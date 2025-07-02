import os
import logging
import yaml
import shutil
import json
from pathlib import Path
from typing import List, Dict, Optional
from ultralytics import YOLO
from control_models.base import ControlModel, get_bool
from label_studio_sdk.label_interface.control_tags import ControlTag


logger = logging.getLogger(__name__)


class RectangleLabelsModelWithFinetune(ControlModel):
    """
    Enhanced RectangleLabels model with fine-tuning capabilities for YOLO
    """

    type = "RectangleLabels"
    model_path = "best.pt"
    
    # 声明额外的属性
    training_data_dir: str = None
    finetune_model_path: str = None
    annotation_counter_file: str = None
    annotation_counter: int = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 为每个项目创建独立的训练目录
        self.project_id = self.label_studio_ml_backend.project_id
        self.training_data_dir = f"/app/training_data/project_{self.project_id}"
        self.finetune_model_path = f"/app/models/finetune_project_{self.project_id}.pt"
        self.annotation_counter_file = f"{self.training_data_dir}_counter.json"
        self.setup_training_dirs()
        self.load_annotation_counter()

    def setup_training_dirs(self):
        """创建训练数据目录结构"""
        os.makedirs(f"{self.training_data_dir}/images/train", exist_ok=True)
        os.makedirs(f"{self.training_data_dir}/labels/train", exist_ok=True)
        os.makedirs(f"{self.training_data_dir}/images/val", exist_ok=True)
        os.makedirs(f"{self.training_data_dir}/labels/val", exist_ok=True)
        os.makedirs(f"/app/models/runs/project_{self.project_id}", exist_ok=True)

    def load_annotation_counter(self):
        """加载标注计数器"""
        try:
            if os.path.exists(self.annotation_counter_file):
                with open(self.annotation_counter_file, 'r') as f:
                    data = json.load(f)
                    self.annotation_counter = data.get('count', 0)
            else:
                self.annotation_counter = 0
        except Exception as e:
            logger.warning(f"Failed to load annotation counter: {e}")
            self.annotation_counter = 0

    def save_annotation_counter(self):
        """保存标注计数器"""
        try:
            with open(self.annotation_counter_file, 'w') as f:
                json.dump({'count': self.annotation_counter}, f)
        except Exception as e:
            logger.error(f"Failed to save annotation counter: {e}")

    @classmethod
    def is_control_matched(cls, control) -> bool:
        """检查控制标签是否匹配并启用了微调功能"""
        if control.objects[0].tag != "Image":
            return False
        if is_obb(control):
            return False
        # 检查是否启用微调模式
        is_finetune = get_bool(control.attr, "model_finetune", "false")
        is_rectangle = control.tag == cls.type
        
        logger.info(f"Control matching - Tag: {control.tag}, Finetune: {is_finetune}, Rectangle: {is_rectangle}")
        return is_rectangle and is_finetune

    def predict_regions(self, path) -> List[Dict]:
        """使用最佳可用模型进行预测"""
        # 优先使用微调后的模型
        model_to_use = self.get_best_model()
        
        results = model_to_use.predict(path)
        self.debug_plot(results[0].plot())

        # 检查OBB
        if results[0].obb is not None and results[0].boxes is None:
            raise ValueError(
                "Oriented bounding boxes are detected in the YOLO model results. "
                'However, `model_obb="true"` is not set at the RectangleLabels tag '
                "in the labeling config."
            )

        return self.create_rectangles(results, path)

    def get_best_model(self):
        """获取最佳可用模型（微调模型或原始模型）"""
        if os.path.exists(self.finetune_model_path):
            logger.info(f"Using fine-tuned model: {self.finetune_model_path}")
            return YOLO(self.finetune_model_path)
        else:
            logger.info(f"Using original model: {self.model_path}")
            return self.model

    def get_local_image_path(self, image_url, task_id):
        """获取本地图像路径（支持多种方式）"""
        logger.info(f"Getting local path for: {image_url}")
        
        # 方式1：尝试使用原始方法（如果API可用）
        try:
            if hasattr(self.label_studio_ml_backend, 'get_local_path'):
                local_path = self.label_studio_ml_backend.get_local_path(image_url, task_id=task_id)
                if os.path.exists(local_path):
                    logger.info(f"Found via API: {local_path}")
                    return local_path
        except Exception as e:
            logger.debug(f"API method failed: {e}")
        
        # 方式2：检查本地挂载的目录
        if image_url.startswith('/data/upload/'):
            # 尝试多个可能的挂载路径
            possible_paths = [
                f"/label-studio-data{image_url.replace('/data', '')}",  # 去掉/data前缀
                f"/label-studio-data{image_url}",  # 完整路径（备用）
                f"/label-studio-data/upload/{os.path.basename(image_url)}",  # 只有文件名（备用）
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    logger.info(f"Found local file: {path}")
                    return path
        
        # 方式3：如果是HTTP URL，尝试下载
        if image_url.startswith('http'):
            try:
                import requests
                temp_dir = f"/tmp/ml-backend/task_{task_id}"
                os.makedirs(temp_dir, exist_ok=True)
                
                filename = image_url.split('/')[-1]
                local_path = os.path.join(temp_dir, filename)
                
                if not os.path.exists(local_path):
                    response = requests.get(image_url, timeout=10)
                    response.raise_for_status()
                    
                    with open(local_path, 'wb') as f:
                        f.write(response.content)
                    logger.info(f"Downloaded to: {local_path}")
                
                return local_path
            except Exception as e:
                logger.error(f"Failed to download: {e}")
        
        # 方式4：返回原始URL，让后续处理
        logger.warning(f"Could not resolve local path for: {image_url}")
        return image_url

    def create_rectangles(self, results, path):
        """创建矩形标注（与原版相同）"""
        logger.debug(f"create_rectangles: {self.from_name}")
        data = results[0].boxes
        model_names = results[0].names
        regions = []

        for i in range(data.shape[0]):
            score = float(data.conf[i])
            x, y, w, h = data.xywhn[i].tolist()
            model_label = model_names[int(data.cls[i])]

            logger.debug(
                "----------------------\n"
                f"task id > {path}\n"
                f"type: {self.control}\n"
                f"x, y, w, h > {x, y, w, h}\n"
                f"model label > {model_label}\n"
                f"score > {score}\n"
            )

            # 置信度过滤
            if score < self.model_score_threshold:
                continue

            # 标签映射检查
            if model_label not in self.label_map:
                continue
            output_label = self.label_map[model_label]

            # 创建区域
            region = {
                "from_name": self.from_name,
                "to_name": self.to_name,
                "type": "rectanglelabels",
                "value": {
                    "rectanglelabels": [output_label],
                    "x": (x - w / 2) * 100,
                    "y": (y - h / 2) * 100,
                    "width": w * 100,
                    "height": h * 100,
                },
                "score": score,
            }
            regions.append(region)
        return regions

    def fit(self, event, data, **kwargs):
        """实现微调训练逻辑"""
        logger.info(f"Fine-tuning fit called with event: {event}")
        
        if event not in ['ANNOTATION_CREATED', 'ANNOTATION_UPDATED']:
            logger.info(f"Skipping training for event: {event}")
            return False

        try:
            # 提取任务和标注数据
            task = data.get('task', {})
            
            # 修复：处理单个annotation
            annotation = data.get('annotation', {})
            annotations = [annotation] if annotation else []
            
            if not annotations:
                logger.warning("No annotations found for training")
                return False

            logger.info(f"Processing {len(annotations)} annotations for task {task.get('id')}")

            # 处理每个标注
            processed_count = 0
            for annotation in annotations:
                if self.process_annotation(task, annotation):
                    self.annotation_counter += 1
                    processed_count += 1

            if processed_count > 0:
                self.save_annotation_counter()
                logger.info(f"Processed {processed_count} annotations. Total count: {self.annotation_counter}")

            # 检查是否需要开始微调
            min_annotations = int(self.control.attr.get("model_min_annotations", "10"))
            retrain_interval = int(self.control.attr.get("model_retrain_interval", "5"))

            logger.info(f"Training check - Current: {self.annotation_counter}, Min: {min_annotations}, Interval: {retrain_interval}")

            if (self.annotation_counter >= min_annotations and 
                self.annotation_counter % retrain_interval == 0):
                
                logger.info(f"Starting fine-tuning with {self.annotation_counter} annotations")
                return self.start_finetune()
            else:
                logger.info("Not enough annotations or not at training interval")
                return True

        except Exception as e:
            logger.error(f"Error in fit method: {e}", exc_info=True)
            return False

    def process_annotation(self, task, annotation):
        """处理单个标注并保存训练数据"""
        try:
            # 获取图像URL
            image_url = task['data'].get('image', '')
            if not image_url:
                logger.warning("No image URL found in task data")
                return False

            logger.info(f"Processing annotation {annotation.get('id')} for task {task.get('id')}")

            # 获取本地图像路径（使用新方法）
            local_image_path = self.get_local_image_path(image_url, task['id'])

            if not os.path.exists(local_image_path):
                logger.error(f"Local image path does not exist: {local_image_path}")
                return False

            # 复制图像到训练目录
            image_filename = f"task_{task['id']}_ann_{annotation['id']}.jpg"
            train_image_path = f"{self.training_data_dir}/images/train/{image_filename}"
            shutil.copy2(local_image_path, train_image_path)

            # 生成YOLO格式的标签文件
            label_content = self.convert_annotation_to_yolo(annotation)
            if not label_content:
                logger.warning(f"No valid labels found for annotation {annotation['id']}")
                return False

            label_filename = f"task_{task['id']}_ann_{annotation['id']}.txt"
            label_path = f"{self.training_data_dir}/labels/train/{label_filename}"

            with open(label_path, 'w') as f:
                f.write(label_content)

            logger.info(f"Saved training data: {train_image_path}, {label_path}")
            return True

        except Exception as e:
            logger.error(f"Error processing annotation: {e}", exc_info=True)
            return False

    def convert_annotation_to_yolo(self, annotation):
        """将Label Studio标注转换为YOLO格式"""
        label_lines = []
        
        for result in annotation.get('result', []):
            if result.get('type') != 'rectanglelabels':
                continue

            value = result.get('value', {})
            rectanglelabels = value.get('rectanglelabels', [])
            
            if not rectanglelabels:
                continue

            # 获取边界框坐标（Label Studio格式：左上角 + 宽高，百分比）
            x_percent = value.get('x', 0) / 100
            y_percent = value.get('y', 0) / 100
            width_percent = value.get('width', 0) / 100
            height_percent = value.get('height', 0) / 100

            # 转换为YOLO格式（中心点 + 宽高，归一化）
            center_x = x_percent + width_percent / 2
            center_y = y_percent + height_percent / 2

            # 获取类别ID
            label_name = rectanglelabels[0]
            class_id = self.get_class_id(label_name)

            # YOLO格式：class_id center_x center_y width height
            label_line = f"{class_id} {center_x:.6f} {center_y:.6f} {width_percent:.6f} {height_percent:.6f}"
            label_lines.append(label_line)

        return '\n'.join(label_lines)

    def get_class_id(self, label_name):
        """获取标签对应的类别ID"""
        # 创建反向映射：从Label Studio标签到模型类别ID
        if not hasattr(self, '_label_to_id_map'):
            self._label_to_id_map = {}
            
            # 获取所有唯一的Label Studio标签
            unique_labels = sorted(set(self.label_map.values()))
            
            # 为每个标签分配一个类别ID
            for i, label in enumerate(unique_labels):
                self._label_to_id_map[label] = i

        return self._label_to_id_map.get(label_name, 0)

    def create_training_config(self):
        """创建YOLO训练配置文件"""
        # 获取所有唯一的Label Studio标签
        unique_labels = sorted(set(self.label_map.values()))
        
        config = {
            'path': self.training_data_dir,
            'train': 'images/train',
            'val': 'images/val',
            'names': {}
        }

        # 构建类别名称映射
        for i, label in enumerate(unique_labels):
            config['names'][i] = label

        config_path = f"{self.training_data_dir}/dataset.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        logger.info(f"Created training config: {config_path}")
        logger.info(f"Classes: {config['names']}")
        return config_path

    def start_finetune(self):
        """开始微调训练"""
        try:
            # 检查训练数据
            train_images = list(Path(f"{self.training_data_dir}/images/train").glob("*.jpg"))
            train_labels = list(Path(f"{self.training_data_dir}/labels/train").glob("*.txt"))
            
            if len(train_images) == 0:
                logger.error("No training images found")
                return False
                
            if len(train_labels) == 0:
                logger.error("No training labels found")
                return False
                
            logger.info(f"Found {len(train_images)} images and {len(train_labels)} labels")

            # 创建验证集
            self.create_validation_split()

            # 创建训练配置
            config_path = self.create_training_config()

            # 获取训练参数
            epochs = int(self.control.attr.get("model_finetune_epochs", "20"))
            batch_size = int(self.control.attr.get("model_finetune_batch", "8"))
            learning_rate = float(self.control.attr.get("model_finetune_lr", "0.001"))
            patience = int(self.control.attr.get("model_finetune_patience", "5"))

            logger.info(f"Training parameters - epochs: {epochs}, batch: {batch_size}, lr: {learning_rate}, patience: {patience}")

            # 初始化模型进行微调
            base_model_path = f"/app/models/{self.model_path}"
            model = YOLO(base_model_path)

            # 开始训练
            logger.info("Starting YOLO fine-tuning...")
            
            results = model.train(
                data=config_path,
                epochs=epochs,
                batch=batch_size,
                lr0=learning_rate,
                patience=patience,
                save=True,
                project=f"/app/models/runs/project_{self.project_id}",
                name="finetune",
                exist_ok=True,
                verbose=True
            )

            # 保存最佳模型
            best_model_path = results.save_dir / "weights" / "best.pt"
            if os.path.exists(best_model_path):
                shutil.copy2(best_model_path, self.finetune_model_path)
                logger.info(f"Fine-tuned model saved to: {self.finetune_model_path}")

                # 更新当前使用的模型
                self.model = YOLO(self.finetune_model_path)
                
                return True
            else:
                logger.error("Best model not found after training")
                return False

        except Exception as e:
            logger.error(f"Error during fine-tuning: {e}", exc_info=True)
            return False

    def create_validation_split(self):
        """创建验证集"""
        train_images = list(Path(f"{self.training_data_dir}/images/train").glob("*.jpg"))
        
        # 简单的验证集分割（取20%作为验证集）
        val_ratio = 0.2
        val_count = max(1, int(len(train_images) * val_ratio))

        logger.info(f"Creating validation split: {val_count} images out of {len(train_images)}")

        for i, img_file in enumerate(train_images[-val_count:]):
            # 复制图像
            val_img_path = f"{self.training_data_dir}/images/val/{img_file.name}"
            shutil.copy2(img_file, val_img_path)

            # 复制对应的标签文件
            label_file = f"{self.training_data_dir}/labels/train/{img_file.stem}.txt"
            if os.path.exists(label_file):
                val_label_path = f"{self.training_data_dir}/labels/val/{img_file.stem}.txt"
                shutil.copy2(label_file, val_label_path)

    def reset_training(self):
        """重置训练数据和模型"""
        try:
            if os.path.exists(self.training_data_dir):
                shutil.rmtree(self.training_data_dir)
            
            if os.path.exists(self.finetune_model_path):
                os.remove(self.finetune_model_path)
                
            if os.path.exists(self.annotation_counter_file):
                os.remove(self.annotation_counter_file)
                
            self.annotation_counter = 0
            self.setup_training_dirs()
            logger.info("Training data and fine-tuned model reset")
            return True
        except Exception as e:
            logger.error(f"Error resetting training: {e}")
            return False


def is_obb(control: ControlTag) -> bool:
    """Check if the model should use oriented bounding boxes (OBB)"""
    return get_bool(control.attr, "model_obb", "false")


# 预加载缓存模型
RectangleLabelsModelWithFinetune.get_cached_model(RectangleLabelsModelWithFinetune.model_path)