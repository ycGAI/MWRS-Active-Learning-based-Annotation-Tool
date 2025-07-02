import os
import logging
import yaml
import shutil
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from ultralytics import YOLO
from control_models.base import ControlModel, get_bool
from label_studio_sdk.label_interface.control_tags import ControlTag


logger = logging.getLogger(__name__)


class RectangleLabelsModelWithFinetune(ControlModel):
    """
    Enhanced RectangleLabels model with fine-tuning capabilities for YOLO
    支持模型预测后的人工修正和自动fine-tune
    """

    type = "RectangleLabels"
    model_path = "best.pt"  # 默认基础模型

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 项目相关路径
        self.project_id = self.label_studio_ml_backend.project_id
        self.training_data_dir = f"/app/training_data/project_{self.project_id}"
        self.models_dir = f"/app/models/project_{self.project_id}"
        
        # 模型版本管理
        self.base_model_path = f"/app/models/{self.model_path}"
        self.current_model_path = f"{self.models_dir}/current_model.pt"
        self.finetune_history_dir = f"{self.models_dir}/history"
        
        # 数据统计文件
        self.stats_file = f"{self.training_data_dir}/training_stats.json"
        
        # 初始化
        self.setup_directories()
        self.load_training_stats()
        self.initialize_model()

    def setup_directories(self):
        """创建必要的目录结构"""
        directories = [
            f"{self.training_data_dir}/images/train",
            f"{self.training_data_dir}/labels/train",
            f"{self.training_data_dir}/images/val",
            f"{self.training_data_dir}/labels/val",
            f"{self.training_data_dir}/corrections",  # 保存人工修正记录
            self.models_dir,
            self.finetune_history_dir,
            f"/app/models/runs/project_{self.project_id}"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def initialize_model(self):
        """初始化模型（使用当前最佳模型或基础模型）"""
        if os.path.exists(self.current_model_path):
            logger.info(f"Loading current fine-tuned model: {self.current_model_path}")
            self.model = YOLO(self.current_model_path)
        else:
            logger.info(f"Loading base model: {self.base_model_path}")
            self.model = YOLO(self.base_model_path)
            # 复制基础模型作为当前模型
            if os.path.exists(self.base_model_path):
                shutil.copy2(self.base_model_path, self.current_model_path)

    def load_training_stats(self):
        """加载训练统计信息"""
        default_stats = {
            'total_annotations': 0,
            'total_corrections': 0,
            'finetune_history': [],
            'last_finetune_date': None,
            'model_version': 1,
            'annotation_history': []
        }
        
        try:
            if os.path.exists(self.stats_file):
                with open(self.stats_file, 'r') as f:
                    self.stats = json.load(f)
                    # 确保所有必要字段都存在
                    for key, value in default_stats.items():
                        if key not in self.stats:
                            self.stats[key] = value
            else:
                self.stats = default_stats
        except Exception as e:
            logger.error(f"Failed to load stats: {e}")
            self.stats = default_stats

    def save_training_stats(self):
        """保存训练统计信息"""
        try:
            with open(self.stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save stats: {e}")

    @classmethod
    def is_control_matched(cls, control) -> bool:
        """检查控制标签是否匹配"""
        if control.objects[0].tag != "Image":
            return False
        if is_obb(control):
            return False
        # 检查是否启用微调模式
        is_finetune = get_bool(control.attr, "model_finetune", "false")
        is_rectangle = control.tag == cls.type
        
        return is_rectangle and is_finetune

    def predict_regions(self, path) -> List[Dict]:
        """使用当前最佳模型进行预测"""
        results = self.model.predict(path)
        self.debug_plot(results[0].plot())
        
        return self.create_rectangles(results, path)

    def create_rectangles(self, results, path):
        """创建矩形标注结果"""
        logger.debug(f"create_rectangles: {self.from_name}")
        data = results[0].boxes
        model_names = results[0].names
        regions = []

        for i in range(data.shape[0]):
            score = float(data.conf[i])
            x, y, w, h = data.xywhn[i].tolist()
            model_label = model_names[int(data.cls[i])]

            # 置信度过滤
            if score < self.model_score_threshold:
                continue

            # 标签映射
            if model_label not in self.label_map:
                continue
            output_label = self.label_map[model_label]

            # 创建预测区域
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
                "model_version": self.stats['model_version']  # 添加模型版本信息
            }
            regions.append(region)
            
        return regions

    def fit(self, event, data, **kwargs):
        """处理标注事件并触发fine-tune"""
        logger.info(f"Fit called with event: {event}")
        
        if event not in ['ANNOTATION_CREATED', 'ANNOTATION_UPDATED']:
            return False

        try:
            task = data.get('task', {})
            annotations = data.get('annotations', [])
            
            if not annotations:
                return False

            # 处理每个标注
            for annotation in annotations:
                is_correction = self.is_manual_correction(annotation, task)
                
                if self.process_annotation(task, annotation, is_correction):
                    self.stats['total_annotations'] += 1
                    if is_correction:
                        self.stats['total_corrections'] += 1
                    
                    # 记录标注历史
                    self.stats['annotation_history'].append({
                        'task_id': task.get('id'),
                        'annotation_id': annotation.get('id'),
                        'timestamp': datetime.now().isoformat(),
                        'is_correction': is_correction
                    })

            self.save_training_stats()
            
            # 检查是否需要触发fine-tune
            return self.check_and_trigger_finetune()

        except Exception as e:
            logger.error(f"Error in fit: {e}", exc_info=True)
            return False

    def is_manual_correction(self, annotation, task):
        """判断是否为人工修正的标注"""
        # 方法1：检查是否有模型预测结果
        predictions = task.get('predictions', [])
        if predictions:
            # 如果存在预测但标注与预测不同，认为是修正
            for pred in predictions:
                if pred.get('model_version') == self.stats['model_version']:
                    # 比较标注和预测的差异
                    return True
        
        # 方法2：检查标注的来源
        if annotation.get('was_cancelled', False):
            return False
            
        # 方法3：检查标注时间（如果标注在预测之后创建）
        created_at = annotation.get('created_at')
        if created_at and predictions:
            # 实现时间比较逻辑
            pass
            
        return True  # 默认认为是人工标注

    def process_annotation(self, task, annotation, is_correction=False):
        """处理并保存标注数据"""
        try:
            # 获取图像
            image_url = task['data'].get('image', '')
            if not image_url:
                return False

            local_image_path = self.label_studio_ml_backend.get_local_path(
                image_url, task_id=task['id']
            )

            # 生成唯一文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_name = f"task_{task['id']}_ann_{annotation['id']}_{timestamp}"
            
            # 保存到训练目录
            train_image_path = f"{self.training_data_dir}/images/train/{base_name}.jpg"
            shutil.copy2(local_image_path, train_image_path)

            # 生成YOLO格式标签
            label_content = self.convert_annotation_to_yolo(annotation)
            if not label_content:
                return False

            label_path = f"{self.training_data_dir}/labels/train/{base_name}.txt"
            with open(label_path, 'w') as f:
                f.write(label_content)

            # 如果是人工修正，额外保存修正记录
            if is_correction:
                correction_record = {
                    'task_id': task['id'],
                    'annotation_id': annotation['id'],
                    'timestamp': timestamp,
                    'image_file': base_name + '.jpg',
                    'label_file': base_name + '.txt',
                    'original_predictions': task.get('predictions', [])
                }
                
                correction_file = f"{self.training_data_dir}/corrections/{base_name}.json"
                with open(correction_file, 'w') as f:
                    json.dump(correction_record, f, indent=2)

            logger.info(f"Saved training data: {train_image_path}")
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

            # Label Studio格式转YOLO格式
            x_percent = value.get('x', 0) / 100
            y_percent = value.get('y', 0) / 100
            width_percent = value.get('width', 0) / 100
            height_percent = value.get('height', 0) / 100

            center_x = x_percent + width_percent / 2
            center_y = y_percent + height_percent / 2

            # 获取类别ID
            label_name = rectanglelabels[0]
            class_id = self.get_class_id(label_name)

            label_line = f"{class_id} {center_x:.6f} {center_y:.6f} {width_percent:.6f} {height_percent:.6f}"
            label_lines.append(label_line)

        return '\n'.join(label_lines)

    def get_class_id(self, label_name):
        """获取标签对应的类别ID"""
        if not hasattr(self, '_label_to_id_map'):
            self._label_to_id_map = {}
            unique_labels = sorted(set(self.label_map.values()))
            for i, label in enumerate(unique_labels):
                self._label_to_id_map[label] = i

        return self._label_to_id_map.get(label_name, 0)

    def check_and_trigger_finetune(self):
        """检查是否满足fine-tune条件并触发训练"""
        # 获取配置参数
        min_total_annotations = int(self.control.attr.get("model_min_annotations", "20"))
        min_corrections = int(self.control.attr.get("model_min_corrections", "10"))
        retrain_interval = int(self.control.attr.get("model_retrain_interval", "5"))
        
        # 条件1：总标注数量达标
        if self.stats['total_annotations'] < min_total_annotations:
            logger.info(f"Not enough total annotations: {self.stats['total_annotations']} < {min_total_annotations}")
            return False
        
        # 条件2：人工修正数量达标
        corrections_since_last_train = self.get_corrections_since_last_train()
        if corrections_since_last_train < min_corrections:
            logger.info(f"Not enough corrections: {corrections_since_last_train} < {min_corrections}")
            return False
        
        # 条件3：满足重训练间隔
        if self.stats['total_corrections'] % retrain_interval != 0:
            logger.info(f"Not at training interval: {self.stats['total_corrections']} % {retrain_interval} != 0")
            return False
        
        logger.info("All conditions met, starting fine-tune...")
        return self.start_finetune()

    def get_corrections_since_last_train(self):
        """获取上次训练后的修正数量"""
        if not self.stats['finetune_history']:
            return self.stats['total_corrections']
        
        last_train_corrections = self.stats['finetune_history'][-1].get('total_corrections', 0)
        return self.stats['total_corrections'] - last_train_corrections

    def start_finetune(self):
        """开始fine-tune训练"""
        try:
            # 创建验证集
            self.create_validation_split()
            
            # 创建训练配置
            config_path = self.create_training_config()
            
            # 获取训练参数
            epochs = int(self.control.attr.get("model_finetune_epochs", "50"))
            batch_size = int(self.control.attr.get("model_finetune_batch", "16"))
            learning_rate = float(self.control.attr.get("model_finetune_lr", "0.0001"))
            patience = int(self.control.attr.get("model_finetune_patience", "10"))
            
            logger.info(f"Training parameters - epochs: {epochs}, batch: {batch_size}, lr: {learning_rate}")
            
            # 开始训练
            model = YOLO(self.current_model_path)
            
            train_name = f"finetune_v{self.stats['model_version']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            results = model.train(
                data=config_path,
                epochs=epochs,
                batch=batch_size,
                lr0=learning_rate,
                patience=patience,
                save=True,
                project=f"/app/models/runs/project_{self.project_id}",
                name=train_name,
                exist_ok=True,
                verbose=True
            )
            
            # 保存新模型
            best_model_path = results.save_dir / "weights" / "best.pt"
            if os.path.exists(best_model_path):
                # 备份当前模型
                backup_path = f"{self.finetune_history_dir}/model_v{self.stats['model_version']}.pt"
                shutil.copy2(self.current_model_path, backup_path)
                
                # 更新当前模型
                shutil.copy2(best_model_path, self.current_model_path)
                self.model = YOLO(self.current_model_path)
                
                # 更新统计信息
                self.stats['model_version'] += 1
                self.stats['last_finetune_date'] = datetime.now().isoformat()
                self.stats['finetune_history'].append({
                    'version': self.stats['model_version'],
                    'timestamp': datetime.now().isoformat(),
                    'total_annotations': self.stats['total_annotations'],
                    'total_corrections': self.stats['total_corrections'],
                    'metrics': {
                        'mAP': float(results.results_dict.get('metrics/mAP50-95', 0)),
                        'precision': float(results.results_dict.get('metrics/precision', 0)),
                        'recall': float(results.results_dict.get('metrics/recall', 0))
                    }
                })
                
                self.save_training_stats()
                logger.info(f"Fine-tune completed! New model version: {self.stats['model_version']}")
                return True
            else:
                logger.error("Best model not found after training")
                return False
                
        except Exception as e:
            logger.error(f"Error during fine-tune: {e}", exc_info=True)
            return False

    def create_validation_split(self):
        """创建验证集（20%的数据）"""
        train_images = list(Path(f"{self.training_data_dir}/images/train").glob("*.jpg"))
        
        val_ratio = 0.2
        val_count = max(1, int(len(train_images) * val_ratio))
        
        # 随机选择验证集
        import random
        random.shuffle(train_images)
        
        for img_file in train_images[-val_count:]:
            # 移动到验证集
            val_img_path = f"{self.training_data_dir}/images/val/{img_file.name}"
            shutil.move(str(img_file), val_img_path)
            
            # 移动对应的标签
            label_file = f"{self.training_data_dir}/labels/train/{img_file.stem}.txt"
            if os.path.exists(label_file):
                val_label_path = f"{self.training_data_dir}/labels/val/{img_file.stem}.txt"
                shutil.move(label_file, val_label_path)

    def create_training_config(self):
        """创建YOLO训练配置文件"""
        unique_labels = sorted(set(self.label_map.values()))
        
        config = {
            'path': self.training_data_dir,
            'train': 'images/train',
            'val': 'images/val',
            'names': {i: label for i, label in enumerate(unique_labels)}
        }
        
        config_path = f"{self.training_data_dir}/dataset.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
        return config_path

    def reset_training(self):
        """重置训练数据和模型（用于调试）"""
        try:
            if os.path.exists(self.training_data_dir):
                shutil.rmtree(self.training_data_dir)
            
            if os.path.exists(self.models_dir):
                shutil.rmtree(self.models_dir)
            
            self.setup_directories()
            self.stats = {
                'total_annotations': 0,
                'total_corrections': 0,
                'finetune_history': [],
                'last_finetune_date': None,
                'model_version': 1,
                'annotation_history': []
            }
            self.save_training_stats()
            self.initialize_model()
            
            logger.info("Training data and models reset successfully")
            return True
        except Exception as e:
            logger.error(f"Error resetting training: {e}")
            return False


def is_obb(control: ControlTag) -> bool:
    """检查是否使用定向边界框"""
    return get_bool(control.attr, "model_obb", "false")