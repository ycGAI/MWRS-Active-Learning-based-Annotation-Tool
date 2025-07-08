"""
增强版的YOLO微调模型，支持：
1. 显示每个bbox的置信度
2. 基于不确定性的主动学习
3. 优先标注低置信度样本
"""

import os
import shutil
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from ultralytics import YOLO
import yaml

from ..control_models.rectangle_labels import RectangleLabelsModel, is_obb
from ..control_models.base import ControlModel
from ..utils import get_bool
from .uncertainty_sampler import UncertaintySampler

logger = logging.getLogger(__name__)


class RectangleLabelsModelWithFinetune(RectangleLabelsModel):
    """
    增强版YOLO模型，支持微调和置信度显示
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 初始化不确定性采样器
        self.uncertainty_sampler = UncertaintySampler(
            project_id=self.project_id,
            confidence_threshold=float(self.control.attr.get("uncertainty_threshold", "0.5")),
            min_confidence_for_training=float(self.control.attr.get("min_confidence_for_training", "0.3"))
        )
        
        # 微调相关设置
        self.enable_finetuning = get_bool(self.control.attr, "model_enable_finetune", "false")
        self.show_confidence = get_bool(self.control.attr, "model_show_confidence", "true")
        self.prioritize_uncertain = get_bool(self.control.attr, "model_prioritize_uncertain", "true")
        
        # 训练数据目录
        self.training_data_dir = f"/app/training_data/project_{self.project_id}"
        self.setup_training_dirs()
        
        # 微调模型路径
        self.finetune_model_path = f"/app/models/project_{self.project_id}_finetuned.pt"
        
        # 标注计数器
        self.annotation_counter_file = f"/app/models/project_{self.project_id}_counter.json"
        self.annotation_counter = self.load_annotation_counter()
        
        logger.info(f"RectangleLabelsModelWithFinetune initialized for project {self.project_id}")
        logger.info(f"Fine-tuning enabled: {self.enable_finetuning}")
        logger.info(f"Show confidence: {self.show_confidence}")
        logger.info(f"Prioritize uncertain: {self.prioritize_uncertain}")
        
    def predict_regions(self, path):
        """预测区域并更新置信度信息"""
        # 获取最佳可用模型
        model = self.get_best_model()
        
        # 执行预测
        results = model.predict(
            path,
            conf=self.model_conf_threshold,
            device=self.device,
            max_det=self.model_max_det,
            iou=self.model_iou_threshold,
        )
        
        # 检查OBB模式
        if results[0].obb is not None:
            if not is_obb(self.control):
                logger.warning(
                    "Oriented bounding box (OBB) model detected. "
                    'However, `model_obb="true"` is not set at the RectangleLabels tag '
                    "in the labeling config."
                )
        
        # 创建预测结果
        regions = self.create_enhanced_rectangles(results, path)
        
        # 更新不确定性采样器
        task_id = os.path.basename(path).split('.')[0]  # 从路径提取task_id
        self.uncertainty_sampler.update_task_confidence(task_id, regions)
        
        return regions
    
    def create_enhanced_rectangles(self, results, path):
        """创建增强的矩形标注，包含置信度显示"""
        logger.debug(f"create_enhanced_rectangles: {self.from_name}")
        data = results[0].boxes
        model_names = results[0].names
        regions = []
        
        for i in range(data.shape[0]):
            score = float(data.conf[i])
            x, y, w, h = data.xywhn[i].tolist()
            model_label = model_names[int(data.cls[i])]
            
            if score < self.model_score_threshold:
                continue
            
            if model_label not in self.label_map:
                continue
            output_label = self.label_map[model_label]
            
            # 创建region，确保包含score
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
                "score": float(score),  # 确保是float类型
            }
            
            regions.append(region)
            
            # 记录日志
            logger.info(f"Detection: {output_label} (confidence: {score:.3f})")
        
        return regions
    
    def get_prediction_priority(self, tasks):
        """
        根据不确定性对任务进行优先级排序
        
        Args:
            tasks: 任务列表
            
        Returns:
            排序后的任务列表
        """
        if not self.prioritize_uncertain:
            return tasks
        
        # 获取不确定的任务ID列表
        uncertain_task_ids = self.uncertainty_sampler.get_uncertain_tasks(
            n=len(tasks),
            strategy="least_confident"
        )
        uncertain_task_id_set = {task_id for task_id, _ in uncertain_task_ids}
        
        # 分离确定和不确定的任务
        uncertain_tasks = []
        certain_tasks = []
        
        for task in tasks:
            task_id = str(task.get('id', ''))
            if task_id in uncertain_task_id_set:
                uncertain_tasks.append(task)
            else:
                certain_tasks.append(task)
        
        # 不确定的任务优先
        return uncertain_tasks + certain_tasks
    
    def get_training_statistics(self):
        """获取训练统计信息"""
        stats = {
            "annotation_count": self.annotation_counter,
            "has_finetuned_model": os.path.exists(self.finetune_model_path),
            "confidence_distribution": self.uncertainty_sampler.get_confidence_distribution(),
            "annotation_statistics": self.uncertainty_sampler.get_annotation_statistics()
        }
        
        # 如果有微调模型，添加模型信息
        if stats["has_finetuned_model"]:
            stats["finetuned_model_info"] = {
                "path": self.finetune_model_path,
                "modified_time": datetime.fromtimestamp(
                    os.path.getmtime(self.finetune_model_path)
                ).isoformat()
            }
        
        return stats
    
    def setup_training_dirs(self):
        """设置训练数据目录结构"""
        dirs = [
            f"{self.training_data_dir}/images/train",
            f"{self.training_data_dir}/images/val",
            f"{self.training_data_dir}/labels/train",
            f"{self.training_data_dir}/labels/val"
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def load_annotation_counter(self):
        """加载标注计数器"""
        if os.path.exists(self.annotation_counter_file):
            try:
                with open(self.annotation_counter_file, 'r') as f:
                    data = json.load(f)
                    return data.get('count', 0)
            except Exception as e:
                logger.error(f"Error loading annotation counter: {e}")
        return 0
    
    def save_annotation_counter(self):
        """保存标注计数器"""
        try:
            with open(self.annotation_counter_file, 'w') as f:
                json.dump({'count': self.annotation_counter}, f)
        except Exception as e:
            logger.error(f"Error saving annotation counter: {e}")
    
    def get_best_model(self):
        """获取最佳可用模型（微调模型或原始模型）"""
        if os.path.exists(self.finetune_model_path):
            logger.info(f"Using fine-tuned model: {self.finetune_model_path}")
            return YOLO(self.finetune_model_path)
        else:
            logger.info(f"Using original model: {self.model_path}")
            return self.model
    
    def fit(self, event, data, **kwargs):
        """实现微调训练逻辑"""
        if not self.enable_finetuning:
            logger.info("Fine-tuning is disabled")
            return False
        
        logger.info(f"Fine-tuning fit called with event: {event}")
        
        if event not in ['ANNOTATION_CREATED', 'ANNOTATION_UPDATED']:
            logger.info(f"Skipping training for event: {event}")
            return False
        
        try:
            # 提取任务和标注数据
            task = data.get('task', {})
            annotations = data.get('annotations', [])
            
            if not annotations:
                logger.warning("No annotations found for training")
                return False
            
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
                
                if (self.annotation_counter >= min_annotations and 
                    self.annotation_counter % retrain_interval == 0):
                    
                    logger.info(f"Starting fine-tuning with {self.annotation_counter} annotations")
                    return self.start_finetune()
            
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
            
            # 下载图像到本地
            local_image_path = self.label_studio_ml_backend.get_local_path(
                image_url, task_id=task['id']
            )
            
            if not os.path.exists(local_image_path):
                logger.error(f"Local image path does not exist: {local_image_path}")
                return False
            
            # 复制图像到训练目录
            image_filename = f"task_{task['id']}_ann_{annotation['id']}.jpg"
            train_image_path = f"{self.training_data_dir}/images/train/{image_filename}"
            shutil.copy2(local_image_path, train_image_path)
            
            # 生成YOLO格式的标签文件
            label_filename = f"task_{task['id']}_ann_{annotation['id']}.txt"
            label_path = f"{self.training_data_dir}/labels/train/{label_filename}"
            
            # 转换标注为YOLO格式
            yolo_labels = self.convert_to_yolo_format(annotation, local_image_path)
            
            # 保存标签文件
            with open(label_path, 'w') as f:
                for label in yolo_labels:
                    f.write(f"{label}\n")
            
            logger.info(f"Saved training data for task {task['id']}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing annotation: {e}", exc_info=True)
            return False
    
    def convert_to_yolo_format(self, annotation, image_path):
        """将Label Studio标注转换为YOLO格式"""
        from PIL import Image
        
        # 获取图像尺寸
        with Image.open(image_path) as img:
            img_width, img_height = img.size
        
        yolo_labels = []
        
        # 处理每个结果
        for result in annotation.get('result', []):
            if result['type'] != 'rectanglelabels':
                continue
            
            # 获取标签
            labels = result['value'].get('rectanglelabels', [])
            if not labels:
                continue
            
            label = labels[0]  # 取第一个标签
            
            # 获取类别索引
            if label not in self.label_attrs:
                logger.warning(f"Label '{label}' not found in label_attrs")
                continue
            
            class_idx = list(self.label_attrs.keys()).index(label)
            
            # 获取边界框坐标（Label Studio使用百分比）
            x = result['value']['x'] / 100.0
            y = result['value']['y'] / 100.0
            width = result['value']['width'] / 100.0
            height = result['value']['height'] / 100.0
            
            # 转换为YOLO格式（中心点坐标）
            x_center = x + width / 2
            y_center = y + height / 2
            
            # 添加YOLO格式标签
            yolo_labels.append(f"{class_idx} {x_center} {y_center} {width} {height}")
        
        return yolo_labels
    
    def start_finetune(self):
        """开始微调训练"""
        try:
            # 检查是否有足够的训练数据
            train_images = list(Path(f"{self.training_data_dir}/images/train").glob("*.jpg"))
            if len(train_images) < 5:
                logger.warning(f"Not enough training images: {len(train_images)}")
                return False
            
            logger.info(f"Starting fine-tuning with {len(train_images)} images")
            
            # 创建验证集
            self.create_validation_split()
            
            # 创建训练配置
            config_path = self.create_training_config()
            
            # 获取训练参数
            epochs = int(self.control.attr.get("model_finetune_epochs", "20"))
            batch_size = int(self.control.attr.get("model_finetune_batch", "8"))
            learning_rate = float(self.control.attr.get("model_finetune_lr", "0.001"))
            patience = int(self.control.attr.get("model_finetune_patience", "5"))
            
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
    
    def create_training_config(self):
        """创建YOLO训练配置文件"""
        config = {
            'path': self.training_data_dir,
            'train': 'images/train',
            'val': 'images/val',
            'names': {i: label for i, label in enumerate(self.label_attrs.keys())}
        }
        
        config_path = f"{self.training_data_dir}/data.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        return config_path
    
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