import os
import json
import logging
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class UncertaintySampler:
    """
    基于不确定性的主动学习采样器
    用于优先选择模型置信度低的样本进行标注
    """
    
    def __init__(self, 
                 project_id: str,
                 cache_dir: str = "/app/uncertainty_cache",
                 confidence_threshold: float = 0.5,
                 min_confidence_for_training: float = 0.3):
        """
        初始化不确定性采样器
        
        Args:
            project_id: 项目ID
            cache_dir: 缓存目录
            confidence_threshold: 置信度阈值，低于此值的样本被认为是不确定的
            min_confidence_for_training: 最低置信度阈值，用于过滤训练数据
        """
        self.project_id = project_id
        self.cache_dir = cache_dir
        self.confidence_threshold = confidence_threshold
        self.min_confidence_for_training = min_confidence_for_training
        
        # 任务置信度缓存: {task_id: {"avg_confidence": float, "predictions": list, "timestamp": str}}
        self.task_confidence_cache = {}
        
        # 确保缓存目录存在
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, f"project_{project_id}_confidence.json")
        
        # 加载已有的置信度数据
        self.load_confidence_cache()
        
    def load_confidence_cache(self):
        """从文件加载置信度缓存"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self.task_confidence_cache = json.load(f)
                logger.info(f"Loaded {len(self.task_confidence_cache)} task confidences from cache")
            except Exception as e:
                logger.error(f"Error loading confidence cache: {e}")
                self.task_confidence_cache = {}
    
    def save_confidence_cache(self):
        """保存置信度缓存到文件"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.task_confidence_cache, f, indent=2)
            logger.debug(f"Saved {len(self.task_confidence_cache)} task confidences to cache")
        except Exception as e:
            logger.error(f"Error saving confidence cache: {e}")
    
    def calculate_prediction_confidence(self, predictions: List[Dict]) -> Dict:
        """
        计算预测结果的置信度统计
        
        Args:
            predictions: 预测结果列表
            
        Returns:
            包含置信度统计信息的字典
        """
        if not predictions:
            return {
                "avg_confidence": 0.0,
                "min_confidence": 0.0,
                "max_confidence": 0.0,
                "num_predictions": 0,
                "low_confidence_ratio": 1.0
            }
        
        scores = [p.get('score', 0.0) for p in predictions]
        low_confidence_count = sum(1 for s in scores if s < self.confidence_threshold)
        
        return {
            "avg_confidence": np.mean(scores),
            "min_confidence": np.min(scores),
            "max_confidence": np.max(scores),
            "num_predictions": len(predictions),
            "low_confidence_ratio": low_confidence_count / len(scores) if scores else 0
        }
    
    def update_task_confidence(self, task_id: str, predictions: List[Dict]):
        """
        更新任务的置信度信息
        
        Args:
            task_id: 任务ID
            predictions: 预测结果列表
        """
        confidence_stats = self.calculate_prediction_confidence(predictions)
        
        self.task_confidence_cache[task_id] = {
            "avg_confidence": confidence_stats["avg_confidence"],
            "min_confidence": confidence_stats["min_confidence"],
            "max_confidence": confidence_stats["max_confidence"],
            "num_predictions": confidence_stats["num_predictions"],
            "low_confidence_ratio": confidence_stats["low_confidence_ratio"],
            "predictions": predictions,  # 保存完整预测结果
            "timestamp": datetime.now().isoformat()
        }
        
        # 定期保存缓存
        if len(self.task_confidence_cache) % 10 == 0:
            self.save_confidence_cache()
    
    def get_uncertain_tasks(self, n: int = 10, strategy: str = "least_confident") -> List[Tuple[str, float]]:
        """
        获取最不确定的任务
        
        Args:
            n: 返回的任务数量
            strategy: 采样策略
                - "least_confident": 平均置信度最低
                - "min_confidence": 包含最低置信度预测
                - "high_variance": 置信度方差最大
                - "low_confidence_ratio": 低置信度预测比例最高
                
        Returns:
            [(task_id, uncertainty_score), ...] 列表，按不确定性降序排列
        """
        if not self.task_confidence_cache:
            return []
        
        task_scores = []
        
        for task_id, info in self.task_confidence_cache.items():
            if strategy == "least_confident":
                # 使用平均置信度的反向值作为不确定性分数
                uncertainty_score = 1.0 - info["avg_confidence"]
            elif strategy == "min_confidence":
                # 使用最低置信度的反向值
                uncertainty_score = 1.0 - info["min_confidence"]
            elif strategy == "high_variance":
                # 计算置信度方差
                if info["predictions"]:
                    scores = [p.get('score', 0.0) for p in info["predictions"]]
                    uncertainty_score = np.var(scores) if len(scores) > 1 else 0
                else:
                    uncertainty_score = 0
            elif strategy == "low_confidence_ratio":
                # 使用低置信度预测的比例
                uncertainty_score = info["low_confidence_ratio"]
            else:
                uncertainty_score = 1.0 - info["avg_confidence"]
            
            task_scores.append((task_id, uncertainty_score))
        
        # 按不确定性降序排序
        task_scores.sort(key=lambda x: x[1], reverse=True)
        
        return task_scores[:n]
    
    def get_task_confidence_info(self, task_id: str) -> Optional[Dict]:
        """获取特定任务的置信度信息"""
        return self.task_confidence_cache.get(task_id)
    
    def get_confidence_distribution(self) -> Dict:
        """
        获取所有任务的置信度分布统计
        
        Returns:
            置信度分布统计信息
        """
        if not self.task_confidence_cache:
            return {
                "total_tasks": 0,
                "avg_confidence": 0,
                "confidence_buckets": {}
            }
        
        all_confidences = [info["avg_confidence"] for info in self.task_confidence_cache.values()]
        
        # 计算置信度分布
        buckets = defaultdict(int)
        for conf in all_confidences:
            bucket = int(conf * 10) / 10  # 0.0, 0.1, 0.2, ..., 0.9, 1.0
            buckets[bucket] += 1
        
        return {
            "total_tasks": len(all_confidences),
            "avg_confidence": np.mean(all_confidences),
            "std_confidence": np.std(all_confidences),
            "min_confidence": np.min(all_confidences),
            "max_confidence": np.max(all_confidences),
            "confidence_buckets": dict(buckets),
            "uncertain_tasks": sum(1 for c in all_confidences if c < self.confidence_threshold)
        }
    
    def should_annotate_task(self, task_id: str) -> bool:
        """
        判断是否应该标注该任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            是否应该标注
        """
        info = self.get_task_confidence_info(task_id)
        if not info:
            # 没有预测信息的任务应该被标注
            return True
        
        # 如果平均置信度低于阈值，应该标注
        return info["avg_confidence"] < self.confidence_threshold
    
    def get_training_priorities(self, annotated_tasks: List[Dict]) -> List[Dict]:
        """
        根据置信度对已标注任务进行优先级排序，用于训练
        
        Args:
            annotated_tasks: 已标注的任务列表
            
        Returns:
            按优先级排序的任务列表
        """
        # 计算每个任务的训练优先级
        task_priorities = []
        
        for task in annotated_tasks:
            task_id = str(task.get('id', ''))
            info = self.get_task_confidence_info(task_id)
            
            if info:
                # 优先级基于：低置信度 + 预测数量
                priority = (1.0 - info["avg_confidence"]) * np.log1p(info["num_predictions"])
            else:
                # 没有预测信息的任务给予中等优先级
                priority = 0.5
            
            task_priorities.append((task, priority))
        
        # 按优先级降序排序
        task_priorities.sort(key=lambda x: x[1], reverse=True)
        
        return [task for task, _ in task_priorities]
    
    def get_annotation_statistics(self) -> Dict:
        """获取标注统计信息"""
        stats = self.get_confidence_distribution()
        
        # 添加更多统计信息
        if self.task_confidence_cache:
            # 计算需要标注的任务数量
            tasks_need_annotation = sum(
                1 for info in self.task_confidence_cache.values()
                if info["avg_confidence"] < self.confidence_threshold
            )
            
            # 计算不同置信度区间的任务数量
            confidence_ranges = {
                "very_low": 0,    # [0, 0.3)
                "low": 0,         # [0.3, 0.5)
                "medium": 0,      # [0.5, 0.7)
                "high": 0,        # [0.7, 0.9)
                "very_high": 0    # [0.9, 1.0]
            }
            
            for info in self.task_confidence_cache.values():
                conf = info["avg_confidence"]
                if conf < 0.3:
                    confidence_ranges["very_low"] += 1
                elif conf < 0.5:
                    confidence_ranges["low"] += 1
                elif conf < 0.7:
                    confidence_ranges["medium"] += 1
                elif conf < 0.9:
                    confidence_ranges["high"] += 1
                else:
                    confidence_ranges["very_high"] += 1
            
            stats.update({
                "tasks_need_annotation": tasks_need_annotation,
                "confidence_ranges": confidence_ranges,
                "annotation_progress": {
                    "completed": stats["total_tasks"] - tasks_need_annotation,
                    "remaining": tasks_need_annotation,
                    "percentage": (1 - tasks_need_annotation / stats["total_tasks"]) * 100 
                                  if stats["total_tasks"] > 0 else 0
                }
            })
        
        return stats