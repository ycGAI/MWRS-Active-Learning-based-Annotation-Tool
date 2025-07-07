"""
增强版的Label Studio ML后端
支持置信度显示、不确定性采样和训练统计
"""

from label_studio_ml.api import init_app
from label_studio_ml.utils import get_single_tag_keys, get_env
from flask import jsonify, request
import logging

# 导入增强版模型
from control_models.rectangle_labels_finetune_enhanced import RectangleLabelsModelWithFinetune

logger = logging.getLogger(__name__)

# Label Studio ML后端类
class YOLOBackend(object):
    
    def __init__(self, project_id=None, label_config=None):
        self.project_id = project_id
        self.label_config = label_config
        
        # 解析标签配置
        self.from_name, self.to_name, self.value, self.labels = get_single_tag_keys(
            self.label_config, 'RectangleLabels', 'Image'
        )
        
        # 初始化模型
        self.model = RectangleLabelsModelWithFinetune(
            label_config=label_config,
            project_id=project_id
        )
        
        logger.info(f"YOLOBackend initialized for project {project_id}")
    
    def predict(self, tasks, context=None, **kwargs):
        """
        预测任务并返回结果
        支持优先级排序
        """
        predictions = []
        
        # 如果启用了不确定性优先级，对任务进行排序
        if hasattr(self.model, 'get_prediction_priority'):
            tasks = self.model.get_prediction_priority(tasks)
        
        for task in tasks:
            # 获取图像路径
            image_url = task['data'].get(self.value)
            if not image_url:
                predictions.append({'result': []})
                continue
            
            # 获取本地路径
            image_path = self.get_local_path(image_url, task_id=task.get('id'))
            
            # 执行预测
            regions = self.model.predict_regions(image_path)
            
            # 构建预测结果
            prediction = {
                'result': regions,
                'score': sum(r['score'] for r in regions) / len(regions) if regions else 0
            }
            
            # 添加模型版本信息
            if hasattr(self.model, 'model_version'):
                prediction['model_version'] = self.model.model_version
            
            predictions.append(prediction)
        
        return predictions
    
    def fit(self, event, data, **kwargs):
        """
        处理标注事件，触发模型训练
        """
        logger.info(f"Fit called with event: {event}")
        
        if hasattr(self.model, 'fit'):
            return self.model.fit(event, data, **kwargs)
        
        return {'status': 'ok'}
    
    def get_local_path(self, url, task_id=None):
        """获取图像的本地路径"""
        # 这里需要实现从URL到本地路径的转换
        # 可以使用Label Studio的存储API
        return url  # 简化示例，实际需要下载文件


# 创建Flask应用
_model = YOLOBackend

# 初始化应用
app = init_app(
    model_class=_model,
    # 添加自定义路由
    additional_routes=[
        {
            'rule': '/uncertainty_stats',
            'view_func': lambda: get_uncertainty_stats(),
            'methods': ['GET']
        },
        {
            'rule': '/training_stats', 
            'view_func': lambda: get_training_stats(),
            'methods': ['GET']
        },
        {
            'rule': '/reset_training',
            'view_func': lambda: reset_training(),
            'methods': ['POST']
        }
    ]
)

# 自定义API端点
def get_uncertainty_stats():
    """获取不确定性统计信息"""
    try:
        model = app.config.get('model')
        if hasattr(model, 'uncertainty_sampler'):
            stats = model.uncertainty_sampler.get_annotation_statistics()
            return jsonify(stats)
        else:
            return jsonify({'error': 'Uncertainty sampler not available'}), 404
    except Exception as e:
        logger.error(f"Error getting uncertainty stats: {e}")
        return jsonify({'error': str(e)}), 500

def get_training_stats():
    """获取训练统计信息"""
    try:
        model = app.config.get('model')
        if hasattr(model, 'get_training_statistics'):
            stats = model.get_training_statistics()
            return jsonify(stats)
        else:
            return jsonify({'error': 'Training statistics not available'}), 404
    except Exception as e:
        logger.error(f"Error getting training stats: {e}")
        return jsonify({'error': str(e)}), 500

def reset_training():
    """重置训练数据和模型"""
    try:
        model = app.config.get('model')
        if hasattr(model, 'reset_training'):
            success = model.reset_training()
            if success:
                return jsonify({'status': 'success', 'message': 'Training data and model reset'})
            else:
                return jsonify({'status': 'error', 'message': 'Failed to reset training'}), 500
        else:
            return jsonify({'error': 'Reset training not available'}), 404
    except Exception as e:
        logger.error(f"Error resetting training: {e}")
        return jsonify({'error': str(e)}), 500

# 添加健康检查端点
@app.route('/health')
def health():
    return jsonify({'status': 'UP'})

# 启动应用
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9090, debug=True)