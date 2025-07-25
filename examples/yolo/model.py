import os
import logging

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

from control_models.base import ControlModel
# from control_models.choices import ChoicesModel
from control_models.rectangle_labels import RectangleLabelsModel
from control_models.rectangle_labels_finetune import RectangleLabelsModelWithFinetune  # 新增微调模型
from control_models.rectangle_labels_obb import RectangleLabelsObbModel
# from control_models.polygon_labels import PolygonLabelsModel
# from control_models.keypoint_labels import KeypointLabelsModel
# from control_models.video_rectangle import VideoRectangleModel
#from control_models.timeline_labels import TimelineLabelsModel
from typing import List, Dict, Optional


logger = logging.getLogger(__name__)
if not os.getenv("LOG_LEVEL"):
    logger.setLevel(logging.INFO)

# 注册可用模型类（优先级顺序很重要！微调模型需要放在前面）
available_model_classes = [
    #ChoicesModel,
    RectangleLabelsModelWithFinetune,  # 微调模型优先检查
    RectangleLabelsModel,              # 普通矩形标签模型
    RectangleLabelsObbModel,
    # PolygonLabelsModel,
    # KeypointLabelsModel,
    # VideoRectangleModel,
    #TimelineLabelsModel,
]


class YOLO(LabelStudioMLBase):
    """Label Studio ML Backend based on Ultralytics YOLO"""

    def __init__(self, **kwargs):
        """Initialize the model"""
        super(YOLO, self).__init__(**kwargs)
        # 设置是否显示置信度（从环境变量读取，默认为true）
        self.show_confidence = os.getenv('MODEL_SHOW_CONFIDENCE', 'true').lower() == 'true'
        logger.info(f"YOLO ML Backend initialized with show_confidence={self.show_confidence}")

    def setup(self):
        """Configure any parameters of your model here"""
        self.set("model_version", "yolo")

    def detect_control_models(self) -> List[ControlModel]:
        """Detect control models based on the labeling config.
        Control models are used to predict regions for different control tags in the labeling config.
        """
        control_models = []

        for control in self.label_interface.controls:
            # skipping tags without toName
            if not control.to_name:
                logger.warning(
                    f'{control.tag} {control.name} has no "toName" attribute, skipping it'
                )
                continue

            # match control tag with available control models
            # 注意：顺序很重要，微调模型会优先匹配
            matched = False
            for model_class in available_model_classes:
                if model_class.is_control_matched(control):
                    instance = model_class.create(self, control)
                    if not instance:
                        logger.debug(
                            f"No instance created for {control.tag} {control.name}"
                        )
                        continue
                    if not instance.label_map:
                        logger.error(
                            f"No label map built for the '{control.tag}' control tag '{instance.from_name}'.\n"
                            f"This indicates that your Label Studio config labels do not match the model's labels.\n"
                            f"To fix this, ensure that the 'value' or 'predicted_values' attribute "
                            f"in your Label Studio config matches one or more of these model labels.\n"
                            f"If you don't want to use this control tag for predictions, "
                            f'add `model_skip="true"` to it.\n'
                            f"Examples:\n"
                            f'  <Label value="Car"/>\n'
                            f'  <Label value="YourLabel" predicted_values="label1,label2"/>\n'
                            f"Labels provided in your labeling config:\n"
                            f"  {str(control.labels_attrs)}\n"
                            f"Available '{instance.model_path}' model labels:\n"
                            f"  {list(instance.model.names.values())}"
                        )
                        continue

                    control_models.append(instance)
                    logger.info(f"Control tag with model detected: {model_class.__name__}")
                    matched = True
                    break
            
            if not matched:
                logger.warning(f"No matching control model found for {control.tag} {control.name}")

        if not control_models:
            control_tags = ", ".join([c.type for c in available_model_classes])
            raise ValueError(
                f"No suitable control tags (e.g. {control_tags} connected to Image or Video object tags) "
                f"detected in the label config:\n{self.label_config}"
            )

        return control_models

    def predict(
        self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs
    ) -> ModelResponse:
        """Run YOLO predictions on the tasks
        :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
        :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create)
        :return model_response
            ModelResponse(predictions=predictions) with
            predictions [Predictions array in JSON format]
            (https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        logger.info(
            f"Run prediction on {len(tasks)} tasks, project ID = {self.project_id}"
        )
        control_models = self.detect_control_models()

        predictions = []
        for task in tasks:

            regions = []
            for model in control_models:
                try:
                    path = model.get_path(task)
                    regions += model.predict_regions(path)
                except Exception as e:
                    logger.error(f"Error in prediction for {model.__class__.__name__}: {e}")
                    continue

            # calculate final score
            all_scores = [region["score"] for region in regions if "score" in region]
            avg_score = sum(all_scores) / max(len(all_scores), 1)

            # compose final prediction
            prediction = {
                "result": regions,
                "score": avg_score,
                "model_version": self.model_version,
            }
            
            # 添加置信度显示到meta字段
            if self.show_confidence:
                for region in prediction.get("result", []):
                    if "score" in region and "value" in region:
                        score = region["score"]
                        labels = region["value"].get("rectanglelabels", ["Unknown"])
                        label = labels[0] if labels else "Unknown"
                        if "meta" not in region:
                            region["meta"] = {}
                        region["meta"]["text"] = [f"{label}: {score:.1%}"]
                        logger.debug(f"Added confidence display: {label}: {score:.1%}")
            
            predictions.append(prediction)

        return ModelResponse(predictions=predictions)

    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated.
        Now supports fine-tuning for models that implement it.
        """
        logger.info(f"Fit method called with event: {event}, project_id: {self.project_id}")
        
        results = {}
        control_models = self.detect_control_models()
        
        for model in control_models:
            try:
                logger.info(f"Calling fit on {model.__class__.__name__}")
                training_result = model.fit(event, data, **kwargs)
                results[model.from_name] = training_result
                
                # 记录训练结果
                if training_result:
                    logger.info(f"Training successful for {model.__class__.__name__}")
                else:
                    logger.info(f"Training skipped for {model.__class__.__name__}")
                    
            except Exception as e:
                logger.error(f"Training failed for {model.__class__.__name__}: {e}", exc_info=True)
                results[model.from_name] = False

        logger.info(f"Fit method results: {results}")