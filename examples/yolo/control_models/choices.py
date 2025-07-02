"""
Choices control model for Label Studio
This is a placeholder implementation
"""
import logging
from typing import List, Dict
from control_models.base import ControlModel

logger = logging.getLogger(__name__)


class ChoicesModel(ControlModel):
    """
    Choices control model - placeholder implementation
    """
    type = "Choices"
    
    @classmethod
    def is_control_matched(cls, control) -> bool:
        """Check if control tag matches Choices"""
        return control.tag == cls.type
    
    def predict_regions(self, path) -> List[Dict]:
        """
        Placeholder predict method for Choices
        Returns empty list as this is not implemented
        """
        logger.debug(f"ChoicesModel.predict_regions called for {path}")
        return []
    
    def fit(self, event, data, **kwargs):
        """
        Placeholder fit method
        """
        logger.debug(f"ChoicesModel.fit called with event: {event}")
        return True
