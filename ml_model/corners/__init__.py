"""
Corner Kick modeling package within the soccer-predictions project.
Contains feature extraction, XGBoost routine/outcome models, and training pipelines.
"""
from .models import CornerRoutineXGB, CornerOutcomeXGB

__all__ = ["CornerRoutineXGB", "CornerOutcomeXGB"]
