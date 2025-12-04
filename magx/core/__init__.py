"""
Core infrastructure for MagX
- Model wrapper abstraction
- Base explainer interfaces
"""

from .model import ModelWrapper
from .base_explainer import ExplanationResult

__all__ = [
    "ModelWrapper",
    "ExplanationResult",
]
