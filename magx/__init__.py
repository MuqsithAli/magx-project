"""
MagX - Model-Agnostic Explanations

High-level entry point for explanation workflows.
"""

from .core.model import ModelWrapper
from .core.base_explainer import ExplanationResult
from .magx_explainer import MagXExplainer

__all__ = [
    "ModelWrapper",
    "ExplanationResult",
    "MagXExplainer",
]
