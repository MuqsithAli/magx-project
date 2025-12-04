from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

import numpy as np
import pandas as pd

from .model import ModelWrapper


@dataclass
class ExplanationResult:
    """
    Container for explanation outputs.

    Attributes
    ----------
    values : Any
        Numeric explanation values (e.g., importances, contributions).
    meta : Dict[str, Any]
        Metadata describing the explanation (feature names, method, etc.)
    """
    values: Any
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"values": self.values, "meta": self.meta}


class GlobalExplainer(Protocol):
    """
    Base interface for global explainers.
    """

    def fit(
        self,
        model: ModelWrapper,
        X: pd.DataFrame | np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[list[str]] = None,
    ) -> None:
        ...

    def explain_global(self) -> ExplanationResult:
        ...


class LocalExplainer(Protocol):
    """
    Base interface for local explainers.
    """

    def fit(
        self,
        model: ModelWrapper,
        X_background: pd.DataFrame | np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> None:
        """
        X_background is used to create local distributions for instance explanation.
        """
        ...

    def explain_local(
        self,
        x: np.ndarray | pd.Series,
        instance_id: Optional[Any] = None,
    ) -> ExplanationResult:
        ...
