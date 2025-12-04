from __future__ import annotations
from typing import Protocol, Any, Optional
import numpy as np


class PredictOnlyModel(Protocol):
    def predict(self, X: Any) -> Any:
        ...


class ProbabilisticModel(PredictOnlyModel, Protocol):
    def predict_proba(self, X: Any) -> Any:
        ...


class ModelWrapper:
    def __init__(
        self,
        model: Any,
        task_type: str = "auto",
        class_names: Optional[list[str]] = None,
    ) -> None:
        self.model = model
        self.task_type = task_type
        self.class_names = class_names
        if self.task_type == "auto":
            self.task_type = self._infer_task_type()

    def _infer_task_type(self) -> str:
        if hasattr(self.model, "predict_proba"):
            return "classification"
        return "regression"

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.model.predict(X))

    def predict_proba(self, X: Any) -> Optional[np.ndarray]:
        if hasattr(self.model, "predict_proba"):
            return np.asarray(self.model.predict_proba(X))
        return None

    def is_classification(self) -> bool:
        return self.task_type == "classification"

    def is_regression(self) -> bool:
        return self.task_type == "regression"
