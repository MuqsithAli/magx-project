from __future__ import annotations
from typing import Optional, Any, List

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from ..core.model import ModelWrapper
from ..core.base_explainer import ExplanationResult


class LimeLikeExplainer:
    """
    Lightweight, model-agnostic local surrogate explainer.
    """

    def __init__(
        self,
        kernel_width: float = 0.75,
        n_samples: int = 500,
        random_state: Optional[int] = 42,
    ) -> None:
        self.kernel_width = kernel_width
        self.n_samples = n_samples
        self.random_state = random_state

        self.model: Optional[ModelWrapper] = None
        self.X_background: Optional[pd.DataFrame] = None
        self.feature_names: Optional[List[str]] = None

    def fit(
        self,
        model: ModelWrapper,
        X_background: pd.DataFrame | np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> None:

        if not isinstance(X_background, pd.DataFrame):
            X_background = pd.DataFrame(
                X_background,
                columns=feature_names
                if feature_names is not None
                else [f"f{i}" for i in range(X_background.shape[1])],
            )

        self.model = model
        self.X_background = X_background.reset_index(drop=True)
        self.feature_names = list(self.X_background.columns)

    def _sample_neighborhood(self, x: np.ndarray) -> np.ndarray:
        """Generate synthetic neighborhood around x."""
        rng = np.random.RandomState(self.random_state)
        mean = x
        std = self.X_background.std().values + 1e-6  # avoid zero std
        return rng.normal(loc=mean, scale=std, size=(self.n_samples, len(x)))

    def _kernel(self, x: np.ndarray, samples: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(samples - x, axis=1)
        weights = np.exp(-(distances**2) / (2 * self.kernel_width**2))

        # Avoid zero-sum weights (numerical stability)
        if weights.sum() == 0:
            weights += 1e-8

        return weights


    def explain_local(
        self,
        x: np.ndarray | pd.Series,
        instance_id: Optional[Any] = None,
    ) -> ExplanationResult:

        if isinstance(x, pd.Series):
            x = x.values
        x = np.asarray(x)

        if self.model is None or self.X_background is None:
            raise RuntimeError("Call fit() before explain_local().")

        # 1️⃣ Sample local data
        neighborhood = self._sample_neighborhood(x)

        # 2️⃣ Model predictions
        neigh_df = pd.DataFrame(neighborhood, columns=self.feature_names)
        preds = self.model.predict(neigh_df)


        # 3️⃣ Kernel weights
        weights = self._kernel(x, neighborhood)

        if np.allclose(weights.sum(), 0):
            weights = np.ones_like(weights)

        weights = weights / (weights.sum() + 1e-12)

        # 4️⃣ Train local surrogate
        model_local = Ridge(alpha=1.0)  # stable linear surrogate
        model_local.fit(neighborhood, preds, sample_weight=weights)

        # 5️⃣ Coefficients = feature attributions
        attributions = model_local.coef_

        return ExplanationResult(
            values=attributions,
            meta={
                "feature_names": self.feature_names,
                "method": "lime_like",
                "instance_id": instance_id,
            },
        )
