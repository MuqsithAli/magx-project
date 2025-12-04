from __future__ import annotations
from typing import Optional, Callable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score

from ..core.model import ModelWrapper
from ..core.base_explainer import ExplanationResult


class PermutationImportanceExplainer:
    """
    Model-agnostic Permutation Feature Importance.
    """

    def __init__(
        self,
        n_repeats: int = 5,
        random_state: Optional[int] = 42,
        scoring: Optional[Callable] = None,
    ) -> None:
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.scoring = scoring

        self.model: Optional[ModelWrapper] = None
        self.X: Optional[pd.DataFrame] = None
        self.y: Optional[np.ndarray] = None
        self.feature_names: Optional[List[str]] = None
        self.importances_: Optional[np.ndarray] = None
        self.importances_std_: Optional[np.ndarray] = None

    def fit(
        self,
        model: ModelWrapper,
        X: pd.DataFrame | np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(
                X,
                columns=feature_names
                if feature_names is not None
                else [f"f{i}" for i in range(X.shape[1])],
            )

        self.model = model
        self.X = X.reset_index(drop=True)
        self.y = y
        self.feature_names = list(self.X.columns)

    def _default_scoring(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> float:
        """Pick default scoring based on task type."""
        if self.model.is_classification():
            return accuracy_score(y_true, y_pred)
        else:
            # Lower MSE = better → convert to positive score
            return -mean_squared_error(y_true, y_pred)

    def explain_global(self) -> ExplanationResult:
        if self.model is None or self.X is None:
            raise RuntimeError("fit() must be called before explain_global().")

        # Base score (unmodified dataset)
        base_preds = self.model.predict(self.X)
        scorer = self.scoring or self._default_scoring
        base_score = scorer(self.y, base_preds)

        rng = np.random.RandomState(self.random_state)
        n_features = len(self.feature_names)

        importances = np.zeros((self.n_repeats, n_features))

        for repeat in range(self.n_repeats):
            for i, feat in enumerate(self.feature_names):
                X_perm = self.X.copy()
                X_perm[feat] = rng.permutation(X_perm[feat].values)

                preds_perm = self.model.predict(X_perm)
                perm_score = scorer(self.y, preds_perm)

                importances[repeat, i] = base_score - perm_score

        # Compute mean reduction and std for feature ranking
        mean_importances = importances.mean(axis=0)
        std_importances = importances.std(axis=0)

        # Sort by importance descending
        order = np.argsort(mean_importances)[::-1]
        mean_importances = mean_importances[order]
        std_importances = std_importances[order]
        sorted_features = [self.feature_names[i] for i in order]

        self.importances_ = mean_importances
        self.importances_std_ = std_importances

        return ExplanationResult(
            values=mean_importances,
            meta={
                "feature_names": sorted_features,
                "std": std_importances,
                "method": "permutation_importance",
                "n_repeats": self.n_repeats,
                "base_score": base_score,
            },
        )
