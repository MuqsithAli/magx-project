from __future__ import annotations
from typing import Optional, Literal, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from .core.model import ModelWrapper
from .core.base_explainer import ExplanationResult
from .explainers.global_permutation import PermutationImportanceExplainer
from .explainers.local_lime import LimeLikeExplainer
from .viz import plot_global_importance, plot_local_contributions

from .eval import (
    local_faithfulness_deletion,
    local_sparsity,
    local_topk_coverage,
)


GlobalMethod = Literal["permutation"]
LocalMethod = Literal["lime"]


class MagXExplainer:
    """
    High-level interface for MagX explainability.

    Usage:
    ------
    magx = MagXExplainer(model, X_train, y_train, feature_names=...)
    global_exp = magx.explain_global(method="permutation")
    local_exp = magx.explain_local(x_instance, method="lime")
    """

    def __init__(
        self,
        model: Any,
        X_train: pd.DataFrame | np.ndarray,
        y_train: Optional[np.ndarray] = None,
        feature_names: Optional[list[str]] = None,
        task_type: str = "auto",
        class_names: Optional[list[str]] = None,
    ) -> None:

        # Wrap in consistent model protocol
        self.wrapper = ModelWrapper(
            model,
            task_type=task_type,
            class_names=class_names,
        )

        # Standardize X_train as DataFrame with column names
        if isinstance(X_train, pd.DataFrame):
            self.X_train = X_train.reset_index(drop=True)
            self.feature_names = feature_names or list(self.X_train.columns)
        else:
            X_train = np.asarray(X_train)
            self.feature_names = (
                feature_names or [f"f{i}" for i in range(X_train.shape[1])]
            )
            self.X_train = pd.DataFrame(X_train, columns=self.feature_names)

        self.y_train = y_train
        self.class_names = class_names

        # Lazy initialization for explainers
        self._global_explainers: Dict[str, Any] = {}
        self._local_explainers: Dict[str, Any] = {}

    # ==============================================================
    # GLOBAL EXPLANATIONS
    # ==============================================================

    def explain_global(
        self,
        method: GlobalMethod = "permutation",
    ) -> ExplanationResult:

        if method == "permutation":
            if method not in self._global_explainers:
                explainer = PermutationImportanceExplainer()
                explainer.fit(
                    model=self.wrapper,
                    X=self.X_train,
                    y=self.y_train,
                    feature_names=self.feature_names,
                )
                self._global_explainers[method] = explainer

            return self._global_explainers[method].explain_global()

        raise ValueError(f"Unsupported global method: {method}")
    
    # ==============================================================
    # GLOBAL TEXT EXPLANATION
    # ==============================================================

    def explain_global_text(
        self,
        method: GlobalMethod = "permutation",
        top_k: int = 5,
    ) -> str:
        """
        Generate a natural-language summary of global feature importance.

        Parameters
        ----------
        method : str
            Global explanation method. Currently only 'permutation'.
        top_k : int
            Number of most important features to mention explicitly.

        Returns
        -------
        text : str
            Human-readable explanation.
        """
        exp = self.explain_global(method=method)
        importances = exp.values
        meta = exp.meta

        feature_names: List[str] = meta.get("feature_names", [])
        std = meta.get("std", None)
        base_score = meta.get("base_score", None)

        # Truncate to top_k
        pairs: List[Tuple[str, float]] = list(zip(feature_names, importances))
        pairs_sorted = sorted(pairs, key=lambda t: t[1], reverse=True)
        pairs_sorted = pairs_sorted[:top_k]

        lines: List[str] = []

        # Intro sentence
        if self.wrapper.is_classification():
            lines.append(
                "Globally, this model's classification decisions are most influenced "
                "by the following features:"
            )
        else:
            lines.append(
                "Globally, this regression model's predictions are most influenced "
                "by the following features:"
            )

        # Feature bullet-style explanations
        for name, score in pairs_sorted:
            lines.append(f"- {name}: importance {score:.3f} (larger score = stronger influence)")

        # Brief interpretation paragraph
        if self.wrapper.is_classification():
            tail = (
                "These importance scores measure how much the model's accuracy drops "
                "when each feature is randomly permuted. Features with higher scores "
                "are those the model relies on most to distinguish between classes. "
                "Note that this reflects association learned by the model, not causal effect."
            )
        else:
            tail = (
                "These importance scores measure how much the model's error increases "
                "when each feature is randomly permuted. Features with higher scores "
                "are those the model relies on most to adjust its predicted values. "
                "As always, this reflects patterns in the data and does not imply causation."
            )

        lines.append("")
        lines.append(tail)

        if base_score is not None:
            if self.wrapper.is_classification():
                lines.append(f"\nBaseline validation score (before permutation): {base_score:.3f}")
            else:
                lines.append(
                    f"\nBaseline validation score (negative MSE, higher is better): {base_score:.3f}"
                )

        return "\n".join(lines)


    # ==============================================================
    # LOCAL EXPLANATIONS
    # ==============================================================

    def explain_local(
        self,
        x: np.ndarray | pd.Series,
        method: LocalMethod = "lime",
        instance_id: Optional[Any] = None,
    ) -> ExplanationResult:

        # Standardize instance to 1D numpy vector
        if isinstance(x, pd.Series):
            x_vec = x.values
        else:
            x_vec = np.asarray(x)

        if method == "lime":
            if method not in self._local_explainers:
                explainer = LimeLikeExplainer()
                explainer.fit(
                    model=self.wrapper,
                    X_background=self.X_train,
                    feature_names=self.feature_names,
                )
                self._local_explainers[method] = explainer

            return self._local_explainers[method].explain_local(
                x_vec,
                instance_id=instance_id,
            )

        raise ValueError(f"Unsupported local method: {method}")
    
     # ==============================================================
    # LOCAL TEXT EXPLANATION
    # ==============================================================

    def explain_local_text(
        self,
        x: np.ndarray | pd.Series,
        method: LocalMethod = "lime",
        instance_id: Optional[Any] = None,
        top_k: int = 5,
    ) -> str:
        """
        Generate a natural-language explanation for a single instance.

        Parameters
        ----------
        x : array-like or pd.Series
            Instance to explain.
        method : str
            Local explainer method. Currently only 'lime'.
        instance_id : Any, optional
            Optional identifier for the instance (e.g., index).
        top_k : int
            Number of most influential features to mention.

        Returns
        -------
        text : str
            Human-readable explanation of local contributions.
        """
        # Get local explanation
        exp = self.explain_local(
            x=x,
            method=method,
            instance_id=instance_id,
        )

        contribs = np.asarray(exp.values)
        feature_names: List[str] = exp.meta.get("feature_names", [])

        # Sort by absolute influence
        pairs: List[Tuple[str, float]] = list(zip(feature_names, contribs))
        pairs_sorted = sorted(pairs, key=lambda t: abs(t[1]), reverse=True)
        pairs_sorted = pairs_sorted[:top_k]

        positive_factors = [(n, v) for n, v in pairs_sorted if v > 0]
        negative_factors = [(n, v) for n, v in pairs_sorted if v < 0]

        # Compute model prediction for the instance
        if isinstance(x, pd.Series):
            x_vec = x.values
        else:
            x_vec = np.asarray(x)

        # Use DataFrame to avoid "feature names" warnings
        x_df = pd.DataFrame([x_vec], columns=self.feature_names)
        pred = self.wrapper.predict(x_df)[0]

        proba_str = ""
        if self.wrapper.is_classification():
            probs = self.wrapper.predict_proba(x_df)
            if probs is not None:
                # Get probability of predicted class
                pred_idx = int(pred)
                prob_pred = probs[0][pred_idx]
                # Map to class name if available
                label = (
                    self.class_names[pred_idx]
                    if self.class_names is not None and pred_idx < len(self.class_names)
                    else str(pred)
                )
                proba_str = (
                    f"The model predicts class '{label}' with estimated probability "
                    f"around {prob_pred:.2f}.\n"
                )
            else:
                label = str(pred)
                proba_str = f"The model predicts class '{label}'.\n"
        else:
            proba_str = f"The model predicts a value of approximately {float(pred):.3f}.\n"

        # Build explanation text
        lines: List[str] = []

        if instance_id is not None:
            lines.append(f"Local explanation for instance {instance_id}:")
        else:
            lines.append("Local explanation for the given instance:")

        lines.append("")
        lines.append(proba_str.strip())
        lines.append("")

        if positive_factors:
            lines.append("Features pushing the prediction *upwards* (toward the positive outcome):")
            for name, val in positive_factors:
                lines.append(f"- {name}: contribution +{val:.3f}")
        else:
            lines.append("No strong positive contributions among the top features.")

        lines.append("")

        if negative_factors:
            lines.append("Features pushing the prediction *downwards* (away from the positive outcome):")
            for name, val in negative_factors:
                lines.append(f"- {name}: contribution {val:.3f}")
        else:
            lines.append("No strong negative contributions among the top features.")

        lines.append("")
        lines.append(
            "These contributions come from a local linear surrogate model fitted in the neighbourhood "
            "of this instance. Larger absolute values indicate stronger influence on the final prediction. "
            "The explanation is faithful to the model's local behaviour, but should not be interpreted as a "
            "causal statement about the underlying real-world process."
        )

        return "\n".join(lines)


    # ==============================================================
    # PLOTTING HELPERS
    # ==============================================================

    def plot_global(
        self,
        method: GlobalMethod = "permutation",
        top_k: Optional[int] = None,
        theme: str = "light",
        ax: Optional[Any] = None,
    ):
        """
        Convenience wrapper to plot global explanations.
        """
        explanation = self.explain_global(method=method)
        return plot_global_importance(
            explanation=explanation,
            top_k=top_k,
            theme=theme,
            ax=ax,
        )

    def plot_local(
        self,
        x: np.ndarray | pd.Series,
        method: LocalMethod = "lime",
        instance_id: Optional[Any] = None,
        top_k: int = 10,
        theme: str = "light",
        ax: Optional[Any] = None,
    ):
        """
        Convenience wrapper to plot local explanations for a single instance.
        """
        explanation = self.explain_local(
            x=x,
            method=method,
            instance_id=instance_id,
        )
        return plot_local_contributions(
            explanation=explanation,
            top_k=top_k,
            theme=theme,
            ax=ax,
        )

    # ==============================================================
    # LOCAL EXPLANATION EVALUATION
    # ==============================================================

    def evaluate_local(
        self,
        x: np.ndarray | pd.Series,
        method: LocalMethod = "lime",
        instance_id: Optional[Any] = None,
        steps: int = 10,
        top_k: int = 5,
    ) -> Dict[str, float]:
        """
        Compute basic evaluation metrics for a local explanation.

        Metrics:
        - faithfulness_deletion: does removing important features
          actually change the model output?
        - sparsity: how many features are effectively used?
        - topk_coverage: how much of the explanation is concentrated
          in the top-k features?
        """
        exp = self.explain_local(
            x=x,
            method=method,
            instance_id=instance_id,
        )

        faith = local_faithfulness_deletion(
            model=self.wrapper,
            x=x,
            explanation=exp,
            X_background=self.X_train,
            steps=steps,
        )
        spars = local_sparsity(exp)
        cov = local_topk_coverage(exp, top_k=top_k)

        return {
            "faithfulness_deletion": float(faith),
            "sparsity": float(spars),
            "topk_coverage": float(cov),
        }
