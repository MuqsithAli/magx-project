from __future__ import annotations
from typing import Union, List

import numpy as np
import pandas as pd

from ..core.model import ModelWrapper
from ..core.base_explainer import ExplanationResult


def local_sparsity(explanation: ExplanationResult) -> float:
    """
    Measure how sparse a local explanation is.

    Returns a value in [0, 1], where:
    - 1.0 means very sparse (most contributions are near zero)
    - 0.0 means dense (most features contribute significantly)
    """
    values = np.asarray(explanation.values, dtype=float)
    if values.size == 0:
        return 0.0

    abs_vals = np.abs(values)
    max_abs = abs_vals.max()
    if max_abs == 0:
        # All zero -> perfectly sparse by construction
        return 1.0

    # Define a relative threshold: below 1% of max is effectively zero
    threshold = 0.01 * max_abs
    nonzero = (abs_vals > threshold).sum()
    sparsity = 1.0 - nonzero / float(len(values))
    return float(sparsity)


def local_topk_coverage(
    explanation: ExplanationResult,
    top_k: int = 5,
) -> float:
    """
    Fraction of total absolute contribution captured by the top-k features.
    Value in [0, 1], higher is better (more concentrated explanations).
    """
    values = np.asarray(explanation.values, dtype=float)
    if values.size == 0:
        return 0.0

    abs_vals = np.abs(values)
    total = abs_vals.sum()
    if total == 0:
        return 0.0

    k = min(top_k, len(values))
    topk_sum = np.sort(abs_vals)[::-1][:k].sum()
    return float(topk_sum / total)


def local_faithfulness_deletion(
    model: ModelWrapper,
    x: Union[np.ndarray, pd.Series],
    explanation: ExplanationResult,
    X_background: pd.DataFrame,
    steps: int = 10,
) -> float:
    """
    Simple deletion-based local faithfulness metric.

    Intuition:
    - Rank features by |contribution|
    - Gradually replace the most important features with baseline values
      (feature-wise mean from X_background)
    - Measure how much the model prediction drops on average.

    Returns a scalar in R (higher = more faithful, negative possible if
    deletion sometimes increases the score).
    """
    feature_names: List[str] = explanation.meta.get("feature_names", [])
    contribs = np.asarray(explanation.values, dtype=float)

    if len(feature_names) == 0 or contribs.size == 0:
        return 0.0

    # Ensure background has the same columns
    X_bg = X_background[feature_names]
    baseline = X_bg.mean(axis=0)  # pd.Series

    # Order features by absolute contribution
    order = np.argsort(np.abs(contribs))[::-1]
    n_features = len(feature_names)

    # Represent x as a Series aligned with feature_names
    if isinstance(x, pd.Series):
        x_series = x[feature_names]
    else:
        x_series = pd.Series(np.asarray(x), index=feature_names)

    # Base prediction
    x_df = pd.DataFrame([x_series.values], columns=feature_names)

    if model.is_classification():
        probs = model.predict_proba(x_df)
        if probs is not None:
            pred_class = int(np.argmax(probs[0]))
            base = float(probs[0][pred_class])
        else:
            # Fallback: treat predicted label as score (less ideal)
            base = float(model.predict(x_df)[0])
            pred_class = None
    else:
        base = float(model.predict(x_df)[0])
        pred_class = None

    if np.isclose(base, 0.0):
        denom = 1.0
    else:
        denom = abs(base)

    drops = []

    for s in range(1, steps + 1):
        k = max(1, int(round(s * n_features / steps)))
        idx_to_replace = order[:k]

        x_mod = x_series.copy()
        for j in idx_to_replace:
            feat = feature_names[j]
            x_mod[feat] = baseline[feat]

        x_mod_df = pd.DataFrame([x_mod.values], columns=feature_names)

        if model.is_classification():
            probs2 = model.predict_proba(x_mod_df)
            if probs2 is not None and pred_class is not None:
                val = float(probs2[0][pred_class])
            else:
                val = float(model.predict(x_mod_df)[0])
        else:
            val = float(model.predict(x_mod_df)[0])

        drop = base - val
        drops.append(drop)

    drops = np.asarray(drops, dtype=float)
    norm_drops = drops / denom
    return float(norm_drops.mean())
