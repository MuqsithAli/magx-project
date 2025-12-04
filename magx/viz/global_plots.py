from __future__ import annotations
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ..core.base_explainer import ExplanationResult


def _apply_theme(theme: str = "light") -> None:
    """
    Apply a simple light/dark theme using seaborn/matplotlib.
    """
    theme = (theme or "light").lower()
    if theme == "dark":
        sns.set_theme(style="darkgrid")
        plt.rcParams["figure.facecolor"] = "#121212"
        plt.rcParams["axes.facecolor"] = "#121212"
        plt.rcParams["axes.edgecolor"] = "white"
        plt.rcParams["text.color"] = "white"
        plt.rcParams["axes.labelcolor"] = "white"
        plt.rcParams["xtick.color"] = "white"
        plt.rcParams["ytick.color"] = "white"
    else:
        sns.set_theme(style="whitegrid")
        plt.rcParams["figure.facecolor"] = "white"
        plt.rcParams["axes.facecolor"] = "white"
        plt.rcParams["axes.edgecolor"] = "black"
        plt.rcParams["text.color"] = "black"
        plt.rcParams["axes.labelcolor"] = "black"
        plt.rcParams["xtick.color"] = "black"
        plt.rcParams["ytick.color"] = "black"


def plot_global_importance(
    explanation: ExplanationResult,
    top_k: Optional[int] = None,
    theme: str = "light",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot global feature importance from a permutation explanation result.

    Parameters
    ----------
    explanation : ExplanationResult
        Output from MagXExplainer.explain_global().
    top_k : Optional[int]
        If set, only the top_k most important features are plotted.
    theme : str
        'light' or 'dark'.
    ax : Optional[plt.Axes]
        Existing matplotlib Axes to draw on. If None, a new figure+axes is created.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes containing the plot.
    """
    _apply_theme(theme)

    importances = np.asarray(explanation.values)
    meta = explanation.meta
    feature_names = meta.get("feature_names", [f"f{i}" for i in range(len(importances))])
    std = np.asarray(meta.get("std", np.zeros_like(importances)))

    # Reduce to top_k if requested
    if top_k is not None and top_k < len(importances):
        importances = importances[:top_k]
        std = std[:top_k]
        feature_names = feature_names[:top_k]

    # Horizontal bar plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(feature_names))))

    y_pos = np.arange(len(feature_names))

    ax.barh(y_pos, importances, xerr=std)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    ax.invert_yaxis()  # most important on top
    ax.set_xlabel("Importance (score drop on permutation)")
    ax.set_title("Global Feature Importance (Permutation)")

    plt.tight_layout()
    return ax
