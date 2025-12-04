from __future__ import annotations
from typing import Optional, List, Tuple

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


def plot_local_contributions(
    explanation: ExplanationResult,
    top_k: int = 10,
    theme: str = "light",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot local feature contributions for a single instance.

    Parameters
    ----------
    explanation : ExplanationResult
        Output from MagXExplainer.explain_local().
    top_k : int
        Number of highest-absolute-contribution features to show.
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

    values = np.asarray(explanation.values)
    meta = explanation.meta
    feature_names: List[str] = meta.get("feature_names", [f"f{i}" for i in range(len(values))])

    # Sort by absolute contribution
    pairs: List[Tuple[str, float]] = list(zip(feature_names, values))
    pairs_sorted = sorted(pairs, key=lambda t: abs(t[1]), reverse=True)
    pairs_sorted = pairs_sorted[:top_k]

    names = [p[0] for p in pairs_sorted][::-1]   # reverse for barh
    contribs = np.array([p[1] for p in pairs_sorted])[::-1]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(names))))

    y_pos = np.arange(len(names))

    ax.barh(y_pos, contribs)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Local Contribution (surrogate coefficient)")
    ax.set_title("Local Feature Contributions (LIME-like)")

    plt.tight_layout()
    return ax
