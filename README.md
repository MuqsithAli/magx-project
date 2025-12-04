# MagX — Model-Agnostic Explainability Toolkit

**MagX** is a research-focused, model-agnostic explainability framework for machine learning.  
It provides **global** and **local** interpretability methods that work with *any* black-box model  
(Classification or Regression), with visualizations and natural-language summaries.

MagX is ideal for:
- Machine learning interpretability research
- Transparency in field deployments
- Educational demos of explainability concepts

---

## Features

| Capability | Status |
|-----------|:-----:|
| Global explainability (Permutation Importance) | ✔ |
| Local explainability (LIME-style surrogate) | ✔ |
| Model-agnostic (works with any `.predict()`) | ✔ |
| Natural-language explanations | ✔ |
| Visualizations (light/dark themes) | ✔ |
| Local explanation evaluation metrics | ✔ |
| Titanic demo notebook | ✔ (coming soon) |
| Counterfactual explanations | Planned |

---

## 🚀 Quickstart

This example demonstrates how to train a model, generate global and local
explanations, and visualize feature contributions using MagX.

```
from magx import MagXExplainer
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train any model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Initialize MagX
magx = MagXExplainer(
    model=model,
    X_train=X_train,
    y_train=y_train,
    feature_names=list(X_train.columns),
    class_names=list(data.target_names),
)

# Global explanation — importance ranking
print(magx.explain_global_text(top_k=5))
magx.plot_global(top_k=10, theme="light")
plt.show()

# Local explanation — single instance
x0 = X_test.iloc[0]
print(magx.explain_local_text(x0, top_k=5))
magx.plot_local(x0, top_k=10, theme="dark")
plt.show()
```

---

## Visualizations

MagX supports light and dark themes:

```
magx.plot_global(theme="dark")
magx.plot_local(x0, theme="light")
```

---

## Evaluation Metrics

Measure local explanation quality:

```
scores = magx.evaluate_local(x0)
print(scores)
```

Metrics included:

| Metric | Purpose |
|--------|:---------:|
| Faithfulness (deletion) | Does removing important features change prediction?|
| Sparsity | How concise is the explanation?|
| Top-K coverage | How much influence do top features capture?|

---

## Architecture

```
magx/
 ├── core/           # Base interfaces + model wrapper
 ├── explainers/     # Global & local explanation algorithms
 ├── eval/           # Explanation metrics
 ├── viz/            # Visualization helpers
 ├── magx_explainer.py  # Unified interface for users
 └── __init__.py
```

---

## Why MagX?

- No model internals needed → safe for black-box systems

- Consistent API across methods

- Human-readable textual insight

- Built for research reproducibility

---

## Roadmap

- Counterfactual explanations

- Text attribution visualization

- Jupyter notebooks for benchmark datasets

- Packaging automation + CI/CD

- Hyperparameter-based stability analysis

---

## Contributing

Contributions are welcome!
Please open an issue with ideas, bugs, or suggestions.

---

## License

MIT License
Free for commercial and research use.

---

## Citation

If you use MagX in academic work, please cite the repository ❤️
(BibTeX will be added when published).

---

## 📦 Installation

```bash
install latest from GitHub: 
pip install git+https://github.com/MuqsithAli/magx.git
```

---

![GitHub tag (latest SemVer)](https://img.shields.io/github/v/tag/MuqsithAli/magx-project?label=release)
