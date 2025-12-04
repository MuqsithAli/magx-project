import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from magx import MagXExplainer


def main():
    # 1. Load a simple dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Train a simple model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 3. Wrap with MagX
    magx = MagXExplainer(
        model=model,
        X_train=X_train,
        y_train=y_train,
        feature_names=list(X_train.columns),
        task_type="classification",
        class_names=list(data.target_names),
    )

    # 4. Global explanation (permutation importance)
    print("\n=== GLOBAL EXPLANATION (Permutation Importance) ===")
    global_exp = magx.explain_global(method="permutation")

    importances = global_exp.values
    meta = global_exp.meta
    feat_names = meta["feature_names"]
    std = meta["std"]

    for f, imp, s in zip(feat_names[:10], importances[:10], std[:10]):
        print(f"{f:35s}  importance={imp: .4f}  std={s: .4f}")

    # 5. Local explanation (LIME-like) for a single instance
    print("\n=== LOCAL EXPLANATION (LIME-like) ===")
    x0 = X_test.iloc[0]
    local_exp = magx.explain_local(x0, method="lime", instance_id=0)

    local_attr = local_exp.values
    local_meta = local_exp.meta
    local_features = local_meta["feature_names"]

    # Show top positive / negative contributions
    contrib_pairs = list(zip(local_features, local_attr))
    contrib_pairs_sorted = sorted(contrib_pairs, key=lambda t: abs(t[1]), reverse=True)

    print("Top 10 feature contributions (by absolute value):")
    for name, contrib in contrib_pairs_sorted[:10]:
        print(f"{name:35s}  contrib={contrib: .4f}")

    metrics = magx.evaluate_local(x0, top_k=5)
    print("\n=== LOCAL EXPLANATION METRICS ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    text_global = magx.explain_global_text(top_k=5)
    text_local = magx.explain_local_text(x0, top_k=5)

    print(text_global)
    print(text_local)


    # 6. Plotting tests
    import matplotlib.pyplot as plt

    print("\nShowing global and local plots...")
    magx.plot_global(theme="light", top_k=10)
    plt.show()

    magx.plot_local(x0, theme="dark", top_k=10)
    plt.show()



if __name__ == "__main__":
    main()
