"""Shared data loaders for the variational classifier weeks (15-17, 21).

Iris-2-class returns float32 features and float32 {0,1} labels, deterministic
80/20 split. We standardise per-feature on the train split only — same
preprocessing for the quantum and classical baselines so the comparison is fair.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def iris_two_class(class_a=0, class_b=1, test_size=0.2, seed=0, scale=True):
    """Iris setosa (0) vs versicolor (1) by default — linearly separable;
    a stress test of encodings, not of model capacity. All four features kept.

    Returns (X_train, y_train, X_test, y_test) with X in float32 and y in {0,1}.
    """
    iris = load_iris()
    mask = (iris.target == class_a) | (iris.target == class_b)
    X = iris.data[mask].astype(np.float32)
    y_raw = iris.target[mask]
    y = np.where(y_raw == class_a, 0, 1).astype(np.float32)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    if scale:
        mu = X_tr.mean(axis=0)
        sd = X_tr.std(axis=0) + 1e-8
        X_tr = (X_tr - mu) / sd
        X_te = (X_te - mu) / sd
    return X_tr, y_tr, X_te, y_te


def angle_normalise(X, scale=np.pi / 2):
    """Map standardised features into [-scale, scale] for angle encoding via tanh."""
    return scale * np.tanh(X)
