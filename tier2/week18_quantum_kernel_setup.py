"""Week 18 — Quantum kernel setup with the ZZ feature map.

Tier 2 / 2D.1. Working through:
  - the data-encoding -> kernel-matrix pipeline
        K(x, x') = |<phi(x) | phi(x')>|^2,    |phi(x)> = U_ZZ(x) |0...0>
  - ZZFeatureMap (Havlicek et al. 2019, Suzuki et al. 2020):
      H^n - U_Z(x) - H^n - U_Z(x), with U_Z(x) = prod_i exp(i x_i Z_i)
                                                 prod_{i<j} exp(i (pi - x_i)(pi - x_j) Z_i Z_j)
  - building a 40-point Iris (1 vs 2) Gram matrix via FidelityQuantumKernel
  - PSD verification (min eigenvalue ~ 0); spectrum and condition number;
    diagonal = 1 (each |phi(x)> is a normalised pure state)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings

warnings.filterwarnings("ignore")

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

from tier2.utils.data import iris_two_class

N_FEATURES = 4
REPS = 2
N_POINTS = 40   # 20 per class for 5-fold CV in week 19
SEED = 0


def section(title):
    print("\n" + title)
    print("-" * len(title))


def text_heatmap(K, labels, title="K"):
    """Tiny ASCII heat-map of an (n,n) kernel matrix sorted by label."""
    order = np.argsort(labels)
    Ks = K[np.ix_(order, order)]
    ys = labels[order]
    n = Ks.shape[0]
    chars = " .:-=+*#%@"
    K01 = np.clip(Ks, 0, 1)
    print(f"  {title}: rows/cols sorted by label (class 0 first).")
    print(f"  legend: ' ' = 0.0  ...  '@' = 1.0")
    sep = None
    for k in range(1, n):
        if ys[k] != ys[k - 1]:
            sep = k
            break
    for r in range(n):
        row = ""
        for c in range(n):
            row += chars[min(int(K01[r, c] * (len(chars) - 1)), len(chars) - 1)]
            if sep is not None and c == sep - 1:
                row += "|"
        print(f"  {row}")
        if sep is not None and r == sep - 1:
            print(f"  {'-' * (n + 1)}")


def main():
    section("1. Data: 40-point Iris-0-vs-1 subset, MinMax-scaled to [0, pi]")
    # Use the easy pair (setosa vs versicolor) for the kernel pipeline so the
    # SVM in week 19 has a clear signal. Pair (1, 2) is harder and runs into
    # ZZ feature-map kernel concentration -- which week 20 demonstrates by
    # sweeping the reps parameter.
    # MinMax scaling to [0, pi] is the Havlicek-paper convention for the ZZ
    # feature map; it keeps the (pi - x_i)(pi - x_j) terms in [0, pi^2] rather
    # than blowing past 2 pi which destroys the kernel structure.
    X_full, y_full, _, _ = iris_two_class(
        class_a=0, class_b=1, seed=SEED, scale=False
    )
    X_full = MinMaxScaler(feature_range=(0.0, float(np.pi))).fit_transform(
        X_full
    ).astype(np.float32)
    # take 20 per class for an evenly-balanced kernel block
    idx0 = np.where(y_full == 0)[0][:20]
    idx1 = np.where(y_full == 1)[0][:20]
    keep = np.concatenate([idx0, idx1])
    X = X_full[keep].astype(np.float32)
    y = y_full[keep].astype(np.int64)
    print(f"  shape: {X.shape}, class balance: {[int((y == c).sum()) for c in (0, 1)]}")

    section("2. ZZFeatureMap construction")
    fm = ZZFeatureMap(feature_dimension=N_FEATURES, reps=REPS)
    print(f"  qubits     : {fm.num_qubits}")
    print(f"  parameters : {fm.num_parameters} (one per feature)")
    print(f"  reps       : {REPS} (alternating H^n - U_Z(x) layers)")
    print(f"  depth      : {fm.decompose().depth()}")

    section("3. Compute K = |<phi(x)|phi(x')>|^2 over all 40x40 pairs")
    qk = FidelityQuantumKernel(feature_map=fm)
    import time
    t0 = time.time()
    K = qk.evaluate(X)
    elapsed = time.time() - t0
    print(f"  evaluated  : {K.shape[0]}*{K.shape[1]} entries in {elapsed:.2f} s")
    print(f"  symmetry   : ||K - K^T||_inf = {np.abs(K - K.T).max():.2e}")
    print(f"  diagonal   : min {np.diag(K).min():.6f},  max {np.diag(K).max():.6f}")
    print(f"  K range    : [{K.min():.4f}, {K.max():.4f}]")

    section("4. PSD check: full eigenvalue spectrum")
    evals = np.sort(np.linalg.eigvalsh(K))[::-1]
    print(f"  largest 5  : {evals[:5]}")
    print(f"  smallest 5 : {evals[-5:]}")
    print(f"  min eig    : {evals[-1]:.3e}  (PSD threshold > -1e-8)")
    rank_eps = float((evals > 1e-6 * evals[0]).sum())
    cond = float(evals[0] / max(evals[-1], 1e-30))
    print(f"  effective rank (>1e-6 of lambda_max): {rank_eps:.0f} of {len(evals)}")
    print(f"  condition number lambda_max/lambda_min: {cond:.2e}")

    section("5. Block structure: in-class vs cross-class similarity")
    mask_same = (y[:, None] == y[None, :]) & (np.eye(len(y), dtype=bool) == False)
    mask_diff = y[:, None] != y[None, :]
    mean_same = float(K[mask_same].mean())
    mean_diff = float(K[mask_diff].mean())
    print(f"  mean K within-class  (off-diagonal) : {mean_same:.4f}")
    print(f"  mean K between-class                : {mean_diff:.4f}")
    print(f"  within - between gap                : {mean_same - mean_diff:+.4f}")
    print(f"  -> a positive gap is what makes the SVM in week 19 work")

    section("6. ASCII heat-map (rows/cols sorted by class)")
    text_heatmap(K, y, title="K (class 0 block | class 1 block)")

    section("Checkpoint assertions")
    # symmetric within numerical precision
    assert np.abs(K - K.T).max() < 1e-9, "K not symmetric"
    # diagonal essentially 1
    assert abs(float(np.diag(K).min()) - 1.0) < 1e-6, "K diagonal != 1"
    # PSD up to numerical noise
    assert evals[-1] > -1e-8, f"K not PSD: min eig {evals[-1]}"
    # in-class > between-class -> kernel separates the labels
    assert mean_same > mean_diff, (
        f"within-class similarity ({mean_same:.4f}) <= "
        f"between-class ({mean_diff:.4f}) -> SVM unlikely to work"
    )
    print(f"  PASS: K is PSD (min eig {evals[-1]:.2e}); "
          f"in-class > between-class by {(mean_same - mean_diff)*100:.1f} pp.")


if __name__ == "__main__":
    main()
