"""Week 20 — Wider quantum-vs-RBF kernel benchmark.

Tier 2 / 2D.3. Working through:
  - sweeping the ZZ-feature-map depth (reps) in {1, 2, 3} and the training-set
    size in {20, 40, 80}, with RBF on the same splits as the control
  - 5-fold CV mean and held-out test accuracy reported for every (depth, n)
    cell; results dumped to a CSV alongside this script
  - accuracy vs depth at the committed input scaling (MinMax to [0, pi])
  - text plots of accuracy vs n_train (one curve per depth) and accuracy vs
    depth (one curve per n_train); RBF and quantum overlaid

POST-HOC control (added after the [0, pi] results above were written up,
so it is not part of the original design and is not a pass gate):
  - the off-diagonal mean and std of every training Gram matrix, so the
    spread of kernel values is measured rather than inferred from accuracy
  - the same quantum sweep (same splits, C grid, CV and seed) with the
    ZZ-map inputs multiplied by a fixed factor POSTHOC_SCALE = 0.1. The
    value comes from a review probe that tried this one factor (no search
    over factors, no selection on test data); its outcome was known before
    this control was added. The factor is the kernel bandwidth of
    Shaydulin & Wild, "Importance of Kernel Bandwidth in Quantum Machine
    Learning", arXiv:2111.05451: a smaller factor makes the kernel vary
    more slowly with the inputs.
  CSV columns for this control start with "posthoc_".
"""

import sys
import os
import csv
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings

warnings.filterwarnings("ignore")

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics.pairwise import rbf_kernel

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

from tier2.utils.data import iris_two_class

DEPTHS = (1, 2, 3)
N_TRAIN_SIZES = (20, 40, 80)
N_TEST = 20
C_GRID = (0.1, 1.0, 10.0, 100.0)
N_CV_FOLDS = 5
SEED = 0
# POST-HOC control (see docstring): input scale for the second quantum sweep.
POSTHOC_SCALE = 0.1
# Off-diagonal fidelity statistics for independent Haar-random states in
# dimension d = 2^4 (fidelity ~ Beta(1, d - 1)); printed as a reference.
HAAR_D = 2 ** 4
HAAR_MEAN = 1.0 / HAAR_D
HAAR_STD = float(np.sqrt((HAAR_D - 1) / (HAAR_D ** 2 * (HAAR_D + 1))))
PH = f"posthoc_x{POSTHOC_SCALE}_"   # CSV prefix for the post-hoc sweep
CSV_FIELDS = [
    "depth", "n_train", "kernel", "best_C", "cv_mean", "cv_std",
    "test_acc", "wall_s",
    "posthoc_offdiag_mean", "posthoc_offdiag_std",
    PH + "best_C", PH + "cv_mean", PH + "cv_std", PH + "test_acc",
    PH + "offdiag_mean", PH + "offdiag_std",
]
CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "week20_results.csv"
)


def section(title):
    print("\n" + title)
    print("-" * len(title))


def cv_best(K, y, n_folds, seed):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    best_C, best_m, best_s = None, -1.0, 0.0
    for C in C_GRID:
        accs = []
        for tr, va in skf.split(np.zeros_like(y), y):
            clf = SVC(kernel="precomputed", C=C).fit(
                K[np.ix_(tr, tr)], y[tr]
            )
            accs.append(float(clf.score(K[np.ix_(va, tr)], y[va])))
        m = float(np.mean(accs))
        if m > best_m:
            best_C, best_m, best_s = C, m, float(np.std(accs, ddof=0))
    return best_C, best_m, best_s


def held_out(K_tr, K_te, y_tr, y_te, C):
    clf = SVC(kernel="precomputed", C=C).fit(K_tr, y_tr)
    return float(clf.score(K_te, y_te))


def offdiag_stats(K):
    """Mean and std (ddof = 0) of the off-diagonal entries of a Gram matrix."""
    off = K[~np.eye(len(K), dtype=bool)]
    return float(off.mean()), float(off.std())


def text_plot(rows, xs, kernel_label, width=44, lo=0.4, hi=1.02):
    """One row per series; 'Q'/'R' marker per (depth or n) cell."""
    print(f"  acc range: {lo:.2f} -> {hi:.2f}    legend: Q=quantum, R=rbf, *=overlap")
    header = f"{'cell':>6s} | {lo:.2f}" + " " * (width // 2 - 4)
    header += f"{(lo + hi) / 2:.2f}" + " " * (width // 2 - 4) + f"{hi:.2f}"
    print(f"  {header}")
    for label, q, r in rows:
        line = [" "] * width
        for ch, val in (("Q", q), ("R", r)):
            j = int(round((val - lo) / (hi - lo) * (width - 1)))
            j = max(0, min(width - 1, j))
            line[j] = ch if line[j] == " " else "*"
        print(f"  {label:>6s} |{''.join(line)}|")


def main():
    section("1. Data and preprocessing setup")
    X_full, y_full, X_te_full, y_te_full = iris_two_class(
        class_a=0, class_b=1, seed=SEED, scale=False
    )
    # Test set (fixed across all cells): 10 per class
    idx_te = np.concatenate(
        [np.where(y_te_full == c)[0][:N_TEST // 2] for c in (0, 1)]
    )
    X_te_raw, y_te = X_te_full[idx_te], y_te_full[idx_te]
    # Train pool: take 40 per class up front, then slice to n_train sizes
    idx_tr = np.concatenate(
        [np.where(y_full == c)[0][:max(N_TRAIN_SIZES) // 2] for c in (0, 1)]
    )
    X_tr_pool, y_tr_pool = X_full[idx_tr], y_full[idx_tr]
    print(f"  train pool: {X_tr_pool.shape}, test: {X_te_raw.shape}")
    print(f"  depths sweep   : {DEPTHS}")
    print(f"  n_train sweep  : {N_TRAIN_SIZES}")
    print(f"  CV folds, C grid: {N_CV_FOLDS}, {C_GRID}")

    section("2. Run the full 3 x 3 sweep (saves to week20_results.csv)")
    print(f"  {'depth':>5s}  {'n_train':>7s}  {'kernel':>7s}  {'best_C':>6s}  "
          f"{'CV mean':>8s}  {'CV std':>7s}  {'test':>6s}  {'time(s)':>7s}")
    rows = []
    for n_train in N_TRAIN_SIZES:
        # Balanced subset of size n_train (n_train // 2 per class)
        per_class = n_train // 2
        idx = np.concatenate(
            [np.where(y_tr_pool == c)[0][:per_class] for c in (0, 1)]
        )
        X_tr_raw, y_tr = X_tr_pool[idx], y_tr_pool[idx]

        # Preprocess once per kernel
        mm = MinMaxScaler(feature_range=(0.0, float(np.pi))).fit(X_tr_raw)
        ss = StandardScaler().fit(X_tr_raw)
        X_tr_q = mm.transform(X_tr_raw).astype(np.float32)
        X_te_q = mm.transform(X_te_raw).astype(np.float32)
        X_tr_r = ss.transform(X_tr_raw).astype(np.float32)
        X_te_r = ss.transform(X_te_raw).astype(np.float32)
        # POST-HOC control: the same MinMax features times POSTHOC_SCALE
        X_tr_p = (POSTHOC_SCALE * mm.transform(X_tr_raw)).astype(np.float32)
        X_te_p = (POSTHOC_SCALE * mm.transform(X_te_raw)).astype(np.float32)

        # RBF (depth-independent; compute once per n_train)
        K_rr = rbf_kernel(X_tr_r, X_tr_r)
        K_rt = rbf_kernel(X_te_r, X_tr_r)
        t0 = time.time()
        bC_r, m_r, s_r = cv_best(K_rr, y_tr, N_CV_FOLDS, SEED)
        te_r = held_out(K_rr, K_rt, y_tr, y_te, bC_r)
        rbf_time = time.time() - t0

        for depth in DEPTHS:
            fm = ZZFeatureMap(feature_dimension=4, reps=depth)
            qk = FidelityQuantumKernel(feature_map=fm)
            t0 = time.time()
            K_qq = qk.evaluate(X_tr_q)
            K_qt = qk.evaluate(X_te_q, X_tr_q)
            bC_q, m_q, s_q = cv_best(K_qq, y_tr, N_CV_FOLDS, SEED)
            te_q = held_out(K_qq, K_qt, y_tr, y_te, bC_q)
            q_time = time.time() - t0
            print(f"  {depth:>5d}  {n_train:>7d}  {'Q':>7s}  {bC_q:>6.1f}  "
                  f"{m_q:>8.4f}  {s_q:>7.4f}  {te_q:>6.4f}  {q_time:>7.2f}")
            # POST-HOC control, computed after the timed block above
            K_pp = qk.evaluate(X_tr_p)
            K_pt = qk.evaluate(X_te_p, X_tr_p)
            bC_p, m_p, s_p = cv_best(K_pp, y_tr, N_CV_FOLDS, SEED)
            te_p = held_out(K_pp, K_pt, y_tr, y_te, bC_p)
            off_q = offdiag_stats(K_qq)
            off_p = offdiag_stats(K_pp)
            rows.append({
                "depth": depth, "n_train": n_train, "kernel": "Q",
                "best_C": bC_q, "cv_mean": m_q, "cv_std": s_q,
                "test_acc": te_q, "wall_s": q_time,
                "posthoc_offdiag_mean": off_q[0],
                "posthoc_offdiag_std": off_q[1],
                PH + "best_C": bC_p, PH + "cv_mean": m_p, PH + "cv_std": s_p,
                PH + "test_acc": te_p,
                PH + "offdiag_mean": off_p[0], PH + "offdiag_std": off_p[1],
            })
        # one RBF row per n_train (depth-independent)
        print(f"  {0:>5d}  {n_train:>7d}  {'RBF':>7s}  {bC_r:>6.1f}  "
              f"{m_r:>8.4f}  {s_r:>7.4f}  {te_r:>6.4f}  {rbf_time:>7.2f}")
        off_r = offdiag_stats(K_rr)
        rows.append({
            "depth": 0, "n_train": n_train, "kernel": "RBF",
            "best_C": bC_r, "cv_mean": m_r, "cv_std": s_r,
            "test_acc": te_r, "wall_s": rbf_time,
            "posthoc_offdiag_mean": off_r[0],
            "posthoc_offdiag_std": off_r[1],
        })

    # CSV
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"\n  saved {len(rows)} rows to {CSV_PATH}")

    section("3. Test accuracy vs n_train  (one row per depth + RBF baseline)")
    plot_rows = []
    rbf_mean = float(np.mean([
        r["test_acc"] for r in rows if r["kernel"] == "RBF"
    ]))
    for depth in DEPTHS:
        for n in N_TRAIN_SIZES:
            q = next(r["test_acc"] for r in rows
                     if r["kernel"] == "Q" and r["depth"] == depth and r["n_train"] == n)
            r_v = next(r["test_acc"] for r in rows
                       if r["kernel"] == "RBF" and r["n_train"] == n)
            plot_rows.append((f"d{depth}n{n}", q, r_v))
    text_plot(plot_rows, None, "depth/n_train cells")

    section("4. Test accuracy vs depth  (averaged across n_train)")
    print(f"  {'depth':>5s}  {'mean Q test':>11s}  {'mean RBF':>9s}  "
          f"{'gap (Q-R) pp':>13s}")
    for depth in DEPTHS:
        q_mean = float(np.mean([
            r["test_acc"] for r in rows
            if r["kernel"] == "Q" and r["depth"] == depth
        ]))
        gap = (q_mean - rbf_mean) * 100
        print(f"  {depth:>5d}  {q_mean:>11.4f}  {rbf_mean:>9.4f}  {gap:>+13.2f}")
    print( "\n  reading: at this [0, pi] input scaling, Q test acc falls with")
    print( "  depth. RBF is depth-independent (it has no depth) and stays at its")
    print( "  ceiling. Accuracy alone does not show why; section 6 (post-hoc)")
    print( "  measures the Gram spread and reruns the sweep with smaller inputs.")

    section("5. Honest summary")
    overall_q = float(np.mean([r["test_acc"] for r in rows if r["kernel"] == "Q"]))
    overall_r = float(np.mean([r["test_acc"] for r in rows if r["kernel"] == "RBF"]))
    q_wins = sum(
        1 for d in DEPTHS for n in N_TRAIN_SIZES
        if next(r["test_acc"] for r in rows if r["kernel"] == "Q"
                and r["depth"] == d and r["n_train"] == n)
        > next(r["test_acc"] for r in rows
               if r["kernel"] == "RBF" and r["n_train"] == n)
    )
    n_cells = len(DEPTHS) * len(N_TRAIN_SIZES)
    print(f"  overall mean test acc:  Q = {overall_q:.4f}  vs  RBF = {overall_r:.4f}")
    print(f"  cells where Q > RBF  :  {q_wins} of {n_cells}")
    print(f"  total quantum kernel time across the sweep: "
          f"{sum(r['wall_s'] for r in rows if r['kernel'] == 'Q'):.1f} s")
    print(f"  total RBF time                          : "
          f"{sum(r['wall_s'] for r in rows if r['kernel'] == 'RBF'):.3f} s")

    section(f"6. POST-HOC control: Gram spread, and inputs x {POSTHOC_SCALE}")
    print("  Added after the [0, pi] results were written up; not a gate.")
    print("  Off-diagonal entries of the training Gram matrix (ddof = 0),")
    print("  and test accuracy, each averaged over the 3 n_train cells.")
    print(f"  Reference, independent Haar-random 4-qubit states: mean "
          f"{HAAR_MEAN:.4f}, std {HAAR_STD:.4f}.")
    print(f"  {'':5s}  {'[0, pi] inputs (committed)':^28s}  "
          f"{f'inputs x {POSTHOC_SCALE} (post-hoc)':^28s}")
    print(f"  {'depth':>5s}  {'off mean':>8s}  {'off std':>8s}  {'test':>8s}  "
          f"{'off mean':>8s}  {'off std':>8s}  {'test':>8s}")
    q_rows = [r for r in rows if r["kernel"] == "Q"]
    for depth in DEPTHS:
        rd = [r for r in q_rows if r["depth"] == depth]
        vals = [float(np.mean([r[k] for r in rd])) for k in (
            "posthoc_offdiag_mean", "posthoc_offdiag_std", "test_acc",
            PH + "offdiag_mean", PH + "offdiag_std", PH + "test_acc")]
        print(f"  {depth:>5d}  " + "  ".join(f"{v:>8.4f}" for v in vals))
    rd = [r for r in rows if r["kernel"] == "RBF"]
    print(f"  {'RBF':>5s}  {np.mean([r['posthoc_offdiag_mean'] for r in rd]):>8.4f}"
          f"  {np.mean([r['posthoc_offdiag_std'] for r in rd]):>8.4f}"
          f"  {np.mean([r['test_acc'] for r in rd]):>8.4f}")
    overall_p = float(np.mean([r[PH + "test_acc"] for r in q_rows]))
    rbf_by_n = {r["n_train"]: r["test_acc"] for r in rows if r["kernel"] == "RBF"}
    p_wins = sum(r[PH + "test_acc"] > rbf_by_n[r["n_train"]] for r in q_rows)
    p_ties = sum(r[PH + "test_acc"] == rbf_by_n[r["n_train"]] for r in q_rows)
    print(f"  overall mean test acc, inputs x {POSTHOC_SCALE}: Q = {overall_p:.4f}"
          f"  vs  RBF = {overall_r:.4f}")
    print(f"  cells where Q x {POSTHOC_SCALE} > RBF: {p_wins} of {n_cells}; "
          f"equal to RBF: {p_ties} of {n_cells}")

    section("Checkpoint assertions")
    # Plan: full sweep runs end to end and produces both CSV + plot
    assert os.path.exists(CSV_PATH)
    assert len(rows) == n_cells + len(N_TRAIN_SIZES)  # Q cells + RBF rows
    # RBF baseline must be reasonable across n_train
    assert overall_r > 0.85, f"RBF mean test acc {overall_r:.4f} too low"
    # Quantum side must compute (we don't require it to win)
    assert all(0.0 <= r["test_acc"] <= 1.0 for r in rows), "bad accuracies"
    # Accuracy trend at the committed [0, pi] input scaling: depth=3 Q acc
    # should be no better than depth=1 Q acc. This checks accuracy only; it
    # does not measure kernel concentration (section 6 prints the spread).
    q_d1 = np.mean([r["test_acc"] for r in rows
                    if r["kernel"] == "Q" and r["depth"] == 1])
    q_d3 = np.mean([r["test_acc"] for r in rows
                    if r["kernel"] == "Q" and r["depth"] == 3])
    assert q_d3 <= q_d1 + 0.05, (
        f"quantum kernel accuracy did not fall with depth at the [0, pi] "
        f"scaling: depth=1 mean {q_d1:.3f}, depth=3 mean {q_d3:.3f}"
    )
    # POST-HOC control: only checked for completeness, not gated on value
    assert all(0.0 <= r[PH + "test_acc"] <= 1.0 for r in q_rows), \
        "bad post-hoc accuracies"
    p_d1 = np.mean([r[PH + "test_acc"] for r in q_rows if r["depth"] == 1])
    p_d3 = np.mean([r[PH + "test_acc"] for r in q_rows if r["depth"] == 3])
    print(f"  PASS: full 3x3 sweep + RBF rows; CSV at week20_results.csv; "
          f"at the [0, pi] input scaling Q d=1 mean {q_d1:.3f} -> d=3 mean "
          f"{q_d3:.3f}. Post-hoc inputs x {POSTHOC_SCALE} (not a gate): "
          f"d=1 {p_d1:.3f} -> d=3 {p_d3:.3f}.")


if __name__ == "__main__":
    main()
