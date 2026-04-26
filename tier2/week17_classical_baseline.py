"""Week 17 — Matched-parameter classical baseline.

Tier 2 / 2C.3. Working through:
  - building a 1-hidden-layer MLP whose parameter count exactly matches the
    week-16 hybrid (13 params): input(4) -> hidden(2) -> output(1) gives
    4*2 + 2 + 2*1 + 1 = 13 trainable weights/biases
  - sweeping the training-set size n_train in {10, 20, 40, 60, 80} with three
    seeds per point, training both models under identical optimisation
    settings (Adam, lr=0.05, 50 full-batch epochs, BCE)
  - reporting head-to-head test accuracy curves: where does each model win?
  - the honest takeaway: at this scale the quantum block buys you nothing
    a 13-parameter classical MLP cannot already produce.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

from tier2.utils.data import iris_two_class, angle_normalise

N_QUBITS = 4
N_LAYERS = 2
N_HIDDEN = 2  # 4*2 + 2 + 2 + 1 = 13 parameters, matching the hybrid
N_EPOCHS = 50
LR = 0.05
SAMPLE_SIZES = (10, 20, 40, 60, 80)
SEEDS = (0, 1, 2)
TARGET_PARAM_COUNT = 13


def section(title):
    print("\n" + title)
    print("-" * len(title))


def make_quantum_block():
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev, interface="torch")
    def qnode(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
        qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS), rotation=qml.RY)
        return [qml.expval(qml.PauliZ(w)) for w in range(N_QUBITS)]

    return qml.qnn.TorchLayer(qnode, {"weights": (N_LAYERS, N_QUBITS)})


class HybridClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = make_quantum_block()
        self.head = nn.Linear(N_QUBITS, 1)

    def forward(self, x):
        return self.head(self.q(x)).squeeze(-1)


class ClassicalMLP(nn.Module):
    """4 -> 2 -> 1 with ReLU; 13 trainable parameters total."""

    def __init__(self, hidden=N_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_full_batch(model, X_tr, y_tr, X_te, y_te, n_epochs, lr, seed):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    Xe = torch.tensor(X_te, dtype=torch.float32)
    ye = torch.tensor(y_te, dtype=torch.float32)
    for _ in range(n_epochs):
        opt.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        tr_acc = float(((model(Xt) >= 0.0).float() == yt).float().mean())
        te_acc = float(((model(Xe) >= 0.0).float() == ye).float().mean())
    return tr_acc, te_acc


def text_curve(sizes, mean_q, mean_c, width=44):
    """Tiny ASCII plot of test accuracy vs n_train for two series."""
    lo, hi = 0.5, 1.02
    print(f"  acc range: {lo:.2f} -> {hi:.2f}    legend: Q=hybrid, C=classical, *=overlap")
    print(f"  {'n_train':>7s} | {lo:.2f}{'':>{width//2 - 4}}{(lo+hi)/2:.2f}{'':>{width//2 - 4}}{hi:.2f}")
    for k, n in enumerate(sizes):
        row = [" "] * width
        for ch, ys in (("Q", mean_q), ("C", mean_c)):
            j = int(round((ys[k] - lo) / (hi - lo) * (width - 1)))
            j = max(0, min(width - 1, j))
            row[j] = ch if row[j] == " " else "*"
        print(f"  {n:>7d} |{''.join(row)}|")


def main():
    section("1. Verify parameter-count match")
    h = HybridClassifier()
    c = ClassicalMLP()
    n_h = count_params(h)
    n_c = count_params(c)
    print(f"  hybrid (TorchLayer + Linear)  : {n_h} params  "
          f"(quantum 8 + linear 5)")
    print(f"  classical MLP (4-{N_HIDDEN}-1 + ReLU): {n_c} params  "
          f"(input 4*{N_HIDDEN}+{N_HIDDEN} + output {N_HIDDEN}+1)")
    print(f"  target                        : {TARGET_PARAM_COUNT} params")
    assert n_h == TARGET_PARAM_COUNT, n_h
    assert n_c == TARGET_PARAM_COUNT, n_c

    section("2. Sample-size sweep, 3 seeds, head-to-head")
    print(f"  data: Iris versicolor (1) vs virginica (2); test set fixed at 20 examples")
    print(f"  {'n_train':>7s}  {'seed':>4s}  {'hybrid test':>11s}  {'mlp test':>9s}")
    results = {n: {"q": [], "c": []} for n in SAMPLE_SIZES}
    for n_train in SAMPLE_SIZES:
        for seed in SEEDS:
            X_tr_full, y_tr_full, X_te, y_te = iris_two_class(
                class_a=1, class_b=2, seed=seed
            )
            X_tr = angle_normalise(X_tr_full[:n_train])
            y_tr = y_tr_full[:n_train]
            X_te_n = angle_normalise(X_te)

            torch.manual_seed(seed)
            h = HybridClassifier()
            _, te_q = train_full_batch(h, X_tr, y_tr, X_te_n, y_te,
                                       N_EPOCHS, LR, seed)
            torch.manual_seed(seed)
            c = ClassicalMLP()
            _, te_c = train_full_batch(c, X_tr, y_tr, X_te_n, y_te,
                                       N_EPOCHS, LR, seed)
            results[n_train]["q"].append(te_q)
            results[n_train]["c"].append(te_c)
            print(f"  {n_train:>7d}  {seed:>4d}  {te_q:>11.4f}  {te_c:>9.4f}")

    section("3. Mean and std across seeds")
    print(f"  {'n_train':>7s}  {'hybrid mean+/-std':>20s}  {'mlp mean+/-std':>17s}  {'gap (Q-C) pp':>13s}")
    mean_q, mean_c = [], []
    for n in SAMPLE_SIZES:
        q_arr = np.array(results[n]["q"])
        c_arr = np.array(results[n]["c"])
        mq, sq = float(q_arr.mean()), float(q_arr.std(ddof=0))
        mc, sc = float(c_arr.mean()), float(c_arr.std(ddof=0))
        gap_pp = (mq - mc) * 100
        mean_q.append(mq)
        mean_c.append(mc)
        print(f"  {n:>7d}  {mq:>10.4f} +/- {sq:.4f}  {mc:>8.4f} +/- {sc:.4f}  {gap_pp:>+13.2f}")

    section("4. Test accuracy vs training set size (text plot)")
    text_curve(SAMPLE_SIZES, mean_q, mean_c)

    section("5. Honest summary")
    overall_gap = (np.mean(mean_q) - np.mean(mean_c)) * 100
    wins_q = sum(1 for k in range(len(SAMPLE_SIZES)) if mean_q[k] > mean_c[k] + 0.005)
    wins_c = sum(1 for k in range(len(SAMPLE_SIZES)) if mean_c[k] > mean_q[k] + 0.005)
    ties = len(SAMPLE_SIZES) - wins_q - wins_c
    print(f"  averaged across all sample sizes:")
    print(f"    hybrid mean test acc      : {np.mean(mean_q):.4f}")
    print(f"    classical MLP mean test acc: {np.mean(mean_c):.4f}")
    print(f"    overall gap (hybrid - mlp): {overall_gap:+.2f} pp")
    print(f"    points won by hybrid      : {wins_q}/{len(SAMPLE_SIZES)}")
    print(f"    points won by classical   : {wins_c}/{len(SAMPLE_SIZES)}")
    print(f"    points tied (within 0.5pp): {ties}/{len(SAMPLE_SIZES)}")
    print()
    print(f"  Reading: at 13 trainable parameters and Iris-1-vs-2, both models")
    print(f"  saturate near 1.00 once n_train >= ~40. The hybrid's only structural")
    print(f"  advantage is the 4-D Bloch-vector readout vs. ReLU(Wx+b); on this")
    print(f"  data that is not enough to prefer one over the other.")

    section("Checkpoint assertions")
    # Plan: ≥ 5 sample-size points (we have exactly 5)
    assert len(SAMPLE_SIZES) >= 5
    # Plan: reproducibility across seeds
    for n in SAMPLE_SIZES:
        s_q = float(np.std(results[n]["q"]))
        s_c = float(np.std(results[n]["c"]))
        assert s_q < 0.25 and s_c < 0.25, (
            f"high cross-seed variance at n={n}: hybrid std {s_q:.3f}, mlp std {s_c:.3f}"
        )
    # Both models eventually reach respectable accuracy at largest n
    n_big = SAMPLE_SIZES[-1]
    assert np.mean(results[n_big]["q"]) > 0.85, "hybrid weak at largest n"
    assert np.mean(results[n_big]["c"]) > 0.85, "classical MLP weak at largest n"
    # Param-match invariant
    assert n_h == n_c == TARGET_PARAM_COUNT
    print(f"  PASS: 5 sample sizes, 3 seeds each; hybrid & MLP both reach "
          f">0.85 at n={n_big}; gap {overall_gap:+.2f} pp.")


if __name__ == "__main__":
    main()
