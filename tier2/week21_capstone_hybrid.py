"""Week 21 — Tier 2 capstone: hybrid model on a non-toy dataset.

Pulls everything together:
  - non-toy data: scikit-learn 'digits' (8x8 grayscale, MNIST-style),
    binary 0-vs-1 subset, PCA to 4 features so the quantum block stays
    in a trainable depth regime
  - hybrid model: AngleEmbedding + 2 RY-entangler layers (week 15's winner)
    wrapped in qml.qnn.TorchLayer + nn.Linear classical head (week 16)
  - end-to-end training in PyTorch (Adam, BCEWithLogitsLoss)
  - barren-plateau check at the model's working qubit count, with the
    n in {4, 6, 8, 10} sweep from week 14 confirming we are not on the
    plateau side of the cliff
  - artifacts saved alongside the script:
      week21_loss_curve.png, week21_barren_histogram.png
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
import matplotlib

matplotlib.use("Agg")  # no display in CI / headless runs
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from tier2.utils.barren import gradient_variance

N_QUBITS = 4
N_LAYERS = 2
N_EPOCHS = 60
LR = 0.05
SEED = 0

BARREN_QUBITS = (4, 6, 8, 10)
BARREN_SAMPLES = 100

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LOSS_PNG = os.path.join(THIS_DIR, "week21_loss_curve.png")
BARREN_PNG = os.path.join(THIS_DIR, "week21_barren_histogram.png")


def section(title):
    print("\n" + title)
    print("-" * len(title))


def load_digits_binary(class_a=0, class_b=1, test_size=0.2, seed=SEED):
    digits = load_digits()
    mask = (digits.target == class_a) | (digits.target == class_b)
    X = digits.data[mask].astype(np.float32)
    y = (digits.target[mask] == class_b).astype(np.float32)
    return train_test_split(X, y, test_size=test_size,
                            stratify=y, random_state=seed)


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


def train(model, X_tr, y_tr, X_te, y_te, n_epochs, lr, seed):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    Xe = torch.tensor(X_te, dtype=torch.float32)
    ye = torch.tensor(y_te, dtype=torch.float32)
    history = []
    for _ in range(n_epochs):
        opt.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        opt.step()
        with torch.no_grad():
            tr_acc = float(((model(Xt) >= 0.0).float() == yt).float().mean())
            te_acc = float(((model(Xe) >= 0.0).float() == ye).float().mean())
        history.append((float(loss), tr_acc, te_acc))
    return history


def main():
    section("1. Data: scikit-learn digits 0-vs-1, PCA(64 -> 4)")
    X_tr_raw, X_te_raw, y_tr, y_te = load_digits_binary()
    print(f"  raw shape: train {X_tr_raw.shape}, test {X_te_raw.shape}  "
          f"(64 grayscale features per 8x8 image)")
    pca = PCA(n_components=N_QUBITS, random_state=SEED).fit(X_tr_raw)
    X_tr = pca.transform(X_tr_raw).astype(np.float32)
    X_te = pca.transform(X_te_raw).astype(np.float32)
    # angle-embed scaling: tanh into (-pi/2, pi/2)
    X_tr = (np.pi / 2) * np.tanh(X_tr / X_tr.std(axis=0, keepdims=True))
    X_te = (np.pi / 2) * np.tanh(X_te / X_tr.std(axis=0, keepdims=True))
    print(f"  PCA explained variance ratio: "
          f"{[f'{v:.3f}' for v in pca.explained_variance_ratio_.tolist()]}")
    print(f"  total var captured by 4 PCs: "
          f"{float(pca.explained_variance_ratio_.sum()):.3f}")
    print(f"  encoded X range: [{X_tr.min():.3f}, {X_tr.max():.3f}]")

    section("2. Build and train the hybrid model")
    torch.manual_seed(SEED)
    model = HybridClassifier()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable parameters: {n_params}  (8 quantum + 5 classical head)")
    history = train(model, X_tr, y_tr, X_te, y_te, N_EPOCHS, LR, SEED)
    print(f"  {'epoch':>5s}  {'loss':>8s}  {'train':>8s}  {'test':>8s}")
    for ep in (0, 5, 10, 20, 30, 40, N_EPOCHS - 1):
        L, tr, te = history[ep]
        print(f"  {ep:>5d}  {L:>8.4f}  {tr:>8.4f}  {te:>8.4f}")

    final_loss, final_tr, final_te = history[-1]
    print(f"\n  final train acc : {final_tr:.4f}")
    print(f"  final test acc  : {final_te:.4f}")

    section("3. Save the loss-curve plot")
    losses = [h[0] for h in history]
    tr_accs = [h[1] for h in history]
    te_accs = [h[2] for h in history]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(losses, "k-", label="loss (BCE)")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax2 = ax1.twinx()
    ax2.plot(tr_accs, "b--", label="train acc")
    ax2.plot(te_accs, "r-", label="test acc")
    ax2.set_ylabel("accuracy")
    ax2.set_ylim(0.0, 1.05)
    fig.legend(loc="lower right", bbox_to_anchor=(0.95, 0.15))
    fig.suptitle(f"Tier 2 capstone: digits 0-vs-1, hybrid model (n={N_QUBITS} qubits)")
    fig.tight_layout()
    fig.savefig(LOSS_PNG, dpi=120)
    plt.close(fig)
    print(f"  saved {LOSS_PNG}")

    section("4. Barren-plateau check at the model's qubit count and beyond")
    print(f"  observable: <Z_0 Z_1>;  hardware-efficient ansatz, depth = n")
    print(f"  samples per n: {BARREN_SAMPLES}\n")
    print(f"  {'n':>3s}  {'#params':>8s}  {'Var[dE/dtheta_0]':>18s}  {'mean |grad|':>11s}")
    variances, all_grads = {}, {}
    for n_q in BARREN_QUBITS:
        v, grads = gradient_variance(
            n_qubits=n_q, n_layers=n_q, n_samples=BARREN_SAMPLES, seed=42
        )
        variances[n_q] = v
        all_grads[n_q] = grads
        print(f"  {n_q:>3d}  {n_q * n_q:>8d}  {v:>18.4e}  "
              f"{float(np.mean(np.abs(grads))):>11.4e}")

    fig, axes = plt.subplots(1, len(BARREN_QUBITS), figsize=(13, 3.5),
                             sharey=True)
    for ax, n_q in zip(axes, BARREN_QUBITS):
        ax.hist(all_grads[n_q], bins=20, color="steelblue", edgecolor="white")
        ax.set_title(f"n={n_q}, Var={variances[n_q]:.2e}")
        ax.set_xlabel(r"$\partial \langle O\rangle / \partial \theta_0$")
        ax.axvline(0.0, color="k", linewidth=0.7)
    axes[0].set_ylabel("count")
    fig.suptitle("Barren-plateau probe: gradient distribution by qubit count")
    fig.tight_layout()
    fig.savefig(BARREN_PNG, dpi=120)
    plt.close(fig)
    print(f"  saved {BARREN_PNG}")

    print(f"\n  the model trains at n={N_QUBITS}, where Var = {variances[4]:.4e}.")
    print(f"  by n=10 the variance has dropped to {variances[10]:.4e} -- "
          f"{variances[4]/variances[10]:.1f}x smaller -- consistent with the")
    print(f"  exponential decay measured in week 14. Choosing n=4 is what")
    print(f"  keeps the capstone in the trainable regime.")

    section("5. Cross-tier reflection (one-line per sub-project)")
    print( "  2A VQE on H2          : exact within 0.001 mHa of FCI (week 10)")
    print( "  2B QAOA on MaxCut     : rho 0.85 -> 0.98 across p=1..3 (week 13)")
    print( "  2C variational class. : encoding choice dominated; classical MLP")
    print( "                          edged out the quantum hybrid head-to-head")
    print( "                          on Iris-1-vs-2 (week 17)")
    print( "  2D quantum kernel     : 0/9 cells beat RBF, depth made it worse")
    print( "                          (week 20)")
    print( "  See TIER2_REVIEW.md for the full writeup.")

    section("Checkpoint assertions")
    # Plan: model trains to non-trivial accuracy
    assert final_te > 0.85, f"capstone test acc {final_te:.4f} below 0.85"
    # Loss must drop substantially
    assert losses[-1] < losses[0] * 0.5, (
        f"loss did not halve: {losses[0]:.3f} -> {losses[-1]:.3f}"
    )
    # Both PNGs exist
    assert os.path.exists(LOSS_PNG)
    assert os.path.exists(BARREN_PNG)
    # Barren-plateau decay confirmed
    assert variances[BARREN_QUBITS[0]] / variances[BARREN_QUBITS[-1]] > 4.0, (
        "expected variance to decay > 4x from n=4 to n=10"
    )
    print(f"  PASS: capstone test acc {final_te:.4f} > 0.85; "
          f"barren Var decays {variances[4]/variances[10]:.1f}x; "
          f"loss curve and gradient histogram saved.")


if __name__ == "__main__":
    main()
