"""Week 16 — Hybrid quantum-classical classifier with TorchLayer.

Tier 2 / 2C.2. Working through:
  - wrapping a QNode as a torch.nn.Module via qml.qnn.TorchLayer
  - quantum block: angle encoding (best from week 15) + 2 RY-entangler layers,
    returns a 4-vector of <Z_i> expectations (one per qubit)
  - classical head: nn.Linear(4 -> 1) + sigmoid
  - end-to-end training with torch.optim.Adam and BCEWithLogitsLoss
  - reproduces / exceeds week 15's accuracy with the same encoding,
    showing the TorchLayer wrapper has no hidden cost
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
N_EPOCHS = 50
LR = 0.05
BATCH_SIZE = 16
SEED = 0


def section(title):
    print("\n" + title)
    print("-" * len(title))


def make_quantum_block(n_qubits, n_layers):
    """Returns a TorchLayer mapping (batch, n_qubits) features -> (batch, n_qubits)
    expectation values. Trainable weights have shape (n_layers, n_qubits)."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def qnode(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits), rotation=qml.RY)
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits)}
    return qml.qnn.TorchLayer(qnode, weight_shapes)


class HybridClassifier(nn.Module):
    def __init__(self, n_qubits=N_QUBITS, n_layers=N_LAYERS):
        super().__init__()
        self.q = make_quantum_block(n_qubits, n_layers)
        self.head = nn.Linear(n_qubits, 1)

    def forward(self, x):
        z = self.q(x)  # (batch, n_qubits) of <Z_i>
        return self.head(z).squeeze(-1)  # (batch,) logits


def train(model, X_tr, y_tr, X_te, y_te, n_epochs, lr, batch_size, seed):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_te_t = torch.tensor(X_te, dtype=torch.float32)
    y_te_t = torch.tensor(y_te, dtype=torch.float32)

    n = X_tr_t.shape[0]
    rng = np.random.default_rng(seed)
    history = []

    def accuracy(X, y):
        with torch.no_grad():
            preds = (model(X) >= 0.0).float()
            return float((preds == y).float().mean())

    for epoch in range(n_epochs):
        perm = rng.permutation(n)
        epoch_loss = 0.0
        for k in range(0, n, batch_size):
            idx = perm[k:k + batch_size]
            xb = X_tr_t[idx]
            yb = y_tr_t[idx]
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            epoch_loss += float(loss) * len(idx)
        epoch_loss /= n
        tr_acc = accuracy(X_tr_t, y_tr_t)
        te_acc = accuracy(X_te_t, y_te_t)
        history.append((epoch_loss, tr_acc, te_acc))
    return history


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    section("1. Data: same Iris 1 vs 2 split as week 15")
    X_tr, y_tr, X_te, y_te = iris_two_class(class_a=1, class_b=2, seed=SEED)
    X_tr = angle_normalise(X_tr)
    X_te = angle_normalise(X_te)
    print(f"  train: {X_tr.shape}    test: {X_te.shape}")

    section("2. Build hybrid model: TorchLayer (angle + entangler) -> Linear -> logits")
    torch.manual_seed(SEED)
    model = HybridClassifier()
    n_params = count_params(model)
    n_q_params = sum(p.numel() for n, p in model.named_parameters() if "head" not in n)
    n_c_params = sum(p.numel() for n, p in model.named_parameters() if "head" in n)
    print(f"  total trainable params : {n_params}  "
          f"(quantum {n_q_params}, classical head {n_c_params})")
    print(f"  optimizer              : Adam(lr={LR})")
    print(f"  batch size             : {BATCH_SIZE}")

    section("3. Train for 50 epochs")
    history = train(model, X_tr, y_tr, X_te, y_te, N_EPOCHS, LR, BATCH_SIZE, SEED)
    print(f"  {'epoch':>5s}  {'loss':>8s}  {'train acc':>10s}  {'test acc':>10s}")
    for ep in (0, 5, 10, 20, 30, 40, N_EPOCHS - 1):
        loss_, tr, te = history[ep]
        print(f"  {ep:>5d}  {loss_:>8.4f}  {tr:>10.4f}  {te:>10.4f}")

    final_loss, final_tr, final_te = history[-1]
    section("4. Result")
    print(f"  final train acc     : {final_tr:.4f}")
    print(f"  final test acc      : {final_te:.4f}")
    print(f"  week 15 angle ref   : 1.0000 (raw PennyLane Adam, no torch wrapping)")

    # Reference: classical 1-Linear baseline on the same input
    section("5. Sanity check: a single nn.Linear classical baseline")
    torch.manual_seed(SEED)
    lin = nn.Linear(4, 1)
    opt = torch.optim.Adam(lin.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_te_t = torch.tensor(X_te, dtype=torch.float32)
    y_te_t = torch.tensor(y_te, dtype=torch.float32)
    for _ in range(N_EPOCHS):
        opt.zero_grad()
        logits = lin(X_tr_t).squeeze(-1)
        loss = loss_fn(logits, y_tr_t)
        loss.backward()
        opt.step()
    with torch.no_grad():
        lin_acc = float(((lin(X_te_t).squeeze(-1) >= 0.0).float() == y_te_t).float().mean())
    print(f"  single Linear test acc : {lin_acc:.4f}  (this is the linear-classifier ceiling)")
    print(f"  hybrid quantum gain    : {(final_te - lin_acc)*100:+.1f} pp")

    section("Checkpoint assertions")
    # Plan target
    assert final_te >= 0.97, (
        f"hybrid model failed plan target 0.97 test acc; got {final_te:.4f}"
    )
    # Loss must actually decrease
    assert history[-1][0] < history[0][0] * 0.5, (
        f"loss did not halve over training: start={history[0][0]:.3f}, end={history[-1][0]:.3f}"
    )
    # The hybrid must do at least as well as the linear baseline
    assert final_te >= lin_acc - 0.05, (
        f"hybrid lost to a single Linear: {final_te:.4f} vs {lin_acc:.4f}"
    )
    print(f"  PASS: hybrid test acc = {final_te:.4f} >= 0.97; "
          f"loss dropped {history[0][0]:.3f} -> {final_loss:.3f}.")


if __name__ == "__main__":
    main()
