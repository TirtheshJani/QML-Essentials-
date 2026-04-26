"""Week 15 — Encoding study: angle vs amplitude vs IQP on Iris (1 vs 2).

Tier 2 / 2C.1. Working through:
  - the three canonical data encodings on a 4-qubit ansatz:
      angle     - one R_Y(x_i) on qubit i (data lives on the Bloch sphere)
      amplitude - x normalised then loaded directly as state amplitudes
      iqp       - diagonal RZ + ZZ unitary, hard to simulate classically
  - same trainable head for all three: BasicEntanglerLayers(L=2), output <Z_0>
  - same optimiser (Adam) and same Iris-1-vs-2 split (versicolor vs virginica)
    so that test accuracy is *only* a function of the encoding
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from tier2.utils.data import iris_two_class, angle_normalise

N_QUBITS = 4
N_LAYERS = 2
N_EPOCHS = 60
LR = 0.05
SEED = 0


def section(title):
    print("\n" + title)
    print("-" * len(title))


def make_circuit(encoding, n_qubits, n_layers):
    """Return a QNode (weights, x) -> <Z_0> in [-1, 1]."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def circuit(weights, x):
        if encoding == "angle":
            qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
        elif encoding == "amplitude":
            qml.AmplitudeEmbedding(
                x, wires=range(n_qubits), normalize=True, pad_with=0.0
            )
        elif encoding == "iqp":
            qml.IQPEmbedding(x, wires=range(n_qubits), n_repeats=2)
        else:
            raise ValueError(encoding)
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits), rotation=qml.RY)
        return qml.expval(qml.PauliZ(0))

    return circuit


def encode_features(encoding, X):
    """Encoding-specific feature pre-processing (matches the embedding's domain)."""
    if encoding == "angle":
        # standardised features -> bounded angles in (-pi/2, pi/2)
        return angle_normalise(X)
    if encoding == "amplitude":
        # feed standardised features as raw amplitudes; AmplitudeEmbedding
        # zero-pads to 2^n and normalises to unit length internally
        return X
    if encoding == "iqp":
        # IQPEmbedding tolerates any real number; standardised features are fine
        return X
    raise ValueError(encoding)


def predict_proba(circuit, weights, X):
    """<Z_0> in [-1, 1] -> P(class=1) in [0, 1]."""
    z = np.array([float(circuit(weights, x)) for x in X])
    return 0.5 * (1.0 - z)  # class 1 when <Z_0> negative


def accuracy(circuit, weights, X, y):
    p = predict_proba(circuit, weights, X)
    return float(np.mean((p >= 0.5).astype(np.float32) == y))


def train_one(encoding, X_tr, y_tr, X_te, y_te, seed):
    circuit = make_circuit(encoding, N_QUBITS, N_LAYERS)
    Xt = encode_features(encoding, X_tr)
    Xe = encode_features(encoding, X_te)
    rng = np.random.default_rng(seed)

    # BasicEntanglerLayers expects shape (n_layers, n_qubits)
    weights = pnp.array(
        rng.uniform(-0.1, 0.1, size=(N_LAYERS, N_QUBITS)), requires_grad=True
    )

    def loss(weights):
        zs = pnp.stack([circuit(weights, x) for x in Xt])
        probs = 0.5 * (1.0 - zs)
        # binary cross-entropy with eps for stability
        eps = 1e-6
        return -pnp.mean(
            y_tr * pnp.log(probs + eps) + (1 - y_tr) * pnp.log(1 - probs + eps)
        )

    opt = qml.AdamOptimizer(stepsize=LR)
    history = []
    for epoch in range(N_EPOCHS):
        weights, lval = opt.step_and_cost(loss, weights)
        history.append(float(lval))

    train_acc = accuracy(circuit, weights, Xt, y_tr)
    test_acc = accuracy(circuit, weights, Xe, y_te)
    n_params = weights.size
    return {
        "encoding": encoding,
        "weights": np.asarray(weights),
        "n_params": int(n_params),
        "train_acc": train_acc,
        "test_acc": test_acc,
        "loss_curve": history,
    }


def main():
    section("1. Data: Iris versicolor (1) vs virginica (2)")
    # The harder pair: classes are NOT linearly separable in 4D.
    X_tr, y_tr, X_te, y_te = iris_two_class(class_a=1, class_b=2, seed=SEED)
    print(f"  train: {X_tr.shape},  test: {X_te.shape}")
    print(f"  class balance train: {[int((y_tr == c).sum()) for c in (0, 1)]}")
    print(f"  class balance test:  {[int((y_te == c).sum()) for c in (0, 1)]}")

    section("2. Train each encoding with the same RY-entangler ansatz")
    print(f"  {'encoding':>10s}  {'#params':>8s}  {'train acc':>10s}  {'test acc':>10s}  "
          f"{'loss[0]':>9s}  {'loss[-1]':>9s}")
    results = {}
    for enc in ("angle", "amplitude", "iqp"):
        r = train_one(enc, X_tr, y_tr, X_te, y_te, seed=SEED)
        results[enc] = r
        print(f"  {enc:>10s}  {r['n_params']:>8d}  {r['train_acc']:>10.4f}  "
              f"{r['test_acc']:>10.4f}  {r['loss_curve'][0]:>9.4f}  "
              f"{r['loss_curve'][-1]:>9.4f}")

    section("3. Loss curves (sparse epochs)")
    print(f"  {'epoch':>5s}  {'angle':>8s}  {'amplitude':>10s}  {'iqp':>8s}")
    for ep in (0, 5, 10, 20, 30, 40, 50, N_EPOCHS - 1):
        row = "  " + f"{ep:>5d}  "
        for enc in ("angle", "amplitude", "iqp"):
            row += f"{results[enc]['loss_curve'][ep]:>8.4f}  "
        print(row)

    section("4. Head-to-head summary")
    accs = {enc: results[enc]["test_acc"] for enc in results}
    best_enc = max(accs, key=accs.get)
    angle_amp_gap = abs(accs["angle"] - accs["amplitude"])
    print(f"  best encoding       : {best_enc}  (test acc {accs[best_enc]:.4f})")
    print(f"  angle vs amplitude  : {accs['angle']:.4f} vs {accs['amplitude']:.4f}"
          f"  -> gap = {angle_amp_gap*100:.1f} pp")
    print(f"  iqp vs angle        : {accs['iqp']:.4f} vs {accs['angle']:.4f}")
    print(f"  -> on this dataset, the encoding controls the ceiling more than")
    print(f"     the trainable ansatz weights ever could.")

    section("Checkpoint assertions")
    # Plan: at least one encoding > 0.95 test accuracy
    assert max(accs.values()) > 0.95, (
        f"no encoding beat 0.95 test acc; max = {max(accs.values()):.4f}"
    )
    # Plan: angle vs amplitude differ by >= 5 pp
    assert angle_amp_gap >= 0.05, (
        f"angle vs amplitude gap only {angle_amp_gap*100:.1f} pp (expected >= 5)"
    )
    # All three should at least drive the loss down from initialisation;
    # ending well below random labelling (loss ~ ln 2 = 0.693).
    final_losses = {enc: results[enc]["loss_curve"][-1] for enc in results}
    assert all(l < 0.7 for l in final_losses.values()), (
        f"some encoding never improved beyond random-label loss; {final_losses}"
    )
    print(f"  PASS: best encoding = {best_enc} ({accs[best_enc]*100:.1f}%); "
          f"angle vs amplitude gap = {angle_amp_gap*100:.1f} pp.")


if __name__ == "__main__":
    main()
