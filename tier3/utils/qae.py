"""Quantum-autoencoder ansatz, Romero local cost, and a training loop.

Encoder is a hardware-efficient ansatz of `n_layers` layers, each = RY on
every qubit + nearest-neighbour CNOT ladder. This restricts to real-valued
amplitudes — fine for H2 ground states, which are real in STO-3G — and
matches the barren-plateau probe ansatz from `tier2/utils/barren.py` so
the trainability behaviour is consistent across tiers.

Training uses PennyLane's autograd interface with `qml.AdamOptimizer`.
The cost is the Romero local fidelity averaged over a training batch:

  C(alpha) = 1 - (1/N) * sum_i  P_i(trash bits = 00 | alpha)

Gradient of probabilities is supported by `default.qubit` with backprop, so
each gradient step costs one forward pass per training example.
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from tier3.utils.states import (
    N_QUBITS,
    CODE_WIRES,
    TRASH_WIRES,
    DIM_TRASH,
    local_fidelity as np_local_fidelity,
)


def _shape(n_layers):
    return (n_layers, N_QUBITS)


def encoder(params, wires=range(N_QUBITS)):
    """Hardware-efficient encoder: depth-L RY + linear CNOT ladder.

    `params.shape == (n_layers, n_qubits)`.
    """
    n_layers, n_qubits = params.shape
    wires = list(wires)
    for L in range(n_layers):
        for w in range(n_qubits):
            qml.RY(params[L, w], wires=wires[w])
        for w in range(n_qubits - 1):
            qml.CNOT(wires=[wires[w], wires[w + 1]])


def make_trash_probs_qnode(n_layers):
    """QNode mapping (input_state, params) -> P(trash bitstring).

    Returns a length-4 vector: (P(00), P(01), P(10), P(11)) on TRASH_WIRES.
    """
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def qnode(state, params):
        qml.StatePrep(state, wires=range(N_QUBITS))
        encoder(params, wires=range(N_QUBITS))
        return qml.probs(wires=TRASH_WIRES)

    return qnode


def make_encoder_unitary(n_layers):
    """Return a function `params -> U` giving the encoder as a 16x16 matrix.

    Used by `reconstruction_fidelity` (which is purely numerical, not a
    QNode). Implemented by constructing the matrix manually so we don't need
    to round-trip through a QNode.
    """
    def U(params):
        n_layers_p, n_qubits = params.shape
        assert n_qubits == N_QUBITS
        # build 16x16 matrix step by step
        M = np.eye(2 ** N_QUBITS, dtype=complex)
        for L in range(n_layers_p):
            # single-qubit RY layer
            for w in range(N_QUBITS):
                ry = _ry_matrix(params[L, w])
                M = _apply_single_qubit(ry, w, M)
            # CNOT ladder
            for w in range(N_QUBITS - 1):
                M = _apply_cnot(w, w + 1, M)
        return M
    return U


def _ry_matrix(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _apply_single_qubit(g, wire, M):
    """Left-multiply M by a single-qubit gate `g` on `wire`.

    Wire 0 is most significant in our convention. We do this by reshaping
    the 16-dim space into a (2, ..., 2, dim) tensor and contracting g into
    the appropriate axis.
    """
    n = N_QUBITS
    M_t = M.reshape([2] * n + [2 ** n])
    M_t = np.tensordot(g, M_t, axes=([1], [wire]))
    # tensordot moves the contracted axis; need to put it back.
    M_t = np.moveaxis(M_t, 0, wire)
    return M_t.reshape(2 ** n, 2 ** n)


def _apply_cnot(control, target, M):
    """Left-multiply M by CNOT(control, target). Wire 0 most significant."""
    n = N_QUBITS
    out = np.zeros_like(M)
    for i in range(2 ** n):
        bits = [(i >> (n - 1 - k)) & 1 for k in range(n)]
        if bits[control] == 1:
            bits_new = bits.copy()
            bits_new[target] = 1 - bits_new[target]
            j = sum(b << (n - 1 - k) for k, b in enumerate(bits_new))
        else:
            j = i
        out[j] = M[i]
    return out


def romero_cost(qnode, params, states):
    """Mean local fidelity loss: 1 - mean_i P(trash = 00)."""
    total = 0.0
    for psi in states:
        probs = qnode(psi, params)
        total = total + probs[0]
    return 1.0 - total / len(states)


def train_qae(
    train_states,
    n_layers=4,
    n_epochs=200,
    lr=0.05,
    seed=0,
    verbose=False,
    log_grad_at=None,
):
    """Train the QAE on `train_states` (a list/array of length-16 vectors).

    Returns `(history, final_params)` where `history` is a list of
    (epoch, loss, mean_local_fid) tuples.

    `log_grad_at` is an optional iterable of epoch numbers; at those epochs
    we compute the gradient norm of the cost wrt `params` and append it to
    `history` as a 4th tuple entry. This is the inline barren-plateau
    monitor (Tier 2 review item 4).
    """
    rng = np.random.default_rng(seed)
    qnode = make_trash_probs_qnode(n_layers)
    params = pnp.array(
        rng.uniform(0.0, 2 * np.pi, size=_shape(n_layers)),
        requires_grad=True,
    )
    opt = qml.AdamOptimizer(stepsize=lr)
    history = []
    log_set = set(log_grad_at) if log_grad_at else set()

    def cost_fn(p):
        return romero_cost(qnode, p, train_states)

    for ep in range(n_epochs):
        params, loss = opt.step_and_cost(cost_fn, params)
        mean_fid = 1.0 - float(loss)
        entry = (ep, float(loss), mean_fid)
        if ep in log_set or ep == n_epochs - 1:
            grad_fn = qml.grad(cost_fn)
            g = grad_fn(params)
            grad_norm = float(np.linalg.norm(np.asarray(g)))
            entry = entry + (grad_norm,)
        history.append(entry)
        if verbose and ep % 20 == 0:
            print(f"  epoch {ep:>3d}: loss = {loss:.4f}  mean_fid = {mean_fid:.4f}")
    return history, np.asarray(params)


def evaluate(params, states, encoder_unitary_fn=None):
    """Mean and per-state local fidelity on `states`. Numpy-only.

    `encoder_unitary_fn(params) -> U` lets the caller reuse a cached U
    factory. If None, builds one.
    """
    if encoder_unitary_fn is None:
        encoder_unitary_fn = make_encoder_unitary(params.shape[0])
    U = encoder_unitary_fn(np.asarray(params))
    fids = np.array([np_local_fidelity(psi, U) for psi in states])
    return float(np.mean(fids)), fids
