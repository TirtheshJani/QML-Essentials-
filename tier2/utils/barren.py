"""Gradient-variance probe used by the barren-plateau studies (week 14, week 21).

Following McClean et al. 2018 (Nat. Commun. 9:4812):
  - sample random parameters from U[0, 2 pi) for a hardware-efficient ansatz
  - measure d<O>/dtheta_0 for a fixed observable O via reverse-mode autodiff
  - Var over random samples decays exponentially with the qubit count n

The ansatz used here is the standard "hardware-efficient" alternating layers:
each layer = RY(theta_iL) on every qubit + nearest-neighbour CNOT ladder.

Implementation note: only the gradient wrt theta_0 is needed, but the full
parameter-shift sweep would compute all n*L derivatives. We use backprop on
default.qubit and split the parameter signature so only theta_0 carries
requires_grad — this is ~30x faster at n=10.
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


def hardware_efficient_qnode(n_qubits, n_layers):
    """Return a QNode with signature (theta0, theta_rest), returning <Z_0 Z_1>.
    Only theta0 is differentiated when used with qml.grad(_, argnums=0)."""
    dev = qml.device("default.qubit", wires=n_qubits)
    obs = qml.PauliZ(0) @ qml.PauliZ(1)

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def circuit(theta0, theta_rest):
        full = pnp.concatenate([pnp.array([theta0]), theta_rest]).reshape(
            n_layers, n_qubits
        )
        for L in range(n_layers):
            for w in range(n_qubits):
                qml.RY(full[L, w], wires=w)
            for w in range(n_qubits - 1):
                qml.CNOT(wires=[w, w + 1])
        return qml.expval(obs)

    return circuit


def gradient_variance(n_qubits, n_layers, n_samples, seed=0):
    """Estimate Var[d<O>/dtheta_0] over n_samples random parameter draws.
    Returns (variance, raw_gradients_array)."""
    circuit = hardware_efficient_qnode(n_qubits, n_layers)
    rng = np.random.default_rng(seed)

    n_params = n_qubits * n_layers
    grads = np.empty(n_samples)
    for s in range(n_samples):
        theta = rng.uniform(0.0, 2 * np.pi, size=n_params)
        th0 = pnp.array(theta[0], requires_grad=True)
        th_rest = pnp.array(theta[1:], requires_grad=False)
        grads[s] = float(qml.grad(circuit, argnums=0)(th0, th_rest))

    return float(np.var(grads, ddof=1)), grads
