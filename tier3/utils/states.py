"""H2 ground-state dataset + fidelity / partial-trace helpers.

The dataset is a 1-parameter family of 4-qubit pure states: ground states of
the H2 STO-3G electronic Hamiltonian at varying bond length r. We build the
Hamiltonian using `tier2/utils/chem.py:h2_hamiltonian` (already exercised by
the VQE work in weeks 9-11), then exact-diagonalize the resulting 16x16
matrix.

PennyLane convention: when `qml.matrix(H, wire_order=range(4))` is called,
basis states are ordered with wire 0 most significant. So `state[12]` is
amplitude on |1100> = |q0 q1 q2 q3>. We keep this convention throughout —
both the encoder QNode (which uses `qml.StatePrep(state, wires=range(4))`)
and the partial-trace helper rely on it.

For QAE work (weeks 22-27) we standardize on:
  - 4 qubits total, wires = [0, 1, 2, 3]
  - 2 code qubits (wires [0, 1]) preserve information
  - 2 trash qubits (wires [2, 3]) should be driven to |00>

The reshape `state.reshape(4, 4)[code_idx, trash_idx]` therefore gives
amplitudes indexed by (code basis, trash basis) — used by `local_fidelity`
and `reconstruction_fidelity` below.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np
import pennylane as qml

from tier2.utils.chem import h2_hamiltonian, BOHR_PER_ANGSTROM

# wire layout used everywhere in tier 3
N_QUBITS = 4
CODE_WIRES = [0, 1]
TRASH_WIRES = [2, 3]
N_CODE = len(CODE_WIRES)
N_TRASH = len(TRASH_WIRES)
DIM = 2 ** N_QUBITS
DIM_CODE = 2 ** N_CODE
DIM_TRASH = 2 ** N_TRASH


def h2_ground_state(r_angstrom):
    """Exact ground-state vector of H2/STO-3G at bond length r (in Angstrom).

    Returns `(psi, energy)` where `psi` is a length-16 complex vector with
    PennyLane wire-order convention (wire 0 most significant) and `energy`
    is the FCI energy in Hartree.
    """
    r_bohr = r_angstrom * BOHR_PER_ANGSTROM
    H, n_qubits = h2_hamiltonian(r_bohr)
    assert n_qubits == N_QUBITS, f"expected 4-qubit H2, got {n_qubits}"
    M = qml.matrix(H, wire_order=range(N_QUBITS))
    vals, vecs = np.linalg.eigh(M)
    psi = vecs[:, 0].astype(complex)
    return psi, float(vals[0])


def build_h2_dataset(r_values):
    """Stack `[h2_ground_state(r) for r in r_values]` into a (k, 16) array.

    Phases of degenerate eigenvectors from `eigh` are arbitrary; we fix the
    global phase so the largest-magnitude entry is real and positive. This
    is purely for plotting / latent-trajectory comparability across r.
    """
    states = np.empty((len(r_values), DIM), dtype=complex)
    energies = np.empty(len(r_values))
    for i, r in enumerate(r_values):
        psi, e = h2_ground_state(r)
        # canonicalize global phase
        k = int(np.argmax(np.abs(psi)))
        if abs(psi[k]) > 1e-12:
            psi = psi * np.conj(psi[k]) / abs(psi[k])
        states[i] = psi
        energies[i] = e
    return states, energies


def state_overlap(psi, phi):
    """Pure-state fidelity: |<psi|phi>|^2."""
    return float(np.abs(np.vdot(psi, phi)) ** 2)


def reduced_density_matrix(psi, keep_wires=CODE_WIRES):
    """Partial-trace the wires NOT in `keep_wires` from `|psi><psi|`.

    Implementation uses the reshape-and-contract trick. Assumes 4-qubit
    state in standard PennyLane index ordering (wire 0 most significant).
    """
    assert psi.shape == (DIM,)
    if keep_wires == CODE_WIRES:
        # T[c, t]: c indexes code, t indexes trash. State index = c * 4 + t.
        T = psi.reshape(DIM_CODE, DIM_TRASH)
        return T @ np.conj(T.T)  # 4x4
    if keep_wires == TRASH_WIRES:
        T = psi.reshape(DIM_CODE, DIM_TRASH)
        return T.T @ np.conj(T)  # 4x4
    raise ValueError(f"unsupported keep_wires {keep_wires}")


def local_fidelity(psi, encoder_unitary):
    """Romero local cost: P(trash bits = 00) after encoding.

    For pure-state inputs this is also a tight surrogate for the
    reconstruction fidelity (equality in the perfect-compression limit).
    """
    enc = encoder_unitary @ psi
    T = enc.reshape(DIM_CODE, DIM_TRASH)
    # P(trash = |00>) = sum over code of |T[c, 0]|^2
    return float(np.sum(np.abs(T[:, 0]) ** 2))


def reconstruction_fidelity(psi, encoder_unitary):
    """Full QAE reconstruction fidelity F = <psi| U^dag (rho_code (x) |0><0|) U |psi>.

    Protocol (Romero 2017):
      1. encode:   |psi> -> U|psi>
      2. partial-trace over trash: rho_code = Tr_trash(U|psi><psi|U^dag)
      3. inject fresh |00> on the trash wires: sigma = rho_code (x) |0><0|
      4. decode:   rho_out = U^dag sigma U
      5. compare:  F = <psi| rho_out |psi>
    """
    enc = encoder_unitary @ psi
    T = enc.reshape(DIM_CODE, DIM_TRASH)
    rho_code = T @ np.conj(T.T)  # (4, 4)
    zero_trash = np.zeros((DIM_TRASH, DIM_TRASH), dtype=complex)
    zero_trash[0, 0] = 1.0
    sigma = np.kron(rho_code, zero_trash)  # (16, 16)
    Udag = np.conj(encoder_unitary.T)
    rho_out = Udag @ sigma @ encoder_unitary
    return float(np.real(np.conj(psi) @ rho_out @ psi))


def uhlmann_fidelity(rho, sigma):
    """F(rho, sigma) = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2.

    Used by week 25 to compare reconstructed density matrices under noise.
    """
    from scipy.linalg import sqrtm
    sr = sqrtm(rho)
    inner = sr @ sigma @ sr
    s = sqrtm(inner)
    f = float(np.real(np.trace(s)) ** 2)
    return max(0.0, min(1.0, f))


def effective_dimension(states, threshold=1e-6):
    """Number of singular values of the (k, 16) state matrix above `threshold`.

    Geometric upper bound on how many qubits a code must span to represent
    the dataset losslessly. For H2 across r in [0.4, 2.5] A this is 4 — i.e.
    a 2-qubit code is enough.
    """
    s = np.linalg.svd(states, compute_uv=False)
    return int(np.sum(s > threshold)), s
