"""Week 14 — QAOA cross-implementation in Qiskit + barren-plateau probe.

Tier 2 / 2B.3. Working through:
  - building the same MaxCut H_C as a qiskit SparsePauliOp and as a PennyLane
    Hamiltonian, confirming they agree term-for-term
  - sampling the p=1 (gamma, beta) energy surface from both backends and
    confirming they match within numerical noise
  - optimizing p=1 in Qiskit and recovering the same approximation ratio
  - the barren-plateau probe (McClean et al. 2018): for a hardware-efficient
    ansatz with depth = qubit count, Var[d<O>/dtheta_0] decays exponentially
    in the number of qubits.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*sparse.*")

import numpy as np
import pennylane as qml
from scipy.optimize import minimize

from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp, Statevector

from tier2.utils.maxcut import (
    random_regular_graph,
    brute_force_maxcut,
    cost_hamiltonian,
)
from tier2.utils.barren import gradient_variance

N_NODES = 6
DEGREE = 3
SEED = 0

QUBIT_RANGE = (4, 6, 8, 10)
N_BARREN_SAMPLES = 100


def section(title):
    print("\n" + title)
    print("-" * len(title))


# --- Qiskit construction ---------------------------------------------------


def maxcut_sparse_pauli(graph):
    """Same H_C as tier2.utils.maxcut.cost_hamiltonian, in qiskit format.

    Qiskit Pauli strings are right-to-left in qubit index (qubit 0 = last char).
    """
    n = graph.number_of_nodes()
    paulis, coeffs = [], []
    for i, j in graph.edges:
        chars = ["I"] * n
        chars[n - 1 - i] = "Z"
        chars[n - 1 - j] = "Z"
        paulis.append("".join(chars))
        coeffs.append(-0.5)
        paulis.append("I" * n)
        coeffs.append(+0.5)
    return SparsePauliOp(paulis, coeffs=coeffs).simplify()


def qiskit_qaoa_energy(ansatz, H_q, beta_val, gamma_val):
    """Bind one (beta, gamma) into the QAOAAnsatz and return <H_C>."""
    bound = ansatz.assign_parameters(
        {ansatz.parameters[0]: beta_val, ansatz.parameters[1]: gamma_val}
    )
    return float(Statevector(bound).expectation_value(H_q).real)


def pennylane_qaoa_energy(graph, gamma_val, beta_val):
    n = graph.number_of_nodes()
    H = cost_hamiltonian(graph)
    edges = list(graph.edges)
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def circuit():
        for w in range(n):
            qml.Hadamard(w)
        for i, j in edges:
            qml.CNOT(wires=[i, j])
            qml.RZ(-gamma_val, wires=j)
            qml.CNOT(wires=[i, j])
        for w in range(n):
            qml.RX(2 * beta_val, wires=w)
        return qml.expval(H)

    return float(circuit())


# --- Main ------------------------------------------------------------------


def main():
    section("1. Build the same MaxCut H_C in Qiskit and PennyLane")
    g = random_regular_graph(N_NODES, DEGREE, seed=SEED)
    n = g.number_of_nodes()
    best, _ = brute_force_maxcut(g)
    H_q = maxcut_sparse_pauli(g)
    H_pl = cost_hamiltonian(g)
    print(f"  graph              : {n} nodes, {g.number_of_edges()} edges, MaxCut = {best}")
    print(f"  Qiskit H_C terms   : {len(H_q.paulis)} (after simplify)")
    print(f"  PennyLane H_C terms: {len(H_pl.terms()[0])}")

    # eigenvalue agreement
    pl_mat = qml.matrix(H_pl, wire_order=range(n))
    pl_evals = np.sort(np.linalg.eigvalsh(pl_mat))
    qk_evals = np.sort(np.linalg.eigvalsh(H_q.to_matrix()))
    max_eig_diff = float(np.max(np.abs(pl_evals - qk_evals)))
    print(f"  max |spectrum diff|: {max_eig_diff:.2e}")

    section("2. p=1 (gamma, beta) energy surface: agreement on a 5x5 grid")
    grid_g = np.linspace(0.0, np.pi, 5)
    grid_b = np.linspace(0.0, np.pi / 2, 5)
    ansatz1 = QAOAAnsatz(cost_operator=H_q, reps=1)
    print(f"  {'gamma':>6s}  {'beta':>6s}  {'qiskit':>10s}  {'pennylane':>10s}  {'|diff|':>9s}")
    max_surf_diff = 0.0
    for gamma in grid_g[::2]:
        for beta in grid_b[::2]:
            e_q = qiskit_qaoa_energy(ansatz1, H_q, beta, gamma)
            e_p = pennylane_qaoa_energy(g, gamma, beta)
            d = abs(e_q - e_p)
            max_surf_diff = max(max_surf_diff, d)
            print(f"  {gamma:>6.3f}  {beta:>6.3f}  {e_q:>10.5f}  {e_p:>10.5f}  {d:>9.1e}")
    # full 5x5 sweep for the assertion (cheaper than printing all 25)
    for gamma in grid_g:
        for beta in grid_b:
            d = abs(
                qiskit_qaoa_energy(ansatz1, H_q, beta, gamma)
                - pennylane_qaoa_energy(g, gamma, beta)
            )
            max_surf_diff = max(max_surf_diff, d)
    print(f"\n  max |Qiskit - PennyLane| over 5x5 grid: {max_surf_diff:.2e}")

    section("3. Optimize p=1 in Qiskit (COBYLA) — reproduce week 13's rho ~ 0.85")

    def neg_qiskit_energy(x):
        beta, gamma = x
        return -qiskit_qaoa_energy(ansatz1, H_q, beta, gamma)

    rng = np.random.default_rng(0)
    best_e = -np.inf
    best_x = None
    for r in range(8):
        x0 = rng.uniform(low=[0.0, 0.0], high=[np.pi / 2, np.pi], size=2)
        res = minimize(neg_qiskit_energy, x0, method="COBYLA",
                       options={"rhobeg": 0.3, "maxiter": 200})
        if -res.fun > best_e:
            best_e, best_x = -res.fun, res.x
    rho_q = best_e / best
    print(f"  best Qiskit p=1 <H_C>: {best_e:.5f}  (beta={best_x[0]:+.3f}, gamma={best_x[1]:+.3f})")
    print(f"  rho                  : {rho_q:.4f}")
    print(f"  PennyLane week 13 p=1: 0.8485")

    section("4. Barren-plateau probe: hardware-efficient ansatz, L = n")
    print(f"  observable: <Z_0 Z_1>;  parameters: n*L drawn ~ U[0, 2 pi)")
    print(f"  samples per n: {N_BARREN_SAMPLES}\n")
    print(f"  {'n':>3s}  {'#params':>8s}  {'Var[dE/dtheta_0]':>18s}  "
          f"{'log_2 ratio to n=4':>20s}")
    variances = {}
    var4 = None
    for n_q in QUBIT_RANGE:
        v, _ = gradient_variance(
            n_qubits=n_q, n_layers=n_q, n_samples=N_BARREN_SAMPLES, seed=42
        )
        variances[n_q] = v
        if var4 is None:
            var4 = v
            ratio_str = "(reference)"
        else:
            ratio_str = f"{np.log2(v / var4):+.2f}"
        print(f"  {n_q:>3d}  {n_q*n_q:>8d}  {v:>18.4e}  {ratio_str:>20s}")

    # Linear fit log Var = a - c * n -> halving rate per qubit added
    ns = np.array(QUBIT_RANGE, dtype=float)
    log_v = np.log(np.array([variances[n_q] for n_q in QUBIT_RANGE]))
    slope, intercept = np.polyfit(ns, log_v, 1)
    halving_per_qubit = -slope / np.log(2)
    print(f"\n  exponential fit log Var(n) ~ {intercept:+.2f} + {slope:+.4f} * n")
    print(f"  -> Var halves every {1/halving_per_qubit:.2f} qubit(s) added "
          f"(decays by factor {np.exp(-slope):.2f}x per qubit)")

    section("Checkpoint assertions")
    # Cross-implementation: spectrum and surface match
    assert max_eig_diff < 1e-9, f"H_C spectra disagree: {max_eig_diff}"
    assert max_surf_diff < 1e-9, f"p=1 surfaces disagree: {max_surf_diff}"
    # Qiskit COBYLA recovers a comparable approximation ratio
    assert rho_q >= 0.80, f"Qiskit p=1 rho too low: {rho_q}"
    # Barren plateau: variance at n=10 must be substantially smaller than at n=4
    decay = variances[QUBIT_RANGE[0]] / variances[QUBIT_RANGE[-1]]
    assert decay > 4.0, f"expected variance decay > 4x from n=4 to n=10, got {decay:.2f}x"
    # And monotonic non-increasing
    vals = [variances[n_q] for n_q in QUBIT_RANGE]
    assert all(vals[i] >= vals[i + 1] / 1.5 for i in range(len(vals) - 1)), (
        f"variance not monotonically decaying: {vals}"
    )
    print(f"  PASS: cross-impl agrees to {max_surf_diff:.0e}; "
          f"barren-plateau Var decays {decay:.1f}x from n=4 to n=10.")


if __name__ == "__main__":
    main()
