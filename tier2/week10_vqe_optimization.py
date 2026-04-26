"""Week 10 — VQE optimization on H2.

Tier 2 / 2A.2. Working through:
  - the AllSinglesDoubles ansatz on top of the HF reference
    (3 parameters: singles [0,2], [1,3] and the double [0,1,2,3])
  - the variational principle: <psi(theta)| H |psi(theta)> >= E_FCI for any theta
  - Adam optimization of the energy expectation value
  - convergence to chemical accuracy (1.6 mHa) within ~100 iterations
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from tier2.utils.chem import h2_hamiltonian, fci_energy, hf_energy

R_EQ_BOHR = 1.398
CHEM_ACC_HA = 1.6e-3  # 1.6 mHa = chemical accuracy
MAX_ITERS = 200
TOL_HA = 1e-7


def section(title):
    print("\n" + title)
    print("-" * len(title))


def build_vqe(H, n_qubits, n_electrons=2):
    """Returns (energy_fn, theta_init, n_params, hf_state) for AllSinglesDoubles."""
    singles, doubles = qml.qchem.excitations(n_electrons, n_qubits)
    hf = qml.qchem.hf_state(n_electrons, n_qubits)
    n_params = len(singles) + len(doubles)
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def energy(theta):
        qml.AllSinglesDoubles(
            weights=theta,
            wires=range(n_qubits),
            hf_state=hf,
            singles=singles,
            doubles=doubles,
        )
        return qml.expval(H)

    theta_init = pnp.zeros(n_params, requires_grad=True)
    return energy, theta_init, n_params, hf


def main():
    section("1. Build H2 Hamiltonian and the AllSinglesDoubles ansatz")
    H, n_qubits = h2_hamiltonian(R_EQ_BOHR)
    energy, theta, n_params, hf = build_vqe(H, n_qubits)
    e_hf = hf_energy(H, n_qubits)
    e_fci = fci_energy(H, n_qubits)
    print(f"  n_qubits           : {n_qubits}")
    print(f"  hf state           : {hf}")
    print(f"  ansatz params      : {n_params}  (2 singles + 1 double)")
    print(f"  energy at theta=0  : {float(energy(theta)):+.6f} Ha  (= E_HF)")
    print(f"  E_HF               : {e_hf:+.6f} Ha")
    print(f"  E_FCI (target)     : {e_fci:+.6f} Ha")

    section("2. Adam optimization of <H>(theta)")
    opt = qml.AdamOptimizer(stepsize=0.1)
    print(f"  {'iter':>4s}  {'energy':>12s}  {'gap to FCI (mHa)':>16s}  {'|grad|':>10s}")
    history = []
    e_prev = float(energy(theta))
    history.append(e_prev)
    for k in range(1, MAX_ITERS + 1):
        theta, e_now = opt.step_and_cost(energy, theta)
        e_now = float(e_now)
        history.append(e_now)
        if k <= 5 or k % 20 == 0:
            grad_norm = float(np.linalg.norm(qml.grad(energy)(theta)))
            print(f"  {k:>4d}  {e_now:>+12.6f}  {(e_now - e_fci)*1000:>+16.3f}  {grad_norm:>10.2e}")
        if abs(e_prev - e_now) < TOL_HA and k > 5:
            print(f"  converged: |dE| < {TOL_HA:.0e} Ha at iter {k}")
            break
        e_prev = e_now
    e_vqe = history[-1]

    section("3. Final theta (optimal AllSinglesDoubles weights)")
    print(f"  theta              : {np.array2string(np.asarray(theta), precision=5)}")
    print(f"  E_VQE              : {e_vqe:+.6f} Ha")
    print(f"  E_FCI              : {e_fci:+.6f} Ha")
    print(f"  gap                : {(e_vqe - e_fci)*1000:+.4f} mHa  (target < 1.6 mHa)")
    print(f"  HF correlation captured: "
          f"{(e_hf - e_vqe)/(e_hf - e_fci)*100:.2f}% of HF-FCI gap")

    section("4. Loss curve (sparse)")
    print(f"  {'iter':>4s}  {'energy':>12s}")
    for k in (0, 1, 2, 5, 10, 20, 50, 100, len(history) - 1):
        if k < len(history):
            print(f"  {k:>4d}  {history[k]:>+12.6f}")

    section("Checkpoint assertions")
    assert e_vqe >= e_fci - 1e-9, f"variational principle violated: {e_vqe} < {e_fci}"
    assert abs(e_vqe - e_fci) < CHEM_ACC_HA, (
        f"VQE missed chemical accuracy: gap = {(e_vqe - e_fci)*1000:.3f} mHa"
    )
    assert e_vqe < e_hf - 5e-3, "VQE failed to improve substantially over HF"
    print(f"  PASS: E_VQE = {e_vqe:+.6f} Ha within "
          f"{(e_vqe - e_fci)*1000:+.3f} mHa of FCI.")


if __name__ == "__main__":
    main()
