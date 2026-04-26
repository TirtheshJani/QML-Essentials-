"""Week 9 — H2 Hamiltonian and the Hartree-Fock reference.

Tier 2 / 2A.1. Working through:
  - building the H2 electronic Hamiltonian via qml.qchem.molecular_hamiltonian
    at the equilibrium bond length 0.74 A (= 1.398 bohr)
  - reading the Pauli decomposition (15 terms on 4 qubits in STO-3G)
  - measuring the Hartree-Fock energy from the Slater determinant |1100>
  - exact FCI energy by direct diagonalization (cheap at 4 qubits)
  - the static correlation gap E_FCI - E_HF ~ 20 mHa even at equilibrium
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pennylane as qml

from tier2.utils.chem import h2_hamiltonian, hf_energy, fci_energy

R_EQ_BOHR = 1.398  # 0.74 A
E_HF_REF = -1.117  # Ha, STO-3G reference (Sherrill notes; PennyLane qchem demo)
E_FCI_REF = -1.137  # Ha


def section(title):
    print("\n" + title)
    print("-" * len(title))


def main():
    section("1. Build the H2 Hamiltonian at r = 1.398 bohr (~0.74 A)")
    H, n_qubits = h2_hamiltonian(R_EQ_BOHR)
    coeffs, ops = H.terms()
    print(f"  n_qubits           : {n_qubits}")
    print(f"  n_pauli_terms      : {len(coeffs)}")
    print(f"  identity coefficient (constant shift): {float(coeffs[0]):+.6f}")

    section("2. Pauli decomposition (top-5 by |coefficient|)")
    order = np.argsort(-np.abs([float(c) for c in coeffs]))
    for k in order[:5]:
        print(f"  {float(coeffs[k]):+.6f}   {ops[k]}")

    section("3. Hartree-Fock energy on the |1100> reference")
    e_hf = hf_energy(H, n_qubits)
    print(f"  E_HF (computed)    : {e_hf:+.6f} Ha")
    print(f"  E_HF (reference)   : {E_HF_REF:+.6f} Ha")
    print(f"  abs error          : {abs(e_hf - E_HF_REF)*1000:.3f} mHa")

    section("4. FCI ground state via direct diagonalization")
    e_fci = fci_energy(H, n_qubits)
    print(f"  E_FCI              : {e_fci:+.6f} Ha")
    print(f"  E_FCI (reference)  : {E_FCI_REF:+.6f} Ha")
    print(f"  static-corr gap    : {(e_hf - e_fci)*1000:.3f} mHa  (HF - FCI)")

    section("5. Why HF is not enough")
    print("  HF puts both electrons in the lowest spatial orbital -> |1100>.")
    print("  At equilibrium the gap is small (~20 mHa) but already above")
    print("  chemical accuracy (1.6 mHa). VQE in week 10 will close it.")

    section("Checkpoint assertions")
    assert n_qubits == 4, f"H2/STO-3G should give 4 qubits, got {n_qubits}"
    assert len(coeffs) == 15, f"H2/STO-3G should give 15 Pauli terms, got {len(coeffs)}"
    assert abs(e_hf - E_HF_REF) < 5e-3, f"E_HF off: got {e_hf}, expected ~{E_HF_REF}"
    assert abs(e_fci - E_FCI_REF) < 5e-3, f"E_FCI off: got {e_fci}, expected ~{E_FCI_REF}"
    assert e_fci < e_hf, "FCI must be at or below HF (variational principle)"
    print("  PASS: Hamiltonian shape, HF energy, and FCI reference all match.")


if __name__ == "__main__":
    main()
