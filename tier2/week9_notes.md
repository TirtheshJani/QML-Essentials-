# Week 9 — H₂ Hamiltonian and the Hartree–Fock reference

**Tier 2 / 2A.1.** Companion to `week9_vqe_h2_setup.py`.

## 1. The electronic Hamiltonian

For a molecule with fixed nuclear positions (Born–Oppenheimer), the electronic Hamiltonian in second quantization is

$$\hat H = \sum_{pq} h_{pq}\,a_p^\dagger a_q + \tfrac12 \sum_{pqrs} h_{pqrs}\,a_p^\dagger a_q^\dagger a_r a_s + E_{nn},$$

with $h_{pq}$ the one-body integrals (kinetic + nuclear-attraction), $h_{pqrs}$ the two-body Coulomb integrals, and $E_{nn} = \sum_{A<B} Z_A Z_B / R_{AB}$ the constant nuclear-nuclear repulsion. The full procedure is:

1. Choose a basis (here STO-3G — one contracted Gaussian per H atom).
2. Solve the self-consistent Hartree–Fock equations to get molecular orbitals.
3. Transform the integrals into the MO basis.
4. Map fermionic operators to qubits via Jordan–Wigner: $a_j \mapsto \frac12(X_j + iY_j)\prod_{k<j} Z_k$.

PennyLane's `qml.qchem.molecular_hamiltonian` does all four in one call and returns a Pauli-string `Hamiltonian` plus the qubit count.

## 2. The Jordan–Wigner image of H₂/STO-3G

H₂ in STO-3G has 2 spatial × 2 spin = **4 spin orbitals → 4 qubits**. The Hamiltonian comes out as **15 Pauli strings** of weight ≤ 2:

$$\hat H = c_0 I + \sum_i c_i Z_i + \sum_{i<j} c_{ij} Z_i Z_j + c_{XY} \big(Y_0 X_1 X_2 Y_3 - X_0 X_1 Y_2 Y_3 - \cdots\big).$$

The $Z$-only block is diagonal and just shifts basis-state energies; the $X/Y$ four-body terms couple Slater determinants — they are what HF cannot capture.

## 3. The Hartree–Fock reference

In the spin-orbital ordering $(1\alpha, 1\beta, 2\alpha, 2\beta)$ the closed-shell singlet puts both electrons in the lowest spatial orbital, giving the Slater determinant

$$|\Phi_{HF}\rangle = |1100\rangle.$$

Its energy is just $\langle 1100|\hat H|1100\rangle$ — a single circuit with `qml.BasisState` and `qml.expval(H)`. At $r = 1.398$ bohr (= 0.74 Å):

| quantity | value (Ha) |
|---------|-----------:|
| $E_{HF}$ | −1.1168 |
| $E_{FCI}$ | −1.1373 |
| $E_{HF} - E_{FCI}$ | +0.0205 |

The 20 mHa gap is **static (left-right) correlation**: at finite separation the true ground state has weight on $|0011\rangle$ as well, which $|1100\rangle$ alone cannot represent.

## 4. Why this matters for VQE

Chemical accuracy is **1.6 mHa** ($\approx 1$ kcal/mol). HF misses by 20 mHa even at equilibrium, and the error blows up at dissociation (week 11). VQE's job is to close that gap with a parametrized ansatz that *can* mix in the doubly-excited determinant $|0011\rangle$.

## 5. What the script verifies

- `qml.qchem.molecular_hamiltonian` returns 4 qubits and 15 Pauli terms for H₂/STO-3G.
- $E_{HF}$ from the quantum circuit agrees with the textbook STO-3G value within 1 mHa.
- The exact diagonalization $E_{FCI}$ is below $E_{HF}$, satisfying the variational bound that week 10's VQE must also respect.
