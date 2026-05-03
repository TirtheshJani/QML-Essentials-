# Week 22 — QAE Dataset and Fidelity Infrastructure

## What we built

A 1-parameter family of 4-qubit pure states from a quantum-native source:
the H₂/STO-3G electronic ground states $\ket{\psi(r)}$ for
$r \in \{0.4, 0.5, \dots, 2.5\}$ Å.

Each state is obtained by exact diagonalization of the 16×16 matrix form
of the Hamiltonian assembled by `qml.qchem.molecular_hamiltonian` (the
same backend used by the Tier 2 VQE work in weeks 9–11). Concretely:

$$
H(r) = \sum_k c_k(r) P_k, \quad P_k \in \{I, X, Y, Z\}^{\otimes 4},
$$

then
$$
\ket{\psi(r)} = \arg\min_{\ket\phi : \langle\phi|\phi\rangle = 1}
\langle\phi| H(r) |\phi\rangle = v_0(M(r))
$$
where $M(r)$ is the dense matrix of $H(r)$ and $v_0$ is the eigenvector
of the smallest eigenvalue.

## Why this is the right dataset for a QAE

Three reasons, all about quantum structure (Tier 2 review item 1):

1. **Inputs are quantum states by construction.** No encoding choice to
   second-guess (which dominated the Tier 2 classifier results, week 15).
2. **The dataset is geometrically simple.** We measured the singular
   values of the stacked $(22, 16)$ amplitude matrix and the top 4 capture
   > 99.9 % of the variance — i.e. the entire $r$-curve lives in a 4-D
   linear subspace of $\mathbb{C}^{16}$. A 2-qubit code (also 4-D) is the
   *minimal* quantum bottleneck that can fit the dataset losslessly.
3. **It connects Tier 2 to Tier 3.** Same `qml.qchem` machinery; same
   chem helper; the QAE is now compressing the *output* of the VQE
   pipeline.

## Conventions used everywhere in Tier 3

We standardize wire layout once and use it for the rest of the tier.
PennyLane orders basis states with wire 0 most significant, so:

$$
\ket{\psi} = \sum_{q_0 q_1 q_2 q_3} \psi_{q_0 q_1 q_2 q_3} \ket{q_0 q_1 q_2 q_3}
$$

with `psi[q_0*8 + q_1*4 + q_2*2 + q_3]` matching `qml.matrix(H, wire_order=range(4))`.
Then **code wires = [0, 1]**, **trash wires = [2, 3]**, and the reshape
`psi.reshape(4, 4)[c, t]` indexes amplitudes by (code basis, trash basis).

This convention is encoded in `tier3/utils/states.py` constants and
respected by every subsequent week's encoder QNode and partial-trace.

## Partial trace and reduced density matrices

For a pure state $\ket\psi$, the reduced density matrix on the code wires is

$$
\rho_{\text{code}} = \mathrm{Tr}_{\text{trash}} \ket\psi\bra\psi
= T T^\dagger,
\qquad T_{c, t} = \psi_{c \cdot 4 + t}.
$$

Symmetrically, $\rho_{\text{trash}} = T^\dagger T$. The Schmidt
decomposition guarantees $\mathrm{Tr}(\rho_{\text{code}}^2) =
\mathrm{Tr}(\rho_{\text{trash}}^2)$ for any pure $\ket\psi$ — this
identity is asserted at the bottom of the script as a sanity check on the
partial-trace implementation.

## Fidelity definitions used in Tier 3

- **Pure-state overlap:** $F(\psi, \phi) = |\langle\psi | \phi\rangle|^2$.
- **Local trash fidelity (Romero training cost):**
  $F_{\text{loc}}(\psi, U) = \langle U\psi |\, I_{\text{code}} \otimes \ket{0}\bra{0}_{\text{trash}}\, |U\psi \rangle$
  — i.e. probability of measuring the trash bits in $\ket{00}$ after
  encoding.
- **Reconstruction fidelity:**
  $F_{\text{recon}}(\psi, U) = \langle\psi | U^\dagger (\rho_{\text{code}}(U,\psi) \otimes \ket{0}\bra{0}_{\text{trash}}) U | \psi\rangle$
  — the textbook QAE fidelity of the full encode → discard → re-inject
  $\ket{00}$ → decode protocol.
- **Uhlmann fidelity (mixed states):**
  $F(\rho, \sigma) = \big(\mathrm{Tr}\sqrt{\sqrt\rho \, \sigma \, \sqrt\rho}\,\big)^2$ — used in week 25 for noisy reconstructions.

In the perfect-compression limit ($U\ket\psi = \ket{\phi_{\text{code}}}\ket{0}_{\text{trash}}$), all four metrics agree at 1. Local fidelity and
reconstruction fidelity diverge only when compression is imperfect, which
is exactly the regime training has to navigate.

## What week 23 needs from this

A `(22, 16)` complex array of states, a `state_overlap` and a
`reduced_density_matrix` function, and the standard wire layout. All of
these are now exposed by `tier3/utils/states.py`.

## Pass criterion

`abs(F(self, self) - 1) < 1e-10`, all 22 states unit-norm, effective rank
$\le 4$, top-4 SVD captures > 0.999 of variance, partial-trace consistency
on a representative state — every check passes from a fresh checkout.
