# Week 10 — VQE optimization on H₂

**Tier 2 / 2A.2.** Companion to `week10_vqe_optimization.py`.

## 1. The variational principle

For any normalized state $|\psi(\theta)\rangle$,

$$E(\theta) := \langle\psi(\theta)|\hat H|\psi(\theta)\rangle \;\ge\; E_0,$$

with equality iff $|\psi(\theta)\rangle = |E_0\rangle$. VQE picks a parametrized circuit $U(\theta)$, prepares $|\psi(\theta)\rangle = U(\theta)|\Phi_{HF}\rangle$, and minimizes $E(\theta)$ over $\theta$ classically. The quantum device only measures the energy.

## 2. The AllSinglesDoubles ansatz

For an $n$-spin-orbital Hamiltonian with $\eta$ electrons, the single and double excitations are the operators

$$\hat T_1 = \sum_{ia} t_i^a\,a_a^\dagger a_i, \qquad \hat T_2 = \sum_{ijab} t_{ij}^{ab}\,a_a^\dagger a_b^\dagger a_j a_i,$$

with $i,j$ occupied and $a,b$ virtual in the HF reference. The unitary coupled-cluster (UCCSD) ansatz exponentiates the anti-Hermitized version:

$$|\psi(\theta)\rangle = e^{\hat T - \hat T^\dagger}|\Phi_{HF}\rangle.$$

`qml.AllSinglesDoubles` implements a hardware-friendly Trotter-1 approximation: for each excitation, a `SingleExcitation` or `DoubleExcitation` Givens rotation by angle $\theta_k$.

For H₂/STO-3G:

- HF reference $|1100\rangle$,
- singles $\{(0\to 2),\ (1\to 3)\}$,
- one double $(0,1\to 2,3)$,
- → **3 trainable parameters**.

## 3. Why singles drop out for H₂

The closed-shell singlet HF state has both spin orbitals of one spatial orbital occupied. The single excitations $a_2^\dagger a_0$ and $a_3^\dagger a_1$ break that symmetry (they would create an $|1010\rangle$ or $|0101\rangle$ component). Spin-symmetry of the exact ground state forces their amplitudes to **zero** at the minimum. The script confirms this: $\theta_1 = \theta_2 = 0$, only the double $\theta_3 \approx 0.225$ rad is non-zero.

## 4. Optimization

Adam (`qml.AdamOptimizer`) with step size 0.1 reaches chemical accuracy in ~30 iterations and converges to $|\Delta E| < 10^{-7}$ Ha by iteration ~100. PennyLane gradients are exact via the parameter-shift rule

$$\frac{\partial E}{\partial \theta_k} = \tfrac12\big[E(\theta + \tfrac\pi2 e_k) - E(\theta - \tfrac\pi2 e_k)\big]$$

— no finite differences, no shot noise on the simulator.

## 5. Result

| quantity | value (Ha) |
|---------|-----------:|
| $E_{HF}$ | −1.11677 |
| $E_{VQE}$ | −1.13728 |
| $E_{FCI}$ | −1.13729 |
| $E_{VQE} - E_{FCI}$ | < 0.001 mHa |

For H₂ the ansatz is *exact* in this basis: AllSinglesDoubles spans the full 2-electron 4-orbital singlet manifold. Week 11 will trace this same calculation across the full bond-dissociation curve and check how well it holds when HF starts to fail badly.

## 6. What the script verifies

- The variational bound $E_{VQE} \ge E_{FCI}$ is respected to numerical precision.
- $E_{VQE}$ lands within 1.6 mHa of FCI (chemical accuracy).
- The HF–FCI correlation gap is closed to ≥ 99 %.
