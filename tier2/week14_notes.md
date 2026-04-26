# Week 14 — Qiskit cross-implementation + barren-plateau probe

**Tier 2 / 2B.3.** Companion to `week14_qaoa_qiskit_compare.py`.

## 1. Why a second backend at all

Two reasons. First, the QAOA literature lives mostly in Qiskit; being able to round-trip a circuit and a Hamiltonian between PennyLane and Qiskit means you can read a paper's reference implementation without re-deriving it. Second, this is a sanity check on week 13: if both frameworks produce the same energy at the same angles, the bug-surface is the *physics*, not the differentiation engine.

## 2. The two cost-Hamiltonian objects

Same operator, two object types:

- PennyLane: `qml.Hamiltonian(coeffs, observables)` — list of Pauli words and floats.
- Qiskit: `SparsePauliOp(["IIZIIZ", ...], coeffs=[...])` — Pauli string in **right-to-left** convention (qubit 0 is the *last* character).

Index-flip is the one place this gets routinely wrong; the helper builds the Qiskit string with `chars[n - 1 - i] = "Z"` so that string position $n - 1 - i$ refers to qubit $i$. The script verifies the two operators have **identical spectra** ($\le 10^{-14}$) before doing anything else.

## 3. The QAOA ansatz parameter ordering

`QAOAAnsatz(cost_operator=H_q, reps=p)` exposes parameters as `[β[0], γ[0], β[1], γ[1], …]` — **β before γ**. PennyLane circuits in week 13 used `params[layer] = (γ, β)`. Mixing those up looks like the optimization is broken when really only the labels are swapped. The script assigns by parameter object, not by position, to side-step the issue.

## 4. The 5×5 grid sanity check

Qiskit and PennyLane evaluate $\langle\psi(\gamma,\beta)|\hat H_C|\psi(\gamma,\beta)\rangle$ at 25 grid points; the maximum disagreement is **$2.3 \times 10^{-14}$** — full statevector precision. COBYLA over the Qiskit cost recovers $\rho = 0.8485$ at the same angles week 13's PennyLane Adam found. Two independent implementations of the same algorithm landing on the same number is the cleanest possible cross-validation.

## 5. The barren-plateau phenomenon

McClean, Boixo, Smelyanskiy, Babbush, Neven (Nat. Commun. 2018) showed that for sufficiently expressive *hardware-efficient* ansätze, the gradient of the cost with respect to any single parameter has variance

$$\mathrm{Var}\!\left[\frac{\partial\langle O\rangle}{\partial \theta_k}\right] \;=\; \mathcal{O}\!\big(2^{-n}\big),$$

over uniform parameter draws. Concentration is exponential in the qubit count. At reasonable $n$ a randomly initialised circuit has gradients indistinguishable from zero, and gradient-based optimization stalls before training begins. The condition is roughly that the ansatz be a **2-design** on the relevant subspace — depth $\ge \mathcal O(\mathrm{poly}(n))$ for the alternating RY/CNOT family used here.

## 6. The probe in numbers

Hardware-efficient ansatz: each layer = `RY(θ_{Lw})` on every qubit + nearest-neighbour CNOT ladder; depth $L = n$ (so the parameter count grows as $n^2$). Observable $O = Z_0 Z_1$. 100 samples of $\theta \sim U[0, 2\pi)^{nL}$ per $n$.

| $n$ | params | $\mathrm{Var}[\partial_{\theta_0}\langle O\rangle]$ | log₂ ratio to $n=4$ |
|---:|---:|---:|---:|
| 4 | 16 | $9.89 \times 10^{-2}$ | (reference) |
| 6 | 36 | $3.04 \times 10^{-2}$ | −1.70 |
| 8 | 64 | $1.45 \times 10^{-2}$ | −2.77 |
| 10 | 100 | $8.68 \times 10^{-3}$ | −3.51 |

Linear regression on $\log\mathrm{Var}$ versus $n$ gives slope $\approx -0.40$, i.e. **$\mathrm{Var}$ halves every $\sim 1.7$ added qubits** — exactly the exponential decay McClean predicts. The total drop from $n=4$ to $n=10$ is $11.4\times$.

## 7. Why this matters for everything that follows

- **2C variational classifier** uses 4 qubits — small enough to dodge the plateau.
- **2D quantum kernel** is depth-shallow (kernel matrix needs only one circuit per pair), insulating it.
- **Capstone hybrid model** scales features via PCA precisely to avoid this regime: the gradient-variance histogram in week 21 will reuse this same probe to confirm the chosen depth is still trainable.

The lesson the README put in scare quotes — *"watch optimization stall — that is the lesson"* — is now a number on a page.

## 8. Performance footnote

The naive `qml.grad(circuit)(theta)` computes the full gradient vector, which is wasteful when only $\partial_{\theta_0}$ is needed. `tier2/utils/barren.py` splits the parameter signature so only `theta_0` carries `requires_grad`, then uses `diff_method="backprop"`. The result is identical to the parameter-shift rule but ~30× faster at $n=10$.

## 9. What the script verifies

- Qiskit's `SparsePauliOp` and PennyLane's `qml.Hamiltonian` have **identical spectra** (machine zero).
- Their $p=1$ energy surfaces agree to $\sim 10^{-14}$ across a 5×5 $(\gamma,\beta)$ grid.
- Qiskit's COBYLA optimum reproduces week 13's $\rho \approx 0.85$ at the same angles.
- $\mathrm{Var}[\partial_{\theta_0}\langle O\rangle]$ shrinks by more than $4\times$ from $n=4$ to $n=10$ (it shrinks by $11.4\times$ in practice) and is monotone non-increasing in $n$.
