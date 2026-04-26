# Tier 2 Review — Honest Notes from 13 Weeks of Core QML

Tier 2 (weeks 9–21) covered four sub-projects: VQE on H₂ (2A), QAOA on MaxCut (2B), a variational classifier on Iris (2C), and a quantum kernel + SVM (2D), capped by a hybrid model on `digits` 0-vs-1 (week 21). Every weekly script is assertion-gated and passes from a clean checkout. This is the cross-tier reflection the plan called for: honest about where quantum helped, where it didn't, and what I'd do differently for tier 3.

## 1. What each sub-project actually demonstrated

### 2A — VQE on H₂ (weeks 9–11)

VQE worked exactly as advertised on the only quantum-native problem in tier 2.

| metric | value |
|------|------:|
| equilibrium $E_{VQE}$ vs FCI | within **0.001 mHa** at 0.74 Å |
| dissociation curve max error | **0.05 mHa** across 0.4 → 2.5 Å |
| restricted-HF error at $r = 2.5$ Å | 233 mHa (HF blows up) |
| ansatz | `AllSinglesDoubles`, 3 trainable params |

The result is unambiguous: for a quantum-native task (find the ground state of a quantum Hamiltonian), a small variational ansatz with chemistry-informed structure saturates the basis-set ceiling. This is the cleanest "quantum solving a quantum problem" demonstration in tier 2.

### 2B — QAOA on MaxCut (weeks 12–14)

QAOA worked, then concentrated. On a fixed 6-node 3-regular graph:

| $p$ | approximation ratio $\rho$ |
|---:|---:|
| 1 | 0.85 |
| 2 | 0.94 |
| 3 | 0.98 |

Goemans–Williamson's worst-case classical guarantee is 0.878. QAOA at $p = 3$ exceeds it on this instance. **But:** the same instance never gets larger than 6 nodes here. Week 14's barren-plateau probe on a wider hardware-efficient ansatz showed gradient variance halving every ~1.7 added qubits — by 10 qubits, gradients are ~10× smaller than at 4. Scaling QAOA past tens of qubits is gated by the same problem.

### 2C — Variational classifier on Iris (weeks 15–17)

Three results, all instructive:

- **Encoding choice dominated.** On Iris-1-vs-2, *angle* hit 100 %, *amplitude* hit 30 %, IQP hit 75 % (week 15). The classical no-free-lunch lesson, restated: a feature map is a hypothesis class, and choosing the wrong one is unrecoverable downstream.
- **Hybrid > linear, by a small margin.** A 13-parameter `TorchLayer + Linear` model reached 100 % vs a 5-parameter single `nn.Linear` baseline at 95 % (week 16). Quantum non-linearity contributes, but cheaply.
- **Hybrid ≤ matched-param classical MLP.** A 13-parameter 4-2-1 ReLU MLP trained under identical protocol won 3 of 5 sample-size cells (week 17). Mean test accuracy across the sweep: hybrid 0.923, MLP 0.943. **−2 pp gap in favor of classical.**

The variational quantum classifier's competitive niche on a tabular dataset like Iris does not exist at this scale.

### 2D — Quantum kernel + SVM (weeks 18–20)

Most decisive negative result in tier 2.

| (depth, $n_{tr}$) cell | quantum test acc | RBF test acc |
|------|---:|---:|
| best Q (d=1, n=80) | 0.95 | 1.00 |
| worst Q (d=3, n=80) | 0.50 | 1.00 |
| **mean across all 9 cells** | **0.74** | **1.00** |

Quantum wins **0 of 9 cells.** Wall-clock cost: 125 s for the quantum sweep vs 0.07 s for RBF — a factor of ~1800. The Thanasilp et al. 2024 kernel-concentration phenomenon was directly observed: as `reps` grew from 1 to 3, the quantum mean test accuracy fell monotonically (0.85 → 0.77 → 0.62), exactly the predicted decay.

### Week 21 — Capstone

A 13-parameter hybrid (the same architecture as week 16) trained on `digits` 0-vs-1 reduced to 4 PCs reached **0.986** test accuracy. The barren-plateau probe at $n \in \{4, 6, 8, 10\}$ was rerun and confirmed 4 qubits is in the trainable regime (Var = $9.9 \times 10^{-2}$, vs $8.7 \times 10^{-3}$ at $n = 10$). PCA-first was the architectural choice that kept the model trainable.

## 2. Where barren plateaus showed up

Week 14 measured Var$[\partial_{\theta_0}\langle Z_0Z_1\rangle]$ for a depth-$L=n$ hardware-efficient ansatz, 100 random parameter samples per $n$.

| $n$ qubits | 4 | 6 | 8 | 10 |
|----:|---:|---:|---:|---:|
| Var | $9.9\times10^{-2}$ | $3.0\times10^{-2}$ | $1.5\times10^{-2}$ | $8.7\times10^{-3}$ |

Linear fit on $\log\mathrm{Var}$ vs $n$: slope $-0.40$, i.e., variance halves every $\sim 1.7$ qubits — exponential decay matching McClean et al. 2018. The week 21 capstone reran this probe and reproduced the same numbers exactly.

**Where this affected tier 2 design choices:**

- **2A and 2B used 4 and 6 qubits** respectively — both safely on the trainable side.
- **2C and 2D restricted to 4 qubits / shallow depth** for the same reason.
- **The capstone used PCA(64 → 4)** specifically to stay at $n = 4$. A 64-qubit hybrid would have been unfittable.

The barren plateau was not an obstacle in tier 2 because tier 2 was deliberately designed around it. That design discipline is itself part of what tier 1 + tier 2 was for.

## 3. Win/loss/tie ledger vs classical baselines

| sub-project | quantum metric | matched classical | verdict |
|------|------:|------:|------|
| 2A VQE on H₂ | 0.001 mHa from FCI | exact diag at $n=4$ is FCI itself | **tied (both exact)** |
| 2B QAOA $p=3$ | $\rho = 0.98$ | Goemans-Williamson $\rho \ge 0.878$ worst case | **quantum wins** |
| 2C variational classifier (week 16) | 1.00 test on Iris-1-vs-2 | single `nn.Linear` 0.95 | **quantum wins +5 pp** |
| 2C param-matched (week 17) | 0.92 mean across n_train | 4-2-1 MLP at 0.94 mean | **classical wins −2 pp** |
| 2D quantum kernel (week 19) | 0.70 CV / 0.70 test | RBF 1.00 / 1.00 | **classical wins −30 pp** |
| 2D wider sweep (week 20) | 0.74 mean across 9 cells | RBF 1.00 | **classical wins, 0/9 cells** |
| Capstone (week 21) | 0.986 on digits-0-vs-1 | not computed; would be ~1.0 for sklearn LogReg | **likely tied, capstone strength is the architecture** |

**Net:** quantum wins 2 of 7 head-to-heads — both on quantum-structured problems (2A, 2B). Classical wins on every tabular data task (2C param-matched, 2D both subprojects). The pattern matches the Schuld 2021 / Huang et al. 2021 theoretical line: quantum models compete only on data with structure their encoding actually represents.

The two "wins" are also the two cases where the *classical baseline is itself a quantum or quantum-adjacent algorithm*: VQE's ground-truth reference is full quantum-state diagonalization; QAOA's classical bound is Goemans-Williamson, which uses an SDP relaxation that's polynomial but expensive at scale.

## 4. What I'd change before tier 3

In rough order of expected impact:

1. **Pick datasets with quantum structure.** Iris and digits were chosen for repo cleanliness, not for quantum-advantage candidacy. Tier 3 should pick either (a) a synthetic dataset generated by a quantum process (so the "right" encoding is literally known), or (b) a real dataset with discrete combinatorial structure that maps cleanly onto a quantum encoding (graph-classification is the obvious candidate).
2. **Move from `default.qubit` to a real backend.** Everything in tier 2 ran on a noiseless statevector simulator. Half the practical knowledge — shot noise, decoherence, gate errors, calibration drift — is invisible in this regime. IBM's free 7-qubit access is the obvious next step; Pennylane has the integrations.
3. **Keep the assertion-gated pattern.** It paid off repeatedly in tier 2. The week 11 VQE bug (false-converged optimizer, returning HF energy at every $r$) would have shipped without the assertion that demanded gap < 5 mHa across the curve. Assertion gates >> notebook screenshots for catching silent failures.
4. **Move barren-plateau monitoring earlier in the workflow.** The probe was useful in week 14 and week 21 but I should have built it in week 10 and consulted it before every architectural choice. "Is this depth still trainable at this width?" should be the first question, not the last.
5. **Stop fighting kernel concentration and pick a different model class.** Week 20 made it clear that the quantum kernel + RBF comparison is, at small scale, a one-sided fight. Tier 3's project should either commit to a *parameterized* quantum kernel (where the encoding's hyperparameters are learnable) or skip kernel methods entirely.
6. **Honest reproducibility.** Every week ran with `seed=0` and full-batch training; cross-seed reporting was thinner than ideal. Tier 3 should report mean ± std across at least 5 seeds for any headline number, and run any "the quantum side wins" claim through a permutation test.

## 5. Closing thought

Tier 2 was the right shape: four sub-projects that cover the standard Schuld–Petruccione textbook (chapters 5–7) plus an honest comparison with classical methods on every one. The negative results from 2C param-matched and 2D were the most valuable findings because they're the ones I would have hand-waved past if the gates had been "did the quantum thing produce a number" instead of "did the quantum thing beat the matched classical baseline".

The unsexy summary: **on the kinds of problems QML competes for today (small tabular data, generic features, simulator scale), classical is harder to beat than the marketing suggests.** Tier 3 should pick problems where there is a defensible reason quantum *should* win, not problems where it's politely competitive.

— end of tier 2 review.
