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

$\rho$ is QAOA's expected cut divided by the optimum $C^* = 7$, which week 13 finds by brute force over all $2^6 = 64$ cuts. That brute-force search is an exact classical solve of this instance ($\rho = 1$). Goemans–Williamson's 0.878 is a worst-case guarantee over all graphs, so QAOA's 0.98 on one instance exceeding it is not a head-to-head comparison. **But:** the same instance never gets larger than 6 nodes here. Week 14's barren-plateau probe on a wider hardware-efficient ansatz showed gradient variance halving every ~1.7 added qubits — by 10 qubits, gradients are ~10× smaller than at 4. Scaling QAOA past tens of qubits is gated by the same problem.

### 2C — Variational classifier on Iris (weeks 15–17)

Three results, all instructive:

- **Encoding choice dominated.** On Iris-1-vs-2, *angle* hit 100 %, *amplitude* hit 30 %, IQP hit 75 % (week 15). The classical no-free-lunch lesson, restated: a feature map is a hypothesis class, and choosing the wrong one is unrecoverable downstream.
- **Hybrid ahead of linear by one test example.** A 13-parameter `TorchLayer + Linear` model reached 100 % vs a 5-parameter single `nn.Linear` baseline at 95 % (week 16). That is one test example of 20 from a single seed, so it is within noise.
- **Hybrid ≤ matched-param classical MLP.** A 13-parameter 4-2-1 ReLU MLP trained under identical protocol won 3 of 5 sample-size cells (week 17). Mean test accuracy across the sweep: hybrid 0.923, MLP 0.943. **−2 pp gap in favor of classical**, within seed noise (3 seeds, a 20-example test set where one example is 5 pp).

The variational quantum classifier's competitive niche on a tabular dataset like Iris does not exist at this scale.

### 2D — Quantum kernel + SVM (weeks 18–20)

The largest gap in tier 2, at the input scaling the scripts used. A post-hoc control shows that most of it comes from that scaling.

| (depth, $n_{tr}$) cell | quantum test acc | RBF test acc |
|------|---:|---:|
| best Q (d=1, n=80) | 0.95 | 1.00 |
| worst Q (d=3, n=80) | 0.50 | 1.00 |
| **mean across all 9 cells** | **0.74** | **1.00** |

Quantum wins **0 of 9 cells.** Wall-clock cost: 125 s for the quantum sweep vs 0.07 s for RBF — a factor of ~1800. At the committed input scaling (MinMax to $[0, \pi]$) the quantum mean test accuracy fell as `reps` grew from 1 to 3 (0.85 → 0.77 → 0.62). Accuracy alone does not say why. Week 20 now also runs a post-hoc control, added after these results were written up and not a pass gate: it prints the off-diagonal spread of each training Gram matrix, and reruns the quantum sweep with the ZZ-map inputs multiplied by 0.1 (one factor, taken from a review probe, not tuned on test data). Means over the three $n_{tr}$ cells:

| reps | $[0, \pi]$ inputs: Gram off-diagonal std | test acc | inputs × 0.1 (post-hoc): off-diagonal std | test acc |
|---:|---:|---:|---:|---:|
| 1 | 0.111 | 0.85 | 0.247 | 1.00 |
| 2 | 0.098 | 0.77 | 0.257 | 0.97 |
| 3 | 0.086 | 0.62 | 0.258 | 0.98 |

For independent Haar-random 4-qubit states the off-diagonal std would be 0.059. At $[0, \pi]$ each added repetition moves the kernel values toward that random-state level and accuracy falls. With the inputs scaled by 0.1 the spread stays near 0.25 and the quantum kernel reaches 0.98 over the 9 cells against RBF's 1.00 (7 of 9 cells equal, none better). At 4 qubits this is a bandwidth effect: the input scale sets how quickly the kernel falls off between nearby points (Shaydulin & Wild, *Importance of Kernel Bandwidth in Quantum Machine Learning*, arXiv:2111.05451). The RBF kernel got a width calibrated to its input scale (week 19) and the quantum kernel did not. Thanasilp et al., *Exponential concentration in quantum kernel methods*, Nat. Commun. 15 (2024), DOI 10.1038/s41467-024-49287-w, concerns concentration that grows exponentially with the number of qubits, which a 1–3 reps sweep at 4 qubits does not test.

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

| sub-project | quantum metric | classical baseline | verdict |
|------|------:|------:|------|
| 2A VQE on H₂ | 0.001 mHa from FCI | exact diag at $n=4$ is FCI itself | **tied (both exact)** |
| 2B QAOA $p=3$ | $\rho = 0.98$ (expected cut / optimum) | brute force over 64 cuts, exact ($\rho = 1$) | **classical exact; GW's 0.878 worst-case bound is not a head-to-head** |
| 2C variational classifier (week 16) | 1.00 test on Iris-1-vs-2 | single `nn.Linear` 0.95 | **hybrid +5 pp: one test example of 20, single seed, within noise** |
| 2C param-matched (week 17) | 0.92 mean across n_train | 4-2-1 MLP at 0.94 mean | **classical +2 pp, within seed noise (3 seeds)** |
| 2D quantum kernel (week 19) | 0.70 CV / 0.70 test | RBF 1.00 / 1.00 | **classical wins −30 pp at the $[0, \pi]$ input scaling** |
| 2D wider sweep (week 20) | 0.74 mean across 9 cells; 0.98 with inputs × 0.1 (post-hoc) | RBF 1.00 | **classical wins, 0/9 cells; with inputs × 0.1, 0/9 wins and 7/9 ties** |
| Capstone (week 21) | 0.986 on digits-0-vs-1 | not computed; would be ~1.0 for sklearn LogReg | **likely tied, capstone strength is the architecture** |

**Net:** no clear quantum win in the 7 rows. 2A is a tie (both exact). In 2B the brute-force classical solve is exact on this 6-node instance, and QAOA's 0.98 is measured against it. The two 2C rows differ by one test example from one seed (week 16) or by 2 pp within seed noise (week 17). Classical wins on the kernel task (2D, both rows) at the input scaling the scripts used; with the inputs scaled by 0.1 (a post-hoc control) the quantum kernel comes within 2 pp (0.98 vs 1.00). The capstone has no classical run. This fits the kernel view in Schuld, *Supervised quantum machine learning models are kernel methods*, arXiv:2101.11020 (2021): the way data is encoded into quantum states is the main ingredient that can set a quantum model apart from a classical one. Huang et al. 2021 make the related point that quantum models compete only on data with structure their encoding actually represents.

The two quantum-structured rows (2A, 2B) are also the two where the classical reference is exact at this size: VQE is checked against exact diagonalization (FCI), and QAOA against a brute-force optimum. Neither is a quantum win; both show the quantum method reaching (2A) or approaching (2B) a classical answer that is cheap to compute on 4 or 6 qubits.

## 4. What I'd change before tier 3

In rough order of expected impact:

1. **Pick datasets with quantum structure.** Iris and digits were chosen for repo cleanliness, not for quantum-advantage candidacy. Tier 3 should pick either (a) a synthetic dataset generated by a quantum process (so the "right" encoding is literally known), or (b) a real dataset with discrete combinatorial structure that maps cleanly onto a quantum encoding (graph-classification is the obvious candidate).
2. **Move from `default.qubit` to a real backend.** Everything in tier 2 ran on a noiseless statevector simulator. Half the practical knowledge — shot noise, decoherence, gate errors, calibration drift — is invisible in this regime. IBM's free 7-qubit access is the obvious next step; Pennylane has the integrations.
3. **Keep the assertion-gated pattern.** It paid off repeatedly in tier 2. The week 11 VQE bug (false-converged optimizer, returning HF energy at every $r$) would have shipped without the assertion that demanded gap < 5 mHa across the curve. Assertion gates >> notebook screenshots for catching silent failures.
4. **Move barren-plateau monitoring earlier in the workflow.** The probe was useful in week 14 and week 21 but I should have built it in week 10 and consulted it before every architectural choice. "Is this depth still trainable at this width?" should be the first question, not the last.
5. **Set the quantum kernel's bandwidth before comparing, or pick a different model class.** At the $[0, \pi]$ input scaling, week 20's quantum kernel + RBF comparison was a one-sided fight; the post-hoc control shows that scaling the inputs by 0.1 closes most of the gap (0.98 vs 1.00), so the fight was mostly about input scale. Tier 3's project should either commit to a *parameterized* quantum kernel (where the encoding's hyperparameters are learnable) or skip kernel methods entirely.
6. **Honest reproducibility.** Every week ran with `seed=0` and full-batch training; cross-seed reporting was thinner than ideal. Tier 3 should report mean ± std across at least 5 seeds for any headline number, and run any "the quantum side wins" claim through a permutation test.

## 5. Closing thought

Tier 2 was the right shape: four sub-projects that cover the standard Schuld–Petruccione textbook (chapters 5–7) plus an honest comparison with classical methods on every one. The negative results from 2C param-matched and 2D (whose gap turned out to be mostly input scaling) were the most valuable findings because they're the ones I would have hand-waved past if the gates had been "did the quantum thing produce a number" instead of "did the quantum thing beat the matched classical baseline".

The unsexy summary: **on the kinds of problems QML competes for today (small tabular data, generic features, simulator scale), classical is harder to beat than the marketing suggests.** Tier 3 should pick problems where there is a defensible reason quantum *should* win, not problems where it's politely competitive.

— end of tier 2 review.
