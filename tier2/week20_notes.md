# Week 20 — Wider quantum-vs-RBF kernel benchmark

**Tier 2 / 2D.3.** Companion to `week20_kernel_benchmark.py`.

## 1. The sweep grid

Two axes, 9 + 3 = 12 cells:

- **depth** (`reps`) ∈ $\{1, 2, 3\}$ — the only knob the ZZ feature map exposes.
- **$n_{\text{train}}$** ∈ $\{20, 40, 80\}$ — sample-size axis.
- **kernels**: quantum at every (depth, $n$) cell; RBF once per $n$ (it has no depth).

Every cell records: best $C$ from a 5-fold CV grid over $\{0.1, 1, 10, 100\}$, the CV mean ± std at that $C$, the held-out test accuracy, and wall-clock seconds. Results dumped to `week20_results.csv`.

## 2. Headline numbers

| depth | $n_{tr}=20$ Q test | $n_{tr}=40$ Q test | $n_{tr}=80$ Q test | mean Q | RBF |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.65 | 0.95 | 0.95 | **0.85** | 1.00 |
| 2 | 0.80 | 0.70 | 0.80 | 0.77 | 1.00 |
| 3 | 0.65 | 0.70 | 0.50 | 0.62 | 1.00 |

**Quantum wins in 0 of 9 cells.** The closest the quantum kernel gets is `depth=1, n_train=40 or 80` at 0.95 — five points behind RBF's perfect score.

## 3. The kernel-concentration trajectory

Mean quantum test accuracy across $n_{\text{train}}$ as a function of depth:

| depth | mean Q test | gap to RBF (pp) |
|---:|---:|---:|
| 1 | 0.850 | −15 |
| 2 | 0.767 | −23 |
| 3 | 0.617 | −38 |

Each added repetition makes the ZZ feature map more expressive and pushes more pairs of fidelities $K_{ij}$ toward the same value (Thanasilp et al. 2024, Larocca et al. 2025). The SVM is then optimizing on noise, and accuracy drifts toward the chance baseline of 0.5 — at depth=3, $n_{tr}=80$ we hit exactly 0.50.

This is the same exponential-concentration phenomenon as the McClean barren plateau (week 14), reframed for kernel methods: **expressivity is not free**. A maximally expressive feature map is also maximally indistinguishable across inputs, and the kernel becomes a constant.

## 4. The runtime accounting

Total wall-clock across the sweep:

| kernel | total time |
|-------|-----------:|
| quantum | 124.9 s |
| RBF | 0.07 s |

A factor of **~1800×** between them, with the quantum side losing on accuracy in every cell. On real hardware the gap widens further: every fidelity needs $\sim 10^4$ shots for $10^{-2}$ precision, taking the per-Gram-matrix cost from 30 s of simulation to minutes of QPU time.

## 5. What would change the verdict

The Schuld 2021 *PRL* "Quantum machine learning models are kernel methods" line of work makes the equivalence formal: any variational quantum classifier *is* a kernel method with the embedding's induced kernel. The question is therefore not "should I use a quantum kernel?" but "**does my data live in a structure this particular embedding represents well?**" Iris does not. Datasets where quantum kernels have been published as competitive include:

- **Quantum-generated data** (Huang et al. 2021, *Nat. Commun.*): synthetic labels engineered from a quantum process. The kernel is then optimal by construction.
- **Datasets with explicit graph or sequence structure** for which an encoding can be designed (Skolik et al. 2023). Generic tabular data lacks such structure.
- **Hardware noise as inductive bias** (Suzuki et al. 2024, preliminary): noisy quantum kernels sometimes regularize better than noiseless ones.

None of these conditions hold for the Iris benchmark. The honest record stays.

## 6. Reading the text plot

For each cell label `dDnN` (depth `D`, train size `N`), one `Q` and one `R` marker. RBF clusters near the right edge (≥ 0.95); quantum scatters left of it. The leftmost point — `d3n80` at 0.50 — is the depth-3 catastrophe: 80 examples are *not enough to overcome* the loss of signal from concentration; more data would just let the SVM fit the constant kernel more confidently.

## 7. What week 21 (capstone) does next

Trade the kernel-method approach back for a *trainable* hybrid model on a non-toy dataset (MNIST 0-vs-1 reduced to 4 features via PCA), and use the gradient-variance probe from week 14 to monitor whether the chosen depth is still trainable. The capstone's deliverables: loss curves, accuracy, and the gradient-variance histogram — all saved alongside `TIER2_REVIEW.md`.

## 8. What the script verifies

- All 9 (depth, $n_{tr}$) Q cells + 3 RBF cells run to completion.
- CSV written with all 12 rows.
- RBF mean test accuracy across $n_{tr}$ exceeds 0.85 (it's 1.00).
- Quantum kernel mean test accuracy *decreases* from depth 1 to depth 3 — the concentration trend baked into the assertions.
