# Week 20 — Wider quantum-vs-RBF kernel benchmark

**Tier 2 / 2D.3.** Companion to `week20_kernel_benchmark.py`.

## 1. The sweep grid

Two axes, 9 + 3 = 12 cells:

- **depth** (`reps`) ∈ $\{1, 2, 3\}$ — the only knob the ZZ feature map exposes. The input scale is a second knob, set by the preprocessing (MinMax to $[0, \pi]$); section 3 varies it as a post-hoc control.
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

## 3. Accuracy vs depth, and what the Gram matrix shows

Mean quantum test accuracy across $n_{\text{train}}$ as a function of depth, at the committed input scaling (MinMax to $[0, \pi]$):

| depth | mean Q test | gap to RBF (pp) |
|---:|---:|---:|
| 1 | 0.850 | −15 |
| 2 | 0.767 | −23 |
| 3 | 0.617 | −38 |

At depth 3, $n_{tr}=80$ the test accuracy is exactly 0.50, the chance level.

The first version of these notes read this drop as kernel concentration from accuracy alone. The script now measures the kernel. As a post-hoc control, added after the numbers above were written up and not a pass gate, section 6 of the script prints the off-diagonal mean and std of each training Gram matrix and reruns the quantum sweep with the ZZ-map inputs multiplied by 0.1 (one factor, taken from a review probe, not tuned on test data; CSV columns `posthoc_*`). Means over the three $n_{\text{train}}$ cells:

| depth | $[0, \pi]$: off-diag mean | off-diag std | test | × 0.1: off-diag mean | off-diag std | test |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.097 | 0.111 | 0.850 | 0.500 | 0.247 | 1.000 |
| 2 | 0.089 | 0.098 | 0.767 | 0.417 | 0.257 | 0.967 |
| 3 | 0.083 | 0.086 | 0.617 | 0.303 | 0.258 | 0.983 |

Independent Haar-random 4-qubit states would give mean 1/16 = 0.0625 and std 0.059. At $[0, \pi]$ each added repetition moves the kernel values toward that random-state level, and accuracy falls. With the inputs scaled by 0.1 the spread stays near 0.25 and accuracy stays at 0.97 to 1.00 at every depth (0.983 over the 9 cells against RBF's 1.000; 7 of 9 cells equal RBF, none better).

So at 4 qubits the accuracy drop is a bandwidth effect: the input scale decides how quickly the kernel falls off between nearby points (Shaydulin & Wild, *Importance of Kernel Bandwidth in Quantum Machine Learning*, arXiv:2111.05451). The kernel does not become a constant (its off-diagonal std at depth 3 is still 0.086). Thanasilp et al., *Exponential concentration in quantum kernel methods*, Nat. Commun. 15 (2024), DOI 10.1038/s41467-024-49287-w, is about concentration that grows exponentially with the number of qubits, the same kind of exponential concentration that underlies barren plateaus (Larocca et al., *Barren plateaus in variational quantum computing*, Nat. Rev. Phys. 7, 174–189 (2025), DOI 10.1038/s42254-025-00813-9). This sweep stays at 4 qubits, so it does not test that.

## 4. The runtime accounting

Total wall-clock across the sweep:

| kernel | total time |
|-------|-----------:|
| quantum | 124.9 s |
| RBF | 0.07 s |

A factor of **~1800×** between them, with the quantum side losing on accuracy in every cell. On real hardware the gap widens further: every fidelity needs $\sim 10^4$ shots for $10^{-2}$ precision, taking the per-Gram-matrix cost from 30 s of simulation to minutes of QPU time.

## 5. What would change the verdict

Schuld, *Supervised quantum machine learning models are kernel methods*, arXiv:2101.11020 (2021), makes the equivalence formal: any variational quantum classifier *is* a kernel method with the embedding's induced kernel. The question is therefore not "should I use a quantum kernel?" but "**does my data live in a structure this particular embedding represents well?**" Iris does not. Settings where quantum models have been published as competitive include:

- **Engineered quantum data** (Huang et al., *Power of data in quantum machine learning*, Nat. Commun. 12, 2631 (2021), DOI 10.1038/s41467-021-22539-9): datasets built so that a projected quantum kernel's geometry differs from the classical kernels', where the quantum kernel predicts better. The same paper shows classical models trained on data are often competitive otherwise.
- **Datasets with explicit graph structure** for which an encoding can be designed to respect the graph's symmetries (Skolik et al., *Equivariant quantum circuits for learning on weighted graphs*, npj Quantum Inf. 9, 47 (2023), DOI 10.1038/s41534-023-00710-y; a variational model rather than a kernel). Generic tabular data lacks such structure.

None of these conditions hold for the Iris benchmark. The honest record stays.

## 6. Reading the text plot

For each cell label `dDnN` (depth `D`, train size `N`), one `Q` and one `R` marker. RBF clusters near the right edge (≥ 0.95); quantum scatters left of it. The leftmost point, `d3n80` at 0.50, is the depth-3 cell at the $[0, \pi]$ scaling; with the inputs scaled by 0.1 the same cell reaches 1.00 (section 3).

## 7. What week 21 (capstone) does next

Trade the kernel-method approach back for a *trainable* hybrid model on a non-toy dataset (MNIST 0-vs-1 reduced to 4 features via PCA), and use the gradient-variance probe from week 14 to monitor whether the chosen depth is still trainable. The capstone's deliverables: loss curves, accuracy, and the gradient-variance histogram — all saved alongside `TIER2_REVIEW.md`.

## 8. What the script verifies

- All 9 (depth, $n_{tr}$) Q cells + 3 RBF cells run to completion.
- CSV written with all 12 rows.
- RBF mean test accuracy across $n_{tr}$ exceeds 0.85 (it's 1.00).
- Quantum kernel mean test accuracy at the $[0, \pi]$ scaling does not rise from depth 1 to depth 3 (depth-3 mean ≤ depth-1 mean + 0.05). This checks accuracy, not concentration.
- The post-hoc control (Gram spread, and inputs × 0.1) is printed and written to the `posthoc_*` CSV columns. It is not a gate.
