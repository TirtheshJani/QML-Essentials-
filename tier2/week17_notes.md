# Week 17 — Matched-parameter classical baseline

**Tier 2 / 2C.3.** Companion to `week17_classical_baseline.py`.

## 1. The trick of fair comparison

A quantum advantage claim only counts if the classical comparator is given the same budget. Week 16's hybrid model had 13 trainable parameters (8 quantum + 5 classical). The natural classical opponent at the same budget is a **1-hidden-layer MLP** sized so that

$$\underbrace{4 \cdot h}_{\text{input weights}} + \underbrace{h}_{\text{input bias}} + \underbrace{h \cdot 1}_{\text{output weight}} + \underbrace{1}_{\text{output bias}} \;=\; 6h + 1 \;=\; 13 \;\Longleftrightarrow\; h = 2.$$

A `4 → 2 → 1` ReLU MLP it is. Same input, same loss (`BCEWithLogitsLoss`), same optimizer (Adam, `lr=0.05`), same epochs (50, full-batch).

## 2. The sample-size sweep

For each $n_{\text{train}} \in \{10, 20, 40, 60, 80\}$ and each of three seeds, both models train on the first $n_{\text{train}}$ Iris-1-vs-2 examples (after deterministic 80/20 split) and are evaluated on the fixed 20-example test set. Mean and per-seed std reported below.

| $n_{\text{train}}$ | hybrid (mean ± std) | MLP (mean ± std) | gap (Q − C) pp |
|---:|---:|---:|---:|
| 10 | 0.883 ± 0.047 | 0.917 ± 0.047 | −3.33 |
| 20 | 0.850 ± 0.041 | 0.917 ± 0.047 | −6.67 |
| 40 | 0.950 ± 0.041 | 0.967 ± 0.024 | −1.67 |
| 60 | 0.950 ± 0.000 | 0.950 ± 0.041 | +0.00 |
| 80 | 0.983 ± 0.024 | 0.967 ± 0.024 | +1.67 |

Overall: **hybrid 1 win, classical 3 wins, 1 tie.** Mean test accuracy across the curve: hybrid 0.923, MLP 0.943 — a **−2 pp** gap.

## 3. Reading the curve

Both models *saturate near 1.0* once $n_{\text{train}} \ge 40$. They are doing equally good jobs at the same model class — *small, regularized, easy-to-train* — on a 100-example dataset that a linear classifier already nearly solves (week 16's single `nn.Linear` baseline got 95 %). At $n_{\text{train}} = 10\text{–}20$ the classical MLP edges ahead, mostly because ReLU + bias gives a tiny non-linearity that the quantum block's $\langle Z_w\rangle$ readout matches but does not exceed when data is scarce.

## 4. The honest takeaway

At 13 parameters and Iris-1-vs-2, the quantum block does not justify itself. The README is explicit: *"Be honest about whether the quantum kernel actually wins."* Same standard applies to variational classifiers. This isn't a failure of the encoding (week 15) or of the optimisation (week 16); it's the data not being structured in a way the quantum non-linearity can exploit better than ReLU. Iris features are *almost* linearly separable; both models converge to the same boundary.

The cases where the literature has shown hybrid models winning over matched classical baselines are:

- **structured input** with topology the encoding respects (graph-structured QGNNs, geometric quantum encodings),
- **larger $n$** where the quantum block's $2^n$-dimensional state space might offer effective regularisation,
- **datasets with explicit quantum structure** (entangled measurement outcomes, expectation values of Hamiltonians).

Iris is none of those. The capstone (week 21) will use a higher-dimensional dataset (MNIST 0-vs-1 reduced via PCA) and the kernel comparison (weeks 18–20) will probe a *different* class of model entirely. Whether they tip the balance is the open question of tier 2.

## 5. Variance footnote

Per-seed standard deviation is at most 0.047 (a single-example flip on a 20-test set). Both models are reproducible across seeds within $< 5$ pp, well below the plan's threshold. The cross-seed stability is itself a small piece of evidence that the optimization is not the issue.

## 6. What the script verifies

- Hybrid and MLP both have **exactly 13** trainable parameters (verified at the top of `main`).
- Per-seed std at every sample size is below 0.25 — reproducibility threshold from the plan.
- Both models reach > 0.85 mean test accuracy at the largest training-set size.
- Five sample-size points, three seeds each — the plan's specified sweep grid.

## 7. What week 18 starts on

The next sub-project (2D, weeks 18–20) abandons trainable circuits entirely and uses the quantum device as a **kernel evaluator**: a `ZZFeatureMap` defines a fidelity-based Gram matrix, and a classical SVM does the rest. The comparison is then quantum *kernel* vs RBF *kernel* with the same downstream classifier — a structurally cleaner head-to-head than this week's parameterised tussle.
