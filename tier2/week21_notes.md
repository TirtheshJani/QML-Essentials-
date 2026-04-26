# Week 21 — Tier 2 capstone: hybrid model on a non-toy dataset

**Tier 2 capstone.** Companion to `week21_capstone_hybrid.py`.
Cross-tier writeup in `TIER2_REVIEW.md` (repo root).

## 1. Why digits and not Iris

Iris has 100 examples per pair and is linearly separable for the easy class pair. Tier 2's capstone needs to *strain* the hybrid pipeline, not coast on it. The scikit-learn `digits` dataset is the natural step up: 8×8 grayscale images of handwritten digits, 64 raw features, ~180 examples per class. The 0-vs-1 binary subset has ~360 examples (vs 100 for Iris-2-class) — almost 4× the data, an order of magnitude more raw features, and a real preprocessing decision (PCA) before the model sees a thing.

`load_digits` is the standard sklearn substitute for "MNIST in two lines". MNIST proper at 28×28 = 784 features would force a much heavier compression; the 8×8 version makes the PCA → 4 features story obvious.

## 2. The pipeline

| stage | dim | what it does |
|------|----:|------|
| raw `digits[0,1]` | (360, 64) | grayscale 8×8 images |
| 80/20 split | (288, 64) + (72, 64) | stratified, fixed seed |
| `PCA(n_components=4)` | 64 → 4 | 73.7 % of variance retained |
| $\frac{\pi}{2}\tanh(z / \sigma_z)$ | 4 → 4 | bounded to $\pm \pi/2$ for `AngleEmbedding` |
| `AngleEmbedding` + `BasicEntanglerLayers(L=2)` | 4 → 4 | quantum block, 8 trainable params |
| `nn.Linear(4, 1)` | 4 → 1 | classical head, 5 trainable params |
| `BCEWithLogitsLoss` | — | binary cross-entropy w/ sigmoid built in |

Same 13-parameter budget as week 16 (the Iris hybrid). Same encoding (week 15's winner). Same optimizer protocol (Adam, lr 0.05, full-batch, 60 epochs). The capstone tests whether the same architecture survives a real preprocessing step + larger non-toy dataset.

## 3. Result

Final test accuracy: **0.986** (71 of 72 test examples correct). Loss drops from 0.70 to 0.04 over 60 epochs; train accuracy plateaus at 0.99 by epoch 20. The single misclassification is presumably a borderline 0/1 image whose first 4 PCs straddle the decision boundary; without the per-example error analysis it's hard to be more specific.

The **plan target was test acc > 0.85**; the model clears it by 14 pp.

## 4. The barren-plateau check

The same `tier2/utils/barren.py` probe from week 14 reruns the exponential-decay measurement at $n \in \{4, 6, 8, 10\}$ qubits with the hardware-efficient $L = n$ ansatz:

| $n$ | params | Var$[\partial_{\theta_0}\langle O\rangle]$ | mean $|\nabla|$ |
|---:|---:|---:|---:|
| 4 | 16 | $9.9 \times 10^{-2}$ | 0.246 |
| 6 | 36 | $3.0 \times 10^{-2}$ | 0.130 |
| 8 | 64 | $1.5 \times 10^{-2}$ | 0.096 |
| 10 | 100 | $8.7 \times 10^{-3}$ | 0.075 |

Variance drops **11.4×** from $n = 4$ to $n = 10$ — same number as week 14, reproducible across seeds. The capstone runs at $n = 4$ where Var is ~ $10^{-1}$ — solidly trainable. The histogram (`week21_barren_histogram.png`) shows the gradient distribution narrowing as $n$ grows, exactly the shape the plateau theorem predicts.

The architectural lesson: **PCA-first matters**. If we had skipped PCA and tried a 64-qubit hybrid, gradients would be ~ $10^{-15}$ — undetectable from optimizer noise. The whole capstone is trainable *because* the encoding budget was capped by a deliberate PCA cut, not despite it.

## 5. Artifacts saved

- `week21_loss_curve.png` — three series on twin axes: training loss (left) and train/test accuracy (right). Loss falls smoothly; train and test accuracy track each other within < 1 pp throughout, indicating no overfitting on this scale.
- `week21_barren_histogram.png` — four-panel histogram of $\partial_{\theta_0}\langle O\rangle$ at $n \in \{4, 6, 8, 10\}$, 100 samples each. Bell shapes centered on zero, with width shrinking visibly with $n$.

## 6. The reductive one-liner per sub-project

The script ends with a five-line tier 2 summary, expanded in `TIER2_REVIEW.md`:

- **2A VQE on H₂**: trainable problem, exact answer within 0.001 mHa of FCI (week 10).
- **2B QAOA on MaxCut**: ρ 0.85 → 0.98 across p = 1 → 3 (week 13).
- **2C variational classifier**: encoding dominated; classical 13-param MLP edged out the hybrid 3/5 → 1/5 across sample sizes on Iris-1-vs-2 (week 17).
- **2D quantum kernel**: 0/9 cells beat RBF on the depth × $n_\text{train}$ sweep; depth made things worse (week 20).
- **Capstone**: 98.6 % on digits 0-vs-1 with the same 13-parameter hybrid, gradients still healthy at $n = 4$.

## 7. What the script verifies

- Test accuracy > 0.85 (actual 0.986).
- Loss falls by more than $2\times$ over training (actual: 0.70 → 0.04, $17\times$).
- Both PNGs (`week21_loss_curve.png` and `week21_barren_histogram.png`) exist on disk.
- Variance ratio $V(n=4) / V(n=10) > 4$ — the plateau check is alive (actual 11.4).

## 8. What was *not* attempted

- Multi-class classification (digits 0–9). Would need a one-vs-rest or amplitude-coded readout; out of scope for binary tier 2.
- Robustness to label noise. The capstone is on clean labels; quantum kernels' robustness story (Suzuki et al. 2024) belongs in tier 3.
- Hardware execution. Everything here ran on `default.qubit`; running the same circuits on a real backend with shot noise is the natural tier 3 escalation.
