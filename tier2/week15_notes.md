# Week 15 — Encoding study on Iris (versicolor vs virginica)

**Tier 2 / 2C.1.** Companion to `week15_encoding_study.py`.

## 1. The dataset choice

Setosa-vs-versicolor (classes 0 and 1) is **linearly separable** in 4-D — every encoding nails 100 % and the comparison is uninformative. Versicolor-vs-virginica (classes **1 and 2**) is the canonical *hard* Iris pair: a linear SVM gets ~95 % and the classes overlap in two of the four features. That overlap is the differential the encoding study needs.

## 2. Three encodings, one trainable head

All three quantum models share:

- 4 qubits (one per Iris feature),
- `BasicEntanglerLayers(n_layers=2, rotation=RY)` as the trainable head — 8 parameters,
- output $\langle Z_0\rangle \mapsto \tfrac12(1 - \langle Z_0\rangle)$ for $P(\text{class}=1)$,
- 60 epochs of Adam at lr 0.05, BCE loss.

The only thing that varies is the embedding stage:

### Angle encoding
$$|\psi(x)\rangle = \bigotimes_{i=1}^4 R_Y(x_i)|0\rangle, \qquad x_i = \tfrac\pi2 \tanh(\bar x_i).$$

One rotation per feature; data lives on the Bloch sphere. Local in features ($i$-th feature only touches qubit $i$) until the entanglers in the head mix them.

### Amplitude encoding
$$|\psi(x)\rangle = \frac{1}{\|x'\|}\sum_{k=0}^{2^n - 1} x'_k\,|k\rangle, \qquad x' = (x_1, x_2, x_3, x_4, 0, \ldots, 0).$$

The 4-D feature vector is zero-padded to 16 amplitudes and L²-normalized. Maximally information-dense — $n$ qubits store $2^n$ real numbers — but **non-linear in features** in a way the trainable head cannot easily undo: the encoder strips both global sign and length, so $x$ and $-x$ get identical states, and so do $x$ and $2x$.

### IQP encoding
$$|\psi(x)\rangle = U_Z(x)\,H^{\otimes n}\,U_Z(x)\,H^{\otimes n}|0\rangle, \qquad U_Z(x) = \prod_i e^{i x_i Z_i}\prod_{i<j} e^{i x_i x_j Z_i Z_j}.$$

Diagonal in the computational basis between Hadamard layers, with single-qubit and pairwise products. Believed to be **classically hard to simulate** at fixed depth (Bremner–Jozsa–Shepherd 2010); the canonical "quantum-only" encoding family that motivates supremacy claims.

## 3. Results

| encoding | params | train acc | test acc | final loss |
|---------|------:|----------:|---------:|-----------:|
| angle | 8 | 0.85 | **1.00** | 0.45 |
| amplitude | 8 | 0.64 | 0.30 | 0.66 |
| IQP | 8 | 0.75 | 0.75 | 0.59 |

The plan's pass criteria — *(a) some encoding > 0.95 and (b) angle vs amplitude gap ≥ 5 pp* — are both satisfied (100 % and 70 pp respectively).

## 4. Reading the amplitude failure

Amplitude at **30 %** test accuracy on a balanced 2-class problem is below chance — the model has confidently learned the wrong decision boundary. This is not a training failure; the loss does decrease from $0.98$ to $0.66$. It is a **representational failure**: amplitude embedding factors out norm and global sign, and the standardised Iris features differ between virginica and versicolor partly *in* their norms. The encoder discards exactly the information that distinguishes the classes.

The honest takeaway is the one Schuld & Petruccione hammer in chapter 6 of *Machine Learning with Quantum Computers*: **encoding choice imposes a hard ceiling no amount of variational training can lift**. Capacity arguments based on Hilbert-space dimension are misleading — the embedding map is doing a lot of unstated work.

## 5. Why angle wins on this pair

Iris features after standardization sit roughly in $[-2, 2]$. After $\frac\pi2 \tanh(\cdot)$ they sit in $\sim(-1.4, 1.4)$ rad, which is the sweet spot of $R_Y$: large enough to push the $\langle Z_0\rangle$ expectation toward $\pm 1$, small enough to stay non-degenerate. The linearity of $R_Y(x_i) \cdot R_Y(x_j) = R_Y(x_i + x_j)$ on a single qubit is broken by the trainable entanglers, giving the model just enough non-linearity for this pair while preserving feature-coordinate identity that amplitude embedding crushes.

## 6. What week 16 will do

Take the **angle** encoding from this comparison, wrap it in `qml.qnn.TorchLayer`, stack a one-layer classical head, and train with PyTorch's optimizer. With ≥ 0.97 test accuracy as the gate (the plan's threshold), week 16 stress-tests whether the hybrid wrapping costs anything vs the raw PennyLane training here.

## 7. What the script verifies

- The best of three encodings exceeds 95 % test accuracy on Iris-1-vs-2.
- The accuracy gap between `angle` and `amplitude` is at least 5 pp (the plan's quoted threshold) — it is in fact 70 pp.
- All three encodings drive their training loss below the random-label baseline $\ln 2 \approx 0.693$, so the training itself is healthy across all three; the differences are about embeddings, not optimization.
