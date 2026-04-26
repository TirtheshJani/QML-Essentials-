# Week 18 — Quantum kernel setup with the ZZ feature map

**Tier 2 / 2D.1.** Companion to `week18_quantum_kernel_setup.py`.

## 1. The shift from variational to kernel

Sub-projects 2A–2C trained *parametrised* circuits (VQE, QAOA, variational classifier). Sub-project 2D abandons trainable circuits entirely and uses the quantum device as a **kernel evaluator**. The trainable model is a classical SVM that consumes a Gram matrix; the quantum part contributes only the feature map $x \mapsto |\phi(x)\rangle$ and the inner product

$$K(x, x') = |\langle\phi(x)\,|\,\phi(x')\rangle|^2.$$

There are no quantum optimization parameters and therefore no gradients, no barren plateaus, and no warm-start issues. The price is that all model expressivity comes from the (data-independent) feature map alone.

## 2. The ZZ feature map (Havlíček et al. 2019)

Two alternating Hadamard / data-encoding layers:

$$U_{ZZ}(x) = U_Z(x)\,H^{\otimes n}\,U_Z(x)\,H^{\otimes n}, \qquad U_Z(x) = \prod_i e^{i x_i Z_i}\prod_{i<j} e^{i (\pi - x_i)(\pi - x_j) Z_i Z_j}.$$

The single-qubit factors encode each feature individually; the pairwise $ZZ$ factors entangle them. Because the encoding is diagonal in the computational basis between the Hadamards, the IQP-class circuit is conjectured **classically hard to simulate** at fixed depth (Bremner–Jozsa–Shepherd 2010). That hardness is the origin of any "advantage" the kernel might confer.

Parameter `reps=2` doubles the ZZ block once, producing a 4-qubit, ~30-depth circuit when `feature_dimension = 4`.

## 3. Computing the Gram matrix

`FidelityQuantumKernel(feature_map=fm)` evaluates $K_{ij} = |\langle 0|U_{ZZ}^\dagger(x_i)U_{ZZ}(x_j)|0\rangle|^2$ — a single fidelity per pair, simulated exactly on `default.qubit`. Cost: $\mathcal O(n^2)$ circuit evaluations. At $n = 40$ this takes ~5 s on a laptop.

Sanity properties:

- **Symmetry.** $K_{ij} = K_{ji}$ to within $10^{-16}$ (statevector simulation).
- **Diagonal $= 1$.** Each $|\phi(x)\rangle$ is a normalised pure state.
- **PSD.** $K = \Phi^\dagger \Phi$ where $\Phi_i = |\phi(x_i)\rangle$ — automatic from construction. Empirically $\lambda_{\min}(K) = 0.28$ on this 40-point set, well clear of the $-10^{-8}$ numerical-PSD threshold.

## 4. The first whiff of kernel concentration

Mean within-class similarity is **0.0788**; mean between-class similarity is **0.0733**. The gap is just 0.6 pp. That tiny separation is what the SVM in week 19 has to amplify into a decision boundary. Compare to a trivial RBF kernel on the same standardised features, where within-class fidelity is ~0.6 and between-class ~0.1 — the quantum kernel is *less* discriminative on average.

This is the **kernel concentration** phenomenon (Thanasilp et al. 2024, *Nat. Commun.* 15:5200): for sufficiently expressive feature maps, $K_{ij}$ concentrates around a constant value as $n$ qubits or circuit depth grows, and the SVM is left fitting noise. The harder Iris pair (1 vs 2) shows it more starkly — `reps=2` there gives within − between = $-0.0001$, *negative*. Week 20 sweeps `reps ∈ {1, 2, 3}` and watches concentration tighten in real time.

## 5. ASCII heat-map reading

Rows and columns sorted by class. The block structure is visible if you squint: the upper-left 20×20 (class 0) and lower-right 20×20 (class 1) blocks are slightly denser than the off-diagonal blocks. The "@" markers on the diagonal are the diagonal $K_{ii} = 1$. Off-diagonal density is mostly 0.05–0.20 with occasional outliers.

## 6. What the script verifies

- $K = K^\top$ to $10^{-9}$.
- $\mathrm{diag}(K) = \mathbf{1}$ to $10^{-6}$.
- $\lambda_{\min}(K) \ge -10^{-8}$ (numerical PSD; in practice $0.28$).
- Mean within-class similarity exceeds mean between-class similarity (by 0.6 pp here — small but real).

## 7. What week 19 will do

Hand the precomputed Gram matrix to `sklearn.svm.SVC(kernel='precomputed')`, train it via 5-fold CV, and compare head-to-head against `SVC(kernel='rbf')` on the same features and the same `C`. Both numbers reported, no cherry-picking — the readme's standing instruction.
