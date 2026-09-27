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

`FidelityQuantumKernel(feature_map=fm)` evaluates $K_{ij} = |\langle 0|U_{ZZ}^\dagger(x_i)U_{ZZ}(x_j)|0\rangle|^2$ — a single fidelity per pair, computed exactly from statevectors (no shot noise). Cost: $\mathcal O(n^2)$ circuit evaluations. At $n = 40$ this takes ~5 s on a laptop.

Sanity properties:

- **Symmetry.** $K_{ij} = K_{ji}$ to within $10^{-16}$ (statevector simulation).
- **Diagonal $= 1$.** Each $|\phi(x)\rangle$ is a normalised pure state.
- **PSD.** $K = \Phi^\dagger \Phi$ where $\Phi_i = |\phi(x_i)\rangle$ — automatic from construction. Empirically $\lambda_{\min}(K) = 0.107$ on this 40-point set, well clear of the $-10^{-8}$ numerical-PSD threshold.

## 4. Within- vs between-class similarity

Mean within-class similarity (off-diagonal) is **0.0849**; mean between-class similarity is **0.0656**. The gap is 1.9 pp. That small separation is what the SVM in week 19 has to turn into a decision boundary.

Small, nearly uniform kernel values are what **kernel concentration** looks like (Thanasilp et al., *Nat. Commun.* 15 (2024), DOI 10.1038/s41467-024-49287-w): for sufficiently expressive feature maps, $K_{ij}$ concentrates around a constant value, exponentially in the number of qubits $n$, and the SVM is left fitting noise. This script does not test that: it runs one depth at 4 qubits. Week 20 sweeps `reps ∈ {1, 2, 3}` and prints the Gram matrix's off-diagonal spread at each depth; at 4 qubits its post-hoc control finds that the input scale matters more (week 20 notes, section 3).

## 5. ASCII heat-map reading

Rows and columns sorted by class. The block structure is visible if you squint: the upper-left 20×20 (class 0) and lower-right 20×20 (class 1) blocks are slightly denser than the off-diagonal blocks. The "@" markers on the diagonal are the diagonal $K_{ii} = 1$. Off-diagonal density is mostly 0.05–0.20 with occasional outliers.

## 6. What the script verifies

- $K = K^\top$ to $10^{-9}$.
- $\mathrm{diag}(K) = \mathbf{1}$ to $10^{-6}$.
- $\lambda_{\min}(K) \ge -10^{-8}$ (numerical PSD; in practice $0.107$).
- Mean within-class similarity exceeds mean between-class similarity (by 1.9 pp here).

## 7. What week 19 will do

Hand the precomputed Gram matrix to `sklearn.svm.SVC(kernel='precomputed')`, train it via 5-fold CV, and compare head-to-head against `SVC(kernel='rbf')` on the same features and the same `C`. Both numbers reported, no cherry-picking — the readme's standing instruction.
