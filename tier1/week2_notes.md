# Week 2 — Multi-qubit states, Bell pairs, and GHZ

**Codebook I.5–I.7.** Companion to `week2_bell_and_ghz.py`.

## 1. Tensor products

The state space of $n$ qubits is $\mathcal{H} = (\mathbb{C}^2)^{\otimes n}$, dimension $2^n$. The computational basis is

$$|x_1 x_2 \cdots x_n\rangle = |x_1\rangle\otimes|x_2\rangle\otimes\cdots\otimes|x_n\rangle, \qquad x_i\in\{0,1\}.$$

A general $n$-qubit state has $2^n - 1$ complex degrees of freedom (after normalization and global-phase removal):

$$|\psi\rangle = \sum_{x\in\{0,1\}^n} c_x |x\rangle, \qquad \sum_x |c_x|^2 = 1.$$

A state is **separable** if it factors as $|\alpha\rangle\otimes|\beta\rangle$; otherwise it is **entangled**.

## 2. CNOT

$$\mathrm{CNOT} = |0\rangle\langle 0|\otimes I + |1\rangle\langle 1|\otimes X = \begin{pmatrix}1&0&0&0\\0&1&0&0\\0&0&0&1\\0&0&1&0\end{pmatrix}.$$

Acting on the basis: $\mathrm{CNOT}|c, t\rangle = |c, t\oplus c\rangle$.

## 3. The four Bell states

$$|\Phi^+\rangle = \tfrac{1}{\sqrt2}(|00\rangle + |11\rangle), \qquad |\Phi^-\rangle = \tfrac{1}{\sqrt2}(|00\rangle - |11\rangle),$$

$$|\Psi^+\rangle = \tfrac{1}{\sqrt2}(|01\rangle + |10\rangle), \qquad |\Psi^-\rangle = \tfrac{1}{\sqrt2}(|01\rangle - |10\rangle).$$

They are orthonormal and form a basis of $\mathbb{C}^4$ (the **Bell basis**). Construction:

$$|\Phi^+\rangle = \mathrm{CNOT}\,(H\otimes I)\,|00\rangle.$$

The other three are obtained by applying $X$ or $Z$ on the input qubits before $H$/CNOT — equivalently $|\beta_{ij}\rangle = (Z^i X^j \otimes I)|\Phi^+\rangle$.

## 4. Pauli-correlation fingerprint

For two-qubit observables $\sigma_a\otimes\sigma_b$, the four Bell states have a unique signature:

| state | $\langle XX\rangle$ | $\langle YY\rangle$ | $\langle ZZ\rangle$ |
|------|------|------|------|
| $\|\Phi^+\rangle$ | $+1$ | $-1$ | $+1$ |
| $\|\Phi^-\rangle$ | $-1$ | $+1$ | $+1$ |
| $\|\Psi^+\rangle$ | $+1$ | $+1$ | $-1$ |
| $\|\Psi^-\rangle$ | $-1$ | $-1$ | $-1$ |

Sample derivation for $\langle ZZ\rangle$ on $|\Phi^+\rangle$: $ZZ|00\rangle = +|00\rangle$, $ZZ|11\rangle = +|11\rangle$, so the eigenvalue is $+1$.

For $\langle XX\rangle$ on $|\Phi^+\rangle$: rewrite $|00\rangle = \tfrac12(|+\rangle+|-\rangle)(|+\rangle+|-\rangle)$ etc., or simply note $XX(|00\rangle+|11\rangle) = |11\rangle+|00\rangle$, so $|\Phi^+\rangle$ is a $+1$ eigenvector of $XX$.

## 5. Partial trace and the entanglement signature

For a bipartite pure state $|\psi\rangle_{AB}$, the **reduced density matrix** on $A$ is

$$\rho_A = \mathrm{Tr}_B(|\psi\rangle\langle\psi|).$$

For $|\Phi^+\rangle$,

$$\rho_A = \tfrac12(|0\rangle\langle 0| + |1\rangle\langle 1|) = \tfrac{I}{2}.$$

The single-qubit reduced state is **maximally mixed** — *no local measurement on $A$ extracts any information*. This is the operational definition of maximal entanglement; it is impossible for a separable state.

More generally, for a pure $|\psi\rangle_{AB}$,

$$\text{entropy of entanglement} = S(\rho_A) = -\mathrm{Tr}(\rho_A\log\rho_A),$$

which is $1$ bit for any Bell state.

## 6. GHZ states

$$|\mathrm{GHZ}_n\rangle = \tfrac{1}{\sqrt2}(|0\rangle^{\otimes n} + |1\rangle^{\otimes n}).$$

Construction: $H$ on wire 0, then a chain of CNOTs $0\to 1\to 2\to\cdots\to n-1$.

In the $Z$ basis, sampling $|\mathrm{GHZ}_n\rangle$ yields *only* $0\cdots 0$ or $1\cdots 1$, each with probability $1/2$ — perfect $n$-way correlation. GHZ states are maximally fragile (loss of any qubit collapses the entanglement), in contrast to **W states** $\propto |10\cdots 0\rangle + |01\cdots 0\rangle + \cdots$.

## 7. What the script verifies

- All four Bell amplitudes match the textbook coefficients.
- The Pauli-fingerprint table above is reproduced exactly.
- Both single-qubit marginals of $|\Phi^+\rangle$ and $|\Psi^-\rangle$ are $(0.5, 0.5)$.
- 2000 GHZ$_3$ shots split between $|000\rangle$ and $|111\rangle$ only — zero forbidden outcomes.
