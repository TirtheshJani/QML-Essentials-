# Week 6 — The Quantum Fourier Transform

**Codebook I.14–I.15.** Companion to `week6_qft.py`.

## 1. Definition

For $N = 2^n$, the **quantum Fourier transform** acts on basis states as

$$\mathrm{QFT}\,|x\rangle = \frac{1}{\sqrt N}\sum_{y=0}^{N-1} e^{2\pi i\,xy/N}\,|y\rangle.$$

It is the discrete Fourier transform read as a unitary on $\mathbb{C}^N$, with matrix elements $F_{yx} = \tfrac{1}{\sqrt N}\omega_N^{xy}$, $\omega_N = e^{2\pi i/N}$.

Linearity: a general state transforms as

$$\mathrm{QFT}\sum_x c_x|x\rangle = \sum_y \tilde c_y|y\rangle, \qquad \tilde c_y = \tfrac{1}{\sqrt N}\sum_x e^{2\pi i\,xy/N} c_x.$$

## 2. Tensor-product form

Write $x = x_1 x_2\cdots x_n$ in binary, and the binary fraction $0.x_k x_{k+1}\cdots x_n := \sum_{j\ge k} x_j 2^{-(j-k+1)}$. Then

$$\mathrm{QFT}\,|x\rangle = \frac{1}{\sqrt N}\bigotimes_{k=1}^{n}\Big(|0\rangle + e^{2\pi i\,0.x_k x_{k+1}\cdots x_n}|1\rangle\Big).$$

Each output qubit is a single-qubit superposition with a phase that depends on a successively shorter tail of the input bits. This decomposition is what makes the QFT efficient: the $2^n$-dim transform compiles to $O(n^2)$ gates.

## 3. The textbook circuit

For each qubit $j = 1,\ldots,n$:

1. Apply $H$ to qubit $j$.
2. For each $k > j$, apply a controlled phase $R_{k-j+1} = \mathrm{diag}(1,\,e^{2\pi i / 2^{\,k-j+1}})$ from qubit $k$ onto qubit $j$.

Finish by a chain of $\mathrm{SWAP}$ gates that reverses the qubit order.

Gate count: $\binom{n}{2} + n = O(n^2)$. The classical FFT is $O(N\log N)$, so the QFT is exponentially faster — but you pay for it: the result is encoded in amplitudes, not in directly-readable values.

## 4. Inverse

Since $\mathrm{QFT}$ is unitary,

$$\mathrm{QFT}^{-1}\,|y\rangle = \frac{1}{\sqrt N}\sum_{x}e^{-2\pi i\,xy/N}|x\rangle,$$

and $\mathrm{QFT}^{-1} \cdot \mathrm{QFT} = I$. The circuit is the same with conjugated phases (or equivalently, run in reverse with $\mathrm{adjoint}$).

## 5. Period → frequency (the Shor structure)

Take a state periodic with period $r$ that divides $N$:

$$|\psi\rangle = \frac{1}{\sqrt K}\sum_{j=0}^{K-1}|x_0 + j r\rangle, \qquad K = N/r.$$

Apply QFT and use $\sum_{j=0}^{K-1} e^{2\pi i\,jr y/N} = K\,\mathbb{1}[y\equiv 0\bmod K]$:

$$\mathrm{QFT}|\psi\rangle = \frac{1}{\sqrt r}\sum_{m=0}^{r-1} e^{2\pi i\,x_0 m K / N}|m K\rangle.$$

The amplitude is supported **only on multiples of $K = N/r$**. Measuring yields a multiple of $N/r$, which (with continued fractions) recovers the period $r$. This is the engine of Shor's factoring algorithm.

The script verifies this on $n=4$, $N=16$:

| period $r$ | $N/r$ | nonzero peaks |
|----:|----:|------|
| 2 | 8 | $y\in\{0, 8\}$ |
| 4 | 4 | $y\in\{0, 4, 8, 12\}$ |
| 8 | 2 | $y\in\{0, 2, 4, \ldots, 14\}$ |

## 6. Why amplitudes, not values

After QFT we cannot read off $\tilde c_y$ directly — only sample $y$ with probability $|\tilde c_y|^2$. Algorithms that use QFT (Shor, QPE, hidden-subgroup) must arrange for the *probability distribution* to encode the answer, not the amplitudes themselves.

## 7. What the script verifies

- Hand-rolled QFT matches `qml.QFT` and the analytic formula to $\sim 10^{-15}$ across all 8 inputs at $n=3$.
- $\mathrm{QFT}|0\rangle^{\otimes n}$ is the uniform distribution.
- $\mathrm{QFT}^{-1}\circ\mathrm{QFT}$ is the identity to machine precision on every basis state.
- A periodic input with period $r$ produces output peaks only at multiples of $N/r$.
