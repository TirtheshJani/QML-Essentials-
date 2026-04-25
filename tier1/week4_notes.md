# Week 4 — Deutsch–Jozsa and oracles

**Codebook I.11–I.12.** Companion to `week4_deutsch_jozsa.py`.

## 1. The promise problem

Given a black-box function $f:\{0,1\}^n\to\{0,1\}$ promised to be either

- **constant** ($f(x)$ is the same for all $x$), or
- **balanced** ($f(x)=0$ for exactly half the inputs, $=1$ for the other half),

decide which.

**Classical query complexity:** in the worst case, $2^{n-1}+1$ queries — you can see $2^{n-1}$ identical outputs and still be unable to distinguish "constant 0" from "balanced with that half $=0$".

**Quantum query complexity:** $1$. This is the first algorithm that demonstrates a separation between classical and quantum query complexity.

## 2. Phase oracle

Encode $f$ as a unitary that imprints its value as a phase:

$$U_f|x\rangle = (-1)^{f(x)}|x\rangle.$$

(Equivalent to a standard XOR oracle $|x\rangle|y\rangle \to |x\rangle|y\oplus f(x)\rangle$ acting on an ancilla in $|-\rangle$, by the *phase kickback* trick.)

## 3. The Deutsch–Jozsa circuit

$$|0\rangle^{\otimes n} \xrightarrow{H^{\otimes n}} \tfrac{1}{\sqrt N}\sum_x |x\rangle \xrightarrow{U_f} \tfrac{1}{\sqrt N}\sum_x (-1)^{f(x)}|x\rangle \xrightarrow{H^{\otimes n}} \sum_y c_y |y\rangle,$$

where $N = 2^n$ and, using $H^{\otimes n}|x\rangle = \tfrac{1}{\sqrt N}\sum_y (-1)^{x\cdot y}|y\rangle$,

$$c_y = \frac{1}{N}\sum_x (-1)^{f(x) \oplus (x\cdot y)}.$$

## 4. The verdict

The amplitude on $|0\rangle^{\otimes n}$ ($y=0$) is

$$c_0 = \frac{1}{N}\sum_x (-1)^{f(x)}.$$

- **$f$ constant**: $(-1)^{f(x)}$ is the same for every $x$, so $c_0 = \pm 1$ and $|c_0|^2 = 1$.
- **$f$ balanced**: $(-1)^{f(x)}$ takes $+1$ on half the inputs and $-1$ on the other half, so $c_0 = 0$.

Therefore

$$P(\text{measure } 0\cdots 0) = \begin{cases}1 & f\ \text{constant},\\ 0 & f\ \text{balanced}.\end{cases}$$

A single shot suffices.

## 5. Concrete phase oracles (the four used in the script)

| function | $f(x)$ | type | phase oracle |
|----------|--------|------|--------------|
| constant $0$ | $0$ | constant | $I$ |
| constant $1$ | $1$ | constant | $-I$ (e.g. $XZXZ$ on any wire) |
| first-bit | $x_0$ | balanced | $Z_0$ |
| parity | $x_0\oplus x_1\oplus\cdots\oplus x_{n-1}$ | balanced | $Z_0 Z_1 \cdots Z_{n-1}$ |

The $Z$-on-wire-$i$ implementation gives $Z_i|x\rangle = (-1)^{x_i}|x\rangle$, which composes additively in the exponent under tensor products — exactly $(-1)^{x_0\oplus\cdots\oplus x_{n-1}}$ for the parity case.

## 6. Query-count comparison

| $n$ | classical worst case | DJ quantum |
|-----|---------------------:|-----------:|
| 2 | 3 | 1 |
| 5 | 17 | 1 |
| 10 | 513 | 1 |
| 20 | 524{,}289 | 1 |

The advantage is exponential, but the problem is contrived (a *promise*). DJ is best understood as a proof of principle: quantum interference can encode global properties of $f$ in a single query.

## 7. What the script verifies

- For $n\in\{2,3,4,5\}$ and all four oracles, $P(0\cdots 0)$ is **exactly** $1.0$ for constant and $0.0$ for balanced.
- 1000-shot runs at $n=4$ give $1000/1000$ all-zero outcomes for constant and $0/1000$ for balanced.
