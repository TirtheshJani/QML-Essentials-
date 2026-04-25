# Week 5 — Grover's search and amplitude amplification

**Codebook I.13.** Companion to `week5_grover.py`.

## 1. The unstructured search problem

Search a database of $N=2^n$ items for an unknown subset of $M$ marked items, given black-box access via a phase oracle

$$O|x\rangle = \begin{cases}-|x\rangle & x\in \text{marked}\\ +|x\rangle & x\notin \text{marked}\end{cases}.$$

Classical: $\Theta(N/M)$ queries on average. Quantum (Grover): $O(\sqrt{N/M})$ queries — a **quadratic speedup**.

## 2. Two-dimensional reformulation

Define normalized "good" and "bad" superpositions:

$$|G\rangle = \tfrac{1}{\sqrt M}\sum_{x\in\text{marked}}|x\rangle, \qquad |B\rangle = \tfrac{1}{\sqrt{N-M}}\sum_{x\notin\text{marked}}|x\rangle.$$

These are orthogonal. The uniform superposition decomposes as

$$|s\rangle = H^{\otimes n}|0\rangle^{\otimes n} = \sin\theta\,|G\rangle + \cos\theta\,|B\rangle, \qquad \sin\theta = \sqrt{M/N}.$$

The whole algorithm lives in the 2D plane $\mathrm{span}(|G\rangle, |B\rangle)$.

## 3. The two reflections

The oracle is a reflection about the $|B\rangle$ axis:

$$O = I - 2|G\rangle\langle G|.$$

The **diffusion operator** is a reflection about $|s\rangle$:

$$D = 2|s\rangle\langle s| - I.$$

In circuit form, $D = H^{\otimes n}(2|0\rangle\langle 0| - I)H^{\otimes n}$, and $2|0\rangle\langle 0| - I = -X^{\otimes n}\,\mathrm{MCZ}\,X^{\otimes n}$ up to global phase.

## 4. The Grover iteration

$$G = D\cdot O.$$

Composition of two reflections is a rotation by twice the angle between the reflection axes — i.e. by $2\theta$ in the $(|G\rangle, |B\rangle)$ plane:

$$G^k|s\rangle = \sin\big((2k+1)\theta\big)\,|G\rangle + \cos\big((2k+1)\theta\big)\,|B\rangle.$$

Hence

$$P(\text{measure marked at step}\ k) = \sin^2\big((2k+1)\theta\big).$$

## 5. The optimal iteration count

Maximum at $(2k+1)\theta = \pi/2$, giving

$$k^\star = \left\lfloor \frac{\pi}{4\theta} - \tfrac12 \right\rfloor \approx \left\lfloor \frac{\pi}{4}\sqrt{\frac{N}{M}}\,\right\rfloor.$$

For $N=16, M=1$: $\theta = \arcsin(1/4) \approx 0.2527$, $k^\star = \lfloor\pi/4\cdot 4\rfloor = 3$, with $P\approx \sin^2(7\theta) \approx 0.961$.

For $N=16, M=4$: $\theta = \arcsin(1/2) = \pi/6$, $k^\star = 1$, with $P = \sin^2(3\cdot\pi/6) = \sin^2(\pi/2) = 1$ — exact.

## 6. Iteration table

| $n$ | $N$ | $M$ | classical avg ($\approx N/M$) | Grover $k^\star$ |
|----:|----:|----:|------------------------------:|-----------------:|
| 4 | 16 | 1 | 8.5 | 3 |
| 4 | 16 | 4 | 3.4 | 1 |
| 6 | 64 | 1 | 32.5 | 6 |
| 8 | 256 | 1 | 128.5 | 12 |
| 10 | 1024 | 1 | 512.5 | 25 |

## 7. Caveats

- Iterating *past* $k^\star$ over-rotates and the success probability falls (then revives at $3k^\star$, $5k^\star$, ...).
- If $M$ is unknown, run with random $k\in[0, K]$ for growing $K$. **Fixed-point Grover** algorithms remove the over-rotation issue at the cost of more iterations.
- $\sqrt N$ is *optimal* among query algorithms (BBBV lower bound, 1997). No quantum algorithm can do better than quadratic for unstructured search.

## 8. What the script verifies

- $P(\text{marked})$ traces the $\sin^2((2k+1)\theta)$ curve over $k=0\dots 6$ and peaks at $k=3$ with $0.9613$.
- For $M\in\{1,2,4,6\}$ within $N=16$, the predicted $k^\star$ matches the empirical first-cycle peak.
- $M=4$ achieves $P = 1.000000$ exactly at $k=1$.
