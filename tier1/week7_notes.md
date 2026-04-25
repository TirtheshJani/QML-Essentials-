# Week 7 — Quantum Phase Estimation (skim)

**Codebook I.16.** Companion to `week7_phase_estimation.py`.

## 1. The problem

Given a unitary $U$ and an eigenvector $|\psi\rangle$ with eigenvalue $e^{2\pi i\varphi}$, estimate the phase $\varphi\in[0,1)$ to $t$ bits of precision.

Resources: a *counting register* of $t$ qubits, a register holding $|\psi\rangle$, and the ability to apply controlled-$U^{2^k}$ for $k=0,\ldots,t-1$.

## 2. Phase kickback

Controlling $U$ on $|\psi\rangle$:

$$\mathrm{C}\text{-}U\,(|c\rangle\otimes|\psi\rangle) = |c\rangle\otimes U^c|\psi\rangle = e^{2\pi i\,\varphi c}\,|c\rangle\otimes|\psi\rangle.$$

The phase moves from the target onto the control — *kickback*. The eigenstate is unchanged; the control picks up the phase.

## 3. The QPE circuit

1. Initialize the counting register in $|0\rangle^{\otimes t}$ and the target in $|\psi\rangle$.
2. Apply $H^{\otimes t}$ to the counting register: it becomes $\tfrac{1}{\sqrt{2^t}}\sum_{c=0}^{2^t-1}|c\rangle$.
3. Apply controlled-$U^{2^k}$ for $k=0,\ldots,t-1$ (qubit $k$ is the control). By kickback, the joint state becomes

$$\frac{1}{\sqrt{2^t}}\sum_{c=0}^{2^t-1} e^{2\pi i\,\varphi c}\,|c\rangle\otimes|\psi\rangle.$$

4. Apply $\mathrm{QFT}^{-1}$ on the counting register and measure.

## 4. The recovery formula

The state in step 3 is exactly $\mathrm{QFT}\,|2^t \varphi\rangle\otimes|\psi\rangle$ when $2^t\varphi\in\mathbb{Z}$. Then $\mathrm{QFT}^{-1}$ recovers the basis state and measurement returns $y = 2^t\varphi$ with probability $1$:

$$\widehat{\varphi} = \frac{y}{2^t} = \varphi \qquad (\text{exact, when } \varphi\ \text{is a }t\text{-bit dyadic}).$$

For arbitrary $\varphi$, write $2^t\varphi = a + \delta$ with $a\in\{0,\ldots,2^t-1\}$ and $\delta\in[0,1)$. The probability of measuring $y$ is

$$P(y) = \frac{1}{2^{2t}}\,\frac{\sin^2\big(\pi(2^t\varphi - y)\big)}{\sin^2\big(\pi(2^t\varphi - y)/2^t\big)}.$$

The closest integer $y^\star$ has

$$P(y^\star) \ge \frac{4}{\pi^2} \approx 0.405,$$

and the error is bounded:

$$\big|\widehat\varphi - \varphi\big| \le 2^{-t}.$$

So **each additional counting qubit halves the error** — exponential precision in $t$.

## 5. Worked example: a non-dyadic phase

Take $\varphi = 1/3$. Then $2^t\varphi$ is never an integer; the closest integer is at distance $\delta\in\{1/3, 2/3\}$ from $y^\star$. The script reports:

| $t$ | $y^\star$ | $\widehat\varphi$ | $\|\widehat\varphi - 1/3\|$ | $P(y^\star)$ | $2^{-t}$ |
|---:|---:|------:|------:|------:|------:|
| 3 | 3 | 0.3750 | 4.17e-2 | 0.6878 | 1.25e-1 |
| 4 | 5 | 0.3125 | 2.08e-2 | 0.6849 | 6.25e-2 |
| 5 | 11 | 0.3438 | 1.04e-2 | 0.6842 | 3.13e-2 |
| 6 | 21 | 0.3281 | 5.21e-3 | 0.6840 | 1.56e-2 |
| 7 | 43 | 0.3359 | 2.60e-3 | 0.6839 | 7.81e-3 |
| 8 | 85 | 0.3320 | 1.30e-3 | 0.6839 | 3.91e-3 |

Each row halves the error and tracks the $2^{-t}$ bound exactly. The peak probability stabilizes at $P(y^\star)\approx 0.684$, well above the worst-case bound $4/\pi^2$.

## 6. Why QPE matters

QPE is the substrate underneath:

- **Shor's algorithm** — period finding via QPE on the modular-multiplication unitary.
- **HHL** — solving linear systems by phase-estimating the matrix exponential $e^{iAt}$.
- **VQE / quantum chemistry** — energy estimation via QPE on $e^{-iHt}$ (or its variational analogues, where the phase becomes the eigenenergy).
- **Quantum counting** — Grover with QPE on the Grover iterator $G$ counts how many marked items exist.

## 7. What the script verifies

- Dyadic phases ($Z\to 1/2$, $S\to 1/4$, $T\to 1/8$) are recovered with $P(y^\star) = 1$ when $t$ suffices, and remain exact when $t$ exceeds the minimum.
- For $\varphi = 1/3$, the error halves per added bit ($t=3\to 8$), matching $2^{-t}$.
- Manual QPE agrees with `qml.QuantumPhaseEstimation` to $\sim 3\times 10^{-15}$.
