# Week 8 — Tier 1 checkpoint synthesis

Companion to `week8_checkpoint_bell_grover.py`.

## 1. What "no-docs" means

The Tier 1 checkpoint asks: rebuild Bell-state preparation and Grover's algorithm in PennyLane *without referring back to weeks 2 or 5*. The criterion is operational — not literal recall, but the ability to derive the constructions from first principles fast enough to land them in one session.

## 2. Bell state construction (rederived)

Goal: produce the four Bell states

$$|\Phi^\pm\rangle = \tfrac{1}{\sqrt2}(|00\rangle \pm |11\rangle), \qquad |\Psi^\pm\rangle = \tfrac{1}{\sqrt2}(|01\rangle \pm |10\rangle).$$

Recipe — derive directly:

1. $|\Phi^+\rangle = \mathrm{CNOT}\,(H\otimes I)\,|00\rangle$, by computation: $H|0\rangle = |+\rangle$, then $\mathrm{CNOT}(|+\rangle|0\rangle) = \tfrac{1}{\sqrt2}(|00\rangle + |11\rangle)$.
2. $|\Psi^+\rangle = (X\otimes I)|\Phi^+\rangle$ — but easier: prepare $|10\rangle$ first instead of $|00\rangle$ and run the same circuit; the X kicks the input into the "Psi" sector, which swaps the two computational components.

Actually, applying $X$ on wire 1 (not 0) before $H \otimes I$ + CNOT gives $|\Psi^+\rangle$ (verify: $X_1|00\rangle = |01\rangle$, then $H_0|01\rangle = |+\rangle|1\rangle$, then $\mathrm{CNOT}|+\rangle|1\rangle = \tfrac{1}{\sqrt2}(|01\rangle + |10\rangle)$).

3. $|\Phi^-\rangle$ and $|\Psi^-\rangle$ pick up a relative minus by applying $Z$ on wire 0 *after* the CNOT. $Z_0$ flips the sign of the $|1\cdot\rangle$ component, exactly the desired $-$ on the $|11\rangle$ (or $|10\rangle$) branch.

So the single recipe is:

$$|\beta_{ab}\rangle = Z_0^a\,\mathrm{CNOT}_{0,1}\,(H_0)\,X_1^b\,|00\rangle, \qquad a,b\in\{0,1\}.$$

**Verification trick (Pauli fingerprints).** Each Bell state is uniquely identified by the triple $(\langle XX\rangle, \langle YY\rangle, \langle ZZ\rangle)$:

| state | $XX$ | $YY$ | $ZZ$ |
|---|---:|---:|---:|
| $\|\Phi^+\rangle$ | $+1$ | $-1$ | $+1$ |
| $\|\Phi^-\rangle$ | $-1$ | $+1$ | $+1$ |
| $\|\Psi^+\rangle$ | $+1$ | $+1$ | $-1$ |
| $\|\Psi^-\rangle$ | $-1$ | $-1$ | $-1$ |

The script asserts each row to $10^{-10}$.

## 3. Grover construction (rederived)

Goal for $N = 2^n$ items, $M$ marked:

1. **Initial state.** $|s\rangle = H^{\otimes n}|0\rangle^{\otimes n}$.
2. **Phase oracle.** For each marked bitstring $m$, surround a multi-controlled $Z$ with $X$ gates on the zero-positions of $m$. This applies a $-1$ phase exactly on $|m\rangle$.
3. **Diffusion.** $D = H^{\otimes n}(2|0\rangle\langle 0| - I)H^{\otimes n}$. Implementation: $H^{\otimes n}\,X^{\otimes n}\,\mathrm{MCZ}\,X^{\otimes n}\,H^{\otimes n}$, exploiting $2|0\rangle\langle 0| - I = -X^{\otimes n}\,\mathrm{MCZ}\,X^{\otimes n}$ up to global phase.
4. **Iterate.** Apply oracle + diffusion $k$ times.

The amplitude of the marked subspace after $k$ iterations is

$$\sin\!\big((2k+1)\theta\big), \qquad \sin\theta = \sqrt{M/N}.$$

So $P(\text{marked}) = \sin^2((2k+1)\theta)$, peaking at $k^\star \approx \lfloor\frac{\pi}{4}\sqrt{N/M}\rfloor$.

## 4. Numerical predictions tested

For $N=16$:

- **$M=1$:** $\sin\theta = 1/4$, so $\theta\approx 0.2527$. Predicted optimum $k^\star = \lfloor \pi/4 \cdot 4\rfloor = 3$, with $P = \sin^2(7\cdot 0.2527) = \sin^2(1.769) \approx 0.961$. **Empirical:** $0.9613$.
- **$M=4$:** $\sin\theta = 1/2$, so $\theta = \pi/6$. Predicted optimum $k^\star = 1$, with $P = \sin^2(3\cdot\pi/6) = \sin^2(\pi/2) = 1$ exactly. **Empirical:** $1.000000$.

The $M=4$ case is the cleanest demonstration that Grover is an *amplitude-amplification* algorithm: when the geometry aligns, one rotation lands you exactly on the target.

## 5. Checkpoint assertions (in the script)

```text
assert bell_pass                                  # all four Pauli fingerprints
assert peak_p > 0.96                              # Grover M=1 at k=3
assert abs(p_total - 1.0) < 1e-10                 # Grover M=4 at k=1
```

All three pass.

## 6. What Tier 1 actually delivered

Eight runnable scripts plus matching notes:

| week | topic | core math identity |
|------|-------|--------|
| 1 | single-qubit gates, Bloch sphere | $\vec r = (\langle X\rangle, \langle Y\rangle, \langle Z\rangle)$, $P(0\|R_Y(\theta)\|0\rangle) = \cos^2\tfrac\theta 2$ |
| 2 | Bell + GHZ | $\rho_A = \mathrm{Tr}_B|\Phi^+\rangle\langle\Phi^+| = I/2$ |
| 3 | measurement + teleportation | $\|\psi\rangle\|\Phi^+\rangle = \tfrac12\sum_{m_0,m_1}\|m_0 m_1\rangle\otimes Z^{m_0}X^{m_1}\|\psi\rangle$ |
| 4 | Deutsch–Jozsa | $c_0 = \tfrac1N\sum_x (-1)^{f(x)} \in \{0, \pm 1\}$ |
| 5 | Grover | $P(\text{marked}) = \sin^2((2k+1)\theta)$, $\sin\theta = \sqrt{M/N}$ |
| 6 | QFT | $\|x\rangle\mapsto \tfrac1{\sqrt N}\sum_y e^{2\pi i xy/N}\|y\rangle$ |
| 7 | QPE | $\widehat\varphi = y/2^t$, $\|\widehat\varphi - \varphi\| \le 2^{-t}$ |
| 8 | checkpoint | rebuild Bell + Grover from scratch |

Tier 1 outcome (per README): *read a quantum circuit fluently, predict measurement outcomes, explain entanglement and interference without hand-waving.* Achieved.

## 7. What's next

**Tier 2** moves to variational QML: VQE on $H_2$, QAOA on MaxCut, a variational classifier with PennyLane's `TorchLayer`, and a quantum kernel for SVM. The math turns from unitaries-and-measurements to *cost-function landscapes* — gradients, barren plateaus, encoding choices.
