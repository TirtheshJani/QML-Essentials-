# Week 13 — p-layer QAOA on MaxCut

**Tier 2 / 2B.2.** Companion to `week13_qaoa_optimization.py`.

## 1. The QAOA ansatz

For a problem Hamiltonian $\hat H_C$ (week 12) and the standard X-mixer

$$\hat H_M = \sum_{v} X_v,$$

the depth-$p$ QAOA state is

$$|\psi_p(\boldsymbol\gamma, \boldsymbol\beta)\rangle = \prod_{l=1}^{p} e^{-i\beta_l \hat H_M}\,e^{-i\gamma_l \hat H_C}\;|s\rangle, \qquad |s\rangle = H^{\otimes n}|0\rangle^{\otimes n}.$$

Two angles per layer; $2p$ total parameters. The classical outer loop maximizes $\langle\psi_p|\hat H_C|\psi_p\rangle$ over $(\boldsymbol\gamma, \boldsymbol\beta)$.

## 2. Decomposing the layers

**Cost layer.** Each summand of $\hat H_C$ is $\tfrac12(I - Z_iZ_j)$ on its edge. The identity contributes a global phase. The $Z_iZ_j$ piece is implemented exactly by

$$e^{+i\frac{\gamma}{2}Z_iZ_j} = \mathrm{CNOT}_{ij}\;R_Z(-\gamma)_j\;\mathrm{CNOT}_{ij},$$

since $R_Z(\theta) = e^{-i\theta Z/2}$ and the CNOT sandwich rotates $Z_j \to Z_iZ_j$ in the conjugating frame. All edge terms commute (all-Z), so the layer is **exact** with no Trotter error.

**Mixer layer.** $\hat H_M$ is a sum of single-qubit $X_v$, all commuting:

$$e^{-i\beta\hat H_M} = \prod_v e^{-i\beta X_v} = \prod_v R_X(2\beta).$$

## 3. The optimization landscape

QAOA cost surfaces in $(\boldsymbol\gamma, \boldsymbol\beta)$ have three properties that make naive gradient descent fragile:

1. **Many local minima.** The expectation is a trigonometric polynomial of degree $\le 2p$ in each angle.
2. **Symmetries that aren't degeneracies.** $\gamma \to \gamma + \pi$ on every edge is a global phase only when all weights are equal; in general it changes phases across edges and *does* matter.
3. **Periodicity.** $\gamma \in [0, \pi)$, $\beta \in [0, \pi/2)$ already covers a fundamental domain.

The script does **10 random restarts** per $p$ inside that domain and keeps the best — cheap insurance against local minima at this scale.

## 4. Results on the 6-node 3-regular graph

| $p$ | $\langle H_C\rangle$ | $\rho = \langle H_C\rangle / C^*$ |
|----:|--------------------:|----------------------------------:|
| 1 | 5.94 | 0.85 |
| 2 | 6.60 | 0.94 |
| 3 | 6.88 | 0.98 |

The Farhi–Goldstone–Gutmann (2014) **worst-case** bound for $p = 1$ on 3-regular graphs is $\rho \ge 0.6924$; the empirical 0.85 here is well above that — small graphs leave a lot of slack. Each additional layer roughly halves the residual gap to 1. At $p \to \infty$ QAOA is provably optimal (it can simulate adiabatic evolution).

## 5. Reading the output state

At the best $p = 3$ angles, the measurement distribution puts **91.6 %** of its mass on bitstrings achieving $C(x) = 7$ — the six optimal partitions, each with probability $\approx 0.154 = 0.92/6$. The exact $1/6$ split reflects the $\mathbb Z_2$ flip symmetry of MaxCut and the permutation symmetry of the graph; the optimizer exploits no asymmetry it cannot see.

## 6. What the script verifies

- Approximation ratio at $p = 1$ exceeds the FGG'14 worst-case bound.
- Ratio is monotone non-decreasing in $p$.
- $p = 3$ ratio $\ge 0.85$ (the threshold quoted in the tier 2 plan).
- The final $p = 3$ output state places more than half its probability mass on bitstrings that achieve the brute-force MaxCut optimum.

## 7. Foreshadowing barren plateaus (week 14)

At $n = 6$ the landscape is friendly: 10 random restarts always find a $\rho > 0.85$ basin. As $n$ grows, the gradient variance at random initialization decays as $\Theta(2^{-n})$ (McClean et al. 2018 for hardware-efficient ansätze; QAOA's exponential decay was confirmed by Wang et al. 2021). Week 14 measures this decay directly on the same circuit family — that is the point at which "watch optimization stall" becomes the lesson.
