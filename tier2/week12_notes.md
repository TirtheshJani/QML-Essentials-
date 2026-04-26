# Week 12 — QAOA MaxCut: graph, cost Hamiltonian, brute force

**Tier 2 / 2B.1.** Companion to `week12_qaoa_maxcut_setup.py`.

## 1. The MaxCut problem

Given an undirected graph $G = (V, E)$, partition $V = S \sqcup \bar S$ to **maximize** the number of "cut" edges (edges with one endpoint in $S$ and one in $\bar S$). Decision-MaxCut is NP-complete; the best classical poly-time approximation guarantee is the Goemans–Williamson 0.878 bound (assuming the Unique Games Conjecture, this is optimal). A 6-node 3-regular instance with $|E| = 9$ has optimum 7 — at the small scale we're in, brute force is the cheapest baseline.

## 2. Encoding as an Ising Hamiltonian

Assign $z_v \in \{+1, -1\}$ to each vertex (sign = side of the cut). An edge $(i, j)$ contributes $1$ to the cut iff $z_i \neq z_j$, i.e. iff $\tfrac12(1 - z_i z_j) = 1$. Sum:

$$C(z) = \sum_{(i,j)\in E} \tfrac12 (1 - z_i z_j).$$

Promote each $z_v$ to a Pauli $Z_v$ acting on qubit $v$. Then $C \to \hat H_C$ is **diagonal** in the computational basis with eigenvalues equal to the cut counts:

$$\hat H_C = \sum_{(i,j)\in E} \tfrac12 (I - Z_i Z_j), \qquad \hat H_C |x\rangle = C(x)|x\rangle.$$

The script verifies this by sampling 6 basis states and matching `<x|H_C|x>` to the brute-force count.

## 3. The uniform-superposition baseline

The starting state of every QAOA run is

$$|s\rangle = H^{\otimes n}|0\rangle^{\otimes n} = \frac{1}{\sqrt{2^n}}\sum_x |x\rangle.$$

Because $\hat H_C$ is diagonal,

$$\langle s|\hat H_C|s\rangle = \frac{1}{2^n}\sum_x C(x) = \mathbb{E}_{x \sim U}[C(x)] = \frac{|E|}{2},$$

since each edge is cut with probability $1/2$ for a uniformly random vertex assignment. The script sees exactly $4.5 = 9/2$. This is the **random-guess** baseline that QAOA must beat.

## 4. The approximation ratio

$$\rho := \frac{\langle\psi|\hat H_C|\psi\rangle}{C^{*}}, \qquad C^{*} = \max_x C(x).$$

For our graph, $C^{*} = 7$ and the uniform baseline is $\rho = 4.5/7 \approx 0.643$. Goemans–Williamson guarantees $\rho \ge 0.878$ in expectation; QAOA at $p \to \infty$ converges to $\rho = 1$ exactly; week 13 measures finite-$p$ behavior.

## 5. Why this is the right toy

- Small enough to fully diagonalize ($N = 64$): every QAOA result has a ground-truth comparison.
- 3-regular: the standard QAOA-on-MaxCut benchmark, with known $p = 1$ analytical bounds (Farhi–Goldstone–Gutmann 2014).
- Symmetric ($Z_2$ flip $z \to -z$ leaves $C$ invariant): the optimum has even multiplicity. The script finds 6 optimal partitions, which by symmetry come in 3 complementary pairs.

## 6. What the script verifies

- The cost Hamiltonian has $2|E| = 18$ Pauli terms ($|E|$ identity halves and $|E|$ negative ZZ pairs).
- For 6 sample bitstrings, `<x|H_C|x>` equals the brute-force cut count.
- The uniform-superposition energy equals $|E|/2 = 4.5$ exactly.
- The largest eigenvalue of $\hat H_C$ equals the brute-force MaxCut value (7), so QAOA at the optimum can in principle saturate $\rho = 1$.
