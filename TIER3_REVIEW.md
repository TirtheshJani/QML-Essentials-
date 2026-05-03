# Tier 3 Review — Quantum Autoencoder for H₂ Ground States

Tier 3 (weeks 22–27) trained a 4-qubit quantum autoencoder on a 1-parameter
family of H₂/STO-3G ground states, evaluated generalization, swept
depolarizing noise on `default.mixed`, and compared head-to-head against
two classical autoencoder baselines. Every weekly script is assertion-gated
and the week-27 capstone re-runs the whole pipeline at 5 seeds (3 for the
noise sweep) with a 2σ regression check on every headline number.

This review is the structural twin of `TIER2_REVIEW.md`: what the artifact
demonstrated, where barren plateaus did and didn't appear, the honest
quantum-vs-classical scoreboard, and what would change before any
publication-quality follow-up.

## 1. What the QAE actually demonstrated

### 1.1 Compression worked at the geometric minimum

The H₂ ground-state set lives in a 4-D linear subspace of $\mathbb{C}^{16}$
(week 22 SVD: top 4 singular values capture > 99.9 % of variance). A
2-qubit code is the *minimal* quantum bottleneck that can fit it
losslessly. The QAE saturated the achievable bound:

| metric | mean ± std (5 seeds) |
|------|------:|
| training local fidelity (full curve) | **0.987 ± 0.011** |
| training reconstruction fidelity | **0.972 ± 0.014** |
| held-out test reconstruction fidelity | **0.946 ± 0.018** |
| generalization gap (recon, train − test) | **+2.6 pp** |
| init gradient norm $\|\nabla C\|_2$ | 1.31 |
| final gradient norm $\|\nabla C\|_2$ | 4.6 × 10⁻³ |

The 16-parameter, depth-4 RY+CNOT encoder (matched parameter count to
*exactly* the quantum block from Tier 2 week 21) reaches near-perfect
local fidelity and stays well above the held-out gate. The local-cost-vs-
reconstruction-fidelity gap that the Romero paper warns about closed to
~1.5 pp at convergence — i.e. the local cost was a tight surrogate.

### 1.2 The latent code recovered the bond-length axis

Week 24 projected the encoded code-qubit reduced density matrices onto
their PCA axes across the 22 $r$ values. Spearman rank correlation
between $r$ and PC1 of $\rho_{\text{code}}$ came out at
$\rho_{\text{Spearman}} = +0.998$ (seed-0 model). The latent trajectory
is a smooth 1-D arc parameterized by the bond length, with no fold-overs
or discontinuities.

This is the part that's not just "the cost is low" — it's *interpretable
compression*. The encoder discovered that the only varying physical
parameter in the dataset is $r$, and used its 2-qubit code as a
1-parameter encoding of that axis (with one extra "noise" axis that
spreads orthogonally and carries no physical signal).

### 1.3 Noise robustness was real but limited

| depolarizing rate $p$ | trained-test fidelity | random-encoder | Δ |
|---:|---:|---:|---:|
| 0.000 | 0.95 ± 0.02 | 0.27 | **+68 pp** |
| 0.001 | 0.92 ± 0.02 | 0.27 | **+65 pp** |
| 0.005 | 0.87 ± 0.03 | 0.26 | **+61 pp** |
| 0.010 | 0.81 ± 0.04 | 0.26 | **+55 pp** |
| 0.020 | 0.71 ± 0.05 | 0.25 | **+46 pp** |

Per-gate $p \approx 5\times10^{-3}$ corresponds to current IBM 2-qubit
gate error rates. At that operating point the trained QAE was 61 pp
above a random encoder — not a small effect — but the absolute fidelity
has dropped from 0.95 to 0.87. That delta would compound in any
downstream computation that fed the decoded state into another circuit.

Interpreting the slope: the dominant trend is approximately linear in
$p$ across the trained range, not exponential, which is consistent with
the depolarizing channel's contribution to expectation values for
shallow circuits ($\langle O\rangle \to (1-p)^{n_g} \langle O\rangle$
with $n_g \approx 30$ gates would give a steeper curve; the milder
slope here reflects the local-cost cost function being dominated by
stochastic projection onto $\ket{0}^{\text{trash}}$ rather than the
full state).

## 2. Where barren plateaus showed up — and where they didn't

Tier 2 review item 4 said "move barren-plateau monitoring earlier in the
workflow." Tier 3 followed it: the gradient norm was logged inline at
epochs 0, 50, 100, 150, 199 of every training run. Result:

| location | observation |
|------|------|
| init, $n=4$, $L=4$ | $\|\nabla C\|_2 = 1.31$ — well above the plateau |
| training, all weeks | gradient norm decayed by ~3 orders of magnitude as the cost converged, but never crossed the $10^{-3}$ trainability floor |
| under noise, $p = 0.02$ | $\|\nabla C\|_2$ at init dropped to 0.78 — depolarizing noise *flattens* the cost landscape, contributing additively to the plateau effect |

In short: the choice of $n_{\text{qubits}} = 4$ and $L = 4$ from the
Tier 2 week-14 probe made the QAE trainable by construction. The
**design discipline** that the Tier 2 review identified as Tier 3's
pre-requisite worked exactly as predicted. We never had to react to a
barren-plateau failure during training; we paid for it once at the
ansatz-selection step.

The unexplored region is $n \ge 6$. At $n = 6$ qubits, the week-14
probe measured Var ≈ $3 \times 10^{-2}$ at init — still trainable, but
3× tighter than $n = 4$. A QAE with a 4-qubit code + 2-qubit trash
on $n = 6$ would be the next experiment; depending on the dataset's
effective dimension, it might also be unnecessary.

## 3. Quantum-vs-classical scoreboard

| model | params | test recon fidelity | notes |
|------|---:|---:|------|
| Linear classical AE | 256 | **0.999 ± 0.001** | oracle on a 4-D subspace |
| QAE | 16 | **0.946 ± 0.018** | the headline result |
| Matched nonlinear classical AE | 32 | **0.86 ± 0.04** | smallest sensible classical fight |

Three honest readings of this table:

- **The linear AE wins decisively, but it's an oracle.** The H₂
  ground-state manifold is a linear subspace by construction (the
  Hamiltonian is parameterized by $r$ and the ground state is an
  eigenvector of a continuously-varying matrix; a low-rank subspace
  approximation is the right classical tool). You can't beat an oracle
  with a model that doesn't know the geometry; you can only match it.
  At 256 parameters the linear AE essentially *is* the SVD.

- **The QAE outperforms the matched-parameter classical AE by
  ~9 pp.** This is real, but the comparison is structurally awkward —
  16 quantum parameters generate 16-D unitary transformations on a
  16-D Hilbert space, which has more *expressivity per parameter* than
  16 real linear weights on a 32-D real space. The "matched" classical
  baseline is fighting with one hand tied. We report it because not
  reporting it is dishonest, but the conclusion isn't "quantum wins
  at parameter count" — it's "parameter count isn't the right
  comparison axis."

- **The QAE has structural advantages the classical AE can't compete
  with.** The classical AE needs the input as a $\mathbb{C}^{16}$
  amplitude vector, which on real hardware costs full state tomography
  — exponentially more measurements than the QAE needs (which
  consumes the state directly). The classical AE also produces an
  amplitude-vector output that has to be re-prepared on a quantum
  device for any downstream computation — also exponential overhead.
  Neither cost shows up in our simulator-based comparison, but they
  are why a QAE makes sense as a building block at all.

This pattern matches Tier 2's: quantum methods compete only on
problems whose structure the encoding represents *and* where you stay
in the quantum domain end-to-end. The QAE on simulator state vectors
is a fair laboratory test; the QAE in a hardware quantum-data pipeline
is the realistic deployment.

## 4. What I would change before any follow-up

In rough order of expected impact:

1. **Move to real hardware.** Tier 2 review item 2 carried over to
   Tier 3 (we hit `default.mixed`, but not IBM). The next step is to
   run a *trained* QAE on actual `ibm_kyoto` or a current device's
   free tier, with a SWAP-test ancilla for fidelity measurement. The
   noise model used here (uniform per-gate depolarizing) is the
   simplest plausible — real devices have correlated, non-Markovian,
   and gate-specific errors that this sweep doesn't capture.

2. **Pick a non-linear dataset.** H₂ ground states are a 4-D linear
   subspace, which is exactly the regime where a linear classical AE
   wins by construction. The next experiment is a dataset that's
   geometrically linear in *some* state-space representation but not
   in amplitude space — e.g., random circuit states under a controlled
   structural prior, or eigenstates of a non-quadratic Hamiltonian
   like the transverse-field Ising model away from its critical
   point. There the linear-AE oracle goes away.

3. **Train on the reconstruction cost directly.** We trained on the
   Romero local cost because it's cheap and differentiable on
   `default.qubit`. With `default.mixed` available, the full
   reconstruction-fidelity cost (encode → trace → re-inject → decode →
   overlap) is computable but slow. A side-by-side comparison of the
   two cost functions on the same dataset would tighten the local-
   vs-recon-fidelity argument that Romero left implicit.

4. **Add a SWAP-test estimator.** All fidelities here are computed by
   exact state-vector simulation. On hardware they would be measured
   via SWAP test or destructive overlap test, which carries shot
   noise. A finite-shot version of the assertions would be a faithful
   step closer to the hardware regime.

5. **Compare against compressed-sensing classical baselines.** The
   linear AE is one classical compressor; LASSO-style sparse coding,
   tensor-train decomposition, and explicit PCA on the amplitude
   basis are others. Tier 3 reported one classical comparison;
   Tier 4 (if there is one) should report several.

6. **Graph-classification / discrete-structure datasets.** Tier 2's
   review pointed at graph data as a place where quantum encodings
   might compete on their own terms. We didn't go there in Tier 3.
   It remains the most promising untouched direction in the original
   roadmap.

## 5. Cross-tier reflection

Three tiers, ~21 weeks of sustained effort, one sentence of summary
each:

- **Tier 1**: read a quantum circuit fluently — gates, Bell, Grover,
  QFT, no hand-waving.
- **Tier 2**: train and *honestly evaluate* variational QML on real
  small datasets — VQE, QAOA, classifier, kernel — and write up where
  quantum lost.
- **Tier 3**: produce a single end-to-end quantum-native artifact (the
  H₂-ground-state QAE) with reproducibility, noise robustness, and
  matched classical baselines, and write up what the result means.

The biggest tier-over-tier delta in the writing is honesty under
pressure. Tier 1 was structured by Codebook progress; Tier 2 by a
plan with assertion gates; Tier 3 by *its own falsifiable
predictions*. The week-25 noise sweep, the week-24 Spearman test, and
the week-26 head-to-head all had pre-registered pass criteria that
could have failed (and would have, on a worse experiment). They
didn't, but the discipline is what keeps the result trustworthy if it
ever did.

## 6. Closing thought

Tier 2 ended with: *on the kinds of problems QML competes for today,
classical is harder to beat than the marketing suggests*. Tier 3
extends that with one nuance: **on quantum-native tasks where the
input is already a quantum state and the output needs to feed into
another quantum operation, classical models aren't competing at all
— they need exponential pre- and post-processing to even enter the
ring**. The QAE is the cleanest demonstration in this curriculum of a
problem class where the question isn't "can quantum beat classical"
but "is there a sensible classical comparator at all."

The repo evolution closes cleanly:
- `tier1/` — 8 weeks of literacy
- `tier2/` — 13 weeks of core QML with honest cross-method comparisons
- `tier3/` — 6 weeks of one quantum-native artifact, end-to-end

What I would *not* do: pretend any of this is publishable. Tier 3 was
designed as a curriculum-finishing artifact, not a paper. It does
exactly what it claims to do, and no more. That's what the gates were
for.

— end of tier 3 review.
