# Tier 3 Review — Quantum Autoencoder for H₂ Ground States

Tier 3 (weeks 22–27) trained a 4-qubit quantum autoencoder on a 1-parameter
family of H₂/STO-3G ground states, evaluated generalization, swept
depolarizing noise on `default.mixed`, and compared head-to-head against
two classical autoencoder baselines. Every weekly script is assertion-gated
(week 23 reports one missed gate instead of failing, see §1.1)
and the week-27 capstone re-runs the whole pipeline at 5 seeds (3 for the
noise sweep) with a 2σ regression check on every headline number.

This review is the structural twin of `TIER2_REVIEW.md`: what the artifact
demonstrated, where barren plateaus did and didn't appear, the honest
quantum-vs-classical scoreboard, and what would change before any
publication-quality follow-up.

## 1. What the QAE actually demonstrated

### 1.1 Compression worked, with room to spare

The H₂ ground-state set lives in a 2-D linear subspace of $\mathbb{C}^{16}$
(week 22 SVD: singular values 4.584 and 0.993, the rest below $10^{-15}$;
every state is a real combination of $\ket{1100}$ and $\ket{0011}$). A
2-qubit code (4-D) fits it losslessly with room to spare; strictly, one
code qubit would already be enough. Numbers below are what the scripts
print with the pinned `requirements.txt`:

| metric | mean ± std (5 seeds) |
|------|------:|
| training local fidelity (full curve, week 23) | **0.9825 ± 0.0214** |
| training reconstruction fidelity (full curve, week 23) | **0.9719 ± 0.0349** |
| held-out test reconstruction fidelity (week 24) | **0.9871 ± 0.0258** |
| generalization gap (recon, train − test, week 24) | **+0.05 pp** |
| init gradient norm $\|\nabla C\|_2$ (week 23) | 0.44 |
| final gradient norm $\|\nabla C\|_2$ (week 23) | 2.3 × 10⁻³ |

The 16-parameter, depth-4 RY+CNOT encoder (twice the 8 quantum weights
of the Tier 2 week 21 block) reaches local fidelity 1.0000 on seeds 0, 3
and 4 in week 23. Seeds 1 and 2 stop in a local minimum (0.9576 and
0.9551), which puts the across-seed std at 0.0214 and misses the
pre-registered std < 0.02 gate; the script reports the miss rather than
hiding it (`tier3/week23_notes.md`). The held-out gate passes: in week
24, four seeds reconstruct the unseen bond lengths at 1.0000 and seed 2
at 0.9355. The mean gap between local cost and reconstruction fidelity
in week 23 is 1.1 pp (0.9825 vs 0.9719), so the local cost was a tight
surrogate.

### 1.2 The latent code recovered the bond-length axis

Week 24 projected the encoded code-qubit reduced density matrices onto
their PCA axes across the 22 $r$ values. Spearman rank correlation
between $r$ and PC1 of $\rho_{\text{code}}$ came out at
$\rho_{\text{Spearman}} = -1.000$ (seed-0 model; the sign of a PCA axis
is arbitrary, so $|\rho| = 1.000$ is the number that matters). The latent
trajectory is a smooth 1-D arc parameterized by the bond length, with no
fold-overs or discontinuities (`tier3/week24_latent_trajectory.png`).

This is the part that's not just "the cost is low" — it's *interpretable
compression*. The encoder discovered that the only varying physical
parameter in the dataset is $r$, and used its 2-qubit code as a
1-parameter encoding of that axis (PC1 carries 97.5 % of the variance;
PC2 is the arc's curvature, itself a smooth function of $r$).

### 1.3 Noise robustness was real but limited

Week 25, test local fidelity $P(\text{trash} = 00)$, mean ± std over 3
seeds:

| depolarizing rate $p$ | trained-test fidelity | random-encoder | Δ |
|---:|---:|---:|---:|
| 0.000 | 0.971 ± 0.021 | 0.182 | **+79 pp** |
| 0.001 | 0.948 ± 0.018 | 0.185 | **+76 pp** |
| 0.005 | 0.871 ± 0.009 | 0.195 | **+68 pp** |
| 0.010 | 0.803 ± 0.011 | 0.207 | **+60 pp** |
| 0.020 | 0.678 ± 0.010 | 0.224 | **+45 pp** |

Per-gate $p \approx 5\times10^{-3}$ corresponds to current IBM 2-qubit
gate error rates. At that operating point the trained QAE was 68 pp
above a random encoder — not a small effect — but the absolute fidelity
has dropped from 0.97 to 0.87. That delta would compound in any
downstream computation that fed the decoded state into another circuit.

Interpreting the slope: the decline flattens as $p$ grows (about 23
fidelity per unit $p$ near $p = 0$, about 13 between $p = 0.01$ and
$0.02$) and stays milder than a naive exponential, which is consistent with
the depolarizing channel's contribution to expectation values for
shallow circuits ($\langle O\rangle \to (1-p)^{n_g} \langle O\rangle$
with $n_g \approx 30$ gates would give a steeper curve; the milder
slope here reflects the local-cost cost function being dominated by
stochastic projection onto $\ket{0}^{\text{trash}}$ rather than the
full state).

## 2. Where barren plateaus showed up — and where they didn't

Tier 2 review item 4 said "move barren-plateau monitoring earlier in the
workflow." Tier 3 followed it: the gradient norm was logged inline at
epochs 0, 50, 100, 150 and 199 in week 23 (0 and 199 in weeks 24 and
27). Result:

| location | observation |
|------|------|
| init, $n=4$, $L=4$ | $\|\nabla C\|_2 = 0.44$ (5-seed mean, range 0.28 to 0.51), well above the plateau |
| training, week 23 | the 5-seed mean fell from 0.44 to $2.3 \times 10^{-3}$; seeds 2, 3 and 4 end below $10^{-3}$ because they converged (two at local fidelity 1.0000, one in the 0.955 local minimum), not because they stalled |
| under noise | not measured: the week 25 and week 27 noise runs do not log gradient norms |

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

Week 26, same train/test split, 5 seeds:

| model | params | test recon fidelity | notes |
|------|---:|---:|------|
| Linear classical AE | 256 | **1.0000 ± 0.0000** | oracle on a 2-D subspace |
| QAE | 16 | **0.9871 ± 0.0258** | the headline result |
| Small nonlinear classical AE | 136 | **0.9811 ± 0.0211** | closer small-model comparison, not parameter-matched |

Three honest readings of this table:

- **The linear AE wins decisively, but it's an oracle.** The H₂
  ground-state manifold is a linear subspace by construction (the
  Hamiltonian is parameterized by $r$ and the ground state is an
  eigenvector of a continuously-varying matrix; a low-rank subspace
  approximation is the right classical tool). You can't beat an oracle
  with a model that doesn't know the geometry; you can only match it.
  At 256 parameters the linear AE essentially *is* the SVD.

- **The QAE and the small nonlinear classical AE tie.** The QAE is
  ahead by 0.6 pp (0.9871 vs 0.9811), well inside one standard
  deviation, and both have seeds that stop lower (for the QAE, seed 2
  at 0.9355 in week 24). The comparison is also not
  parameter-matched: the classical AE has 136 weights to the QAE's 16,
  and no classical AE that reads all 32 input numbers can get down to
  16 (a $32 \to 1 \to 32$ AE already has 64). The only
  parameter-matched comparison in this curriculum is tier 2 week 17
  (13 vs 13 parameters, a classifier). We report it because not
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

2. **Pick a non-linear dataset.** H₂ ground states are a 2-D linear
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
  classical autoencoder baselines (not parameter-matched), and write up
  what the result means.

The biggest tier-over-tier delta in the writing is honesty under
pressure. Tier 1 was structured by Codebook progress; Tier 2 by a
plan with assertion gates; Tier 3 by *its own falsifiable
predictions*. The week-25 noise sweep, the week-24 Spearman test, and
the week-26 head-to-head all had pre-registered pass criteria that
could have failed (and would have, on a worse experiment). They
didn't, but the discipline is what keeps the result trustworthy if it
ever did. One pre-registered gate did fail: week 23's across-seed std
(0.0214 against < 0.02, §1.1). The script reports the miss instead of
moving the threshold.

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
