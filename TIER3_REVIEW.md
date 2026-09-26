# Tier 3 Review — Quantum Autoencoder for H₂ Ground States

Tier 3 (weeks 22–27) trained a 4-qubit quantum autoencoder on a 1-parameter
family of H₂/STO-3G ground states, evaluated it on a held-out split (which
here tests interpolation), swept depolarizing noise on `default.mixed`,
and compared head-to-head against two classical autoencoder baselines. Every weekly script is assertion-gated
(week 23 reports one missed gate instead of failing, see §1.1)
and the week-27 capstone re-runs weeks 23–26 at 5 seeds (3 for the noise
sweep) and re-checks the weekly pass gates, the week 23–25 fidelity gates
loosened by 2σ of the run. It does not compare against the weekly
numbers, but every run is seeded, and on the pinned requirements it
reproduces weeks 23–26 to the printed 4 decimals.

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
| held-out (interpolation) gap (recon, train − test, week 24) | **+0.05 pp** |
| init gradient norm $\|\nabla C\|_2$ (week 23) | 0.44 |
| final gradient norm $\|\nabla C\|_2$ (week 23) | 2.3 × 10⁻³ |

The 16-parameter, depth-4 RY+CNOT encoder (twice the 8 quantum weights
of the Tier 2 week 21 block) reaches local fidelity 1.0000 on seeds 0, 3
and 4 in week 23. Seed 2 stops at a stationary point at 0.9551
(gradient norm 0.0000 at the last epoch). Seed 1 (0.9576) is still on a
slow plateau at epoch 200 (gradient norm 0.010), the plateau near cost
0.04 that seed 0 left at about epoch 130 (`tier3/week23_loss_curves.png`).
Together they put the across-seed std at 0.0214, which misses the
pre-registered std < 0.02 gate; the script reports the miss rather than
hiding it (`tier3/week23_notes.md`). The held-out gate passes: in week
24, four seeds reconstruct the unseen bond lengths at 1.0000 and seed 2
at 0.9355. On this 2-D linear dataset that is interpolation, not
generalization: the 11 training states span the subspace that holds the
test states, and there both fidelities are fixed polynomials of the
state, so held-out fidelity follows from training fidelity for any
encoder. That includes $r = 2.5$, the one test bond length just outside
the training range (which ends at 2.4). Reconstruction fidelity in week
23 is 1.1 pp below local (trash) fidelity (0.9719 vs 0.9825). That
closeness is guaranteed, not found: for a pure input, $F_{\text{loc}}^2 \le F_{\text{recon}} \le
F_{\text{loc}}$ (`tier3/check_qae_bounds.py`), so high trash fidelity
forces high reconstruction fidelity, and the 5-seed means sit inside
that range ($0.9825^2 = 0.9653$).

### 1.2 The latent arc is a property of the data, not of training

Week 24 projected the encoded code-qubit reduced density matrices onto
their PCA axes across the 22 $r$ values. Spearman rank correlation
between $r$ and PC1 of $\rho_{\text{code}}$ came out at
$\rho_{\text{Spearman}} = -1.000$ (seed-0 model; the sign of a PCA axis
is arbitrary, so $|\rho| = 1.000$), with PC1 carrying 97.5 % of the
variance. The latent trajectory is a smooth 1-D arc with no fold-overs
(`tier3/week24_latent_trajectory.png`).

That is not evidence that the encoder discovered $r$. Every state is
$\cos t\,\ket{1100} + \sin t\,\ket{0011}$ with $t$ moving monotonically
in $r$, so any encoder maps the curve to a smooth arc of code states.
The week-24 control runs the same features, PCA and Spearman on 1000
untrained RY+CNOT encoders with seeded random angles: median
$|\rho| = 1.0000$, mean $0.9994 \pm 0.0081$, and 999 of 1000 pass the
pre-registered $|\rho| > 0.9$ gate. No encoder at all (the identity)
also gives $|\rho| = 1.000$. The monotone latent arc is a property of
the data, and the Spearman gate could essentially not fail.

### 1.3 Noise robustness was real but limited

Week 25, test local fidelity $P(\text{trash} = 00)$, mean ± std over 3
seeds for the trained QAE and over 20 seeded random (untrained) encoders,
the same 20 at every $p$:

| depolarizing rate $p$ | trained-test fidelity | random encoders (20) | Δ |
|---:|---:|---:|---:|
| 0.000 | 0.971 ± 0.021 | 0.2619 ± 0.1325 | **+71 pp** |
| 0.001 | 0.948 ± 0.018 | 0.2616 ± 0.1278 | **+69 pp** |
| 0.005 | 0.871 ± 0.009 | 0.2606 ± 0.1111 | **+61 pp** |
| 0.010 | 0.803 ± 0.011 | 0.2595 ± 0.0934 | **+54 pp** |
| 0.020 | 0.678 ± 0.010 | 0.2579 ± 0.0669 | **+42 pp** |

The $p$ values are sweep points, not calibrated to any device. At the
mid-sweep point $p = 5\times10^{-3}$ the trained QAE was 61 pp above the
mean of the random encoders, but the absolute fidelity has dropped from
0.97 to 0.87. That delta would compound in any
downstream computation that fed the decoded state into another circuit.

Interpreting the slope: the decline flattens as $p$ grows (about 23
fidelity per unit $p$ near $p = 0$, about 13 between $p = 0.01$ and
$0.02$). That is what exponential decay toward a floor looks like: as
the noise grows the trash register tends to the maximally mixed state,
so $P(\text{trash} = 00)$ decays toward $1/4$, not toward 0. With that
floor,
$F(p) = 1/4 + (F(0) - 1/4)(1 - p)^n$ gives, for each noisy point in the
table, $n = \ln[(F(p) - 1/4)/(F(0) - 1/4)] / \ln(1 - p)$ = 32, 30, 26
and 26 at $p$ = 0.001, 0.005, 0.01 and 0.02. So the curve is consistent
with a plain exponential in about 26 to 32 channels. The circuit inserts
40 depolarizing channels (in each of the 4 layers, one after each of the
4 RY gates and two after each of the 3 CNOTs), and the model is
retrained at each $p$, so this is a consistency check, not a model of
the mechanism.

## 2. Where barren plateaus showed up — and where they didn't

Tier 2 review item 4 said "move barren-plateau monitoring earlier in the
workflow." Tier 3 followed it: the gradient norm was logged inline at
epochs 0, 50, 100, 150 and 199 in week 23 (0 and 199 in weeks 24 and
27). Result:

| location | observation |
|------|------|
| init, $n=4$, $L=4$ | $\|\nabla C\|_2 = 0.44$ (5-seed mean, range 0.28 to 0.51), well above the plateau |
| training, week 23 | the 5-seed mean fell from 0.44 to $2.3 \times 10^{-3}$; seeds 2, 3 and 4 end below $10^{-3}$ because they converged (two at local fidelity 1.0000, one at a stationary point at 0.955), not because they stalled |
| under noise | not measured: the week 25 and week 27 noise runs do not log gradient norms |

In short: the choice of $n_{\text{qubits}} = 4$ and $L = 4$ from the
Tier 2 week-14 probe kept the QAE's initial gradients away from the
plateau. The **design discipline** that the Tier 2 review identified as
Tier 3's pre-requisite worked as intended, though not as predicted in
size: the measured init gradient norm (0.44) is below the ~1.3 estimate
in `tier3/week23_notes.md`, but above the 0.3 plateau gate. We never
had to react to a barren-plateau failure during training; we paid for
it once at the ansatz-selection step.

One caveat on the cost itself. The training cost
$C_G = 1 - P(\text{trash} = 00)$ projects onto both trash qubits at
once. Cerezo et al., *Cost function dependent barren plateaus in
shallow parametrized quantum circuits*, Nat. Commun. 12, 1791 (2021),
DOI 10.1038/s41467-021-21728-w, classify this QAE cost as global (the
kind that shows barren plateaus even for shallow circuits as qubits are
added) and contrast it with a local version that averages
$P(\text{trash bit } j = 0)$ over single trash qubits. With 2 trash
qubits the two bound each other, $C_L \le C_G \le 2\,C_L$ (the upper
bound is a union bound over the two trash bits), so here the choice
changes the cost by at most a factor of 2; it becomes a trainability
question only for larger trash registers. The scripts still call $P(\text{trash} = 00)$
"local fidelity"; the name is historical.

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
  ground-state manifold is a 2-D linear subspace by symmetry (the
  Hamiltonian conserves particle number and spin, and the bonding and
  antibonding orbitals have opposite parity, so at every $r$ only
  $\ket{1100}$ and $\ket{0011}$ are in the ground state's sector; a
  low-rank subspace approximation is the right classical tool). You
  can't beat an oracle with a model that doesn't know the geometry; you
  can only match it. At 256 parameters the linear AE essentially *is* the SVD.

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

- **At larger $n$ the QAE would have structural advantages (argued
  here, not measured; see §6).** The classical AE needs the input as a
  $\mathbb{C}^{16}$ amplitude vector, which on real hardware costs full
  state tomography — in general exponentially more measurements than the
  QAE needs (which consumes the state directly). The classical AE also produces an
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
   run a *trained* QAE on a current IBM device's free tier, with a
   SWAP-test ancilla for fidelity measurement. The
   noise model used here (uniform per-gate depolarizing) is the
   simplest plausible — real devices have correlated, non-Markovian,
   and gate-specific errors that this sweep doesn't capture.

2. **Test the QAE on data access, not on compression power.** A
   non-linear dataset would not help the QAE. Its encoder is a unitary
   followed by discarding the trash qubits, so over any set of training
   states its mean trash fidelity can never exceed the sum of the $2^k$
   largest eigenvalues of the states' average density matrix ($k$ code
   qubits, so 4 eigenvalues here). A linear map that projects onto the
   matching $2^k$ eigenvectors (PCA with a $2^k$-dimensional complex
   code) reaches exactly that sum, and reconstruction fidelity is at
   most trash fidelity (§1.1). So a dataset that is not low-rank in
   amplitude space would limit the QAE as much as the linear AE, and
   would favour a nonlinear classical AE. `tier3/check_qae_bounds.py`
   (not one of the week scripts) checks this on a curved 30-state family
   with 13 eigenvalues above $10^{-6}$: an encoder built from the top 4
   eigenvectors and the rank-4 projection both reach 0.6894, and no
   Haar-random or RY+CNOT encoder goes above it. The case for a QAE is that it acts on the quantum
   state without tomography (§3, §6). The next experiment should keep a
   low-rank ensemble and test, at larger $n$, a setting where a
   classical AE cannot read the amplitudes.

3. **Train on the reconstruction cost directly.** We trained on the
   Romero trash-fidelity cost because it's cheap and differentiable on
   `default.qubit`. With `default.mixed` available, the full
   reconstruction-fidelity cost (encode → trace → re-inject → decode →
   overlap) is computable but slow. A side-by-side comparison of the
   two cost functions on the same dataset would tighten the trash-
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

Three tiers, 27 weeks of sustained effort, one sentence of summary
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
predictions*, though not all of them could fail. In this review
"pre-registered" means fixed in the committed scripts at commit
71ca939, before the first full run recorded in the history (in which
week 23 stopped on its own std assert). `TIER3_PLAN.md` landed in the
same commit and lists only some of the gates: week 23 has 5 gates in
its script and 3 in the plan, and the plan's final gradient *variance*
gate is a gradient *norm* gate in the script. The week-25 noise
sweep had pre-registered pass criteria that could have failed (test
fidelity > 0.85 at $p = 0.005$ and monotone decay in $p$; the script
also requires more than 30 pp over random encoders at $p = 0.005$).
Two other checks could not: 999 of 1000 untrained encoders pass the
week-24 Spearman gate (§1.2), and week 26's pre-registered criterion
was only that the head-to-head table exists. Week 23's final-gradient
gate (5-seed mean norm > $10^{-3}$) is no evidence against a plateau
either: a converged run has near-zero gradient, and the gate passes
(mean 0.0023) because seeds 0 and 1 have not converged. It stays in
the script because it was fixed before the first run; the
barren-plateau evidence is the init-gradient gate (0.44 > 0.3).
The week-24 held-out split is weaker than it looks as well, since on
this dataset held-out fidelity follows from training fidelity (§1.1).
One pre-registered gate did fail: week 23's across-seed std
(0.0214 against < 0.02, §1.1). The script reports the miss instead of
moving the threshold.

## 6. Closing thought

Tier 2 ended with: *on the kinds of problems QML competes for today,
classical is harder to beat than the marketing suggests*. Tier 3 adds
one nuance, as an argument rather than a result: on quantum-native
tasks where the input is already a quantum state and the output feeds
another quantum operation, a classical autoencoder would need
tomography on the way in and state preparation on the way out, and for
large $n$ both costs grow exponentially in general. This study does
not test that. At 4 qubits everything, the QAE included, is simulated
classically at negligible cost, and on these states the 256-parameter
linear AE reconstructs exactly while the 136-parameter AE ties the QAE.
The QAE is the problem class in this curriculum where the question "is
there a sensible classical comparator at all" comes up; answering it
would need larger $n$ and quantum data that cannot be read out
classically.

The repo evolution closes cleanly:
- `tier1/` — 8 weeks of literacy
- `tier2/` — 13 weeks of core QML with honest cross-method comparisons
- `tier3/` — 6 weeks of one quantum-native artifact, end-to-end

What I would *not* do: pretend any of this is publishable. Tier 3 was
designed as a curriculum-finishing artifact, not a paper. It does
exactly what it claims to do, and no more. That's what the gates were
for.

— end of tier 3 review.
