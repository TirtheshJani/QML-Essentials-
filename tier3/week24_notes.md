# Week 24 — Held-out Split (Interpolation) and Latent Geometry

## What changed from week 23

Week 23 trained on the full 22-state $r$-curve. Week 24 splits it
50 / 50: train on the 11 even-indexed $r$ values, evaluate on the 11
odd-indexed $r$ values that the model never saw. In general a split
like this asks whether a model does more than memorize its training set.

On this dataset it tests interpolation, not generalization. The 11
training states already span the 2-D subspace that holds the 11 test
states (on the same curve: 10 of them sit between training bond lengths,
and $r = 2.5$ is just outside the training range, which ends at 2.4),
and on that real
subspace both fidelities are fixed polynomials of the state (quadratic
for local fidelity, quartic for reconstruction fidelity). So for any
encoder the training fidelities determine the held-out ones, including
at $r = 2.5$, and a small gap is expected by construction.

## Held-out metric

Two numbers, each averaged over 5 seeds:

- **Local fidelity gap:** $C_{\text{train}} - C_{\text{test}}$, where $C$
  is the trash-zero probability. On this dataset the held-out
  fidelities are fixed by the training ones (see above), so the gap is
  expected to be small for any encoder; it checks the pipeline, not
  memorization.
- **Reconstruction fidelity gap:** same idea but with the full
  $F_{\text{recon}}$ from week 22.

**Pass criterion:** test recon fidelity > 0.85 *and* held-out gap
< 10 pp. Both are required.

## Latent geometry

The "latent code" of an input $\ket{\psi(r)}$ is the reduced density
matrix on the code wires after encoding:
$$
\rho_{\text{code}}(r) = \mathrm{Tr}_{\text{trash}}\big(V \ket{\psi(r)}\bra{\psi(r)} V^\dagger\big).
$$
A 4×4 Hermitian matrix has $4^2 = 16$ real degrees of freedom (with
trace = 1 reducing to 15). We flatten and concatenate
$(\mathrm{Re}\rho, \mathrm{Im}\rho)$ into a 32-D feature vector per state,
then run PCA across the 22 $r$ points.

The test committed with the scripts in 71ca939 was that **PC1 should
track $r$ monotonically** if the QAE learned that $r$ is the only varying axis,
measured with Spearman rank correlation:

$$
\rho_{\text{Spearman}}(r, \text{PC1}) \in [-1, 1].
$$

Pass threshold: $|\rho_{\text{Spearman}}| > 0.9$. The trained seed-0
encoder gives $-1.000$.

**The test cannot tell a trained encoder from an untrained one.** Every
state is $\cos t\,\ket{1100} + \sin t\,\ket{0011}$ with $t$ moving
monotonically in $r$, so for any encoder $\rho_{\text{code}}(r)$ is a
quadratic function of $(\cos t, \sin t)$ and the features trace a
smooth arc. The script's control (section 3b) runs the same features,
PCA and Spearman on 1000 untrained RY+CNOT encoders with seeded random
angles: median $|\rho| = 1.0000$, mean $0.9994 \pm 0.0081$, and 999 of
1000 pass the 0.9 gate. No encoder at all (the identity) also gives
$|\rho| = 1.0000$. The monotone latent arc is a property of the data,
not evidence that training discovered $r$.

## Why we keep the *encoder* unitary, not the decoder

The Romero protocol is a tied-weight autoencoder: the decoder is the
adjoint of the encoder. Storing one parameter array $\boldsymbol\alpha$
gives both. We use `tier3.utils.qae.make_encoder_unitary` to construct
the 16×16 unitary numerically (so `reconstruction_fidelity` doesn't
have to differentiate through a partial trace), and we use the trained
$V(\boldsymbol\alpha)$ for the latent-space probe.

## Failure modes, and what this week's gates can show

- **Overfitting** — large train/test gap. The QAE has 16 parameters and
  the dataset has 22 states in a 2-D subspace. This split cannot show
  overfitting: the training states span that subspace, so the held-out
  fidelities follow from the training ones (see above).
- **Trash-fidelity success but recon failure** (high $P(\text{trash}=00)$,
  low $F_{\text{recon}}$) cannot happen here. For a pure input,
  $F_{\text{loc}}^2 \le F_{\text{recon}} \le F_{\text{loc}}$
  (`tier3/check_qae_bounds.py`), so high trash fidelity forces high
  reconstruction fidelity.
- **Latent collapse** (a high-fidelity QAE that maps every $r$ to almost
  the same code state) is possible here, and the mean-fidelity gates
  cannot detect it. The training states' average density matrix has one
  eigenvalue of 0.955872, so an encoder that keeps only that eigenvector
  and gives every $r$ the same code state scores 0.9559 on the training
  states and 0.9540 on the test states, for both fidelities, and passes
  every fidelity gate this week (post-hoc check,
  `tier3/check_dominant_eigvec_baseline.py`). Seed 2 is a near-instance:
  it keeps only that eigenvector ($\langle v_2|P|v_2\rangle = 0.0000$
  for $P = U^\dagger (I_{\text{code}} \otimes \ket{00}\bra{00}) U$),
  its Spearman$(r, \text{PC1})$ is $+0.24$, and its test reconstruction
  (0.9355) is below that constant-code encoder's 0.9540. Only the
  Spearman gate could flag it, and it is applied to seed 0 only.

## Output

`tier3/week24_latent_trajectory.png` — a 2-D PCA projection of the
$\rho_{\text{code}}(r)$ trajectory, colored by $r$. A scrambled color
gradient would mean a broken pipeline; a smooth one is what untrained
encoders give too, so it is not evidence of a good encoding.
