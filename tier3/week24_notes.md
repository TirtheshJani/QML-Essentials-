# Week 24 — Generalization and Latent Geometry

## What changed from week 23

Week 23 trained on the full 22-state $r$-curve. Week 24 splits it
50 / 50: train on the 11 even-indexed $r$ values, evaluate on the 11
odd-indexed $r$ values that the model never saw. This is the smallest
possible "is the QAE doing more than memorizing the training set?" test.

## Generalization metric

Two numbers, each averaged over 5 seeds:

- **Local fidelity gap:** $C_{\text{train}} - C_{\text{test}}$, where $C$
  is the trash-zero probability. If the QAE memorizes individual
  training states, this gap will be large; if it discovers a generic
  encoding for the H₂ ground-state manifold, the gap will be small.
- **Reconstruction fidelity gap:** same idea but with the full
  $F_{\text{recon}}$ from week 22.

**Pass criterion:** test recon fidelity > 0.85 *and* generalization gap
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

If the QAE learned that $r$ is the only varying axis, **PC1 should track
$r$ monotonically.** We test this with Spearman rank correlation:

$$
\rho_{\text{Spearman}}(r, \text{PC1}) \in [-1, 1].
$$

A value near $\pm 1$ means the QAE used its 2-qubit code as a
1-D parameterization of the bond-length axis (with one extra degree of
freedom representing where the input states actually differ within the
4-D subspace). That's not just "low loss"; that's *interpretable
compression*.

Pass threshold: $|\rho_{\text{Spearman}}| > 0.9$.

## Why we keep the *encoder* unitary, not the decoder

The Romero protocol is a tied-weight autoencoder: the decoder is the
adjoint of the encoder. Storing one parameter array $\boldsymbol\alpha$
gives both. We use `tier3.utils.qae.make_encoder_unitary` to construct
the 16×16 unitary numerically (so `reconstruction_fidelity` doesn't
have to differentiate through a partial trace), and we use the trained
$V(\boldsymbol\alpha)$ for the latent-space probe.

## Failure modes this week catches

- **Overfitting** — large train/test gap. The QAE has 16 parameters;
  the dataset has 22 states living in a 4-D subspace, so memorization
  is geometrically possible. The held-out evaluation rules it out (or
  confirms it, depending on how training goes).
- **Local-cost-success but recon-failure** — high $P(\text{trash}=00)$
  but low $F_{\text{recon}}$. This happens when the encoder learns to
  zero out the trash *for one of the Schmidt sectors only*, leaving
  the other sectors poorly encoded. The reconstruction fidelity check
  catches it; the local cost alone wouldn't.
- **Latent collapse** — a high-fidelity QAE that maps every $r$ to
  almost the same code state. Spearman correlation < 0.5 catches this.

## Output

`tier3/week24_latent_trajectory.png` — a 2-D PCA projection of the
$\rho_{\text{code}}(r)$ trajectory, colored by $r$. A smooth color
gradient along the curve means we have a good encoding; a scrambled
gradient means we don't.
