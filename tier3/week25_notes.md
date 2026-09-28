# Week 25 — Noise Robustness

## What changed

- Device: `default.qubit` → `default.mixed` (density-matrix simulator).
- Cost: same Romero trash-fidelity cost, but now $P(\text{trash} = 00)$ is
  computed on a *mixed* state, not a pure one.
- Noise model: a single-qubit depolarizing channel
  $$
  \mathcal{D}_p(\rho) = (1 - p)\rho + \frac{p}{3}\big(X \rho X + Y \rho Y + Z \rho Z\big)
  $$
  (PennyLane's `DepolarizingChannel` convention; equivalently
  $(1 - 4p/3)\rho + (4p/3)\,I/2$, so the Bloch vector shrinks by
  $1 - 4p/3$) applied after every gate, with $p \in \{0, 10^{-3}, 5\cdot10^{-3}, 10^{-2}, 2\cdot10^{-2}\}$. The 2-qubit CNOT contributes
  $\mathcal{D}_p$ on each of its two wires.

## Why this noise sweep

The five values of $p$ are sweep points, not calibrated to any device:
they run from no noise through $10^{-3}$ to $2\cdot10^{-2}$, with
`p = 0.005` as the mid-sweep point where the pass gate sits. The *shape*
of the curve (how fidelity decays with $p$) is the falsifiable
prediction. Mapping $p$ onto a real device would need that device's
calibration data and a gate-specific noise model.

## Random-encoder baseline

We also evaluate 20 *random* (untrained) encoders at each $p$ (seeded,
the same 20 draws at every $p$) and report their mean ± std. Reason:
on a depolarizing channel of any nonzero $p$, *every* encoder loses
fidelity. The interesting question is whether a trained QAE has a real
"signal" advantage over an arbitrary one. Concretely:

$$
\Delta(p) = F_{\text{trained}}(p) - F_{\text{random}}(p).
$$

If $\Delta(p) \to 0$ at large $p$, training stops mattering. If
$\Delta(p)$ stays > 30 pp through $p = 0.005$, the QAE is still doing
better than an arbitrary encoder at the mid-sweep point. The
assertion at the bottom requires the latter.

Measured (pinned `requirements.txt`): at $p = 0.005$ the trained QAE's
test fidelity is 0.8707 ± 0.0090 (3 seeds) against 0.2606 ± 0.1111 for
the random encoders, $\Delta = +61$ pp; at $p = 0.02$, $\Delta = +42$ pp.

Post-hoc, this bar turned out to be low: the trained-vs-random gap is
mostly what any encoder that keeps the dominant eigenvector $v_1$ of the
training states gets (`tier3/check_dominant_eigvec_baseline.py`, written
in review round 4). Week-24 seed 2, which keeps only $v_1$, scores
0.8648 at $p = 0.005$ under this noise model without retraining, +60 pp
over the random encoders and 0.0058 below the trained mean, and passes
every gate in this script. Two of the three seeds trained at
$p = 0.005$ also keep only $v_1$ ($\langle v_2|P|v_2\rangle$ = 0.0034
and 0.0001 for $P = U^\dagger (I_{\text{code}} \otimes
\ket{00}\bra{00}) U$; seed 1 keeps both, 0.9904).

## How training under noise differs

Two practical changes from week 23:

1. **Slower per epoch.** `default.mixed` does $4^n$ density-matrix
   matrix-vector products vs $2^n$ on `default.qubit`. At $n = 4$ this is
   a ~4× wallclock hit per gradient step.
2. **Different optimum.** Under nonzero $p$, the *optimal* encoder is
   not the noiseless optimum. Training discovers a noise-aware code
   that's slightly different — not necessarily better at $p = 0$, but
   robust to the specific channel.

This is why we retrain at each $p$ rather than evaluating the noiseless
encoder under noise. The latter would underestimate trained
performance and make the noise tax look worse than it is. That was the
expectation; it was not measured here. The post-hoc check above
evaluated two noiseless-trained week-24 encoders at $p = 0.005$ without
retraining: seed 0 (keeps both directions) scores 0.8788 and seed 2
(keeps only $v_1$) 0.8648, against 0.8707 ± 0.0090 for the retrained
seeds.

## Failure modes this catches

- **Training fails entirely under noise.** If train fidelity at
  $p = 0.001$ already drops below 0.85, the gradient signal is being
  drowned out by the noise; we'd need either more samples or a
  better-conditioned cost function.
- **Non-monotone fidelity.** If $F(p)$ goes up at some $p > 0$ relative
  to $p = 0$, something is wrong (numerical or methodological — the
  fidelity should never *improve* with more noise on the same task).
- **No advantage over random.** $\Delta(p) < 5$ pp at any $p$ in the
  sweep would mean the encoder isn't doing anything useful in that
  noise regime.

## Output

`tier3/week25_noise_curve.png`: errorbars for trained-train,
trained-test, and random-encoder fidelity vs $p$, with a symlog $x$-axis
so the $p = 0$ point is visible alongside the log spacing of the noisy
points.
