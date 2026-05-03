# Week 25 — Noise Robustness

## What changed

- Device: `default.qubit` → `default.mixed` (density-matrix simulator).
- Cost: same Romero local cost, but now $P(\text{trash} = 00)$ is
  computed on a *mixed* state, not a pure one.
- Noise model: a single-qubit depolarizing channel
  $$
  \mathcal{D}_p(\rho) = (1 - p)\rho + \frac{p}{3}\big(X \rho X + Y \rho Y + Z \rho Z\big)
  $$
  applied after every parameterized gate, with $p \in \{0, 10^{-3}, 5\cdot10^{-3}, 10^{-2}, 2\cdot10^{-2}\}$. The 2-qubit CNOT contributes
  $\mathcal{D}_p$ on each of its two wires, mirroring how IBM
  characterizes 2-qubit gate errors.

## Why this is the right noise sweep

Per-gate $p \approx 10^{-3}$ matches what IBM publishes for current
superconducting qubits (median 1q error ~$3 \times 10^{-4}$, 2q error
~$10^{-2}$). Our `p = 0.005` sits in the realistic mid-range; `p = 0.02`
is near the brink-of-uselessness end. The sweep brackets actual
hardware regimes — the *shape* of the curve (how fidelity decays with
$p$) is the falsifiable prediction.

## Random-encoder baseline

We also evaluate a *random* (untrained) encoder at each $p$. Reason:
on a depolarizing channel of any nonzero $p$, *every* encoder loses
fidelity. The interesting question is whether a trained QAE has a real
"signal" advantage over an arbitrary one. Concretely:

$$
\Delta(p) = F_{\text{trained}}(p) - F_{\text{random}}(p).
$$

If $\Delta(p) \to 0$ at large $p$, training stops mattering. If
$\Delta(p)$ stays > 30 pp through $p = 0.005$, the QAE is still
extracting useful structure in the realistic-noise regime. The
assertion at the bottom requires the latter.

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
performance and make the noise tax look worse than it is.

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
