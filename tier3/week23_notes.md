# Week 23 — Training the Quantum Autoencoder

## Setup

- **Encoder ansatz:** hardware-efficient,
  $V(\boldsymbol\alpha) = \prod_{L=1}^{4} \big(\bigotimes_{w=0}^{3} R_Y(\alpha_{Lw})\big)\, U_{\text{CNOT-ladder}}$.
  16 trainable parameters total.
- **Wire layout:** code = [0, 1], trash = [2, 3] (week 22 convention).
- **Cost:** Romero "local" trash-fidelity loss
  $$
  C(\boldsymbol\alpha) = 1 - \frac{1}{N} \sum_{i=1}^{N} P_i(\text{trash} = 00 \mid \boldsymbol\alpha).
  $$
- **Optimizer:** PennyLane's `AdamOptimizer`, `lr = 0.05`, 200 epochs.
- **Seeds:** 0..4 (Tier 2 review item 6 — every headline gets mean ± std).

## Why this ansatz, and why depth 4

Two constraints, one design:

1. The dataset's effective dim is 4 (week 22). To re-express each state in
   a (code, $\ket{0}_{\text{trash}}$) factored form, the encoder needs at
   minimum $\log_2 4 = 2$ layers' worth of expressivity — but this is a
   lower bound, not a tight one. Depth 4 gives ~2× headroom over the
   minimum.
2. The Tier 2 week 14 / week 21 barren-plateau probe measured $\mathrm{Var}[\partial_{\alpha} \langle Z_0 Z_1\rangle] \approx 9.9 \times 10^{-2}$
   for $n = 4$ qubits, depth $L = n$ on the same RY+CNOT ansatz. Going
   deeper than $L = n$ starts to flatten the landscape; staying at $L = n$
   keeps initial gradient magnitude $|\nabla C|_2 \approx \sqrt{16 \times
   10^{-1}} \approx 1.3$, which is comfortably trainable.

So depth 4 is the sweet spot: expressive enough for a 4-D dataset, not
yet stuck on the plateau.

## Why local cost (not reconstruction cost)

Reconstruction fidelity requires building $\rho_{\text{out}} = U^\dagger
(\rho_{\text{code}} \otimes \ket{0}\bra{0}) U$ and computing
$\langle\psi|\rho_{\text{out}}|\psi\rangle$. PennyLane can't differentiate
through the partial trace + density-matrix construction natively, so this
would require either density-matrix simulation (`default.mixed`) or a
SWAP-test ancilla. Both work but cost more — and Romero (2017) §III shows
the local cost is a tight surrogate in the high-fidelity regime.

We **train on local cost** and **report reconstruction fidelity**
numerically (via `tier3.utils.states.reconstruction_fidelity`) on the
finished encoder. If they disagree by more than a couple of percentage
points, that signals the local cost is loose — a useful diagnostic, and
the assert at the bottom catches it.

## Inline barren-plateau monitoring

Tier 2 review item 4 said "move barren-plateau monitoring earlier." Here
we log $\|\nabla C\|_2$ at epochs $\{0, 50, 100, 150, 199\}$ during
training, not after the fact. The check is cheap (one extra `qml.grad`
call per logged epoch) and catches the failure mode where the optimizer
would otherwise drag itself onto the plateau and quietly stop moving.

Empirically: $\|\nabla C\|_2$ at epoch 0 is ~1.3 (plateau-free); at the
end of training it drops by 2–3 orders of magnitude as the model
converges, but never crosses the $10^{-3}$ threshold. The assertion at
the bottom requires that it stays above $10^{-3}$ — we are training
*toward* a low-cost minimum, not falling into a flat region of cost.

## Reproducibility

5 seeds × 200 epochs × 22 states × ~3 ms/forward ≈ 7 minutes wall-clock
on `default.qubit`. Writeup target is `mean fidelity ± std`, with the
hard pass criteria:

- mean local fidelity > 0.95
- std across seeds < 0.02
- mean reconstruction fidelity > 0.93
- final gradient norm > $10^{-3}$ (still moving)
- initial gradient norm > 0.3 (not on the plateau at init)

All five are asserted at the bottom of the script. Five-of-five required
to pass.

## What the next week needs from this

A *trained* QAE — specifically, the parameter array $\boldsymbol\alpha^*$
from the seed-0 run. Week 24 splits the dataset (train on every other
$r$, test on the remaining 11 $r$ values) to test generalization, and
adds the latent-trajectory visualization on the code qubits.
