# Week 26 — Classical Autoencoder Baseline

## The honest comparison

Tier 2 review item 5: pick a classical model and compare *fairly*. For
state-vector compression, the obvious classical baseline is a real-valued
autoencoder operating on the concatenated real and imaginary parts of the
amplitude vector:

$$
\ket{\psi} = \mathbf{a} + i\mathbf{b}, \quad \mathbf{a}, \mathbf{b} \in \mathbb{R}^{16}
\quad\longrightarrow\quad
\mathbf{x} = [\mathbf{a}; \mathbf{b}] \in \mathbb{R}^{32}.
$$

A classical AE then reconstructs $\hat{\mathbf{x}} \to \hat{\ket\psi}$
which we L2-normalize before computing fidelity.

## Two classical comparisons, neither perfect

**Linear AE (256 params)** — encoder
$\mathbf{x} \mapsto \mathbf{c} = W_e \mathbf{x}$,
decoder $\mathbf{c} \mapsto \hat{\mathbf{x}} = W_d \mathbf{c}$, with
$W_e \in \mathbb{R}^{4 \times 32}$, $W_d \in \mathbb{R}^{32 \times 4}$.

This is the classical *upper bound*. Because the H₂ ground-state set
is a 4-D subspace of $\mathbb{C}^{16}$ (week 22 SVD), a 4-D linear
bottleneck *can* losslessly reconstruct the entire dataset. Trained
to convergence, the linear AE will achieve ~1.0 fidelity. The QAE
beating this would imply the dataset is *not* 4-D linear, which we know
to be false.

**Param-matched nonlinear AE (~32 params)** — the fair comparison: a
$32 \to 2 \to 2 \to 2 \to 32$ AE with `tanh` non-linearities, no
biases, code dim = 2. It's still bigger than the QAE's 16 parameters
but it's the smallest sensible classical architecture that touches
the full 32-D input space and uses a 2-D code matching the QAE's code
qubits.

We can't get the classical AE down to 16 parameters and still touch
all 32 input axes — a $32 \to 1$ encoder has 32 parameters by itself.
This **structural asymmetry** is itself one of the points of this
week: the parameter-counting argument doesn't translate cleanly between
quantum and classical models. The QAE's 16 *complex-amplitude*
parameters generate a Lie group transformation on a 16-D Hilbert space,
which is genuinely more expressive per parameter than 16 real linear
weights.

## Expected ranking

From the dataset's geometric structure:

1. Linear AE → ~1.0 (oracle on a linear subspace)
2. QAE → 0.95 ± few pp (limited by the local-cost surrogate gap)
3. Matched nonlinear AE → 0.85 ± noise (small, has to learn)

This is not a story of quantum advantage on this specific dataset. The
H₂ ground-state manifold is *too easy* for the classical baseline at
unconstrained capacity. The QAE's interest comes from elsewhere:

- it doesn't *need* the dataset to be linear in amplitude space (week
  27 capstone could swap the dataset for non-linear families);
- it operates on the actual quantum state, which on real hardware
  cannot be exposed as an amplitude vector to a classical AE without
  expensive tomography;
- under noise (week 25), the comparison structure changes — the
  classical AE never sees the encoded-then-noisy state, so it doesn't
  pay the noise tax the QAE does, but it also can't act on a quantum
  output that one would want to feed downstream into another quantum
  algorithm.

## What the script reports

- 5-seed mean ± std for each of three models on test reconstruction
  fidelity.
- QAE − matched and QAE − linear deltas in pp.
- A note that the linear AE is an oracle, not a fair fight.

## Pass criterion

QAE test recon > 0.85 *and* linear AE > 0.95 (the geometric upper
bound). No assertion that the QAE *beats* the classical: the
statement of week 26 is that we have an honest comparison, not a
quantum win.
