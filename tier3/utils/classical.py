"""Parameter-matched classical autoencoder for week 26.

Train target: take a length-32 vector (real and imag parts of a 16-dim
complex amplitude vector concatenated), bottleneck to a 4-dim code,
reconstruct. Comparable to the QAE because the code dimension matches
(2-qubit code = 4 complex amplitudes ~ 8 real parameters of information).

Parameter budget (matched to a 4-layer hardware-efficient QAE):
  QAE trainable parameters = n_layers * n_qubits = 4 * 4 = 16

We hit the same total trainable parameter count by choosing the AE widths
to land on 16 trainable scalars too. The cleanest matched architecture
turns out to be:

  encoder: Linear(32 -> 4, no bias)    -> 32 * 4 = 128 weights, BUT we
  decoder: Linear(4 -> 32, no bias)    -> 4 * 32 = 128 weights

That's 256 parameters — vastly more than 16. The QAE will look unfairly
disadvantaged. So we do two comparisons in week 26:

  - "Tiny classical AE" with widths shrunk to land at ~16 trainable params
    (e.g. encoder Linear(32 -> 1) + decoder Linear(1 -> 32) = 64 params; the
    smallest matched architecture is bottleneck = 1, no bias, but that
    can't represent a 4-D subspace at all).
  - "Standard classical AE" at unconstrained capacity (256 params) —
    serves as the upper bound. Honest comparison reports both.

The classical AE is trained in PyTorch with Adam on MSE loss between
input and output of the L2-normalized network. After training,
reconstruction fidelity = `|<psi_recon | psi>|^2` where `psi_recon` is
the (re-normalized) reconstructed amplitude vector.
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


def states_to_real(states):
    """Stack `(k, 16)` complex states into a `(k, 32)` real array."""
    states = np.asarray(states)
    return np.concatenate([states.real, states.imag], axis=1).astype(np.float32)


def real_to_states(reals):
    """Inverse of `states_to_real`. Reconstructs `(k, 16)` complex array,
    L2-normalized so each row is a valid quantum state."""
    reals = np.asarray(reals)
    half = reals.shape[1] // 2
    z = reals[:, :half] + 1j * reals[:, half:]
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return z / norms


class ClassicalAE(nn.Module):
    """Linear autoencoder, no bias, optional hidden layer."""

    def __init__(self, input_dim=32, code_dim=4, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            self.encoder = nn.Linear(input_dim, code_dim, bias=False)
            self.decoder = nn.Linear(code_dim, input_dim, bias=False)
        else:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim, bias=False),
                nn.Tanh(),
                nn.Linear(hidden_dim, code_dim, bias=False),
            )
            self.decoder = nn.Sequential(
                nn.Linear(code_dim, hidden_dim, bias=False),
                nn.Tanh(),
                nn.Linear(hidden_dim, input_dim, bias=False),
            )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_classical_ae(
    train_states,
    code_dim=4,
    hidden_dim=None,
    n_epochs=200,
    lr=0.05,
    seed=0,
) -> Tuple[ClassicalAE, list]:
    torch.manual_seed(seed)
    X = torch.tensor(states_to_real(train_states))
    model = ClassicalAE(input_dim=X.shape[1], code_dim=code_dim,
                        hidden_dim=hidden_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for ep in range(n_epochs):
        opt.zero_grad()
        recon = model(X)
        loss = nn.functional.mse_loss(recon, X)
        loss.backward()
        opt.step()
        history.append((ep, float(loss)))
    return model, history


def reconstruction_fidelity_classical(model, states):
    """Mean and per-state pure-state fidelity after the classical AE.

    Reconstruction is L2-normalized to land on the unit sphere before
    fidelity is computed (otherwise a perfectly-zeroed-out output looks
    great by MSE but isn't a valid quantum state).
    """
    X = torch.tensor(states_to_real(states))
    with torch.no_grad():
        Y = model(X).numpy()
    recon_states = real_to_states(Y)
    fids = np.empty(len(states))
    for i, (psi, phi) in enumerate(zip(states, recon_states)):
        fids[i] = float(np.abs(np.vdot(psi, phi)) ** 2)
    return float(np.mean(fids)), fids
