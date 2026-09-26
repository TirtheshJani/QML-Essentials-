"""Classical autoencoder baselines for week 26.

Input: a length-32 real vector (real and imaginary parts of a 16-dim
complex amplitude vector, concatenated). Week 26 trains two baselines:

  - Linear AE: Linear(32 -> 4) + Linear(4 -> 32), no bias = 256 weights.
    Its 4-dim code is larger than the 2-D data subspace (week 22), so it
    can reconstruct the dataset exactly.
  - Small nonlinear AE: 32 -> 2 -> 2 -> 2 -> 32 with tanh, no bias
    = 32*2 + 2*2 + 2*2 + 2*32 = 136 weights, with a 2-dim code.

Neither is parameter-matched to the QAE's 16 trainable angles
(n_layers * n_qubits = 4 * 4). A classical AE that reads all 32 inputs
cannot get that small: 32 -> 1 -> 32 without bias already has 64 weights,
and a 1-dim code cannot represent the 2-D data subspace exactly.

The classical AE is trained in PyTorch with Adam on the MSE between the
input and the raw network output (no normalization during training).
After training,
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
    # float64: normalizing the float32 model output in single precision
    # leaves fidelities up to ~2e-7 above 1 for a perfect reconstruction.
    reals = np.asarray(reals, dtype=np.float64)
    half = reals.shape[1] // 2
    z = reals[:, :half] + 1j * reals[:, half:]
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return z / norms


class ClassicalAE(nn.Module):
    """Autoencoder without bias: linear, or with a tanh hidden layer of
    width `hidden_dim` on each side of the code when it is set."""

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
