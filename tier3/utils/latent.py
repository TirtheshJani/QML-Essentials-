"""Latent-space probe shared by weeks 24 and 27, plus its random-encoder control.

The probe: encode every state, take the code-qubit reduced density matrix,
flatten its real and imaginary parts into a feature vector, run PCA across
the states and report Spearman(r, PC1).

The control runs the same probe on untrained encoders (same RY+CNOT ansatz,
angles drawn uniformly in [0, 2*pi) like the training initialization). If
untrained encoders give the same |Spearman|, the statistic measures the data,
not what training learned.
"""

import numpy as np
from scipy.stats import spearmanr

from tier3.utils.states import DIM_CODE, DIM_TRASH, N_QUBITS
from tier3.utils.qae import make_encoder_unitary


def code_bloch_vectors(states, encoder_unitary):
    """Real and imaginary parts of each state's encoded code-qubit reduced
    density matrix, flattened. Returns an (n, 32) array suitable for PCA."""
    feats = []
    for psi in states:
        enc = encoder_unitary @ psi
        T = enc.reshape(DIM_CODE, DIM_TRASH)
        rho_code = T @ np.conj(T.T)
        feats.append(rho_code.flatten())
    feats = np.array(feats)
    return np.concatenate([feats.real, feats.imag], axis=1)


def latent_pca(states, encoder_unitary):
    """PCA via SVD on the centered features. Returns (pcs, singular_values)."""
    feats = code_bloch_vectors(states, encoder_unitary)
    feats_c = feats - feats.mean(axis=0, keepdims=True)
    _, sing, Vt = np.linalg.svd(feats_c, full_matrices=False)
    return feats_c @ Vt.T, sing


def pc1_spearman(r_values, states, encoder_unitary):
    """Spearman rank correlation between r and PC1 of the latent features."""
    pcs, _ = latent_pca(states, encoder_unitary)
    rho, _ = spearmanr(r_values, pcs[:, 0])
    return float(rho)


def random_encoder_spearman(r_values, states, n_layers, n_draws, seed):
    """|Spearman(r, PC1)| for `n_draws` untrained encoders (seeded)."""
    rng = np.random.default_rng(seed)
    U_fn = make_encoder_unitary(n_layers)
    out = np.empty(n_draws)
    for k in range(n_draws):
        params = rng.uniform(0.0, 2 * np.pi, size=(n_layers, N_QUBITS))
        out[k] = abs(pc1_spearman(r_values, states, U_fn(params)))
    return out
