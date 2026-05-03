"""Week 24 - Generalization and latent-space geometry.

Splits the 22-state H2 dataset into 11 train + 11 test (every other r),
retrains the QAE on the train half, evaluates on the held-out half, and
visualizes the latent code: as r varies, the encoded code-qubit reduced
density matrix sweeps out a smooth trajectory in the Bloch ball. We
report the Spearman rank correlation between r and the leading PCA axis
of those Bloch points - a clean monotonicity test for whether the QAE
discovered the bond-length axis as its latent variable.

Multi-seed: held-out fidelity reported as mean +/- std over 5 seeds.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
from scipy.stats import spearmanr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tier3.utils.states import (
    build_h2_dataset,
    reduced_density_matrix,
    reconstruction_fidelity,
    local_fidelity,
)
from tier3.utils.qae import (
    train_qae,
    make_encoder_unitary,
)

R_GRID = np.round(np.arange(0.4, 2.51, 0.1), 2)
TRAIN_IDX = np.arange(0, len(R_GRID), 2)  # even indices
TEST_IDX = np.arange(1, len(R_GRID), 2)   # odd indices

N_LAYERS = 4
N_EPOCHS = 200
LR = 0.05
SEEDS = (0, 1, 2, 3, 4)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LATENT_PNG = os.path.join(THIS_DIR, "week24_latent_trajectory.png")


def section(title):
    print("\n" + title)
    print("-" * len(title))


def code_bloch_vectors(states, encoder_unitary):
    """Project each state's encoded code-qubit reduced density matrix onto
    a 15-dim "Bloch-like" vector (real flatten of Hermitian rho_code minus
    its trace). Returns (n, 16) array suitable for PCA."""
    feats = []
    for psi in states:
        enc = encoder_unitary @ psi
        T = enc.reshape(4, 4)
        rho_code = T @ np.conj(T.T)
        feats.append(rho_code.flatten())
    feats = np.array(feats)
    return np.concatenate([feats.real, feats.imag], axis=1)


def main():
    section("1. Train/test split on the H2 dataset")
    states, energies = build_h2_dataset(R_GRID)
    train_states = states[TRAIN_IDX]
    test_states = states[TEST_IDX]
    print(f"  train r values : {[f'{R_GRID[i]:.2f}' for i in TRAIN_IDX]}")
    print(f"  test  r values : {[f'{R_GRID[i]:.2f}' for i in TEST_IDX]}")
    print(f"  n_train = {len(train_states)}, n_test = {len(test_states)}")

    section("2. Train QAE per seed; report train + test fidelity")
    print(f"  {'seed':>4s}  {'train (loc)':>11s}  {'test (loc)':>10s}  "
          f"{'train (rec)':>11s}  {'test (rec)':>10s}")
    train_loc = np.empty(len(SEEDS))
    test_loc = np.empty(len(SEEDS))
    train_rec = np.empty(len(SEEDS))
    test_rec = np.empty(len(SEEDS))
    seed0_params = None
    seed0_U = None
    U_fn = make_encoder_unitary(N_LAYERS)
    for i, seed in enumerate(SEEDS):
        history, params = train_qae(
            train_states,
            n_layers=N_LAYERS,
            n_epochs=N_EPOCHS,
            lr=LR,
            seed=seed,
            log_grad_at=(0,),
            verbose=False,
        )
        U = U_fn(np.asarray(params))
        if seed == 0:
            seed0_params = params
            seed0_U = U
        train_loc[i] = float(np.mean([local_fidelity(p, U) for p in train_states]))
        test_loc[i] = float(np.mean([local_fidelity(p, U) for p in test_states]))
        train_rec[i] = float(np.mean([reconstruction_fidelity(p, U) for p in train_states]))
        test_rec[i] = float(np.mean([reconstruction_fidelity(p, U) for p in test_states]))
        print(f"  {seed:>4d}  {train_loc[i]:>11.4f}  {test_loc[i]:>10.4f}  "
              f"{train_rec[i]:>11.4f}  {test_rec[i]:>10.4f}")

    print(f"\n  mean +/- std across seeds:")
    print(f"    train local fid : {train_loc.mean():.4f} +/- {train_loc.std():.4f}")
    print(f"    test  local fid : {test_loc.mean():.4f} +/- {test_loc.std():.4f}")
    print(f"    train recon fid : {train_rec.mean():.4f} +/- {train_rec.std():.4f}")
    print(f"    test  recon fid : {test_rec.mean():.4f} +/- {test_rec.std():.4f}")

    gap_loc = train_loc.mean() - test_loc.mean()
    gap_rec = train_rec.mean() - test_rec.mean()
    print(f"  generalization gap (local): {gap_loc*100:+.2f} pp")
    print(f"  generalization gap (recon): {gap_rec*100:+.2f} pp")

    section("3. Latent-space trajectory (seed = 0 model, all 22 states)")
    feats = code_bloch_vectors(states, seed0_U)
    # PCA via SVD on centered features
    feats_c = feats - feats.mean(axis=0, keepdims=True)
    _, sing, Vt = np.linalg.svd(feats_c, full_matrices=False)
    pcs = feats_c @ Vt.T  # (22, k)
    print(f"  feature dim (rho_code real+imag flatten) : {feats.shape[1]}")
    print(f"  top-3 singular values                     : "
          f"{[f'{s:.3e}' for s in sing[:3]]}")
    print(f"  variance captured by PC1                  : "
          f"{(sing[0] ** 2) / (sing ** 2).sum():.4f}")

    rho_pc1, p_pc1 = spearmanr(R_GRID, pcs[:, 0])
    print(f"  Spearman( r , PC1 ) : rho = {rho_pc1:+.4f}  p = {p_pc1:.2e}")

    section("4. Save latent-trajectory plot")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=R_GRID, cmap="viridis",
                    s=70, edgecolor="k", linewidth=0.4)
    for i, r in enumerate(R_GRID):
        if i % 2 == 0:
            ax.annotate(f"{r:.1f}", (pcs[i, 0], pcs[i, 1]),
                        fontsize=7, xytext=(4, 4), textcoords="offset points")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("bond length r (A)")
    ax.set_xlabel("PC1 of code-qubit rho")
    ax.set_ylabel("PC2 of code-qubit rho")
    ax.set_title(f"Tier 3 / week 24: latent trajectory  "
                 f"(Spearman(r, PC1) = {rho_pc1:+.3f})")
    fig.tight_layout()
    fig.savefig(LATENT_PNG, dpi=120)
    plt.close(fig)
    print(f"  saved {LATENT_PNG}")

    section("Checkpoint assertions")
    assert train_loc.mean() > 0.95, \
        f"train local fid {train_loc.mean():.4f} < 0.95"
    assert test_loc.mean() > 0.90, \
        f"test local fid {test_loc.mean():.4f} < 0.90 (held-out gate)"
    assert test_rec.mean() > 0.85, \
        f"test recon fid {test_rec.mean():.4f} < 0.85"
    assert gap_loc < 0.10, \
        f"generalization gap {gap_loc:.4f} > 0.10 (overfitting)"
    assert abs(rho_pc1) > 0.9, \
        f"PC1 fails to track r monotonically: |Spearman| = {abs(rho_pc1):.3f}"
    assert os.path.exists(LATENT_PNG)
    print(f"  PASS: held-out recon fid {test_rec.mean():.3f}; "
          f"latent PC1 tracks r at |rho|={abs(rho_pc1):.3f}.")


if __name__ == "__main__":
    main()
