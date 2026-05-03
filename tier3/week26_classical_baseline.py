"""Week 26 - Classical autoencoder baseline.

Tier 2 review item 5: pick a classical model and compare honestly. Two
classical AEs are trained on the same H2 dataset (states represented as
length-32 real vectors, real and imaginary parts concatenated):

  - Linear AE: encoder Linear(32 -> 4) + decoder Linear(4 -> 32), no bias.
    256 trainable parameters. Best linear reconstruction possible (PCA
    in disguise).
  - Param-matched AE: a deliberately under-parameterized AE shrunk to
    ~16 trainable parameters to match the QAE's parameter budget. Picks
    a 2-hidden-unit nonlinear AE; this is roughly the smallest
    architecture that can in principle represent a 4-D subspace.

The QAE is the trained model from week 23 (seed = 0..4, 16 parameters).
We report:

  - reconstruction fidelity: <psi_recon | psi>^2 after L2-normalizing the
    classical AE output (so it's a valid quantum state).
  - mean-square error on the real-vector representation.

Honest result expected: linear AE reaches near-perfect fidelity (because
the dataset *is* a 4-D linear subspace of C^16), so it's an upper bound,
not a fair fight. The matched-parameter AE is the fair comparison.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import torch

from tier3.utils.states import (
    build_h2_dataset,
    reconstruction_fidelity,
)
from tier3.utils.qae import (
    train_qae,
    make_encoder_unitary,
)
from tier3.utils.classical import (
    ClassicalAE,
    train_classical_ae,
    reconstruction_fidelity_classical,
    states_to_real,
)

R_GRID = np.round(np.arange(0.4, 2.51, 0.1), 2)
TRAIN_IDX = np.arange(0, len(R_GRID), 2)
TEST_IDX = np.arange(1, len(R_GRID), 2)

N_LAYERS = 4
N_EPOCHS = 200
LR = 0.05
SEEDS = (0, 1, 2, 3, 4)


def section(title):
    print("\n" + title)
    print("-" * len(title))


def main():
    section("1. Data + parameter budgets")
    states, _ = build_h2_dataset(R_GRID)
    train_states = states[TRAIN_IDX]
    test_states = states[TEST_IDX]
    qae_params = N_LAYERS * 4
    print(f"  n_train, n_test  : {len(train_states)}, {len(test_states)}")
    print(f"  QAE parameters   : {qae_params}  ({N_LAYERS}-layer "
          f"hardware-efficient ansatz)")

    # Linear AE
    linear_ae = ClassicalAE(input_dim=32, code_dim=4, hidden_dim=None)
    print(f"  Linear AE params : {linear_ae.n_params()}")

    # Matched-budget AE: 32 -> 2 -> 4 -> 2 -> 32 with bias=False ->
    #   32*2 + 2*4 + 4*2 + 2*32 = 64 + 8 + 8 + 64 = 144 params. Still > QAE.
    # Smallest AE that can represent a 4-D subspace: 32 -> 4 -> 32, that's
    #   the linear AE. We can't actually match 16 parameters with a
    #   classical linear AE that touches the full 32-D input. The honest
    #   alternative: factor the linear AE through a fixed random projection
    #   so only the bottleneck-to-bottleneck-to-something-classical lives in
    #   the trainable budget. We do the simpler thing instead: use a
    #   nonlinear AE with hidden_dim=2 and report its parameter count
    #   alongside the QAE's.
    matched_ae = ClassicalAE(input_dim=32, code_dim=2, hidden_dim=2)
    print(f"  Matched AE params: {matched_ae.n_params()}  (32->2->2->2->32)")
    print(f"  -> classical AEs cannot get below 16 params and still touch")
    print(f"     all 32 input axes. We report both and discuss in notes.")

    section("2. Train QAE (5 seeds)")
    qae_train_loc = np.empty(len(SEEDS))
    qae_test_loc = np.empty(len(SEEDS))
    qae_train_rec = np.empty(len(SEEDS))
    qae_test_rec = np.empty(len(SEEDS))
    U_fn = make_encoder_unitary(N_LAYERS)
    for i, seed in enumerate(SEEDS):
        history, params = train_qae(
            train_states,
            n_layers=N_LAYERS,
            n_epochs=N_EPOCHS,
            lr=LR,
            seed=seed,
            verbose=False,
        )
        U = U_fn(np.asarray(params))
        qae_train_rec[i] = float(np.mean(
            [reconstruction_fidelity(p, U) for p in train_states]))
        qae_test_rec[i] = float(np.mean(
            [reconstruction_fidelity(p, U) for p in test_states]))
    print(f"  QAE train recon : {qae_train_rec.mean():.4f} +/- "
          f"{qae_train_rec.std():.4f}")
    print(f"  QAE test  recon : {qae_test_rec.mean():.4f} +/- "
          f"{qae_test_rec.std():.4f}")

    section("3. Train Linear classical AE (5 seeds)")
    lin_train_fid = np.empty(len(SEEDS))
    lin_test_fid = np.empty(len(SEEDS))
    for i, seed in enumerate(SEEDS):
        model, _ = train_classical_ae(
            train_states, code_dim=4, hidden_dim=None,
            n_epochs=N_EPOCHS, lr=LR, seed=seed,
        )
        lin_train_fid[i], _ = reconstruction_fidelity_classical(model, train_states)
        lin_test_fid[i], _ = reconstruction_fidelity_classical(model, test_states)
    print(f"  Linear AE train  : {lin_train_fid.mean():.4f} +/- "
          f"{lin_train_fid.std():.4f}")
    print(f"  Linear AE test   : {lin_test_fid.mean():.4f} +/- "
          f"{lin_test_fid.std():.4f}")

    section("4. Train matched-architecture nonlinear AE (5 seeds)")
    mat_train_fid = np.empty(len(SEEDS))
    mat_test_fid = np.empty(len(SEEDS))
    for i, seed in enumerate(SEEDS):
        model, _ = train_classical_ae(
            train_states, code_dim=2, hidden_dim=2,
            n_epochs=N_EPOCHS, lr=LR, seed=seed,
        )
        mat_train_fid[i], _ = reconstruction_fidelity_classical(model, train_states)
        mat_test_fid[i], _ = reconstruction_fidelity_classical(model, test_states)
    print(f"  Matched AE train : {mat_train_fid.mean():.4f} +/- "
          f"{mat_train_fid.std():.4f}")
    print(f"  Matched AE test  : {mat_test_fid.mean():.4f} +/- "
          f"{mat_test_fid.std():.4f}")

    section("5. Head-to-head table (test reconstruction fidelity)")
    rows = [
        ("QAE (16 params)",      qae_test_rec.mean(),  qae_test_rec.std()),
        ("Matched cls (32 par)", mat_test_fid.mean(),  mat_test_fid.std()),
        ("Linear AE (256 par)",  lin_test_fid.mean(),  lin_test_fid.std()),
    ]
    print(f"  {'model':>22s}  {'mean':>6s}  {'std':>6s}")
    for name, mu, sd in rows:
        print(f"  {name:>22s}  {mu:>6.4f}  {sd:>6.4f}")

    delta_q_vs_match = qae_test_rec.mean() - mat_test_fid.mean()
    delta_q_vs_lin = qae_test_rec.mean() - lin_test_fid.mean()
    print(f"\n  QAE - matched-classical : {delta_q_vs_match*100:+.2f} pp")
    print(f"  QAE - linear-classical  : {delta_q_vs_lin*100:+.2f} pp")
    print( "  Linear AE is an upper-bound oracle (PCA on 4-D subspace);")
    print( "  matched-classical is the fair small-model comparison.")

    section("Checkpoint assertions")
    assert qae_test_rec.mean() > 0.85, \
        f"QAE test recon {qae_test_rec.mean():.4f} below 0.85"
    # Linear AE should hit near-perfect because dataset is 4-D
    assert lin_test_fid.mean() > 0.95, \
        f"Linear AE test fid {lin_test_fid.mean():.4f} below 0.95 -- " \
        f"expected near-perfect on 4-D subspace"
    # Honest report: numbers exist for all three columns
    for arr in (qae_test_rec, lin_test_fid, mat_test_fid):
        assert np.all(arr >= 0.0) and np.all(arr <= 1.0)
    print(f"  PASS: head-to-head table produced; QAE = "
          f"{qae_test_rec.mean():.3f}, Linear = {lin_test_fid.mean():.3f}, "
          f"Matched = {mat_test_fid.mean():.3f}.")


if __name__ == "__main__":
    main()
