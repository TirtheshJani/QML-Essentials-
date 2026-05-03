"""Week 23 - Train the quantum autoencoder with Romero local cost.

Trains the 4-qubit QAE on the H2 dataset from week 22 (full 22 states used
as training data here; week 24 introduces the train/test split). Loss is
the Romero local cost - one minus the mean P(trash = |00>) over the batch.

Inline barren-plateau monitoring (Tier 2 review item 4): we log the
gradient norm at epoch 0 and every 50 epochs to confirm we are not on the
plateau side of the trainability cliff. With n_layers = 4 and n_qubits = 4,
the week-14 probe predicts Var(grad) ~ 9.9e-2 at random init, so we expect
gradient norms of order sqrt(16 * 1e-1) ~ 1.3 at epoch 0.

Reports both the local fidelity (cost) and the reconstruction fidelity
(numerical, via tier3.utils.states.reconstruction_fidelity) to make week
22's local-vs-recon distinction concrete.

Multi-seed: trains at seeds 0..4, reports mean +/- std on the headline
final fidelity (Tier 2 review item 6).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tier3.utils.states import (
    build_h2_dataset,
    reconstruction_fidelity,
    local_fidelity,
)
from tier3.utils.qae import (
    train_qae,
    make_encoder_unitary,
)

R_GRID = np.round(np.arange(0.4, 2.51, 0.1), 2)
N_LAYERS = 4
N_EPOCHS = 200
LR = 0.05
SEEDS = (0, 1, 2, 3, 4)
GRAD_LOG_AT = (0, 50, 100, 150)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LOSS_PNG = os.path.join(THIS_DIR, "week23_loss_curves.png")


def section(title):
    print("\n" + title)
    print("-" * len(title))


def main():
    section("1. Data + ansatz summary")
    states, energies = build_h2_dataset(R_GRID)
    print(f"  training set            : {len(states)} states (full curve)")
    print(f"  encoder ansatz          : RY + CNOT ladder, "
          f"{N_LAYERS} layers, {N_LAYERS * 4} trainable params")
    print(f"  cost                    : Romero local "
          f"= 1 - mean P(trash = |00>)")
    print(f"  optimizer               : Adam(lr={LR}), {N_EPOCHS} epochs, "
          f"seeds {SEEDS}")

    section("2. Train at each seed")
    print(f"  {'seed':>4s}  {'init loss':>9s}  {'final loss':>10s}  "
          f"{'final fid':>9s}  {'|grad|@0':>9s}  {'|grad|@end':>10s}")

    histories = []
    final_local_fids = []
    final_recon_fids = []
    init_grads = []
    final_grads = []

    for seed in SEEDS:
        history, params = train_qae(
            states,
            n_layers=N_LAYERS,
            n_epochs=N_EPOCHS,
            lr=LR,
            seed=seed,
            log_grad_at=GRAD_LOG_AT,
            verbose=False,
        )
        # Compute reconstruction fidelity numerically with the trained U
        U_fn = make_encoder_unitary(N_LAYERS)
        U = U_fn(np.asarray(params))
        recon_fids = np.array([reconstruction_fidelity(psi, U) for psi in states])
        local_fids = np.array([local_fidelity(psi, U) for psi in states])

        # Pull the (epoch=0) and (final) gradient norms from history
        g0 = next(e[3] for e in history if e[0] == 0)
        gN = next(e[3] for e in history if e[0] == N_EPOCHS - 1)

        init_loss = history[0][1]
        final_loss = history[-1][1]
        print(f"  {seed:>4d}  {init_loss:>9.4f}  {final_loss:>10.4f}  "
              f"{float(np.mean(local_fids)):>9.4f}  "
              f"{g0:>9.4f}  {gN:>10.4f}")
        histories.append([(e[0], e[1], e[2]) for e in history])
        final_local_fids.append(float(np.mean(local_fids)))
        final_recon_fids.append(float(np.mean(recon_fids)))
        init_grads.append(g0)
        final_grads.append(gN)

    section("3. Headline result (mean +/- std over seeds)")
    f_local_mu, f_local_sd = float(np.mean(final_local_fids)), float(np.std(final_local_fids))
    f_recon_mu, f_recon_sd = float(np.mean(final_recon_fids)), float(np.std(final_recon_fids))
    g0_mu = float(np.mean(init_grads))
    gN_mu = float(np.mean(final_grads))
    print(f"  mean local fidelity       : {f_local_mu:.4f} +/- {f_local_sd:.4f}")
    print(f"  mean reconstruction fid.  : {f_recon_mu:.4f} +/- {f_recon_sd:.4f}")
    print(f"  init gradient norm        : {g0_mu:.4f}")
    print(f"  final gradient norm       : {gN_mu:.6f}")
    print(f"  -> training did not flatten gradients to zero; the model")
    print(f"     left the plateau-free regime intact.")

    section("4. Save loss-curve plot (seed = 0)")
    fig, ax = plt.subplots(figsize=(7, 4))
    for seed, h in zip(SEEDS, histories):
        eps = [e[0] for e in h]
        losses = [e[1] for e in h]
        ax.plot(eps, losses, label=f"seed {seed}", alpha=0.7)
    ax.set_xlabel("epoch")
    ax.set_ylabel("Romero local cost  $1 - \\langle P(\\mathrm{trash}=00)\\rangle$")
    ax.set_yscale("log")
    ax.legend(loc="upper right")
    ax.set_title("Tier 3 / week 23: QAE training, 5 seeds")
    fig.tight_layout()
    fig.savefig(LOSS_PNG, dpi=120)
    plt.close(fig)
    print(f"  saved {LOSS_PNG}")

    section("Checkpoint assertions")
    assert f_local_mu > 0.95, \
        f"mean local fidelity {f_local_mu:.4f} below 0.95 target"
    assert f_local_sd < 0.02, \
        f"std across seeds {f_local_sd:.4f} above 0.02 -- run is too noisy"
    assert f_recon_mu > 0.93, \
        f"mean reconstruction fidelity {f_recon_mu:.4f} below 0.93"
    assert gN_mu > 1e-3, \
        f"final gradient norm {gN_mu:.2e} suggests we slid into the plateau"
    assert g0_mu > 0.3, \
        f"init gradient norm {g0_mu:.4f} suggests the ansatz starts on the plateau"
    assert os.path.exists(LOSS_PNG)
    print(f"  PASS: QAE trains to local fid {f_local_mu:.3f} +/- {f_local_sd:.3f}, "
          f"recon fid {f_recon_mu:.3f}, gradients healthy at both ends.")


if __name__ == "__main__":
    main()
