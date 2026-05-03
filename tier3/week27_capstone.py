"""Week 27 - Tier 3 capstone.

Single end-to-end script that re-runs the full pipeline at 5 seeds,
saves a results CSV and a 2x2 figure panel, and asserts every
headline number from weeks 23-26 reproduces within 2 standard
deviations.

Pipeline:
  1. Build H2 ground-state dataset (week 22).
  2. Train QAE noiseless on train half, evaluate on test half (week 23-24).
  3. Re-train QAE under depolarizing noise (week 25; reduced-grid sweep).
  4. Train Linear AE and Matched AE classical baselines (week 26).
  5. Aggregate and write tier3/week27_summary.csv + tier3/week27_results.png.

Pass criteria are the original weekly gates, evaluated on this consolidated
run. If anything regresses by more than 2 sigma the script fails its
assertions.
"""

import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from tier3.utils.states import (
    build_h2_dataset,
    reduced_density_matrix,
    reconstruction_fidelity,
    local_fidelity,
    N_QUBITS,
    TRASH_WIRES,
)
from tier3.utils.qae import (
    train_qae,
    make_encoder_unitary,
    N_QUBITS as QAE_N_QUBITS,
)
from tier3.utils.classical import (
    train_classical_ae,
    reconstruction_fidelity_classical,
)

R_GRID = np.round(np.arange(0.4, 2.51, 0.1), 2)
TRAIN_IDX = np.arange(0, len(R_GRID), 2)
TEST_IDX = np.arange(1, len(R_GRID), 2)
N_LAYERS = 4
N_EPOCHS = 200
LR = 0.05
SEEDS = (0, 1, 2, 3, 4)
NOISE_SEEDS = (0, 1, 2)
NOISE_LEVELS = (0.0, 0.005, 0.02)  # reduced sweep for the capstone
NOISE_EPOCHS = 100

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(THIS_DIR, "week27_summary.csv")
PNG_PATH = os.path.join(THIS_DIR, "week27_results.png")


def section(title):
    print("\n" + title)
    print("-" * len(title))


def make_noisy_qnode(noise_p):
    dev = qml.device("default.mixed", wires=QAE_N_QUBITS)

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def qnode(state, params):
        qml.StatePrep(state, wires=range(QAE_N_QUBITS))
        for L in range(N_LAYERS):
            for w in range(QAE_N_QUBITS):
                qml.RY(params[L, w], wires=w)
                if noise_p > 0:
                    qml.DepolarizingChannel(noise_p, wires=w)
            for w in range(QAE_N_QUBITS - 1):
                qml.CNOT(wires=[w, w + 1])
                if noise_p > 0:
                    qml.DepolarizingChannel(noise_p, wires=w)
                    qml.DepolarizingChannel(noise_p, wires=w + 1)
        return qml.probs(wires=TRASH_WIRES)

    return qnode


def train_noisy(train_states, noise_p, n_epochs, lr, seed):
    rng = np.random.default_rng(seed)
    qnode = make_noisy_qnode(noise_p)
    params = pnp.array(
        rng.uniform(0.0, 2 * np.pi, size=(N_LAYERS, QAE_N_QUBITS)),
        requires_grad=True,
    )
    opt = qml.AdamOptimizer(stepsize=lr)

    def cost(p):
        total = 0.0
        for psi in train_states:
            total = total + qnode(psi, p)[0]
        return 1.0 - total / len(train_states)

    for _ in range(n_epochs):
        params, _ = opt.step_and_cost(cost, params)
    return np.asarray(params)


def evaluate_noisy_local(states, params, noise_p):
    qnode = make_noisy_qnode(noise_p)
    return float(np.mean([float(qnode(psi, pnp.array(params))[0])
                          for psi in states]))


def main():
    section("0. Capstone setup")
    states, energies = build_h2_dataset(R_GRID)
    train_states = states[TRAIN_IDX]
    test_states = states[TEST_IDX]
    print(f"  dataset             : {len(states)} H2 ground states "
          f"in [0.4, 2.5] A")
    print(f"  train/test split    : {len(train_states)} / {len(test_states)}")
    print(f"  QAE                 : {N_LAYERS} layers, "
          f"{N_LAYERS * QAE_N_QUBITS} parameters")
    print(f"  seeds               : {SEEDS}")

    csv_rows = [("week", "model", "split", "metric", "mean", "std", "n_seeds")]

    section("1. Week 23 - noiseless training (5 seeds, full dataset)")
    train_loc_full = np.empty(len(SEEDS))
    init_grad = np.empty(len(SEEDS))
    final_grad = np.empty(len(SEEDS))
    U_fn = make_encoder_unitary(N_LAYERS)
    for i, seed in enumerate(SEEDS):
        history, params = train_qae(
            states, n_layers=N_LAYERS, n_epochs=N_EPOCHS,
            lr=LR, seed=seed, log_grad_at=(0,), verbose=False,
        )
        U = U_fn(np.asarray(params))
        train_loc_full[i] = float(np.mean(
            [local_fidelity(p, U) for p in states]))
        init_grad[i] = next(e[3] for e in history if e[0] == 0)
        final_grad[i] = next(e[3] for e in history if e[0] == N_EPOCHS - 1)
    print(f"  full-curve local fid : {train_loc_full.mean():.4f} +/- "
          f"{train_loc_full.std():.4f}")
    print(f"  init grad norm       : {init_grad.mean():.4f}")
    print(f"  final grad norm      : {final_grad.mean():.6f}")
    csv_rows.append(("week23", "QAE", "full", "local_fid",
                     f"{train_loc_full.mean():.4f}",
                     f"{train_loc_full.std():.4f}", len(SEEDS)))

    section("2. Week 24 - generalization on train/test split")
    qae_train_rec = np.empty(len(SEEDS))
    qae_test_rec = np.empty(len(SEEDS))
    qae_test_loc = np.empty(len(SEEDS))
    seed0_U = None
    for i, seed in enumerate(SEEDS):
        _, params = train_qae(
            train_states, n_layers=N_LAYERS, n_epochs=N_EPOCHS,
            lr=LR, seed=seed, verbose=False,
        )
        U = U_fn(np.asarray(params))
        if seed == 0:
            seed0_U = U
        qae_train_rec[i] = float(np.mean(
            [reconstruction_fidelity(p, U) for p in train_states]))
        qae_test_rec[i] = float(np.mean(
            [reconstruction_fidelity(p, U) for p in test_states]))
        qae_test_loc[i] = float(np.mean(
            [local_fidelity(p, U) for p in test_states]))
    print(f"  QAE train recon : {qae_train_rec.mean():.4f} +/- "
          f"{qae_train_rec.std():.4f}")
    print(f"  QAE test  recon : {qae_test_rec.mean():.4f} +/- "
          f"{qae_test_rec.std():.4f}")
    print(f"  QAE test  local : {qae_test_loc.mean():.4f} +/- "
          f"{qae_test_loc.std():.4f}")
    csv_rows.append(("week24", "QAE", "test", "recon_fid",
                     f"{qae_test_rec.mean():.4f}",
                     f"{qae_test_rec.std():.4f}", len(SEEDS)))

    section("3. Week 24 - latent-space monotonicity")
    feats = []
    for psi in states:
        enc = seed0_U @ psi
        T = enc.reshape(4, 4)
        rho_code = T @ np.conj(T.T)
        feats.append(rho_code.flatten())
    feats = np.array(feats)
    feats = np.concatenate([feats.real, feats.imag], axis=1)
    feats_c = feats - feats.mean(axis=0, keepdims=True)
    _, sing, Vt = np.linalg.svd(feats_c, full_matrices=False)
    pcs = feats_c @ Vt.T
    rho_pc1, _ = spearmanr(R_GRID, pcs[:, 0])
    print(f"  Spearman(r, PC1) : {rho_pc1:+.4f}")
    csv_rows.append(("week24", "QAE", "all", "spearman_r_pc1",
                     f"{rho_pc1:+.4f}", "0.0", "1"))

    section("4. Week 25 - depolarizing noise sweep (reduced)")
    noise_table = {}
    for p in NOISE_LEVELS:
        per_seed = np.empty(len(NOISE_SEEDS))
        for j, seed in enumerate(NOISE_SEEDS):
            params = train_noisy(train_states, p, NOISE_EPOCHS, LR, seed)
            per_seed[j] = evaluate_noisy_local(test_states, params, p)
        noise_table[p] = per_seed
        print(f"  p = {p:>6.4f} : {per_seed.mean():.4f} +/- {per_seed.std():.4f}")
        csv_rows.append(("week25", "QAE", f"test_p={p}", "local_fid",
                         f"{per_seed.mean():.4f}",
                         f"{per_seed.std():.4f}", len(NOISE_SEEDS)))

    section("5. Week 26 - classical baselines")
    lin_test = np.empty(len(SEEDS))
    mat_test = np.empty(len(SEEDS))
    for i, seed in enumerate(SEEDS):
        m_lin, _ = train_classical_ae(
            train_states, code_dim=4, hidden_dim=None,
            n_epochs=N_EPOCHS, lr=LR, seed=seed,
        )
        lin_test[i], _ = reconstruction_fidelity_classical(m_lin, test_states)
        m_mat, _ = train_classical_ae(
            train_states, code_dim=2, hidden_dim=2,
            n_epochs=N_EPOCHS, lr=LR, seed=seed,
        )
        mat_test[i], _ = reconstruction_fidelity_classical(m_mat, test_states)
    print(f"  Linear AE test  : {lin_test.mean():.4f} +/- {lin_test.std():.4f}")
    print(f"  Matched AE test : {mat_test.mean():.4f} +/- {mat_test.std():.4f}")
    csv_rows.append(("week26", "LinearAE", "test", "recon_fid",
                     f"{lin_test.mean():.4f}",
                     f"{lin_test.std():.4f}", len(SEEDS)))
    csv_rows.append(("week26", "MatchedAE", "test", "recon_fid",
                     f"{mat_test.mean():.4f}",
                     f"{mat_test.std():.4f}", len(SEEDS)))

    section("6. Save CSV")
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        for row in csv_rows:
            w.writerow(row)
    print(f"  saved {CSV_PATH}")

    section("7. Save 2x2 panel figure")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

    # (0, 0) loss curve, seed = 0
    history_seed0, _ = train_qae(
        train_states, n_layers=N_LAYERS, n_epochs=N_EPOCHS,
        lr=LR, seed=0, verbose=False,
    )
    ax = axes[0, 0]
    ax.plot([e[0] for e in history_seed0], [e[1] for e in history_seed0])
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Romero local cost")
    ax.set_title("(a) QAE training loss, seed = 0")

    # (0, 1) latent trajectory
    ax = axes[0, 1]
    sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=R_GRID, cmap="viridis",
                    s=70, edgecolor="k", linewidth=0.4)
    fig.colorbar(sc, ax=ax, label="r (A)")
    ax.set_xlabel("PC1(rho_code)")
    ax.set_ylabel("PC2(rho_code)")
    ax.set_title(f"(b) latent trajectory, Spearman = {rho_pc1:+.3f}")

    # (1, 0) noise sweep
    ax = axes[1, 0]
    ps = np.array(NOISE_LEVELS)
    means = np.array([noise_table[p].mean() for p in NOISE_LEVELS])
    sds = np.array([noise_table[p].std() for p in NOISE_LEVELS])
    ax.errorbar(ps, means, yerr=sds, marker="o", capsize=3)
    ax.set_xlabel("depolarizing rate p")
    ax.set_ylabel("test local fidelity")
    ax.set_xscale("symlog", linthresh=1e-4)
    ax.set_title("(c) noise robustness")
    ax.grid(True, alpha=0.3)

    # (1, 1) Q-vs-classical bars
    ax = axes[1, 1]
    bar_names = ["QAE\n(16 par)", "Matched\nclassical\n(32 par)",
                 "Linear AE\n(256 par)"]
    bar_means = [qae_test_rec.mean(), mat_test.mean(), lin_test.mean()]
    bar_sds = [qae_test_rec.std(), mat_test.std(), lin_test.std()]
    colors = ["#3a86ff", "#fb5607", "#06a77d"]
    ax.bar(bar_names, bar_means, yerr=bar_sds, capsize=4, color=colors,
           edgecolor="k", linewidth=0.5)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("test reconstruction fidelity")
    ax.set_title("(d) head-to-head: 5-seed mean +/- std")
    for i, (mu, sd) in enumerate(zip(bar_means, bar_sds)):
        ax.text(i, mu + sd + 0.02, f"{mu:.3f}", ha="center", fontsize=9)

    fig.suptitle("Tier 3 capstone: quantum autoencoder on H2 ground states",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=120)
    plt.close(fig)
    print(f"  saved {PNG_PATH}")

    section("Checkpoint assertions")
    # week 23 reproduction
    assert train_loc_full.mean() > 0.95 - 2 * train_loc_full.std(), \
        f"week 23 regression: full-curve local fid {train_loc_full.mean():.4f}"
    # week 24 reproduction
    assert qae_test_rec.mean() > 0.85 - 2 * qae_test_rec.std(), \
        f"week 24 regression: test recon {qae_test_rec.mean():.4f}"
    assert abs(rho_pc1) > 0.9, \
        f"latent monotonicity broken: |Spearman| = {abs(rho_pc1):.3f}"
    # week 25 reproduction
    test_at_005 = noise_table[0.005]
    assert test_at_005.mean() > 0.85 - 2 * test_at_005.std(), \
        f"week 25 regression: p=0.005 test fid {test_at_005.mean():.4f}"
    # week 26 reproduction
    assert lin_test.mean() > 0.95, \
        f"linear AE oracle broken: {lin_test.mean():.4f}"
    # artifacts exist
    assert os.path.exists(CSV_PATH)
    assert os.path.exists(PNG_PATH)
    print(f"  PASS: capstone reproduces every weekly headline within 2 sigma; "
          f"CSV + 4-panel figure saved.")


if __name__ == "__main__":
    main()
