"""POST-HOC check: the encoder that keeps only the dominant eigenvector.

POST-HOC. Written in review round 4, after every week 22-27 result was
known, in response to a reviewer probe. It is not a week script and not
a gate: the README loop and CI run tier*/week*.py only, and nothing here
changes a week script or a threshold. Run it with
`python tier3/check_dominant_eigvec_baseline.py`. It is seeded and
deterministic, and its output is committed next to it as
`check_dominant_eigvec_baseline.log`.

Why. Every H2 state here is a real combination of |1100> and |0011>, so
the ensemble's average density matrix rho = mean_i |psi_i><psi_i| has two
nonzero eigenvalues, lambda1 > lambda2 = 1 - lambda1, with eigenvectors
v1, v2. Take an encoder that sends v1 to |c>|00> and v2 to |c>|01> for one
fixed code state |c> (the "constant-code encoder"). For psi = a v1 + b v2
it gives P(trash = 00) = |a|^2 = |<v1|psi>|^2, its decoder returns |v1>
for every input, so reconstruction fidelity is the same number, and the
mean over the ensemble is lambda1. Its code state is the same for every
r, so it keeps no information about r. Every tier-3 fidelity gate is a
mean over states, so this is the level a gate has to clear before it can
say anything about whether the code carries r.

For a trained encoder U, P = U^dag (I_code x |00><00|_trash) U projects
onto the inputs U compresses without loss. <v1|P|v1> and <v2|P|v2> say
which data directions it keeps: an encoder that keeps both has
<v2|P|v2> near 1; one that keeps only v1 has <v2|P|v2> near 0.

Sections:
  1. lambda1 for the 22-state set and for the 11-state training half.
  2. The constant-code encoder: local and reconstruction fidelity (all,
     train, test) and the week 23 / 24 fidelity gates it would pass.
  3. Weeks 23 and 24 retrained at their seeds with the same calls as the
     week scripts (same numbers): per seed <v1|P|v1>, <v2|P|v2> and
     Spearman(r, PC1) over the 22 states.
  4. The week-24 seed-0 and seed-2 encoders (section 3 prints which
     directions each keeps) under the week-25 noise model at every swept
     p, without retraining, next to the week-25 random encoders.
  5. Week 25 retrained at p = 0.005: each seed's noiseless <v2|P|v2>.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
from pennylane import numpy as pnp

from tier3.utils.states import (
    N_QUBITS,
    build_h2_dataset,
    local_fidelity,
    reconstruction_fidelity,
)
from tier3.utils.qae import train_qae, make_encoder_unitary
from tier3.utils.latent import code_bloch_vectors, pc1_spearman
from tier3 import week23_qae_training as w23
from tier3 import week24_generalization as w24
from tier3 import week25_noise_robustness as w25

# Projector onto code (x) |00>_trash; state index = code * 4 + trash.
P_CODE_00 = np.kron(np.eye(4), np.diag([1.0, 0.0, 0.0, 0.0]))
LAST = 20   # epochs over which the end-of-training loss range is printed


def section(title):
    print("\n" + title)
    print("-" * len(title))


def top_two(states):
    """Eigenvalues (descending) and the top-2 eigenvectors of mean |psi><psi|."""
    rho = np.einsum("ki,kj->ij", states, states.conj()) / len(states)
    evals, evecs = np.linalg.eigh(rho)
    order = np.argsort(evals)[::-1]
    return evals[order], evecs[:, order[0]], evecs[:, order[1]]


def constant_code_encoder(v1, v2):
    """Unitary sending v1 -> |c=0>|00> (index 0) and v2 -> |c=0>|01> (index 1)."""
    Q, _ = np.linalg.qr(np.column_stack([v1, v2, np.eye(16)]))
    Q = Q[:, :16]
    assert abs(abs(np.vdot(Q[:, 0], v1)) - 1) < 1e-12
    assert abs(abs(np.vdot(Q[:, 1], v2)) - 1) < 1e-12
    return Q.conj().T


def kept_directions(U, v1, v2):
    P = U.conj().T @ P_CODE_00 @ U
    return (float(np.real(np.vdot(v1, P @ v1))),
            float(np.real(np.vdot(v2, P @ v2))))


def fids(states, U):
    loc = np.array([local_fidelity(s, U) for s in states])
    rec = np.array([reconstruction_fidelity(s, U) for s in states])
    return loc, rec


def verdict(ok):
    return "pass" if ok else "FAIL"


def main():
    R = w24.R_GRID
    TR, TE = w24.TRAIN_IDX, w24.TEST_IDX
    states, _ = build_h2_dataset(R)
    states = np.asarray(states, dtype=complex)
    U_fn = make_encoder_unitary(w23.N_LAYERS)
    print("POST-HOC check, written in review round 4 after all week 22-27 results")
    print("were known. Not a gate; no week script or threshold depends on it.")

    section("1. Eigenvalues of the ensemble's average density matrix")
    ev_all, v1_all, v2_all = top_two(states)
    ev_tr, v1_tr, v2_tr = top_two(states[TR])
    print(f"  22 states (week 23 training set) : lambda1 {ev_all[0]:.6f}, "
          f"lambda2 {ev_all[1]:.6f}, rest below {np.abs(ev_all[2:]).max():.1e} in size")
    print(f"  11 train states (week 24)        : lambda1 {ev_tr[0]:.6f}, "
          f"lambda2 {ev_tr[1]:.6f}")
    ov1 = np.abs(states @ v1_all.conj()) ** 2
    print(f"  min over the 22 states of |<v1|psi>|^2 : {ov1.min():.4f} "
          f"(at r = {R[ov1.argmin()]:.2f})")

    section("2. Constant-code encoder (code state identical for every r)")
    print(f"  {'built from':<22s} {'':>5s} {'all 22':>7s} {'train':>7s} {'test':>7s}")
    rows = {}
    for tag, v1, v2 in (("22 states (week 23)", v1_all, v2_all),
                        ("train half (week 24)", v1_tr, v2_tr)):
        Uc = constant_code_encoder(v1, v2)
        loc, rec = fids(states, Uc)
        ov = np.abs(states @ v1.conj()) ** 2
        assert np.allclose(loc, ov, atol=1e-12) and np.allclose(rec, ov, atol=1e-12)
        feats = code_bloch_vectors(states, Uc)
        spread = float(np.abs(feats - feats[0]).max())
        rows[tag] = (loc, rec)
        for name, f in (("local", loc), ("recon", rec)):
            print(f"  {tag:<22s} {name:>5s} {f.mean():>7.4f} {f[TR].mean():>7.4f} "
                  f"{f[TE].mean():>7.4f}")
        print(f"  {'':<22s} largest change in the code-state features across r: "
              f"{spread:.1e}")

    loc23, rec23 = rows["22 states (week 23)"]
    loc24, rec24 = rows["train half (week 24)"]
    # Thresholds copied from the asserts in week23 / week24 (unchanged).
    print("  week 23 gates (built from the 22 states):")
    print(f"    mean local fid > 0.95 : {loc23.mean():.4f} {verdict(loc23.mean() > 0.95)}")
    print(f"    mean recon fid > 0.93 : {rec23.mean():.4f} {verdict(rec23.mean() > 0.93)}")
    print("  week 24 gates (built from the train half):")
    gap = loc24[TR].mean() - loc24[TE].mean()
    print(f"    train local fid > 0.95: {loc24[TR].mean():.4f} {verdict(loc24[TR].mean() > 0.95)}")
    print(f"    test local fid > 0.90 : {loc24[TE].mean():.4f} {verdict(loc24[TE].mean() > 0.90)}")
    print(f"    test recon fid > 0.85 : {rec24[TE].mean():.4f} {verdict(rec24[TE].mean() > 0.85)}")
    print(f"    held-out gap < 0.10   : {gap:+.4f} {verdict(gap < 0.10)}")
    print("    |Spearman(r, PC1)| > 0.9: not defined (the code state does not "
          "change with r)")

    section("3. Trained encoders: which data directions does each seed keep?")
    print("  P = U^dag (I_code x |00><00|) U; v1, v2 from the states each run "
          "trained on.")
    print("  loc = mean P(trash=00) on those states; test rec = mean "
          "reconstruction fidelity on the 11 odd-index r.")
    print("  Spearman(r, PC1) over all 22 states, as in week 24 (gate |rho| > 0.9 "
          "is applied to seed 0 only).")
    trained = {}
    for tag, wk, idx, v1, v2, lam1, log_at in (
            ("week 23 (22 states)", w23, np.arange(len(R)), v1_all, v2_all,
             ev_all[0], w23.GRAD_LOG_AT),
            ("week 24 (train half)", w24, TR, v1_tr, v2_tr, ev_tr[0], (0,))):
        print(f"\n  {tag}: 1 - lambda1 = {1 - lam1:.6f}")
        print(f"  {'seed':>4s} {'final loss':>10s} {'loss range, last ' + str(LAST):>22s} "
              f"{'|grad|@end':>10s} {'<v1|P|v1>':>9s} {'<v2|P|v2>':>9s} "
              f"{'Spearman':>8s} {'loc':>7s} {'test rec':>8s}")
        for seed in wk.SEEDS:
            hist, params = train_qae(states[idx], n_layers=wk.N_LAYERS,
                                     n_epochs=wk.N_EPOCHS, lr=wk.LR, seed=seed,
                                     log_grad_at=log_at, verbose=False)
            U = U_fn(np.asarray(params))
            k1, k2 = kept_directions(U, v1, v2)
            rho_s = pc1_spearman(R, states, U)
            loc, _ = fids(states[idx], U)
            _, rec = fids(states[TE], U)
            tail = [e[1] for e in hist[-LAST:]]
            trained[(tag, seed)] = params
            print(f"  {seed:>4d} {hist[-1][1]:>10.6f} "
                  f"{min(tail):>10.2e} to {max(tail):>8.2e} {hist[-1][3]:>10.4f} "
                  f"{k1:>9.4f} {k2:>9.4f} {rho_s:>+8.4f} {loc.mean():>7.4f} "
                  f"{rec.mean():>8.4f}")

    section("4. Week-24 encoders under the week-25 noise model (no retraining)")
    enc = {sd: trained[("week 24 (train half)", sd)] for sd in (0, 2)}
    for sd, prm in enc.items():
        k1, k2 = kept_directions(U_fn(np.asarray(prm)), v1_tr, v2_tr)
        print(f"  week-24 seed {sd}: <v1|P|v1> {k1:.4f}, <v2|P|v2> {k2:.4f}")
    rng = np.random.default_rng(w25.RANDOM_SEED)   # same draws as week 25
    random_params = [pnp.array(rng.uniform(0.0, 2 * np.pi,
                                           size=(w25.N_LAYERS, N_QUBITS)),
                               requires_grad=False)
                     for _ in range(w25.N_RANDOM)]
    print(f"  test P(trash=00); random = the week-25 random encoders "
          f"(n={w25.N_RANDOM}, mean +/- std)")
    print(f"  {'p':>6s} {'seed 0':>8s} {'seed 2':>8s} {'random':>18s} "
          f"{'seed 2 - random':>16s}")
    noisy = {sd: {} for sd in enc}   # seed -> p -> (train, test)
    rand_mu = {}
    for p in w25.NOISE_LEVELS:
        for sd, prm in enc.items():
            noisy[sd][p] = (w25.evaluate_local_fidelity(states[TR], prm, p),
                            w25.evaluate_local_fidelity(states[TE], prm, p))
        rnd = np.array([w25.evaluate_local_fidelity(states[TE], rp, p)
                        for rp in random_params])
        rand_mu[p] = rnd.mean()
        print(f"  {p:>6.4f} {noisy[0][p][1]:>8.4f} {noisy[2][p][1]:>8.4f} "
              f"{rnd.mean():>9.4f} +/- {rnd.std():.4f} "
              f"{(noisy[2][p][1] - rnd.mean()) * 100:>+13.2f} pp")
    i5 = 0.005
    print("  week 25 gates applied to these encoders (thresholds copied from "
          "week25, unchanged):")
    for sd in enc:
        tests = [noisy[sd][p][1] for p in w25.NOISE_LEVELS]
        gap5 = noisy[sd][i5][1] - rand_mu[i5]
        print(f"    seed {sd}: noiseless train > 0.95 "
              f"{noisy[sd][0.0][0]:.4f} {verdict(noisy[sd][0.0][0] > 0.95)}; "
              f"test at 0.005 > 0.85 {noisy[sd][i5][1]:.4f} "
              f"{verdict(noisy[sd][i5][1] > 0.85)}; "
              f"gap > 30 pp {gap5 * 100:+.2f} {verdict(gap5 > 0.3)}; "
              f"monotone {verdict(bool(np.all(np.diff(tests) <= 1e-3)))}")

    section("5. Week 25 retrained at p = 0.005: which directions does each seed keep?")
    print("  noiseless P of each trained encoder; v1, v2 from the train half")
    print(f"  {'seed':>4s} {'test P00 at 0.005':>17s} {'<v1|P|v1>':>9s} {'<v2|P|v2>':>9s}")
    f5 = []
    n_keep_v1 = 0
    for seed in w25.SEEDS:
        prm = w25.train_noisy(states[TR], i5, w25.N_EPOCHS, w25.LR, seed)
        f = w25.evaluate_local_fidelity(states[TE], prm, i5)
        b1, b2 = kept_directions(U_fn(np.asarray(prm)), v1_tr, v2_tr)
        f5.append(f)
        n_keep_v1 += b2 < 0.5
        print(f"  {seed:>4d} {f:>17.4f} {b1:>9.4f} {b2:>9.4f}")
    f5 = np.array(f5)
    print(f"  trained mean (as in week 25): {f5.mean():.4f} +/- {f5.std():.4f}; "
          f"week-24 seed 2 (section 4): {noisy[2][i5][1]:.4f}, "
          f"{noisy[2][i5][1] - f5.mean():+.4f} from the trained mean; "
          f"week-24 seed 0: {noisy[0][i5][1]:.4f}")
    print(f"  seeds with <v2|P|v2> < 0.5 (keep v1 only): {n_keep_v1} of {len(w25.SEEDS)}")


if __name__ == "__main__":
    main()
