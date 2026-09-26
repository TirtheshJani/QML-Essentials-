"""Numerical check of two bounds on QAE fidelities.

Not a week script: the README loop and CI run tier*/week*.py only. Run it
with `python tier3/check_qae_bounds.py`. Seeded, so the output repeats.

1. Trash-fidelity bound (Ky Fan's maximum principle). The encoder U is
   unitary and P = I_code (x) |00><00|_trash is a rank-4 projector, so the
   mean trash fidelity over an ensemble of states,

       mean_i P_i(trash = 00) = Tr[U^dag P U rho],   rho = mean_i |psi_i><psi_i|,

   is at most the sum of the 4 largest eigenvalues of rho, for every U. A
   linear projection onto the top-4 eigenvectors of rho (PCA with a 4-dim
   complex code) reaches exactly that sum as its mean reconstruction
   fidelity. An ensemble that is not low-rank in amplitude space therefore
   limits the QAE exactly as much as it limits that linear map.
   Checked on a curved 30-state family that is far from low-rank.

2. Reconstruction vs trash fidelity for a pure input:
       F_loc**2 <= F_recon <= F_loc,
   where F_loc = P(trash = 00) and F_recon is the encode -> reset trash ->
   decode fidelity (tier3.utils.states). Writing
   U|psi> = sqrt(q)|phi>|00> + sqrt(1 - q)|chi> with chi's trash part
   orthogonal to |00>, F_recon = q <phi|rho_code|phi> lies in [q**2, q].
   Checked with the repo's own fidelity functions.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.linalg import expm
from scipy.stats import unitary_group

from tier3.utils.states import (
    build_h2_dataset,
    local_fidelity,
    reconstruction_fidelity,
)
from tier3.utils.qae import make_encoder_unitary

DIM = 16
CODE_DIM = 4
N_CURVE = 30
N_HAAR = 2000
N_ANSATZ = 200
N_LAYERS = 4
SEED = 0
TOL = 1e-12
R_GRID = np.round(np.arange(0.4, 2.51, 0.1), 2)


def section(title):
    print("\n" + title)
    print("-" * len(title))


def mean_trash_fidelity(U, states):
    return float(np.mean([local_fidelity(s, U) for s in states]))


def curved_family(rng):
    """psi(t) = exp(-2i t H) psi0 for a seeded random Hermitian H, t in [0, 1]."""
    H = rng.normal(size=(DIM, DIM)) + 1j * rng.normal(size=(DIM, DIM))
    H = H + H.conj().T
    psi0 = rng.normal(size=DIM) + 1j * rng.normal(size=DIM)
    psi0 /= np.linalg.norm(psi0)
    return np.array([expm(-2j * t * H) @ psi0 for t in np.linspace(0, 1, N_CURVE)])


def top_eigen(states):
    rho = np.einsum("ki,kj->ij", states, states.conj()) / len(states)
    evals, evecs = np.linalg.eigh(rho)
    order = np.argsort(evals)[::-1]
    return evals[order], evecs[:, order]


def optimal_encoder(evecs):
    """Unitary whose rows send the top-4 eigenvectors to code (x) |00>."""
    # State index = code * 4 + trash, so trash = 00 means indices 0, 4, 8, 12.
    code_rows = [c * 4 for c in range(CODE_DIM)]
    other_rows = [i for i in range(DIM) if i not in code_rows]
    U = np.zeros((DIM, DIM), dtype=complex)
    for j, row in enumerate(code_rows + other_rows):
        U[row, :] = evecs[:, j].conj()
    return U


def main():
    rng = np.random.default_rng(SEED)

    section("1. Trash-fidelity bound on a curved family that is not low-rank")
    states = curved_family(rng)
    evals, evecs = top_eigen(states)
    bound = float(evals[:CODE_DIM].sum())
    print(f"  states                      : {len(states)}, eigenvalues of rho "
          f"above 1e-6: {int(np.sum(evals > 1e-6))} of {DIM}")
    print(f"  top-4 eigenvalue sum        : {bound:.6f}")

    haar_means = np.array([
        mean_trash_fidelity(unitary_group.rvs(DIM, random_state=SEED + k), states)
        for k in range(N_HAAR)
    ])
    print(f"  {N_HAAR} Haar-random U, mean P(trash=00): max {haar_means.max():.6f}")

    U_fn = make_encoder_unitary(N_LAYERS)
    ansatz_means = np.array([
        mean_trash_fidelity(
            U_fn(rng.uniform(0.0, 2 * np.pi, size=(N_LAYERS, 4))), states)
        for _ in range(N_ANSATZ)
    ])
    print(f"  {N_ANSATZ} random RY+CNOT encoders (the QAE ansatz), mean "
          f"P(trash=00): max {ansatz_means.max():.6f}")

    f_opt = mean_trash_fidelity(optimal_encoder(evecs), states)
    print(f"  encoder built from the top-4 eigenvectors: {f_opt:.6f}")

    proj = evecs[:, :CODE_DIM] @ evecs[:, :CODE_DIM].conj().T
    # Fidelity of the normalized projection with the input: ||P psi||^2.
    f_lin = float(np.mean([np.real(s.conj() @ proj @ s) for s in states]))
    print(f"  rank-4 linear projection (PCA), mean reconstruction fidelity: "
          f"{f_lin:.6f}")

    h2_states, _ = build_h2_dataset(R_GRID)
    h2_evals, _ = top_eigen(np.asarray(h2_states, dtype=complex))
    h2_bound = float(h2_evals[:CODE_DIM].sum())
    print(f"  H2 dataset (22 states): top-4 eigenvalue sum {h2_bound:.6f}, "
          f"eigenvalues above 1e-6: {int(np.sum(h2_evals > 1e-6))}")

    section("2. F_loc**2 <= F_recon <= F_loc for pure inputs")
    n_low = n_high = 0
    for k in range(N_HAAR):
        U = unitary_group.rvs(DIM, random_state=10_000 + k)
        psi = rng.normal(size=DIM) + 1j * rng.normal(size=DIM)
        psi /= np.linalg.norm(psi)
        f_loc = local_fidelity(psi, U)
        f_rec = reconstruction_fidelity(psi, U)
        n_low += f_rec < f_loc ** 2 - TOL
        n_high += f_rec > f_loc + TOL
    print(f"  {N_HAAR} random pure states x Haar-random U: "
          f"F_recon < F_loc^2 in {n_low}, F_recon > F_loc in {n_high}")

    n_low_h2 = n_high_h2 = 0
    for _ in range(N_ANSATZ):
        U = U_fn(rng.uniform(0.0, 2 * np.pi, size=(N_LAYERS, 4)))
        for psi in h2_states:
            f_loc = local_fidelity(psi, U)
            f_rec = reconstruction_fidelity(psi, U)
            n_low_h2 += f_rec < f_loc ** 2 - TOL
            n_high_h2 += f_rec > f_loc + TOL
    print(f"  {N_ANSATZ} random RY+CNOT encoders x 22 H2 states: "
          f"F_recon < F_loc^2 in {n_low_h2}, F_recon > F_loc in {n_high_h2}")

    section("Checks")
    assert haar_means.max() <= bound + TOL, "Haar U above the Ky Fan bound"
    assert ansatz_means.max() <= bound + TOL, "ansatz U above the Ky Fan bound"
    assert abs(f_opt - bound) < 1e-9, "optimal encoder does not reach the bound"
    assert abs(f_lin - bound) < 1e-9, "linear projection does not reach the bound"
    assert n_low == n_high == n_low_h2 == n_high_h2 == 0, "recon bound violated"
    print("  PASS: no encoder exceeds the top-4 eigenvalue sum; the eigenvector "
          "encoder and the rank-4 linear projection both reach it; "
          "F_loc^2 <= F_recon <= F_loc in every case.")


if __name__ == "__main__":
    main()
