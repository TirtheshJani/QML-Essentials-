"""Week 22 - Tier 3 / quantum autoencoder, part 1.

Builds the input distribution for the QAE: ground states of H2 (STO-3G) at
22 bond lengths in [0.4, 2.5] A. Verifies that the resulting (22, 16) state
matrix has effective rank ~ 4, which is the geometric reason a 2-qubit code
(4 amplitudes) can fit them. Also exercises the fidelity / partial-trace
helpers in `tier3/utils/states.py`.

No training this week - this is pure infrastructure. The output is checked
end-to-end by the assertions at the bottom.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np

from tier3.utils.states import (
    DIM,
    DIM_CODE,
    DIM_TRASH,
    build_h2_dataset,
    effective_dimension,
    reduced_density_matrix,
    state_overlap,
)

R_GRID = np.round(np.arange(0.4, 2.51, 0.1), 2)  # 22 points


def section(title):
    print("\n" + title)
    print("-" * len(title))


def main():
    section("1. Build H2 ground-state dataset across 22 bond lengths")
    print(f"  r grid (Angstrom): {list(R_GRID)}")
    states, energies = build_h2_dataset(R_GRID)
    print(f"  states.shape = {states.shape}  (k_states, 16-dim amplitudes)")
    norms = np.linalg.norm(states, axis=1)
    print(f"  norms (should all be 1.0): "
          f"min {norms.min():.6f}  max {norms.max():.6f}")
    print(f"  energies range: {energies.min():.4f} -> {energies.max():.4f} Ha")
    print(f"  E(r=0.74) = {energies[np.argmin(np.abs(R_GRID - 0.7))]:.4f} Ha "
          f"(near equilibrium)")

    section("2. Effective dimension via SVD")
    rank, sing = effective_dimension(states, threshold=1e-3)
    print(f"  singular values (top 8): "
          f"{[f'{s:.3e}' for s in sing[:8]]}")
    print(f"  effective rank (sigma > 1e-3) : {rank}")
    print(f"  variance captured by top 4    : "
          f"{(sing[:4] ** 2).sum() / (sing ** 2).sum():.6f}")
    print( "  -> 4 dominant components means the entire dataset lives in a")
    print( "     4-dim linear subspace of C^16. A 2-qubit code (also 4-dim)")
    print( "     is exactly the right size to compress it.")

    section("3. Round-trip fidelity sanity check")
    f_self = state_overlap(states[0], states[0])
    f_eq_close = state_overlap(states[3], states[4])
    f_far = state_overlap(states[0], states[-1])
    print(f"  <psi_0 | psi_0>     : {f_self:.10f}  (must be ~1.0)")
    print(f"  <psi(0.7)|psi(0.8)> : {f_eq_close:.6f}  (close r values overlap)")
    print(f"  <psi(0.4)|psi(2.5)> : {f_far:.6f}  (far apart)")

    section("4. Partial-trace utilities")
    psi = states[6]  # near equilibrium
    rho_code = reduced_density_matrix(psi, keep_wires=[0, 1])
    rho_trash = reduced_density_matrix(psi, keep_wires=[2, 3])
    print(f"  rho_code.shape  = {rho_code.shape}  (4x4)")
    print(f"  rho_trash.shape = {rho_trash.shape}  (4x4)")
    print(f"  Tr(rho_code)    = {float(np.real(np.trace(rho_code))):.10f}  "
          f"(must be 1.0)")
    print(f"  Tr(rho_trash)   = {float(np.real(np.trace(rho_trash))):.10f}  "
          f"(must be 1.0)")
    purity_code = float(np.real(np.trace(rho_code @ rho_code)))
    purity_trash = float(np.real(np.trace(rho_trash @ rho_trash)))
    print(f"  purity(code)    = {purity_code:.6f}")
    print(f"  purity(trash)   = {purity_trash:.6f}")
    print( "  Equal purities -> the input state is entangled across the")
    print( "  code/trash bipartition. The QAE's job is to disentangle it.")

    section("5. Mutual fidelity matrix (preview)")
    n_show = 6
    idx = np.linspace(0, len(states) - 1, n_show).astype(int)
    F = np.empty((n_show, n_show))
    for i, ii in enumerate(idx):
        for j, jj in enumerate(idx):
            F[i, j] = state_overlap(states[ii], states[jj])
    print(f"  fidelity matrix on r = {[f'{R_GRID[k]:.2f}' for k in idx]}:")
    for row in F:
        print("    " + "  ".join(f"{x:.3f}" for x in row))

    section("Checkpoint assertions")
    assert states.shape == (len(R_GRID), DIM)
    assert np.allclose(norms, 1.0, atol=1e-10), \
        "all states must be unit-normalized"
    assert rank <= 4, f"H2 dataset should sit in a 4-dim subspace, got {rank}"
    assert (sing[:4] ** 2).sum() / (sing ** 2).sum() > 0.999, \
        "top 4 singular values should capture > 99.9% of variance"
    assert abs(f_self - 1.0) < 1e-10
    assert f_far < f_eq_close, \
        "states at far-apart r should overlap less than nearby r"
    assert abs(np.trace(rho_code) - 1.0) < 1e-10
    assert abs(np.trace(rho_trash) - 1.0) < 1e-10
    assert abs(purity_code - purity_trash) < 1e-10, \
        "Schmidt symmetry: bipartite purities must match for a pure state"
    print(f"  PASS: 22 H2 states built, effective dim 4, all fidelity / "
          f"partial-trace checks green.")


if __name__ == "__main__":
    main()
