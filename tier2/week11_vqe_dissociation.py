"""Week 11 — H2 bond-dissociation curve via VQE.

Tier 2 / 2A.3. Working through:
  - sweeping the H-H separation r over compression -> equilibrium -> dissociation
  - rerunning VQE at every grid point; tracking E_VQE, E_HF, E_FCI
  - watching restricted HF fail catastrophically beyond ~1.7 A:
      the closed-shell determinant cannot describe two separated H atoms
  - VQE with AllSinglesDoubles tracks FCI to within ~1 mHa across the curve
  - text-mode plot of E(r) for the three energies
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from tier2.utils.chem import (
    h2_hamiltonian,
    fci_energy,
    hf_energy,
    BOHR_PER_ANGSTROM,
)

# Grid: 0.4 -> 2.5 A in 12 points, equilibrium 0.74 A included.
R_ANGSTROM = np.array(
    [0.40, 0.55, 0.65, 0.74, 0.90, 1.10, 1.30, 1.50, 1.75, 2.00, 2.25, 2.50]
)


def section(title):
    print("\n" + title)
    print("-" * len(title))


def vqe_at_r(r_bohr, max_iters=200, tol=1e-7):
    """Optimize AllSinglesDoubles on H2 at separation r_bohr; return final energy."""
    H, n = h2_hamiltonian(r_bohr)
    singles, doubles = qml.qchem.excitations(2, n)
    hf = qml.qchem.hf_state(2, n)
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev, interface="autograd")
    def energy(theta):
        qml.AllSinglesDoubles(
            weights=theta,
            wires=range(n),
            hf_state=hf,
            singles=singles,
            doubles=doubles,
        )
        return qml.expval(H)

    theta = pnp.zeros(len(singles) + len(doubles), requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=0.1)
    # Warm up before checking convergence: step_and_cost returns the cost at
    # the pre-step theta, so the very first iteration would otherwise show no
    # change and trigger the break before any optimization happens.
    e_prev = None
    for k in range(max_iters):
        theta, e_now = opt.step_and_cost(energy, theta)
        e_now = float(e_now)
        if e_prev is not None and k > 5 and abs(e_now - e_prev) < tol:
            break
        e_prev = e_now
    # one final evaluation at the optimised theta (step_and_cost is one step behind)
    e_final = float(energy(theta))
    return e_final, np.asarray(theta)


def text_plot(rs, series, labels, width=60):
    """Tiny ASCII plot of multiple y-series sharing the same x grid."""
    flat = np.concatenate([np.asarray(s) for s in series])
    y_lo, y_hi = float(flat.min()), float(flat.max())
    pad = 0.05 * (y_hi - y_lo) if y_hi > y_lo else 1e-3
    y_lo, y_hi = y_lo - pad, y_hi + pad

    print(f"  y range: [{y_lo:+.4f}, {y_hi:+.4f}] Ha    legend: " +
          ", ".join(f"{c}={lab}" for c, lab in zip("OXC", labels)))
    print(f"  {'r (A)':>6s} | " +
          "".join(f"{(y_lo + (y_hi - y_lo) * i / (width - 1)):+.3f}".ljust(6)
                  for i in (0, width // 2, width - 1)))
    for k, r in enumerate(rs):
        row = [" "] * width
        for c, ys in zip("OXC", series):
            j = int(round((ys[k] - y_lo) / (y_hi - y_lo) * (width - 1)))
            j = max(0, min(width - 1, j))
            row[j] = c if row[j] == " " else "*"
        print(f"  {r:>6.2f} |{''.join(row)}|")


def main():
    section("1. Sweep r and run VQE / HF / FCI at each point")
    print(f"  {'r (A)':>6s}  {'r (bohr)':>8s}  {'E_HF':>10s}  {'E_VQE':>10s}  "
          f"{'E_FCI':>10s}  {'|VQE-FCI| mHa':>14s}")
    e_hf_arr, e_vqe_arr, e_fci_arr = [], [], []
    for r_a in R_ANGSTROM:
        r_b = r_a * BOHR_PER_ANGSTROM
        H, n = h2_hamiltonian(r_b)
        e_hf = hf_energy(H, n)
        e_fci = fci_energy(H, n)
        e_vqe, _ = vqe_at_r(r_b)
        e_hf_arr.append(e_hf)
        e_vqe_arr.append(e_vqe)
        e_fci_arr.append(e_fci)
        print(f"  {r_a:>6.2f}  {r_b:>8.4f}  {e_hf:>+10.5f}  {e_vqe:>+10.5f}  "
              f"{e_fci:>+10.5f}  {(e_vqe - e_fci)*1000:>+14.4f}")

    e_hf_arr = np.array(e_hf_arr)
    e_vqe_arr = np.array(e_vqe_arr)
    e_fci_arr = np.array(e_fci_arr)

    section("2. Text-mode plot E(r): O = HF, X = VQE, C = FCI")
    text_plot(R_ANGSTROM, [e_hf_arr, e_vqe_arr, e_fci_arr], ["HF", "VQE", "FCI"])

    section("3. Equilibrium and dissociation diagnostics")
    k_eq = int(np.argmin(e_fci_arr))
    print(f"  equilibrium (min FCI):  r = {R_ANGSTROM[k_eq]:.2f} A,  "
          f"E_FCI = {e_fci_arr[k_eq]:+.6f} Ha")
    print(f"  HF-FCI gap at r = 2.5 A: {(e_hf_arr[-1] - e_fci_arr[-1])*1000:>7.2f} mHa")
    print(f"  HF-FCI gap at r = 0.74 A:{(e_hf_arr[3] - e_fci_arr[3])*1000:>7.2f} mHa")
    print( "  -> HF degrades by ~10x toward dissociation; VQE stays flat.")

    max_err_mha = float(np.max(np.abs(e_vqe_arr - e_fci_arr)) * 1000)
    print(f"\n  max |E_VQE - E_FCI| across the curve: {max_err_mha:.4f} mHa")

    section("Checkpoint assertions")
    # variational principle pointwise
    assert np.all(e_vqe_arr >= e_fci_arr - 1e-8), "VQE went below FCI somewhere"
    # VQE tracks FCI within 5 mHa across the entire curve
    assert max_err_mha < 5.0, f"VQE drifts from FCI by {max_err_mha:.3f} mHa > 5 mHa"
    # HF must blow up at dissociation: gap at r=2.5 A is much larger than at equilibrium
    gap_diss = e_hf_arr[-1] - e_fci_arr[-1]
    gap_eq = e_hf_arr[3] - e_fci_arr[3]
    assert gap_diss > 5 * gap_eq, (
        f"expected restricted-HF to fail at dissociation: gap_diss={gap_diss*1000:.2f} mHa "
        f"vs gap_eq={gap_eq*1000:.2f} mHa"
    )
    print(f"  PASS: VQE within {max_err_mha:.3f} mHa of FCI everywhere; "
          f"HF gap grows from {gap_eq*1000:.1f} -> {gap_diss*1000:.1f} mHa.")


if __name__ == "__main__":
    main()
