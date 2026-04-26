# Week 11 — H₂ bond-dissociation curve

**Tier 2 / 2A.3.** Companion to `week11_vqe_dissociation.py`.

## 1. Why the dissociation curve matters

Equilibrium energies are easy. The hard test is the *full* potential energy surface, especially the **dissociation limit** $r \to \infty$, where the molecule splits into two non-interacting H atoms:

$$E(r\to\infty) \;\to\; 2 E_H \;=\; -1\ \text{Ha (FCI/STO-3G ≈ -0.93 Ha after basis-set error)}.$$

A method that gets equilibrium right but fails at dissociation is useless for chemistry — bond breaking, reaction barriers, and transition states all live in this regime.

## 2. Where restricted Hartree–Fock fails

Restricted HF (RHF) constrains both electrons to occupy the same spatial orbital with opposite spin. Near equilibrium that's roughly true. At large $r$, the true ground state is

$$|\psi_{\text{true}}\rangle = \tfrac{1}{\sqrt 2}\big(|\uparrow_L\,\downarrow_R\rangle - |\downarrow_L\,\uparrow_R\rangle\big),$$

i.e. one electron localized on each atom. RHF can only see the molecular bonding orbital $\sigma_g = (1s_L + 1s_R)/\sqrt 2$, which forces the two electrons to share both $L$ and $R$ — wave-function weight is wasted on ionic configurations $|\uparrow_L\downarrow_L\rangle$ and $|\uparrow_R\downarrow_R\rangle$ that the true state should suppress. The result is a runaway energy error:

| $r$ (Å) | $E_{HF} - E_{FCI}$ (mHa) |
|--------:|-------------------------:|
| 0.40 | +9.8 |
| 0.74 (eq.) | +20.5 |
| 1.50 | +87.3 |
| 2.50 | +233.1 |

This is the **static (left-right) correlation** problem. No amount of basis-set improvement fixes it; you need a multi-determinant ansatz.

## 3. Why VQE / AllSinglesDoubles tracks FCI

For 2 electrons in 4 spin orbitals, the singlet sector has dimension 3, spanned by:

$$|1100\rangle,\quad |0011\rangle,\quad \tfrac{1}{\sqrt2}(|1001\rangle - |0110\rangle).$$

The double excitation in `AllSinglesDoubles`, $\hat T_{0,1\to 2,3}$, mixes $|1100\rangle$ with $|0011\rangle$. With its angle $\theta_d$ free, the ansatz can interpolate continuously from pure HF ($\theta_d = 0$) to a 50/50 superposition ($\theta_d = \pi/2$) — exactly the parametric family needed to follow the bond from equilibrium to dissociation. The script confirms VQE stays within **0.05 mHa** of FCI across the whole sweep.

## 4. Reading the text plot

```
   r |  -1.16            -0.92            -0.68
0.40 |                              *O
...
2.50 |                            *                           O
```

`*` is where the VQE and FCI markers overlap (within plot resolution). `O` is the restricted HF. They start nearly together, then HF drifts steadily right as the bond breaks. The asterisk traces the textbook H₂ Morse-like potential.

## 5. The optimizer pitfall (worth flagging)

`opt.step_and_cost(f, theta)` returns `(theta_new, f(theta_old))` — the *previous* iteration's energy. A naive `e_prev = f(theta); for k: theta, e = step_and_cost(f, theta); if abs(e_prev - e) < tol: break` triggers an immediate false convergence on iteration 1, since `e == e_prev` by construction. Fix: ignore the convergence check for the first few iterations, or compare against the energy at the optimised theta after the loop. The script does both.

## 6. What the script verifies

- Pointwise variational principle $E_{VQE}(r) \ge E_{FCI}(r)$ across the full grid.
- $\max_r |E_{VQE}(r) - E_{FCI}(r)| < 5$ mHa (in practice ~0.05 mHa).
- HF–FCI gap grows by more than $5\times$ from equilibrium to $r = 2.5$ Å — the static-correlation failure made empirical.
