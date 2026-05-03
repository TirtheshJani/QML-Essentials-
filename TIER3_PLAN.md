# Tier 3 Plan — Quantum Autoencoder for Compressing Physics-Generated States

## Context

Tier 2 (weeks 9–21) finished green: VQE on H₂, QAOA on MaxCut, variational
classifier on Iris, quantum kernel + SVM on Iris, plus a hybrid capstone on
`digits` 0-vs-1. The cross-tier review (`TIER2_REVIEW.md`) landed five
recommendations for Tier 3. They are the design constraints below.

`README.md:86–98` lists four candidate Tier 3 projects:

- quantum autoencoder
- hybrid model on a real dataset (MonitSharma's portfolio)
- tensor network (MPS) methods
- reproduce one paper from artix41's awesome list

This plan picks **quantum autoencoder** and reuses one Tier 2 component as the
data source: the VQE machinery from 2A produces a 1-parameter family of
4-qubit ground states (H₂ at varying bond length), which becomes the input
distribution for the autoencoder. The artifact therefore connects Tier 2's
"quantum solving a quantum problem" (VQE) to Tier 3's "quantum compressing
quantum data" (QAE) without inventing fresh data.

## Reference

- Romero, Olson, Aspuru-Guzik, *Quantum autoencoders for efficient
  compression of quantum data*, **arXiv:1612.02806** (2017). Eq. 6 (local
  trash-fidelity cost) is the training target. Sec. III.B is the H₂
  application that we will reproduce in spirit at smaller scale.
- Schuld & Petruccione ch. 9 (autoencoders + state compression).
- Tier 2 review §4 — five lessons that this plan operationalizes.

## Why this project

QAE checks every box from the Tier 2 review:

| Tier 2 review item | how QAE addresses it |
|------|------|
| 1. data with quantum structure | inputs are quantum states by construction |
| 2. real / noisy backend | week 25 swaps `default.qubit` → `default.mixed` + depolarizing channel |
| 3. assertion-gated weeks | every script keeps the tier-1/2 assert pattern |
| 4. barren-plateau monitoring early | week 23 logs gradient variance from epoch 1 |
| 5. honest classical baseline | week 26 trains a parameter-matched classical AE |
| 6. multi-seed reproducibility | every headline number reported as mean ± std over 5 seeds |

It is also a *quantum-native* task: input is quantum states, output is
quantum states. The classical baseline (PCA / classical AE on amplitude
vectors) is competing on the quantum side's home turf, which is the inverse
of Tier 2's tabular-data fight.

## Decisions

- **Layout:** `tier3/weekN_*.py` + `weekN_notes.md`, plus `tier3/utils/`
  for shared QAE/state helpers. Same convention as `tier2/`.
- **Length:** 6 weeks (weeks 22–27). Week 27 = capstone artifact + writeup.
  Tier 2 was 13 weeks; Tier 3 is shorter on purpose — depth-over-breadth, one
  high-quality result instead of four medium-quality ones.
- **Model size:** 4-qubit input, 2-qubit code, 2-qubit trash. Inside the
  trainable regime per the week 14 / week 21 barren-plateau probe.
- **Dataset:** H₂ ground states ψ(r) for r ∈ {0.4, 0.5, …, 2.5} Å. Generated
  with `tier2/utils/chem.py` so we exercise the Tier 2 pipeline.
- **Cost function:** Romero local cost — `1 − E[P(trash = 00)]` over the
  training distribution.
- **Pass criteria:** assertion-gated in every script, mirroring
  `tier1/week8_checkpoint_bell_grover.py:151–154`.

## Directory layout

```
tier3/
  utils/
    __init__.py
    states.py        # H2 ground-state dataset, fidelity, partial-trace
    qae.py           # encoder ansatz, Romero cost, training loop
    classical.py     # parameter-matched classical AE baseline
  week22_qae_dataset.py            + week22_notes.md
  week23_qae_training.py           + week23_notes.md
  week24_generalization.py         + week24_notes.md
  week25_noise_robustness.py       + week25_notes.md
  week26_classical_baseline.py     + week26_notes.md
  week27_capstone.py               + week27_notes.md
TIER3_REVIEW.md
```

## Reusable patterns from Tier 1 + Tier 2

- `section()` print helper + assertion-gated `main()` —
  `tier1/week8_checkpoint_bell_grover.py:110–154`. Same shape every week.
- `tier2/utils/chem.py` — H₂ Hamiltonian + ground-state diagonalization for
  generating the dataset. **Direct reuse**, no copy-paste.
- `tier2/utils/barren.py` — gradient-variance probe. Imported by week 23 to
  monitor trainability throughout training (review item 4).
- Hardware-efficient ansatz (`RY` + nearest-neighbour CNOTs) from week 14 /
  week 21 — used as the encoder's parametric circuit.
- `qml.qnn.TorchLayer` wrapping pattern from week 16 / week 21 — used for
  Adam optimization with `torch`-native control flow.

## Week-by-week

### Week 22 — Dataset + fidelity infrastructure

Build the input distribution and the fidelity metric, no training yet.

- Compute H₂ ground states via exact diagonalization of the Tier 2
  Hamiltonians at `r ∈ {0.4, 0.5, …, 2.5} Å` (22 states total).
- Verify each state is a valid 16-dim unit vector and that the set has
  effective dimension ≪ 16 (SVD of the stacked matrix; the leading 4
  singular values should capture > 0.999 of the variance — that is the
  geometric reason a 2-qubit code can fit them).
- Implement `state_fidelity(rho, sigma)` for both pure states (overlap²)
  and mixed states (Uhlmann formula via `scipy.linalg.sqrtm`).
- Pass: 22 states constructed, all unit-norm; rank-4 SVD captures > 0.999
  variance; round-trip fidelity of any state with itself is 1.0 ± 1e-10.

### Week 23 — Train the quantum autoencoder

The actual QAE training run. This is where the model is born.

- 4-qubit hardware-efficient encoder, depth = 4 layers (16 trainable
  parameters — well below the trainable cliff measured in week 14).
- Romero local cost: `C(α) = 1 − (1/N) Σ_i P(trash bits = 0 | input ψ_i)`
  computed via `qml.probs(wires=trash_wires)`.
- Adam, lr = 0.05, 200 epochs, 5 seeds.
- **Barren-plateau monitoring inline:** log gradient variance at epoch 0
  and epoch 200 to confirm we never enter the plateau regime during
  training (Tier 2 review item 4).
- Pass: mean training fidelity > 0.95 over 5 seeds with std < 0.02; final
  gradient variance still > 1e-3 (not on the plateau).

### Week 24 — Generalization + latent geometry

Held-out fidelity and a look at what the code qubits learned.

- Train on every other r value (11 states), test on the other 11.
- Reconstruction fidelity on the held-out set, mean ± std over 5 seeds.
- Latent-space visualization: for each input state, compute the reduced
  density matrix on the code qubits after encoding, extract the dominant
  Bloch vector, plot it as r varies. Smooth ordering = the QAE found the
  bond-length axis.
- Pass: held-out mean fidelity > 0.90 over 5 seeds; latent trajectory has
  monotone projection onto its first PCA axis with r (Spearman ρ > 0.9).

### Week 25 — Noise robustness

Move from `default.qubit` to `default.mixed`. Tier 2 review item 2.

- Add depolarizing noise after every gate via PennyLane's
  `qml.transforms.insert(qml.DepolarizingChannel, p)` for
  p ∈ {0.0, 0.001, 0.005, 0.01, 0.02}.
- For each noise level: retrain QAE for 100 epochs, evaluate test
  fidelity. Plot fidelity vs p curve.
- Compare against a "noisy classical" sanity check: same fidelity metric
  on random unitaries instead of the trained encoder, to show that the
  trained model degrades gracefully but a random model collapses.
- Pass: at p = 0.005 mean fidelity > 0.85; degradation curve is monotone
  in p; figure saved to `tier3/week25_noise_curve.png`.

### Week 26 — Classical autoencoder baseline

The honest comparison. Tier 2 review item 5.

- Build a classical AE with PyTorch: input ∈ ℝ³² (real + imag parts of
  the 16-dim state vector), bottleneck = 4, output ∈ ℝ³², MSE loss.
- Match the trainable parameter count to the QAE's exactly (16 + small
  classical post-processing on the QAE side ≈ 32 classical parameters
  for the AE — pick widths that hit this).
- Train on the same train/test split, 5 seeds, identical optimizer.
- Compare on (a) reconstruction fidelity (after L2-normalizing the
  classical AE output and treating it as a quantum state) and (b) raw
  MSE.
- Be honest in the writeup. The classical AE can in principle nail this
  because the data lives in a 4-D linear subspace of ℝ³².
- Pass: head-to-head table exists with mean ± std over 5 seeds for both
  models; both numbers reported, neither cherry-picked.

### Week 27 — Capstone + writeup

Pull everything into one runnable artifact + a long-form review.

- `week27_capstone.py`: rerun the whole pipeline (dataset → QAE → noise
  sweep → classical baseline) end-to-end with `seed = 0..4`, save
  `week27_summary.csv` + a 2x2 figure panel
  (`week27_results.png`: loss curve, latent trajectory, noise curve,
  Q-vs-classical bars).
- `TIER3_REVIEW.md`: structured the same way as `TIER2_REVIEW.md`. Sections:
  (1) what the QAE actually demonstrated;
  (2) where barren plateaus showed up and where they didn't;
  (3) honest QAE-vs-classical-AE numbers;
  (4) what would change before tier 4 / publication-quality work.
- Pass: `week27_capstone.py` reproduces every headline number from
  weeks 23–26 within 2 std, writes the CSV + PNG, and asserts each gate.

## Critical files to modify

- `requirements.txt` — already covers `pennylane`, `numpy`, `scipy`, `torch`,
  `matplotlib`, `scikit-learn`. No new top-level deps required (PySCF only
  matters for the chem helper, which is already gated by Tier 2).
- `tier3/utils/__init__.py`, `tier3/utils/states.py`,
  `tier3/utils/qae.py`, `tier3/utils/classical.py` — new shared helpers.
- `tier3/weekN_<topic>.py` (×6) and `tier3/weekN_notes.md` (×6).
- `TIER3_REVIEW.md` (repo root) — capstone writeup.
- `README.md` — append a Tier 3 status block under the existing Tier 3
  section, with checkmarks + commit hashes as work lands (mirrors Tier 2).

## Verification

Tier 3 ends green when, from a fresh checkout:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

for f in tier3/week*.py; do echo "== $f =="; python "$f" || exit 1; done

test -f tier3/week27_capstone.py
test -f tier3/week27_results.png
test -f tier3/week27_summary.csv
test -f TIER3_REVIEW.md
```

Per-week pass conditions are inline above; each script asserts its own gate
so the loop is a green-or-red signal for the whole tier.
