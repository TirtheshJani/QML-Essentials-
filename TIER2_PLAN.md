# Tier 2 Plan — Core QML

## Context

Tier 1 (8 weeks of quantum literacy: gates, Bell/GHZ, measurement, Deutsch–Jozsa,
Grover, QFT, QPE, from-scratch checkpoint) is complete and merged to `main`
(commits `6d7e254` … `b62e217`). Each tier-1 week is a single runnable
`tier1/weekN_<topic>.py` script paired with a LaTeX-math `weekN_notes.md`,
ending in assertions that gate the checkpoint. The README (`README.md:51–83`)
already sketches Tier 2 as four sub-projects across weeks 9–20:

- **2A.** VQE on H₂ (weeks 9–11)
- **2B.** QAOA on MaxCut (weeks 12–14)
- **2C.** Variational classifier on Iris/MNIST (weeks 15–17)
- **2D.** Quantum kernel + classical SVM (weeks 18–20)

The tier 2 outcome (`README.md:53`) is to *train and analyze variational QML
models on real (small) datasets, encounter the barren plateau problem
first-hand, and benchmark quantum vs classical*. This plan turns that sketch
into a concrete week-by-week schedule and a directory layout, adds a capstone
hybrid-model week (week 21) plus a `TIER2_REVIEW.md` writeup, and identifies
which tier-1 patterns and helpers carry forward.

## Decisions

- **Layout: hybrid.** Flat `tier2/weekN_*.py` + `weekN_notes.md` files (tier-1
  consistency, chronological scan), plus a `tier2/utils/` module for shared
  helpers that would otherwise be copy-pasted across sub-projects.
- **Checkpoint: capstone week + writeup.** Week 20 finishes 2D; week 21 is a
  standalone integrative hybrid-model run on a non-toy dataset; `TIER2_REVIEW.md`
  is the honest cross-tier writeup on barren plateaus and quantum-vs-classical
  tradeoffs. Total tier length: 13 weeks (9–21).
- **Pass criteria stay assertion-gated** in every weekly script, matching
  `tier1/week8_checkpoint_bell_grover.py:151–154`.

## Directory layout

```
tier2/
  utils/
    __init__.py
    data.py                 # Iris 2-class, MNIST 2-class, train/test split
    classical_baselines.py  # RBF-SVM, 1-layer MLP, HF/FCI references
    plotting.py             # text tables + matplotlib helpers
    barren.py               # gradient-variance sampler over random params
  week9_vqe_h2_setup.py            + week9_notes.md
  week10_vqe_optimization.py       + week10_notes.md
  week11_vqe_dissociation.py       + week11_notes.md
  week12_qaoa_maxcut_setup.py      + week12_notes.md
  week13_qaoa_optimization.py      + week13_notes.md
  week14_qaoa_qiskit_compare.py    + week14_notes.md
  week15_encoding_study.py         + week15_notes.md
  week16_variational_classifier.py + week16_notes.md
  week17_classical_baseline.py     + week17_notes.md
  week18_quantum_kernel_setup.py   + week18_notes.md
  week19_kernel_svm_train.py       + week19_notes.md
  week20_kernel_benchmark.py       + week20_notes.md
  week21_capstone_hybrid.py        + week21_notes.md
TIER2_REVIEW.md
```

Plus a one-line update to `requirements.txt` adding `scikit-learn`, `networkx`,
and `pyscf` (chemistry backend used by `qml.qchem`).

## Reusable patterns from Tier 1

- **Assertion-gated `main()`** with `section()` print helper —
  `tier1/week8_checkpoint_bell_grover.py:110–154`. Copy directly into every
  Tier 2 script.
- **Pauli-expectation tomography** (`qml.expval(qml.PauliZ(0) @ ...)`) used in
  `tier1/week2_bell_and_ghz.py:28–50` is the basis for VQE Hamiltonian
  evaluation and QAOA cost-function evaluation.
- **Phase-oracle / multi-controlled gate idiom** from
  `tier1/week5_grover.py:15–46` informs QAOA's mixer/cost-Hamiltonian
  construction (same X-sandwich pattern for Z-basis bit flips).
- **`qml.draw(circuit)()` introspection** (`tier1/week5_grover.py:142`) for
  sanity-checking ansatz structures before training.

## Week-by-week

### 2A. Variational Quantum Eigensolver — H₂ (weeks 9–11)

- **Week 9 — H₂ Hamiltonian setup.** Build the H₂ molecular Hamiltonian via
  `qml.qchem.molecular_hamiltonian` at the equilibrium bond length 0.74 Å.
  Verify by computing the Hartree–Fock energy with no parameters and
  cross-checking against a known reference (≈ −1.117 Ha). Print the Pauli
  decomposition to make the operator concrete.
  Pass: `abs(E_HF − (−1.117)) < 5e-3`.
- **Week 10 — VQE optimization.** Use `qml.AllSinglesDoubles` (or a
  hardware-efficient ansatz) on 4 qubits with `qml.GradientDescentOptimizer`
  (Adam optional) to drive the energy below FCI − 1 mHa. Log loss curve and
  parameter trajectory.
  Pass: final energy within 1.6 mHa (chemical accuracy) of FCI = −1.1373 Ha.
- **Week 11 — Bond-dissociation curve.** Sweep r ∈ [0.3, 2.5] Å, run VQE at
  each point, plot E(r). Compare against FCI from PySCF; observe the
  static-correlation regime where HF fails. Save a CSV + a text-mode plot.
  Pass: max absolute error vs FCI on the curve < 5 mHa.

### 2B. QAOA on MaxCut (weeks 12–14)

- **Week 12 — Problem and cost Hamiltonian.** Build a 6-node random graph with
  `networkx`. Encode MaxCut as `H_C = ½ Σ_{(i,j)∈E} (1 − Z_i Z_j)`. Compute
  the brute-force optimum on this small graph for ground-truth comparison.
  Pass: bitstring enumeration recovers the known optimum cut value.
- **Week 13 — p-layer QAOA.** Implement the QAOA ansatz (alternating cost +
  X-mixer layers) for p ∈ {1, 2, 3}; outer-loop COBYLA / Adam over
  (γ, β). Track the approximation ratio.
  Pass: p=3 reaches approximation ratio ≥ 0.85 on a 6-node graph.
- **Week 14 — Qiskit cross-implementation + barren-plateau probe.**
  Re-implement the same circuit using `qiskit-machine-learning`'s
  `QAOAAnsatz` and confirm the energy surface matches PennyLane's within
  numerical noise. Use `tier2/utils/barren.py` to measure
  `Var[∂E/∂θ]` at random init for n ∈ {4, 6, 8, 10} qubits and plot the
  exponential decay — this is the README's "watch optimization stall" lesson
  made empirical.
  Pass: variance halves at minimum 2× per qubit added (qualitative trend).

### 2C. Variational classifier (weeks 15–17)

- **Week 15 — Encoding study.** On 2-class Iris (setosa vs versicolor),
  compare angle, amplitude, and IQP encodings: train the *same* 2-layer
  ansatz with each encoding and report test accuracy + decision boundary.
  This makes the README's "encoding choice dominates results" claim concrete.
  Pass: at least one encoding > 0.95 test accuracy; angle vs amplitude differ
  by ≥ 5 pp.
- **Week 16 — Hybrid PyTorch model.** Wrap the best encoding from week 15 in
  `qml.qnn.TorchLayer`, stack with one classical `nn.Linear` head, train end
  to end via `torch.optim.Adam`. Log per-epoch loss/accuracy.
  Pass: ≥ 0.97 test accuracy on Iris-2-class within 50 epochs.
- **Week 17 — Matched-parameter classical baseline.** Build a 1-hidden-layer
  MLP whose parameter count matches the quantum model's trainable θ-count
  exactly, train under the same protocol on the same data. Report a head-to-
  head accuracy + sample-efficiency curve. Be honest in the notes about which
  wins and by how much.
  Pass: head-to-head plot exists with ≥ 5 sample-size points; accuracies
  reproducible across 3 seeds.

### 2D. Quantum kernel + classical SVM (weeks 18–20)

- **Week 18 — Feature map and kernel matrix.** Build a `ZZFeatureMap` (Havlíček
  et al. 2019, `arXiv:1804.11326`) over `qiskit-machine-learning`'s
  `FidelityQuantumKernel`. Compute and plot the Gram matrix on a 40-point
  Iris-2-class subset; verify PSD.
  Pass: minimum eigenvalue of K ≥ −1e-8 (numerical PSD).
- **Week 19 — Train SVM on quantum kernel.** Pass the precomputed kernel to
  `sklearn.svm.SVC(kernel='precomputed')`; report 5-fold CV accuracy.
  Side-by-side compare against RBF on the same split with the same `C`.
  Pass: results table present; both numbers reported, not cherry-picked.
- **Week 20 — Wider benchmark.** Sweep feature-map depth ∈ {1, 2, 3} and
  training set size ∈ {20, 40, 80}. Plot accuracy vs depth and vs n_train for
  both kernels. The deliverable is the *honest table*, not a quantum win.
  Pass: full sweep runs end-to-end and produces both CSV + plot.

### Capstone — week 21

- **`week21_capstone_hybrid.py`.** Train the full hybrid pipeline (best
  encoding from 2C + Torch head + a non-toy dataset such as MNIST 0-vs-1 or
  Fashion-MNIST shirt-vs-pullover, downsampled to 4–8 features via PCA).
  Reuse `tier2/utils/barren.py` to log gradient-variance growth as the model
  scales. Save loss curves, accuracy, and the gradient-variance histogram.
  Pass: model trains to non-trivial accuracy (> 0.85); barren-plateau plot
  produced; final assertion checks both.
- **`TIER2_REVIEW.md`.** Honest cross-tier writeup. Sections:
  (1) what each sub-project actually demonstrated,
  (2) where barren plateaus showed up and at what scale,
  (3) where the quantum approach won, lost, or tied vs the classical
  baseline (concrete numbers from 2C and 2D),
  (4) what I would change before tier 3.

## Critical files to modify

- `requirements.txt` — add `scikit-learn`, `networkx`, `pyscf`.
- `tier2/utils/__init__.py`, `tier2/utils/data.py`,
  `tier2/utils/classical_baselines.py`, `tier2/utils/plotting.py`,
  `tier2/utils/barren.py` — new, shared helpers.
- `tier2/weekN_<topic>.py` (×13) and `tier2/weekN_notes.md` (×13) — one runnable
  + one LaTeX-math notes file per week, identical convention to `tier1/`.
- `TIER2_REVIEW.md` (repo root) — capstone writeup.
- `README.md` — minor: at the end of each tier-2 sub-section, add a checkmark
  + commit hash as work lands (mirrors how tier 1 was tracked).

## Verification

Tier 2 ends green when, from a fresh checkout with the updated
`requirements.txt`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# every weekly script is self-checking via assertions at the bottom
for f in tier2/week*.py; do echo "== $f =="; python "$f" || exit 1; done

# capstone artifacts exist
test -f tier2/week21_capstone_hybrid.py
test -f TIER2_REVIEW.md
```

Per-week pass conditions are listed inline above; each script asserts its own
gate (mirroring `tier1/week8_checkpoint_bell_grover.py:151–154`) so the loop
above is a green-or-red signal for the whole tier.
