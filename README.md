# QML-Essentials

A self-paced Quantum Machine Learning curriculum for someone with a strong
classical ML and physics background, new to quantum computing. Realistic
target: ~5–6 months to genuine fluency at 5–7 hrs/week.

## Frameworks

- **Primary: [PennyLane](https://pennylane.ai/)** — differentiable-first, clean
  PyTorch / JAX integration, best free structured intro
  ([PennyLane Codebook](https://pennylane.ai/codebook)). Maria Schuld (author
  of the canonical QML textbook) works there.
- **Secondary: [Qiskit](https://qiskit.org/)** — heavier, hardware-focused.
  Required for 3 of the linked resources and for free IBM hardware access.

```bash
python -m venv .venv && source .venv/bin/activate
pip install pennylane qiskit qiskit-machine-learning numpy matplotlib torch
```

## How to use this repo

Each tier grows its own directory (`tier1/`, `tier2/`, `tier3/`) as I work
through it. Every commit is one runnable notebook or script — a checkpoint,
not a proof. The roadmap below is the spine; the linked resources are the
muscle.

---

## Tier 1 — Quantum Literacy (~6–8 weeks)

**Outcome:** Read a quantum circuit fluently, predict measurement outcomes,
explain entanglement and interference without hand-waving.

| Week | Topic | Primary resource | Cross-reference |
|------|-------|------------------|-----------------|
| 1 | Single-qubit gates, Bloch sphere | PennyLane Codebook I.1–I.4 | Hands-On Vol.1 ch.1–2 |
| 2 | Multi-qubit, Bell + GHZ states | PennyLane Codebook I.5–I.7 | Hands-On Vol.1 ch.3 |
| 3 | Measurement, teleportation | PennyLane Codebook I.8–I.10 | Nielsen & Chuang §1.3 |
| 4 | Deutsch–Jozsa, oracles | PennyLane Codebook I.11–I.12 | Hands-On Vol.1 ch.4 |
| 5 | Grover's search | PennyLane Codebook I.13 | krishnakumarsekar list — Grover section |
| 6 | Quantum Fourier Transform | PennyLane Codebook I.14–I.15 | Nielsen & Chuang §5.1 |
| 7 | Phase estimation (skim) | PennyLane Codebook I.16 | — |
| 8 | **Checkpoint:** Reimplement Bell + Grover from scratch in PennyLane, no docs | — | — |

**Reference (lookup, not linear read):** Nielsen & Chuang ch.1–4
**Bookmark:** [krishnakumarsekar/awesome-quantum-machine-learning](https://github.com/krishnakumarsekar/awesome-quantum-machine-learning) for landscape orientation.

---

## Tier 2 — Core QML (~8–12 weeks)

**Outcome:** Train and analyze variational QML models on real (small) datasets;
encounter the barren plateau problem first-hand; benchmark quantum vs classical.

### 2A. Variational Quantum Eigensolver — H2 molecule (weeks 9–11) ✓ `84b45b0`
- PennyLane VQE tutorial as starting point
- Hands-On Vol.1 VQE chapter for theory grounding
- Schuld & Petruccione ch.5

### 2B. QAOA on MaxCut (weeks 12–14) ✓ `84b45b0`, `0c03d9d`
- Implement on a 5–8 node graph
- Watch optimization stall — that *is* the lesson (barren plateaus)
- Cross-implement once in `qiskit-machine-learning` for framework comparison

### 2C. Variational classifier (weeks 15–17) ✓ `0c03d9d`
- 2-class subset of Iris or MNIST
- PennyLane `TorchLayer` for a hybrid model
- Hands-On Vol.1 data-encoding chapter — encoding choice dominates results
- Benchmark against a 1-layer classical MLP with matched parameter count

### 2D. Quantum kernel + classical SVM (weeks 18–20) ✓ `902ac72`
- Reference: [Havlíček et al. 2019](https://arxiv.org/abs/1804.11326)
- Use `qiskit-machine-learning`'s `FidelityQuantumKernel`
- Benchmark vs RBF kernel on the same dataset
- Be honest about whether the quantum kernel actually wins

**Primary text (whole tier):** Schuld & Petruccione, *Machine Learning with Quantum Computers* — the one book that matters.
**Bookmark:** [artix41/awesome-quantum-ml](https://github.com/artix41/awesome-quantum-ml) for paper deep-dives by topic.

**Tier 2 checkpoint** ✓ `902ac72` — week 21 capstone (`tier2/week21_capstone_hybrid.py`)
trains a hybrid model on `digits` 0-vs-1 to 0.986 test accuracy with the
barren-plateau probe rerun at the working qubit count. Cross-tier reflection
in `TIER2_REVIEW.md`.

---

## Tier 3 — Stretch (~6–8 weeks)

**Outcome:** Operate at the edge of current research. Produce one
end-to-end quantum-native artifact (script suite + writeup).

The four candidate projects from the original roadmap were:
- **Quantum autoencoder** — data compression in quantum state space
- **Hybrid model on a real dataset** — choose a problem from
  [MonitSharma's portfolio](https://github.com/MonitSharma/Quantum-Machine-Learning-on-Near-Term-Quantum-Devices)
  (galaxy detection, medical imaging, HEP) and implement end-to-end
- **Tensor network (MPS) methods** — classical simulators that bridge to
  active research; great for understanding what quantum buys you
- **Reproduce one paper** from artix41's awesome list

**Project chosen:** quantum autoencoder ([Romero, Olson, Aspuru-Guzik 2017](https://arxiv.org/abs/1612.02806))
on the H₂ ground-state manifold from the Tier 2 VQE pipeline. Detail in
`TIER3_PLAN.md`. Six weeks (22–27); each is one runnable
`tier3/weekN_*.py` paired with `tier3/weekN_notes.md`.

### 3A. QAE foundations (week 22) ✓
Build the H₂ ground-state dataset, partial-trace utilities, and the
fidelity / Uhlmann-fidelity helpers (`tier3/utils/states.py`).

### 3B. QAE training + generalization (weeks 23–24) ✓
4-qubit, 16-parameter, depth-4 RY+CNOT encoder trained on the Romero
local cost; 5-seed mean-±-std reporting; held-out generalization +
latent-space monotonicity (Spearman-ρ test on $r$ vs PC1 of $\rho_{\text{code}}$).

### 3C. Noise robustness (week 25) ✓
Switch from `default.qubit` → `default.mixed` with depolarizing channel
noise; sweep $p \in \{0, 10^{-3}, 5\cdot10^{-3}, 10^{-2}, 2\cdot10^{-2}\}$;
trained-vs-random baseline at every noise level.

### 3D. Classical autoencoder baselines (week 26) ✓
Linear AE oracle (256 params) and a parameter-matched nonlinear AE for
the honest comparison; both reported, neither cherry-picked.

### Capstone — week 27 ✓
`tier3/week27_capstone.py` reruns weeks 23–26 in one execution, writes
`week27_summary.csv` + a 2×2 figure panel, and asserts every prior
weekly headline within 2σ. Cross-tier writeup in `TIER3_REVIEW.md`.

---

## Resource Index

**Active learning materials (in order of use):**
1. [PennyLane Codebook](https://pennylane.ai/codebook) — Tier 1 spine, free, interactive
2. [Hands-On QML With Python Vol.1 (companion repo)](https://github.com/quantum-machine-learning/Hands-On-Quantum-Machine-Learning-With-Python-Vol-1) — Tier 1–2 structured book
3. Schuld & Petruccione, *Machine Learning with Quantum Computers* — Tier 2 primary text
4. [qiskit-machine-learning](https://github.com/qiskit-community/qiskit-machine-learning) — Tier 2 production library

**Lab inspiration:**
- [MonitSharma/QML on Near-Term Devices](https://github.com/MonitSharma/Quantum-Machine-Learning-on-Near-Term-Quantum-Devices) — Tier 3 project ideas

**Reference / lookup only:**
- [artix41/awesome-quantum-ml](https://github.com/artix41/awesome-quantum-ml) — paper finder by topic
- [krishnakumarsekar/awesome-quantum-machine-learning](https://github.com/krishnakumarsekar/awesome-quantum-machine-learning) — landscape map
- Nielsen & Chuang, *Quantum Computation and Quantum Information* — quantum reference

**Video / community:**
- Maria Schuld's YouTube lectures
- [QHack tutorials](https://github.com/XanaduAI/QHack) for stretch problems
- PennyLane Discussion forum

---

## Pacing reality

5–7 hrs/week for ~5–6 months. Two-job constraint is real; the schedule is
designed for sustainable progress, not heroics. If a tier checkpoint takes an
extra 2 weeks, take it — skipping foundations is how QML becomes cargo-culting.

## Repo evolution

This README is the spine. The repo now ends green across all three
tiers:

- `tier1/` — 8 weeks of literacy: gates, Bell/GHZ, measurement,
  Deutsch–Jozsa, Grover, QFT, QPE, plus the from-scratch checkpoint.
- `tier2/` — 13 weeks of core QML across four sub-projects (VQE,
  QAOA, variational classifier, quantum kernel) + the week-21 hybrid
  capstone. `TIER2_REVIEW.md` is the cross-tier honest writeup.
- `tier3/` — 6 weeks on the H₂-ground-state quantum autoencoder
  (`TIER3_PLAN.md` → weeks 22–27 → `TIER3_REVIEW.md` → capstone CSV +
  figure).
- `requirements.txt` covers every tier's deps.

Each tier directory holds flat `weekN_<topic>.py` + `weekN_notes.md`
files plus a `utils/` submodule for shared helpers. Every week is a
self-checking script: assertion-gated `main()`, exits non-zero if any
gate fails. From a clean checkout:

```bash
pip install -r requirements.txt
for f in tier{1,2,3}/week*.py; do echo "== $f =="; python "$f" || exit 1; done
```

is the green-or-red signal for the whole curriculum.
