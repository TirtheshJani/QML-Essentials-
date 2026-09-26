# QML-Essentials

[![CI](https://github.com/TirtheshJani/QML-Essentials-/actions/workflows/ci.yml/badge.svg)](https://github.com/TirtheshJani/QML-Essentials-/actions/workflows/ci.yml)

A self-paced Quantum Machine Learning curriculum for someone with a strong
classical ML and physics background, new to quantum computing. Realistic
target: ~5–6 months to genuine fluency at 5–7 hrs/week.

## For reviewers

This repo is the curriculum as runnable scripts: circuit basics
(`tier1/`), core QML with VQE, QAOA, a variational classifier and a
quantum kernel (`tier2/`), and a 4-qubit, 16-parameter quantum
autoencoder (QAE) trained on H₂ ground states (`tier3/`). CI runs every
week script. The numbers below are what the scripts print with the
pinned `requirements.txt` on Python 3.11; ± is the population standard
deviation across seeds (numpy's default, ddof = 0). The tier 3 capstone
reruns weeks 23-26 and writes the QAE and classical-AE rows to
`tier3/week27_summary.csv` and `tier3/week27_results.png`.

| tier 3 result (test = 11 held-out bond lengths) | value |
|------|------:|
| QAE local fidelity, all 22 states, 5 seeds (week 23) | 0.9825 ± 0.0214 |
| QAE test reconstruction fidelity, 5 seeds (week 24) | 0.9871 ± 0.0258 |
| Spearman(r, latent PC1), seed 0 (week 24); 1000 untrained encoders give median abs. value 1.000 | -1.000 |
| QAE test local fidelity at depolarizing p = 0.005, 3 seeds (week 25) | 0.8707 ± 0.0090 (20 random encoders: 0.2606 ± 0.1111) |
| Linear classical AE, 256 params, test reconstruction (week 26) | 1.0000 ± 0.0000 |
| Nonlinear classical AE, 136 params, test reconstruction (week 26) | 0.9811 ± 0.0211 |

Four caveats. The 22 H₂ states span only a 2-D subspace, so the linear
AE reconstructs them exactly and the QAE cannot beat it. Because of that
geometry the Spearman row is a property of the data, not of training
(every state is cos t|1100⟩ + sin t|0011⟩, and 999 of 1000 untrained
encoders also pass the |ρ| > 0.9 gate), and the held-out split tests
interpolation along the curve, not generalization. Neither
classical AE is parameter-matched to the QAE's 16 parameters, and the
QAE and the 136-parameter AE tie within one standard deviation; the only
parameter-matched comparison in this repo is tier 2 week 17, where a
13-parameter MLP edges out a 13-parameter hybrid by 2 pp (0.943 vs 0.923
mean test accuracy), within seed noise (3 seeds, 20 test examples, where
one example is 5 pp). In week 23, seed 2 stops at a stationary point at
0.955 and seed 1 (0.958) is still on a slow plateau when training ends
at epoch 200, which misses my pre-registered std < 0.02 gate; the script
reports the miss rather than failing. `TIER3_REVIEW.md` has the full
writeup.

To reproduce tier 3 in one script (about 22 minutes on 4 CPU cores):

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python tier3/week27_capstone.py
```

The loop under "Repo evolution" below runs every week script.

## Frameworks

- **Primary: [PennyLane](https://pennylane.ai/)** — differentiable-first, clean
  PyTorch / JAX integration, best free structured intro
  ([PennyLane Codebook](https://pennylane.ai/codebook)). Maria Schuld (author
  of the canonical QML textbook) works there.
- **Secondary: [Qiskit](https://qiskit.org/)** — heavier, hardware-focused.
  Required for 3 of the linked resources and for free IBM hardware access.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # pinned; tested on Python 3.11
```

## How to use this repo

Each tier grows its own directory (`tier1/`, `tier2/`, `tier3/`) as I work
through it. Each week is one runnable script plus notes: a checkpoint,
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
trash-fidelity cost (a global cost in the terminology of
[Cerezo et al. 2021](https://doi.org/10.1038/s41467-021-21728-w));
5-seed mean-±-std reporting; a held-out split (on this 2-D dataset it
tests interpolation, not generalization) and a latent-space Spearman-ρ
test on $r$ vs PC1 of $\rho_{\text{code}}$, with an untrained-encoder
control that passes the same test.

### 3C. Noise robustness (week 25) ✓
Switch from `default.qubit` → `default.mixed` with depolarizing channel
noise; sweep $p \in \{0, 10^{-3}, 5\cdot10^{-3}, 10^{-2}, 2\cdot10^{-2}\}$;
trained QAE (3 seeds) vs 20 seeded random encoders at every noise level.

### 3D. Classical autoencoder baselines (week 26) ✓
Linear AE oracle (256 params) and a small nonlinear AE (136 params; not
parameter-matched to the QAE's 16) for the honest comparison; both
reported, neither cherry-picked.

### Capstone — week 27 ✓
`tier3/week27_capstone.py` reruns weeks 23–26 in one execution, writes
`week27_summary.csv` + a 2×2 figure panel, and re-checks the weekly
pass gates (the week 23, 24 and 25 fidelity gates loosened by 2σ of the
run). It does not compare against the weekly scripts' numbers, but
every run is seeded, and on the pinned requirements its numbers match
weeks 23–26 to the printed 4 decimals. Cross-tier writeup in
`TIER3_REVIEW.md`.

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
- `requirements.txt` pins every tier's deps (tested on Python 3.11).
- `.github/workflows/ci.yml` runs every week script, one job per tier.

Each tier directory holds flat `weekN_<topic>.py` + `weekN_notes.md`
files plus a `utils/` submodule for shared helpers. Every week from
week 8 on is a self-checking script: assertion-gated `main()`, exits
non-zero if any gate fails; tier 1 weeks 1-7 are print-only
walkthroughs that fail only if they crash. One gate is reported rather
than asserted: week 23's across-seed std gate, which the first full run
missed, prints `MISSED pre-registered gate` (see
`tier3/week23_notes.md`). From a clean checkout:

```bash
pip install -r requirements.txt
for f in tier{1,2,3}/week*.py; do echo "== $f =="; python "$f" || exit 1; done
```

is the green-or-red signal for the whole curriculum. The loop rewrites
`tier2/week20_results.csv` on every run: its `wall_s` column is
wall-clock time, so `git status` shows it as modified afterwards.
