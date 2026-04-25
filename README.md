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

### 2A. Variational Quantum Eigensolver — H2 molecule (weeks 9–11)
- PennyLane VQE tutorial as starting point
- Hands-On Vol.1 VQE chapter for theory grounding
- Schuld & Petruccione ch.5

### 2B. QAOA on MaxCut (weeks 12–14)
- Implement on a 5–8 node graph
- Watch optimization stall — that *is* the lesson (barren plateaus)
- Cross-implement once in `qiskit-machine-learning` for framework comparison

### 2C. Variational classifier (weeks 15–17)
- 2-class subset of Iris or MNIST
- PennyLane `TorchLayer` for a hybrid model
- Hands-On Vol.1 data-encoding chapter — encoding choice dominates results
- Benchmark against a 1-layer classical MLP with matched parameter count

### 2D. Quantum kernel + classical SVM (weeks 18–20)
- Reference: [Havlíček et al. 2019](https://arxiv.org/abs/1804.11326)
- Use `qiskit-machine-learning`'s `FidelityQuantumKernel`
- Benchmark vs RBF kernel on the same dataset
- Be honest about whether the quantum kernel actually wins

**Primary text (whole tier):** Schuld & Petruccione, *Machine Learning with Quantum Computers* — the one book that matters.
**Bookmark:** [artix41/awesome-quantum-ml](https://github.com/artix41/awesome-quantum-ml) for paper deep-dives by topic.

**Tier 2 checkpoint:** Train a hybrid model on a non-toy dataset and write up
honest observations on barren plateaus and quantum-vs-classical tradeoffs.

---

## Tier 3 — Stretch (~6–8 weeks)

**Outcome:** Operate at the edge of current research. Produce one
publishable-quality artifact (notebook + writeup).

Pick **one** of:
- **Quantum autoencoder** — data compression in quantum state space
- **Hybrid model on a real dataset** — choose a problem from
  [MonitSharma's portfolio](https://github.com/MonitSharma/Quantum-Machine-Learning-on-Near-Term-Quantum-Devices)
  (galaxy detection, medical imaging, HEP) and implement end-to-end
- **Tensor network (MPS) methods** — classical simulators that bridge to
  active research; great for understanding what quantum buys you
- **Reproduce one paper** from artix41's awesome list

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

This README is the spine. As work progresses:
- `tier1/` — one notebook per Codebook section, plus checkpoint scripts
- `tier2/` — VQE, QAOA, classifier, kernel directories
- `tier3/` — single chosen project
- `requirements.txt` added once first code lands
