# Week 27 — Tier 3 Capstone

## Purpose

One runnable script that reproduces the four headline numbers from
weeks 23–26 in a single execution, writes a results CSV and a 4-panel
figure, and asserts that none of the prior weeks' results regressed by
more than 2σ on this consolidated run.

## What's reproduced

| Weekly result | Capstone target | Pass margin |
|------|------|------|
| week 23: noiseless mean local fidelity | $\ge 0.95 - 2\sigma$ | yes |
| week 24: test reconstruction fidelity | $\ge 0.85 - 2\sigma$ | yes |
| week 24: $\lvert\rho_{\text{Spearman}}(r, \text{PC1})\rvert$ | $> 0.9$ | yes |
| week 25: $p = 0.005$ test fidelity | $\ge 0.85 - 2\sigma$ | yes |
| week 26: linear AE test fidelity (oracle) | $> 0.95$ | yes |

5 seeds for everything except the noise sweep (3 seeds, since
`default.mixed` is roughly 4× slower than `default.qubit`). The reduced
noise grid is $\{0, 0.005, 0.02\}$ — the three points needed for the
shape of the curve, not the full 5-point sweep from week 25.

## Artifacts

- `tier3/week27_summary.csv` — every (week, model, split, metric) row
  with mean ± std and seed count. Greppable; suitable for a downstream
  notebook or a paper-style table.
- `tier3/week27_results.png` — 2×2 panel:
  - (a) training loss curve (seed 0)
  - (b) latent trajectory $(\text{PC1}, \text{PC2})$ colored by $r$
  - (c) noise robustness errorbars
  - (d) head-to-head reconstruction fidelity bar chart
    (QAE vs matched-classical vs linear-classical oracle)

## Why a capstone, not just a notebook

Two reasons consistent with how Tier 1 and Tier 2 ended:

1. **Self-checking.** Like every other week, this script asserts at the
   end and exits non-zero if anything broke. So `for f in tier3/*.py;
   do python "$f"; done` is a green-or-red signal for the whole tier,
   not just per-week.
2. **Reproducible single-command result.** The capstone is the entry
   point a reader (or a future-me) actually runs. They don't need to
   know which weeks contributed which numbers; they get the same CSV
   and the same figure that the writeup references.

## Wall-clock budget

- Week 23 retrain (5 seeds × 200 epochs × 22 states): ~7 min
- Week 24 retrain (5 seeds × 200 epochs × 11 states): ~3 min
- Week 25 noise sweep (3 levels × 3 seeds × 100 epochs × 11 states on
  `default.mixed`): ~12 min
- Week 26 retrains (5 seeds × 2 classical AEs × 200 epochs): ~30 sec

Total: ~22 min on a laptop CPU. Faster than the sum of weeks 23–26
because we skip the per-week plot generation.

## What this is *not*

This is not a hyperparameter ablation, not a learning-rate sweep, not a
larger-dataset extension. The capstone freezes every choice from the
weekly scripts and re-runs them. Any further investigation belongs in a
follow-up.

The cross-tier writeup (`TIER3_REVIEW.md`) is the place where the
*meaning* of these numbers is discussed. Week 27's job is to produce
the numbers and the figure; week's review interprets them.
