# CLAUDE.md — Research Programme Reference
# Read this file completely before writing any code or running any agent.
# This is the authoritative guide for all engineers and all Claude Code agents.

## This Repository

**DOCR-MARL Thesis** — Multi-Agent RL for Directional Overcurrent Relay
coordination with PV penetration.
**Status:** Phase 1 complete. Core runs A–K implemented in `run_core_training.py`.
**Operational reference:** `EXPERIMENT_GUIDE.md` — read that file for all experiment
commands, figure generation, reproducibility steps, and how to run the pipeline
in Docker / on a remote server. Do not duplicate it here.

**GPU-vectorized-environment redesign (2026-11-02 → 2026-11-04, additional/opt-in,
not a replacement):** `run_core_training_vectorized.py` and
`run_post_processing_vectorized.py` are a second, parallel implementation of
Cores A–K and figure generation, built as new files specifically so the real,
already-verified single-environment path (`run_core_training.py`,
`marl/marl_agents.py`, `marl/safe_rl.py`, `experiments/training_pipeline.py`,
`core_eval_bridge.py`) stays completely untouched. Status: correctness/schema-
verified at small scale on CPU only — no GPU benchmark has ever been run on
this redesign, and it is NOT used to produce any real thesis figure. Full
detail (scope, cache/figure-directory conventions, what's excluded, and the
standing non-goal) is in `EXPERIMENT_GUIDE.md` §12 — read it before touching
any `*_vectorized.py` file or anything under `marl/vectorized_*.py`,
`network/vectorized_*.py`, `environment/vectorized_*.py`.

**Correction (2026-08-08):** this file previously also documented "Programme 2:
USS v2.0 Research Papers" — a University Scheduling System publication effort
that shared this repository. That programme's code was already reverted in an
earlier pass (it never belonged here — see git history), and its 12 paper
drafts under `papers/` have now been deleted at the user's request. This
repository has exactly one active research programme. If USS v2.0 work
resumes, it belongs in its own repository, not here.

---

## Repository Structure

```
DOCR_Research/
├── CLAUDE.md                        ← THIS FILE
├── EXPERIMENT_GUIDE.md              ← operational guide (commands, figures, Docker/cloud)
├── Dockerfile, docker-compose.yml,  ← containerized training/experimentation
│   .dockerignore, docker/
├── network/                         ← DOCR relay network models, relay physics, topology
├── marl/                            ← MARL agents, environment, comm graph, safe RL, EWC
├── baselines/                       ← GA-SQP / HWOA / HHO / JAYA / EEO metaheuristics
├── evaluation/                      ← statistical analysis, evaluation utilities
├── environment/                     ← dynamic network wrapper, robustness experiments
├── pv_model/                        ← PV inverter / scenario generation
├── deployment/                      ← inference latency, model compression, HILS bridge
├── experiments/, mains/,            ← standalone entry-point scripts (exp_fig*.py, main*.py)
│   phase1_scripts/, phase2_scripts/
├── run_core_training.py             ← the 11 core training runs (A–K)
├── run_thesis_experiments.py        ← main orchestrator (phases 1–9)
├── run_post_processing.py           ← figure generation dispatch
├── run_core_training_vectorized.py  ← additional/opt-in: vectorized-env Cores A–K
│                                       (see EXPERIMENT_GUIDE §12 — never touch the
│                                       real run_core_training.py path from here)
├── run_post_processing_vectorized.py ← additional/opt-in: figures from the
│                                       vectorized path's own "_vectorized" caches only
├── results/
│   ├── core_cache/                  ← Core A–K training cache (cleared 2026-08-08; Cores
│   │                                   have since been re-run — populated with real
│   │                                   results again, see EXPERIMENT_GUIDE §2.2.1).
│   │                                   Also holds the vectorized path's own
│   │                                   "<name>_vectorized" entries, coexisting
│   │                                   with the real ones under distinct names.
│   ├── figures/, logs/, reports/    ← generated output (real pipeline only)
│   └── figures_vectorized/          ← vectorized-path figures ONLY; never the real ones
└── tests/                           ← pytest suite (299 tests as of 2026-11-04, run
                                        `pytest tests/ -q` for the current count)
```

---

## What NOT to Do

- Do not modify DOCR/MARL code (`network/`, `marl/`, `baselines/`) without
  reading `EXPERIMENT_GUIDE.md` first
- Do not use `random.random()` / unseeded RNGs — every experiment script
  takes `--seed` on the CLI and constructs `np.random.default_rng(seed)`
  explicitly (see `EXPERIMENT_GUIDE.md` for the convention)
- Do not add USS v2.0 (or any other unrelated research programme's) code,
  papers, or scaffolding to this repository — see the correction above
- Do not modify `run_core_training.py`, `marl/marl_agents.py`, `marl/safe_rl.py`,
  `experiments/training_pipeline.py`, or `core_eval_bridge.py` when working on
  the vectorized redesign — new work there goes in `*_vectorized.py` files or
  `marl/vectorized_*.py`/`network/vectorized_*.py`/`environment/vectorized_*.py`
  (new, parallel files), per EXPERIMENT_GUIDE §12
- Do not treat a figure under `results/figures_vectorized/` as a real thesis
  result, cite its numbers, or let it overwrite/collide with anything under
  `results/figures/` — see EXPERIMENT_GUIDE §12's non-goal note
