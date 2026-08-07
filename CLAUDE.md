# CLAUDE.md — Research Programme Reference
# Read this file completely before writing any code or running any agent.
# This is the authoritative guide for all engineers and all Claude Code agents.

## This Repository Contains Two Active Research Programmes

### Programme 1: DOCR-MARL Thesis
Multi-Agent RL for Directional Overcurrent Relay coordination with PV penetration.
**Status:** Phase 1 complete. Core runs A–K implemented in `run_core_training.py`.
**Operational reference:** `EXPERIMENT_GUIDE.md` — read that file for all experiment
commands, figure generation, and reproducibility steps. Do not duplicate it here.

### Programme 2: USS v2.0 Research Papers
University Scheduling System v2.0 — 12 targeted publications.
**Status:** 3 paper drafts exist. 9 paper scaffolds to be written.
**Operational reference:** this file and the sections below.

Both programmes share the same Python environment, logging infrastructure,
and caching system. They do NOT share experiment namespaces — log all USS
runs under `results/uss/` and all DOCR runs under `results/` (existing).

---

## USS v2.0 Paper Portfolio

### Papers That Exist (inspect before writing any code for them)

**Correction (2026-08-07): all 12 papers already exist as drafts in `papers/`** —
see `papers/INDEX.md` for the master table. The section below previously claimed
papers 01–05, 07–09, 11 had not been written yet; that was stale. Every draft
still has open `[PLACEHOLDER]` tags (counts below) that map 1:1 to a result file —
treat placeholder count, not paper existence, as the completion signal.

| File | Target Venue | Open Placeholders |
|------|-------------|--------------------|
| `papers/paper_01_hybrid_optimization_JoS.md` | Journal of Scheduling | 7 |
| `papers/paper_02_weight_learning_AIED.md` | AIED | 7 |
| `papers/paper_03_conflict_resolution_AAAI.md` | AAAI | 9 |
| `papers/paper_04_multiobjective_EMO.md` | EMO | 9 |
| `papers/paper_05_whatif_DSS.md` | Decision Support Systems | 16 |
| `papers/paper_06_fairness_FAccT.md` | FAccT | Tables 1–4, Figs 1–4 (need experiments) |
| `papers/paper_07_sustainability_JCP.md` | J. Cleaner Production | 8 |
| `papers/paper_08_blockchain_TSC.md` | IEEE TSC | 7 |
| `papers/paper_09_iot_IoTJ.md` | IEEE IoT Journal | 10 |
| `papers/paper_10_nlp_interface_IUI.md` | IUI '26 | Fig 1 (arch), Tables 2–5, Figs 2–4 (need NLP runs + user study) |
| `papers/paper_11_privacy_PoPETs.md` | PoPETs | 5 |
| `papers/paper_12_full_system_INFORMS.md` | INFORMS Applied Analytics | ~50% — full deployment section, KPI tables |

Before writing code for any track, re-run `grep -c '\[PLACEHOLDER\]' papers/paper_NN_*.md`
to confirm current state — this table will drift again as placeholders get filled.

---

## Paper-First Development Rule

**Do not write USS research code without a corresponding paper section.**

The correct order for any new research module:
1. Open (or write) the paper's Section 3–4 (definitions + methods)
2. Identify every `[PLACEHOLDER]` tag — each maps to one result file
3. Feed the method section to Claude Code → generate the research module
4. Feed the experimental setup (Section 5) → generate the experiment runner
5. Run experiments → fill placeholders
6. Submit paper

A paper is submission-ready when `grep -c '\[PLACEHOLDER\]'` returns 0.

---

## Repository Structure

```
DOCR_Research/
├── CLAUDE.md                        ← THIS FILE
├── EXPERIMENT_GUIDE.md              ← DOCR/MARL thesis operational guide
├── papers/
│   ├── paper_06_fairness_FAccT.md   ← COMPLETE except experiments
│   ├── paper_10_nlp_interface_IUI.md← COMPLETE except NLP runs + study
│   ├── paper_12_full_system_INFORMS.md
│   └── paper_NN_<topic>_<venue>.md  ← scaffold format
├── solver/                          ← USS constraint solver (to create)
│   ├── abstract_solver.py           ← ALL USS solvers implement this
│   └── cp_sat_solver.py
├── research/
│   ├── logging/
│   │   └── experiment_logger.py     ← SEED FILE — all USS runs write here
│   ├── synthetic_data/
│   │   └── uss_syn_generator.py     ← SEED FILE — USS-SYN-3200 dataset
│   ├── equity/                      ← paper_06
│   ├── nlp_interface/               ← paper_10
│   ├── sustainability/              ← paper_07
│   ├── multiobj/                    ← paper_04
│   ├── blockchain_audit/            ← paper_08
│   ├── iot_integration/             ← paper_09
│   └── privacy/                     ← paper_11
├── uss/                             ← production USS v2.0 system (Phase 3+)
│   ├── api/
│   ├── db/
│   ├── scheduler/
│   └── ml/
├── network/                         ← DOCR relay network models (existing)
├── marl/                            ← DOCR MARL agents (existing)
├── baselines/                       ← DOCR baselines (existing)
├── results/
│   ├── core_cache/                  ← DOCR experiment cache (existing)
│   ├── figures/                     ← DOCR figures (existing)
│   └── uss/                         ← USS experiment outputs (to create)
│       ├── track_04_fairness/
│       ├── track_06_nlp/
│       └── ...
└── tests/
    ├── constraints/                 ← USS constraint unit tests
    ├── solver/                      ← USS solver unit tests
    └── research/                    ← USS research module tests
```

---

## 6 USS Seed Files (Create Before Anything Else)

These six files must exist and pass `pytest` before any USS paper experiment runs.
Create them in this order — later files import earlier ones.

**1. `solver/abstract_solver.py`**
Generate from USS_Requirements_Document_v2.0.md Appendix D.
The AbstractSolver interface is append-only after first commit — never
remove or rename a method.

```python
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any

class AbstractSolver(ABC):
    @abstractmethod
    def build_model(self, constraints: List[Dict], objective: Dict) -> None: ...
    @abstractmethod
    def set_warm_start(self, previous_solution: Dict[str, Any]) -> None: ...
    @abstractmethod
    def solve(self, time_limit_seconds: int, callbacks: Optional[List] = None) -> Dict: ...
    @abstractmethod
    def get_status(self) -> str: ...        # FEASIBLE|INFEASIBLE|OPTIMAL|TIME_LIMIT_REACHED
    @abstractmethod
    def get_objective_value(self) -> float: ...
    @abstractmethod
    def get_optimality_gap(self) -> Optional[float]: ...
    @abstractmethod
    def get_conflict_reasons(self) -> List[Dict]: ...
    @abstractmethod
    def supports_incremental(self) -> bool: ...
    @abstractmethod
    def update_model(self, changed_constraints: List[Dict]) -> None: ...
```

**2. `research/logging/experiment_logger.py`**
Writes JSONL (one JSON object per line). Schema is fixed after first
paper experiment — fields are append-only, never rename.

Required fields every USS log entry must contain:
```python
{
    "run_id": str,          # UUID4
    "timestamp": str,       # ISO 8601
    "paper_track": str,     # e.g. "track_04_fairness"
    "experiment_name": str,
    "seed": int,
    "instance_name": str,   # e.g. "USS-SYN-3200"
    "n_courses": int,
    "n_rooms": int,
    "n_time_slots": int,
    "n_students": int,
    "solver_name": str,
    "time_limit_s": int,
    "warm_start": bool,
    "constraint_weights": dict,
    "status": str,
    "solve_time_s": float,
    "objective_value": float,
    "optimality_gap": Optional[float],
    "hard_constraint_satisfaction_rate": float,
    "soft_constraint_satisfaction_by_id": dict,
    "paper_file": str,      # e.g. "papers/paper_06_fairness_FAccT.md"
    "result_table": str,    # e.g. "Table 1"
    # Track-specific optional fields:
    "equity_score": Optional[float],        # track_04
    "carbon_reduction_pct": Optional[float],# track_03
    "pareto_point": Optional[dict],         # track_09
}
```

**3. `research/synthetic_data/uss_syn_generator.py`**
Generate from paper_06 Section 5.1 spec (below). Seed 42 is the
canonical dataset. Never regenerate with a different seed without
bumping `data/uss_syn_3200/VERSION`.

USS-SYN-3200 spec:
- 3,200 students · 480 courses · 12 departments · 210 lecturers
- 95 teaching rooms · 8 buildings · 2 campuses
- 15 weeks · 45 time slots/week (09:00–21:00, Mon–Fri, 1-hr blocks)
- 4 penalty slots: 07:00–08:00 and 19:00–21:00
- Demographic marginals (calibrated to HESA/IPEDS):
  - Disability: 12.3% (mobility, sensory, cognitive, chronic)
  - First-generation: 28.7%
  - Low socioeconomic tier: 31.2%
  - Gender: 52.1% women / 44.6% men / 3.3% non-binary
- Structural confound: first-gen and low-SES students over-enrolled in
  large introductory courses historically assigned to peripheral slots.

Usage: `python research/synthetic_data/uss_syn_generator.py --seed 42 --output data/uss_syn_3200/`

**4. `solver/cp_sat_solver.py`**
Implement using OR-Tools 9.8. Generate from the constraint ID registry
below — one method per constraint family.

**5. `research/equity/eam.py`**
Equity Audit Module. Generate from paper_06 Sections 3 and 4:
- Definition 1–6: USE, DP_gap, EO_viol, IF_ratio, composite E ∈ [0,1]
- Algorithm 1: Disparate Impact Detection
  (chi-square + Bonferroni + Cohen's d + permutation test, 30 seeds)
- Algorithm 2: Equity Remediation Pipeline
  (detect → report → remediate, warm-start, max 5 iterations)

**6. `research/equity/experiments.py`**
Generated from paper_06 Section 5 (experimental setup).
Runs 5 methods × 30 seeds. Writes to `results/uss/track_04_fairness/`.

---

## USS Constraint ID Registry

Every USS constraint implementation MUST begin its docstring with the ID.
The ID is the canonical cross-reference between the requirements document,
the code, and the paper text.

### Lecture Hard Constraints (L-H1 to L-H12)
- L-H1: No Room Conflict
- L-H2: No Lecturer Conflict
- L-H3: No Student Group Conflict
- L-H4: Room Capacity
- L-H5: Room Suitability
- L-H6: Lecture Duration
- L-H7: Pre-assigned Slots
- L-H8: Valid Time Window
- L-H9: Course-Group Mapping
- L-H10: Maximum Daily Load
- L-H11: Prerequisite Chain Ordering
- L-H12: Cross-Campus Time Zone Alignment

### Lecture Soft Constraints (L-S1 to L-S11)
- L-S1: Lecturer Preferences (adaptive weight)
- L-S2: Compact Student Schedule
- L-S3: Balanced Weekly Load
- L-S4: Room Proximity
- L-S5: Lunch Break Preservation
- L-S6: Early/Late Slot Avoidance
- L-S7: Back-to-Back Same Lecturer
- L-S8: Preferred Room
- L-S9: Carbon Footprint Minimization
- L-S10: Equity Distribution ← paper_06 primary constraint
- L-S11: Revenue Optimization

### Exam Hard Constraints (E-H1 to E-H12)
- E-H1: No Student Exam Conflict
- E-H2: No Invigilator Conflict
- E-H3: No Room Conflict
- E-H4: Room Capacity
- E-H5: Exam Duration
- E-H6: Minimum Gap Between Exams
- E-H7: Examiner Availability
- E-H8: Special Needs Accommodation
- E-H9: Maximum Exams Per Day
- E-H10: Exam Period Boundary
- E-H11: Seating Distance Requirement
- E-H12: Invigilator Route Feasibility

### Exam Soft Constraints (E-S1 to E-S10)
- E-S1: Exam Spacing
- E-S2: Room Utilization
- E-S3: Invigilator Load Balance
- E-S4: Large Exam Priority
- E-S5: Departmental Clustering
- E-S6: Time-of-Day Preference
- E-S7: Invigilator Room Familiarity
- E-S8: Minimum Invigilator Ratio
- E-S9: Seating Pattern Optimization
- E-S10: Invigilator Walking Distance

---

## Paper-Code Correspondence

**Correction (2026-08-07):** this table previously mismatched paper numbers to
topics from paper_07 onward (it labeled paper_07 as blockchain, paper_08 as IoT,
paper_09 as privacy, paper_11 as an applied study). The actual filenames in
`papers/` are authoritative — cross-check against `papers/INDEX.md` before
generating code from a "paper_NN" reference.

| Paper | Actual File | Research Module | Key Implementation |
|-------|-------------|-----------------|--------------------|
| paper_01 | paper_01_hybrid_optimization_JoS.md | research/or_optimization/ | ITC2002Benchmarks |
| paper_02 | paper_02_weight_learning_AIED.md | research/adaptive_weights/ | WeightLearner, ABTestFramework |
| paper_03 | paper_03_conflict_resolution_AAAI.md | research/conflict_resolution/ | ImpactAwareResolver |
| paper_04 | paper_04_multiobjective_EMO.md | research/multiobj/ | ParetoFrontierTracker |
| paper_05 | paper_05_whatif_DSS.md | research/decision_support/ | WhatIfSimulator |
| paper_06 | paper_06_fairness_FAccT.md | research/equity/ | EquityAuditModule |
| paper_07 | paper_07_sustainability_JCP.md | research/sustainability/ | CarbonCalculator |
| paper_08 | paper_08_blockchain_TSC.md | research/blockchain_audit/ | ChainAnchorer |
| paper_09 | paper_09_iot_IoTJ.md | research/iot_integration/ | OccupancyPredictor |
| paper_10 | paper_10_nlp_interface_IUI.md | research/nlp_interface/ | BertIntentClassifier, SchedulingNER |
| paper_11 | paper_11_privacy_PoPETs.md | research/privacy/ | DifferentialPrivacyScheduler |
| paper_12 | paper_12_full_system_INFORMS.md | uss/ | full production system |

---

## Publication Priority and Deadlines

**Immediate target (Track 4 — paper_06_fairness_FAccT.md):**
- Paper is 90% written. Needs 6 seed files + experiment runs.
- Target: FAccT 2026. Action: create seed files this week.

**Near-term target (Track 6 — paper_10_nlp_interface_IUI.md):**
- Paper is ~85% written. Needs BERT models fine-tuned and user study.
- Target: IUI '26 (March 2026, Cagliari). Action: assess feasibility.

**Near-term target (Track 8/12 — paper_12_full_system_INFORMS.md):**
- Paper is ~50% written. Needs full USS v2.0 deployment and KPI data.
- Target: INFORMS Applied Analytics. Longer horizon.

---

## Team Ownership (5 Engineers)

| Engineer | Owns | Leads Papers |
|----------|------|-------------|
| Tech Lead | `solver/`, `data/constraints/`, architecture | paper_01, paper_04 |
| Backend Eng | `uss/api/`, `uss/db/`, `research/blockchain_audit/`, `research/privacy/` | paper_08, paper_11 |
| Frontend Eng | `uss/dashboards/`, `research/nlp_interface/` | paper_10 |
| AI/ML Eng | `uss/ml/`, `research/equity/`, `research/sustainability/` | paper_02, paper_07, paper_06 |
| DevOps Eng | `uss/infra/`, `research/iot_integration/`, CI/CD | paper_09 |

DOCR/MARL thesis work is the responsibility of the engineer who ran Phase 1.
New team members do not modify DOCR code without reading EXPERIMENT_GUIDE.md.

---

## Phase Gate Checklist (USS only)

A USS development phase is complete when ALL of the following are true:
- [ ] All deliverables committed to the relevant `research/<track>/` directory
- [ ] All `[PLACEHOLDER]` tags in dependent papers are filled
- [ ] `/code-review` passed on all new code
- [ ] Experiment results committed to `results/uss/<track>/`
- [ ] Paper(s) submitted or staged for submission

---

## Coding Conventions (USS modules)

**Python version:** 3.11 (OR-Tools 9.8 requires this)

**Every constraint implementation starts with its ID:**
```python
def add_room_conflict_constraint(model, assignments):
    # L-H1: No Room Conflict
    ...
```

**Every USS experiment run uses the logger:**
```python
from research.logging.experiment_logger import ExperimentLogger
logger = ExperimentLogger(output_dir="results/uss/track_04_fairness/")
```

**Random seeds from CLI only:**
```python
# Good
parser.add_argument("--seed", type=int, default=42)
rng = np.random.default_rng(args.seed)

# Bad
rng = np.random.default_rng()  # non-reproducible
```

**No comments that describe what code does.** One line for non-obvious WHY.

**Test coverage:** Every constraint has a unit test in `tests/constraints/`.
Every research module has an integration test on 10 students (not 3,200).

**Result files are NOT committed to git.** All experiment outputs go in
`results/uss/` which is in `.gitignore`.

---

## IRB / Ethics

Mandatory before any real student or staff data is collected:
1. IRB application submitted and approved
2. Approval: `compliance/irb_approval.pdf`
3. Data management plan: `compliance/irb_data_management_plan.md`

USS-SYN-3200 is synthetic — IRB not required for it.
Real institutional data requires separate IRB per institution.

---

## What NOT to Do

- Do not add error handling for scenarios that cannot happen inside OR-Tools — trust the solver
- Do not build USS features not required by a specific paper's method section
- Do not rename experiment logger fields after the first paper run — breaks history
- Do not run USS experiments without the logger — retroactive logging is impossible
- Do not modify `solver/abstract_solver.py` interface without major version bump + team sign-off
- Do not collect real demographic data without IRB in `compliance/`
- Do not use `random.random()` unseeded
- Do not modify DOCR/MARL code (`network/`, `marl/`, `baselines/`) without reading EXPERIMENT_GUIDE.md first

---

## Placeholder Protocol

Every `[PLACEHOLDER]` in a paper follows this format:
```
[PLACEHOLDER: Table N — <description>. Required data: <result file path>]
[PLACEHOLDER: Figure N — <description>. Required script: <script path>]
```

When filled, replace with the actual LaTeX table or `\includegraphics` reference.
Commit the result file to `results/uss/<track>/`.

A paper is submission-ready when:
```bash
grep -c '\[PLACEHOLDER\]' papers/paper_NN_*.md
# must return: 0
```
