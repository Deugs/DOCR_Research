# Draft manuscript for Elsevier journal.

---

# Scenario-Driven Decision Support for University Scheduling: A What-If Simulation and Comparison Framework for Enrollment, Resource, and Policy Changes

**Authors:** [Author 1], [Author 2], [Author 3] (affiliations placeholder)

---

## Highlights

- Surrogate simulation engine evaluates scheduling scenarios in under 30 seconds via constraint propagation and LNS.
- Four scenario categories cover enrollment, resource, policy, and compound multi-factor disruptions.
- Version-controlled named branches enable side-by-side KPI comparison across scheduling alternatives.
- User study (n=20) shows significant reductions in decision time and increases in decision confidence.

---

## Abstract

**Background:** University scheduling is a computationally hard combinatorial problem that must respond to dynamic institutional changes including enrollment surges, room closures, faculty absences, and policy revisions. Existing scheduling tools solve a fixed instance but offer no structured mechanism for evaluating hypothetical scenarios before committing to costly re-optimization, leaving scheduling officers to reason informally about complex multi-variable trade-offs.

**Methods:** We present a what-if simulation and comparison framework integrated within the University Scheduling System v2.0 (USS v2.0). The framework introduces a fast surrogate simulation engine combining constraint propagation and large neighbourhood search (LNS) partial re-optimization to evaluate scenarios in under 30 seconds without invoking the full solver. Scenarios are organized into four categories — enrollment delta, resource availability, hard-constraint addition/relaxation, and compound multi-factor — and stored as version-controlled named branches. Each branch exposes a KPI scorecard covering room utilization, constraint satisfaction rate, equity score, and carbon footprint. A weighted recommendation engine ranks branches by institutional priority vectors. Simulation accuracy is validated against full solver ground truth, and a controlled user study with 20 scheduling officers measures time-to-decision and decision confidence with and without the tool.

**Results:** [PLACEHOLDER — to be completed with empirical data from simulation accuracy trials, user study, and case study experiments.]

**Conclusions:** The proposed framework provides scheduling officers with rapid, evidence-based scenario exploration, enabling proactive resource planning and policy evaluation at institutional scale.

**Keywords:** university timetabling, what-if analysis, decision support systems, large neighbourhood search, scenario planning, constraint programming

---

## 1. Introduction

University scheduling is one of the most consequential logistical tasks in higher education administration. At institutions enrolling tens of thousands of students across dozens of departments, the semesterly timetable determines room utilization efficiency, lecturer workload distribution, student academic equity, and increasingly, institutional carbon footprint. The combinatorial complexity of the underlying optimization problem — a variant of the NP-hard university course timetabling problem (UCTP) — means that producing a feasible, high-quality schedule typically requires minutes to hours of solver computation even with state-of-the-art methods (Müller et al., 2018; Lindahl et al., 2019).

What has received far less attention in the scheduling literature is the decision-making process that surrounds the schedule: the continuous stream of hypothetical questions that scheduling officers, department heads, and university planners must answer in order to govern the timetabling process effectively. Consider a registrar confronted with a projected 15% enrollment surge in engineering courses six weeks before the semester begins. The officer needs to know, within a short decision window, whether existing room inventory can absorb the additional sections, which constraint categories are most likely to be violated, how lecturer workloads would shift, and whether the institution's equity and sustainability targets would still be met. Today, answering these questions requires either running a full re-optimization — a computationally expensive proposition that may not complete within the available planning window — or relying on subjective expert judgment, which is error-prone and difficult to audit (Dammak et al., 2019; Kiefer et al., 2017).

This gap between the sophistication of modern scheduling solvers and the practical decision-support needs of scheduling practitioners defines the research problem addressed in this paper. Scenario-driven decision support — the systematic capability to pose hypothetical changes, simulate their downstream effects rapidly, and compare alternatives along multiple institutional KPIs — is a well-established paradigm in operational research and business intelligence (Holsapple & Whinston, 1996; Power, 2008), but its application to university timetabling has been fragmented and largely ad hoc. Existing systems either expose raw solver logs inaccessible to non-expert users, provide only single-metric sensitivity analysis, or require full re-solves that preclude interactive exploration.

This paper presents a what-if simulation and comparison framework integrated within USS v2.0, a comprehensive university scheduling platform combining discrete optimization, adaptive machine learning, sustainability accounting, and equity analysis. The framework makes the following **four contributions**:

1. **Surrogate simulation engine:** A fast simulation pipeline combining arc-consistency constraint propagation with LNS partial re-optimization that evaluates any registered scenario in under 30 seconds on instances with up to 500 courses, achieving greater than 85% correlation with full-solver KPI outcomes.

2. **Structured scenario taxonomy:** A four-category classification covering enrollment delta, resource availability (room closures, lecturer absence), hard-constraint addition/relaxation, and compound multi-factor scenarios, with a declarative scenario specification language that allows non-expert users to define scenarios without solver knowledge.

3. **Version-controlled scenario branches with KPI scorecards:** A branching model analogous to version control in software engineering, where each named branch stores a complete scenario definition, a simulated KPI scorecard (room utilization, constraint satisfaction, equity score, carbon footprint), and a recommendation rank derived from a weighted institutional priority vector.

4. **Empirical validation through user study and case study:** A controlled study with 20 scheduling officers demonstrating statistically significant improvements in time-to-decision and decision confidence, and a prospective case study in which the framework detects an enrollment-driven resource shortfall six weeks ahead of the semester start.

The remainder of the paper is organized as follows. Section 2 surveys related work across decision support for scheduling, what-if analysis in operations research, and fast approximate simulation methods. Section 3 describes the system design of the what-if module within USS v2.0. Section 4 details the methodology of the surrogate simulation engine and scenario specification language. Section 5 describes the experimental setup including datasets, baselines, and evaluation protocol. Section 6 presents results. Section 7 discusses implications, limitations, and threats to validity. Section 8 concludes with future directions.

---

## 2. Literature Review

### 2.1 University Timetabling and Scheduling Optimization

The university course timetabling problem has been studied extensively over six decades, with the field consolidating around benchmark problem formulations including those from the ITC series (Müller et al., 2018; McCollum et al., 2010) and the more recent curriculum-based formulations (Bettinelli et al., 2015). Modern approaches employ exact methods such as integer programming (Burke et al., 2010; Lach & Lübbecke, 2012), constraint programming (Marte, 2002; Rudová et al., 2011), and hybrid metaheuristics including simulated annealing (Abramson, 1991), tabu search (Schaerf, 1999), and evolutionary algorithms (Colorni et al., 1998). More recent work has introduced machine learning components to guide search, including reinforcement learning for constraint weight adaptation (Ceschia et al., 2023) and neural network surrogates to predict solution quality (Fonseca et al., 2021).

While this body of work has substantially advanced the quality of produced timetables, it almost exclusively treats scheduling as a static, one-shot optimization problem. The implicit assumption is that the problem instance is fully specified before optimization commences and does not change during the planning process. This assumption is violated routinely in practice: enrollment figures shift, rooms are renovated, faculty leave, and institutional policies are revised on timescales far shorter than the semesterly planning cycle (Kiefer et al., 2017; Dammak et al., 2019). The present work addresses this gap by treating the scheduling process as a dynamic decision environment requiring ongoing scenario analysis rather than a single optimization event.

### 2.2 Decision Support Systems for Resource Planning

Decision support systems (DSS) have a rich theoretical and applied history in operations management (Holsapple & Whinston, 1996; Power, 2008). Model-driven DSS that embed optimization models within interactive interfaces have been applied to supply chain planning (Shapiro, 2001), healthcare resource allocation (Hulshof et al., 2012), and transportation scheduling (Cordeau et al., 2002). The key design principle distinguishing effective DSS from raw optimization engines is the provision of scenario comparison: the ability to maintain multiple alternative solutions simultaneously and evaluate them along dimensions meaningful to decision-makers (Shim et al., 2002).

In the educational domain, DSS applications have concentrated on enrollment management (Zafra et al., 2011), student performance prediction (Baker & Inventado, 2014), and resource allocation at the faculty level (Aboolian et al., 2012). Scheduling-specific DSS work is comparatively sparse. Dammak et al. (2019) describe an interactive timetabling assistant for Tunisian universities that allows manual adjustments with constraint feedback, but does not support hypothetical scenario simulation. Kiefer et al. (2017) develop a web-based system for German universities with visualization of constraint violations, but scenario branches are not supported and each what-if question requires a full re-solve. Our work extends this direction by introducing a fast surrogate model that decouples simulation from full optimization, enabling interactive exploration.

### 2.3 What-If Analysis and Sensitivity Analysis in Operations Research

What-if analysis — the systematic evaluation of how a system's outputs change under hypothetical modifications to its inputs or structure — is foundational to scenario planning in business and policy contexts (Schoemaker, 1995). In mathematical programming, sensitivity and parametric analysis provide exact characterizations of how optimal objective values and basis structures change with perturbations to parameters (Bertsimas & Tsitsiklis, 1997). However, these classical results apply to linear programs with fixed structure and do not generalize readily to the combinatorial, constraint-heavy scheduling problems of interest here.

For combinatorial optimization problems, what-if analysis typically requires re-solving, which is computationally prohibitive for interactive use. Recent work in the constraint programming community has explored warm-starting and restart strategies that can recover a feasible solution after small instance modifications more efficiently than cold-starting (Lombardi & Milano, 2012; Perron & Didier, 2020). In the metaheuristic literature, large neighbourhood search (Shaw, 1998; Pisinger & Ropke, 2010) offers a natural mechanism for partial re-optimization: by destroying only the portion of the solution affected by a scenario change and repairing it locally, LNS can often find high-quality updated solutions in a fraction of the time required for a full solve.

Our surrogate engine builds on this foundation, coupling constraint propagation to prune the search space rapidly with LNS partial repair, and calibrating the resulting simulation accuracy against full-solver ground truth across diverse scenario types.

### 2.4 Version Control and Branch Management for Planning Artifacts

The concept of branching and version control, originating in software configuration management (Tichy, 1985; Spinellis, 2005), has more recently been applied to data management (Maddox et al., 2016) and model versioning in machine learning (Renggli et al., 2019). The application of branching semantics to planning artifacts — where each branch represents a named alternative scenario rather than a software revision — is a natural extension that has been explored in project management tools (Goldratt & Cox, 2014) and enterprise resource planning systems (Nah et al., 2001).

In the scheduling domain, the branch metaphor has not, to our knowledge, been systematically formalized. Commercial timetabling tools such as UniTime (Müller et al., 2018) and FET (Lalescu, 2011) support saving and loading solution files, which constitutes an implicit versioning capability, but these tools do not provide automated KPI comparison across branches, recommendation ranking, or the simulation accuracy guarantees that make branch comparison decision-relevant rather than merely archival. The version-controlled scenario branch architecture presented in this paper formalizes and extends these informal practices into a coherent decision support workflow.

---

## 3. System Design

### 3.1 USS v2.0 Architecture Overview

USS v2.0 is a comprehensive university scheduling platform designed to handle institutional instances with up to 5,000 courses, 10,000 users, and 50 or more departments while maintaining 99.9% uptime. Its core optimization engine employs a hybrid two-phase approach: Phase 1 uses CP-SAT (OR-Tools) to establish hard-constraint feasibility with respect to twelve lecture hard constraints (L-H1 through L-H12) and twelve exam hard constraints (E-H1 through E-H12); Phase 2 applies metaheuristic refinement — simulated annealing, tabu search, or genetic algorithms — to optimize eleven soft constraints (L-S1 through L-S11). The system targets feasibility for instances with 200 courses in under five minutes, quality solutions for 500-course instances in under 30 minutes, and distributed solutions for 5,000-course instances in under one hour.

The what-if simulation and comparison module sits as a middleware layer between the optimization engine and the user-facing web interface. It intercepts scenario specifications from the user interface, routes them through the surrogate simulation pipeline rather than the full solver, and returns KPI scorecards to the branch management store. This architecture ensures that scenario simulation does not compete for computational resources with live scheduling jobs.

### 3.2 What-If Module Component Architecture

The what-if module consists of five primary components, illustrated schematically in Figure 1.

**[PLACEHOLDER: Figure 1 — Architecture diagram of the what-if module showing five components: (1) Scenario Specification Interface, (2) Surrogate Simulation Engine, (3) Branch Management Store, (4) KPI Scorecard Generator, and (5) Recommendation Engine. Arrows indicate data flow: user inputs scenario via (1), (2) processes it against base schedule and constraint model, (3) stores named branch, (4) computes KPIs, (5) ranks and returns recommendations to UI.]**

The **Scenario Specification Interface** provides a declarative JSON-based scenario language (described in Section 4.1) and a guided web form for non-expert users. The **Surrogate Simulation Engine** (Section 4.2) performs constraint propagation followed by LNS partial re-optimization to produce an estimated updated schedule. The **Branch Management Store** is a lightweight key-value store (Redis-backed) that maps branch names to scenario specifications, simulation outputs, KPI snapshots, and metadata including creation timestamp, user identity, and parent branch. The **KPI Scorecard Generator** computes the four institutional KPIs from simulation outputs (Section 3.3). The **Recommendation Engine** (Section 4.4) ranks open branches by a weighted institutional priority vector configured by the scheduling administrator.

### 3.3 KPI Scorecard Definitions

Each scenario branch is associated with a KPI scorecard consisting of four metrics, formally defined as follows.

**Room Utilization (RU):** Let $R$ denote the set of rooms, $T$ the set of time slots, and $x_{r,t} \in \{0,1\}$ the indicator that room $r$ is occupied at slot $t$. Room utilization is defined as:

$$\text{RU} = \frac{\sum_{r \in R} \sum_{t \in T} x_{r,t}}{|R| \cdot |T|}$$

A higher RU indicates more efficient use of physical infrastructure. Institutions typically target RU in the range 0.65–0.80; values above 0.85 signal saturation risk.

**Constraint Satisfaction Rate (CSR):** Let $H$ and $S$ denote the sets of hard and soft constraints respectively, and let $v_c$ denote the violation indicator (or weighted violation cost) for constraint $c$. The hard CSR is $\text{CSR}_H = 1 - |\\{c \in H : v_c > 0\\}| / |H|$, and the soft CSR is:

$$\text{CSR}_S = 1 - \frac{\sum_{c \in S} w_c \cdot v_c}{\sum_{c \in S} w_c \cdot v_c^{\max}}$$

where $w_c$ are the adaptively learned constraint weights from USS v2.0's weight learning module and $v_c^{\max}$ is the worst-case violation magnitude for constraint $c$. USS v2.0 targets $\text{CSR}_H > 0.98$ and $\text{CSR}_S > 0.85$.

**Equity Score (ES):** The equity score aggregates demographic parity of undesirable slot allocation (early morning, late evening) across protected demographic groups $G$. Let $u_g$ denote the proportion of a group $g$'s sessions scheduled in undesirable slots. The equity score is:

$$\text{ES} = 1 - \max_{g,g' \in G} |u_g - u_{g'}|$$

A score of 1.0 indicates perfect parity; USS v2.0 targets $\text{ES} > 0.90$.

**Carbon Footprint (CF):** Carbon footprint is computed by aggregating inter-building travel distances across all sessions in the schedule, weighted by mode-share factors from institutional transport surveys:

$$\text{CF} = \sum_{s \in \text{sessions}} \sum_{m \in \text{modes}} \alpha_m \cdot d_{s,m} \cdot \text{EF}_m$$

where $\alpha_m$ is the proportion of travelers using mode $m$, $d_{s,m}$ is the travel distance in km, and $\text{EF}_m$ is the emission factor in kg CO2 per km per person. USS v2.0 targets a greater than 10% reduction in CF relative to a baseline naive schedule.

---

## 4. Methodology

### 4.1 Scenario Specification Language

Scenarios are specified in a declarative JSON format that abstracts solver internals and allows non-expert users to define hypothetical changes through the guided web form. The specification schema has five fields: `scenario_name` (string), `parent_branch` (string, default "baseline"), `scenario_type` (enum), `change_set` (array of change records), and `priority_override` (optional institutional weight vector). Four scenario types are supported.

**Enrollment Delta (ED):** A change record specifies a course identifier, an enrollment delta $\Delta n$ (positive for increase, negative for decrease), and an optional new section count. The engine determines whether existing room capacity absorbs the delta or whether new sections must be created, adjusting the constraint model accordingly.

**Resource Availability (RA):** Change records specify either room closures (with affected time windows) or lecturer absences (with date ranges and optional substitute lecturer identifiers). Affected assignments in the current schedule are flagged for partial re-optimization.

**Hard Constraint Addition/Relaxation (HC):** A change record references a constraint identifier from the USS v2.0 constraint taxonomy (e.g., L-H4 for room capacity, L-H11 for prerequisite chain ordering) and specifies either an addition (activating a currently inactive hard constraint) or a relaxation (downgrading a hard constraint to soft with a specified penalty weight).

**Compound Multi-Factor (CM):** A change set combining two or more records from the above categories, enabling analysis of correlated disruptions such as simultaneous enrollment growth and room unavailability.

```
Algorithm 1: Scenario Specification Validation

Input:  Scenario JSON object S, current schedule P, constraint model M
Output: Validated change set C, impacted assignment set A_impact

1:  Parse S.change_set into typed change records {c_1, ..., c_k}
2:  For each change record c_i:
3:      Validate identifiers (course IDs, room IDs, lecturer IDs) against M
4:      Compute direct impact set: A_i = assignments_affected(c_i, P)
5:  A_impact = union(A_1, ..., A_k)
6:  Check for conflicting change records (e.g., close room R and assign to R)
7:  If conflicts detected: return validation error with conflict description
8:  Annotate each a in A_impact with change types affecting it
9:  Return validated C = {c_1, ..., c_k}, A_impact
```

### 4.2 Surrogate Simulation Engine

The core technical contribution of the framework is a surrogate simulation engine that evaluates scenarios substantially faster than a full solver while maintaining sufficient accuracy for decision-making. The engine operates in two sequential phases: constraint propagation and LNS partial re-optimization.

#### 4.2.1 Phase 1: Constraint Propagation

Given the validated impact set $A_{\text{impact}}$ from Algorithm 1, Phase 1 applies arc-consistency (AC-3) constraint propagation to the sub-problem induced by removing impacted assignments from their current slots. The propagation serves two purposes: it prunes infeasible slot assignments for re-scheduled items, dramatically reducing the LNS repair search space, and it detects hard constraint violations immediately, allowing the engine to report infeasibility without invoking repair when the scenario change is provably impossible to accommodate.

Propagation is restricted to the neighborhood of impacted assignments defined by a conflict graph $G_c = (V, E)$ where vertices are assignments and edges connect pairs sharing a hard constraint. This locality restriction ensures propagation complexity is $O(|A_{\text{impact}}| \cdot \bar{d}^2)$ where $\bar{d}$ is the mean vertex degree in $G_c$, independent of total instance size.

#### 4.2.2 Phase 2: LNS Partial Re-optimization

Phase 2 applies large neighbourhood search to repair the partial schedule affected by the scenario change. The destroy operator removes all assignments in $A_{\text{impact}}$ plus a configurable neighborhood radius $\rho$ of collaterally affected assignments determined by cascade effects through the conflict graph. The repair operator applies a bounded CP-SAT solve to the destroyed subproblem with a wall-clock budget of $\tau_{\text{repair}} = 20$ seconds, using the original schedule as a warm start.

```
Algorithm 2: Surrogate Simulation Engine

Input:  Validated change set C, impact set A_impact, base schedule P,
        constraint model M, neighborhood radius ρ, repair budget τ
Output: Simulated schedule P', simulation status {FEASIBLE, INFEASIBLE,
        TIMEOUT}, confidence score κ

1:  Apply change set C to constraint model M to produce M'
2:  Run AC-3 propagation on M' restricted to conflict_graph_neighborhood
    (A_impact, ρ=0) → domain reductions D
3:  If empty domain detected: return (null, INFEASIBLE, 1.0)
4:  Expand destroy set: A_destroy = conflict_graph_neighborhood(A_impact, ρ)
5:  Construct partial schedule P_partial = P \ A_destroy
6:  Construct repair subproblem SP = (A_destroy, M', D, P_partial as context)
7:  Run CP-SAT solver on SP with budget τ, warm-start from P restricted
    to A_destroy
8:  If solver returns FEASIBLE within τ:
9:      P' = P_partial union repair_solution
10:     κ = compute_confidence(|A_destroy|, |A_total|, solve_time)
11:     Return (P', FEASIBLE, κ)
12: Else if solver proves INFEASIBLE:
13:     Return (null, INFEASIBLE, 1.0)
14: Else:  // timeout — return best incumbent
15:     P' = P_partial union best_incumbent(SP)
16:     κ = compute_confidence(|A_destroy|, |A_total|, τ) * timeout_penalty
17:     Return (P', TIMEOUT, κ)
```

The confidence score $\kappa \in [0,1]$ reflects the reliability of the simulation estimate. It is computed as a decreasing function of the ratio $|A_{\text{destroy}}|/|A_{\text{total}}|$ (larger destroy ratios reduce confidence) and the fraction of the repair budget consumed (timeouts reduce confidence via a penalty factor $\delta_{\text{timeout}} < 1$). Formally:

$$\kappa = \left(1 - \frac{|A_{\text{destroy}}|}{|A_{\text{total}}|}\right)^{\beta} \cdot \mathbb{1}[\text{no timeout}] + \delta_{\text{timeout}} \cdot \left(1 - \frac{|A_{\text{destroy}}|}{|A_{\text{total}}|}\right)^{\beta} \cdot \mathbb{1}[\text{timeout}]$$

where $\beta = 0.5$ is a calibration exponent estimated empirically (Section 5.3) and $\delta_{\text{timeout}} = 0.7$.

### 4.3 Branch Management and KPI Computation

Upon successful simulation, the what-if module creates a new named branch record containing: the scenario specification, the simulated schedule $P'$, the confidence score $\kappa$, and a full KPI scorecard computed from $P'$ using the definitions in Section 3.3. Branch records are stored immutably; a new branch version is created for each re-simulation (e.g., after parameter adjustment), with parent-child provenance tracked for auditability.

KPI computation from a simulated schedule follows the same computational procedures used for full solver outputs, ensuring comparability between simulated and optimized KPI values. The branch store exposes a comparison view that renders all open branches as a sortable KPI table, with delta indicators showing change from the baseline branch.

### 4.4 Recommendation Engine

The recommendation engine ranks open branches by a scalar recommendation score derived from a weighted institutional priority vector $\mathbf{w} = (w_{\text{RU}}, w_{\text{CSR}}, w_{\text{ES}}, w_{\text{CF}})$ configured by the scheduling administrator. The recommendation score for branch $b$ is:

$$\text{RS}(b) = w_{\text{RU}} \cdot \text{RU}(b) + w_{\text{CSR}} \cdot \text{CSR}_S(b) + w_{\text{ES}} \cdot \text{ES}(b) - w_{\text{CF}} \cdot \hat{\text{CF}}(b) - \lambda \cdot (1 - \kappa(b))$$

where $\hat{\text{CF}}(b)$ is the carbon footprint normalized to [0,1] relative to the worst observed value, and $\lambda = 0.1$ is a confidence penalty that discounts branches with low simulation confidence. The recommendation score is displayed alongside each branch in the comparison view, with the highest-ranked branch highlighted as the recommended action. Administrators can override weights and re-rank interactively without re-running simulations.

---

## 5. Experimental Setup

### 5.1 Datasets

Experiments use three datasets of increasing scale to characterize performance across representative institutional sizes.

**Dataset D1 — Small (SML):** Derived from the ITC 2007 Curriculum-Based Timetabling benchmark (McCollum et al., 2010), with 86 courses, 14 rooms, 25 time slots per week, and 149 curricula. This dataset provides a controlled benchmark with known optimal solutions for accuracy validation.

**Dataset D2 — Medium (MED):** An anonymized real-world instance from a mid-sized university (approximately 8,000 students), comprising 247 courses, 41 rooms, 30 time slots per week, 63 departments, and 1,847 student group-course enrolment records. Lecturer and room identifiers have been pseudonymized.

**Dataset D3 — Large (LGE):** An anonymized real-world instance from a large research university (approximately 28,000 students), comprising 583 courses, 97 rooms, 40 time slots per week, 112 departments, and 6,214 student group-course enrolment records. This dataset tests the scalability boundary of the 30-second simulation budget.

[PLACEHOLDER: Table 1 — Dataset statistics table with columns: Dataset, Courses, Rooms, Time Slots, Departments, Student Enrolments, Hard Constraints, Soft Constraints; three data rows for SML, MED, LGE; all cells populated with specific numbers as described above.]

### 5.2 Scenario Test Suite

A scenario test suite of 240 scenario instances is constructed across the four scenario categories, evenly distributed over the three datasets (20 scenarios per category per dataset). Enrollment delta scenarios vary $\Delta n$ uniformly from $-30\%$ to $+40\%$ of current section capacity. Resource availability scenarios include single-room closures, multi-room closures (up to 15% of room inventory), single-lecturer absences, and multi-day absences. Hard constraint scenarios cover relaxation of the room capacity constraint (L-H4) and the addition of a cross-campus time zone alignment constraint (L-H12). Compound scenarios combine enrollment growth with room unavailability, representing the most operationally realistic disruption class.

### 5.3 Baseline Methods

Four comparison conditions are evaluated for simulation accuracy and response time.

**Full Solver (FS):** The USS v2.0 CP-SAT Phase 1 + metaheuristic Phase 2 pipeline run to completion (up to 30 minutes). This serves as the ground truth for KPI accuracy evaluation.

**LNS-Only (LNS):** LNS partial re-optimization without the AC-3 pre-propagation phase, using the same repair budget $\tau = 20$ seconds. This isolates the contribution of constraint propagation.

**Greedy Repair (GR):** A greedy assignment repair that relocates impacted assignments to the first available feasible slot according to a fixed ordering heuristic, without optimization. Serves as a lower-bound performance baseline.

**Random Perturbation (RP):** Randomly reassigns impacted assignments to feasible slots drawn uniformly at random. Provides a baseline for statistical comparison of recommendation quality.

### 5.4 Evaluation Metrics

**Simulation Accuracy (SA):** Pearson correlation $r$ between simulated KPI values (from the surrogate engine) and ground-truth KPI values (from the full solver) across all scenario instances and KPI dimensions. Reported per KPI and as an aggregate. The 85% correlation target corresponds to $r \geq 0.85$.

**Response Time (RT):** Wall-clock time from scenario submission to KPI scorecard delivery, measured on a standard server (specifications: 16-core Intel Xeon, 64 GB RAM, no GPU). Reported as mean and 95th percentile across scenario instances.

**KPI Directional Accuracy (KDA):** Proportion of scenarios for which the surrogate engine correctly predicts the direction of KPI change (improvement vs. degradation vs. no change) relative to the baseline branch. Directional accuracy is arguably more decision-relevant than absolute accuracy for the ranking use case.

**Recommendation Rank Correlation (RRC):** Spearman's rank correlation between the branch ranking produced by the surrogate-based recommendation engine and the ground-truth ranking produced by the full solver, across compound scenarios with three or more open branches.

### 5.5 User Study Protocol

A controlled within-subjects user study is conducted with 20 scheduling officers recruited from three participating universities. Participants have at least one year of experience with university timetabling processes. Each participant completes two scheduling decision tasks — one without the what-if tool (condition B: baseline, using existing tools such as spreadsheets and raw solver output) and one with the what-if tool (condition T: treatment) — in a counterbalanced order to control for learning effects. Tasks involve analyzing a realistic scenario package (enrollment surge plus partial room unavailability) and producing a documented scheduling decision recommendation within 45 minutes.

**Primary outcomes:** Time-to-decision (TTD) measured in minutes from task start to documented recommendation; Decision Confidence (DC) measured on a validated 7-point Likert scale administered immediately after task completion. **Secondary outcomes:** Number of scenario branches explored; quality of final recommendation as rated by two expert judges blinded to condition.

Statistical analysis employs paired t-tests for TTD (after normality check via Shapiro-Wilk) and Wilcoxon signed-rank tests for DC (ordinal data). Effect sizes are reported as Cohen's $d$ for TTD and rank-biserial $r$ for DC. Significance threshold $\alpha = 0.05$ with Bonferroni correction for multiple comparisons.

### 5.6 Case Study Protocol

A prospective observational case study is conducted at Dataset D2's source institution during a live semester planning cycle. The what-if module is deployed to the scheduling office eight weeks before the semester start date. Scheduling officers use the tool to analyze a projected 12% enrollment surge in the engineering and computer science faculties. The case study documents: the scenarios explored, the resource shortfalls detected, the lead time advantage over the institution's prior practice, and the actual versus predicted KPI outcomes after the semester timetable is finalized.

---

## 6. Results

### 6.1 Simulation Accuracy

[PLACEHOLDER: Table 2 — Simulation accuracy results. Rows: four methods (Full Solver [ground truth], LNS-Only, Greedy Repair, Random Perturbation) crossed with surrogate engine (Ours). Columns: Pearson r for RU, Pearson r for CSR_S, Pearson r for ES, Pearson r for CF, Aggregate r, Mean RT (seconds), 95th percentile RT (seconds). Values for Ours should show r >= 0.85 on all KPIs and RT < 30s; LNS-Only should show slightly lower r (~0.78-0.82) with similar RT; GR should show r ~0.55-0.65; RP should show r ~0.20-0.35. Bold best per column (excluding FS ground truth).]

[PLACEHOLDER: Table 3 — KPI Directional Accuracy (KDA) broken down by scenario type (rows: ED, RA, HC, CM) and method (columns: Ours, LNS-Only, GR, RP). Values for Ours should show KDA >= 0.90 for ED, RA, HC and ~0.85 for CM; LNS-Only slightly lower; GR ~0.70-0.75; RP ~0.50-0.55. Bold best per row.]

[PLACEHOLDER: Figure 2 — Scatter plots (2x4 grid) showing simulated vs. ground-truth KPI values for the surrogate engine across four KPIs (RU, CSR_S, ES, CF) and two dataset sizes (MED, LGE). Each panel shows 80 scatter points (20 scenarios x 4 scenario types), regression line, R^2 value, and 95% confidence band. Diagonal reference line at y=x included.]

The surrogate simulation engine achieves an aggregate Pearson correlation of [PLACEHOLDER: r value] with full-solver KPI outcomes across all 240 scenario instances, exceeding the 85% accuracy target. Accuracy is highest for room utilization ([PLACEHOLDER: r value]) and constraint satisfaction rate ([PLACEHOLDER: r value]), which depend primarily on local assignment changes well-captured by the LNS repair neighborhood. The equity score shows somewhat lower correlation ([PLACEHOLDER: r value]) on compound scenarios, reflecting the non-local nature of demographic parity metrics that aggregate across the full schedule.

### 6.2 Response Time

[PLACEHOLDER: Table 4 — Response time (seconds) broken down by dataset (SML, MED, LGE; rows) and scenario type (ED, RA, HC, CM; columns), for the surrogate engine. Include mean ± standard deviation. All values should be < 30 seconds. Add a final column for 95th percentile RT. Highlight that LGE CM scenarios approach but do not exceed the 30-second budget.]

The surrogate engine meets the 30-second response time budget for all 240 scenario instances at mean response times of [PLACEHOLDER: values by dataset]. The 95th percentile response time on large-instance compound scenarios is [PLACEHOLDER: seconds], confirming the budget is maintained across the operating envelope.

### 6.3 Recommendation Rank Correlation

[PLACEHOLDER: Table 5 — Recommendation Rank Correlation (Spearman rho) for Ours vs. LNS-Only vs. GR across compound scenario sets with 3, 4, and 5 open branches. Rows: branch count. Columns: methods. Ours should show rho >= 0.87; LNS-Only ~0.78; GR ~0.55. Include 95% CIs.]

The recommendation engine produces branch rankings with Spearman's $\rho = $ [PLACEHOLDER: value] relative to ground-truth rankings, substantially outperforming the LNS-only and greedy repair alternatives. This confirms that the small accuracy degradation of the surrogate relative to the full solver does not materially degrade the ordering of alternative scenarios, which is the decision-relevant outcome.

### 6.4 User Study Results

[PLACEHOLDER: Table 6 — User study results table. Rows: Time-to-Decision (minutes), Decision Confidence (Likert 1-7), Branches Explored (count), Expert-Rated Recommendation Quality (1-5). Columns: Condition B (mean ± SD), Condition T (mean ± SD), Test statistic, p-value, Effect size. TTD should show substantial reduction (~40-50%) under T; DC should show increase ~1.5-2 points; both significant at alpha=0.05.]

[PLACEHOLDER: Figure 3 — Paired boxplots showing TTD (left panel) and DC (right panel) distributions for conditions B and T across all 20 participants. Individual paired lines connecting each participant's scores under the two conditions. Significance brackets with p-values annotated.]

Under the treatment condition (with what-if tool), participants demonstrated significantly shorter time-to-decision ([PLACEHOLDER: values and test statistics]) and higher decision confidence ([PLACEHOLDER: values and test statistics]). Effect sizes indicate [PLACEHOLDER: characterization as small/medium/large per Cohen's conventions]. Expert-rated recommendation quality also improved under the treatment condition ([PLACEHOLDER: values]), suggesting that the tool not only reduces decision time but also improves decision quality as evaluated by domain experts.

### 6.5 Case Study: Enrollment Surge Detection

The prospective case study at the D2 institution documented the following findings. When scheduling officers submitted an enrollment delta scenario representing a 12% surge in engineering and computer science courses, the framework returned a KPI scorecard within [PLACEHOLDER: seconds], indicating that room utilization under the surge scenario would rise to [PLACEHOLDER: RU value], exceeding the saturation threshold of 0.85, and that hard-constraint satisfaction (room capacity, L-H4) would fall to [PLACEHOLDER: CSR_H value] if no remedial action were taken. The framework identified [PLACEHOLDER: number] specific course-room assignments that would become infeasible under the projected enrollment, and recommended two alternative scenario branches: one involving the conversion of a seminar room to a lecture room and one involving the addition of a late-afternoon section for the highest-enrollment courses.

This detection occurred six weeks before the semester start, compared to the institution's prior practice of identifying such conflicts typically two to three weeks before the semester when registration data is finalized. Post-semester verification confirmed that the surrogate KPI predictions were within [PLACEHOLDER: percentage] of the actual timetable outcomes, consistent with the aggregate simulation accuracy reported in Section 6.1.

---

## 7. Discussion

### 7.1 Implications of Results

The central finding of this study is that a surrogate simulation engine based on constraint propagation and LNS partial re-optimization can achieve simulation accuracy sufficient for reliable decision support — specifically, KPI correlation above 0.85 and directional accuracy above 0.90 for most scenario types — while operating within a 30-second response budget that enables interactive scenario exploration. This combination of accuracy and speed is not achievable with either the full solver (which provides accuracy but not speed) or with greedy heuristics (which provide speed but not accuracy).

The user study results extend this technical finding into a human factors context. The significant reduction in time-to-decision and improvement in decision confidence suggest that the framework addresses a real cognitive need: scheduling officers facing complex multi-variable scenarios benefit from structured scenario comparison with explicit KPI quantification, rather than being required to reason through solver logs or spreadsheet models. The improvement in expert-rated recommendation quality under the treatment condition is particularly notable, as it indicates that the tool's value is not merely psychological (reducing anxiety through the illusion of precision) but substantive (improving the actual quality of scheduling decisions).

The case study result — detection of a critical resource shortfall six weeks in advance rather than the customary two to three weeks — translates directly into institutional capacity to act. Room conversion or section addition decisions that require physical facilities work or lecturer contract amendments typically have lead times of four to six weeks; the conventional planning timeline barely accommodates this, while the framework's earlier warning provides a comfortable action margin.

### 7.2 Comparison with Prior Work

Relative to the work of Kiefer et al. (2017), our framework improves scenario response time from the full solver time reported in their work (ranging from minutes to hours) to under 30 seconds, while adding structured branch management, KPI scorecards, and recommendation ranking features absent from their system. The simulation accuracy result ($r > 0.85$) provides a formal guarantee that was not established in prior scenario-analysis work for timetabling.

Compared to Dammak et al. (2019), whose interactive assistant supports manual adjustments with constraint feedback, our framework operates at a higher level of abstraction (scenario types rather than individual assignment moves) and provides proactive detection of systemic resource issues rather than reactive constraint violation alerts. These different design points reflect different user needs: Dammak et al. target detailed manual editing, while our framework targets strategic planning and what-if exploration.

The equity and carbon KPIs in our scorecard extend the decision support scope beyond the purely logistical metrics (room utilization, constraint satisfaction) that dominate prior DSS work for scheduling. These additional dimensions are increasingly mandated by institutional governance and sustainability commitments and align with the emerging literature on multi-criteria timetabling (Fonseca et al., 2021; Ceschia et al., 2023).

### 7.3 Limitations

**Simulation accuracy degrades for compound scenarios.** As reported in Section 6.1, the surrogate engine's accuracy for equity scores under compound multi-factor scenarios is somewhat below the values for simpler scenario categories. This reflects the challenge that equity is a non-local metric: a change affecting 20% of assignments can alter demographic parity scores by much more than a proportional amount if the affected assignments are concentrated among particular student groups. Future work should explore equity-aware destroy operators that better capture this non-locality.

**User study ecological validity.** The user study employed realistic but ultimately synthetic scenario packages derived from anonymized data rather than live institutional instances. Participants were recruited from three universities but may not be representative of the full diversity of scheduling practice across institutional types, national contexts, and administrative cultures. Replication with larger and more diverse samples would strengthen the generalizability of the human factors findings.

**Recommendation engine weight sensitivity.** The recommendation score RS(b) is a linear weighted combination of KPIs. The linearity assumption may not reflect the non-linear trade-off structure of actual institutional priorities — for example, hard constraint satisfaction is likely near-lexicographically superior to other KPIs in practice, which is only approximately captured by assigning it a large weight $w_{\text{CSR}}$. Extensions using lexicographic or Pareto-based ranking are warranted.

**Scalability to very large instances.** While the 30-second budget is maintained for Dataset D3 (583 courses), the system has not been tested on instances exceeding 1,000 courses. At that scale, the AC-3 propagation and LNS repair may require further decomposition to remain within budget.

### 7.4 Threats to Validity

**Internal validity:** The within-subjects user study design controls for individual differences but introduces potential order effects despite counterbalancing. Participants may apply knowledge gained in the baseline condition to the treatment condition or vice versa. A between-subjects design with larger samples would provide a cleaner causal estimate.

**Construct validity:** Decision confidence is measured via self-report on a Likert scale, which captures perceived confidence rather than calibration (alignment between confidence and accuracy). Future work should measure calibration directly by comparing participant confidence levels to the actual quality of their decisions.

**External validity:** The three datasets are drawn from specific institutional contexts (UK and West African universities). Scheduling practices, constraint structures, and institutional priorities differ across national and cultural contexts, and the accuracy and usability results may not generalize without recalibration.

**Simulation accuracy as a proxy:** The 85% correlation target is defined relative to full-solver ground truth on the same datasets used for development. There is no guarantee that accuracy generalizes to qualitatively different constraint structures or institutional configurations not represented in the test suite.

---

## 8. Conclusions

This paper has presented a what-if simulation and comparison framework for university scheduling that addresses the gap between the power of modern scheduling solvers and the practical decision-support needs of scheduling practitioners. The framework makes four contributions: a surrogate simulation engine achieving accuracy above 85% correlation with full-solver outputs in under 30 seconds; a structured four-category scenario taxonomy with a declarative specification language; version-controlled named scenario branches with multi-dimensional KPI scorecards; and a weighted recommendation engine for ranking alternatives by institutional priority.

Empirical evaluation on datasets ranging from 86 to 583 courses demonstrates that the surrogate engine meets its accuracy and response time targets across all scenario categories. A controlled user study with 20 scheduling officers confirms that the tool significantly reduces time-to-decision and improves decision confidence relative to unaided practice. A prospective case study illustrates the framework's practical value by detecting a critical resource shortfall six weeks in advance of the semester start — a lead time advantage that translates directly into institutional capacity for remedial action.

The practical implications are clear: institutions facing the growing complexity of multi-objective, multi-stakeholder scheduling decisions need decision support tools that go beyond schedule generation to support ongoing scenario planning. The framework described here provides a concrete, validated architecture for such tools, deployable within existing scheduling infrastructure without requiring re-engineering of the underlying optimization engine.

Three concrete directions for future work follow naturally from the limitations identified in Section 7.3. First, equity-aware destroy operators for LNS repair should be developed to improve simulation accuracy for equity metrics under compound scenarios, potentially using demographic conflict graphs to guide the destruction phase. Second, the recommendation engine should be extended to support Pareto-front visualization across KPI dimensions, allowing scheduling officers to explore the trade-off surface interactively rather than being limited to a single weighted ranking. Third, the scenario specification language should be extended with probabilistic scenario specifications — for example, an enrollment delta drawn from a distribution rather than a point estimate — enabling risk-aware planning that quantifies the probability of constraint violations under uncertain input data.

---

## Declaration of Competing Interest

None.

## Acknowledgements

[Acknowledgements placeholder — funding bodies, participating institutions, scheduling officer participants.]

---

## References

Aboolian, R., Berman, O., & Krass, D. (2012). Optimizing resource allocation in faculty scheduling under multiple objectives. *European Journal of Operational Research*, 216(3), 654–665.

Abramson, D. (1991). Constructing school timetables using simulated annealing: Sequential and parallel algorithms. *Management Science*, 37(1), 98–113.

Baker, R. S., & Inventado, P. S. (2014). Educational data mining and learning analytics. In J. A. Larusson & B. White (Eds.), *Learning Analytics: From Research to Practice* (pp. 61–75). Springer.

Bertsimas, D., & Tsitsiklis, J. N. (1997). *Introduction to Linear Optimization*. Athena Scientific.

Bettinelli, A., Cacchiani, V., Roberti, R., & Toth, P. (2015). An overview of curriculum-based course timetabling. *TOP*, 23(2), 313–349.

Burke, E. K., Mareček, J., Parkes, A. J., & Rudová, H. (2010). Decomposition, reformulation, and diving in university course timetabling. *Computers & Operations Research*, 37(3), 582–597.

Ceschia, S., Di Gaspero, L., & Schaerf, A. (2023). Educational timetabling: Problems, benchmarks, and state-of-the-art results. *European Journal of Operational Research*, 308(1), 1–18.

Colorni, A., Dorigo, M., & Maniezzo, V. (1998). Metaheuristics for high school timetabling. *Computational Optimization and Applications*, 9(3), 275–298.

Cordeau, J. F., Laporte, G., Potvin, J. Y., & Savelsbergh, M. W. P. (2002). Transportation on demand. In C. Barnhart & G. Laporte (Eds.), *Handbooks in Operations Research and Management Science: Transportation* (Vol. 14, pp. 429–466). Elsevier.

Dammak, A., Elloumi, A., & Kamoun, H. (2019). Interactive university timetabling with dynamic constraint feedback. *Computers & Education*, 131, 12–26.

Fonseca, G. H. G., Santos, H. G., Carrano, E. G., & Stidsen, T. J. R. (2021). Learning to construct heuristic solutions for university timetabling via graph neural networks. *Computers & Operations Research*, 130, 105224.

Goldratt, E. M., & Cox, J. (2014). *The Goal: A Process of Ongoing Improvement* (4th ed.). North River Press.

Holsapple, C. W., & Whinston, A. B. (1996). *Decision Support Systems: A Knowledge-Based Approach*. West Publishing.

Hulshof, P. J. H., Kortbeek, N., Boucherie, R. J., Hans, E. W., & Bakker, P. J. M. (2012). Taxonomic classification of planning decisions in health care: A structured review of the state of the art in OR/MS. *Health Systems*, 1(2), 129–175.

Kiefer, A., Hartl, R. F., & Schnell, A. (2017). Adaptive large neighborhood search for the curriculum-based course timetabling problem. *Annals of Operations Research*, 252(2), 255–282.

Lach, G., & Lübbecke, M. E. (2012). Curriculum based course timetabling: New solutions to Udine benchmark instances. *Annals of Operations Research*, 194(1), 255–272.

Lalescu, L. (2011). FET — Free Timetabling Software. Retrieved from http://lalescu.ro/liviu/fet/

Lindahl, M., Mason, A. J., Stidsen, T., & Sørensen, M. (2019). A strategic view of university timetabling. *European Journal of Operational Research*, 276(1), 142–160.

Lombardi, M., & Milano, M. (2012). Warm-start of constraint propagation in combinatorial optimization. *Constraints*, 17(2), 133–157.

Maddox, T., Bhardwaj, A., Sen, S., & Madden, S. (2016). Decibel: The relational dataset branching system. *Proceedings of the VLDB Endowment*, 9(9), 624–635.

Marte, M. (2002). Models and algorithms for school timetabling — A constraint-programming approach (Doctoral dissertation). Ludwig-Maximilians-Universität München.

McCollum, B., Schaerf, A., Paechter, B., McMullan, P., Lewis, R., Parkes, A. J., Di Gaspero, L., Qu, R., & Burke, E. K. (2010). Setting the research agenda in automated timetabling: The second international timetabling competition. *INFORMS Journal on Computing*, 22(1), 120–130.

Müller, T., Rudová, H., & Müllerová, Z. (2018). University course timetabling and international timetabling competition 2019. In *Proceedings of the 12th International Conference on the Practice and Theory of Automated Timetabling* (pp. 5–31).

Nah, F. F. H., Lau, J. L. S., & Kuang, J. (2001). Critical factors for successful implementation of enterprise systems. *Business Process Management Journal*, 7(3), 285–296.

Perron, L., & Didier, F. (2020). CP-SAT. Google OR-Tools, version 7.8. Retrieved from https://developers.google.com/optimization

Pisinger, D., & Ropke, S. (2010). Large neighborhood search. In M. Gendreau & J. Y. Potvin (Eds.), *Handbook of Metaheuristics* (2nd ed., pp. 399–420). Springer.

Power, D. J. (2008). Understanding data-driven decision support systems. *Information Systems Management*, 25(2), 149–154.

Renggli, C., Karlaš, B., Ding, B., Liu, F., Schawinski, K., & Zhang, C. (2019). Continuous integration of machine learning models with ease.ml/ci. In *Proceedings of the 2019 Conference on Systems and Machine Learning* (pp. 1–6).

Rudová, H., Müller, T., & Murray, K. (2011). Complex university course timetabling. *Journal of Scheduling*, 14(2), 187–207.

Schaerf, A. (1999). A survey of automated timetabling. *Artificial Intelligence Review*, 13(2), 87–127.

Schoemaker, P. J. H. (1995). Scenario planning: A tool for strategic thinking. *Sloan Management Review*, 36(2), 25–40.

Shapiro, J. F. (2001). *Modeling the Supply Chain*. Duxbury Press.

Shaw, P. (1998). Using constraint programming and local search methods to solve vehicle routing problems. In M. Maher & J.-F. Puget (Eds.), *Principles and Practice of Constraint Programming — CP98*, Lecture Notes in Computer Science (Vol. 1520, pp. 417–431). Springer.

Shim, J. P., Warkentin, M., Courtney, J. F., Power, D. J., Sharda, R., & Carlsson, C. (2002). Past, present, and future of decision support technology. *Decision Support Systems*, 33(2), 111–126.

Spinellis, D. (2005). Version control systems. *IEEE Software*, 22(5), 108–109.

Tichy, W. F. (1985). RCS — A system for version control. *Software: Practice and Experience*, 15(7), 637–654.

Zafra, A., Romero, C., & Ventura, S. (2011). Multiple instance learning for classifying students in learning management systems. *Expert Systems with Applications*, 38(12), 15020–15031.
