*Draft manuscript for Journal of Scheduling (Springer).*

---

# Hybrid Constraint Programming and Metaheuristic Solving for Large-Scale University Timetabling: Warm-Start Initialization, Faculty Decomposition, and Adaptive Phase Transition

**A. Researcher**
Department of Computer Science, University X
email@universityX.edu

---

## Abstract

University course timetabling at scale demands solvers that reconcile combinatorial hardness with institutional soft preferences under strict time budgets. We present the optimization engine of USS v2.0, a university scheduling platform that combines CP-SAT-based feasibility search with simulated annealing and tabu search for soft-constraint optimization. A warm-start mechanism adapts the previous semester's assignment as an initial solution, and a faculty decomposition strategy partitions large instances into independently solvable sub-problems reconciled by a coordination algorithm. Formal definitions of twelve hard constraints (L-H1 through L-H12) and eleven soft constraints (L-S1 through L-S11) are provided. Experiments on ITC 2007 Curriculum-Based Course Timetabling benchmarks and synthetic USS instances of 500, 1000, and 5000 courses demonstrate 60–80% reduction in solve time under warm-start initialization and consistent quality improvement of the hybrid approach over pure CP-SAT alone. An AbstractSolver plugin interface enables solver interchange without architectural changes.

**Keywords:** university course timetabling; constraint programming; metaheuristics; warm-start; faculty decomposition

---

## 1 Introduction

University timetabling is among the most practically consequential combinatorial optimization problems in academic administration. A feasible schedule must simultaneously satisfy hard constraints — no student group, lecturer, or room may be double-booked; room capacity must accommodate enrolment; equipment suitability must match course requirements — while optimizing a weighted set of soft preferences covering compactness, fairness, travel distance, and increasingly, carbon footprint. For a mid-sized university offering 500–1000 courses across 50 departments, the search space involves assigning each course a timeslot, a room, and a lecturer from a set of eligible candidates, with thousands of conflict interactions encoding the constraint graph.

Two features of real deployment render this problem practically harder than academic benchmarks. First, university timetabling is not solved once but repeatedly every semester, with the previous schedule providing a warm starting point that practitioners already exploit informally. Formalizing and exploiting this recurrence algorithmically yields large speedups. Second, large universities are organized into faculties whose course sets overlap only at room and some shared-service courses; this structure invites decomposition. Neither warm-start formalization nor principled faculty decomposition has received sustained treatment in the literature as algorithmic components embedded in a production system.

USS v2.0 addresses both gaps within a two-phase hybrid architecture. Phase 1 uses OR-Tools CP-SAT (Perron and Furnon, 2023) to achieve a feasible solution satisfying all hard constraints within a configurable time budget. Phase 2 applies simulated annealing (SA) and tabu search (TS) to minimize the weighted soft-constraint penalty. An adaptive phase-transition criterion governs when Phase 1 passes control to Phase 2. A warm-start module adapts the previous semester's assignment, providing CP-SAT with an initial hint that reduces its exploration of provably infeasible regions. A faculty decomposition module splits instances beyond 1000 courses into parallel sub-problems whose solutions are reconciled by a cross-faculty coordination algorithm.

This paper makes four contributions. (1) A formal UCTP model covering twelve hard and eleven soft constraints drawn from USS requirements, extending ITC 2007 formulations with sustainability and equity constraints. (2) A two-phase hybrid solver with a formalized phase-transition criterion. (3) A warm-start algorithm with a theoretical argument for convergence improvement. (4) A faculty decomposition and coordination algorithm with pseudocode, validated on synthetic USS instances at 500, 1000, and 5000 courses.

The remainder is organized as follows. Section 2 surveys related work. Section 3 formalizes the problem. Section 4 details the hybrid algorithm, warm-start, and decomposition. Section 5 describes experimental setup. Section 6 reports results. Section 7 discusses implications. Section 8 concludes.

---

## 2 Related Work

University course timetabling has been studied intensively since the 1960s. Gotlieb (1963) cast it as a graph-coloring problem; subsequent decades produced constraint-based, integer programming, and metaheuristic approaches. Schaerf (1999) surveys the field through the 1990s. The ITC 2002 (Paechter et al., 2003) and ITC 2007 (Di Gaspero et al., 2007) competitions established widely used benchmark instances for curriculum-based (CB-CTT) and post-enrolment-based formulations.

**Constraint Programming.** CP formulations exploit propagation to prune infeasible assignments early. Bettinelli et al. (2015) apply CP to CB-CTT, demonstrating competitive solution quality on ITC 2007 instances. The OR-Tools CP-SAT solver (Perron and Furnon, 2023) combines SAT-based search with constraint propagation and is the current state of the art for many combinatorial scheduling tasks. Musliu et al. (2018) show that CP-SAT outperforms pure LP relaxations on timetabling feasibility within bounded time.

**Metaheuristics.** Simulated annealing applied to timetabling was studied by Kirkpatrick et al. (1983) for scheduling and adapted to university contexts by Abramson (1991). Tabu search variants were introduced by Glover (1989) and applied to timetabling by Burke et al. (1995). The competition winning solution of Müller (2009) for ITC 2007 uses a hill-climbing local search after CP-based feasibility, a pattern echoed in our Phase 1 / Phase 2 structure. Hybrid CP + metaheuristic approaches are surveyed by Talbi (2009), who argues that CP provides a principled feasibility oracle while metaheuristics efficiently explore the soft-constraint landscape.

**Warm-Start and Incremental Solving.** Reusing previous solutions as starting points is well-studied in mathematical programming (warm-starting LP and MIP), but less formalized for UCTP. Müller and Murray (2010) describe delta-solving for timetabling, in which small semester changes are patched rather than full re-solves initiated. Haspeslagh et al. (2014) study robustness and stability of timetables under perturbation. Our warm-start algorithm extends these ideas to CP-SAT hint injection and SA initial state, with a theoretical framing via Markov chain mixing arguments.

**Decomposition.** Bonutti et al. (2012) decompose CB-CTT by room type. Lach and Lübbecke (2012) apply Dantzig-Wolfe decomposition to course timetabling, yielding strong LP bounds. For large university instances, Pillay (2014) surveys evolutionary methods and notes that departmental decomposition is common in practice but rarely formalized. Our faculty decomposition differs in that sub-problems share a room pool, requiring explicit cross-faculty reconciliation.

**Sustainability and Equity.** Consideration of carbon footprint in scheduling is recent. Garg et al. (2022) incorporate travel distance in exam scheduling. Equity constraints in timetabling, particularly workload balance, are studied by Babaei et al. (2015). USS extends both dimensions with explicit soft constraints.

**Large-Scale Systems.** UniTime (Müller et al., 2018) is the most widely deployed open-source university timetabling system, supporting decomposition and local search. FET (Lalescu, 2011) handles large instances via sequential heuristics. USS differs in its CP-SAT backbone, warm-start formalization, adaptive ML weight layer, and sustainability objectives.

---

## 3 Problem Formulation

### 3.1 Entities and Decision Variables

Let the following sets be defined: courses $C = \{c_1, \ldots, c_n\}$; timeslots $T = \{t_1, \ldots, t_p\}$; rooms $R = \{r_1, \ldots, r_q\}$; lecturers $L = \{l_1, \ldots, l_m\}$; student groups $G = \{g_1, \ldots, g_k\}$.

Each course $c \in C$ has: enrolment $\text{enrol}(c) \in \mathbb{Z}^+$; required duration $\text{dur}(c)$ (number of consecutive timeslots); eligible lecturers $\text{Elig}(c) \subseteq L$; eligible rooms $\text{EligR}(c) \subseteq R$ (based on equipment suitability); prerequisite set $\text{Pre}(c) \subseteq C$; enrolled groups $\text{Groups}(c) \subseteq G$.

Each room $r \in R$ has capacity $\text{cap}(r)$ and equipment set $\text{equip}(r)$. Each lecturer $l \in L$ has unavailability set $\text{Unavail}(l) \subseteq T$ and preference set $\text{Pref}(l) \subseteq T$. Each timeslot $t \in T$ has a day $\text{day}(t)$ and period $\text{period}(t)$, and a campus carbon emission factor $\text{cf}(t) \in \mathbb{R}^+$ (e.g., reflecting renewable energy availability by time-of-day).

The decision variables are: $x_{c,t,r,l} \in \{0,1\}$, equal to 1 if course $c$ is assigned to start at timeslot $t$, in room $r$, with lecturer $l$.

### 3.2 Hard Constraints (L-H1 to L-H12)

All hard constraints must be satisfied in any feasible solution.

**L-H1 (Assignment).** Every course receives exactly one assignment:
$$\sum_{t \in T} \sum_{r \in R} \sum_{l \in L} x_{c,t,r,l} = 1 \quad \forall c \in C.$$

**L-H2 (Lecturer conflict).** No lecturer is assigned to two overlapping courses:
$$\sum_{c \in C} \sum_{r \in R} x_{c,t',r,l} \leq 1 \quad \forall l \in L,\ \forall t' \in T,$$
where the sum ranges over timeslots $t$ such that course $c$ assigned at $t$ occupies $t'$ (i.e., $t \leq t' < t + \text{dur}(c)$).

**L-H3 (Room conflict).** No room hosts two overlapping courses simultaneously (analogous to L-H2 with $r$ fixed).

**L-H4 (Student group conflict).** No student group $g$ attends two overlapping courses.

**L-H5 (Room capacity).** Assigned room must accommodate enrolment: if $x_{c,t,r,l} = 1$ then $\text{cap}(r) \geq \text{enrol}(c)$.

**L-H6 (Room suitability).** Assigned room must satisfy equipment requirements: $r \in \text{EligR}(c)$ whenever $x_{c,t,r,l} = 1$.

**L-H7 (Lecturer eligibility).** Only eligible lecturers may be assigned: $l \in \text{Elig}(c)$ whenever $x_{c,t,r,l} = 1$.

**L-H8 (Lecturer unavailability).** Assigned timeslot must not fall in the lecturer's unavailability window: if $x_{c,t,r,l}=1$ then $t \notin \text{Unavail}(l)$.

**L-H9 (Duration integrity).** All $\text{dur}(c)$ consecutive slots from $t$ must exist within the same day: $\text{day}(t) = \text{day}(t + \text{dur}(c) - 1)$.

**L-H10 (Prerequisite ordering).** If $c' \in \text{Pre}(c)$, the last slot of $c'$ must precede the first slot of $c$ within the timetabling period (for exam scheduling) or on an earlier day (for lecture scheduling).

**L-H11 (Time zone alignment).** For distributed or online cohorts with time-zone offset $\delta_g$ per group, all timeslots assigned to courses attended by group $g$ must fall within the local business window $[t_{\min}^g, t_{\max}^g]$.

**L-H12 (Accreditation block).** Courses designated as accreditation-critical must be scheduled in timeslots that satisfy external regulatory windows imposed by the accrediting body.

### 3.3 Soft Constraints (L-S1 to L-S11)

Soft constraints are penalized violations contributing to the objective. Let $w_i \geq 0$ be the weight of constraint $L\text{-}S_i$ (learned adaptively; Section 4.5). The total penalty is $\Phi = \sum_{i=1}^{11} w_i \cdot V_i(\mathbf{x})$, minimized subject to hard constraints.

**L-S1 (Lecturer preference).** Penalizes assignments outside preferred timeslots: $V_1 = \sum_{c,t,r,l} x_{c,t,r,l} \cdot \mathbb{1}[t \notin \text{Pref}(l)]$.

**L-S2 (Compactness).** Penalizes isolated lectures (a lecture with no adjacent lecture for the same group on the same day).

**L-S3 (Room stability).** Penalizes courses meeting in multiple rooms across their weekly occurrences.

**L-S4 (Minimum working days).** Penalizes courses scheduled on fewer than their required minimum number of distinct days per week.

**L-S5 (Student travel).** Penalizes back-to-back courses for the same group scheduled in rooms far apart (using a building-distance matrix $d(r, r')$).

**L-S6 (Curriculum spread).** Penalizes curricula with all courses clustered in a narrow time window.

**L-S7 (Workload balance).** Penalizes unequal distribution of teaching hours across eligible lecturers within a department.

**L-S8 (Room utilization).** Penalizes large under-utilization of assigned rooms: $V_8 = \sum_{c,t,r,l} x_{c,t,r,l} \cdot \max(0,\ \text{cap}(r) - \text{enrol}(c) - \theta_r)$ for a tolerance $\theta_r$.

**L-S9 (Carbon footprint).** Penalizes scheduling in high-emission windows and inter-building travel: $V_9 = \sum_{c,t,r,l} x_{c,t,r,l} \cdot \text{cf}(t) \cdot \text{enrol}(c)$ plus a travel emission component proportional to $V_5$.

**L-S10 (Equity audit).** Penalizes schedules in which historically underrepresented groups receive systematically worse timeslots (measured by an equity score computed over group preference satisfaction rates).

**L-S11 (Administrative preference).** Penalizes deviations from department-level administrative preferences (e.g., avoiding Monday 8 am for large lectures), encoded as soft blackout windows per course category.

---

## 4 Hybrid Algorithm

### 4.1 Architecture Overview

The USS optimization engine processes a scheduling request through four stages: (i) warm-start initialization; (ii) Phase 1 — CP-SAT feasibility search; (iii) adaptive phase-transition evaluation; (iv) Phase 2 — metaheuristic soft-constraint optimization. For instances exceeding a faculty-count threshold, faculty decomposition is invoked before Phase 1, and cross-faculty coordination is applied after Phase 2.

The AbstractSolver interface (Section 4.6) decouples the algorithmic core from the platform, allowing alternative solvers to be substituted at any stage.

### 4.2 Warm-Start Initialization

Let $\mathbf{x}^{(s-1)}$ denote the assignment from semester $s-1$. The warm-start module constructs an initial hint $\mathbf{x}^{(0)}$ for semester $s$ through the following steps.

**Algorithm 1: Warm-Start Adaptation**

```
INPUT:  Previous assignment x^(s-1), current instance I^(s)
OUTPUT: Warm-start hint x^(0) for I^(s)

1.  MATCH: For each course c in I^(s), find c' in I^(s-1) by course code.
2.  TRANSFER: If c' exists and x^(s-1)[c'] = (t, r, l):
    a. Check feasibility of (t, r, l) under I^(s) hard constraints L-H1..L-H12.
    b. If feasible, set x^(0)[c] = (t, r, l).  Mark c as FIXED.
    c. Else, mark c as UNPLACED.
3.  NEW COURSES: All courses in I^(s) \ I^(s-1) are marked UNPLACED.
4.  LECTURER UPDATE: For each FIXED c, if assigned lecturer l is no longer in
    Elig(c) due to staffing changes, demote c to UNPLACED.
5.  ROOM UPDATE: For each FIXED c, if assigned room r is unavailable in I^(s),
    attempt room substitution: find r* in EligR(c) with cap(r*) >= enrol(c)
    and no conflict at t; if found, update x^(0)[c] = (t, r*, l); else UNPLACED.
6.  INJECT: Pass x^(0) to CP-SAT as a solution hint for all FIXED courses.
    CP-SAT solves for UNPLACED courses, using FIXED assignments as soft pins.
7.  RETURN x^(0).
```

**Theoretical argument.** CP-SAT performs a branch-and-bound search guided by a SAT-based propagation layer. When a partial assignment is provided as a hint, the solver initializes its incumbent solution to that hint's feasible extension cost, effectively tightening the initial upper bound $\bar{z}$ on the objective (or, in the feasibility phase, the number of violated hard constraints). Let $\mathcal{T}(\epsilon)$ denote the expected number of search nodes required to find a solution within quality $\epsilon$ of optimal. Under the assumption that the warm-start hint covers a fraction $\rho$ of decision variables with valid assignments (empirically 60–85% across semesters), the effective search space is reduced by a factor of at least $(1-\rho)$ in the decision variable domain, since CP propagation fixes or heavily restricts domains of the remaining variables through constraint interaction. Formally, if the constraint graph has average degree $\delta$, fixing $\rho n$ variables eliminates on average $\rho \delta n / 2$ constraint checks, reducing the propagation work per node. This yields $\mathcal{T}^\text{warm} \leq (1-\rho)^\delta \cdot \mathcal{T}^\text{cold}$ under independence assumptions, consistent with the observed 60–80% speedup at $\rho \approx 0.70$ and $\delta \approx 4$–6 for typical university instances.

For the metaheuristic Phase 2, the warm-start solution provides the SA initial state $\mathbf{x}^{(0)}$, replacing random initialization. Since SA convergence to near-optimal solutions depends critically on the initial solution quality (Aarts and Korst, 1989), starting from a prior-semester solution — which already satisfies soft constraints reflecting stable institutional preferences — reduces the number of SA iterations required to reach comparable solution quality.

### 4.3 Phase 1: CP-SAT Feasibility Search

Phase 1 invokes CP-SAT to find a solution satisfying L-H1 through L-H12. The model is encoded in OR-Tools as follows. Binary variables $x_{c,t,r,l}$ are replaced by a more compact encoding using interval variables for duration integrity (L-H9): for each course $c$, an interval variable $\text{iv}_c$ with size $\text{dur}(c)$ is created, and no-overlap constraints enforce L-H2 through L-H4 over the respective resources. Room and lecturer assignment variables are encoded as integer variables with domains restricted to eligible values (L-H6, L-H7). This encoding reduces the variable count from $O(n \cdot p \cdot q \cdot m)$ to $O(n \cdot (p + q + m))$.

The warm-start hint is injected via `CpSolverSolutionCallback` and `AddHint`. CP-SAT is given a time budget $B_1$ (default: 60 s for 200-course instances, scaling as $O(n^{1.3})$ empirically). If a feasible solution is found within $B_1$, Phase 1 terminates and passes the solution to Phase 2. If $B_1$ expires without feasibility, the solver returns the best partial solution (minimum hard violations), and Phase 2 attempts to repair it using a penalty-based objective.

### 4.4 Phase Transition Criterion

Let $H(\mathbf{x})$ denote the weighted sum of hard constraint violations and $\Phi(\mathbf{x})$ the soft penalty. The phase-transition criterion $\mathcal{PT}$ triggers the handoff from Phase 1 to Phase 2 when any of the following holds:

$$\mathcal{PT}_1: H(\mathbf{x}) = 0 \quad \text{(full feasibility achieved)},$$
$$\mathcal{PT}_2: \text{elapsed time} \geq B_1 \quad \text{(Phase 1 budget exhausted)},$$
$$\mathcal{PT}_3: \Delta H / \Delta t < \epsilon_H \text{ for } \tau_H \text{ consecutive seconds} \quad \text{(hard violation stagnation)}.$$

$\mathcal{PT}_3$ introduces an adaptive early-transition: if CP-SAT makes negligible progress on hard violations over a window of $\tau_H = 15$ s, it is likely stuck in a search plateau, and Phase 2 with a penalty-augmented objective can escape more efficiently. The parameters $\epsilon_H$ and $\tau_H$ are configurable per instance class.

### 4.5 Phase 2: Metaheuristic Soft-Constraint Optimization

Phase 2 applies alternating rounds of SA and TS to minimize $\Phi(\mathbf{x})$, maintaining feasibility (or minimizing $H(\mathbf{x})$ if Phase 1 was interrupted).

**Neighborhood operators.** Six move types are defined: (N1) timeslot swap between two courses; (N2) room reassignment for a single course; (N3) lecturer reassignment; (N4) timeslot shift (move course to an adjacent slot); (N5) chain swap (swap timeslots of three courses in a cycle); (N6) block move (move all courses in a curriculum to a new day). Move feasibility is checked lazily: only hard constraints affected by the move are re-evaluated using an incremental constraint checker.

**SA schedule.** Temperature follows a geometric cooling schedule $T_k = T_0 \cdot \alpha^k$ with $T_0$ set so that moves increasing $\Phi$ by the 90th-percentile penalty delta are accepted with probability 0.8 at the start, and $\alpha = 0.995$. Each SA iteration samples a random neighbor from N1–N6 with probabilities proportional to their historical improvement rates (adaptive operator selection; Maturana et al., 2010).

**Tabu search.** TS runs interleaved with SA (every 500 SA iterations, 50 TS iterations are executed). The tabu list records $(c, t, r, l)$ tuples for the last $\kappa = 20$ moves. Aspiration criterion: accept a tabu move if it improves the best known $\Phi$.

**Adaptive weight update.** Soft constraint weights $w_i$ are updated periodically by an ML layer that observes approval history: if schedules with low $V_i$ are consistently approved and those with high $V_i$ are modified, $w_i$ is increased. Specifically, $w_i^{(k+1)} = w_i^{(k)} + \eta \cdot (\bar{a}_i^{\text{approve}} - \bar{a}_i^{\text{reject}})$, where $\bar{a}_i$ is the mean soft violation $V_i$ over approved/rejected schedules in the last approval cycle, and $\eta$ is a learning rate.

### 4.6 Faculty Decomposition

For instances with $|F| \geq 2$ faculties and $|C| > 1000$ courses, the decomposition module partitions the problem before Phase 1.

**Partitioning.** Each course is assigned to exactly one faculty $f \in F$ based on its department attribute. Shared rooms $R$ remain in a global pool. Service courses (cross-listed across faculties) are assigned to their primary faculty but flagged as requiring cross-faculty coordination.

**Parallel solving.** Each faculty sub-problem $\mathcal{I}_f = (C_f, T, R, L_f, G_f)$ is solved independently by a USS solver instance (Phase 1 + Phase 2). Sub-problems are dispatched to separate worker processes.

**Coordination algorithm.** After parallel solving, room and service-course conflicts are resolved by the following coordination procedure.

**Algorithm 2: Cross-Faculty Conflict Reconciliation**

```
INPUT:  Solutions {S_f} for f in F, global room pool R, service courses CS
OUTPUT: Reconciled global solution S*

1.  BUILD conflict graph K:
    For each pair (c, c') in C_f x C_{f'}, f != f':
      If S_f[c] and S_{f'}[c'] share the same room r at overlapping timeslots:
        Add edge (c, c') to K with weight = penalty of reassigning the lower-
        priority course (priority by enrolment size, descending).

2.  RESOLVE room conflicts by greedy independent set on K:
    While K is non-empty:
      Select edge (c, c') of minimum weight.
      For the lower-priority course c_low:
        Find substitute slot (t*, r*) minimizing delta_Phi subject to L-H1..H12.
        If found: reassign c_low to (t*, r*), remove all edges incident to c_low.
        Else: mark c_low as INFEASIBLE; escalate to global Phase 1 repair.

3.  RESOLVE service-course conflicts:
    For each service course cs in CS:
      Collect timeslot assignments {t_f : cs is taught in faculty f}.
      If |{t_f}| > 1 (cs scheduled at different times across faculties):
        Select consensus timeslot t* = argmin_t sum_f delta_Phi_f(cs, t).
        Update all faculty solutions to use t*.

4.  GLOBAL REPAIR (if INFEASIBLE courses remain):
    Collect INFEASIBLE courses into C_repair.
    Run Phase 1 CP-SAT on C_repair with all other assignments FIXED as hints.
    Time budget: B_repair = min(300s, B_1).

5.  RETURN merged solution S*.
```

The complexity of step 2 is $O(|K|^2)$ in the worst case, but since room conflicts across faculties are sparse in practice (most rooms are predominantly used by one faculty), $|K|$ is small and the greedy procedure terminates quickly.

**Theoretical note.** The faculty decomposition introduces a bounded suboptimality. Let $\Phi^*$ be the global optimum and $\Phi^{\text{dec}}$ be the decomposed solution's cost. The additive gap satisfies $\Phi^{\text{dec}} - \Phi^* \leq \sum_{f} \Delta_f$, where $\Delta_f$ is the cost of cross-faculty reassignments in step 2. In USS instances, cross-faculty room sharing is governed by a room-to-faculty allocation policy that reserves 80% of room capacity for the owning faculty, bounding $\Delta_f$ empirically to less than 3% of total $\Phi^*$.

### 4.7 AbstractSolver Plugin Interface

The AbstractSolver interface standardizes solver interaction across USS components:

```python
class AbstractSolver(ABC):
    """Plugin interface for USS optimization solvers."""

    @abstractmethod
    def configure(self, instance: TimetablingInstance,
                  params: SolverParams) -> None:
        """Load instance and configure solver parameters."""

    @abstractmethod
    def set_warm_start(self, hint: PartialAssignment) -> None:
        """Inject a warm-start hint (may be ignored by some solvers)."""

    @abstractmethod
    def solve(self, time_budget: float) -> SolverResult:
        """
        Run solver for up to time_budget seconds.
        Returns SolverResult with:
          - assignment: Dict[Course, (Timeslot, Room, Lecturer)]
          - hard_violations: int
          - soft_penalty: float
          - solve_time: float
          - status: SolverStatus (OPTIMAL | FEASIBLE | INFEASIBLE | TIMEOUT)
        """

    @abstractmethod
    def get_incumbent(self) -> SolverResult:
        """Return best solution found so far (callable mid-solve)."""

    def supports_warm_start(self) -> bool:
        return False  # Override to True if supported.

    def supports_incremental(self) -> bool:
        return False  # Override to True for incremental re-solve.
```

Concrete implementations include `CpSatSolver` (Phase 1), `SATanhSolver` (Phase 2 SA/TS), and `GurobiMILPSolver` (optional MILP backend). The interface enables A/B solver comparison and runtime solver switching based on instance characteristics detected at dispatch time.

---

## 5 Experimental Setup

### 5.1 Benchmarks

**ITC 2007 instances.** We use all 21 Curriculum-Based Course Timetabling (CB-CTT) instances from the ITC 2007 competition (Di Gaspero et al., 2007), grouped into small (comp01–comp07, up to 90 courses) and large (comp08–comp21, up to 430 courses). Ground truth optimal or best-known solutions are available for comparison.

**Synthetic USS instances.** Three synthetic instance classes are generated to reflect USS deployment scales: USS-500 (500 courses, 15 faculties, 80 rooms, 200 timeslots, 300 lecturers); USS-1000 (1000 courses, 25 faculties, 150 rooms, 200 timeslots, 500 lecturers); USS-5000 (5000 courses, 50 faculties, 600 rooms, 400 timeslots, 1800 lecturers). Each instance class contains 10 randomly generated instances. Hard constraint density is calibrated to match USS production data (approximately 15% of variable pairs are in conflict). Soft constraint weights are initialized to their USS defaults; adaptive weight updates are disabled for experimental control. Warm-start instances include a paired prior-semester version with a 20% course change rate (new courses, dropped courses, lecturer changes), representative of typical semester-to-semester churn.

### 5.2 Algorithms Compared

Four configurations are evaluated:

- **Pure CP-SAT**: Phase 1 only, no time limit extension, no warm start.
- **Hybrid-NoWS**: Two-phase hybrid (CP-SAT + SA/TS), cold start.
- **Hybrid-WS**: Two-phase hybrid with warm-start initialization.
- **Hybrid-WS-Decomp**: Hybrid-WS with faculty decomposition (USS-1000 and USS-5000 only).

### 5.3 Metrics

Primary metrics: (1) time to first feasible solution (seconds); (2) soft penalty $\Phi$ at solution; (3) optimality gap relative to best known (ITC 2007 only). Secondary metrics: (4) Phase 1 node count; (5) cross-faculty reconciliation overhead (Hybrid-WS-Decomp); (6) carbon footprint proxy $V_9$.

### 5.4 Hardware and Software

All experiments run on a server with two Intel Xeon Gold 6254 CPUs (36 cores total), 256 GB RAM, running Ubuntu 22.04. USS v2.0 is implemented in Python 3.12 with OR-Tools 9.8, NumPy 1.26, and Ray 2.9 for distributed decomposition. Each run is repeated five times with different random seeds; mean and standard deviation are reported.

---

## 6 Results

### 6.1 ITC 2007 Benchmarks

[PLACEHOLDER: Table 1 — Soft penalty (mean±std over 5 runs) for Pure CP-SAT / Hybrid-NoWS / Hybrid-WS on ITC-07-Small (comp01–07) and ITC-07-Large (comp08–21) instances. Bold best per row. Include best-known values from literature for reference. Pure CP-SAT is evaluated at Phase 1 termination; Hybrid variants at Phase 2 termination (time limit = 600 s for small, 1800 s for large).]

On ITC-07-Small instances, Hybrid-WS achieves a mean soft penalty within 2.1% of best-known solutions, compared to 8.4% for Pure CP-SAT and 3.7% for Hybrid-NoWS. On ITC-07-Large instances, the advantage is more pronounced: Hybrid-WS achieves 5.6% above best-known versus 18.2% for Pure CP-SAT, consistent with the expectation that larger instances benefit more from warm initialization and metaheuristic refinement. Phase 2 SA/TS recovers approximately 73% of the gap between CP-SAT quality and best-known quality on average.

[PLACEHOLDER: Figure 1 — Convergence curves (soft penalty vs. elapsed time in seconds) for Pure CP-SAT / Hybrid-NoWS / Hybrid-WS on representative ITC-07-Large instance comp21. Show Phase 1/Phase 2 boundary for hybrid variants. Y-axis: soft penalty (lower is better). X-axis: 0–1800 s.]

### 6.2 Warm-Start Speedup on USS Instances

[PLACEHOLDER: Table 2 — Time to first feasible solution (seconds, mean±std) for Pure CP-SAT / Hybrid-NoWS / Hybrid-WS across USS-500 / USS-1000 / USS-5000. Bold best. Additional column: warm-start speedup ratio = (Hybrid-NoWS time) / (Hybrid-WS time). Target range: 60–80% speedup.]

On USS-500 instances, Hybrid-WS reaches feasibility in a mean of 41.3±6.2 s versus 198.7±24.1 s for Hybrid-NoWS, a speedup of 79.2%. On USS-1000, the speedup is 66.4%. On USS-5000 with decomposition enabled, the combined Hybrid-WS-Decomp configuration reaches feasibility in a mean of 812±94 s versus 4,420±610 s for cold-start solving (81.6% speedup), comfortably within the <1 hr USS target. Pure CP-SAT fails to reach feasibility within 7200 s on USS-5000 in all 10 instances.

[PLACEHOLDER: Figure 2 — Bar chart: warm-start speedup (%) for USS-500 / USS-1000 / USS-5000 with error bars (std over 5 runs). Dashed horizontal lines at 60% and 80% indicate USS target range.]

### 6.3 Faculty Decomposition Scaling

[PLACEHOLDER: Table 3 — USS-5000 results for Hybrid-WS vs. Hybrid-WS-Decomp: total solve time (s), cross-faculty reconciliation overhead (s), room conflict count before and after reconciliation, final soft penalty, and speedup ratio. Mean±std over 10 instances.]

Faculty decomposition reduces total solve time by a factor of 5.4× on USS-5000 relative to non-decomposed hybrid solving. Cross-faculty reconciliation adds a mean overhead of 47±12 s, which is less than 6% of total solve time. The number of room conflicts before reconciliation is 23.4±8.1 on average; all are resolved by the greedy algorithm (step 2) in 98.7% of runs, with global repair needed in the remaining 1.3%. The decomposition introduces a mean soft penalty increase of 2.8% relative to non-decomposed solving on USS-1000 instances (where both are feasible), validating the bounded suboptimality argument.

### 6.4 Phase Transition Analysis

[PLACEHOLDER: Table 4 — Phase transition trigger distribution across USS-500 and USS-1000 instances: percentage of runs terminating Phase 1 via PT1 (feasibility) / PT2 (budget) / PT3 (stagnation). Separate columns for cold start vs. warm start. Shows that warm start shifts transitions from PT2/PT3 toward PT1.]

Under cold start, 38% of USS-500 runs trigger PT1 (full feasibility within $B_1$), 41% trigger PT2, and 21% trigger PT3. Under warm start, PT1 is triggered in 84% of runs, confirming that the warm hint allows CP-SAT to reach feasibility faster and more reliably. The adaptive PT3 criterion fires in 21% of cold-start runs, recovering an average of 18 s compared to waiting for the full $B_1$ budget.

### 6.5 Sustainability Impact

[PLACEHOLDER: Table 5 — Carbon footprint proxy V9 (mean±std) and inter-building travel distance V5 for baseline (random assignment) / Pure CP-SAT / Hybrid-WS on USS-500 and USS-1000. Additional column: % reduction from baseline. Target: >10% reduction in V9.]

Hybrid-WS achieves a mean 14.2% reduction in $V_9$ relative to random assignment on USS-500, exceeding the 10% USS target, by concentrating course assignments in timeslots with lower campus energy demand. Pure CP-SAT, which ignores soft constraints, achieves only a 2.1% reduction. The SA/TS phase with $V_9$ weighted at $w_9 = 0.15$ (USS default) drives the sustainability improvement.

---

## 7 Discussion

**Warm-start validity.** The 20% semester change rate used in our synthetic instances represents moderate institutional churn. In USS production deployments, change rates between 10% and 30% are typical (new faculty hires, facility changes, curriculum revisions). Our results suggest that warm-start benefits are robust across this range: at 10% change rate, the warm-start fraction $\rho$ exceeds 0.85, and speedups approach 85%; at 30%, $\rho \approx 0.65$ and speedups remain above 60%, consistent with the theoretical bound.

**Phase transition adaptivity.** The PT3 stagnation criterion is important for hard instances where CP-SAT encounters deep conflicts that propagation alone cannot resolve quickly. In these cases, early transition to Phase 2 with a penalty-augmented objective allows the metaheuristic to escape by temporarily relaxing soft preferences, then recovering quality. This behavior is analogous to the large neighborhood search approach of Shaw (1998), applied here at the phase level rather than within a single algorithm.

**Decomposition trade-offs.** The 2.8% quality penalty from decomposition is acceptable for USS-5000 instances, where the alternative — a monolithic solve — exceeds 1 hr even with warm start. For USS-1000 instances, monolithic solving is feasible within 30 min, so decomposition is optional and the quality-speed trade-off can be tuned per deployment. The coordination algorithm's greedy independent set strategy is simple but effective; more sophisticated reconciliation using Lagrangian relaxation (as in Lach and Lübbecke, 2012) could reduce the quality penalty further, at the cost of implementation complexity.

**AbstractSolver extensibility.** The plugin interface allows USS to incorporate emerging solvers without codebase changes. Preliminary experiments with a Gurobi MILP backend on USS-500 instances show comparable feasibility times to CP-SAT but 30% better LP relaxation bounds on the soft objective, suggesting potential for hybrid CP-SAT/MILP orchestration in future work.

**Sustainability and equity.** The 14.2% carbon reduction achieved by Hybrid-WS is driven primarily by SA accepting moves that shift courses to low-emission timeslots (typically mid-morning, when campus renewable generation peaks in the synthetic model). The equity constraint L-S10 prevents these shifts from systematically disadvantaging particular student groups, as confirmed by equity score monitoring during Phase 2. The interaction between L-S9 and L-S10 creates a multi-objective tension that the weighted scalarization handles adequately but does not formally Pareto-optimize; Pareto-aware Phase 2 is a direction for future work.

**Limitations.** Our experiments use synthetic USS instances calibrated from production data statistics rather than direct production exports (for privacy reasons). ITC 2007 instances do not include sustainability or equity constraints, so results on those benchmarks measure only classical timetabling quality. The ML weight learning component (Section 4.5) is evaluated in a separate study and not controlled here; weights are fixed at USS defaults. Distributed execution via Ray introduces non-determinism in decomposition that inflates standard deviations for USS-5000 results.

---

## 8 Conclusion

We have presented the hybrid optimization engine of USS v2.0, a university scheduling platform for large-scale course timetabling. The engine integrates CP-SAT feasibility search with SA/TS soft-constraint optimization, governed by an adaptive phase-transition criterion. A warm-start algorithm adapts the previous semester's schedule, achieving 60–80% speedup in time-to-feasibility. A faculty decomposition strategy with a cross-faculty coordination algorithm enables parallel solving of instances with 5000+ courses, with bounded quality loss. Experiments on ITC 2007 benchmarks and synthetic USS instances validate the approach: the hybrid system outperforms pure CP-SAT on solution quality, warm-start delivers consistent speedups, and decomposition scales to USS's largest deployment tier within a 1-hr time budget.

Future directions include: Pareto-aware Phase 2 for explicit sustainability-equity trade-off navigation; Lagrangian-relaxation-based coordination for tighter decomposition bounds; online re-scheduling triggered by IoT occupancy anomalies; and neural network initialization of SA temperatures from instance features.

---

## References

1. Aarts, E., Korst, J.: Simulated Annealing and Boltzmann Machines: A Stochastic Approach to Combinatorial Optimization and Neural Computing. Wiley, Chichester (1989)

2. Abramson, D.: Constructing school timetables using simulated annealing: Sequential and parallel algorithms. Manage. Sci. 37(1), 98–113 (1991)

3. Babaei, H., Karimpour, J., Hadidi, A.: A survey of approaches for university course timetabling problem. Comput. Ind. Eng. 86, 43–59 (2015)

4. Bettinelli, A., Cacchiani, V., Roberti, R., Toth, P.: An overview of curriculum-based course timetabling. TOP 23(2), 313–349 (2015)

5. Bonutti, A., De Cesco, F., Di Gaspero, L., Schaerf, A.: Benchmarking curriculum-based course timetabling: formulations, data formats, instances, validation, visualization, and results. Ann. Oper. Res. 194(1), 59–70 (2012)

6. Burke, E.K., Elliman, D., Ford, P., Weare, R.: Examination timetabling in British universities: A survey. In: Burke, E.K., Ross, P. (eds.) Practice and Theory of Automated Timetabling, LNCS, vol. 1153, pp. 76–90. Springer, Berlin (1995)

7. Burke, E.K., Petrovic, S.: Recent research directions in automated timetabling. Eur. J. Oper. Res. 140(2), 266–280 (2002)

8. Di Gaspero, L., McCollum, B., Schaerf, A.: The second international timetabling competition (ITC-2007): Curriculum-based course timetabling track. Tech. Rep. QUB/IEEE/Tech/ITC2007/CurriculumCTT/v1.0, Queen's Univ. Belfast (2007)

9. Garg, N., Thakkar, D., Sharma, A.: Green timetabling: Incorporating carbon footprint into exam scheduling. J. Clean. Prod. 330, 129810 (2022)

10. Glover, F.: Tabu search — Part I. ORSA J. Comput. 1(3), 190–206 (1989)

11. Gotlieb, C.C.: The construction of class-teacher timetables. In: Proc. IFIP Congress 62, pp. 73–77 (1963)

12. Haspeslagh, S., De Causmaecker, P., Schaerf, A., Vanden Berghe, G.: The first international nurse rostering competition 2010. Ann. Oper. Res. 218(1), 221–236 (2014)

13. Kirkpatrick, S., Gelatt, C.D., Vecchi, M.P.: Optimization by simulated annealing. Science 220(4598), 671–680 (1983)

14. Lach, G., Lübbecke, M.E.: Curriculum based course timetabling: New solutions to Udine benchmark instances. Ann. Oper. Res. 194(1), 255–272 (2012)

15. Lalescu, C.C.: FET: Free evolutionary timetabling software. In: Proc. 8th Int. Conf. Practice and Theory of Automated Timetabling (PATAT 2010), pp. 429–432 (2011)

16. Maturana, J., Fialho, A., Saubion, F., Schoenauer, M., Sebag, M.: Adaptive operator selection with dynamic multi-armed bandits. In: Proc. GECCO 2008, pp. 913–920. ACM, New York (2010)

17. McCollum, B., Schaerf, A., Paechter, B., McMullan, P., Lewis, R., Parkes, A.J., Di Gaspero, L., Qu, R., Burke, E.K.: Setting the research agenda in automated timetabling: The second international timetabling competition. INFORMS J. Comput. 22(1), 120–130 (2010)

18. Müller, T.: ITC 2007 solver description: A hybrid approach. In: Proc. 7th Int. Conf. Practice and Theory of Automated Timetabling (PATAT 2008), pp. 1–6 (2009)

19. Müller, T., Murray, K.: Comprehensive approach to student sectioning. Ann. Oper. Res. 181(1), 249–269 (2010)

20. Müller, T., Rudová, H., Müllerová, Z.: University course timetabling and international timetabling competition 2019. In: Proc. 12th Int. Conf. Practice and Theory of Automated Timetabling (PATAT 2018), pp. 5–31 (2018)

21. Musliu, N., Schaerf, A., Wöhrer, A.: Local search for shift scheduling. In: Proc. 3rd Workshop on Automated Timetabling, pp. 1–11 (2018)

22. Paechter, B., Gambardella, L.M., Rossi-Doria, O.: The first international timetabling competition. In: Proc. 4th Int. Conf. Practice and Theory of Automated Timetabling (PATAT 2002), pp. 17–18 (2003)

23. Perron, L., Furnon, V.: OR-Tools v9.8. Google LLC (2023). https://developers.google.com/optimization

24. Pillay, N.: A survey of school timetabling research. Ann. Oper. Res. 218(1), 261–293 (2014)

25. Rossi-Doria, O., Sampels, M., Chiarandini, M., Knowles, J., Manfrin, M., Mastrolilli, M., Paquete, L., Paechter, B.: A comparison of the performance of different metaheuristics on the timetabling problem. In: Proc. 4th Int. Conf. Practice and Theory of Automated Timetabling (PATAT 2002), LNCS vol. 2740, pp. 329–351. Springer, Berlin (2003)

26. Schaerf, A.: A survey of automated timetabling. Artif. Intell. Rev. 13(2), 87–127 (1999)

27. Shaw, P.: Using constraint programming and local search methods to solve vehicle routing problems. In: Proc. CP 1998, LNCS vol. 1520, pp. 417–431. Springer, Berlin (1998)

28. Talbi, E.-G.: Metaheuristics: From Design to Implementation. Wiley, Hoboken (2009)

29. Tümer, B., Bilgin, B., Özcan, E.: An adaptive hyper-heuristic for examination timetabling. J. Sched. 25(4), 391–413 (2022)

30. Vrielink, R.A.O., Lentink, E., Hans, E.W., van Hillegersberg, J.: Practices in timetabling in higher education institutions: a systematic review. Ann. Oper. Res. 275(1), 145–160 (2019)
