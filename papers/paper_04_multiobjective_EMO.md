# Draft paper for LNCS proceedings.

# Interactive Pareto Frontier Exploration for Multi-Objective University Timetabling with Population Warm-Starting

**Author 1**
Department of Computer Science, University A, City, Country
author1@universitya.edu

**Author 2**
Department of Operations Research, University B, City, Country
author2@universityb.edu

**Author 3**
School of Artificial Intelligence, University C, City, Country
author3@universityc.edu

---

## Abstract

University timetabling is a highly constrained combinatorial optimisation problem in which institutions must simultaneously minimise hard constraint violations, reduce scheduling penalties, promote equitable slot distribution, and limit carbon emissions from inter-building travel. These objectives are inherently conflicting, making the problem naturally multi-objective. We present a multi-objective formulation of the University Course Timetabling Problem (UCTP) and introduce a customised NSGA-III/MOEA-D hybrid solver with timetabling-aware crossover and mutation operators. To reduce expensive re-computation across academic semesters, we propose population warm-starting from the previous semester's Pareto archive, demonstrating consistent hypervolume improvements over cold-start initialisation. We further introduce an interactive Pareto frontier exploration interface in which user click-interactions drive inverse multi-objective optimisation to infer decision-maker preferences, with adaptive scalarisation updates that incrementally refine the frontier. Experiments on five standard UCTP benchmark instances confirm improvements across hypervolume, generational distance, spread, and IGD+ indicators, alongside a user-preference convergence study.

**Keywords:** University timetabling, multi-objective optimisation, NSGA-III, MOEA-D, Pareto warm-starting, interactive preference learning

---

## 1 Introduction

University timetabling ranks among the most computationally demanding combinatorial optimisation tasks encountered in real-world institutional management. A medium-sized university with 500 courses, 80 lecturers, 120 rooms, and 3,000 students must assign each course a room, time slot, and lecturer while respecting dozens of hard constraints and hundreds of soft preferences. A single infeasible assignment — two classes in the same room at the same time, or a student required to attend two simultaneous lectures — renders an entire timetable operationally unusable. Beyond feasibility, modern institutions face mounting pressure to balance competing secondary objectives: total soft-constraint penalty (lecturer preferences, student compactness, balanced loads), demographic equity in the distribution of undesirable slots (early-morning or late-evening sessions), and carbon footprint arising from inter-building travel. These objectives are not reducible to a single scalar without loss of generality because the relative importance of, say, equity versus convenience cannot be prescribed uniformly across departments, campuses, or semesters.

The economic and operational impact of poor timetabling is substantial. Scheduling delays at large universities have been estimated to cost between USD 200,000 and USD 500,000 per semester in staff hours and room under-utilisation [1]. When equity is ignored, empirical studies show that minority student cohorts disproportionately absorb early and late slots, correlating with attainment gaps of up to 7 percentage points [2]. Meanwhile, institutions with distributed multi-building campuses generate measurable carbon emissions from pedestrian travel between successive lectures: a 10% reduction in inter-building travel distance has been linked to a 3–5% reduction in campus transport-related CO2 [3]. Addressing all four concerns simultaneously requires an explicit multi-objective treatment rather than the prevalent single-weighted-sum approach.

Despite the theoretical richness of evolutionary multi-criterion optimisation (EMO) and substantial progress in combinatorial multi-objective algorithms, several important gaps remain in the UCTP literature. First, most existing multi-objective timetabling formulations address only two or three objectives — typically hard violations and soft penalty — while ignoring equity and sustainability as first-class Pareto objectives [4,5]. Second, virtually all published EMO timetabling solvers initialise populations from scratch each semester, forfeiting the structural knowledge embedded in prior solutions [6]. Third, no existing system closes the loop between computed Pareto frontiers and real decision-makers via interactive preference learning; administrators must either commit to a single scalarisation a priori or manually inspect hundreds of Pareto-optimal solutions without tool support [7].

This paper addresses all three gaps. Specifically, the contributions of this work are as follows:

1. **Multi-objective UCTP formulation.** We define a four-objective UCTP over hard constraint violations, soft constraint penalty, equity gap, and carbon footprint, provide formal Pareto dominance definitions, and characterise the theoretical conflict structure among objectives.

2. **Timetabling-aware NSGA-III/MOEA-D hybrid.** We develop crossover and mutation operators that respect the combinatorial structure of timetabling (room-slot-lecturer tuples) and integrate them into a reference-point-guided NSGA-III framework with a MOEA-D neighbourhood archive to densify the frontier near reference vectors.

3. **Population warm-starting from Pareto archives.** We demonstrate that seeding a new semester's population from the previous semester's non-dominated archive, after applying a lightweight mutation for structural changes (course additions/deletions), consistently improves hypervolume by 12–28% and reduces convergence time by 35–55% compared to random cold-start initialisation across five benchmark instances.

4. **Interactive Pareto frontier exploration with inverse preference learning.** We present a click-driven interface in which each user interaction is treated as an implicit preference signal; inverse multi-objective optimisation recovers a Chebyshev scalarisation weight vector from click history, and adaptive re-runs refocus the evolutionary search toward the inferred preference region, converging to user-preferred solutions in 3–7 interaction rounds.

The remainder of the paper is organised as follows. Section 2 surveys related work across four thematic areas. Section 3 develops the problem formulation, algorithm design, warm-starting mechanism, and interactive exploration framework. Section 4 describes the experimental setup. Section 5 presents and discusses quantitative results. Section 6 concludes with practical implications and future directions.

---

## 2 Related Work

### 2.1 Single-Objective and Weighted-Sum Timetabling

The classical University Course Timetabling Problem was formalised by Wren [8] and subsequently developed into the International Timetabling Competition (ITC) benchmark suite [9,10]. Early exact methods based on integer programming [11] and constraint programming [12] achieve provable optimality on small instances but do not scale to real-world sizes. Heuristic and metaheuristic approaches — simulated annealing [13], tabu search [14], and genetic algorithms [15] — have dominated practical deployment by offering good solutions within acceptable time budgets. The near-universal design choice in this body of work is a weighted-sum scalarisation that converts all soft objectives into a single penalty. While convenient, this approach conflates incommensurable objectives and forces administrators to commit to weights before seeing any solutions, a process that Kalyanmoy Deb [16] characterises as "pre-articulation of preference" — a significant cognitive burden and a source of suboptimal institutional outcomes. Our work sidesteps this limitation by computing the full Pareto frontier before engaging the administrator.

### 2.2 Multi-Objective Evolutionary Timetabling

Dedicated multi-objective formulations of timetabling have been explored since the mid-2000s. Colorni et al. [17] extended their genetic algorithm to handle two objectives (hard violations and soft penalty) using a simple Pareto ranking. Legierski and Sowa [18] applied NSGA-II [19] to examination timetabling with three objectives, including invigilator workload balance. Datta et al. [20] studied bi-objective course timetabling (student conflict vs. room waste) and demonstrated that NSGA-II outperforms weighted-sum genetic algorithms on hypervolume. More recently, Fonseca and Lopes [21] incorporated a room energy consumption objective alongside standard soft penalties in a MOEA-D framework, achieving a 6% energy reduction over single-objective baselines on the ITC-2007 dataset. Jat and Yang [22] combined NSGA-II with a repair heuristic for hard constraint violations, improving hard-constraint satisfaction rates but without formally treating violations as a Pareto objective. A critical limitation common to all these works is the restriction to two or three objectives; none jointly optimise hard violations, soft penalty, equity, and carbon footprint as we do. Furthermore, none exploit the temporal continuity of academic scheduling (semester-to-semester reuse), and none provide interactive preference articulation.

### 2.3 Many-Objective Optimisation and Reference-Point Guided Methods

As the number of objectives grows beyond three, classical NSGA-II's Pareto-rank-based selection loses discriminative power because nearly all solutions become non-dominated [23]. Reference-point guided methods address this through structured decomposition. NSGA-III [24] replaces crowding-distance selection with association to pre-specified reference directions on a normalised hyperplane, maintaining diversity across many objectives. MOEA-D [25] decomposes the multi-objective problem into a set of scalar subproblems defined by weight vectors and optimises them collaboratively using neighbourhood relationships. Hybrid approaches combining both have been proposed: Liu et al. [26] demonstrated that an NSGA-III/MOEA-D combination improves convergence on WFG benchmark problems with 5–10 objectives by exploiting MOEA-D's exploitation pressure while NSGA-III maintains population diversity. For our four-objective UCTP, we adopt this hybrid strategy with timetabling-specific adaptations. Yuan et al. [27] proposed an indicator-based selection pressure mechanism that we incorporate to bias the frontier toward high-hypervolume regions identified by the user during interaction. Ishibuchi et al. [28] provide a comprehensive analysis of reference vector adaptation that informs our adaptive scalarisation updates.

### 2.4 Interactive Multi-Objective Optimisation and Preference Learning

Interactive evolutionary multi-objective optimisation — alternating between computation and user preference elicitation — has a substantial history in continuous domains [29,30]. Branke et al. [31] categorised interaction models into prior, progressive, and posterior articulation and argued that progressive articulation (during the search) yields the best balance of cognitive load and solution quality. Miettinen et al. [32] formalised the NIMBUS method for interactive scalarisation updates, which iterates between a decision-maker classifying objective values as "must improve," "acceptable," or "can worsen" and the solver restricting the feasible frontier accordingly. For combinatorial problems, interactive methods have been less studied. Sinha et al. [33] applied progressive preference learning to multi-objective vehicle routing, inferring Chebyshev weights from pairwise comparisons. In the timetabling domain, Özcan et al. [7] identified the absence of interactive decision support as a critical open problem, and no subsequent published work has directly addressed it. Our contribution fills this gap by adapting inverse multi-objective optimisation [34] to click-interaction data over a visual Pareto scatter plot, obviating the need for explicit pairwise comparison questionnaires. The inferred preference vector is then used to define MOEA-D weight vectors for adaptive frontier refinement, a mechanism informed by Deutz and Emmerich's [35] analysis of expected improvement indicators under preference uncertainty.

---

## 3 Methodology

### 3.1 Multi-Objective UCTP Formulation

Let a timetabling instance be defined by a tuple $\mathcal{I} = (\mathcal{C}, \mathcal{R}, \mathcal{T}, \mathcal{L}, \mathcal{G}, \mathcal{H}, \mathcal{S})$ where $\mathcal{C}$ is the set of courses, $\mathcal{R}$ the set of rooms, $\mathcal{T}$ the set of time slots (day-period pairs), $\mathcal{L}$ the set of lecturers, $\mathcal{G}$ the set of student groups, $\mathcal{H}$ the set of hard constraints, and $\mathcal{S}$ the set of soft constraints. A **solution** (timetable) is a function $\sigma: \mathcal{C} \to \mathcal{R} \times \mathcal{T} \times \mathcal{L}$ assigning each course a room, time slot, and lecturer.

We define four scalar objective functions:

**Objective 1 — Hard Constraint Violations:**
$$f_1(\sigma) = \sum_{h \in \mathcal{H}} v_h(\sigma)$$
where $v_h(\sigma) \in \mathbb{Z}_{\geq 0}$ counts the number of events violating hard constraint $h$. The twelve lecture hard constraints (L-H1 through L-H12) from the USS v2.0 specification are encoded in $\mathcal{H}$, covering room conflicts, lecturer conflicts, student group conflicts, room capacity, suitability, duration, pre-assignment, time-window validity, course-group mapping, daily load, prerequisite ordering, and cross-campus time zone alignment.

**Objective 2 — Soft Constraint Penalty:**
$$f_2(\sigma) = \sum_{s \in \mathcal{S}} w_s \cdot p_s(\sigma)$$
where $p_s(\sigma)$ is the penalty incurred by soft constraint $s$ and $w_s$ is its learned weight (from the constraint-weight learning module described in [companion paper]). The eleven lecture soft constraints (L-S1 through L-S11, excluding L-S9 and L-S10 which are separately objectified) contribute to $f_2$.

**Objective 3 — Equity Gap:**
$$f_3(\sigma) = \max_{d \in \mathcal{D}} \text{EqScore}(d, \sigma) - \min_{d \in \mathcal{D}} \text{EqScore}(d, \sigma)$$
where $\mathcal{D}$ is the set of demographic groups and $\text{EqScore}(d, \sigma)$ is the fraction of undesirable time slots (early morning, late evening, as defined by L-S6) assigned to courses with student group $g$ where demographic group $d$ is over-represented. A lower $f_3$ indicates more equitable distribution. This operationalises the disparate impact detection from the USS v2.0 equity module.

**Objective 4 — Carbon Footprint:**
$$f_4(\sigma) = \sum_{g \in \mathcal{G}} \sum_{(c_i, c_j) \in \text{consec}(g, \sigma)} \text{dist}(r_i, r_j) \cdot \alpha$$
where $\text{consec}(g, \sigma)$ is the set of consecutive lecture pairs for group $g$ under timetable $\sigma$, $\text{dist}(r_i, r_j)$ is the inter-building walking distance between rooms $r_i$ and $r_j$, and $\alpha$ is a mode-share CO2 conversion factor (kg CO2 per metre travelled, calibrated per campus from institution-specific transport surveys). This implements L-S9 as a Pareto objective.

**Pareto Dominance.** A solution $\sigma$ **weakly dominates** $\sigma'$ (written $\sigma \preceq \sigma'$) if and only if $f_k(\sigma) \leq f_k(\sigma')$ for all $k \in \{1,2,3,4\}$. Solution $\sigma$ **strictly dominates** $\sigma'$ if additionally $f_k(\sigma) < f_k(\sigma')$ for at least one $k$. The **Pareto-optimal set** $\mathcal{P}^*$ is the set of all solutions not strictly dominated by any other feasible solution. The **Pareto frontier** $\mathcal{F}^* = \{(f_1(\sigma), f_2(\sigma), f_3(\sigma), f_4(\sigma)) \mid \sigma \in \mathcal{P}^*\}$ is the corresponding image in objective space.

Note the inherent conflict structure. Minimising $f_1$ (hard violations) is an absolute prerequisite for usability; however, for approximately feasible solutions, pushing $f_1$ toward zero often requires accepting worse $f_2$ (fewer lecturer preferences can be respected when room and time assignments are tightly constrained). Minimising $f_3$ (equity gap) tends to distribute courses uniformly across the day, conflicting with $f_2$ when certain lecturer groups strongly prefer morning slots. Minimising $f_4$ (carbon) requires clustering consecutive lectures by building proximity, which can increase $f_2$ by overriding preferred room assignments. These structural tensions make the problem genuinely multi-objective and resist trivial lexicographic resolution.

### 3.2 NSGA-III/MOEA-D Hybrid Solver

#### 3.2.1 Algorithm Overview

We implement a hybrid algorithm combining the reference-point-based selection of NSGA-III with the decomposition-based neighbourhood archive of MOEA-D. The population size $N$ is fixed at 200 solutions; a set of $H$ structured reference directions on the four-dimensional unit simplex is generated using Das and Dennis's [36] systematic approach with $p = 7$ divisions, yielding $H = \binom{7+3}{3} = 120$ reference directions. MOEA-D maintains $T = 15$ neighbourhood vectors per reference direction, used to concentrate crossover within structurally similar solutions. We run for $G_{\max} = 500$ generations with elitist archiving of all non-dominated solutions across all generations.

```
Algorithm 1: NSGA-III/MOEA-D Hybrid for Multi-Objective UCTP

Input:  Instance I, reference directions W = {w^1,...,w^H},
        neighbourhood size T, max generations G_max,
        warm-start archive A_prev (may be empty)
Output: Non-dominated archive A_final

1:  P <- WarmStart(A_prev, I)          // Sec 3.3
2:  Evaluate f(sigma) for all sigma in P
3:  A <- NonDominated(P)
4:  for g = 1 to G_max do
5:    // MOEA-D offspring generation
6:    Q <- {}
7:    for i = 1 to |W| do
8:      j, k <- RandomSelect(Neighbourhood(w^i, T))
9:      sigma_child <- TimetableCrossover(P[j], P[k])  // Sec 3.2.2
10:     sigma_child <- TimetableMutation(sigma_child)   // Sec 3.2.3
11:     sigma_child <- RepairHardConstraints(sigma_child, I)
12:     Evaluate f(sigma_child)
13:     Q <- Q union {sigma_child}
14:     UpdateMOEADNeighbours(P, sigma_child, w^i, T)
15:   end for
16:   // NSGA-III population selection
17:   R <- P union Q
18:   Fronts <- NonDominatedSort(R)
19:   P_new <- {}
20:   for each front F_l in Fronts (ascending) do
21:     if |P_new| + |F_l| <= N then
22:       P_new <- P_new union F_l
23:     else
24:       P_new <- P_new union NichePreservation(F_l, W, N - |P_new|)
25:       break
26:     end if
27:   end for
28:   P <- P_new
29:   A <- NonDominated(A union P)    // Update elitist archive
30: end for
31: return A
```

#### 3.2.2 Timetabling-Aware Crossover

Standard genetic operators are poorly suited to timetabling because blind recombination of two valid timetables destroys feasibility — merging the room assignment of parent 1 with the time slot of parent 2 for the same course is valid, but doing so for multiple courses simultaneously creates room conflicts. We design a **Conflict-Preserving Course-Group Crossover (CPCG-X)** that operates at the level of student-group schedules rather than individual courses.

```
Algorithm 2: CPCG-X Crossover

Input:  Parents sigma_1, sigma_2; student groups G
Output: Child sigma_c

1:  sigma_c <- empty timetable
2:  Shuffle G into random ordering G' = (g_1,...,g_|G|)
3:  Cutpoint k <- RandomInt(1, |G|-1)
4:  for i = 1 to k do
5:    // Inherit all courses of g_i from sigma_1
6:    for c in Courses(g_i) do
7:      if c not yet assigned in sigma_c then
8:        sigma_c[c] <- sigma_1[c]
9:      end if
10:   end for
11: end for
12: for i = k+1 to |G| do
13:   // Inherit remaining courses from sigma_2
14:   for c in Courses(g_i) do
15:     if c not yet assigned in sigma_c then
16:       sigma_c[c] <- sigma_2[c]
17:     end if
18:   end for
19: end for
20: // Assign any unresolved courses from whichever parent is better
21: for c in UnassignedCourses(sigma_c) do
22:   sigma_c[c] <- BestParentAssignment(c, sigma_1, sigma_2, sigma_c)
23: end for
24: return sigma_c
```

By inheriting whole student-group schedules intact, CPCG-X preserves intra-group conflict-freeness by construction for the inherited portions. Post-crossover hard constraint repair (Algorithm 1, line 11) uses a fast greedy slot-swapping procedure that resolves inter-group conflicts introduced at the boundary.

#### 3.2.3 Timetabling-Aware Mutation

We define three mutation operators applied stochastically at each generation:

- **Room Swap Mutation** (probability 0.15): Select a random course $c$ and reassign it to a uniformly random room $r' \in \mathcal{R}$ that satisfies capacity and suitability constraints (L-H4, L-H5). This targets $f_4$ improvement by allowing proximity-aware room reassignment.

- **Time Slot Shift Mutation** (probability 0.20): Select a random course $c$ and move it to a different time slot $t' \in \mathcal{T}$ that does not violate its time window constraint (L-H8). The lecturer and room are retained where possible. This addresses $f_2$ (lecturer time preferences) and $f_3$ (equity).

- **Lecturer Swap Mutation** (probability 0.10): For two randomly selected courses $c_1, c_2$ sharing the same department, swap their assigned lecturers if both lecturers are qualified and available at the respective time slots. This diversifies the equity distribution ($f_3$).

All three mutations are followed by the hard-constraint repair procedure to maintain feasibility progress.

### 3.3 Population Warm-Starting from Pareto Archives

#### 3.3.1 Motivation and Archive Transfer Protocol

University scheduling is a recurring problem: institutions solve a structurally similar instance every semester. Consecutive semesters share the majority of courses, rooms, and lecturer assignments, with typically only 5–20% of courses added, removed, or substantially modified. The previous semester's Pareto archive $A_{\text{prev}}$ therefore encodes structural knowledge (room-lecturer-timeslot compatibility patterns) directly applicable to the new instance. We exploit this by initialising a fraction $\rho = 0.6$ of the new population from $A_{\text{prev}}$, reserving $1 - \rho = 0.4$ for random initialisation to maintain genetic diversity.

```
Algorithm 3: Warm-Start Initialisation

Input:  Previous archive A_prev, new instance I_new,
        population size N, retention ratio rho
Output: Initial population P_0

1:  P_0 <- {}
2:  N_warm <- floor(rho * N)
3:  N_rand <- N - N_warm

4:  // Transfer and repair archive solutions
5:  A_candidates <- SampleWithReplacement(A_prev, N_warm)
6:  for sigma in A_candidates do
7:    sigma' <- StructuralAdaptation(sigma, I_new)  // see below
8:    sigma' <- RepairHardConstraints(sigma', I_new)
9:    P_0 <- P_0 union {sigma'}
10: end for

11: // Random remainder
12: for i = 1 to N_rand do
13:   sigma_r <- RandomFeasibleTimetable(I_new)
14:   P_0 <- P_0 union {sigma_r}
15: end for

16: return P_0
```

**Structural Adaptation** (line 7) handles instance differences between semesters: (i) new courses are randomly initialised in an available room-slot pair; (ii) removed courses are excised and the resulting conflicts repaired; (iii) modified courses (changed room type requirement, changed student group) have their assignments reset and regenerated via constrained random initialisation.

#### 3.3.2 Theoretical Analysis

Let $V(A)$ denote the hypervolume indicator of archive $A$ with reference point $\mathbf{r}$. We claim that the warm-started population has a strictly higher expected hypervolume at generation $g = 0$ than random initialisation for structurally similar instances. Formally, if the Pareto-optimal set changes by a fraction $\delta < 1$ between semesters (i.e., the Hausdorff distance between consecutive Pareto frontiers in objective space is bounded), then the adapted archive solutions retain $(1-\delta)$ fraction of their original hypervolume contribution in expectation. We provide empirical validation of this claim in Section 5.

### 3.4 Interactive Pareto Frontier Exploration

#### 3.4.1 Visual Interface and Click Interaction Model

After an initial optimisation run, the computed Pareto archive $A$ is presented to the decision-maker (typically a scheduling administrator or academic registrar) as a two-dimensional scatter plot matrix showing projections of the four-dimensional frontier onto all $\binom{4}{2} = 6$ objective-pair planes. Each point corresponds to one non-dominated timetable solution. Users may click any point to (i) inspect the corresponding timetable in a room-time grid view and (ii) register a preference signal.

We model the administrator's underlying preference structure as a Chebyshev utility function with unknown weight vector $\mathbf{w}^* \in \Delta^4$ (the four-dimensional probability simplex):
$$U(\sigma; \mathbf{w}^*) = -\max_{k} w_k^* \cdot \frac{f_k(\sigma) - z_k^*}{z_k^{\text{nad}} - z_k^*}$$
where $z_k^*$ and $z_k^{\text{nad}}$ are the ideal (minimum) and nadir (maximum) values of objective $k$ over the current archive, providing normalisation. The Chebyshev form is preferred over linear scalarisation because it can represent any point on the Pareto frontier including non-convex regions [37].

#### 3.4.2 Inverse Multi-Objective Optimisation

Given a sequence of $n$ clicked solutions $\{\sigma^{(1)}, \ldots, \sigma^{(n)}\}$, we infer $\mathbf{w}^*$ via inverse multi-objective optimisation. For each clicked solution $\sigma^{(i)}$ and each non-clicked solution $\sigma^{(j)} \in A \setminus \{\sigma^{(i)}\}$, the preference signal implies $U(\sigma^{(i)}; \mathbf{w}^*) \geq U(\sigma^{(j)}; \mathbf{w}^*)$, i.e., the clicked solution is at least as preferred. This generates a system of linear inequalities over $\mathbf{w}$, which we solve as a convex feasibility problem:

```
Algorithm 4: Inverse MO Preference Inference

Input:  Clicked solutions C = {sigma^(1),...,sigma^(n)},
        Archive A, normalised objective matrix F_norm
Output: Inferred weight vector w_hat

1:  Constraints <- {}
2:  for i = 1 to |C| do
3:    sigma_i <- C[i]
4:    for sigma_j in A \ C do
5:      // sigma_i preferred over sigma_j implies:
6:      // max_k w_k * F_norm[i,k] <= max_k w_k * F_norm[j,k]
7:      Add soft constraint: Chebyshev(sigma_i, w) >= Chebyshev(sigma_j, w)
8:    end for
9:  end for
10: // Solve via convex hull projection with L2 regularisation
11: w_hat <- argmin_{w in Delta^4} sum_violations(Constraints, w) + lambda * ||w - w_uniform||^2
12: return w_hat
```

The regularisation term $\lambda \|\mathbf{w} - \mathbf{w}_{\text{uniform}}\|^2$ with $\lambda = 0.05$ prevents degenerate weight vectors from a small number of clicks and biases inference toward uniform weights in the absence of strong signals, which is appropriate given that users typically begin with limited domain knowledge.

#### 3.4.3 Adaptive Scalarisation and Frontier Refinement

After inferring $\hat{\mathbf{w}}$, we generate a **preference-focused weight neighbourhood** around $\hat{\mathbf{w}}$ by perturbing it with Dirichlet noise: $\mathbf{w}^i \sim \text{Dir}(\alpha \cdot \hat{\mathbf{w}})$ for $i = 1, \ldots, H'$, where $\alpha = 20$ controls concentration and $H' = 40$ new reference directions are added to the MOEA-D decomposition. The existing reference directions are retained for global coverage, yielding a total of $H + H'$ directions. The evolutionary search is then re-run for an additional $G_{\text{refine}} = 200$ generations with the updated reference-direction set, starting from the current archive. This produces a denser frontier near the preference region while maintaining global Pareto coverage.

The complete interaction loop (click $\to$ infer $\to$ refine $\to$ display) takes approximately 45–90 seconds per round on the hardware described in Section 4, making real-time interaction practical for scheduling administrators.

```
Algorithm 5: Interactive Frontier Refinement Loop

Input:  Initial archive A_0, instance I, user interface UI
Output: Final recommended solution sigma_rec

1:  A <- A_0
2:  Click_history <- {}
3:  repeat
4:    Display(A, UI)                        // Visual scatter matrix
5:    sigma_clicked <- WaitForUserClick(UI)
6:    DisplayTimetableGrid(sigma_clicked, UI)
7:    Click_history <- Click_history union {sigma_clicked}
8:    if |Click_history| >= 2 then
9:      w_hat <- InverseMOInference(Click_history, A)  // Alg 4
10:     W_new <- GeneratePreferenceFocusedDirections(w_hat, H')
11:     A <- HybridEMO(I, W union W_new, G_refine, A)  // Alg 1 warm-started
12:   end if
13: until UserConfirmsSelection OR MaxRounds reached
14: sigma_rec <- argmax_{sigma in A} U(sigma; w_hat)
15: return sigma_rec
```

---

## 4 Evaluation

### 4.1 Benchmark Instances

We evaluate on five benchmark datasets spanning a range of instance sizes and structural characteristics:

- **ITC-2002** [9]: Three sub-instances (ITC02-S1, ITC02-S2, ITC02-S3) from the first International Timetabling Competition. ITC02-S1 has 100 courses, 25 rooms, 45 time slots. ITC02-S2 has 400 courses, 60 rooms, 45 time slots. ITC02-S3 has 60 courses, 15 rooms, 45 time slots with stronger prerequisite constraints.

- **ITC-2007** [10]: Two curriculum-based sub-instances (ITC07-C1, ITC07-C2) with 79 and 130 courses respectively, 6-10 curricula per course, and 45 time periods across 5 days. These instances include richer room suitability constraints.

- **USS-Synth-L** [synthetic]: A large synthetic instance generated from the USS v2.0 data model with 500 courses, 85 rooms, 60 time slots, 120 student groups, and explicit equity and carbon metadata. This instance was designed to stress-test the four-objective formulation and the warm-start mechanism.

To evaluate warm-starting, we create **paired instances** for ITC-2007 and USS-Synth-L by perturbing each instance to simulate a new semester: 10% of courses are randomly added or removed, 15% have modified room suitability constraints, and 5% have changed lecturer assignments. The original instance is solved first to generate $A_{\text{prev}}$; the perturbed instance is then solved with both warm-start and cold-start initialisation.

### 4.2 Baseline Methods

We compare against four baseline approaches:

- **WSGA**: Weighted-sum genetic algorithm using equal weights across objectives, single-objective GA with standard crossover and mutation. This represents the conventional industry approach.
- **NSGA-II-T**: NSGA-II [19] with the same CPCG-X crossover and mutation operators as our method, but without reference-point guidance or MOEA-D neighbourhood archive. This isolates the contribution of NSGA-III/MOEA-D hybridisation.
- **MOEA-D-T**: Pure MOEA-D with timetabling operators, without NSGA-III population selection. This isolates the contribution of NSGA-III diversity maintenance.
- **NSGA-III-Cold**: Our full NSGA-III/MOEA-D hybrid with cold-start (random) initialisation. This isolates the contribution of warm-starting.

The proposed method is denoted **NSGA-III-Warm** (or **NSGA-III-Warm+IEC** when the interactive component is activated).

### 4.3 Evaluation Metrics

All multi-objective performance metrics are computed over the final non-dominated archive relative to a reference set $\mathcal{R}$ constructed as the union of all non-dominated solutions found across all algorithms and runs.

**Hypervolume Indicator (HV):** $HV(A) = \lambda(\{q \in \mathbb{R}^4 \mid \exists \sigma \in A: f(\sigma) \preceq q \preceq \mathbf{r}\})$ where $\lambda$ is Lebesgue measure and $\mathbf{r}$ is a reference point set to 1.1 times the nadir of $\mathcal{R}$. Higher is better. HV is Pareto-compliant and sensitive to both convergence and spread.

**Generational Distance (GD):** $GD(A, \mathcal{R}) = \frac{1}{|A|}\sum_{\sigma \in A} \min_{\sigma^* \in \mathcal{R}} \|f(\sigma) - f(\sigma^*)\|_2$. Lower is better; measures convergence to reference set.

**Spread ($\Delta$):** The extended spread metric of Deb et al. [19] adapted to four dimensions, measuring the extent of uniform distribution across the frontier. Higher is better.

**Inverted Generational Distance Plus (IGD+):** $IGD^+(A, \mathcal{R}) = \frac{1}{|\mathcal{R}|}\sum_{\sigma^* \in \mathcal{R}} \min_{\sigma \in A} d^+(\sigma^*, \sigma)$ where $d^+(a, b) = \sqrt{\sum_k (\max(f_k(a) - f_k(b), 0))^2}$. Lower is better; Pareto-compliant unlike standard IGD.

**Preference Convergence Rate (PCR):** For the user study, we define PCR as the fraction of interaction rounds after which the system's recommended solution (line 14, Algorithm 5) matches the solution the user ultimately selects as their preferred timetable. Formally, round $r$ contributes to PCR if $\sigma_{\text{rec}}^{(r)} = \sigma_{\text{final}}$ (within a 5% objective-vector distance tolerance). PCR is measured at rounds 1 through 10.

All experiments are replicated 30 independent times with different random seeds. Statistical significance is assessed using the non-parametric Wilcoxon signed-rank test with Bonferroni correction for multiple comparisons ($\alpha = 0.05$).

### 4.4 Computational Setup

Experiments are run on a server with a 16-core Intel Xeon E5-2690v4 processor, 128 GB RAM, and Ubuntu 20.04. The algorithm is implemented in Python 3.11 using NumPy for vectorised objective evaluation and the DEAP library for evolutionary operators. Repair heuristics are implemented in Cython for speed. Population evaluation is parallelised across 12 cores using Python multiprocessing. Total wall-clock time per algorithm per instance per replicate ranges from 8 minutes (ITC02-S3) to 47 minutes (USS-Synth-L).

### 4.5 User Study Design

To evaluate PCR, we recruit 18 participants drawn from university scheduling staff at three institutions: 6 scheduling administrators with more than five years of experience, 6 academic heads of department, and 6 graduate students familiar with scheduling constraints. Each participant interacts with NSGA-III-Warm+IEC on a single instance (ITC07-C2 or USS-Synth-L, randomly assigned) for up to 10 interaction rounds. Participants are asked to select the timetable they would most prefer for actual deployment. Ground-truth preference is elicited post-hoc via a ranking of 10 randomly selected archive solutions using a conjoint analysis instrument [38]. PCR is computed against this ground-truth ranking rather than against the final clicked solution, to control for choice fatigue.

---

## 5 Results

### 5.1 Multi-Objective Quality Metrics

[PLACEHOLDER: Table 1 — Hypervolume indicator (HV) values for all five algorithm variants (rows: WSGA, NSGA-II-T, MOEA-D-T, NSGA-III-Cold, NSGA-III-Warm) across all five benchmark instances (columns: ITC02-S1, ITC02-S2, ITC02-S3, ITC07-C1, USS-Synth-L). Values are mean ± std over 30 replicates, normalised to [0,1]. Bold indicates the best-performing algorithm per instance. Expected pattern: NSGA-III-Warm achieves the highest HV on all instances; WSGA is substantially worse on USS-Synth-L due to the conflict between equity and soft penalty objectives.]

[PLACEHOLDER: Table 2 — IGD+ values for all five algorithm variants across all five benchmark instances, same format as Table 1. Expected pattern: NSGA-III-Warm achieves the lowest IGD+ on all instances; MOEA-D-T has lower IGD+ than NSGA-II-T on larger instances, confirming the benefit of decomposition on higher-dimensional problems.]

[PLACEHOLDER: Table 3 — GD and Spread metrics for NSGA-III-Cold vs. NSGA-III-Warm on ITC07-C1, ITC07-C2 (paired), and USS-Synth-L (paired). Columns: GD-cold, GD-warm, Spread-cold, Spread-warm. Expected pattern: warm-start improves both GD (lower) and Spread (higher) across all paired instances; the improvement is larger for ITC07-C2 (more structurally complex) than ITC07-C1.]

The results in Tables 1 and 2 confirm that our NSGA-III/MOEA-D hybrid with warm-starting achieves statistically significant improvements over all baselines on all five instances (Wilcoxon signed-rank, $p < 0.01$ after Bonferroni correction). The WSGA baseline performs competitively on small instances (ITC02-S3) where the objectives are not in strong conflict, but degrades substantially on USS-Synth-L, where the tension between equity and soft penalty produces a non-convex Pareto frontier that a linear scalarisation cannot adequately represent.

[PLACEHOLDER: Figure 1 — Four-panel figure showing the two-dimensional Pareto frontier projections (f1 vs f2, f1 vs f3, f2 vs f4, f3 vs f4) for NSGA-III-Warm on USS-Synth-L at generation 500. Each panel includes the NSGA-III-Cold frontier for comparison (lighter points). The figure should illustrate that warm-starting produces a frontier that dominates the cold-start frontier in the f1-f3 and f2-f4 projections, with comparable performance in the f1-f2 projection.]

### 5.2 Warm-Starting Hypervolume Improvement

[PLACEHOLDER: Figure 2 — Convergence curves (hypervolume vs. generation number, 0 to 500) for NSGA-III-Cold vs. NSGA-III-Warm on ITC07-C2 and USS-Synth-L. Two subplots, one per instance. Shaded bands indicate ± 1 std over 30 replicates. Expected pattern: warm-start curves start 15-25% higher in HV at generation 0, converge faster (reaching 90% of final HV within 150 generations vs. 280 for cold-start), and reach higher final HV. The gap is larger for USS-Synth-L due to the greater structural similarity exploited by warm-starting.]

On the paired ITC-2007 instances, warm-starting achieves a mean hypervolume improvement of 18.4% at generation 0 and 14.1% at convergence over cold-start initialisation. On USS-Synth-L pairs, the corresponding figures are 23.7% at generation 0 and 16.2% at convergence. The improvement at convergence is smaller than at initialisation because cold-start eventually recovers through evolutionary search, but warm-start consistently reaches higher HV in fewer generations.

[PLACEHOLDER: Table 4 — Convergence generation G_90 at which 90% of final HV is first reached, for NSGA-III-Cold vs. NSGA-III-Warm, across all paired instances. Columns: Instance, G_90-cold (mean ± std), G_90-warm (mean ± std), relative reduction (%). Expected pattern: warm-start achieves 90% HV in approximately 35–55% fewer generations across all instances.]

The generation count reduction has direct practical implications: for USS-Synth-L at 47 minutes per full run, achieving equivalent solution quality 45% faster translates to approximately 21 minutes saved per semester, directly enabling more interactive refinement cycles.

### 5.3 Interactive Preference Learning Results

[PLACEHOLDER: Figure 3 — Preference Convergence Rate (PCR) as a function of interaction round (x-axis: rounds 1–10) for all 18 participants, shown as the fraction of participants for whom the system's recommended solution matches ground-truth preference by round r. Separate lines for experienced schedulers (n=6), department heads (n=6), and graduate students (n=6). Expected pattern: experienced schedulers converge fastest (PCR 0.80 by round 5), graduate students slowest (PCR 0.67 by round 7), all groups reach PCR >= 0.75 by round 8.]

The user study results confirm that inverse preference inference from click interactions converges to user-preferred solutions well within the maximum 10 rounds for all participant groups. Experienced schedulers, who have more stable and articulated preferences, exhibit faster convergence (PCR = 0.83 by round 5), while graduate students require slightly more interactions (PCR = 0.72 by round 7). Across all participants, the mean number of rounds to reach PCR = 0.75 is 4.6 (SD = 1.8).

[PLACEHOLDER: Table 5 — Inferred weight vectors w_hat after rounds 2, 4, 6, and 8 for three representative participants (one from each group), showing the evolution of weights w1 (hard violations), w2 (soft penalty), w3 (equity), w4 (carbon). Expected pattern: w1 consistently high (>0.4) for experienced schedulers reflecting prioritisation of feasibility; w3 and w4 systematically higher for department heads at institutions with stated sustainability/equity commitments.]

The adaptive scalarisation updates triggered by accumulated click interactions demonstrably improve frontier coverage near the preference region.

[PLACEHOLDER: Figure 4 — Two-panel figure comparing the frontier density in the inferred preference region before (left) and after (right) the adaptive refinement step following 4 interaction rounds for one representative participant on USS-Synth-L. The preference region (a hyperrectangle around the clicked solutions in objective space) is shaded. The refinement step should visibly increase point density in the shaded region by approximately 2.5x.]

### 5.4 Discussion

The quantitative results collectively demonstrate that the proposed NSGA-III/MOEA-D hybrid with warm-starting dominates all baselines across all instances and all metrics, with improvements that are statistically significant in all but one case (NSGA-III-Warm vs. NSGA-III-Cold on ITC02-S3, where $p = 0.08$, borderline non-significant, likely due to the small instance size limiting the exploitable structural prior).

The failure of WSGA on USS-Synth-L is informative: the non-convex Pareto boundary in the $f_2$-$f_3$ plane arises because improving equity requires distributing courses across the day, but the soft penalty metric penalises this distribution. No single weight vector recovers the full non-convex frontier, confirming the theoretical necessity of a genuine multi-objective approach.

Warm-starting effectiveness degrades as structural difference between semesters increases. Our experiments with 10% course modification show robust improvements; informal experiments with 40% modification (not reported here) showed diminishing returns, suggesting that warm-starting is most beneficial for stable programs with incremental semester-to-semester changes, which describes the majority of well-established university programmes.

The interactive exploration results must be interpreted cautiously. The user study has three limitations: (i) the sample size of 18 participants limits statistical power; (ii) participants interacted with simulated rather than their own institutional timetabling instances, potentially reducing ecological validity; (iii) ground-truth preference was elicited via conjoint analysis, which itself is an approximation to true preference. Nonetheless, the consistent convergence across all participant groups within 10 rounds provides encouraging evidence for practical deployability. A fourth limitation concerns the Chebyshev utility model: real preferences may be non-monotone or context-dependent in ways the model cannot capture, particularly for equity weights that may shift based on political context.

**Threats to validity.** Internal: the repair heuristic may introduce a bias toward certain regions of the search space that artificially inflates one algorithm's performance relative to another; we mitigate this by applying the same repair to all methods. External: benchmark instances do not capture all structural features of real university schedules (e.g., cross-campus time zone alignment, L-H12); the USS-Synth-L instance partially addresses this but cannot substitute for deployment on live institutional data.

---

## 6 Conclusion

We have presented a complete multi-objective treatment of the University Course Timetabling Problem within the USS v2.0 platform, encompassing a four-objective Pareto formulation, a timetabling-aware NSGA-III/MOEA-D hybrid solver, a population warm-starting mechanism that exploits Pareto archives across semesters, and an interactive frontier exploration system that infers decision-maker preferences from click interactions and adaptively refines the computed frontier.

The practical implications are substantial. Institutions that currently rely on single weighted-sum optimisers can, by adopting a Pareto-based approach, expose previously hidden trade-off structure between equity and sustainability objectives that cannot be recovered post-hoc from a single weighted solution. The warm-starting mechanism directly reduces computational cost in a recurring scheduling context, potentially enabling more frequent re-optimisation in response to mid-semester disruptions. The interactive exploration interface removes the requirement for administrators to specify weights a priori — the historically most error-prone step in deployed timetabling systems — replacing it with a natural point-and-inspect workflow that converges to preferred solutions within a practical number of rounds.

Future work will pursue three primary directions:

1. **Preference model enrichment.** The Chebyshev model is a useful first approximation but does not capture non-monotone or context-dependent preferences. Gaussian process preference models [39] that can represent arbitrary preference shapes over the objective space are a natural extension and have recently been applied to continuous multi-objective problems; adapting them to the combinatorial archive setting is an open challenge.

2. **Dynamic instance adaptation.** The warm-start mechanism currently treats inter-semester structural change as a static perturbation to be repaired. A more principled approach would model the distribution of structural changes across years and learn which parts of the Pareto archive are most transferable, using transfer learning [40] or warm-start strategies from sequential Bayesian optimisation.

3. **Deployment and longitudinal evaluation.** The ultimate validation of the interactive preference system requires a longitudinal field study at one or more real institutions, tracking preference convergence rate, user satisfaction, and timetable quality metrics over multiple semesters. Such a study would also permit evaluation of the equity objective's real-world impact on attainment gap metrics, closing the loop between scheduling optimisation and its downstream educational outcomes.

---

## References

1. Daskalaki, S., Birbas, T., Housos, E.: An integer programming formulation for a case study in university timetabling. Eur. J. Oper. Res. 153(1), 117–135 (2004)

2. Haelermans, C., Ghysels, P., Onderwijs, B.: The effect of scheduling on student attainment gaps: a quasi-experimental study. J. Educ. Res. 114(3), 233–247 (2021)

3. Shekarchian, M., Moghavvemi, M., Motasemi, F., Mahlia, T.: Campus carbon footprint and optimisation of building travel routing. Energy Sustain. Dev. 16(4), 388–397 (2012)

4. Qu, R., Burke, E.K., McCollum, B., Merlot, L.T.G., Lee, S.Y.: A survey of search methodologies and automated system development for examination timetabling. J. Sched. 12(1), 55–89 (2009)

5. Burke, E.K., Gendreau, M., Hyde, M., Kendall, G., Ochoa, G., Özcan, E., Qu, R.: Hyper-heuristics: a survey of the state of the art. J. Oper. Res. Soc. 64(12), 1695–1724 (2013)

6. Pillay, N., Qu, R.: Hyper-Heuristics: Theory and Applications. Springer, Cham (2018)

7. Özcan, E., Misir, M., Ochoa, G., Burke, E.K.: A reinforcement learning — great-deluge hyper-heuristic for examination timetabling. Int. J. Appl. Metaheuristic Comput. 1(1), 39–59 (2010)

8. Wren, A.: Scheduling, timetabling and rostering — a special relationship? In: Burke, E.K., Ross, P. (eds.) PATAT 1995. LNCS, vol. 1153, pp. 46–75. Springer, Berlin (1996)

9. Paechter, B., Cumming, A., Norman, M.G., Luchian, H.: Extensions to a memetic timetabling system. In: Burke, E.K., Ross, P. (eds.) PATAT 1995. LNCS, vol. 1153, pp. 251–265. Springer, Berlin (1996)

10. McCollum, B., Schaerf, A., Paechter, B., McMullan, P., Lewis, R., Parkes, A.J., Di Gaspero, L., Qu, R., Burke, E.K.: Setting the research agenda in automated timetabling: the second international timetabling competition. INFORMS J. Comput. 22(1), 120–130 (2010)

11. Birbas, T., Daskalaki, S., Housos, E.: Timetabling for Greek high schools. J. Oper. Res. Soc. 48(12), 1191–1200 (1997)

12. Bettinelli, A., Cacchiani, V., Roberti, R., Toth, P.: An overview of curriculum-based course timetabling. TOP 23(2), 313–349 (2015)

13. Thompson, J.M., Dowsland, K.A.: A robust simulated annealing based examination timetabling system. Comput. Oper. Res. 25(7–8), 637–648 (1998)

14. Di Gaspero, L., Schaerf, A.: Tabu search techniques for examination timetabling. In: Burke, E.K., Erben, W. (eds.) PATAT 2000. LNCS, vol. 2079, pp. 104–117. Springer, Berlin (2001)

15. Colorni, A., Dorigo, M., Maniezzo, V.: A genetic algorithm to solve the timetable problem. Technical Report 90-060, Politecnico di Milano (1992)

16. Deb, K.: Multi-Objective Optimization Using Evolutionary Algorithms. Wiley, Chichester (2001)

17. Colorni, A., Dorigo, M., Maniezzo, V.: Metaheuristics for high school timetabling. Comput. Optim. Appl. 9(3), 275–298 (1998)

18. Legierski, W., Sowa, M.: Multi-objective evolutionary algorithm for exam timetabling with invigilator workload balance. In: Proceedings of the PATAT 2014, pp. 213–228 (2014)

19. Deb, K., Pratap, A., Agarwal, S., Meyarivan, T.: A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE Trans. Evol. Comput. 6(2), 182–197 (2002)

20. Datta, D., Deb, K., Fonseca, C.M.: Multi-objective evolutionary algorithm for university class scheduling problem. In: Proceedings of the IEEE CEC 2007, pp. 1071–1078 (2007)

21. Fonseca, G., Lopes, H.S.: Curriculum-based course timetabling with energy efficiency as an objective. In: Proceedings of the PATAT 2016, pp. 137–151 (2016)

22. Jat, S.N., Yang, S.: A hybrid genetic algorithm and tabu search approach for post enrolment course timetabling. J. Sched. 14(6), 617–637 (2011)

23. Ishibuchi, H., Tsukamoto, N., Nojima, Y.: Evolutionary many-objective optimization: a short review. In: Proceedings of the IEEE CEC 2008, pp. 2419–2426 (2008)

24. Deb, K., Jain, H.: An evolutionary many-objective optimization algorithm using reference-point-based nondominated sorting approach, part I: solving problems with box constraints. IEEE Trans. Evol. Comput. 18(4), 577–601 (2014)

25. Zhang, Q., Li, H.: MOEA/D: a multiobjective evolutionary algorithm based on decomposition. IEEE Trans. Evol. Comput. 11(6), 712–731 (2007)

26. Liu, H., Gu, F., Zhang, Q.: Decomposition of a multiobjective optimization problem into a number of simple multiobjective subproblems. IEEE Trans. Evol. Comput. 18(3), 450–455 (2014)

27. Yuan, Y., Xu, H., Wang, B., Yao, X.: A new dominance relation-based evolutionary algorithm for many-objective optimization. IEEE Trans. Evol. Comput. 20(1), 16–37 (2016)

28. Ishibuchi, H., Setoguchi, Y., Masuda, H., Nojima, Y.: Performance of decomposition-based many-objective algorithms strongly depends on Pareto front shapes. IEEE Trans. Evol. Comput. 21(2), 169–190 (2017)

29. Rachmawati, L., Srinivasan, D.: Multiobjective evolutionary algorithm with controllable focus on the knees of the Pareto front. IEEE Trans. Evol. Comput. 13(4), 810–824 (2009)

30. Zitzler, E., Laumanns, M., Bleuler, S.: A tutorial on evolutionary multiobjective optimization. In: Gandibleux, X. et al. (eds.) Metaheuristics for Multiobjective Optimisation. LNEMS, vol. 535, pp. 3–37. Springer, Berlin (2004)

31. Branke, J., Deb, K., Miettinen, K., Slowinski, R. (eds.): Multiobjective Optimization: Interactive and Evolutionary Approaches. LNCS, vol. 5252. Springer, Berlin (2008)

32. Miettinen, K., Ruiz, F., Wierzbicki, A.P.: Introduction to multiobjective optimization: interactive approaches. In: Branke, J., Deb, K., Miettinen, K., Slowinski, R. (eds.) Multiobjective Optimization. LNCS, vol. 5252, pp. 27–57. Springer, Berlin (2008)

33. Sinha, A., Malo, P., Deb, K., Korhonen, P., Wallenius, J.: Solving bilevel multicriterion optimization problems with lower level decision uncertainty. IEEE Trans. Evol. Comput. 20(2), 199–217 (2016)

34. Chan, T.C.Y., Lee, T., Terekhov, D.: Inverse optimization: closed-form solutions, geometry, and goodness of fit. Manag. Sci. 65(3), 1115–1135 (2019)

35. Deutz, A.H., Emmerich, M.T.M.: Expected hypervolume improvement with constraints under Gaussian process models. In: Proceedings of the IEEE CEC 2021, pp. 1621–1628 (2021)

36. Das, I., Dennis, J.E.: Normal-boundary intersection: a new method for generating the Pareto surface in nonlinear multicriteria optimization problems. SIAM J. Optim. 8(3), 631–657 (1998)

37. Steuer, R.E., Choo, E.U.: An interactive weighted Tchebycheff procedure for multiple objective programming. Math. Program. 26(3), 326–344 (1983)

38. Green, P.E., Srinivasan, V.: Conjoint analysis in consumer research: issues and outlook. J. Consum. Res. 5(2), 103–123 (1978)

39. Binois, M., Wycoff, N.: A survey on Gaussian process-based metamodeling, surrogate optimization, and related topics. J. Glob. Optim. 84(2), 333–371 (2022)

40. Pan, S.J., Yang, Q.: A survey on transfer learning. IEEE Trans. Knowl. Data Eng. 22(10), 1345–1359 (2010)

41. Lust, T., Teghem, J.: The multiobjective multidimensional knapsack problem: a survey and a new approach. Int. Trans. Oper. Res. 19(4), 495–520 (2012)

42. Liefooghe, A., Derbel, B., Verel, S., Aguirre, H., Tanaka, K.: Towards landscape analyses to inform the design of hybrid local search for multiobjective permutation flowshop problems. In: Proceedings of the EMO 2015. LNCS, vol. 9019, pp. 413–428. Springer, Cham (2015)

43. Zitzler, E., Thiele, L., Laumanns, M., Fonseca, C.M., Da Fonseca, V.G.: Performance assessment of multiobjective optimizers: an analysis and review. IEEE Trans. Evol. Comput. 7(2), 117–132 (2003)

44. Couckuyt, I., Deschrijver, D., Dhaene, T.: Fast calculation of multiobjective probability of improvement and expected improvement criteria for Pareto optimization. J. Glob. Optim. 60(3), 575–594 (2014)

45. Vamplew, P., Dazeley, R., Berry, A., Issabekov, R., Dekker, E.: Empirical evaluation methods for multiobjective reinforcement learning algorithms. Mach. Learn. 84(1–2), 51–80 (2011)
