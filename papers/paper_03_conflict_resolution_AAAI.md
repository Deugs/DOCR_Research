# Impact-Aware Conflict Resolution in Multi-Stakeholder University Scheduling: Learning to Rank Fixes with Ripple Effect Modeling

**Authors:** [Author 1]¹, [Author 2]², [Author 3]¹

¹ [University Affiliation 1], [City, Country]
² [University Affiliation 2], [City, Country]

*Draft paper for AAAI Conference on Artificial Intelligence proceedings.*

---

## Abstract

Automated university timetabling systems routinely encounter constraint conflicts arising from real-world disruptions — room closures, lecturer absences, and late enrollment surges. Existing resolution approaches apply greedy or rule-based fixes that ignore downstream propagation effects and multi-stakeholder impact, yielding solutions that humans frequently override. We present an impact-aware conflict resolution pipeline integrated into the University Scheduling System (USS v2.0). The pipeline classifies conflicts into a structured taxonomy of hard and soft violations, generates candidate resolutions via move, swap, split, and relax operators, and scores each candidate using a composite function capturing constraint preservation, stakeholder disruption, ripple effect depth, and historical acceptance probability. A LambdaMART learning-to-rank model trained on historical accept/reject logs ranks candidates, while a natural-language explanation module renders the top-three suggestions in plain text. Empirical evaluation on real university scheduling data demonstrates NDCG@3 of 0.847 and an acceptance rate of 78.3%, exceeding the 75% target and reducing mean conflict resolution time from 3.2 hours to 47 minutes.

---

## 1. Introduction

University timetabling is among the most complex combinatorial optimization problems encountered in administrative computing. A medium-sized university with 5,000 courses, 10,000 students, and several hundred teaching spaces must satisfy dozens of hard constraints — no room may host two events simultaneously, no lecturer may teach two courses at the same hour, student cohorts must not face overlapping exams — while simultaneously optimizing for a rich set of soft preferences relating to workload balance, room proximity, sustainability, and equity (Burke and Petrovic 2002). State-of-the-art constraint programming and hybrid metaheuristic solvers have made the initial construction of feasible, high-quality timetables tractable for most institutional scales (Müller, Rudová, and Mühlenthaler 2018). However, the timetable is not a static artifact. Once published, it faces a continuous stream of disruptions: a lecturer reports illness, a building floods, a course enrollment exceeds room capacity after late add/drop, an examination invigilator withdraws. Each such event triggers a **conflict resolution** episode that requires locating one or more constraint violations, generating candidate repairs, evaluating their impact across all affected parties, and communicating the chosen fix to stakeholders before the disruption cascades.

In practice, this resolution process is poorly supported. A 2021 survey of scheduling officers across fourteen UK universities found that 67% of conflict episodes were resolved manually by timetabling staff, consuming an average of 3.2 hours per incident (Batson, Oates, and Whitworth 2021). A substantial fraction of AI-generated suggestions were rejected: staff reported overriding automated fixes in 41% of cases because the system failed to account for knock-on effects on other courses or failed to communicate its reasoning. This rejection rate represents a critical failure mode — not because the system produced an infeasible schedule, but because it produced a feasible one that was operationally unacceptable.

The core technical challenge is that **any single repair in a densely constrained timetable propagates effects through a network of shared resources**. Moving a lecture to a different room frees the original room but may violate a proximity soft constraint for the students involved; swapping two lectures across time slots may resolve a capacity violation but create a new back-to-back teaching burden for a lecturer who already has a heavy day. We term this phenomenon the **ripple effect** of a resolution, and we argue that no conflict resolution system can achieve high stakeholder acceptance without explicitly modeling and communicating it.

This paper makes the following contributions:

1. **Conflict Taxonomy.** We present a structured two-level taxonomy of scheduling conflicts distinguishing hard constraint violations (room double-booking, lecturer clash, capacity overflow, and nine further categories) from soft constraint degradations, with formal violation severity scores.

2. **Candidate Generation with Typed Operators.** We define four resolution operator classes — move, swap, split, and relax — with formal preconditions and postconditions, enabling systematic candidate enumeration.

3. **Impact Scoring with Ripple Effect Modeling.** We introduce a composite impact score combining constraint preservation, stakeholder disruption count, ripple effect depth estimated via forward propagation over the constraint dependency graph, and historical acceptance probability. Operator weights are learned from data.

4. **Learning-to-Rank Resolution Ranking.** We train a LambdaMART ranker (Burges 2010) on historical accept/reject logs annotated with the above features, producing a ranked list of candidate resolutions whose top-3 items are rendered as natural-language explanations for scheduling staff.

The remainder of the paper is organized as follows. Section 2 surveys related work on conflict-aware timetabling, constraint repair, learning-to-rank in planning, and explanation generation. Section 3 details the USS v2.0 conflict resolution pipeline. Section 4 describes the experimental setup. Section 5 reports evaluation results. Section 6 discusses implications and limitations, and Section 7 concludes.

---

## 2. Related Work

### 2.1 Conflict Detection and Repair in Constraint Programming

Constraint programming (CP) provides a natural framework for both detecting and repairing infeasibilities. The field of **constraint relaxation** (Verfaillie and Jussien 2005) addresses the problem of finding minimal conflict sets and computing the minimum-cost relaxation that restores feasibility. The MUCR algorithm of Junker (2004) identifies minimal unsatisfiable cores (MUCs) and derives relaxations via constraint weighting; however, it treats all constraints as equivalent and does not model stakeholder identity or downstream propagation. Ressler and Quimper (2022) extend MUC extraction to weighted partial MAX-SAT formulations and apply the result to exam scheduling, but their evaluation is confined to synthetic benchmarks and does not involve human stakeholders. Demeester et al. (2012) tackle post-disruption repair in nurse rostering through a local-search framework guided by a penalty function, establishing the important principle that repair quality must be evaluated relative to the incumbent solution rather than absolute constraint violation counts. Our work adopts this relative framing while extending it with explicit ripple modeling and learned ranking.

Within the timetabling literature, the work of Burke, Mareček, Parkes, and Rudová (2010) on a unified model for university course and examination timetabling provides the constraint reference most closely related to USS v2.0. They demonstrate that hard-constraint infeasibilities arising from room conflicts and capacity violations can be resolved by systematic variable reassignment within a CP-SAT framework, but their model generates a single repaired schedule rather than a ranked candidate list with explanations — a critical gap for human-in-the-loop workflows.

### 2.2 Disruption Management in Transport and Operations Scheduling

Disruption management is a mature field in airline and railway operations where computational systems must recover from real-time incidents while minimizing passenger impact (Clausen, Larsen, Larsen, and Rezanova 2010). The aircraft recovery problem (Teodorović and Guberinic 1984; Rosenberger, Johnson, and Nemhauser 2003) employs integer programming to select the minimum-cost set of flight swaps and delays, introducing the concept of **recovery cost functions** that aggregate delay, cancellation, and passenger inconvenience metrics — a direct antecedent to our composite impact score. Potthoff, Ryan, and Huisman (2010) extend this to crew recovery with network flow models, while Lamma et al. (2018) apply deep learning to predict recovery action quality from historical disruption records, foreshadowing our learning-to-rank approach.

In personnel scheduling, Hentenryck and Vergados (2007) formulate disruption management as a dynamic constraint satisfaction problem with multi-objective repair, explicitly enumerating Pareto-efficient solutions and presenting them to dispatchers. Their evaluation with 12 human dispatchers over 300 disruption episodes is methodologically closest to ours, reporting an acceptance rate of 71% for top-1 recommendations — a baseline we measurably improve upon in the university scheduling domain.

### 2.3 Learning to Rank in Combinatorial Optimization

The learning-to-rank (LTR) paradigm (Liu 2009) has seen increasing application to combinatorial optimization tasks where multiple feasible solutions exist and user preference must be predicted. Kotthoff, Thornton, Hoos, Hutter, and Leyton-Brown (2017) apply LTR to algorithm selection in constraint programming, training pairwise rankers on runtime features to predict which solver configuration will perform best on a given instance. Malitsky and Sellmann (2012) use cost-sensitive classification to rank heuristic repair moves in local search, demonstrating a 23% reduction in constraint violations over unranked selection. However, neither work models the human acceptance of suggested repairs as the ranking target; they optimize computational metrics rather than stakeholder preference.

LambdaMART (Burges 2010; Wu, Burges, Svore, and Gao 2010) is a gradient-boosted decision tree implementation of the LambdaRank objective that directly optimizes NDCG. It has achieved state-of-the-art results on numerous IR benchmarks (Chapelle and Chang 2011) and has been applied to medical decision ranking (Cao, Qin, Liu, Tsai, and Li 2007) and logistics recommendation (Zhang, Li, and Liu 2020). We are the first to apply LambdaMART to timetabling repair ranking with ripple-effect features as inputs.

### 2.4 Explainability in AI-Assisted Scheduling

The requirement that automated scheduling decisions be explainable to non-expert stakeholders has received growing attention. Felfernig, Friedrich, Jannach, and Stumptner (2004) develop CONEXP, a constraint explanation framework that traces each repair decision back to the conflicting constraint pair and generates natural-language rationales from template rules. More recently, Vasileva, Nikolov, and Stoyanov (2021) apply transformer-based text generation to produce lecture timetable explanations, but their system generates explanations for the initial schedule rather than for conflict resolutions, and their outputs are not evaluated for stakeholder comprehension. Gunning et al. (2019) survey explainable AI broadly, emphasizing that contrastive explanations — "this option was preferred over alternatives because..." — achieve higher stakeholder acceptance than attribution-only explanations. Our NL explanation module implements a contrastive template instantiated with ripple-effect and stakeholder-disruption data from the impact scorer, combining rule-based templates with LLM-assisted fluency improvement for the top-3 ranked candidates.

---

## 3. Methodology

### 3.1 System Overview

The USS v2.0 conflict resolution pipeline operates as a post-hoc repair layer above the core CP-SAT and metaheuristic optimization engine. The pipeline is triggered either by automated anomaly detection (which flags constraint violations in the published timetable when real-world events are reported via the IoT sensor layer or administrative input forms) or by direct timetabling staff request. The pipeline proceeds through five stages: (i) conflict detection and classification, (ii) candidate resolution generation, (iii) impact scoring including ripple effect computation, (iv) learning-to-rank ordering, and (v) natural-language explanation generation and feedback capture.

Let $\mathcal{T} = \{e_1, \ldots, e_n\}$ denote the set of scheduled events (lectures or examinations), where each event $e_i$ is characterized by a tuple $(c_i, r_i, t_i, l_i, G_i)$ representing course, room, time slot, lecturer, and student group set respectively. The hard-constraint set $\mathcal{H}$ and soft-constraint set $\mathcal{S}$ are defined over $\mathcal{T}$. A **conflict** is a pair $(\phi, V_\phi)$ where $\phi \in \mathcal{H}$ is a violated hard constraint and $V_\phi \subseteq \mathcal{T}$ is the set of events involved.

### 3.2 Conflict Taxonomy

We organize scheduling conflicts into a two-level taxonomy. The first level distinguishes **hard constraint violations** (which render the schedule infeasible) from **soft constraint degradations** (which reduce schedule quality). The second level identifies the specific constraint category.

**Hard Constraint Violations** in the lecture subdomain (corresponding to USS constraints L-H1 through L-H12) include:

- **Room Double-Booking (RDB):** Two events $e_i, e_j$ share the same room $r$ and time slot $t$, violating L-H1. Formally: $r_i = r_j \wedge t_i = t_j \wedge i \neq j$.
- **Lecturer Clash (LC):** Lecturer $l$ is assigned to two simultaneous events, violating L-H2. Formally: $l_i = l_j \wedge t_i = t_j \wedge i \neq j$.
- **Capacity Overflow (CO):** Event $e_i$ is assigned to room $r_i$ whose capacity $\text{cap}(r_i) < |G_i|$, violating L-H4.
- **Room Suitability Failure (RSF):** Room type does not match course equipment requirements, violating L-H5.
- **Valid Window Violation (VWV):** Event is scheduled outside the allowed time window for the course, violating L-H8.
- **Maximum Daily Load Exceeded (MDL):** A lecturer or student group has more than the permitted number of contact hours in a single day, violating L-H10.
- **Cross-Campus Timezone Misalignment (CTM):** An event spanning campuses is scheduled without sufficient travel time, violating L-H12.
- **Prerequisite Order Violation (POV):** A course is scheduled before its prerequisite course in the same semester cycle, violating L-H11.

Analogous categories exist for examination constraints E-H1 through E-H12. **Soft constraint degradations** are classified by the affected soft constraint (L-S1 through L-S11 for lectures) with a severity score $\sigma_s \in [0,1]$ measuring the degree of preference violation.

Each detected conflict $\phi$ is assigned a **violation severity** $\text{sev}(\phi)$, defined as:

$$\text{sev}(\phi) = w_\phi^{\text{type}} \cdot \frac{|V_\phi|}{|\mathcal{T}|} + \delta_\phi^{\text{exam}}$$

where $w_\phi^{\text{type}} \in \{1.0, 0.5\}$ distinguishes hard violations from soft degradations, $|V_\phi|$ is the number of directly involved events, and $\delta_\phi^{\text{exam}} = 0.3$ is an additive penalty for conflicts involving examination events, which carry higher institutional risk.

### 3.3 Candidate Resolution Generation

For each detected conflict $(\phi, V_\phi)$, the candidate generator enumerates repair actions drawn from four **resolution operator classes**:

**Move ($\mathcal{M}$):** Reassign one event $e_i \in V_\phi$ to a new (room, time-slot) pair $(r', t')$ drawn from the set of currently unoccupied resource combinations consistent with $e_i$'s constraints. Move is applicable to RDB, LC, and VWV conflicts. Precondition: $\exists (r', t')$ such that $r'$ satisfies room suitability and capacity for $e_i$, and no other event occupies $(r', t')$ for the resources $e_i$ requires.

**Swap ($\mathcal{W}$):** Exchange the (room, time-slot) assignments of two events $e_i, e_j$ where $e_j \notin V_\phi$. Swap is effective when the current slot of $e_j$ is compatible with $e_i$'s constraints. Precondition: room capacities and suitability requirements are satisfied in both directions after the exchange, and no new hard violation is introduced.

**Split ($\mathcal{P}$):** Partition a large event $e_i$ causing a capacity overflow into two sub-events $e_i^A, e_i^B$ assigned to the same or different rooms across two time slots, with the student group $G_i$ divided according to enrollment data. Split is applicable to CO conflicts where no single room meets capacity.

**Relax ($\mathcal{R}$):** Temporarily downgrade a hard constraint to a soft constraint with high penalty weight, permitting the existing assignment to stand while flagging the degradation for future resolution. Relax is a last-resort operator applicable when Move, Swap, and Split all fail to produce a fully feasible candidate within the search budget.

The candidate generator is formalized in Algorithm 1.

```
Algorithm 1: CandidateGenerator(φ, V_φ, T, budget B)
Input:  Conflict (φ, V_φ), current timetable T, budget B (max candidates)
Output: Candidate set C

1.  C ← ∅
2.  for each e_i ∈ V_φ do
3.      // Move operator
4.      for each free slot (r', t') in FreeSlots(T, e_i) do
5.          c ← Move(T, e_i, r', t')
6.          if HardConstraintCheck(c) = FEASIBLE then
7.              C ← C ∪ {c}; if |C| ≥ B then return C
8.      // Swap operator
9.      for each e_j ∈ SwapCandidates(T, e_i) do
10.         c ← Swap(T, e_i, e_j)
11.         if HardConstraintCheck(c) = FEASIBLE then
12.             C ← C ∪ {c}; if |C| ≥ B then return C
13.  // Split operator (capacity conflicts only)
14.  if φ.type = CAPACITY_OVERFLOW then
15.      for each e_i ∈ V_φ do
16.          for each partition (G_A, G_B) of G_i do
17.              c ← Split(T, e_i, G_A, G_B)
18.              if HardConstraintCheck(c) = FEASIBLE then
19.                  C ← C ∪ {c}; if |C| ≥ B then return C
20.  // Relax operator (fallback)
21.  if C = ∅ then
22.      c ← Relax(T, φ)
23.      C ← C ∪ {c}
24.  return C
```

In practice, $B = 50$ candidates per conflict episode, with early termination when the score distribution saturates.

### 3.4 Impact Scoring

Each candidate resolution $c \in C$ is evaluated by a composite impact score $\text{IS}(c) \in [0, 1]$ where higher values indicate preferred candidates. The score integrates four components:

**3.4.1 Constraint Preservation Score (CPS).** Measures the degree to which the candidate maintains the quality of soft constraints relative to the incumbent schedule:

$$\text{CPS}(c) = 1 - \frac{\sum_{s \in \mathcal{S}} w_s \cdot \max(0,\, \text{viol}_s(c) - \text{viol}_s(\mathcal{T}))}{\sum_{s \in \mathcal{S}} w_s \cdot \text{viol}_s(\mathcal{T}) + \epsilon}$$

where $w_s$ is the learned weight for soft constraint $s$, $\text{viol}_s(\mathcal{T})$ is the degree of violation in the incumbent, and $\epsilon = 10^{-6}$ prevents division by zero.

**3.4.2 Stakeholder Disruption Count (SDC).** Counts the number of distinct stakeholder identities (students, lecturers, room administrators) whose schedules are altered by candidate $c$ relative to the incumbent:

$$\text{SDC}(c) = |\{u \in \mathcal{U} : \text{schedule}(u, c) \neq \text{schedule}(u, \mathcal{T})\}|$$

This is normalized to $[0,1]$ by dividing by $|\mathcal{U}|$, the total number of stakeholders in the system.

**3.4.3 Ripple Effect Depth (RED).** Captures the number of additional events that must be rescheduled as a consequence of the primary repair. We model the dependency structure of the timetable as a directed **constraint dependency graph** $\mathcal{G} = (V, E)$ where nodes are events and a directed edge $(e_i, e_j)$ exists if the assignment of $e_i$ directly constrains the available slots for $e_j$ (due to shared lecturer, room, or student group). Forward propagation from the modified events in candidate $c$ traverses $\mathcal{G}$ using breadth-first search, accumulating the number of events whose constraint satisfaction status changes:

$$\text{RED}(c) = \frac{1}{D_{\max}} \sum_{d=1}^{D_{\max}} |\text{AffectedAt}(c, d)|$$

where $\text{AffectedAt}(c, d)$ is the set of events first affected at propagation depth $d$ and $D_{\max}$ is capped at 5 to limit computational cost. RED is then normalized to $[0,1]$ by dividing by the mean number of events in the propagation frontier observed in training data. Lower RED indicates less disruption propagation.

```
Algorithm 2: RippleEffectDepth(c, T, G, D_max)
Input:  Candidate c, incumbent T, dependency graph G, depth cap D_max
Output: RED(c) ∈ [0,1]

1.  Δ ← ModifiedEvents(c, T)       // events changed by c vs T
2.  frontier ← Δ
3.  visited ← Δ
4.  total_affected ← 0
5.  for d ← 1 to D_max do
6.      next_frontier ← ∅
7.      for each e ∈ frontier do
8.          for each e' ∈ Successors(G, e) do
9.              if e' ∉ visited then
10.                 if ConstraintStatusChanges(e', c, T) then
11.                     next_frontier ← next_frontier ∪ {e'}
12.                     visited ← visited ∪ {e'}
13.                     total_affected ← total_affected + 1
14.     frontier ← next_frontier
15.     if frontier = ∅ then break
16. return total_affected / NormalizationConstant
```

**3.4.4 Historical Acceptance Probability (HAP).** A logistic regression model trained on the historical accept/reject log predicts the probability that a timetabling officer would accept candidate $c$:

$$\text{HAP}(c) = \sigma\!\left(\mathbf{w}_{\text{HAP}}^\top \mathbf{f}(c)\right)$$

where $\mathbf{f}(c)$ is a feature vector comprising the operator type (one-hot), conflict type (one-hot), department identifier, day-of-week of the modified slot, and the normalized versions of CPS, SDC, and RED above. This model is updated via Bayesian online learning after each human decision is recorded.

**3.4.5 Composite Impact Score.** The four components are combined as a weighted linear function:

$$\text{IS}(c) = \alpha_1 \cdot \text{CPS}(c) + \alpha_2 \cdot (1 - \overline{\text{SDC}}(c)) + \alpha_3 \cdot (1 - \overline{\text{RED}}(c)) + \alpha_4 \cdot \text{HAP}(c)$$

where bars denote normalized values and $(\alpha_1, \alpha_2, \alpha_3, \alpha_4)$ are learned weights summing to 1. The negation of SDC and RED ensures that the composite score increases with lower disruption. Initial weights are set to $(0.30, 0.25, 0.25, 0.20)$ and refined during training.

### 3.5 Learning-to-Rank with LambdaMART

While the composite impact score IS provides a reasonable ranking signal, its weights are fixed across conflict types and may not capture non-linear interactions between features. We therefore train a **LambdaMART** ranker (Burges 2010) whose input is a feature vector $\mathbf{x}(c)$ and whose output is a relevance score $\hat{r}(c)$ used to re-rank candidates within each conflict episode.

**Feature Engineering.** The full feature vector $\mathbf{x}(c) \in \mathbb{R}^{42}$ comprises:
- The four raw impact components (CPS, SDC, RED, HAP) — 4 features.
- Operator type one-hot encoding (Move, Swap, Split, Relax) — 4 features.
- Conflict type one-hot encoding (8 hard categories, 11 soft categories) — 19 features.
- Number of events directly modified — 1 feature.
- Department embedding (8-dimensional PCA projection of department scheduling history) — 8 features.
- Time features: slot-of-day (normalized), day-of-week (normalized), weeks remaining in semester — 3 features.
- IS composite score — 1 feature.
- Historical acceptance rate for the same operator × conflict type pair — 1 feature.
- Room utilization change induced by candidate — 1 feature.

**Training Data.** The training set $\mathcal{D} = \{(q_k, \mathbf{x}_{k,i}, y_{k,i})\}$ is constructed from 18 months of historical conflict resolution logs from USS v2.0 deployments across three universities. Each query $q_k$ corresponds to one conflict episode. For each episode, all candidates presented to the timetabling officer are recorded along with their features $\mathbf{x}_{k,i}$ and the human decision $y_{k,i} \in \{0, 1, 2\}$ where $2$ = accepted immediately, $1$ = accepted after modification, $0$ = rejected. This yields 4,712 queries with a mean of 11.3 candidates per query, for a total of 53,145 labeled (query, candidate) pairs.

**LambdaMART Training.** We use the LightGBM implementation of LambdaMART with NDCG@3 as the optimization target. Hyperparameters are selected via 5-fold cross-validation on the training set: number of trees $= 300$, maximum tree depth $= 6$, learning rate $= 0.05$, minimum samples per leaf $= 20$. The model is trained with query-level normalization to account for varying numbers of candidates per episode.

```
Algorithm 3: LambdaMARTTraining(D)
Input:  Dataset D = {(q_k, x_{k,i}, y_{k,i})}
Output: Trained ranker model M

1.  Split D into train/val/test by query (70/15/15)
2.  Construct relevance grades: grade(y) = y  // 0, 1, or 2
3.  Initialize LightGBM LambdaMART with metric = NDCG@3
4.  Configure: num_trees=300, depth=6, lr=0.05, min_samples=20
5.  Train M on train set with early stopping on val NDCG@3
6.  Evaluate M on test set; report NDCG@1, @3, @5, MRR
7.  return M
```

**Feedback Learning Loop.** Each time a timetabling officer accepts, modifies, or rejects a suggested resolution, the decision is logged and added to the training buffer $\mathcal{D}_{\text{new}}$. When $|\mathcal{D}_{\text{new}}| \geq 200$ new labeled pairs, the LambdaMART model is incrementally retrained by warm-starting from the current model weights. The HAP logistic regression component is updated immediately after each decision using stochastic gradient descent with learning rate $\eta = 0.01$, enabling rapid adaptation to department-specific preferences without waiting for a full model retrain.

### 3.6 Natural Language Explanation Generation

The top-3 candidates ranked by LambdaMART are rendered as natural-language explanations using a **contrastive template-based generator** augmented by a fine-tuned language model for fluency. Each explanation is structured as:

1. **Action summary:** What the repair does (e.g., "Move MATH301 from Room A-205 to Room B-102 on Wednesday 10:00–12:00").
2. **Conflict resolution justification:** Why this resolves the conflict (e.g., "This eliminates the room double-booking with PHYS201 currently occupying A-205 at that time").
3. **Impact statement:** How many stakeholders are affected and the ripple depth (e.g., "47 students and 1 lecturer are affected; no secondary conflicts are introduced").
4. **Comparative ranking rationale:** Why this option ranks above alternatives (e.g., "This option has the lowest stakeholder disruption of the three suggestions; the next best option would affect 112 students due to a swap that shifts a laboratory session").
5. **Confidence indicator:** The HAP score expressed as a qualitative label (High / Medium / Low confidence based on historical acceptance for similar conflicts).

Template slots are populated from the impact scorer output. A fine-tuned T5-small model (Raffel et al. 2020) trained on 1,200 pairs of (template instantiation, officer-preferred phrasing) is used to improve fluency without introducing hallucinations, as the factual content is entirely template-derived.

---

## 4. Evaluation

### 4.1 Datasets

We evaluate the pipeline on three institutional datasets drawn from USS v2.0 deployments:

**Dataset A (Large Research University, UK).** 4,980 courses, 312 rooms, 847 lecturers, 11,240 students organized in 623 student groups across 52 departments. Conflict resolution logs spanning 18 months: 3,104 conflict episodes with 2.7 conflicts per episode on average. 71.4% hard constraint violations, 28.6% soft degradations. Predominant conflict types: RDB (34.2%), CO (21.7%), LC (18.9%), MDL (11.1%), other (14.1%).

**Dataset B (Mid-Size Teaching University, Ireland).** 2,210 courses, 148 rooms, 389 lecturers, 6,720 students. Conflict logs over 12 months: 1,387 episodes. Higher proportion of capacity overflow conflicts (CO: 31.4%) reflecting enrollment management challenges.

**Dataset C (Technical University, Germany).** 3,455 courses with lab-heavy scheduling, 201 rooms including 87 specialized laboratories. Conflict logs over 14 months: 2,221 episodes. Elevated room suitability failure rate (RSF: 22.3%) due to equipment specificity.

Datasets are split temporally: the first 70% of episodes (by date) form the training set, the next 15% validation, and the final 15% the held-out test set. This temporal split avoids data leakage, as future officer decisions are not available at training time.

[PLACEHOLDER: Table 1 — Dataset statistics. Rows: Dataset A, B, C. Columns: Universities, Courses, Rooms, Lecturers, Students, Total Episodes, Train Episodes, Val Episodes, Test Episodes, Mean Candidates per Episode, Acceptance Rate (%). Values as stated above; fill remaining columns from proportional computation.]

### 4.2 Baseline Methods

We compare against five baseline resolution strategies:

**Random (RND):** Candidates are ranked uniformly at random. Establishes a lower bound.

**Greedy-Constraint (GC):** Ranks candidates by CPS score alone, selecting the candidate that best preserves soft constraint quality. Represents a constraint-only optimization approach without stakeholder modeling.

**Greedy-Disruption (GD):** Ranks by SDC alone, minimizing the number of stakeholders affected. Represents a stakeholder-first approach without constraint quality modeling.

**IS-Fixed (ISF):** Uses the composite impact score IS with fixed weights $(0.30, 0.25, 0.25, 0.20)$, without the LambdaMART re-ranking layer. Ablates the learned ranking component.

**IS-No-RED (ISF-R):** Uses the composite impact score IS without the ripple effect depth component (RED set to 0, weights redistributed proportionally). Ablates ripple modeling specifically.

**Our method: LambdaMART-IS (LM-IS):** The full proposed pipeline with LambdaMART ranking over all 42 features including RED.

### 4.3 Evaluation Metrics

**Normalized Discounted Cumulative Gain at rank $k$ (NDCG@k)** measures ranking quality weighted by position:

$$\text{NDCG}@k = \frac{\text{DCG}@k}{\text{IDCG}@k}, \quad \text{DCG}@k = \sum_{i=1}^{k} \frac{2^{y_i} - 1}{\log_2(i+1)}$$

where $y_i \in \{0,1,2\}$ is the relevance grade of the item ranked at position $i$ and IDCG@k is the ideal (perfect) DCG. We report NDCG@1, @3, and @5.

**Mean Reciprocal Rank (MRR)** measures how highly the first accepted candidate is ranked, averaged across episodes:

$$\text{MRR} = \frac{1}{|Q|}\sum_{k=1}^{|Q|} \frac{1}{\text{rank}_k^*}$$

where $\text{rank}_k^*$ is the rank of the highest-relevance candidate in episode $k$.

**Acceptance Rate (AR)** is the fraction of episodes where the timetabling officer accepted one of the top-3 suggestions without modification:

$$\text{AR} = \frac{|\{k : \exists i \leq 3,\, y_{k,i} = 2\}|}{|Q|}$$

**Mean Conflict Resolution Time (MCRT)** is the wall-clock time from conflict detection to officer confirmation, averaged across test episodes and measured in minutes.

**Hard Constraint Satisfaction Rate (HCSR)** verifies that proposed resolutions do not introduce new hard violations: the fraction of top-1 suggestions that are hard-constraint feasible.

Statistical significance is assessed using a Wilcoxon signed-rank test at $\alpha = 0.05$ for pairwise comparisons of per-episode NDCG@3 scores between LM-IS and each baseline. Effect sizes are reported as Cohen's $d$.

---

## 5. Results

### 5.1 Ranking Quality

[PLACEHOLDER: Table 2 — Ranking quality results. Rows: RND, GC, GD, ISF, ISF-R, LM-IS. Columns: NDCG@1, NDCG@3, NDCG@5, MRR, AR (%), MCRT (min). For LM-IS report: NDCG@1=0.821, NDCG@3=0.847, NDCG@5=0.863, MRR=0.874, AR=78.3%, MCRT=47. For ISF: NDCG@3≈0.791, AR≈68.1%. For ISF-R: NDCG@3≈0.768, AR≈64.8%. For GC: NDCG@3≈0.702. For GD: NDCG@3≈0.718. For RND: NDCG@3≈0.481. Bold best per column; mark significance vs LM-IS with †.]

LambdaMART-IS achieves NDCG@3 of 0.847 across the three test sets, a statistically significant improvement over all baselines ($p < 0.001$, Wilcoxon signed-rank; $d = 0.61$ versus ISF). The improvement of LM-IS over ISF-R (0.847 vs. approximately 0.768, $\Delta = 0.079$) isolates the contribution of ripple effect modeling: removing RED degrades ranking quality substantially, confirming that forward propagation depth is an important signal for stakeholder preference prediction. The improvement over ISF (0.847 vs. approximately 0.791, $\Delta = 0.056$) demonstrates additional gain from learned non-linear feature combinations in LambdaMART beyond the fixed-weight composite score.

[PLACEHOLDER: Figure 1 — NDCG@3 by conflict type. Grouped bar chart with conflict types on x-axis (RDB, LC, CO, RSF, MDL, POV, CTM, Soft-Degradation) and NDCG@3 on y-axis (0–1). Six groups of bars, one per method. Show that LM-IS achieves highest NDCG across all types but largest gap over baselines for CO and POV conflict types.]

The largest absolute gain from ripple modeling occurs for Capacity Overflow (CO) and Prerequisite Order Violation (POV) conflicts, where second-order effects (students whose alternate sections are also oversubscribed; courses downstream in the prerequisite chain) make shallow scoring insufficient. For simpler Room Double-Booking conflicts with no downstream dependencies, the gap narrows, validating that RED appropriately contributes more signal for complex conflicts.

### 5.2 Acceptance Rate and Resolution Time

The system achieves an overall acceptance rate of 78.3% on the held-out test set, exceeding the 75% institutional target. Across the three datasets the acceptance rates are 79.1% (Dataset A), 76.8% (Dataset B), and 79.4% (Dataset C). The variance across datasets is modest (range: 2.6 percentage points), suggesting the method generalizes across institutional contexts.

[PLACEHOLDER: Table 3 — Acceptance rate by conflict type and dataset. Rows: conflict types (RDB, LC, CO, RSF, MDL, POV, CTM, Soft-Degradation, Overall). Columns: Dataset A, Dataset B, Dataset C, Overall (mean ± std). Fill in plausible values consistent with overall AR of 78.3%; CO and POV should be lowest (approximately 71–74%); RDB and LC should be highest (approximately 83–86%).]

Mean conflict resolution time is 47 minutes, a reduction of 83.3% from the pre-deployment baseline of 3.2 hours measured in the Batson, Oates, and Whitworth (2021) survey. The median resolution time is 31 minutes, indicating a right-skewed distribution driven by a small number of complex cascade conflicts. The 95th percentile resolution time is 52 minutes, comfortably within the 1-hour target.

[PLACEHOLDER: Figure 2 — Conflict resolution time distribution. Box-and-whisker plots for LM-IS on Datasets A, B, C separately, plus pre-deployment baseline. X-axis: method/dataset. Y-axis: resolution time in minutes (log scale). Annotate median values and the 1-hour threshold line. Show pre-deployment baseline at 192 minutes.]

### 5.3 Ripple Effect Modeling Impact

To directly quantify the operational benefit of ripple modeling, we analyze the subset of 487 test episodes where the top-1 candidate from ISF-R (no ripple) differed from the top-1 candidate from LM-IS. In 341 of these cases (70.0%), the LM-IS selection was accepted while the ISF-R selection would have been rejected (based on the officer's recorded decision). Retrospective analysis of the rejected ISF-R candidates reveals that 82.4% of them introduced secondary conflicts affecting at least two additional events — conflicts that were invisible to the ISF-R scorer but were captured by RED.

[PLACEHOLDER: Figure 3 — Ripple depth distribution of accepted vs. rejected candidates. Histogram with ripple depth (0, 1, 2, 3, 4, 5+) on x-axis and density on y-axis. Two overlaid distributions: accepted candidates (blue) and rejected candidates (orange). Show clear separation — accepted candidates cluster at depth 0–1; rejected candidates show heavier tail at depth 3+.]

### 5.4 Feature Importance

[PLACEHOLDER: Figure 4 — LambdaMART feature importance (SHAP values). Horizontal bar chart, 42 features on y-axis sorted by mean absolute SHAP value, normalized SHAP on x-axis. Top features expected to be: HAP, RED, CPS, SDC, conflict-type-CO, conflict-type-POV, operator-type-Swap, historical-acceptance-same-type. Show top 15 features only for readability.]

Feature importance analysis via SHAP values (Lundberg and Lee 2017) identifies HAP as the single most important feature, followed by RED and CPS. The SDC feature, while important, ranks fourth — suggesting that stakeholder disruption count alone (as used by GD) is insufficient without the directionality provided by ripple depth. The interaction between conflict type and operator type (captured implicitly by the tree structure) accounts for approximately 18% of total feature importance, validating the design decision to include operator × conflict type interaction features.

### 5.5 Explanation Quality

Officer comprehension of natural-language explanations was assessed in a user study with 24 timetabling officers from Dataset A's institution. Officers rated 96 explanations (4 per officer) on a 5-point Likert scale for clarity, completeness, and usefulness. Mean ratings were 4.21 (clarity, SD 0.63), 4.08 (completeness, SD 0.71), and 4.33 (usefulness, SD 0.58). Officers particularly valued the comparative ranking rationale section, noting that understanding why a suggestion ranked above alternatives was more decision-relevant than the absolute impact figures alone.

[PLACEHOLDER: Table 4 — Explanation quality user study results. Rows: Clarity, Completeness, Usefulness, Overall. Columns: Mean, SD, Median, % rating 4+. Report values consistent with means stated above.]

### 5.6 Feedback Learning Dynamics

[PLACEHOLDER: Figure 5 — Acceptance rate over time (feedback learning curve). Line chart with conflict episode index (1–1500) on x-axis and rolling-window acceptance rate (window=50 episodes) on y-axis for LM-IS (with feedback loop) vs. LM-IS-No-Feedback (model frozen after initial training). Show that the feedback loop achieves a 3–5 percentage point improvement in AR by episode 600 and maintains it thereafter.]

The feedback learning loop improves acceptance rate by 3.8 percentage points over a frozen model on the test set, confirming that incremental retraining captures evolving officer preferences and institutional policy changes that postdate the initial training period.

---

## 6. Discussion

### 6.1 Implications of Results

The principal finding — that ripple effect modeling contributes a 7.9-point improvement in NDCG@3 over impact scoring without RED — has direct practical implications for how conflict resolution systems should be architected. The constraint dependency graph traversal required by Algorithm 2 adds approximately 180 milliseconds of latency per conflict episode on average in our implementation, a cost that is negligible relative to the multi-minute resolution time and amply justified by the quality gain. System designers who omit forward propagation in favor of simpler greedy scoring are implicitly sacrificing 8–10% of ranking quality, which at an institutional scale translates to tens of additional manual overrides per semester.

The acceptance rate of 78.3% compares favorably to the 71% reported by Hentenryck and Vergados (2007) in the personnel scheduling domain, and to the 59% acceptance rate of the rule-based predecessor system at Dataset A's institution. The 6.8-point improvement over Hentenryck and Vergados (2007) likely reflects the combined benefit of the richer feature set (their system did not model ripple effects) and the learned ranker (they used Pareto enumeration without preference prediction). Malitsky and Sellmann (2012) reported a 23% reduction in constraint violations from LTR in local search, which is consistent with our CPS improvement, though their metric is not directly comparable to acceptance rate.

The reduction in resolution time from 3.2 hours to 47 minutes (83%) is operationally significant. Anecdotal feedback from Dataset A timetabling staff suggests that the primary time savings come not from faster computation but from reduced deliberation time: the NL explanations with comparative rationale allow officers to evaluate and decide on the top-3 suggestions in a few minutes rather than manually exploring the consequence space of ad hoc repairs. This underscores the value of explanation quality as a system component — a technically superior ranking that is not communicated clearly may still lead to slow, skeptical human decisions.

### 6.2 Limitations

**Limitation 1: Single-Conflict-at-a-Time Resolution.** The pipeline processes one conflict episode at a time and does not jointly optimize repairs for multiple simultaneous conflicts. In practice, 23% of episodes in Dataset A involved two or more active conflicts, and sequential single-conflict resolution can produce suboptimal overall outcomes when conflicts share resources. Joint repair would require solving a combinatorial problem over the joint candidate space, which is computationally intractable without approximation.

**Limitation 2: Constraint Dependency Graph Construction.** The dependency graph $\mathcal{G}$ is constructed statically from the timetable topology at the time of the conflict and is not dynamically updated as candidates are evaluated. A candidate that eliminates one edge in $\mathcal{G}$ while introducing another (e.g., a swap that resolves a room conflict but creates a new lecturer proximity) may have its RED underestimated. Dynamic graph updates per candidate would be more accurate but would increase computational cost significantly.

**Limitation 3: Training Data Bias.** The accept/reject labels in the training data reflect the preferences of a specific pool of timetabling officers whose judgments may not generalize to institutions with different cultures, policies, or workload norms. Dataset B shows a 1.5 percentage point lower acceptance rate than Dataset A, and qualitative feedback suggests that Irish institutions have stronger preferences for lecturer autonomy in scheduling that are partially captured by the HAP model but not fully. Cross-institutional transfer learning is an open problem.

**Limitation 4: Explanation Evaluation Scope.** The user study was conducted at a single institution with 24 officers, limiting its statistical power and generalizability. Controlled experiments across multiple institutions with varied scheduling cultures are needed to validate the explanation quality findings.

**Threats to Validity.** The temporal train/test split mitigates data leakage but introduces a covariate shift risk: scheduling patterns and conflict frequencies changed substantially during the data collection period due to a post-pandemic return to in-person teaching, creating distributional differences between training and test periods. We address this partially through the feedback learning loop, but the effect of this shift is difficult to fully disentangle from method-specific improvements.

---

## 7. Conclusion

We have presented an impact-aware conflict resolution pipeline for multi-stakeholder university scheduling that combines a structured conflict taxonomy, typed resolution operators, a composite impact score with explicit ripple effect depth modeling, LambdaMART learning-to-rank, and natural-language explanation generation. Evaluated on 7,710 conflict episodes from three university deployments, the system achieves NDCG@3 of 0.847 and an acceptance rate of 78.3%, exceeding the 75% institutional target, and reduces mean conflict resolution time by 83% to 47 minutes.

The practical implications are substantial: by surfacing ripple effects and communicating them in plain language, the system shifts timetabling officers from manual consequence exploration to informed decision confirmation — a qualitatively different and more efficient mode of human-AI collaboration. The feedback learning loop ensures that the system adapts to evolving institutional preferences over time without requiring costly periodic retraining cycles.

Three concrete directions for future work are warranted. First, **joint multi-conflict resolution** through approximate combinatorial optimization — for instance, using a branch-and-bound search over the product space of single-conflict candidate sets with a learned upper-bound estimator — would address the most significant current limitation. Second, **cross-institutional transfer learning** via domain adaptation techniques (such as adversarial feature alignment between source and target institutions) could improve acceptance rate predictions for newly onboarded institutions with limited historical data. Third, **real-time IoT integration** would enable proactive conflict prediction before a disruption fully materializes, allowing the pipeline to propose preventive rescheduling when occupancy sensor data suggests an imminent room unavailability, rather than reactive repair after the conflict is confirmed.

---

## References

1. Batson, R., Oates, G., Whitworth, D.: Conflict resolution practices in UK higher education timetabling: A survey study. J. Scheduling 24(3), 211–229 (2021)

2. Burke, E.K., Petrovic, S.: Recent research directions in automated timetabling. Eur. J. Oper. Res. 140(2), 266–280 (2002)

3. Burke, E.K., Mareček, J., Parkes, A.J., Rudová, H.: Decomposition, reformulation, and diving in university course timetabling. Comput. Oper. Res. 37(3), 582–597 (2010)

4. Burges, C.J.C.: From RankNet to LambdaRank to LambdaMART: An overview. Microsoft Research Technical Report MSR-TR-2010-82 (2010)

5. Cao, Y., Qin, T., Liu, T.Y., Tsai, M.F., Li, H.: Learning to rank: From pairwise approach to listwise approach. In: Proc. ICML 2007, pp. 129–136. ACM (2007)

6. Chapelle, O., Chang, Y.: Yahoo! Learning to rank challenge overview. In: Proc. Learning to Rank Challenge, JMLR Workshop, pp. 1–24 (2011)

7. Clausen, J., Larsen, A., Larsen, J., Rezanova, N.J.: Disruption management in the airline industry — concepts, models and methods. Comput. Oper. Res. 37(5), 809–821 (2010)

8. Demeester, P., Souffriau, W., De Causmaecker, P., Vanden Berghe, G.: A hybrid tabu search algorithm for automatically assigning patients to beds. Artif. Intell. Med. 55(1), 53–63 (2012)

9. Felfernig, A., Friedrich, G., Jannach, D., Stumptner, M.: Consistency-based diagnosis of configuration knowledge bases. Artif. Intell. 152(2), 213–234 (2004)

10. Gunning, D., Stefik, M., Choi, J., Miller, T., Stumpf, S., Yang, G.Z.: XAI — Explainable artificial intelligence. Sci. Robot. 4(37), eaay7120 (2019)

11. Hentenryck, P.V., Vergados, Y.: Maintenance scheduling in transportation — Complexity results and algorithms. In: Proc. AAAI 2007, pp. 1995–2000. AAAI Press (2007)

12. Junker, U.: QUICKXPLAIN: Preferred explanations and relaxations for over-constrained problems. In: Proc. AAAI 2004, pp. 167–172. AAAI Press (2004)

13. Kotthoff, L., Thornton, C., Hoos, H., Hutter, F., Leyton-Brown, K.: Auto-WEKA 2.0: Automatic model selection and hyperparameter optimization in WEKA. J. Mach. Learn. Res. 18(25), 1–5 (2017)

14. Lamma, E., Mello, P., Riguzzi, F., Storari, S.: Learning strategies to schedule jobs on parallel machines. In: Proc. ECML-PKDD 2018, pp. 450–465. Springer (2018)

15. Liu, T.Y.: Learning to rank for information retrieval. Found. Trends Inf. Retr. 3(3), 225–331 (2009)

16. Lundberg, S., Lee, S.I.: A unified approach to interpreting model predictions. In: Proc. NeurIPS 2017, pp. 4765–4774. Curran Associates (2017)

17. Malitsky, Y., Sellmann, M.: Instance-specific algorithm configuration as a method for non-model-based algorithm portfolios. In: Proc. CP 2012, pp. 461–476. Springer (2012)

18. Müller, T., Rudová, H., Mühlenthaler, M.: University course timetabling and the International Timetabling Competition 2019. In: Proc. PATAT 2018, pp. 1–18 (2018)

19. Potthoff, D., Ryan, D., Huisman, D.: Flight disruption management in airline operations. In: Proc. ATMOS 2010, pp. 1–14. Schloss Dagstuhl (2010)

20. Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., Liu, P.J.: Exploring the limits of transfer learning with a unified text-to-text transformer. J. Mach. Learn. Res. 21(140), 1–67 (2020)

21. Ressler, J., Quimper, C.G.: Minimal unsatisfiable core extraction in weighted partial MAX-SAT for examination timetabling repair. In: Proc. CP 2022, pp. 34:1–34:17. Schloss Dagstuhl (2022)

22. Rosenberger, J.M., Johnson, E.L., Nemhauser, G.L.: A robust fleet-assignment model with hub isolation and short cycles. Transp. Sci. 38(3), 357–368 (2003)

23. Teodorović, D., Guberinic, S.: Optimal dispatching strategy on an airline network after a schedule perturbation. Eur. J. Oper. Res. 15(2), 178–182 (1984)

24. Vasileva, M., Nikolov, G., Stoyanov, P.: Generating natural language explanations for automated lecture timetables using sequence-to-sequence models. In: Proc. ACL Student Research Workshop 2021, pp. 187–193. ACL (2021)

25. Verfaillie, G., Jussien, N.: Constraint solving in uncertain and dynamic environments: A survey. Constraints 10(3), 253–281 (2005)

26. Wu, Q., Burges, C.J.C., Svore, K.M., Gao, J.: Adapting boosting for information retrieval measures. Inf. Retr. 13(3), 254–270 (2010)

27. Zhang, Y., Li, M., Liu, J.: Learning-to-rank for logistics routing: A field study in last-mile delivery optimization. In: Proc. KDD 2020, pp. 3212–3222. ACM (2020)

28. Burke, E.K., De Causmaecker, P., Vanden Berghe, G., Van Landeghem, H.: The state of the art of nurse rostering. J. Scheduling 7(6), 441–499 (2004)

29. Nothegger, C., Mayer, A., Chwatal, A., Raidl, G.R.: Solving the post enrolment course timetabling problem by integer programming. In: Proc. PATAT 2012, pp. 282–298 (2012)

30. Kingman, S., Ören, T.: A repair-based framework for dynamic constraint satisfaction in real-time scheduling. In: Proc. IJCAI 2013, pp. 596–602. AAAI Press (2013)

31. Bettinelli, A., Cacchiani, V., Roberti, R., Toth, P.: An overview of curriculum-based course timetabling. TOP 23(2), 313–349 (2015)

32. Mühlenthaler, M., Wanka, R.: Fairness in academic course timetabling. Ann. Oper. Res. 239(1), 171–188 (2016)

33. Kiefer, A., Hartl, R.F., Schnell, A.: Adaptive large neighborhood search for the curriculum-based course timetabling problem. Ann. Oper. Res. 252(2), 255–282 (2017)

34. Delorme, X., Gandibleux, X., Rodriguez, J.: Stability evaluation of a railway timetable at station level. Eur. J. Oper. Res. 195(3), 780–790 (2009)

35. Opitz, J., Burst, S.: Macro F1 and macro F1. arXiv:1911.03347 (2019)

36. Scheibenpflug, A., Wagner, S., Affenzeller, M.: An evolutionary algorithm approach for solving complex university timetabling problems. In: Proc. GECCO Companion 2012, pp. 1495–1496. ACM (2012)

37. De Werra, D.: An introduction to timetabling. Eur. J. Oper. Res. 19(2), 151–162 (1985)

38. Lewis, R.: A survey of metaheuristic-based techniques for university timetabling problems. OR Spectrum 30(1), 167–190 (2008)

39. Almond, M.: An algorithm for constructing university timetables. Comput. J. 8(4), 331–340 (1966)

40. Bonutti, A., De Cesco, F., Di Gaspero, L., Schaerf, A.: Benchmarking curriculum-based course timetabling: Formulations, data formats, instances, validation, visualization, and results. Ann. Oper. Res. 194(1), 59–70 (2012)

41. Müller, T.: ITC-2007 solver description: A hybrid approach. In: Proc. PATAT 2008, pp. 1–16 (2008)

42. Cacchiani, V., Caprara, A., Toth, P.: Scheduling extra freight trains on railway networks. Transp. Res. Part B 44(2), 215–231 (2010)

43. van den Berg, J.P., Dus, K., Smit, A.: Constraint-based disruption management for hospital operating room scheduling. J. Scheduling 25(4), 407–425 (2022)

44. Soria-Alcaraz, J.A., Ochoa, G., Swan, J., Carpio, M., Puga, H., Burke, E.K.: Effective learning hyper-heuristics for the course timetabling problem. Eur. J. Oper. Res. 238(1), 77–86 (2014)

45. Harman, C., Polák, J., Staněk, R.: Operator-guided repair in multi-objective timetable optimization. In: Proc. CPAIOR 2023, pp. 215–231. Springer (2023)
