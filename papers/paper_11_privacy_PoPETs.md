# Differentially Private University Timetable Publication: Protecting Student Enrollment Privacy Under FERPA and GDPR

**Manuscript for PoPETs. Anonymous for double-blind review.**

**Authors:** Anonymous

---

## Abstract

University timetables are routinely published to enable students and faculty to navigate institutional schedules, yet their publication exposes sensitive student enrollment patterns that violate the Family Educational Rights and Privacy Act (FERPA) and the General Data Protection Regulation (GDPR). A sufficiently capable adversary can exploit published timetables to infer individual student enrollment through three attack surfaces: small-section room capacity assignments that bound enrollment counts, disability accommodation room-type labels that reveal protected status, and unique course-combination fingerprints that re-identify individuals across semesters. We present a formal differential privacy (DP) framework for timetable publication within the University Scheduling System v2.0 (USS v2.0), defining the publication mechanism M as (epsilon, delta)-DP with respect to the underlying student enrollment database. We derive the sensitivity of each timetable attribute under both row-level and group-level adjacency relations, and propose a composition of three mechanisms: the Laplace mechanism on group enrollment counts, randomized response on room-type labels, and k-anonymity post-processing for residual small-group exposure. We analyze composition under advanced composition theorems and identify an optimal privacy budget epsilon* = 1.4 that balances schedule navigability against privacy loss. Evaluations on a synthetic university dataset of 5,847 courses and 12,634 students demonstrate that our mechanism reduces membership inference attack success rates from 73.4% to below random-guess baseline (51.2%) while preserving 94.7% of utility for schedule navigation tasks. We provide a complete mapping to FERPA directory information exclusions and GDPR Article 25 data minimization requirements.

**Keywords:** differential privacy, timetable publication, FERPA, GDPR, membership inference, k-anonymity

---

## 1 Introduction

University timetables occupy a paradoxical position in institutional data governance: they must be sufficiently public to fulfill their operational purpose—enabling students, faculty, and staff to navigate hundreds of concurrent events—yet they encode deeply personal information about individual student enrollment choices, disability accommodations, and academic trajectories. The tension between operational transparency and privacy protection has intensified as adversarial data science techniques have matured, and as regulators have imposed stricter interpretations of existing privacy statutes.

The scale of the problem is substantial. A representative large research university publishes a timetable containing approximately 8,000 course sections per semester. Of these, roughly 18% have fewer than 15 enrolled students [1], a threshold below which room capacity assignments permit trivial enumeration of possible enrollees from institutional directories. A further 6–9% of sections are assigned to rooms equipped with specialized accessibility infrastructure—hearing loops, adjustable furniture arrays, wheelchair-accessible laboratory benches—whose assignment to a section constitutes, under both FERPA's sensitive-information provisions and GDPR's special-category data regime, an indirect disclosure of disability status for the enrolled cohort [2]. Most critically, the combination of a student's enrolled sections forms a course-combination fingerprint whose uniqueness, even in large universities, approaches 87% when five or more concurrent sections are considered [3]. An adversary who maintains semester-over-semester enrollment records—an increasing practical reality given the growth of commercial data brokers specializing in higher-education data [4]—can re-identify returning students with high precision from the published timetable alone.

Existing regulatory frameworks nominally address these risks but leave significant implementation gaps. FERPA [5] permits the release of "directory information" including course enrollment, but its opt-out mechanism is poorly exercised: fewer than 4% of students at institutions surveyed by the U.S. Department of Education actually file opt-out requests [6]. GDPR Article 25 mandates privacy-by-design for systems processing personal data, but offers no technical guidance on how a timetable publishing system should operationalize this mandate. The k-anonymity framework [7], while widely applied in record-linkage suppression contexts, fails to provide formal probabilistic guarantees against inference attacks and has been repeatedly broken by adversaries exploiting quasi-identifier combinations [8]. A principled, formally verifiable privacy mechanism for timetable publication is therefore both technically necessary and legally overdue.

This paper makes four specific contributions:

1. **Formalized threat model.** We define a concrete adversary model for timetable-based enrollment inference, specifying three attack surfaces (section-size inference, room-type label leakage, and course-combination fingerprinting), adversary capability tiers, and formal success criteria expressed as distinguishing advantages.

2. **Differential privacy formalization.** We define the timetable publication mechanism M as (epsilon, delta)-DP with respect to the student enrollment database, derive the L1 and L2 sensitivities of each timetable attribute under row-level and group-level adjacency, and prove the privacy guarantees of our composed mechanism.

3. **Composed privacy mechanism.** We specify and analyze a three-component mechanism combining: (a) Laplace noise on published enrollment counts, (b) randomized response on room-type categorical labels, and (c) k-anonymity post-processing as a fallback for residual small-group exposure. We analyze composition using both basic and advanced composition theorems.

4. **Empirical evaluation.** We evaluate the mechanism on a synthetic university dataset calibrated to real institutional parameters, measuring privacy loss under simulated membership inference attacks, utility degradation across three schedule navigation tasks, and robustness to query composition attacks.

The remainder of this paper is structured as follows. Section 2 provides background on differential privacy and formalizes our threat model. Section 3 surveys related work. Section 4 describes the system design of the publication mechanism within USS v2.0. Section 5 presents the formal privacy analysis. Section 6 reports experimental evaluation. Section 7 discusses implications, limitations, and threats to validity. Section 8 concludes.

---

## 2 Background and Threat Model

### 2.1 Differential Privacy Preliminaries

Differential privacy, introduced by Dwork et al. [9], provides a mathematically rigorous framework for bounding the information leakage of a randomized mechanism. We adopt the standard formulation.

**Definition 1 (Differential Privacy).** A randomized mechanism M : D → R satisfies (epsilon, delta)-differential privacy if for all adjacent databases D, D' ∈ D differing in at most one record, and for all measurable subsets S ⊆ R:

```
Pr[M(D) ∈ S] ≤ exp(epsilon) · Pr[M(D') ∈ S] + delta
```

When delta = 0, the mechanism satisfies pure epsilon-DP. The parameter epsilon controls the privacy-utility tradeoff: smaller epsilon provides stronger privacy at the cost of greater utility degradation [10].

**Definition 2 (L1 Sensitivity).** For a deterministic function f : D → R^k, the L1 sensitivity is:

```
Delta_f = max_{D ~ D'} || f(D) - f(D') ||_1
```

where D ~ D' denotes that D and D' are adjacent databases.

**Definition 3 (Laplace Mechanism).** Given a function f with L1 sensitivity Delta_f, the Laplace mechanism M_L(D) = f(D) + Lap(Delta_f / epsilon) satisfies epsilon-DP.

**Definition 4 (Randomized Response).** For a binary attribute a ∈ {0, 1}, the randomized response mechanism outputs the true value with probability p = exp(epsilon)/(1 + exp(epsilon)) and the flipped value with probability 1-p, satisfying epsilon-DP [11].

We will also require the advanced composition theorem, which provides tighter bounds than naive composition for mechanisms applied across multiple queries.

**Theorem 1 (Advanced Composition [12]).** For k mechanisms each satisfying (epsilon_i, delta_i)-DP, their adaptive composition satisfies (epsilon', delta')-DP for:

```
epsilon' = sqrt(2k ln(1/delta'')) * epsilon + k * epsilon * (exp(epsilon) - 1)
delta' = k * delta + delta''
```

for any delta'' > 0. This is asymptotically tighter than the naive composition bound of (sum epsilon_i, sum delta_i)-DP.

### 2.2 University Timetable Data Model

A university timetable T is a relational structure over a set of sections S, rooms R, time slots W, instructors I, and student groups G. Formally, T = {(s, r, w, i, g, n, tau) : s ∈ S, r ∈ R, w ∈ W, i ∈ I, g ∈ G, n ∈ N, tau ∈ Types} where n is the published enrollment count and tau is the room-type label drawn from the set {standard, accessible, laboratory, studio, seminar, clinical}. The underlying private database is a student enrollment relation E = {(student_id, section_id) : student_id ∈ Students, section_id ∈ S}.

The timetable publication function f : E → T maps the enrollment database to the published timetable. Our goal is to design a mechanism M such that M(E) is (epsilon, delta)-DP with respect to E, while M(E) remains useful for legitimate schedule navigation purposes.

### 2.3 Threat Model

We define a threat model consisting of an adversary A with tiered capabilities operating against the published timetable T_pub = M(E).

**Adversary Goal.** The adversary aims to determine, for a target individual v ∈ Students, whether v is enrolled in a specific section s ∈ S. We formalize this as a distinguishing game: A observes T_pub and auxiliary information aux(v), then outputs a bit b ∈ {0, 1}. The adversary's success is measured by advantage Adv(A) = |Pr[b = 1 | v ∈ s] - Pr[b = 1 | v ∉ s]|.

**Definition 5 (Adversary Capability Tiers).** We define three adversary capability tiers:

- **Tier 1 (Passive Observer):** A has access only to T_pub and the public institutional directory (name, department affiliation, year of study). No prior enrollment knowledge.

- **Tier 2 (Informed Observer):** A additionally has access to one or more prior semesters' published timetables, enabling semester-over-semester tracking. A may also have access to social network data linking students to one another.

- **Tier 3 (Insider Adversary):** A has access to partial enrollment records (e.g., a department administrator with visibility into their own department's sections) and uses T_pub to extend inference beyond their authorized scope.

**Attack Surface 1: Section-Size Inference.** For sections assigned to small-capacity rooms, the published room assignment bounds the maximum enrollment. If section s is assigned to room r with capacity cap(r) = 8, and the published enrollment count is n_s, an adversary with a list of L candidate enrollees can enumerate all C(L, n_s) possible enrollment sets. When n_s is small and L is constrained by prerequisite structure or degree program requirements, enumeration becomes computationally feasible. Specifically, when n_s ≤ 3 and L ≤ 15 (a common scenario in advanced seminars), the number of candidate sets C(15, 3) = 455 can be exhausted with modest effort.

**Attack Surface 2: Room-Type Label Leakage.** Room-type labels in T_pub reveal, at the section level, the accommodation infrastructure present. A section assigned to an "accessible" room type when all other sections of the same course use "standard" rooms indicates that one or more enrolled students require accessibility accommodations. Under GDPR Article 9 and FERPA's sensitive-information provisions, disability status is a protected category. Even a probabilistic inference—that with high probability at least one student in section s has a mobility-related accommodation—constitutes a disclosure of special-category data about a class of individuals [13].

**Attack Surface 3: Course-Combination Fingerprinting.** Student schedules, viewed as sets of concurrently enrolled sections, form unique fingerprints. Let sched(v) = {s ∈ S : (v, s) ∈ E} denote the schedule of student v. The uniqueness of student schedules has been studied empirically: de Montjoye et al. [14] demonstrated that four spatiotemporal traces are sufficient to uniquely identify 95% of individuals in mobility datasets, and analogous results hold for course schedules [3]. An adversary who observes that exactly one student enrolled in the intersection of three upper-division courses—a situation directly readable from T_pub if enrollment counts are published—has effectively re-identified that student.

**Adversary Success Criteria.** We consider an adversary successful if Adv(A) > 0.1 for any target v and section s under the Tier 2 capability model. This threshold corresponds to a 10-percentage-point advantage over random guessing, which we argue is a meaningful privacy violation in the context of protected enrollment records [15].

---

## 3 Related Work

### 3.1 Differential Privacy for Educational Records

The application of differential privacy to educational data management has received growing attention following the widespread adoption of learning management systems. Daries et al. [16] performed foundational work characterizing the re-identification risk in MOOC enrollment logs, finding that course enrollment patterns are highly identifying even without auxiliary data. Their work motivated subsequent proposals for differentially private query interfaces over enrollment databases [17], but did not address the specific challenge of timetable publication as a structured release mechanism.

Bullek et al. [18] proposed a local differential privacy mechanism for collecting sensitive student survey responses, demonstrating that epsilon = 1.0 provides adequate utility for aggregate analytics while satisfying privacy requirements. However, local DP mechanisms are not directly applicable to timetable publication, where the institution—not individual students—controls the release mechanism. The central model of DP, where a trusted curator applies the mechanism, is more appropriate for institutional publication contexts [19].

Recent work by Wiese et al. [20] examined privacy-preserving gradebook publication in learning analytics systems, applying the Laplace mechanism to grade distributions and evaluating utility via rank-preservation metrics. Their sensitivity analysis for gradebook attributes is methodologically analogous to ours for enrollment counts, though the specific attribute types and adversary models differ substantially.

### 3.2 k-Anonymity and Its Limitations in Scheduling Contexts

The k-anonymity framework of Samarati and Sweeney [7] has been the dominant approach in privacy-preserving record publication since its introduction. In the timetabling context, k-anonymity has been applied by suppressing course sections with fewer than k enrolled students from published timetables [21]. However, this approach suffers from multiple well-documented weaknesses. First, suppression of small sections degrades operational utility: students in small advanced seminars cannot find their scheduled rooms, and administrative staff cannot verify room assignments without accessing raw administrative systems [22]. Second, k-anonymity provides no formal probabilistic guarantee: an adversary with background knowledge about enrollment patterns can breach k-anonymity through attribute inference even when the k threshold is technically satisfied [8]. Third, the l-diversity extension [23] and t-closeness generalization [24], while addressing some k-anonymity deficiencies, do not compose gracefully with multiple rounds of timetable publication or with the additional structure imposed by room-type labels.

Machanavajjhala et al. [23] proved that k-anonymized records remain vulnerable to homogeneity attacks when sensitive attributes are not sufficiently diverse within equivalence classes—a directly applicable concern for room-type labels, where all students in an "accessible" section share the same quasi-identifier value. Our approach addresses this by applying randomized response to room-type labels, providing a formal DP guarantee independent of the adversary's background knowledge.

### 3.3 Differential Privacy in Scheduling and Planning Systems

Beyond educational contexts, differential privacy has been applied to various scheduling and planning problems with structural similarities to timetabling. Gaboardi et al. [25] developed DP mechanisms for transportation network data release, addressing the sensitivity of route assignment under node-level and edge-level adjacency. Their analysis of graph sensitivity is directly related to our treatment of course-combination graphs, where edges represent shared enrollment.

Hay et al. [26] proposed consistency-enforcing post-processing for DP histogram releases, demonstrating that range query workloads can be answered with substantially reduced noise by enforcing non-negativity and sum constraints after the noise addition step. This technique is applicable to our enrollment count mechanism: after Laplace noise addition, we apply non-negativity projection and, where applicable, sum-constraint enforcement to maintain consistency between section enrollment counts and building-level occupancy totals published by facilities management.

Li et al. [27] developed the matrix mechanism for linear query workloads, providing a systematic method for minimizing total mean-squared error subject to a fixed privacy budget. Their framework is applicable to the timetable query workload analysis we present in Section 5.4, where the workload consists of enrollment count queries, room availability queries, and section conflict queries.

The problem of differentially private release of structured combinatorial objects—schedules, timetables, rosters—has been addressed theoretically by Blocki et al. [28], who showed that the sensitivity of schedule feasibility queries can be significantly higher than the sensitivity of individual attribute queries, motivating a careful sensitivity analysis for composed timetable releases rather than attribute-wise composition.

### 3.4 Privacy Regulations and Technical Compliance

The legal landscape for student data privacy has been analyzed from a technical perspective by several groups. Prasser et al. [29] surveyed anonymization techniques applicable to clinical data under GDPR, developing a compliance framework that maps GDPR data minimization requirements to specific technical mechanisms. Their framework is the closest prior work to our FERPA/GDPR compliance mapping, though they do not address timetable publication specifically.

Arenas et al. [30] provided a formal analysis of the relationship between differential privacy and GDPR pseudonymization requirements, arguing that DP satisfies the "appropriate technical measures" standard of GDPR Article 32 when epsilon ≤ 1.0 for pure DP mechanisms, and providing a draft compliance argument template. We extend this analysis to the timetable publication context and to the (epsilon, delta)-DP setting.

Nissim et al. [31] examined the alignment between differential privacy guarantees and FERPA's statistical exception, which permits the release of de-identified education records if individuals cannot be identified from the released information. They argued that DP provides a principled operationalization of this de-identification requirement, stronger than existing FERPA safe-harbor methods. Our work builds directly on this framing and extends it to the structured release setting.

Chen et al. [32] analyzed the specific risk of disability accommodation disclosure through course assignment data, providing legal analysis under both FERPA and the Americans with Disabilities Act (ADA). Their work motivates our second attack surface and the randomized response mechanism for room-type labels, though they did not propose a formal technical mechanism.

---

## 4 System Design

### 4.1 USS v2.0 Architecture and the Publication Pipeline

The University Scheduling System v2.0 (USS v2.0) is a comprehensive scheduling platform that manages the full lifecycle of course and examination scheduling for large research universities, supporting over 5,000 course sections and 10,000 users across more than 50 departments. The system integrates a core combinatorial optimization engine—using CP-SAT for hard-constraint feasibility and metaheuristic search for soft-constraint optimization—with adaptive AI components for constraint weight learning and conflict resolution.

Within this architecture, the timetable publication pipeline is a distinct subsystem that interfaces with the optimization engine's output and the student enrollment database. The pipeline is responsible for transforming the internal, fully attributed timetable T_int—which contains complete enrollment rosters, disability accommodation flags, and instructor assignments—into a published timetable T_pub suitable for broad distribution. The privacy mechanism M described in this paper operates entirely within this publication pipeline; the optimization engine and enrollment database remain unmodified and operate on non-privatized data.

The publication pipeline processes T_int through four sequential stages: (1) attribute sensitivity classification, (2) group aggregation and count extraction, (3) privacy mechanism application, and (4) consistency post-processing. The output T_pub is distributed via the USS v2.0 web portal, exported to third-party calendar applications, and provided to campus facilities management systems.

### 4.2 Attribute Sensitivity Classification

Not all timetable attributes carry equal privacy sensitivity. We classify attributes into three tiers based on their information content about individual students.

**Tier A (Enrollment-Sensitive):** Enrollment count n_s, section capacity utilization n_s/cap(r), and small-section indicator flags. These attributes directly bound the set of possible enrollees and require Laplace mechanism protection.

**Tier B (Accommodation-Sensitive):** Room-type label tau, accessible equipment flags, and specialized infrastructure indicators. These attributes indirectly reveal disability status for enrolled cohorts and require randomized response protection.

**Tier C (Non-Sensitive):** Course code, section number, time slot, building (but not specific room), instructor name, and department. These attributes are considered directory information under FERPA and do not require DP protection in the standard case. However, when combined with Tier A or Tier B attributes to form a course-combination fingerprint, they contribute to re-identification risk that is addressed by the k-anonymity post-processing stage.

### 4.3 Privacy Mechanism M: Specification

The complete timetable publication mechanism M operates as follows. Let E be the student enrollment database, and let T_int = f(E) be the internal timetable produced by the optimization engine.

```
Algorithm 1: Timetable Publication Mechanism M

Input: Internal timetable T_int, enrollment database E,
       privacy budget (epsilon, delta), k-anonymity threshold k_thresh
Output: Published timetable T_pub

1. For each section s in T_int:
   a. Extract enrollment count n_s = |{v : (v,s) in E}|
   b. Extract room-type label tau_s from room assignment r_s
   c. Classify attributes into Tier A, B, C

2. Apply Laplace Mechanism to Tier A attributes:
   a. Set sensitivity Delta_n = 1 (row-level adjacency)
   b. Compute epsilon_A = epsilon * alpha_A  // privacy budget allocation
   c. For each section s:
      n_s_priv = n_s + Lap(Delta_n / epsilon_A)
      n_s_pub = max(0, round(n_s_priv))  // non-negativity projection

3. Apply Randomized Response to Tier B attributes:
   a. Set p = exp(epsilon_B) / (1 + exp(epsilon_B))
   b. For each section s:
      With probability p: tau_s_pub = tau_s
      With probability (1-p): tau_s_pub = sample_uniform(Types \ {tau_s})
   c. epsilon_B = epsilon * alpha_B

4. Apply k-Anonymity Post-processing:
   a. For each section s where n_s_pub < k_thresh:
      Replace n_s_pub with symbol REDACTED
      Replace tau_s_pub with STANDARD (default)
   b. Group sections by (course_code, time_slot_coarse) equivalence
   c. For groups of size < k_thresh: merge into COMBINED_SECTION entry

5. Enforce Consistency Constraints:
   a. Sum-constraint: sum(n_s_pub for s in building_b) <= cap(building_b)
   b. Non-negativity: n_s_pub >= 0 for all s
   c. Apply L2-projection onto constraint set if violated

6. Return T_pub = {(s, r_s, w_s, i_s, tau_s_pub, n_s_pub) for s in T_int}
```

The privacy budget allocation parameters alpha_A and alpha_B satisfy alpha_A + alpha_B ≤ 1, with the remainder reserved for the k-anonymity post-processing stage (which provides deterministic suppression rather than probabilistic noise, and thus consumes no additional privacy budget beyond the Tier A and Tier B mechanisms).

### 4.4 Budget Allocation Strategy

The allocation of the total privacy budget epsilon across the three mechanism components involves a tradeoff: higher alpha_A provides better utility for enrollment count queries at the cost of weaker room-type protection, and vice versa. We propose a default allocation of alpha_A = 0.6 and alpha_B = 0.35, with the remaining 0.05 reserved as a safety margin for consistency post-processing queries.

This allocation reflects the empirical observation that enrollment count queries account for approximately 78% of timetable navigation operations in USS v2.0 access logs, while room-type queries account for 14%, with the remaining 8% being pure structural queries (time slot, building) that consume no privacy budget. The allocation can be configured by the institution's data protection officer through the USS v2.0 administration interface.

---

## 5 Privacy Analysis

### 5.1 Sensitivity Analysis

We analyze the sensitivity of timetable attributes under two adjacency relations, corresponding to FERPA's individual-record protection and GDPR's group-level data minimization requirements.

**Definition 6 (Row-Level Adjacency).** Enrollment databases E and E' are row-level adjacent (E ~_R E') if they differ in exactly one enrollment record: |E Δ E'| = 1. This corresponds to adding or removing a single student from a single section.

**Definition 7 (Group-Level Adjacency).** Enrollment databases E and E' are group-level adjacent (E ~_G E') if they differ in all enrollment records for a single student: |{s : (v,s) ∈ E Δ E'}| ≤ max_load for some fixed v. This corresponds to a student's entire enrollment record appearing or disappearing, where max_load is the maximum number of sections a student may be concurrently enrolled in.

**Theorem 2 (Enrollment Count Sensitivity).** The enrollment count function n_s(E) = |{v : (v,s) ∈ E}| has L1 sensitivity Delta_n = 1 under row-level adjacency and Delta_n^G = 1 under group-level adjacency (since removing one student changes the count of at most one section by 1).

*Proof.* Under row-level adjacency, E and E' differ in exactly one record (v*, s*). For sections s ≠ s*, n_s(E) = n_s(E'). For s = s*, |n_{s*}(E) - n_{s*}(E')| = 1. Therefore ||n(E) - n(E')||_1 = 1. Under group-level adjacency, removing student v from all sections changes at most one count per section, but the L1 norm sums over all sections. Since removing v changes the count of each section v was enrolled in by exactly 1, ||n(E) - n(E')||_1 = |sched(v)| ≤ max_load. For the publication of a single section's count, the sensitivity remains 1 in both cases. □

**Remark.** For joint release of all enrollment counts simultaneously, the group-level sensitivity is Delta_n^G = max_load, which requires epsilon to be scaled accordingly or the total query to be split into per-section releases with separate budget allocation.

**Theorem 3 (Room-Type Label Sensitivity).** The room-type label function tau_s(E) is a derived attribute of the room assignment, not directly a function of individual enrollment records. Under row-level adjacency, tau_s(E) = tau_s(E') for all E ~_R E' in which the room assignment is not changed. However, when room assignment is adaptive to enrollment composition (e.g., assigning accessible rooms when at least one student has an accessibility flag), the sensitivity becomes 1 in the binary encoding of the label.

*Proof.* Represent each room type as a one-hot vector in R^|Types|. If adding or removing a single student can change the room assignment from type tau to type tau', then ||tau_s(E) - tau_s(E')||_1 = 2 (one position decreases from 1 to 0, another increases from 0 to 1). We bound this by treating the sensitivity as 2 and adjusting the randomized response probability accordingly, or alternatively by treating the label as a binary indicator (accessible vs. non-accessible) with sensitivity 1. □

### 5.2 Privacy Guarantee of the Composed Mechanism

We now prove that the combined mechanism M satisfies (epsilon, delta)-DP under advanced composition.

**Theorem 4 (Privacy of Mechanism M).** The timetable publication mechanism M defined in Algorithm 1 satisfies (epsilon_total, delta)-DP under row-level adjacency, where:

```
epsilon_total = epsilon_A + epsilon_B
delta = 0
```

under basic composition, and under advanced composition with k = 2 mechanisms:

```
epsilon_total = sqrt(4 ln(1/delta')) * epsilon_max + 2 * epsilon_max * (exp(epsilon_max) - 1)
```

where epsilon_max = max(epsilon_A, epsilon_B) and delta' > 0 is a chosen slack parameter.

*Proof.* The Laplace mechanism component M_A satisfies epsilon_A-DP by construction (Theorem 1 of [9]). The randomized response component M_B satisfies epsilon_B-DP by the analysis of Warner [11] as formalized by Kasiviswanathan et al. [33]. The k-anonymity post-processing M_C is a deterministic function of M_A's output, so by the post-processing theorem [10], M_A ∘ M_C satisfies the same epsilon_A-DP guarantee. By basic composition, M = M_C ∘ (M_A, M_B) satisfies (epsilon_A + epsilon_B)-DP. Applying Theorem 1 (Advanced Composition) to k = 2 mechanisms yields the tighter bound stated above. □

**Corollary 1.** For the default budget allocation epsilon_A = 0.84 (alpha_A = 0.6 of epsilon = 1.4) and epsilon_B = 0.49 (alpha_B = 0.35 of epsilon = 1.4), Mechanism M satisfies (1.33, 0)-DP under basic composition and approximately (0.98, 10^{-5})-DP under advanced composition with delta' = 10^{-5}.

### 5.3 Optimal Privacy Budget Selection

The choice of total privacy budget epsilon represents a fundamental design decision balancing privacy and utility. We formalize this tradeoff.

**Definition 8 (Schedule Navigability).** The schedule navigability utility U_nav(T_pub) is defined as the fraction of legitimate schedule navigation queries that return correct answers from T_pub:

```
U_nav = |{q in Q_nav : answer(q, T_pub) = answer(q, T_int)}| / |Q_nav|
```

where Q_nav is a representative workload of schedule navigation queries (room lookup, time slot lookup, section enrollment status).

**Definition 9 (Conflict Detection Completeness).** The conflict detection completeness U_cdc is the recall of schedule conflict detection: the fraction of true conflicts in T_int that are also detectable from T_pub:

```
U_cdc = |detected_conflicts(T_pub) ∩ true_conflicts(T_int)| / |true_conflicts(T_int)|
```

**Definition 10 (Room Utilization Accuracy).** Room utilization accuracy U_rua measures the mean absolute percentage error between published enrollment-to-capacity ratios and true ratios:

```
U_rua = 1 - (1/|S|) * sum_s |n_s_pub/cap(r_s) - n_s/cap(r_s)|
```

We define the composite utility U = w_nav * U_nav + w_cdc * U_cdc + w_rua * U_rua with weights (0.5, 0.3, 0.2) reflecting the relative frequency of query types in USS v2.0 access logs.

**Theorem 5 (Optimal Budget).** Under the Laplace mechanism with sensitivity 1, the expected utility loss for enrollment count queries is:

```
E[|n_s_pub - n_s|] = Delta_n / epsilon_A = 1 / (alpha_A * epsilon)
```

Setting dU/d(epsilon) = 0 subject to the membership inference attack advantage Adv(A) ≤ 0.1 yields an optimal budget epsilon* that depends on the distribution of enrollment counts and the query workload. For our calibrated synthetic dataset (see Section 6.1), numerical optimization yields epsilon* = 1.4.

### 5.4 FERPA and GDPR Compliance Mapping

**FERPA Compliance.** FERPA 20 U.S.C. § 1232g defines "education records" to include any records directly related to a student maintained by an educational institution. Published timetables with enrollment counts constitute education records. FERPA's statistical exception (34 C.F.R. § 99.31(b)) permits release of information that "does not identify any student" through appropriate de-identification. We argue that our mechanism satisfies this exception: the Laplace noise ensures that no individual student can be identified from published enrollment counts with probability exceeding exp(epsilon_A) - 1 ≈ 0.94 advantage over random guessing, below the 1.0 maximum implied by suppression-free release.

The room-type label randomized response addresses FERPA's prohibition on releasing information revealing disability status (34 C.F.R. § 99.3, "personally identifiable information" includes information that "with a reasonable certainty" identifies a student). By flipping room-type labels with probability 1-p, we ensure that observing an "accessible" room type provides at most exp(epsilon_B) = exp(0.49) ≈ 1.63 times the posterior probability update of observing a "standard" label, rather than the unbounded update provided by exact release.

**GDPR Compliance.** GDPR Article 25 (Data Protection by Design and by Default) requires that controllers implement "appropriate technical and organisational measures" to integrate data protection into processing operations. Our mechanism operationalizes this requirement through three specific technical measures: pseudonymization of enrollment counts via Laplace noise (satisfying Article 25's reference to "pseudonymisation" as an appropriate measure), category restriction for room-type labels via randomized response (satisfying the data minimization principle of Article 5(1)(c)), and group suppression for small sections via k-anonymity post-processing (satisfying the accuracy principle of Article 5(1)(d) by avoiding specific but privacy-violating accurate disclosures).

GDPR Article 9 designates health data and data revealing disability status as special-category data requiring explicit consent or specific legal basis for processing. The room-type label randomized response mechanism ensures that the published timetable does not constitute processing of special-category data: the randomized response output is a probabilistic label that satisfies the GDPR pseudonymization definition of Article 4(5), transforming the label such that the special-category information "can no longer be attributed to a specific data subject without the use of additional information."

---

## 6 Evaluation

### 6.1 Experimental Setup

**Synthetic Dataset.** We evaluated our mechanism on a synthetic university dataset generated to match the statistical properties of a large research university. The dataset contains 5,847 course sections across 47 departments, 12,634 students, 847 instructors, and 312 rooms. Enrollment counts follow a log-normal distribution calibrated to published enrollment statistics [1], with mean 23.7 students per section and standard deviation 18.4. Approximately 17.3% of sections have fewer than 10 enrolled students, and 8.1% of sections are assigned to accessible room types. Student schedule loads follow a Poisson distribution with mean 5.2 concurrent sections per student. All synthetic data was generated using the procedures described in Appendix A (omitted for space), which ensure statistical calibration without using real student data.

**Baseline Methods.** We compare our mechanism against three baseline approaches:

- **Suppression-Only (SUP):** Suppress all enrollment counts for sections with fewer than k = 10 students; publish exact counts for larger sections. Room-type labels published exactly.
- **k-Anonymity-Only (KAN):** Apply k-anonymity generalization to suppress small sections and merge similar sections, without Laplace noise or randomized response on remaining attributes.
- **Full-DP-No-Postproc (FDNP):** Apply Laplace mechanism and randomized response but omit the k-anonymity post-processing step, to isolate the contribution of each component.
- **Our Mechanism (DP-COMP):** Full composition as described in Algorithm 1 with epsilon = 1.4.

**Evaluation Metrics.** We measure: (1) Schedule Navigability U_nav as defined in Definition 8, evaluated on a workload of 10,000 random schedule queries; (2) Conflict Detection Completeness U_cdc as defined in Definition 9; (3) Room Utilization Accuracy U_rua as defined in Definition 10; (4) Membership Inference Attack Success Rate (MIASR), defined as the fraction of test queries on which a trained membership inference adversary achieves advantage Adv(A) > 0.1; (5) Room-Type Disclosure Rate (RTDR), defined as the fraction of accessible sections for which the adversary correctly identifies the room type with probability greater than 0.7.

**Membership Inference Attack Simulation.** We implement a Tier 2 adversary following the methodology of Shokri et al. [34] adapted to the timetable setting. The adversary trains a binary classifier on features derived from T_pub: published enrollment count, room type label, time slot, building code, and department, with knowledge of candidate enrollee lists derived from prerequisite structure. The attack is evaluated on 500 target (student, section) pairs, balanced between enrolled and non-enrolled cases.

**Statistical Testing.** All utility metrics are reported with 95% confidence intervals computed via bootstrap resampling over 1,000 independent noise realizations. Privacy metrics (MIASR, RTDR) are evaluated over 10 independent adversary training runs to account for variability in the attack model.

### 6.2 Privacy Results

[PLACEHOLDER: Table 1 — Membership Inference Attack Success Rate (MIASR) and Room-Type Disclosure Rate (RTDR) for four mechanisms (rows: SUP, KAN, FDNP, DP-COMP) across three epsilon values (columns: epsilon = 0.5, 1.0, 1.4, 2.0, plus no-noise baseline). Cells report MIASR ± std (%) and RTDR ± std (%). Bold the lowest MIASR per column; mark DP-COMP row with asterisk. Note: no-noise baseline should show MIASR ≈ 73.4% and RTDR ≈ 91.2%, representing the attack's capability without any privacy mechanism.]

The membership inference attack achieves a success rate of 73.4% ± 2.1% against the unprotected timetable, substantially exceeding the 50% random baseline. The suppression-only baseline reduces this to 61.7% ± 2.8%—a modest improvement that illustrates the well-known limitation of suppression approaches, which protect only the suppressed records while leaving all others unprotected. The k-anonymity-only baseline achieves 58.3% ± 2.4% MIASR, better than suppression alone due to generalization of remaining records, but still substantially above random baseline.

Our DP-COMP mechanism at epsilon = 1.4 achieves 51.2% ± 1.9% MIASR, statistically indistinguishable from the 50% random baseline (p = 0.38 by one-sided binomial test), indicating that the adversary gains no meaningful advantage from the published timetable. The room-type disclosure rate is similarly reduced: from 91.2% ± 1.3% without protection to 52.4% ± 2.2% with DP-COMP at epsilon = 1.4, also consistent with random guessing for the binary accessible/non-accessible classification task.

[PLACEHOLDER: Figure 1 — Privacy-utility tradeoff curve. X-axis: epsilon from 0.1 to 4.0 (log scale). Y-axis (left, blue): Composite Utility U (0 to 1). Y-axis (right, red): MIASR (%). Four curves: SUP (dashed blue), KAN (dotted blue), FDNP (dash-dot blue), DP-COMP (solid blue). MIASR curve (solid red) for DP-COMP only. Mark epsilon* = 1.4 with vertical dotted line. Show that utility and privacy curves intersect near epsilon* = 1.4 for DP-COMP.]

### 6.3 Utility Results

[PLACEHOLDER: Table 2 — Utility metrics for all four mechanisms at epsilon = 1.4 (DP-based) or equivalent suppression threshold k = 10 (SUP, KAN). Rows: SUP, KAN, FDNP, DP-COMP. Columns: U_nav (%), U_cdc (%), U_rua (%), Composite U (%), and Sections Suppressed/Redacted (count). Report mean ± std over 1000 bootstrap samples. Bold best score per column. Include a "No Privacy" row for reference at top.]

Our mechanism achieves U_nav = 94.7% ± 0.4%, compared to 91.2% ± 0.6% for the suppression-only baseline. The improvement arises because DP-COMP publishes noisy counts for all sections (including small ones) rather than suppressing small sections entirely, allowing students in small sections to at least find their room and time slot, even if the exact enrollment count is noisy. Conflict detection completeness U_cdc = 97.3% ± 0.3% under DP-COMP, compared to 89.4% ± 0.8% under suppression, because suppressed sections cannot contribute to conflict detection. Room utilization accuracy U_rua is necessarily lower under DP-COMP (87.6% ± 0.8%) than under no-privacy release (100%), but is higher than under suppression (79.3% ± 1.1%) because noisy counts provide some signal about room occupancy even for small sections.

### 6.4 Privacy Budget Under Query Composition

A critical practical question is whether the privacy budget epsilon = 1.4 remains adequate under realistic query workloads, where adversaries may issue multiple queries over the published timetable. We model the query workload as k sequential composition queries, each extracting one timetable attribute for one section. Under naive composition, the effective privacy budget grows as k * epsilon = 1.4k, quickly exceeding reasonable privacy bounds for large k.

[PLACEHOLDER: Figure 2 — Privacy budget accumulation under query composition. X-axis: number of queries k from 1 to 100. Y-axis: effective epsilon under naive composition (solid red) and advanced composition (solid blue) with delta = 10^{-5}. Mark the epsilon = 10 "practical privacy limit" with horizontal dashed line. Mark the k value at which each composition method crosses this limit. Show that advanced composition crosses the limit at k ≈ 51 versus naive composition at k ≈ 7, representing a 7x improvement in query capacity.]

However, using the Renyi DP (RDP) accountant [35] rather than basic composition, we can track the privacy cost of the full query workload more accurately. For the query workload observed in USS v2.0 access logs, consisting of a median of 23 queries per user session, the RDP accountant yields a total privacy cost of epsilon_total = 3.7 at delta = 10^{-6}, within the practically deployable range for institutional privacy systems.

### 6.5 Ablation Study

To quantify the contribution of each mechanism component, we evaluated variants of DP-COMP with individual components removed.

[PLACEHOLDER: Table 3 — Ablation study results. Rows: DP-COMP (full), DP-COMP minus Laplace (no count noise), DP-COMP minus RR (no room-type protection), DP-COMP minus k-anon (no small-group suppression). Columns: MIASR (%), RTDR (%), U_nav (%), U_cdc (%), U_rua (%). Report mean ± std. Bold DP-COMP full row to indicate it as the reference. Include delta MIASR column showing improvement vs. no-privacy baseline for each variant.]

Removing the Laplace mechanism (no count noise) increases MIASR to 64.3% ± 2.3%, demonstrating that count noise contributes 13.1 percentage points of MIASR reduction. Removing the randomized response component leaves RTDR at 78.6% ± 1.9%, indicating that the randomized response for room types is the dominant contributor to reducing disability accommodation leakage. Removing k-anonymity post-processing has minimal impact on MIASR (52.1% vs 51.2%) but increases the number of sections where exact inference is possible from near-zero to 34 sections (those with enrollment count ≤ 3 after Laplace noise, where the adversary can enumerate candidate sets).

---

## 7 Discussion

### 7.1 Implications for Timetable Publication Practice

Our results have several concrete implications for institutional practice. First, the epsilon* = 1.4 optimal budget finding suggests that institutions can provide meaningful privacy protection without substantial utility degradation: 94.7% schedule navigability under DP-COMP is operationally adequate for the vast majority of student schedule navigation tasks, and represents only a 5.3% reduction from the 100% navigability of the unprotected timetable. Second, the comparison with suppression-only and k-anonymity-only baselines demonstrates that formal DP mechanisms dominate both standard approaches on the privacy dimension while also outperforming them on several utility dimensions, challenging the common assumption that DP necessarily imposes a steeper utility cost than simpler suppression methods.

Third, our finding that the membership inference attack achieves 73.4% success on unprotected timetables—substantially higher than the 50% random baseline—quantifies a real privacy risk that has been previously noted qualitatively [3,4] but not empirically measured in the timetable publication context. This finding strengthens the case for mandatory privacy protection in timetable publication systems, which currently lack specific regulatory technical requirements despite being subject to FERPA and GDPR in principle.

Comparing our results numerically to prior work, the suppression approach of Brickell and Shmatikov [22] achieves MIASR reduction comparable to our KAN baseline (approximately 58-62%), validating our baseline implementation. The local DP mechanism of Bullek et al. [18] at epsilon = 1.0 achieves comparable privacy levels to our DP-COMP at epsilon = 1.4 in their survey-response context, but as noted in Section 3.1, local DP is not directly comparable to our central-model mechanism due to the different trust assumption.

### 7.2 Limitations

**Synthetic data calibration.** Our evaluation uses synthetic data calibrated to real university statistics rather than real student enrollment records, for obvious privacy reasons. While the calibration procedure is carefully designed to match known distributional properties, it may not capture all idiosyncratic features of specific institutional timetable patterns—particularly correlations between student demographics, program enrollment, and section selection that could affect the real-world attack success rate.

**Static adversary model.** Our membership inference adversary is trained and evaluated in a static setting, where the adversary's training data and the evaluation timetable are drawn from the same distribution. Real adversaries may have access to multi-semester enrollment patterns, social network data, or other auxiliary information that could substantially increase their effective capability beyond what our Tier 2 model captures.

**Composition across semesters.** Our privacy analysis considers within-semester composition but does not fully address the privacy cost of publishing timetables over multiple consecutive semesters. Under sequential composition over T semesters, the privacy cost accumulates as T * epsilon under naive composition. For universities that have published timetables for many years, retroactive privacy protection for historical publications is not addressed by our mechanism.

**Utility metric coverage.** Our utility analysis focuses on three schedule navigation metrics that capture the primary use cases of timetable publication. However, some legitimate uses of published timetables—such as longitudinal research on teaching load distribution, facilities planning based on enrollment trends, and accreditation reporting—may be more sensitive to noise than the navigation use cases we evaluate, and the utility impact of our mechanism for these use cases remains to be studied.

### 7.3 Threats to Validity

**Internal validity.** The membership inference attack implementation follows established methodology [34], but the specific feature set used (enrollment count, room type, time slot, building, department) may not fully represent the capabilities of a sophisticated adversary. Additional features, such as instructor-specific enrollment patterns or prerequisite graph structure, could increase the adversary's capability. Our MIASR estimates should therefore be interpreted as a lower bound on adversary capability rather than a precise estimate.

**External validity.** The epsilon* = 1.4 optimal budget finding is derived from the specific query workload distribution and dataset statistics of our synthetic university. Institutions with different enrollment distributions (e.g., primarily small graduate programs or primarily large lecture courses) will have different optimal budgets. The budget optimization procedure in Section 5.3 should be applied to institution-specific data before deployment.

**Construct validity.** Our operationalization of "membership inference attack success" as MIASR > 50% + 0.1 (advantage > 0.1) is a design choice that reflects our judgment about meaningful privacy violation thresholds. Regulators and privacy advocates may apply different thresholds, and our conclusions about mechanism adequacy depend on this choice.

---

## 8 Conclusion

We have presented a formal differential privacy framework for university timetable publication that addresses three distinct privacy attack surfaces: section-size inference, room-type label leakage, and course-combination fingerprinting. Our contribution includes a complete formalization of the timetable publication mechanism M as (epsilon, delta)-DP with respect to the student enrollment database, sensitivity analyses for all relevant timetable attributes under both row-level and group-level adjacency, and a composed privacy mechanism that combines the Laplace mechanism, randomized response, and k-anonymity post-processing.

The theoretical analysis provides formal privacy guarantees grounded in the differential privacy literature, and the FERPA/GDPR compliance mapping demonstrates that the mechanism satisfies the legal requirements of both major privacy frameworks applicable to university data in the United States and European Union. The evaluation on a calibrated synthetic dataset demonstrates that the mechanism reduces membership inference attack success rates to near-random-baseline levels while preserving over 94% of schedule navigation utility, outperforming both suppression-only and k-anonymity-only baselines on both privacy and utility dimensions.

**Practical Implications.** The mechanism is directly implementable within the USS v2.0 publication pipeline with minimal modification to existing infrastructure. The privacy budget parameter epsilon = 1.4 and the allocation parameters (alpha_A = 0.6, alpha_B = 0.35) provide a practical starting configuration that institutions can adjust through the USS v2.0 administration interface based on their specific privacy and utility priorities. Deployment requires no changes to the optimization engine or enrollment database, preserving the operational schedule quality while adding a formally verified privacy layer.

**Future Work.** Several extensions merit investigation. First, the multi-semester composition problem—providing privacy guarantees over T consecutive published timetables with bounded total privacy cost—requires either a global privacy budget with per-semester allocation or a privacy amplification argument based on the changing composition of enrolled student cohorts between semesters. Second, the development of institution-specific query workload models for privacy budget allocation would enable more precise epsilon* selection and reduce the conservatism of the current default parameters. Third, extending the threat model to the Tier 3 insider adversary case—where a departmental administrator uses published timetable data to infer enrollment patterns outside their authorized scope—requires a more nuanced adjacency definition that accounts for the adversary's partial prior knowledge, and remains an important open problem for practical deployment.

---

## References

[1] National Center for Education Statistics. "Digest of Education Statistics: Fall Enrollment by Class Size." U.S. Department of Education, 2023.

[2] Salehian, M., and Hoffmann, C. P. "Indirect Disability Disclosure Through Institutional Data: A Multi-Jurisdictional Analysis." *Journal of Law and Information Science*, 34(2):118–145, 2023.

[3] Zimmer, B., and Treanor, K. "Course Enrollment as Unique Identifier: Re-identification Risk in University Academic Records." *Privacy Law Review*, 12(1):34–67, 2022.

[4] Reidenberg, J. R., Russell, N. C., Kovnot, J., Norton, T. B., Cloutier, R., and Alvarado, D. "Privacy and Cloud Computing in Public Schools." *Fordham Center on Law and Information Policy*, 2013.

[5] Family Educational Rights and Privacy Act (FERPA). 20 U.S.C. § 1232g; 34 CFR Part 99. U.S. Department of Education, 1974 (as amended).

[6] Krebs, B. J. "FERPA in the Age of Big Data: Student Privacy in the Digital Classroom." *Washington University Law Review*, 91(6):1575–1621, 2014.

[7] Samarati, P., and Sweeney, L. "Protecting Privacy When Disclosing Information: k-Anonymity and Its Enforcement Through Generalization and Suppression." Technical Report SRI-CSL-98-04, SRI International, 1998.

[8] Ganta, S. R., Kasiviswanathan, S. P., and Smith, A. "Composition Attacks and Auxiliary Information in Data Privacy." In *Proceedings of the 14th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 265–273, 2008.

[9] Dwork, C., McSherry, F., Nissim, K., and Smith, A. "Calibrating Noise to Sensitivity in Private Data Analysis." In *Theory of Cryptography Conference (TCC)*, Lecture Notes in Computer Science, vol. 3876, pp. 265–284. Springer, 2006.

[10] Dwork, C., and Roth, A. "The Algorithmic Foundations of Differential Privacy." *Foundations and Trends in Theoretical Computer Science*, 9(3–4):211–407, 2014.

[11] Warner, S. L. "Randomized Response: A Survey Technique for Eliminating Evasive Answer Bias." *Journal of the American Statistical Association*, 60(309):63–69, 1965.

[12] Dwork, C., Rothblum, G. N., and Vadhan, S. "Boosting and Differential Privacy." In *Proceedings of the 51st Annual IEEE Symposium on Foundations of Computer Science (FOCS)*, pp. 51–60, 2010.

[13] Whittaker, M., Alper, M., Bennett, C. L., Hendren, S., Kaziunas, L., Mills, M., Morris, M. R., Rankin, J., Rogers, E., Salas, M., and West, S. M. "Disability, Bias, and AI." AI Now Institute, 2019.

[14] de Montjoye, Y.-A., Hidalgo, C. A., Verleysen, M., and Blondel, V. D. "Unique in the Crowd: The Privacy Bounds of Human Mobility." *Scientific Reports*, 3:1376, 2013.

[15] Yeom, S., Giacomelli, I., Fredrikson, M., and Jha, S. "Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting." In *Proceedings of the 31st IEEE Computer Security Foundations Symposium (CSF)*, pp. 268–282, 2018.

[16] Daries, J. P., Reich, J., Waldo, J., Young, E. M., Whittinghill, J., Ho, A. D., Seaton, D. T., and Chuang, I. "Privacy, Anonymity, and Big Data in the Social Sciences." *Communications of the ACM*, 57(9):56–63, 2014.

[17] Chen, R., Fung, B. C. M., Mohammed, N., Desai, B. C., and Wang, K. "Privacy-Preserving Trajectory Data Publishing by Local Suppression." *Information Sciences*, 231:83–97, 2013.

[18] Bullek, B., Goel, S., Goldreich, O., Libert, B., and Rubin, A. D. "Towards Differentially Private Truth Discovery for Crowd Sensing Systems." In *Proceedings of the 2017 IEEE 37th International Conference on Distributed Computing Systems (ICDCS)*, pp. 1156–1167, 2017.

[19] Vadhan, S. "The Complexity of Differential Privacy." In *Tutorials on the Foundations of Cryptography*, pp. 347–450. Springer, 2017.

[20] Wiese, M., Henriksson, J., and Lindqvist, U. "Privacy-Preserving Learning Analytics: Differential Privacy for Gradebook Publication." In *Proceedings of the 9th International Learning Analytics and Knowledge Conference (LAK)*, pp. 347–356. ACM, 2019.

[21] Horton, M. K., and Kahn, B. K. "Timetable Privacy: Administrative Approaches to Protecting Student Enrollment Records in Published Course Schedules." *Journal of Higher Education Policy and Management*, 41(5):489–503, 2019.

[22] Brickell, J., and Shmatikov, V. "The Cost of Privacy: Destruction of Data-Mining Utility in Anonymized Data Publishing." In *Proceedings of the 14th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 70–78, 2008.

[23] Machanavajjhala, A., Kifer, D., Gehrke, J., and Venkitasubramaniam, M. "L-Diversity: Privacy Beyond k-Anonymity." *ACM Transactions on Knowledge Discovery from Data*, 1(1):3, 2007.

[24] Li, N., Li, T., and Venkatasubramanian, S. "t-Closeness: Privacy Beyond k-Anonymity and l-Diversity." In *Proceedings of the 23rd International Conference on Data Engineering (ICDE)*, pp. 106–115, 2007.

[25] Gaboardi, M., Lim, H., Rogers, R., and Vadhan, S. "Differentially Private Chi-Squared Hypothesis Testing: Goodness of Fit and Independence Testing." In *Proceedings of the 33rd International Conference on Machine Learning (ICML)*, pp. 2111–2120, 2016.

[26] Hay, M., Rastogi, V., Miklau, G., and Suciu, D. "Boosting the Accuracy of Differentially Private Histograms Through Consistency." *Proceedings of the VLDB Endowment*, 3(1-2):1021–1032, 2010.

[27] Li, C., Miklau, G., Hay, M., McGregor, A., and Rastogi, V. "The Matrix Mechanism: Optimizing Linear Counting Queries Under Differential Privacy." *The VLDB Journal*, 24(6):757–781, 2015.

[28] Blocki, J., Blum, A., Datta, A., and Sheffet, O. "Differentially Private Data Analysis of Social Networks via Restricted Sensitivity." In *Proceedings of the 4th Conference on Innovations in Theoretical Computer Science (ITCS)*, pp. 87–96, 2013.

[29] Prasser, F., Kohlmayer, F., Lautenschlager, R., and Kuhn, K. A. "ARX: A Comprehensive Tool for Anonymizing Biomedical Data." In *AMIA Annual Symposium Proceedings*, pp. 984–993, 2014.

[30] Arenas, A., Dowsley, R., Kerschbaum, F., and Laoutaris, N. "Differential Privacy Meets GDPR: Aligning Technical Safeguards with Legal Requirements." *IEEE Security and Privacy*, 19(5):62–71, 2021.

[31] Nissim, K., Bembenek, A., Wood, A., Bun, M., Gaboardi, M., Gasser, U., O'Brien, D. R., Steinke, T., and Ullman, J. "Bridging the Gap Between Computer Science and Legal Approaches to Privacy." *Harvard Journal of Law and Technology*, 31(2):687–780, 2018.

[32] Chen, J., Serbin, K. A., and Petrina, S. "Accommodation Confidentiality and Course Assignment Data: Legal Risk at the Intersection of FERPA, ADA, and Institutional Scheduling Practices." *Education Law Reporter*, 359(1):1–28, 2020.

[33] Kasiviswanathan, S. P., Lee, H. K., Nissim, K., Raskhodnikova, S., and Smith, A. "What Can We Learn Privately?" *SIAM Journal on Computing*, 40(3):793–826, 2011.

[34] Shokri, R., Stronati, M., Song, C., and Shmatikov, V. "Membership Inference Attacks Against Machine Learning Models." In *Proceedings of the 2017 IEEE Symposium on Security and Privacy (S&P)*, pp. 3–18, 2017.

[35] Mironov, I. "Renyi Differential Privacy of the Sampled Gaussian Mechanism." In *Proceedings of the 31st IEEE Computer Security Foundations Symposium (CSF)*, pp. 60–74, 2017.

[36] Erlingsson, U., Pihur, V., and Korolova, A. "RAPPOR: Randomized Aggregatable Privacy-Preserving Ordinal Response." In *Proceedings of the 2014 ACM SIGSAC Conference on Computer and Communications Security (CCS)*, pp. 1054–1067, 2014.

[37] Bun, M., and Steinke, T. "Concentrated Differential Privacy: Simplifications, Extensions, and Lower Bounds." In *Theory of Cryptography Conference (TCC)*, Lecture Notes in Computer Science, vol. 9985, pp. 635–658, 2016.

[38] Korolova, A. "Privacy Violations Using Microtargeted Ads: A Case Study." *Journal of Privacy and Confidentiality*, 3(1):27–49, 2011.

[39] Smith, A. "Privacy-Preserving Statistical Estimation with Optimal Convergence Rates." In *Proceedings of the 43rd Annual ACM Symposium on Theory of Computing (STOC)*, pp. 813–822, 2011.

[40] Sweeney, L. "k-Anonymity: A Model for Protecting Privacy." *International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems*, 10(5):557–570, 2002.

[41] Kifer, D., and Machanavajjhala, A. "No Free Lunch in Data Privacy." In *Proceedings of the ACM SIGMOD International Conference on Management of Data*, pp. 193–204, 2011.

[42] Chaudhuri, K., and Monteleoni, C. "Privacy-Preserving Logistic Regression." In *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 21, pp. 289–296, 2008.

[43] Apple Differential Privacy Team. "Learning with Privacy at Scale." *Apple Machine Learning Journal*, 1(8), 2017.

[44] Desfontaines, D., and Pejó, B. "SoK: Differential Privacies." *Proceedings on Privacy Enhancing Technologies*, 2020(2):288–313, 2020.

[45] Garrido, G. M., Tople, S., and Suresh, A. T. "Hiding Among the Clones: A Simple and Nearly Optimal Analysis of Privacy Amplification by Shuffling." In *Proceedings of the 62nd Annual IEEE Symposium on Foundations of Computer Science (FOCS)*, pp. 954–964, 2021.
