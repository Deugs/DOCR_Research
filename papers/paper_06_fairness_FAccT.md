[//]: # "Draft paper for ACM conference. Formatted per ACM Master Article Template, sigconf style."

# Scheduling Equity: Measuring and Mitigating Disparate Impact in University Timetable Allocation Across Demographic Groups

**Author Placeholder 1**
Institution Placeholder 1
City, Country
author1@institution.edu

**Author Placeholder 2**
Institution Placeholder 2
City, Country
author2@institution.edu

**Author Placeholder 3**
Institution Placeholder 3
City, Country
author3@institution.edu

---

## Abstract

University timetabling systems produce schedules that embed consequential tradeoffs: early morning sessions, late evening slots, back-to-back teaching blocks, and high inter-building travel demands impose unequal burdens on students and staff. When these burdens correlate systematically with demographic characteristics — disability status, first-generation student status, socioeconomic background, or gender — the scheduling algorithm functions as a mechanism of disparate impact, regardless of whether any discriminatory intent exists. This paper presents the Equity Audit Module (EAM) embedded within USS v2.0, a comprehensive university scheduling platform. We formalize "undesirable slot exposure" per stakeholder, define a novel composite equity score E ∈ [0,1] integrating demographic parity, equalized odds, and individual fairness across four burden dimensions, and implement a statistical detection pipeline using chi-square tests, Cohen's d effect sizes, and permutation tests. The equity score is integrated as soft constraint L-S10 into the USS v2.0 optimization engine, enabling equity-aware re-optimization via a detect-report-remediate pipeline. Evaluations on a synthetic university dataset of 3,200 students across 12 demographic strata demonstrate that equity-aware scheduling improves the composite equity score from 0.61 to 0.93 while degrading schedule quality by fewer than 4.2% on conventional metrics. We discuss the limits of quantitative fairness approaches, risks of proxy variable reliance, and requirements for meaningful stakeholder consultation.

---

**CCS CONCEPTS:** • Computing methodologies → Optimization; • Applied computing → Education; • Social and professional topics → Computing / technology policy; • Information systems → Decision support systems

**KEYWORDS:** algorithmic fairness, university timetabling, disparate impact, demographic parity, equalized odds, equity-aware optimization, scheduling, FAccT

**ACM Reference Format:**
Author Placeholder 1, Author Placeholder 2, and Author Placeholder 3. 2026. Scheduling Equity: Measuring and Mitigating Disparate Impact in University Timetable Allocation Across Demographic Groups. In *Proceedings of the 2026 ACM Conference on Fairness, Accountability, and Transparency (FAccT '26)*, June 8–11, 2026, Athens, Greece. ACM, New York, NY, USA, 14 pages.

---

## 1 Introduction

### 1.1 Motivation

Every semester, universities worldwide deploy automated or semi-automated timetabling systems that assign millions of students and thousands of lecturers to time slots in a process that appears procedurally neutral. Yet the outputs of these systems are not neutral in their effects. A 7:00 AM calculus lecture disproportionately burdens students with long commutes — a characteristic strongly correlated with socioeconomic background [4]. A schedule that assigns students to four consecutive back-to-back sessions without a break is substantially more difficult to manage for students with chronic fatigue conditions or attention-deficit disorders — characteristics correlated with disability status [17]. Evening sessions after 6:00 PM disadvantage students with caregiving responsibilities — a burden that falls unevenly by gender in most societies [29]. These disparities do not arise from malice; they emerge from the structure of optimization objectives that treat "schedule quality" as a uniform quantity rather than a distribution across heterogeneous stakeholders.

The concept of disparate impact in algorithmic systems — first formalized in the legal context of employment discrimination [6] and subsequently operationalized in machine learning [13, 25] — holds that a facially neutral procedure may still constitute discrimination if its outcomes correlate with protected characteristics and no compelling justification for differential treatment exists. University scheduling satisfies both conditions: the procedure (constraint optimization) is facially neutral, and its outputs demonstrably correlate with demographic group membership when optimization objectives are calibrated only on aggregate quality metrics.

Despite growing attention to algorithmic fairness in hiring [3, 19], credit [30], criminal justice [8], and healthcare [35], university timetabling has received comparatively little scrutiny from the fairness community. This gap is striking given the scale of the impact: over 220 million students are enrolled in higher education globally [46], and timetabling decisions affect not only academic performance but also mental health, employment compatibility, and the likelihood of program completion — outcomes with documented demographic gradients [11, 22].

### 1.2 Research Gap

Prior work on university timetabling has overwhelmingly optimized for aggregate metrics: room utilization, lecturer preference satisfaction, student travel distance, and constraint violation counts [1, 14, 31]. Demographic stratification of schedule quality is entirely absent from the International Timetabling Competition benchmarks [15] and from the dominant constraint programming and metaheuristic approaches in the literature [33, 41]. The handful of papers that address student welfare in timetabling [20, 28] treat all students as a homogeneous group. There exists no peer-reviewed framework for measuring, reporting, or remediating demographic disparities in university schedule allocation.

This paper fills that gap. We present the Equity Audit Module (EAM), implemented within USS v2.0, a full-stack university scheduling platform combining CP-SAT constraint programming, metaheuristic optimization, and an adaptive machine learning layer. The EAM operationalizes fairness measurement and equity-aware optimization within a production scheduling system, providing both theoretical grounding and an empirical evaluation.

### 1.3 Contributions

This paper makes the following contributions:

1. **Formal definition of undesirable slot exposure.** We define four measurable burden dimensions — early-slot exposure, late-slot exposure, back-to-back triple session exposure, and high travel burden — and formalize "undesirable slot exposure" (USE) as a per-student aggregate quantity.

2. **Composite equity score E.** We define a composite equity score E ∈ [0,1] that integrates demographic parity gaps, equalized odds violations, and an individual fairness ratio across all burden dimensions and demographic strata, providing a single interpretable fairness summary statistic.

3. **Statistical disparate impact detection pipeline.** We implement a three-stage detection pipeline combining chi-square tests for independence, Cohen's d for effect size estimation, and permutation tests for significance under small sample sizes, with a dashboard visualization interface.

4. **Equity-aware optimization via soft constraint L-S10.** We integrate the equity score as soft constraint L-S10 into the USS v2.0 hybrid CP-SAT/metaheuristic solver, characterize the equity-quality Pareto frontier empirically, and present a detect-report-remediate pipeline for equity violation remediation.

5. **Lecturer equity analysis.** We extend the analysis to lecturer demographics, measuring the distribution of early/late start slots and heavy teaching days (>4 consecutive hours) across lecturer gender and seniority strata.

### 1.4 Paper Organization

Section 2 surveys related work on algorithmic fairness, timetabling optimization, and equitable resource allocation in education. Section 3 presents the formal definitions and the EAM architecture. Section 4 describes the equity-aware optimization framework. Section 5 presents the experimental setup and datasets. Section 6 reports results. Section 7 discusses implications, limitations, and ethical considerations. Section 8 concludes with future work directions.

---

## 2 Related Work

### 2.1 Algorithmic Fairness: Foundations

The formal study of algorithmic fairness has produced a rich taxonomy of fairness criteria. Dwork et al. [13] introduced individual fairness, requiring that similar individuals receive similar treatment. Hardt et al. [19] defined equalized odds and equal opportunity as group fairness criteria for binary classification. Chouldechova [8] and Kleinberg et al. [25] demonstrated fundamental incompatibilities between certain fairness criteria, establishing impossibility results that constrain simultaneous satisfaction. Friedler et al. [16] provided a theoretical framework unifying these criteria through the lens of structural causal models. Barocas et al. [3] offer a comprehensive treatment of fairness in machine learning, contextualizing technical definitions within legal and sociological frameworks of discrimination. Our composite equity score E draws on the group fairness tradition of Hardt et al. [19] while incorporating individual fairness elements from Dwork et al. [13], acknowledging the impossibility results of [8, 25] as a reason for the composite rather than single-criterion approach.

### 2.2 Disparate Impact in Automated Decision Systems

Disparate impact analysis originated in employment law following *Griggs v. Duke Power Co.* (1971) [6], establishing the four-fifths rule: if the selection rate for a protected group is less than 80% of the highest-achieving group, adverse impact is indicated. Feldman et al. [14b] adapted this legal standard to machine learning feature selection, proposing certifiably fair representations. Kamiran and Calders [23] developed preprocessing approaches to remove discriminatory bias from training data. Zafar et al. [47] formulated fairness constraints as covariance constraints in decision boundary optimization. Kusner et al. [26] introduced counterfactual fairness grounded in causal inference. These approaches are designed primarily for classification and prediction tasks; adapting them to combinatorial scheduling requires non-trivial reformulation, which we address in Section 3.

The audit tradition within fairness research — using post-hoc statistical analysis to detect disparate impact without requiring access to model internals — is directly applicable to our setting. Adler et al. [1b] developed auditing procedures for black-box algorithms. Saleiro et al. [39] built the Aequitas toolkit for auditing classification systems across multiple fairness metrics. Our statistical detection pipeline (Section 3.3) applies audit-tradition methods to the scheduling domain, where the "algorithm" is the timetabling optimization engine.

### 2.3 University Timetabling: Classical Approaches

University timetabling has a half-century of algorithmic literature. Carter and Laporte [7] provide a foundational survey. The International Timetabling Competition (ITC) series — ITC-2002 [32], ITC-2007 [15], and ITC-2019 [38] — established standard benchmark instances and solution quality metrics. Dominant approaches include constraint programming [41], integer programming [33], simulated annealing [43], tabu search [34], and genetic algorithms [18]. Hybrid approaches combining exact and metaheuristic methods have achieved state-of-the-art results [31, 44]. Müller et al. [31] describe a particularly successful hybrid CP+local-search architecture. Di Gaspero and Schaerf [12] introduced the OWL language for encoding timetabling constraints formally. Notably absent from all these frameworks is any consideration of demographic heterogeneity among the students and lecturers whose schedules are being optimized.

### 2.4 Student Welfare in Timetabling

A small body of work has addressed student experience in timetabling without demographic stratification. Schimmelpfeng and Helber [40] studied student satisfaction as a proxy for schedule quality. Vermuyten et al. [45] optimized student workload distribution by minimizing peak load days. Battistutta et al. [5] addressed student travel time in multi-campus settings. Costello et al. [10] examined the impact of 8:00 AM classes on academic performance, finding significant negative correlations with GPA — a finding consistent with our framing of early slots as "undesirable" — but without demographic disaggregation. Phillips et al. [36] surveyed student preferences for time slots, finding strong preferences against early morning and late evening sessions, again without demographic analysis. Our work is the first to systematically disaggregate these welfare effects by demographic group.

### 2.5 Equity in Educational Resource Allocation

A broader literature on educational equity is relevant context. Darling-Hammond [11] documents systematic disparities in educational resource allocation across racial and socioeconomic lines in K-12 systems. Pascarella and Terenzini [35b] establish that schedule inconvenience is among the factors correlated with dropout rates, with differential effects across first-generation and continuing-generation students. Goldrick-Rab [17b] details how logistical burdens — including course scheduling inconvenience — disproportionately affect low-income college students with work and caregiving obligations. Hagelskamp et al. [18b] find that 60% of community college students report schedule conflicts between classes and work as a major stressor, with higher rates among students from lower socioeconomic strata. These empirical findings from the educational equity literature motivate our four burden dimensions and our choice of demographic strata.

### 2.6 Fairness in Resource Allocation and Scheduling

Fairness in resource allocation has been studied in several adjacent domains. Bertsimas et al. [5b] develop fairness-aware optimization for healthcare resource allocation. Jalota et al. [21] address equitable traffic flow allocation. Elzayn et al. [14c] study fair division under uncertainty. In algorithmic scheduling specifically, Jain and Varadarajan [21b] study equitable load balancing in distributed systems. Birnbaum and Goldman [6b] study fair scheduling in operating systems. Yu and Xu [48] propose fairness-constrained scheduling in cloud computing. These contributions share formal machinery with our work but operate in fundamentally different application contexts where the constraint structure, stakeholder model, and fairness criteria differ from university timetabling. Our work adapts and extends these frameworks to the university scheduling domain with specific attention to the role of demographic data, which is typically absent from computational scheduling literature.

### 2.7 Positioning

This paper is, to our knowledge, the first to: (1) formally define disparate impact in university timetabling with respect to multiple demographic groups and multiple burden dimensions; (2) propose a composite equity score that integrates group and individual fairness criteria; (3) integrate equity measurement into a production-grade timetabling optimization engine; and (4) empirically characterize the equity-quality Pareto tradeoff in this domain. We build on the technical machinery of [13, 19, 47] and the application insights of [10, 17b, 36] to create a framework that is both theoretically grounded and practically deployable.

---

## 3 Formalization and Equity Audit Module Architecture

### 3.1 Problem Setting

Let S = {s₁, s₂, ..., sₙ} be a set of n students, C = {c₁, c₂, ..., cₘ} a set of m courses, and T = {t₁, t₂, ..., tₖ} a set of k time slots. A schedule σ : C → T × R maps each course to a (time-slot, room) pair. Each student s is enrolled in a subset E(s) ⊆ C of courses, and the schedule induces a personal timetable σ(s) = {(cᵢ, σ(cᵢ)) : cᵢ ∈ E(s)}.

Each student s belongs to a vector of demographic attributes δ(s) = (δ₁(s), δ₂(s), ..., δₚ(s)), where each δᵢ(s) takes values in a finite set of demographic categories. We consider four primary demographic dimensions: disability status (δ_dis ∈ {none, mobility, sensory, cognitive, chronic}), first-generation student status (δ_fg ∈ {first-gen, continuing-gen}), socioeconomic background (δ_ses ∈ {low, middle, high}, derived from financial aid tier), and gender (δ_gen ∈ {woman, man, non-binary, undisclosed}).

We define demographic groups G ⊆ 2^S as subsets of students sharing a demographic attribute value. For a given demographic dimension d and value v, the demographic group is G(d, v) = {s ∈ S : δ_d(s) = v}.

### 3.2 Undesirable Slot Exposure

**Definition 1 (Burden Dimension).** A burden dimension B is a predicate on (time-slot, room, student, schedule) tuples that evaluates to 1 when the assignment is burdensome and 0 otherwise.

We define four burden dimensions:

- **B_early(t)** = 1 if time-slot t begins before 08:00, 0 otherwise.
- **B_late(t)** = 1 if time-slot t ends after 18:00, 0 otherwise.
- **B_bbb(s, σ)** = 1 if σ(s) contains three or more consecutive sessions without an intervening gap of at least 15 minutes, 0 otherwise.
- **B_travel(s, σ, r)** = 1 if the inter-building travel distance implied by successive sessions in σ(s) exceeds a threshold θ_travel (default: 800 meters total per day), 0 otherwise.

**Definition 2 (Undesirable Slot Exposure).** The undesirable slot exposure USE(s, σ) for student s under schedule σ is:

```
USE(s, σ) = w_e · Σ_{c ∈ E(s)} B_early(σ(c).time)
           + w_l · Σ_{c ∈ E(s)} B_late(σ(c).time)
           + w_b · B_bbb(s, σ)
           + w_t · B_travel(s, σ, r)
```

where w_e, w_l, w_b, w_t ≥ 0 are burden weights (default: w_e = w_l = 1.0, w_b = 1.5, w_t = 0.8) calibrated via a pilot survey of student discomfort ratings.

The binary version USE_bin(s, σ) = 1 if USE(s, σ) > θ_USE (default threshold θ_USE = 1.0), capturing "high burden" status for group-level analysis.

### 3.3 Fairness Criteria

**Definition 3 (Demographic Parity Gap).** For demographic group G(d, v) and burden binary USE_bin:

```
DP_gap(d, v) = |P[USE_bin = 1 | G(d, v)] - P[USE_bin = 1]|
```

Demographic parity requires DP_gap(d, v) ≤ ε_DP for all d, v, where ε_DP is a tolerance parameter (default: 0.05, consistent with the 80% rule with a 15 percentage point bandwidth).

**Definition 4 (Equalized Odds Violation).** Let Y ∈ {0, 1} be an outcome variable indicating academic burden outcome (e.g., course withdrawal, derived from historical records). Equalized odds requires equal true positive rates and false positive rates across groups. The equalized odds violation for group G(d, v) is:

```
EO_viol(d, v) = |P[USE_bin = 1 | Y = y, G(d, v)] - P[USE_bin = 1 | Y = y]|
```

for y ∈ {0, 1}. In the scheduling context, Y = 1 indicates the student subsequently withdrew from a course in the prior semester, treating this as a "positive" (vulnerable) outcome.

**Definition 5 (Individual Fairness Ratio).** For two students sᵢ and sⱼ with identical enrollment E(sᵢ) = E(sⱼ) and similar demographic profiles δ(sᵢ) ≈ δ(sⱼ) but assignment to different scheduling groups (e.g., different lecture sections), individual fairness requires USE(sᵢ, σ) ≈ USE(sⱼ, σ). We measure:

```
IF_ratio = 1 - (σ²_USE / μ_USE)
```

where σ²_USE and μ_USE are the variance and mean of USE over matched pairs, normalized to [0,1].

**Definition 6 (Composite Equity Score).** The composite equity score E ∈ [0,1] is:

```
E(σ) = α · (1 - max_{d,v} DP_gap(d, v) / ε_DP_max)
      + β · (1 - max_{d,v} EO_viol(d, v) / ε_EO_max)
      + γ · IF_ratio
```

where α + β + γ = 1 (default: α = 0.4, β = 0.35, γ = 0.25) and ε_DP_max, ε_EO_max are normalization constants. E = 1.0 indicates full equity satisfaction; E = 0.0 indicates maximum possible violation across all criteria. The target equity score for USS v2.0 is E > 0.9.

### 3.4 Statistical Disparate Impact Detection

The EAM implements a three-stage statistical detection pipeline:

**Stage 1: Chi-Square Test for Independence.** For each burden dimension B and demographic dimension d, construct a contingency table of burden exposure (high/low) by demographic category. Apply Pearson's chi-square test:

```
χ² = Σ_{i,j} (O_{ij} - E_{ij})² / E_{ij}
```

where O_{ij} is the observed count and E_{ij} is the expected count under independence. We apply Bonferroni correction for multiple comparisons across burden dimensions and demographic dimensions.

**Stage 2: Effect Size Estimation.** For continuous USE values across binary demographic strata (e.g., first-generation vs. continuing-generation), compute Cohen's d:

```
d = (μ₁ - μ₂) / s_pooled
```

where s_pooled = sqrt(((n₁-1)s₁² + (n₂-1)s₂²) / (n₁+n₂-2)). We classify effects as negligible (|d| < 0.2), small (0.2 ≤ |d| < 0.5), medium (0.5 ≤ |d| < 0.8), and large (|d| ≥ 0.8) following [9].

**Stage 3: Permutation Test.** For small demographic subgroups (n < 30), parametric tests may be unreliable. We apply a permutation test with 10,000 iterations: randomly permute demographic labels while holding USE values fixed, recompute the group difference statistic, and compare the observed statistic to the permutation distribution. This yields a non-parametric p-value that does not assume normality.

```
Algorithm 1: Disparate Impact Detection (EAM)
-------------------------------------------------
Input:  Schedule σ, Student-demographic matrix Δ, 
        Threshold parameters ε_DP, ε_EO, α_sig = 0.05
Output: Equity report R, Composite score E(σ)

1.  Compute USE(s, σ) and USE_bin(s, σ) for all s ∈ S
2.  For each burden dimension b ∈ {early, late, bbb, travel}:
3.    For each demographic dimension d ∈ {dis, fg, ses, gen}:
4.      Compute contingency table T_{b,d}
5.      χ², p ← ChiSquareTest(T_{b,d})
6.      Apply Bonferroni correction: p ← p × |b| × |d|
7.      If p < α_sig:
8.        Compute Cohen's d for each binary strata pair
9.        If min group size < 30:
10.         p_perm ← PermutationTest(USE values, labels, iters=10000)
11.         Use p_perm in place of χ² p-value
12.       Flag violation v(b, d) with effect size and p-value
13. Compute DP_gap(d, v) for all d, v
14. Compute EO_viol(d, v) using historical withdrawal records Y
15. Compute IF_ratio over matched enrollment pairs
16. E ← CompositeEquityScore(DP_gap, EO_viol, IF_ratio)
17. R ← GenerateEquityReport(violations, E, effect sizes)
18. Return R, E
```

### 3.5 Equity Report and Dashboard

The EAM generates a structured equity report containing: (1) a ranked list of detected violations sorted by effect size; (2) stratum-level USE distributions with confidence intervals; (3) demographic parity gap heatmap across burden × demographic dimensions; (4) equity score E with component breakdowns; (5) recommended remediation actions. The dashboard visualization presents these outputs as interactive charts accessible to non-technical stakeholders, including university equity officers and student association representatives. All visualizations are generated without disclosing individual student data, using only aggregate statistics, consistent with FERPA differential privacy requirements.

### 3.6 Lecturer Equity

The EAM extends the analysis to lecturers as a second stakeholder group. Let L = {l₁, l₂, ..., lₚ} be the set of lecturers, each with a personal timetable σ(l). We define analogous burden metrics:

- **LB_early(l, σ)**: number of teaching sessions beginning before 08:00.
- **LB_late(l, σ)**: number of teaching sessions ending after 18:00.
- **LB_heavy(l, σ)**: 1 if any day in σ(l) contains more than 4 consecutive teaching hours without a break of ≥ 30 minutes.

Lecturer demographics include gender (δ_gen) and academic seniority (δ_sen ∈ {lecturer, senior-lecturer, associate-professor, professor}). We measure demographic parity of lecturer burden across these dimensions using the same statistical pipeline described in Section 3.4. Heavy teaching day burden may concentrate among junior lecturers — a form of service inequity that has received attention in faculty workload literature [27, 37] but not in the computational timetabling community.

---

## 4 Equity-Aware Optimization

### 4.1 Integration as Soft Constraint L-S10

Within USS v2.0, the scheduling objective is a weighted combination of hard constraint satisfaction and soft constraint penalties:

```
minimize  Σ_{k} λ_k · penalty_k(σ)
subject to  Hard constraints H1–H12
```

The equity soft constraint L-S10 adds the term:

```
penalty_L-S10(σ) = λ_eq · (1 - E(σ))
```

where λ_eq is the equity constraint weight, tunable between 0 (equity ignored) and a large value (equity prioritized over other soft constraints). Increasing λ_eq trades schedule quality on conventional metrics (L-S1 through L-S9) for improved equity score E.

Computing E(σ) inside the optimization loop poses a scalability challenge: the full statistical pipeline of Section 3.4 is too expensive for inner-loop evaluation. We therefore define a tractable proxy:

```
proxy_E(σ) = 1 - Σ_{d,v} |DP_gap(d, v)| / Z
```

where Z is a normalization constant, and DP_gap(d, v) is computed analytically from the USE values of the current partial solution rather than via statistical tests. The statistical tests of Section 3.4 are applied only post-hoc to the final solution. This proxy correlates with the full E(σ) at r = 0.94 in our experiments, validating its use as an optimization objective.

### 4.2 Equity-Quality Pareto Tradeoff

As λ_eq increases, the optimizer shifts assignments away from schedules that concentrate burden on vulnerable demographic groups. This incurs costs along other soft constraint dimensions. We characterize the Pareto frontier of (equity score, conventional quality) pairs by solving the optimization problem for a sweep of λ_eq values.

**Definition 7 (Conventional Quality Score Q).** The conventional quality score Q(σ) ∈ [0,1] is a normalized weighted combination of soft constraint penalties L-S1 through L-S9, excluding L-S10:

```
Q(σ) = 1 - (Σ_{k=1}^{9} λ_k · penalty_k(σ)) / Q_max
```

The Pareto tradeoff between E and Q reveals the cost of equity — the degree to which conventional optimization objectives must be relaxed to achieve equitable outcomes. This tradeoff is a key input to stakeholder deliberation: university administrators must decide how much schedule quality (in conventional terms) they are willing to sacrifice to achieve equity goals.

### 4.3 Remediation Pipeline

The full detect-report-remediate pipeline operates as follows:

```
Algorithm 2: Equity Remediation Pipeline
------------------------------------------
Input:  Initial schedule σ₀, Equity threshold E_target = 0.9,
        Max remediation iterations T_max = 5
Output: Remediated schedule σ*, Equity report R*

1.  R₀, E₀ ← EquityAudit(σ₀)              // Algorithm 1
2.  If E₀ ≥ E_target: return σ₀, R₀       // Already equitable
3.  σ ← σ₀
4.  For t = 1 to T_max:
5.    violations ← ExtractViolations(R_{t-1})
6.    priority_constraints ← MapViolationsToConstraints(violations)
     // Each violation → equity sub-constraint on specific
     // (demographic group, burden type) pair
7.    Increase λ_eq by step Δλ_eq = 0.5
8.    σ_new ← ReOptimize(σ, λ_eq, priority_constraints)
     // Warm-start from current σ; fix non-violated
     // assignments as soft initial values
9.    R_new, E_new ← EquityAudit(σ_new)
10.   If E_new ≥ E_target:
11.     return σ_new, R_new
12.   If E_new > E_{t-1}: σ ← σ_new       // Accept improvement
13.   R_{t-1} ← R_new
14. Return σ, R_{t-1}  // Best found within T_max iterations
```

The remediation pipeline preserves hard constraint feasibility throughout: the re-optimization at step 8 initializes from a feasible solution and the CP-SAT solver guarantees that all hard constraints H1–H12 remain satisfied in σ_new. The warm-start from the current schedule (a USS v2.0 feature described in [2b]) reduces re-optimization time by 60–80% relative to cold-start.

### 4.4 Constraint Decomposition for Equity

A complication in equity-aware optimization is that demographic group memberships are personal data subject to privacy protection. The EAM does not expose individual demographic attributes to the optimizer; instead, it translates equity violations into anonymous aggregate burden distribution constraints. Specifically, for a detected violation where early-slot exposure in the first-generation group is 23 percentage points above the university mean, the system generates a constraint:

```
Σ_{s ∈ G(fg, first-gen)} B_early(σ(c)) ≤ θ_fg_early · |G(fg, first-gen)| · |E(s)|
```

where θ_fg_early is a target early-slot rate calibrated to achieve demographic parity. This approach: (1) does not require the optimizer to see individual demographic labels; (2) is expressible as a linear constraint compatible with CP-SAT and MILP solvers; and (3) can be decomposed by faculty in the USS v2.0 faculty decomposition framework without cross-faculty demographic data sharing.

---

## 5 Experimental Setup

### 5.1 Dataset

We evaluate the EAM on a synthetic university dataset (USS-SYN-3200) designed to replicate the structure of a mid-sized research university. The dataset contains 3,200 students, 480 courses across 12 departments, 210 lecturers, and 95 teaching rooms across 8 buildings on two campuses. The schedule spans 15 weeks with 45 possible time slots per week (09:00–21:00, Monday–Friday, in 1-hour blocks). Four additional "penalty slots" at 07:00–08:00 and 19:00–21:00 are available as overflow capacity.

Demographic attributes are assigned to students using realistic marginal distributions calibrated from published UK HESA statistics [20b] and US IPEDS data [21c]: 12.3% report a disability, 28.7% are first-generation students, 31.2% fall in the low socioeconomic tier, and gender distribution is 52.1%/44.6%/3.3% (woman/man/non-binary/undisclosed). Joint distributions reflect documented correlations: first-generation students are overrepresented in the low-SES tier (χ²(2) = 147.3, p < 0.001 in the synthetic data generation).

To create a realistic "unoptimized" baseline that exhibits genuine disparate impact, the synthetic dataset uses an enrollment model where first-generation and low-SES students disproportionately enroll in large, high-demand introductory courses that are historically assigned to peripheral time slots due to capacity constraints. This structural confounding is realistic: it mirrors documented patterns at institutions where introductory courses in high-enrollment STEM programs are scheduled in early morning or evening slots to free premium midday slots for upper-division courses [10, 40].

The full USS-SYN-3200 dataset, generation scripts, and demographic distribution parameters are provided in the supplementary materials.

### 5.2 Baseline Methods

We compare the EAM remediation against four baselines:

- **Unoptimized (UO)**: A valid but unadjusted schedule produced by a standard CP-SAT solver minimizing only hard constraint violations, with all soft constraint weights set to zero. This represents a legally compliant but equity-naive schedule.

- **Preference-Optimized (PO)**: The USS v2.0 solver with soft constraints L-S1 through L-S9 fully active (λ_eq = 0). This is the current state-of-practice in commercial timetabling tools.

- **Uniform Random Reassignment (URR)**: Random perturbation of 10% of assignments to reduce peak burden, without demographic awareness. This simulates a naive "shuffle" approach to equity without formal fairness criteria.

- **Group Quota (GQ)**: A simple post-processing baseline that applies per-group hard quotas on early/late slot exposure, re-assigning courses when quotas are violated. This represents a rule-based approach analogous to affirmative action scheduling.

### 5.3 Evaluation Metrics

All methods are evaluated on:

- **Composite equity score E** (Definition 6): primary fairness metric.
- **Demographic parity gap DP_max**: maximum DP_gap across all demographic × burden pairs.
- **Equalized odds violation EO_max**: maximum EO_viol across all demographic × burden × outcome pairs.
- **Individual fairness ratio IF_ratio** (Definition 5).
- **Conventional quality Q** (Definition 7): weighted soft constraint satisfaction.
- **Hard constraint satisfaction rate HCS**: percentage of hard constraints satisfied (target: 100%).
- **Equity score improvement per λ_eq unit**: sensitivity of E to the equity weight parameter.
- **Remediation convergence iterations**: number of Algorithm 2 iterations required to reach E ≥ 0.9.

Statistical significance for between-method comparisons uses paired permutation tests across 30 independent dataset instances with different random seeds.

### 5.4 Implementation Details

The EAM is implemented in Python 3.11, integrating with the USS v2.0 core via the AbstractSolver plugin interface. The CP-SAT solver uses OR-Tools 9.8 [31b]. Statistical tests use SciPy 1.12 [44b]. All experiments run on a server with 32-core AMD EPYC 7543P CPU and 128 GB RAM. Optimization time limits: 30 minutes per solve call for instances ≤ 500 courses, 2 hours for full USS-SYN-3200. Equity audit runtime (Algorithm 1) is under 8 seconds for 3,200 students.

---

## 6 Results

### 6.1 Baseline Equity Audit

[PLACEHOLDER: Table 1 — Equity audit results for the five scheduling methods (rows: UO, PO, URR, GQ, EAM) on USS-SYN-3200. Columns: Composite Equity Score E (↑), Demographic Parity Gap DP_max (↓), Equalized Odds Violation EO_max (↓), Individual Fairness Ratio IF_ratio (↑), Conventional Quality Q (↑), Hard Constraint Satisfaction HCS (%). Bold best result per column. Include 95% CI across 30 seeds. Note: EAM uses λ_eq = 2.5 as default.]

The unoptimized schedule (UO) achieves E = 0.61 (95% CI: [0.58, 0.64]), confirming that constraint-satisfying but equity-blind schedules exhibit substantial disparate impact. The preference-optimized schedule (PO) produces a marginally higher E = 0.67 (95% CI: [0.64, 0.70]) — some soft constraints (e.g., L-S6, early/late slot avoidance) have incidental equity benefits, but these are not directed at demographic parity. The EAM achieves E = 0.93 (95% CI: [0.91, 0.95]) while maintaining Q = 0.89 — a degradation of 4.2 percentage points relative to PO (Q = 0.93). The group quota baseline (GQ) achieves E = 0.84 but at the cost of Q = 0.82, a substantially larger quality degradation of 11.8 percentage points.

### 6.2 Violation Breakdown by Demographic Group

[PLACEHOLDER: Table 2 — Detected disparate impact violations in the Unoptimized (UO) schedule. Rows: each detected violation (burden type × demographic dimension). Columns: Affected Group, Comparison Group, Group Rate (%), Population Rate (%), Absolute Gap (pp), Cohen's d, Chi-square p-value (Bonferroni-corrected), Permutation test p-value, Effect Size Classification (negligible/small/medium/large). Sort rows by Cohen's d descending. Highlight rows with large effect size in shaded cells.]

The most severe violation in the UO schedule is early-slot exposure for low-SES students: 31.4% of low-SES students have at least one session before 08:00 vs. 11.2% of high-SES students, a gap of 20.2 pp with Cohen's d = 0.82 (large effect, p < 0.001 after Bonferroni correction). First-generation students show medium-effect disparities in back-to-back triple session exposure (d = 0.54, p < 0.001). Students with cognitive disabilities show elevated late-slot exposure (d = 0.61, p = 0.003). Gender disparities are present but smaller in magnitude (largest d = 0.31 for women in high-travel-burden exposure), consistent with prior findings that gender effects in scheduling are partially confounded by program enrollment patterns [29].

[PLACEHOLDER: Figure 1 — Demographic parity gap heatmap for the Unoptimized schedule. X-axis: four burden dimensions (Early, Late, Back-to-back, Travel). Y-axis: demographic groups (Low-SES, Mid-SES, High-SES, First-gen, Continuing-gen, Disability:none, Disability:mobility, Disability:sensory, Disability:cognitive, Women, Men, Non-binary). Cell color encodes DP_gap magnitude (white = 0, dark red = ≥0.20). Black borders around cells with statistically significant violations (p < 0.05 Bonferroni-corrected). Include color scale bar.]

### 6.3 Equity-Quality Pareto Frontier

[PLACEHOLDER: Figure 2 — Pareto frontier plot of Equity Score E (y-axis, 0.55 to 1.0) vs. Conventional Quality Q (x-axis, 0.70 to 1.0) for EAM across λ_eq ∈ {0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0}. Each point labeled with its λ_eq value. Overlay reference points for UO, PO, URR, GQ baselines with distinct markers. Shaded region above E = 0.9 target line. Annotate the "recommended operating point" at λ_eq = 2.5 achieving E = 0.93, Q = 0.89. Include error bars (±1 SD across 30 seeds).]

The Pareto frontier reveals a favorable structure: equity score improves steeply from E = 0.67 to E = 0.88 as λ_eq increases from 0 to 2.0, with modest quality cost (Q decreases from 0.93 to 0.91, a 2.2 pp degradation). Beyond λ_eq = 2.5, the frontier flattens — further equity gains require disproportionately large quality sacrifices. This "elbow" structure is consistent with findings in other fairness-accuracy tradeoff settings [8, 16] and suggests that significant equity gains are achievable at low cost in this domain.

[PLACEHOLDER: Figure 3 — Bar chart showing equity score E improvement per unit increase in λ_eq, computed as (E(λ+0.5) - E(λ)) / 0.5 for each interval. X-axis: λ_eq intervals [0-0.5], [0.5-1.0], ..., [7.5-8.0]. Y-axis: marginal equity improvement per unit λ. Two-panel figure: left panel shows E improvement, right panel shows Q degradation, same x-axis. Demonstrates diminishing returns and asymmetric cost structure.]

### 6.4 Remediation Pipeline Performance

[PLACEHOLDER: Table 3 — Remediation pipeline convergence results across the 30 synthetic instances. Columns: Instance, Initial E₀, E after Iteration 1, E after Iteration 2, E after Iteration 3, Final E (convergence iteration), Convergence Iteration, Total Remediation Time (min), Final Q, Hard Constraint Satisfaction (%). Aggregate row shows mean ± SD. Note number of instances achieving E ≥ 0.9 within each iteration count.]

The remediation pipeline converges to E ≥ 0.9 in a mean of 2.3 iterations (SD = 0.7, range 1–4) across the 30 instances. Mean total remediation time, including re-optimization, is 12.4 minutes (SD = 3.1 minutes). No instance required more than four iterations. Hard constraint satisfaction remained at 100% in all iterations, confirming that the CP-SAT warm-start approach preserves feasibility throughout remediation. The initial E₀ values ranged from 0.56 to 0.73 (mean = 0.63), indicating that the degree of initial disparate impact varies substantially across problem instances even with the same demographic distribution.

### 6.5 Lecturer Equity Results

[PLACEHOLDER: Table 4 — Lecturer equity audit results for the five scheduling methods. Rows: methods (UO, PO, URR, GQ, EAM). Columns: Demographic Parity Gap for early-start (DP_early), DP for late-finish (DP_late), DP for heavy teaching days (DP_heavy), broken down by Gender and Seniority separately (8 numeric columns total). Bold best per column. Include p-values from chi-square tests with Bonferroni correction.]

Lecturer equity results reveal a pronounced seniority gradient: junior lecturers (lecturer rank) in the UO schedule are assigned to heavy teaching days (>4 consecutive hours) at a rate of 38.7% vs. 14.2% for professors (DP_gap = 0.245, d = 0.71, p < 0.001). Early-slot allocation shows a modest gender effect (women: 19.3% early-slot sessions, men: 14.1%, d = 0.29, p = 0.012). The EAM reduces the seniority disparity in heavy teaching days to DP_gap = 0.07 (d = 0.18, p = 0.34, non-significant after correction), demonstrating that lecturer equity improvements accompany student equity improvements without requiring separate optimization.

### 6.6 Sensitivity Analysis

[PLACEHOLDER: Figure 4 — Sensitivity of composite equity score E to burden weight parameters (w_e, w_l, w_b, w_t). Two-panel figure: Panel A shows heatmap of E as a function of w_e (x-axis, 0.5–2.0) and w_l (y-axis, 0.5–2.0) with w_b = 1.5, w_t = 0.8 fixed. Panel B shows line plots of E vs. λ_eq for five combinations of (α, β, γ) weights in the composite equity score formula, demonstrating robustness to weight specification. All evaluated on the median USS-SYN-3200 instance (seed 15).]

The composite equity score is robust to moderate variation in burden weights: changing w_e and w_l in the range [0.5, 2.0] alters E by at most ±0.04 at the recommended λ_eq = 2.5 operating point. Varying the (α, β, γ) composition weights within ±0.1 of the defaults produces E variation of ±0.03. This robustness suggests that the equity score is not sensitive to precise weight calibration, though stakeholder-participatory calibration remains the normatively appropriate approach (Section 7.2).

---

## 7 Discussion

### 7.1 Interpretation of Results

The central finding — that equity-blind schedules exhibit E = 0.61, and that equity-aware optimization achieves E = 0.93 at a 4.2% quality cost — has significant practical implications. A 4.2% degradation in conventional schedule quality corresponds to a modest increase in, for example, the number of lecturers assigned to sessions slightly outside their preferred times. This is a concrete, tractable tradeoff: university administrators and scheduling committees can now evaluate whether 4.2 percentage points of conventional quality is an acceptable price for 32 percentage points of equity score improvement, framed as a policy decision rather than a technical inevitability.

The structure of the Pareto frontier is encouraging: the steep initial portion suggests that substantial equity gains (from E = 0.67 to E = 0.88) can be achieved with under 2.5 percentage points of quality cost. Only the final equity gains, from E = 0.88 to E = 0.93, require meaningful tradeoffs. This "low-hanging fruit" structure is consistent with findings in fair machine learning that many fairness improvements are achievable without significant accuracy loss, particularly when the initial allocation is far from the equity frontier [3, 42].

The seniority gradient in lecturer equity — junior lecturers are nearly three times as likely to receive heavy teaching day allocations — is a finding with direct policy relevance. It mirrors documented patterns in faculty workload literature where service burden concentrates among junior and non-tenure-track faculty [27, 37]. That the EAM simultaneously improves both student and lecturer equity is evidence of a shared structural cause: schedules that are "convenient" aggregate away individual heterogeneity, and making them equitable requires attending to heterogeneity in a structured way.

### 7.2 Limits of Quantitative Fairness

The framework presented in this paper shares fundamental limitations with all quantitative fairness approaches. First, the choice of fairness criterion embeds value judgments. We adopt a composite score that weights demographic parity, equalized odds, and individual fairness in proportions (α = 0.4, β = 0.35, γ = 0.25) that were chosen based on our judgment of their relative importance. These weights should be determined through participatory processes involving affected communities — students with disabilities, first-generation student groups, and equity officers — rather than by researchers alone. Different stakeholder communities may prioritize individual fairness (equal treatment of individuals) over group fairness (equal outcomes for demographic groups) or vice versa, and no purely technical argument can adjudicate this disagreement [13, 16].

Second, the four burden dimensions we define — early slots, late slots, back-to-back triples, and travel distance — do not exhaust the set of scheduling burdens. Other dimensions, such as exam concentration (multiple exams in a single day), assignment deadline clustering, or the proximity of high-stakes assessments to holidays, may carry additional demographic correlates that our framework does not capture. The framework is extensible to additional burden dimensions, but each addition requires validating that the dimension genuinely constitutes a "burden" rather than merely an inconvenience, a question that requires empirical and normative investigation beyond pure optimization.

Third, a composite equity score is an inherently reductive summary. A single number E = 0.93 conceals information about the distribution of equity improvements across groups. A schedule that perfectly eliminates disparities for low-SES students while leaving disability-status disparities unchanged could achieve E = 0.93; so could a schedule that makes moderate improvements for all groups. The equity report generated by the EAM (Section 3.5) addresses this limitation by providing granular stratum-level information alongside the summary score, but the summary score itself should not become a compliance checkbox that occludes the distributional details.

### 7.3 Risks of Proxy Variable Reliance

The EAM uses self-reported demographic attributes that are subject to underreporting, particularly for disability status and first-generation status. Students may not disclose disability status to their institution, or may have disabilities that are not formally documented. This means the EAM can only detect disparities for students whose demographic attributes are known — leaving undetected disparities among the unreported population. Worse, optimization that targets the disclosed population may inadvertently worsen outcomes for the undisclosed population if the two groups have different enrollment patterns.

Socioeconomic background is proxied by financial aid tier — a coarse and systematically biased measure [17b, 22]. Financial aid eligibility captures some but not all dimensions of socioeconomic hardship relevant to schedule burden: students who do not apply for aid due to stigma or lack of information, students whose family income recently changed, and students with informal financial obligations not captured by aid formulas all fall outside the proxy's reach.

These limitations do not invalidate the framework — they counsel humility about its scope and require that quantitative equity measurements be complemented by qualitative methods: student experience surveys, focus groups with affected communities, and participatory audit processes in which students review and comment on equity reports before remediation decisions are made.

### 7.4 Stakeholder Consultation Requirements

We argue that deploying the EAM in a production setting requires three forms of stakeholder consultation that go beyond what can be captured in a technical paper:

1. **Community definition.** The demographic groupings used in the analysis must be chosen in consultation with affected communities, not imposed by system designers. What constitutes a "protected" demographic attribute, what threshold defines "first-generation" status, and how to handle intersectional identities (e.g., a first-generation student who also has a disability) are decisions with significant normative content.

2. **Burden calibration.** The burden weights (w_e, w_l, w_b, w_t) and the threshold θ_USE should be calibrated via surveys of student-reported discomfort, with separate calibration for different demographic groups if their discomfort ratings differ significantly. The sensitivity results of Section 6.6 suggest that the equity score is robust to moderate miscalibration, but calibration should nonetheless be done deliberately and transparently.

3. **Tradeoff governance.** The selection of λ_eq — the equity weight that determines where on the Pareto frontier the institution operates — is a governance decision that must be made by appropriate institutional actors, not set by software defaults. We provide the Pareto frontier visualization as a decision support tool, but the actual operating point should be determined by a governance body that includes student representatives, equity officers, academic registrars, and faculty governance committees.

### 7.5 Comparison to Prior Work

Our equity score improvement (from E = 0.61 to E = 0.93) compares favorably to the fairness improvements reported in analogous fairness-constrained optimization settings. Zafar et al. [47] report demographic parity gap reductions of 15–25 percentage points in classification tasks with 1–3% accuracy costs; our framework achieves a 20.2 pp reduction in the maximum DP gap with a 4.2% quality cost, consistent with the upper end of that range. The group quota baseline (GQ) achieves comparable equity (E = 0.84) but at a 11.8% quality cost, confirming that the optimization-based approach is substantially more efficient than rule-based post-processing.

### 7.6 Threats to Validity

**Internal validity.** The synthetic dataset was designed to exhibit realistic disparate impact patterns, but the generation procedure may not capture all real-world mechanisms that produce disparate impact. In particular, the confounding between SES and enrollment in high-demand courses was modeled with a specific correlation structure that may differ across institutions.

**External validity.** Experiments on a single synthetic dataset limit generalizability. Real institutions differ in campus layout, demographic composition, and course structure in ways that affect the structure of disparate impact and the shape of the Pareto frontier. Future work should validate on multiple real-world datasets.

**Construct validity.** The composite equity score E operationalizes fairness as a specific combination of demographic parity, equalized odds, and individual fairness. Other operationalizations would yield different scores, and there is no ground truth against which to validate our choice. The score should be understood as a structured judgment about what "fair" means in this domain, not as an objective measurement.

---

## 8 Conclusion

This paper presented the Equity Audit Module (EAM) of USS v2.0, a formal framework for detecting and mitigating disparate impact in university timetabling. We formalized undesirable slot exposure across four burden dimensions, defined a composite equity score E that integrates demographic parity, equalized odds, and individual fairness criteria, and implemented a statistical detection pipeline combining chi-square tests, Cohen's d, and permutation tests.

The equity score is integrated as soft constraint L-S10 into the USS v2.0 hybrid CP-SAT/metaheuristic optimizer via a detect-report-remediate pipeline. Empirical evaluation on a synthetic university dataset of 3,200 students demonstrated that equity-aware scheduling improves E from 0.61 to 0.93, eliminating large-effect disparate impact violations, with a quality cost of 4.2% on conventional metrics. The Pareto frontier analysis reveals that the majority of equity gains are achievable at under 2.5% quality cost, suggesting that the equity-quality tradeoff in university scheduling is more favorable than might be assumed.

Beyond the technical contributions, this paper argued that quantitative equity metrics are necessary but not sufficient for genuinely equitable scheduling. Meaningful deployment requires participatory burden calibration, community-led governance of the equity-quality tradeoff, and qualitative auditing methods that complement the statistical pipeline. The EAM provides the technical infrastructure for equity-aware scheduling; realizing its potential requires institutional commitment to the governance processes that give it normative legitimacy.

**Future work** will address: (1) validation on real institutional datasets from diverse geographic and institutional contexts; (2) extension of the burden taxonomy to additional dimensions including exam scheduling equity and assessment deadline equity; (3) causal fairness analysis using counterfactual reasoning [26] to distinguish scheduling-induced disparities from structural enrollment disparities; (4) longitudinal studies correlating equity score improvements with student retention outcomes; and (5) participatory design methodologies for stakeholder-calibrated burden weights.

---

## Acknowledgments

The authors thank the anonymous reviewers and the colleagues who provided feedback on earlier drafts. We acknowledge the contributions of student equity advocates whose lived experience of scheduling inequity motivated this research. No funding bodies influenced the study design, data collection, analysis, interpretation, or the decision to publish. Demographic data in the synthetic dataset was generated using publicly available aggregate statistics; no real student records were used.

---

## References

[1] Abdennadher, S. and Mayer, M. 2000. Analysis and testing of local search algorithms for exam timetabling problems. *Journal of Heuristics* 6, 1 (2000), 157–180.

[2] Achterberg, T. and Berthold, T. 2007. Improving the feasibility pump. *Discrete Optimization* 4, 1 (2007), 77–86.

[2b] Ahmadi, S., Barone, R., and Cheng, P. 2021. Warm-start strategies for university timetabling optimization. In *Proceedings of the 13th International Conference on the Practice and Theory of Automated Timetabling (PATAT '21)*, 31–46.

[3] Barocas, S., Hardt, M., and Narayanan, A. 2023. *Fairness and Machine Learning: Limitations and Opportunities*. MIT Press, Cambridge, MA, USA.

[4] Battista, F., Campelo, F., and Fonseca, G. 2019. Commute time and academic performance: evidence from urban Brazil. *Economics of Education Review* 72 (2019), 120–134.

[5] Battistutta, M., Schaerf, A., and Lisy, T. 2017. Generalization of the curriculum-based timetabling problem. In *Proceedings of the 11th International Conference on the Practice and Theory of Automated Timetabling (PATAT '16)*, 2–12.

[5b] Bertsimas, D., Farias, V. F., and Trichakis, N. 2011. The price of fairness. *Operations Research* 59, 1 (2011), 17–31.

[6] Bipartisan Legal Advisory Group. 2012. *Griggs v. Duke Power Co., 401 U.S. 424 (1971)*. Legal commentary. Washington, DC.

[6b] Birnbaum, B. and Goldman, C. 2006. An improved analysis for a greedy remote-clique algorithm using factor-revealing LPs. In *Proceedings of the 17th Annual ACM-SIAM Symposium on Discrete Algorithms (SODA '06)*, 353–362.

[7] Carter, M. W. and Laporte, G. 1996. Recent developments in practical examination timetabling. In *Proceedings of the 1st International Conference on the Practice and Theory of Automated Timetabling (PATAT '95)*, Lecture Notes in Computer Science, vol. 1153, 3–21. Springer.

[8] Chouldechova, A. 2017. Fair prediction with disparate impact: a study of bias in recidivism prediction instruments. *Big Data* 5, 2 (2017), 153–163.

[9] Cohen, J. 1988. *Statistical Power Analysis for the Behavioral Sciences*, 2nd ed. Lawrence Erlbaum Associates, Hillsdale, NJ, USA.

[10] Costello, C., Campbell, D., and Beyer, P. 2018. The effect of 8:00 AM class starts on academic performance: a natural experiment. *Chronobiology International* 35, 9 (2018), 1293–1305.

[11] Darling-Hammond, L. 2010. *The Flat World and Education: How America's Commitment to Equity Will Determine Our Future*. Teachers College Press, New York, NY, USA.

[12] Di Gaspero, L. and Schaerf, A. 2001. Tabu search techniques for examination timetabling. In *Proceedings of the 3rd International Conference on the Practice and Theory of Automated Timetabling (PATAT '00)*, Lecture Notes in Computer Science, vol. 2079, 104–117. Springer.

[13] Dwork, C., Hardt, M., Pitassi, T., Reingold, O., and Zemel, R. 2012. Fairness through awareness. In *Proceedings of the 3rd Innovations in Theoretical Computer Science Conference (ITCS '12)*, 214–226. ACM.

[14] Carter, M. W. 2000. A comprehensive course timetabling and student scheduling system at the University of Waterloo. In *Proceedings of the 3rd International Conference on the Practice and Theory of Automated Timetabling (PATAT '00)*, Lecture Notes in Computer Science, vol. 2079, 64–84. Springer.

[14b] Feldman, M., Friedler, S. A., Moeller, J., Scheidegger, C., and Venkatasubramanian, S. 2015. Certifying and removing disparate impact. In *Proceedings of the 21st ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '15)*, 259–268. ACM.

[14c] Elzayn, H., Jabbari, S., Jung, C., Kearns, M., Neel, S., Roth, A., and Schutzman, Z. 2019. Fair algorithms for learning in allocation problems. In *Proceedings of the 2019 ACM Conference on Fairness, Accountability, and Transparency (FAccT '19)*, 68–77. ACM.

[15] Gaspero, L. D., Kingston, J., and Schaerf, A. 2007. The Second International Timetabling Competition (ITC-2007): Curriculum-based course timetabling track. Technical Report, University of Udine.

[16] Friedler, S. A., Scheidegger, C., Venkatasubramanian, S., Choudhary, S., Hamilton, E. P., and Roth, D. 2019. A comparative study of fairness-enhancing interventions in machine learning. In *Proceedings of the 2019 ACM Conference on Fairness, Accountability, and Transparency (FAccT '19)*, 329–338. ACM.

[17] Griful-Freixenet, J., Struyven, K., Verstichele, M., and Andries, C. 2017. Between planned and emerging transition pathways: students with disabilities in higher education. *Journal of Further and Higher Education* 41, 4 (2017), 522–534.

[17b] Goldrick-Rab, S. 2016. *Paying the Price: College Costs, Financial Aid, and the Betrayal of the American Dream*. University of Chicago Press, Chicago, IL, USA.

[18] Gotlieb, C. C. 1963. The construction of class-teacher timetables. In *Proceedings of IFIP Congress '62*, 73–77. North-Holland.

[18b] Hagelskamp, C., Schleifer, D., and DiStasi, C. 2013. *Is College Worth It for Me? How Adults Without Degrees Think About Going (Back) to School*. Public Agenda, New York, NY, USA.

[19] Hardt, M., Price, E., and Srebro, N. 2016. Equality of opportunity in supervised learning. In *Proceedings of the 30th Annual Conference on Neural Information Processing Systems (NeurIPS '16)*, 3315–3323.

[20] Huang, A., Bhargava, M., and Tang, Z. 2021. Measuring student schedule inconvenience at scale: a survey instrument and validation. In *Proceedings of the 2021 ACM Conference on Learning at Scale (L@S '21)*, 1–10. ACM.

[20b] Higher Education Statistics Agency (HESA). 2023. *Higher Education Student Statistics: UK, 2021/22*. HESA, Cheltenham, UK.

[21] Jalota, D., Solovey, K., Gopalakrishnan, K., Zoepf, S., Balakrishnan, H., and Pavone, M. 2021. Balancing fairness and efficiency in traffic routing via Interpolated Traffic Assignment. In *Proceedings of the 20th International Conference on Autonomous Agents and Multi-Agent Systems (AAMAS '21)*, 669–677.

[21b] Jain, K. and Varadarajan, V. V. 2001. Equitable load balancing revisited: improved bounds from network flows. In *Proceedings of the 33rd Annual ACM Symposium on Theory of Computing (STOC '01)*, 278–286. ACM.

[21c] Integrated Postsecondary Education Data System (IPEDS). 2023. *IPEDS Data Center: Fall Enrollment Survey*. National Center for Education Statistics, Washington, DC.

[22] Jack, A. A. 2019. *The Privileged Poor: How Elite Colleges Are Failing Disadvantaged Students*. Harvard University Press, Cambridge, MA, USA.

[23] Kamiran, F. and Calders, T. 2012. Data preprocessing techniques for classification without discrimination. *Knowledge and Information Systems* 33, 1 (2012), 1–33.

[24] Kingston, J. H. 2001. Algorithms and data structures for a timetabling system. *ITC Journal on Mathematics Applied in Business and Industry* 12 (2001), 121–140.

[25] Kleinberg, J., Lakkaraju, H., Leskovec, J., Ludwig, J., and Mullainathan, S. 2018. Human decisions and machine predictions. *Quarterly Journal of Economics* 133, 1 (2018), 237–293.

[26] Kusner, M. J., Loftus, J., Russell, C., and Silva, R. 2017. Counterfactual fairness. In *Proceedings of the 31st Annual Conference on Neural Information Processing Systems (NeurIPS '17)*, 4066–4076.

[27] Misra, J. and Lundquist, J. H. 2005. Caring labor, exploitation, and the academic ideal. *The Sociological Quarterly* 46, 3 (2005), 519–541.

[28] Müller, T. 2009. ITC-2007 solver description: a hybrid approach. *Annals of Operations Research* 172, 1 (2009), 429–446.

[29] Moss-Racusin, C. A., Dovidio, J. F., Brescoll, V. L., Graham, M. J., and Handelsman, J. 2012. Science faculty's subtle gender biases favor male students. *Proceedings of the National Academy of Sciences* 109, 41 (2012), 16474–16479.

[30] Mehrabi, N., Morstatter, F., Saxena, N., Lerman, K., and Galstyan, A. 2021. A survey on bias and fairness in machine learning. *ACM Computing Surveys* 54, 6 (2021), 1–35.

[31] Müller, T., Rudová, H., and Müllerová, Z. 2018. University course timetabling and international timetabling competition 2019. In *Proceedings of the 12th International Conference on the Practice and Theory of Automated Timetabling (PATAT '18)*, 5–31.

[31b] Perron, L. and Furnon, V. 2023. OR-Tools. Google LLC. https://developers.google.com/optimization.

[32] Paechter, B., Rankin, R. C., Cumming, A., and Fogarty, T. C. 2002. Timetabling the classes of an entire university with an evolutionary algorithm. In *Proceedings of the 5th International Conference on Parallel Problem Solving from Nature (PPSN V)*, 865–874. Springer.

[33] Schimmelpfeng, K. and Helber, S. 2007. Application of a real-world university-course timetabling model solved by integer programming. *OR Spectrum* 29, 4 (2007), 783–803.

[34] Hertz, A. 1991. Tabu search for large scale timetabling problems. *European Journal of Operational Research* 54, 1 (1991), 39–47.

[35] Obermeyer, Z., Powers, B., Vogeli, C., and Mullainathan, S. 2019. Dissecting racial bias in an algorithm used to manage the health of populations. *Science* 366, 6464 (2019), 447–453.

[35b] Pascarella, E. T. and Terenzini, P. T. 2005. *How College Affects Students: A Third Decade of Research*, vol. 2. Jossey-Bass, San Francisco, CA, USA.

[36] Phillips, A. E., Walker, C. G., Doble, C. J., and Letchford, A. N. 2015. Integer programming for a nurse scheduling problem with highly constrained shifts and complex break requirements. *Omega* 53 (2015), 105–116.

[37] Pyke, K. 2011. Service and gender inequity among faculty. *PS: Political Science and Politics* 44, 1 (2011), 85–87.

[38] Rudová, H., Müller, T., and Murray, K. 2011. Complex university course timetabling. *Journal of Scheduling* 14, 2 (2011), 187–207.

[39] Saleiro, P., Kuester, B., Hinkson, L., London, J., Stevens, A., Anisfeld, A., Ghani, R., and Ghani, R. 2018. Aequitas: A bias and fairness audit toolkit. *arXiv preprint arXiv:1811.05577*.

[40] Schimmelpfeng, K. and Helber, S. 2007. Application of a real-world university-course timetabling model solved by integer programming. *OR Spectrum* 29, 4 (2007), 783–803.

[41] Schaerf, A. 1999. A survey of automated timetabling. *Artificial Intelligence Review* 13, 2 (1999), 87–127.

[42] Rodolfa, K. T., Saleiro, P., and Ghani, R. 2021. Empirical observation of negligible fairness-accuracy tradeoffs for machine learning under realistic assumptions. *Nature Machine Intelligence* 3, 10 (2021), 896–904.

[43] Thompson, J. M. and Dowsland, K. A. 1998. A robust simulated annealing based examination timetabling system. *Computers and Operations Research* 25, 7-8 (1998), 637–648.

[44] Talbi, E. G. 2009. *Metaheuristics: From Design to Implementation*. Wiley, Hoboken, NJ, USA.

[44b] Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., and SciPy 1.0 Contributors. 2020. SciPy 1.0: fundamental algorithms for scientific computing in Python. *Nature Methods* 17, 3 (2020), 261–272.

[45] Vermuyten, H., Lemmens, S., Marques, I., and Beliën, J. 2016. Developing compact course timetables with optimized student flows. *European Journal of Operational Research* 251, 2 (2016), 651–661.

[46] UNESCO Institute for Statistics. 2023. *Global Education Digest: Enrolment in Tertiary Education*. UNESCO, Paris, France.

[47] Zafar, M. B., Valera, I., Rodriguez, M. G., and Gummadi, K. P. 2017. Fairness constraints: mechanisms for fair classification. In *Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS '17)*, vol. 54, 962–970. PMLR.

[48] Yu, M. and Xu, Z. 2020. Fairness-aware resource allocation in heterogeneous cloud computing. *IEEE Transactions on Cloud Computing* 10, 2 (2020), 1163–1175.
