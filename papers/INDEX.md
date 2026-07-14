# USS v2.0 Publication Series — Master Paper Index

## Introduction

The **University Scheduling System v2.0 (USS v2.0)** is a comprehensive intelligent scheduling platform designed to address the full complexity of academic timetabling in higher education institutions. Built upon multi-agent reinforcement learning (MARL), combinatorial optimization, and data-driven inference, USS v2.0 integrates constraint satisfaction, stakeholder preference modeling, fairness auditing, sustainability metrics, real-time IoT sensing, privacy-preserving publication, and conversational interfaces into a unified deployable system. This series of 12 papers was produced to disseminate the research contributions embedded in USS v2.0 across the relevant scientific communities — spanning operations research, artificial intelligence, human-computer interaction, data privacy, sustainability, and systems engineering — with each paper targeting the venue of highest impact for its specific theme. Together, the series provides a complete scientific record of USS v2.0: from its algorithmic foundations and learned preference models, through its fairness and sustainability objectives, to its infrastructure guarantees and real-world deployment outcomes.

---

## Master Table

| # | Short Title | Target Venue | Filename | Status |
|---|-------------|--------------|----------|--------|
| 01 | Hybrid CP-Metaheuristic Solving | Journal of Scheduling (Springer) | paper_01_hybrid_optimization_JoS.md | Draft |
| 02 | Preference Weight Learning | AIED | paper_02_weight_learning_AIED.md | Draft |
| 03 | Impact-Aware Conflict Resolution | AAAI | paper_03_conflict_resolution_AAAI.md | Draft |
| 04 | Interactive Pareto Frontier | EMO | paper_04_multiobjective_EMO.md | Draft |
| 05 | What-If Simulation Framework | Decision Support Systems (Elsevier) | paper_05_whatif_DSS.md | Draft |
| 06 | Scheduling Equity & Fairness | FAccT | paper_06_fairness_FAccT.md | Draft |
| 07 | Green Scheduling | Journal of Cleaner Production (Elsevier) | paper_07_sustainability_JCP.md | Draft |
| 08 | Blockchain Audit Trails | IEEE Transactions on Services Computing | paper_08_blockchain_TSC.md | Draft |
| 09 | IoT-Augmented Dynamic Scheduling | IEEE Internet of Things Journal | paper_09_iot_IoTJ.md | Draft |
| 10 | Conversational NLP Interface | IUI | paper_10_nlp_interface_IUI.md | Draft |
| 11 | Differential Privacy Publication | PoPETs | paper_11_privacy_PoPETs.md | Draft |
| 12 | USS v2.0 Full System Paper | INFORMS Journal on Applied Analytics | paper_12_full_system_INFORMS.md | Draft |

---

## Paper Summaries

---

### Paper 01 — Hybrid Constraint Programming and Metaheuristic Solving for Large-Scale University Timetabling: Warm-Start Initialization, Faculty Decomposition, and Adaptive Phase Transition

**Target Venue:** Journal of Scheduling (Springer)
**File:** `paper_01_hybrid_optimization_JoS.md`
**Estimated Submission:** Q3 2026

**Summary:**
This paper presents a hybrid solver that combines constraint programming (CP) with metaheuristic search to tackle large-scale university timetabling at the scale of real institutional deployments. The core contribution is a warm-start initialization strategy that seeds metaheuristic population from CP-derived partial solutions, dramatically reducing convergence time over cold-start baselines. An adaptive phase-transition mechanism dynamically shifts solver effort between CP and metaheuristic search based on solution quality signals, and a faculty decomposition scheme partitions the problem into tractable subproblems that are solved independently before global reconciliation.

**Key Contributions:**
- Warm-start initialization protocol that extracts feasible seed solutions from partial CP runs to initialize metaheuristic populations
- Faculty decomposition strategy that partitions large timetabling instances into independent departmental subproblems for parallel solving
- Adaptive phase-transition controller that monitors search stagnation and switches between CP and metaheuristic modes dynamically
- Empirical evaluation on real university instances demonstrating significant improvements in solution quality and time-to-feasibility over state-of-the-art baselines
- Open benchmark suite of large-scale university timetabling instances derived from anonymized institutional data

---

### Paper 02 — Learning Institutional Scheduling Preferences from Historical Approval Data: Adaptive Soft Constraint Weight Inference for University Timetabling

**Target Venue:** International Conference on Artificial Intelligence in Education (AIED)
**File:** `paper_02_weight_learning_AIED.md`
**Estimated Submission:** Q3 2026

**Summary:**
University timetabling involves numerous soft constraints whose relative importance varies substantially across institutions and changes over time as policies evolve; this paper proposes a machine learning framework that infers soft constraint weights directly from historical scheduling approval data. The approach treats historical approve/reject decisions by scheduling administrators as implicit preference signals and frames weight inference as an inverse optimization problem solved via gradient-based learning. An adaptive update mechanism allows the learned weights to drift with institutional policy shifts, enabling the system to remain aligned with current preferences without manual reconfiguration.

**Key Contributions:**
- Formal inverse optimization formulation for soft constraint weight inference from binary approval history
- Gradient-based weight learning algorithm with convergence guarantees under mild assumptions on the approval data distribution
- Adaptive drift detection mechanism that triggers weight re-estimation when distributional shift is detected in approval patterns
- User study demonstrating that schedules generated with learned weights receive significantly higher administrator satisfaction scores than those with manually set weights
- Integration pathway into USS v2.0 as a plug-in preference calibration module

---

### Paper 03 — Impact-Aware Conflict Resolution in Multi-Stakeholder University Scheduling: Learning to Rank Fixes with Ripple Effect Modeling

**Target Venue:** AAAI Conference on Artificial Intelligence
**File:** `paper_03_conflict_resolution_AAAI.md`
**Estimated Submission:** Q2 2026

**Summary:**
When scheduling conflicts arise — due to last-minute room unavailability, faculty illness, or enrollment surges — resolving them naively often creates cascading secondary conflicts throughout the timetable; this paper formalizes conflict resolution as a learning-to-rank problem where candidate fix actions are ranked by their predicted downstream ripple effect. A graph neural network encodes the timetable as a constraint hypergraph and predicts the propagation depth and scope of each candidate fix, allowing the system to prefer minimally disruptive repairs. The model is trained on simulated conflict-repair trajectories generated from real institutional timetabling data.

**Key Contributions:**
- Graph neural network architecture for ripple-effect prediction over timetable constraint hypergraphs
- Learning-to-rank formulation for conflict resolution that optimizes for minimal downstream disruption
- Trajectory simulation procedure for generating training data from historical timetabling instances without requiring labeled repair logs
- Comprehensive evaluation on a held-out set of real scheduling disruptions, demonstrating reduction in secondary conflicts compared to rule-based and greedy baselines
- Ablation study isolating the contribution of ripple-effect modeling versus simple local-repair heuristics

---

### Paper 04 — Interactive Pareto Frontier Exploration for Multi-Objective University Timetabling with Population Warm-Starting from Historical Solutions

**Target Venue:** International Conference on Evolutionary Multi-Criterion Optimization (EMO)
**File:** `paper_04_multiobjective_EMO.md`
**Estimated Submission:** Q3 2026

**Summary:**
University timetabling is inherently multi-objective, involving trade-offs between student travel time, faculty preference satisfaction, room utilization efficiency, and fairness; this paper presents an interactive Pareto frontier exploration interface backed by a multi-objective evolutionary algorithm (MOEA) whose initial population is warm-started from an archive of high-quality historical schedules. The interactive component allows decision-makers to navigate the Pareto frontier in real time, adjusting objective weights and seeing the frontier shift responsively, supporting progressive articulation of preferences. Population warm-starting is shown to substantially accelerate convergence to a well-spread Pareto approximation set compared to random initialization.

**Key Contributions:**
- Population warm-starting strategy for MOEAs using an archive of historical Pareto-optimal solutions to seed evolutionary search
- Interactive Pareto exploration interface supporting real-time weight adjustment and frontier visualization
- Preference elicitation protocol that translates decision-maker interactions into scalarized objective updates
- Empirical analysis demonstrating faster hypervolume convergence and better frontier spread under warm-start initialization across multiple institutional datasets
- Formal characterization of the historical archive diversity criterion that maximizes warm-start benefit

---

### Paper 05 — Scenario-Driven Decision Support for University Scheduling: A What-If Simulation and Comparison Framework for Enrollment, Resource, and Policy Changes

**Target Venue:** Decision Support Systems (Elsevier)
**File:** `paper_05_whatif_DSS.md`
**Estimated Submission:** Q4 2026

**Summary:**
University administrators must plan timetables under uncertainty about future enrollment, room availability, and policy directives; this paper presents a what-if simulation framework embedded in USS v2.0 that allows planners to define scenarios (e.g., 15% enrollment increase in Computer Science, loss of a large lecture hall) and immediately compare the scheduling implications across multiple proposed timetables. The framework generates a decision report quantifying constraint violation changes, objective score shifts, and stakeholder impact distributions for each scenario-timetable pair. A comparative ranking interface helps administrators identify which current timetable design is most resilient to plausible future changes.

**Key Contributions:**
- Scenario specification language for encoding enrollment, resource, and policy perturbations as structured what-if queries
- Rapid re-optimization module that applies scenario perturbations to an existing timetable and produces updated solutions within interactive time bounds
- Multi-timetable comparison dashboard with automated statistical significance annotation for objective score differences
- Case study with three participating universities demonstrating the framework's utility in strategic facilities planning
- Integration of scenario results into a resilience index that ranks timetable designs by robustness across a scenario portfolio

---

### Paper 06 — Scheduling Equity: Measuring and Mitigating Disparate Impact in University Timetable Allocation Across Demographic Groups

**Target Venue:** ACM Conference on Fairness, Accountability, and Transparency (FAccT)
**File:** `paper_06_fairness_FAccT.md`
**Estimated Submission:** Q2 2026

**Summary:**
University timetabling decisions — such as time-of-day assignments and room location choices — can systematically disadvantage students from specific demographic groups, particularly those with caregiving responsibilities, part-time employment constraints, or longer commutes; this paper introduces a fairness audit framework for university timetabling that measures disparate impact across demographic groups using both individual and group fairness metrics. Post-hoc mitigation strategies are evaluated that add fairness soft constraints to the optimization model without substantially degrading overall schedule quality. The framework is applied to two real institutional datasets with student demographic annotations, revealing previously unnoticed disparities in scheduling outcomes.

**Key Contributions:**
- Taxonomy of fairness criteria applicable to university timetabling, distinguishing individual, group, and intersectional notions
- Automated fairness audit pipeline that computes disparity metrics from a timetable and a student demographic dataset
- Constrained optimization formulation that incorporates demographic fairness as a soft constraint with an adjustable penalty weight
- Empirical analysis on two real datasets quantifying the equity-efficiency trade-off frontier
- Policy recommendations for institutional scheduling offices to operationalize fairness monitoring as a routine audit step

---

### Paper 07 — Green University Scheduling: Integrating Carbon Footprint Minimization as an Optimization Objective in Academic Timetabling

**Target Venue:** Journal of Cleaner Production (Elsevier)
**File:** `paper_07_sustainability_JCP.md`
**Estimated Submission:** Q4 2026

**Summary:**
Universities are significant contributors to urban carbon emissions through building heating, ventilation, and cooling (HVAC) loads driven by occupancy patterns; this paper proposes a carbon-aware timetabling objective that estimates the marginal HVAC energy cost of assigning a course to a specific room at a specific time and incorporates this estimate as an additional optimization objective alongside conventional scheduling criteria. Crucially, the model accounts for inter-period thermal inertia, so scheduling back-to-back courses in the same room yields lower per-student carbon cost than spreading them across rooms. The resulting multi-objective timetabling problem is solved with a carbon-weighted scalarization, and the framework is validated against real building energy monitoring data.

**Key Contributions:**
- Carbon footprint estimation model for timetable assignments based on room thermal profiles and historical HVAC energy data
- Thermal inertia term that captures energy savings from consecutive room occupancy, incentivizing dense scheduling patterns
- Multi-objective timetabling formulation that exposes a quality-versus-carbon trade-off frontier for administrator decision support
- Validation of the energy model against sub-metered building data at a pilot campus
- Estimation of institution-level annual CO2 reduction achievable by adopting carbon-aware scheduling recommendations

---

### Paper 08 — Blockchain-Anchored Audit Trails for Academic Scheduling Systems: Tamper-Evident Records, Digital Approval Chains, and Zero-Trust Integration

**Target Venue:** IEEE Transactions on Services Computing
**File:** `paper_08_blockchain_TSC.md`
**Estimated Submission:** Q4 2026

**Summary:**
Academic scheduling decisions involve multiple approvals from faculty, department chairs, registrars, and facilities managers, and disputes about what was approved when are a recurring source of administrative conflict; this paper presents a blockchain-anchored audit trail architecture integrated into USS v2.0 that records every scheduling decision, approval event, and modification as an immutable on-chain entry linked to a digital identity. The architecture follows zero-trust principles, treating every system component as untrusted and requiring cryptographic proof of authorization before any schedule mutation is accepted. A cost analysis demonstrates that a permissioned blockchain (Hyperledger Fabric) provides sufficient throughput for institutional scheduling workloads while keeping per-transaction costs negligible.

**Key Contributions:**
- Blockchain audit trail schema for scheduling events, capturing actor identity, action type, affected records, and timestamps in a tamper-evident structure
- Digital approval chain protocol enforcing multi-party sign-off workflows with cryptographic non-repudiation
- Zero-trust integration architecture that decouples scheduling system components and enforces authorization at every interface
- Performance evaluation on a Hyperledger Fabric deployment under realistic institutional scheduling event rates
- Compliance mapping demonstrating how the architecture satisfies FERPA record-keeping and institutional accreditation audit requirements

---

### Paper 09 — IoT-Augmented Dynamic University Scheduling: Real-Time Room State Fusion for Adaptive Timetable Optimization

**Target Venue:** IEEE Internet of Things Journal
**File:** `paper_09_iot_IoTJ.md`
**Estimated Submission:** Q3 2026

**Summary:**
Static timetables cannot react to real-world disruptions such as rooms being occupied past their scheduled end time, unexpected equipment failures, or sudden spikes in room temperature that render a space unsuitable for occupancy; this paper presents an IoT-augmented scheduling layer for USS v2.0 that fuses real-time sensor streams — occupancy, temperature, air quality, AV equipment status — into a live room state model and triggers timetable micro-adjustments when state deviations exceed defined thresholds. A lightweight replanning algorithm produces minimally disruptive substitutions (alternative rooms, time swaps) within seconds of a detected anomaly. The system is evaluated in a live campus deployment over one academic semester.

**Key Contributions:**
- Multi-modal room state fusion model integrating occupancy, environmental, and equipment sensor streams via a Kalman filter bank
- Anomaly detection layer that classifies room state deviations by severity and triggers appropriate replanning responses
- Real-time replanning algorithm optimized for sub-10-second response latency under typical campus sensor event rates
- Semester-long live deployment evaluation quantifying the frequency and nature of IoT-triggered schedule adjustments
- Failure mode analysis characterizing system behavior under sensor dropout and communication loss scenarios

---

### Paper 10 — Conversational Scheduling: A Natural Language Query Interface for University Timetabling Administration

**Target Venue:** ACM International Conference on Intelligent User Interfaces (IUI)
**File:** `paper_10_nlp_interface_IUI.md`
**Estimated Submission:** Q2 2026

**Summary:**
Scheduling administrators interact with complex timetabling systems primarily through rigid form-based interfaces that require detailed knowledge of system internals; this paper presents a conversational natural language interface for USS v2.0 that allows administrators to issue scheduling commands, queries, and what-if requests in plain English. The interface uses a domain-specific semantic parsing model fine-tuned on a collected corpus of scheduling intent utterances, maps parsed intents to USS v2.0 API calls, and returns structured responses rendered as natural language summaries or embedded visualizations. A controlled user study demonstrates significantly lower task completion time and higher satisfaction compared to the form-based baseline.

**Key Contributions:**
- Annotated corpus of 4,500 scheduling administration utterances covering commands, queries, and hypothetical scenarios, released as an open dataset
- Domain-adapted semantic parser for scheduling intent recognition with intent-slot structured output aligned to the USS v2.0 API
- Response generation module that translates structured API outputs into fluent natural language summaries and inline chart descriptions
- Controlled user study (n=42 administrators) demonstrating task completion time reduction and satisfaction improvement over form-based interface
- Error analysis of parser failures categorized by intent type, informing future data collection priorities

---

### Paper 11 — Differentially Private University Timetable Publication: Protecting Student Enrollment Privacy Under FERPA and GDPR

**Target Venue:** Proceedings on Privacy Enhancing Technologies (PoPETs)
**File:** `paper_11_privacy_PoPETs.md`
**Estimated Submission:** Q3 2026

**Summary:**
Published university timetables contain implicit information about student enrollment patterns that can be used to re-identify individuals or infer sensitive attributes such as major, religious practice, or disability status through timing of accommodated sections; this paper develops a differential privacy framework for timetable publication that adds calibrated noise to enrollment-linked schedule attributes while preserving the timetable's utility for planning purposes. The privacy mechanism is designed to satisfy both FERPA and GDPR data minimization requirements and is accompanied by a formal utility analysis showing the degradation in schedule quality metrics as a function of the privacy budget epsilon. An audit tool is released that allows institutions to assess the re-identification risk of their published timetables.

**Key Contributions:**
- Formal threat model for timetable publication privacy, enumerating attack vectors that exploit enrollment-linked scheduling patterns
- Differential privacy mechanism for timetable publication with epsilon-delta guarantees calibrated to FERPA and GDPR risk thresholds
- Utility analysis quantifying the trade-off between privacy budget and schedule quality metrics including room utilization and constraint violation rates
- Re-identification risk audit tool released as open-source software for institutional self-assessment
- Legal compliance analysis mapping the technical privacy guarantees to specific FERPA and GDPR regulatory provisions

---

### Paper 12 — USS v2.0: A Comprehensive Intelligent Scheduling Platform for Higher Education — Architecture, Deployment, and Organizational Impact

**Target Venue:** INFORMS Journal on Applied Analytics
**File:** `paper_12_full_system_INFORMS.md`
**Estimated Submission:** Q1 2027

**Summary:**
This paper presents the full USS v2.0 system as an integrated platform, describing its architecture, the multi-year institutional deployment process, and the measurable organizational impact observed at three pilot universities. It synthesizes findings from across the research papers in this series, showing how each component — optimization engine, preference learning, conflict resolution, fairness auditing, sustainability module, IoT layer, privacy framework, and conversational interface — functions as a coherent whole under production conditions. Quantitative outcomes are reported including schedule quality improvements, administrator time savings, student satisfaction increases, and carbon footprint reductions, alongside qualitative organizational change findings from stakeholder interviews.

**Key Contributions:**
- Reference architecture for USS v2.0, documenting component interfaces, data flows, deployment topology, and integration with institutional ERP systems
- Longitudinal deployment study across three universities capturing quantitative KPIs over two academic years
- Synthesis of all series papers into a unified framework narrative, articulating how each research contribution addresses a distinct gap in practical intelligent scheduling
- Organizational impact analysis using mixed-methods evaluation combining KPI dashboards with structured stakeholder interviews
- Roadmap for future USS development directions based on deployment experience and unsolved challenges identified in practice

---

## Recommended Submission Order

The following submission order is recommended based on two principles: (1) **foundational algorithmic papers should be submitted first** so that applied and system papers can cite published or pre-published versions, and (2) **high-impact AI conferences with strict deadlines should be targeted early** to maximize visibility before the full system paper unifies the narrative.

### Wave 1 — Core AI and Algorithms (Submit Q2 2026)

Submit **Paper 03 (AAAI)**, **Paper 06 (FAccT)**, and **Paper 10 (IUI)** first. These three papers target top-tier AI and HCI conferences with fixed annual deadlines. AAAI (Paper 03) establishes the core conflict resolution intelligence of USS v2.0 as a standalone scientific contribution. FAccT (Paper 06) addresses a high-visibility policy-relevant topic where early publication will attract collaborators and institutional partners. IUI (Paper 10) rounds out the interface contribution before the full system paper references it.

### Wave 2 — Optimization Foundations (Submit Q3 2026)

Submit **Paper 01 (JoS)**, **Paper 02 (AIED)**, **Paper 04 (EMO)**, **Paper 09 (IoTJ)**, and **Paper 11 (PoPETs)** in the second wave. Paper 01 is the foundational solver paper; it should be in review before the system paper (Paper 12) needs to cite a stable description. Papers 02 and 04 build directly on the solver and benefit from Paper 01 being under review. Paper 09 is technically self-contained but benefits from the established system context. Paper 11 addresses privacy, a hot topic with fast review cycles at PoPETs.

### Wave 3 — Applied and System Papers (Submit Q4 2026)

Submit **Paper 05 (DSS)**, **Paper 07 (JCP)**, and **Paper 08 (TSC)** in Wave 3. These papers are inherently application-oriented and benefit from citing Waves 1 and 2 results. Journal submissions here allow longer revision cycles that align with the Wave 4 timeline.

### Wave 4 — Full System Synthesis (Submit Q1 2027)

Submit **Paper 12 (INFORMS JAA)** last. This is the system integration paper and must cite all prior papers in the series. Submitting after Waves 1-3 ensures the system paper can reference accepted or at-minimum published preprints of all component papers, significantly strengthening its empirical and scientific grounding.

---

## Cross-Paper Citation Map

The table below indicates citation relationships within the series. A checkmark in row X, column Y means Paper X cites Paper Y. Diagonal cells are omitted. Cells are left blank where no direct citation is anticipated.

| Cites -> | P01 | P02 | P03 | P04 | P05 | P06 | P07 | P08 | P09 | P10 | P11 | P12 |
|----------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| **P01**  |  —  |     |     |     |     |     |     |     |     |     |     |     |
| **P02**  |  Y  |  —  |     |     |     |     |     |     |     |     |     |     |
| **P03**  |  Y  |  Y  |  —  |     |     |     |     |     |     |     |     |     |
| **P04**  |  Y  |  Y  |     |  —  |     |     |     |     |     |     |     |     |
| **P05**  |  Y  |  Y  |  Y  |  Y  |  —  |     |     |     |     |     |     |     |
| **P06**  |  Y  |  Y  |  Y  |  Y  |     |  —  |     |     |     |     |     |     |
| **P07**  |  Y  |     |     |  Y  |     |     |  —  |     |     |     |     |     |
| **P08**  |  Y  |     |     |     |     |     |     |  —  |     |     |  Y  |     |
| **P09**  |  Y  |     |  Y  |     |     |     |     |     |  —  |     |     |     |
| **P10**  |     |  Y  |  Y  |     |  Y  |     |     |     |     |  —  |     |     |
| **P11**  |     |  Y  |     |     |     |  Y  |     |  Y  |     |     |  —  |     |
| **P12**  |  Y  |  Y  |  Y  |  Y  |  Y  |  Y  |  Y  |  Y  |  Y  |  Y  |  Y  |  —  |

**Notes on citation logic:**
- Paper 12 cites all others as the synthesis paper.
- Papers 02, 03, 04, 05, 06 cite Paper 01 because they build on or evaluate using the USS v2.0 solver.
- Paper 03 cites Paper 02 because conflict resolution uses preference weights from the weight learning module.
- Papers 04, 05, 06 cite Paper 03 because their scenarios involve or assume a conflict resolution layer.
- Paper 05 cites Papers 03 and 04 because what-if simulation generates Pareto frontiers and may trigger conflict resolution.
- Paper 06 cites Papers 02, 03, 04 because fairness analysis uses the same weight and objective infrastructure.
- Paper 08 cites Paper 11 because the blockchain audit trail and differential privacy mechanisms share the identity and record-linkage architecture.
- Paper 09 cites Paper 03 because IoT-triggered replanning invokes the conflict resolution module.
- Paper 10 cites Papers 02, 03, 05 because the NL interface exposes preference, conflict, and what-if capabilities.
- Paper 11 cites Papers 02, 06, 08 because privacy guarantees interact with preference data, demographic data, and audit records.
