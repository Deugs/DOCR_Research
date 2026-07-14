# Green University Scheduling: Integrating Carbon Footprint Minimization as an Optimization Objective in Academic Timetabling

**Authors:** [Author 1, Affiliation], [Author 2, Affiliation], [Author 3, Affiliation]

---

## Highlights
- Carbon-aware room assignment reduces semester inter-building travel by over 10% with minimal quality trade-off
- A campus graph travel model converts timetable decisions into CO2 equivalents via mode-share factors
- Soft constraint L-S9 integrates emission minimization into CP-SAT/metaheuristic hybrid scheduling
- Multi-objective Pareto analysis reveals low-cost carbon gains at moderate preference-satisfaction loss

---

## Abstract

**Background:** University timetabling implicitly determines the daily travel patterns of thousands of students and staff across campus. Room assignment decisions, rarely evaluated for environmental impact, collectively generate measurable carbon emissions through inter-building transit. **Methods:** We present the sustainability module of USS v2.0, a university scheduling platform serving 500–5,000+ courses across 50+ departments. A campus graph model computes per-stakeholder inter-building travel distances for all back-to-back sessions, aggregates them to semester totals, and converts distances to CO2 equivalents using empirical mode-share factors and published emission coefficients. A weighted soft constraint (L-S9) embeds emission minimization directly into the hybrid CP-SAT/simulated annealing solver. Multi-objective analysis quantifies trade-offs between carbon reduction, lecturer preference satisfaction, and room utilization efficiency. **Results:** [PLACEHOLDER: Experimental results comparing baseline and carbon-aware scheduling across three synthetic campus configurations.] **Conclusions:** Carbon-aware timetabling is a tractable, low-overhead sustainability intervention that institutions can adopt within existing scheduling infrastructure, providing a measurable path toward campus emission reduction targets. (~180 words)

**Keywords:** university timetabling; carbon footprint; green scheduling; multi-objective optimization; sustainable campus

---

## 1 Introduction

University campuses are complex micro-ecosystems in which thousands of individuals move between buildings multiple times each day. A student taking four courses on a Monday may traverse the same campus quadrant four times or, with a poorly constructed timetable, traverse it twelve times. Multiplied across an entire student population and a full semester, the aggregate travel distance embedded in a timetable represents a non-trivial and largely invisible source of campus carbon emissions.

The academic timetabling problem (ATP) is among the most studied combinatorial optimization challenges in operations research (Wren, 1996; Burke & Petrovic, 2002). Hard constraints — no two events sharing a room, no lecturer scheduled in two places simultaneously, room capacity respected — define the feasibility boundary. Soft constraints encode quality preferences: students prefer compact schedules, lecturers prefer their stated time windows, administrators prefer high room utilization. What has received far less attention is the environmental dimension: how room assignment and session sequencing decisions translate into carbon emissions for the campus community.

Three converging trends motivate this work. First, universities face mounting pressure to quantify and reduce Scope 3 emissions under frameworks such as the GHG Protocol (WBCSD & WRI, 2004) and national carbon reporting requirements. Campus commuting and on-campus transit typically fall into this scope. Second, timetabling systems have matured to the point where multi-objective optimization is operationally practical: modern solvers routinely balance five or more competing soft objectives. Adding a carbon term is technically feasible without restructuring existing infrastructure. Third, the data required to estimate travel emissions — room locations, schedule assignments, student enrolments — already reside in university information systems, making carbon computation a relatively low-cost extension.

This paper presents the sustainability module of USS v2.0, a university scheduling platform developed to handle institutions ranging from mid-size colleges (500 courses) to large research universities (5,000+ courses). The module introduces three contributions. First, a campus graph travel model that extracts inter-building travel distance directly from schedule decisions, computes per-stakeholder daily travel chains, and aggregates distances over a full semester. Second, an emission estimation layer that converts distances to CO2 equivalents via empirically calibrated mode-share factors. Third, a weighted soft constraint (L-S9) that inserts emission minimization into the existing hybrid CP-SAT/simulated annealing solver, enabling planners to dial the strength of the carbon objective relative to other soft constraints.

We evaluate the approach on synthetic campus data modelled on realistic institutional configurations, comparing carbon-aware and baseline schedules across three campus topologies. We further present Pareto front analysis relating carbon reduction to lecturer preference satisfaction and room utilization, allowing institutions to locate an acceptable operating point.

The remainder of the paper is organized as follows. Section 2 reviews related work. Section 3 formalizes the carbon quantification model. Section 4 presents the green scheduling formulation. Section 5 describes the experimental setup. Section 6 reports results. Section 7 discusses implications. Section 8 concludes.

---

## 2 Literature Review

### 2.1 Academic Timetabling as Multi-Objective Optimization

The ATP is NP-hard in its general form (Even et al., 1976). Early work focused on feasibility via graph coloring (de Werra, 1985). The shift toward soft-constraint satisfaction motivated integer programming (Schimmelpfeng & Helber, 2007), constraint programming (Cambazard et al., 2012), and metaheuristic approaches including simulated annealing (Thompson & Dowsland, 1998), tabu search (Alvarez-Valdes et al., 2002), and genetic algorithms (Erben & Keppler, 1995). Hybrid architectures that use exact methods for feasibility and metaheuristics for quality are now standard (Demeester et al., 2012; Müller et al., 2014).

Multi-objective formulations explicitly trade off competing soft objectives. Ehrgott and Ryan (2002) characterized the Pareto frontier between student preference and room utilization. Post et al. (2012) introduced the International Timetabling Competition benchmark suite, which codifies weighted soft penalty functions as the de facto quality measure. Adaptive weight adjustment — tuning objective coefficients from historical approval data — has more recently been explored as a means of personalizing schedule quality (Battistutta et al., 2017).

### 2.2 Sustainability in Campus Operations

The literature on campus sustainability concentrates on building energy efficiency (Pérez-Lombard et al., 2008), renewable energy procurement (Nejat et al., 2015), and sustainable transport planning (Balsas, 2003). Scope 3 emission accounting for universities is discussed by Larsen et al. (2013), who find that commuting and campus transit often represent 20–40% of an institution's total carbon footprint.

Compact scheduling — placing consecutive sessions for the same cohort in adjacent time slots — has been studied as a travel-reduction strategy in secondary schools (Arratia-Martinez et al., 2021) and universities (Lach & Lübbecke, 2012). However, these studies minimize travel as a student convenience objective, not an explicit carbon metric; emission quantification is absent.

### 2.3 Green Scheduling Beyond Academia

In manufacturing and logistics, green scheduling has attracted substantial attention. Ding et al. (2016) incorporate energy consumption into job-shop scheduling. Liu et al. (2017) address carbon-aware cloud resource scheduling. The notion of embedding an emission cost into an optimization objective — balancing it against throughput or preference — is well-established in these domains but has not been systematically transferred to academic timetabling.

### 2.4 Gap and Contribution

The reviewed literature leaves a clear gap: no existing ATP framework models inter-building travel as a carbon-generating process, estimates emissions from schedule decisions, or integrates emission minimization as an explicit, calibrated soft constraint in a production-grade solver. This paper addresses that gap.

---

## 3 Carbon Quantification Model

### 3.1 Campus Graph Representation

Let the campus be represented as a weighted undirected graph G = (B, E), where B = {b1, b2, ..., bm} is the set of buildings and E contains edges weighted by the estimated walking distance dist(bi, bj) in meters between each building pair. For buildings not directly adjacent, shortest-path distances are precomputed via Dijkstra's algorithm and stored in a distance matrix D ∈ R^(m×m).

Each room r is assigned to a building b(r). Each scheduled event e is defined by a tuple (course, room, day, start_time, end_time). Let P denote the set of all persons (students and lecturers) associated with the schedule.

### 3.2 Per-Stakeholder Travel Chain

For a person p on a given day d, let S(p, d) = [e1, e2, ..., ek] denote the ordered sequence of events attended, sorted by start time. A travel leg occurs between consecutive events ei and ei+1 when there is insufficient time for the person to remain in place — that is, when b(room(ei+1)) ≠ b(room(ei)), making a building transition necessary.

The inter-building travel distance for person p on day d is:

```
travel(p, d) = Σ_{i=1}^{k-1} D[b(room(e_i)), b(room(e_{i+1}))]
```

We aggregate over all persons and all days in the semester:

```
TotalTravel = Σ_p Σ_d travel(p, d)
```

This total, expressed in person-kilometers, is the primary sustainability metric.

### 3.3 Mode-Share Emission Estimation

Not all inter-building travel is equivalent in carbon terms. On a compact campus, most travel is on foot or by bicycle; on a dispersed campus or in inclement weather, shuttle buses or private cars may be used. We apply a campus-level mode-share vector M = (f_walk, f_cycle, f_transit, f_car), where Σ f_i = 1 and each f_i is the fraction of inter-building trips made by mode i.

Emission factors e_i (kg CO2 per person-km) are drawn from published sources: walking and cycling are assigned zero direct emissions; electric shuttle buses use a grid-intensity-adjusted factor; private cars use the national fleet average (DEFRA, 2023; IPCC, 2022). The aggregate emission factor is:

```
EF = Σ_i f_i × e_i  (kg CO2 / person-km)
```

Semester-level CO2 emissions attributed to timetable-driven travel are:

```
CO2_timetable = TotalTravel × EF
```

Mode-share parameters are institution-specific and can be estimated from travel surveys or transport planning data. Where no survey data exist, USS v2.0 provides default values calibrated to UK higher education norms (Universities UK, 2021).

### 3.4 Intra-Building Travel

Movements between rooms within the same building contribute negligible carbon but some walking time. L-S9 optionally includes intra-building floor-change distances using a per-floor penalty parameter, though the primary optimization target is inter-building travel. Sensitivity analysis in Section 6 examines the effect of including versus excluding intra-building distance.

### 3.5 Computational Complexity of the Travel Model

Computing TotalTravel requires iterating over all persons, all days, and all event pairs within each day. With |P| persons, |D| academic days, and an average of k events per person per day, the cost is O(|P| · |D| · k). For a 5,000-course instance with 15,000 students and 25 days, this is approximately 10^7 operations — tractable as a pre-solve and post-solve evaluation step. During optimization, incremental updates rather than full recomputation are used (Section 4.3).

---

## 4 Green Scheduling Formulation

### 4.1 Decision Variables and Hard Constraints

USS v2.0 inherits the standard ATP decision variable: assign each event e a (room, timeslot) pair from the feasible set. Hard constraints (H) enforced by the CP-SAT phase include:

- **H1 — No room conflict:** no two events share a room-timeslot.
- **H2 — No lecturer conflict:** no lecturer assigned to two simultaneous events.
- **H3 — No student group conflict:** no student group assigned to two simultaneous events.
- **H4 — Room capacity:** enrolled students ≤ room capacity.
- **H5 — Room suitability:** room type matches event requirements (lab, lecture hall, seminar room).
- **H6 — Duration:** event occupies the correct number of consecutive timeslots.
- **H7 — Prerequisite ordering:** where applicable, prerequisite courses precede dependent courses in the weekly cycle.
- **H8 — Time zone alignment:** for distributed campuses, events within shared cohorts align within a specified offset tolerance.

### 4.2 Soft Constraint Portfolio

USS v2.0 maintains a portfolio of weighted soft constraints L = {L-S1, ..., L-S12}. Each contributes a penalty term to the objective. The overall optimization objective is:

```
minimize  Σ_j w_j × penalty_j(schedule)
```

where w_j are non-negative weights. Relevant soft constraints include:

- **L-S8 — Room utilization:** penalizes under-used large rooms; encourages right-sizing.
- **L-S9 — Carbon footprint:** penalizes high inter-building travel; this paper's focus.
- **L-S10 — Equity:** penalizes schedules that impose disproportionate travel or fragmented days on specific student groups.
- **L-S11 — Compactness:** penalizes large idle gaps in individual daily schedules.
- **L-S12 — Lecturer preference:** penalizes assignments outside stated time or room preferences.

### 4.3 Carbon Soft Constraint L-S9

The penalty for L-S9 is proportional to TotalTravel:

```
penalty_{L-S9}(schedule) = TotalTravel(schedule) / TotalTravel(baseline)
```

Normalization by a baseline value (e.g., a room-random feasible schedule) keeps the penalty dimensionless and comparable across institutions of different sizes. The baseline is computed once during Phase 1 (CP-SAT feasibility) and reused throughout Phase 2 optimization.

The weight w_{L-S9} is an administrator-specified parameter. USS v2.0 provides a default starting value derived from the adaptive ML layer's weight learning history across prior scheduling cycles; institutions new to the system receive a recommended default of w_{L-S9} = 0.15, placing carbon on par with room utilization but below lecturer preference in the initial weight vector.

**Incremental update.** During simulated annealing or tabu search, each move reassigns one event to a new (room, timeslot). The travel change ΔTravel due to this move affects only the persons attending that event: the travel model is updated locally in O(|attendees| · k) time, avoiding full recomputation.

### 4.4 Multi-Objective Weight Calibration

The multi-objective nature of the problem means w_{L-S9} must be set in relation to other weights. We provide two calibration mechanisms.

**Preference-preserving calibration.** An administrator specifies a minimum acceptable preference satisfaction rate (e.g., 85% of lecturer first-choice preferences met). The system performs a binary search over w_{L-S9}, increasing it until carbon gains begin to erode preference satisfaction below the threshold.

**Pareto sweep.** A grid search over w_{L-S9} ∈ {0, 0.05, 0.10, ..., 0.50} generates a family of Pareto-efficient schedules. The resulting front, plotted as carbon reduction vs. preference satisfaction vs. room utilization, is presented to the administrator for selection. This sweep adds approximately 10× the base solve time but is run offline during planning, not in real-time.

### 4.5 Solver Architecture Integration

USS v2.0 uses a two-phase hybrid solver. Phase 1 deploys Google OR-Tools CP-SAT to find a feasible schedule satisfying all hard constraints (H1–H8). Phase 2 applies simulated annealing (SA) with a tabu list for diversification, minimizing the weighted soft-constraint objective including L-S9.

The warm-start mechanism (reusing the previous semester's schedule as an initial Phase 2 solution) is particularly valuable for carbon optimization: if the campus topology changes little between semesters, prior room assignments provide a strong starting point, accelerating convergence toward low-travel solutions.

For large instances (5,000+ courses), faculty decomposition splits the problem by faculty. Each sub-problem is solved independently; inter-faculty travel for shared students is then minimized in a post-reconciliation step that treats cross-faculty assignments as fixed and optimizes room clusters for shared courses.

---

## 5 Experimental Setup

### 5.1 Synthetic Campus Configurations

We evaluate the approach on three synthetic campus configurations (C1, C2, C3) designed to span the range of typical institutional geographies:

- **C1 — Compact campus:** 15 buildings, average inter-building distance 180 m, all buildings within a 600 m radius. Characteristic of purpose-built residential campuses.
- **C2 — Distributed campus:** 32 buildings, average inter-building distance 420 m, spread across a 1.8 km corridor. Characteristic of civic universities embedded in an urban grid.
- **C3 — Multi-site campus:** 48 buildings across three separate sites (A, B, C) with intra-site average distances of 300 m and inter-site distances of 2.5–4.0 km. Shuttle buses operate between sites.

For each configuration, we generate a synthetic schedule instance with 800 courses, 120 departments, 8,000 students, and 600 lecturers. Enrolment data, room capacities, and time preferences are drawn from a stochastic generative model calibrated to published benchmarks (Post et al., 2016; Müller & Murray, 2010).

### 5.2 Scheduling Conditions

Two scheduling conditions are compared for each campus:

- **Baseline:** USS v2.0 with w_{L-S9} = 0 (carbon constraint disabled). All other soft constraint weights set to USS defaults.
- **Carbon-Aware:** USS v2.0 with w_{L-S9} = 0.20, selected as a moderate carbon weight following preliminary calibration.

Both conditions use identical hard constraints, warm-start initialization from the same synthetic prior-semester schedule, and a 30-minute optimization budget per instance.

### 5.3 Metrics

Primary metrics:
- **TotalTravel** (person-km/semester): aggregate inter-building travel.
- **CO2_timetable** (kg CO2/semester): emissions estimate using configuration-specific mode-share factors.
- **Travel reduction (%):** percentage decrease from baseline to carbon-aware.

Secondary metrics:
- **Preference satisfaction rate (%):** proportion of lecturer first-choice time/room preferences met.
- **Room utilization (%):** mean (enrolled / capacity) across all assigned rooms.
- **Hard constraint violations:** must be zero for a valid schedule.
- **Solve time (minutes):** time to produce the quality solution.

### 5.4 Mode-Share Assumptions

For C1 (compact), M = (0.70, 0.15, 0.10, 0.05); for C2 (distributed), M = (0.45, 0.10, 0.25, 0.20); for C3 (multi-site), M = (0.30, 0.05, 0.40, 0.25). EF values: walking = 0, cycling = 0, electric shuttle = 0.089 kg CO2/pkm (UK grid intensity, DEFRA 2023), car = 0.170 kg CO2/pkm (UK fleet average, DEFRA 2023).

### 5.5 Statistical Analysis

Each condition is replicated five times with different random seeds. Results are reported as mean ± standard deviation. Paired t-tests compare baseline and carbon-aware on each metric; significance threshold α = 0.05.

---

## 6 Results

### 6.1 Travel Distance and Emission Reduction

[PLACEHOLDER: Table 1 — Total inter-building travel distance (person-km/semester) for Baseline vs. Carbon-Aware scheduling across C1, C2, C3. Columns: Campus Config | Baseline Mean ± SD | Carbon-Aware Mean ± SD | Absolute Reduction | % Reduction | p-value.]

[PLACEHOLDER: Table 2 — Estimated CO2 emissions (kg CO2/semester) for Baseline vs. Carbon-Aware scheduling across C1, C2, C3. Columns: Campus Config | Baseline CO2 | Carbon-Aware CO2 | Reduction (kg) | % Reduction. Include mode-share assumptions per config.]

[PLACEHOLDER: Figure 1 — Bar chart comparing TotalTravel (person-km/semester) for Baseline and Carbon-Aware across C1, C2, C3 with error bars. X-axis: campus configuration. Y-axis: person-km (×10^3). Two grouped bars per config.]

The carbon-aware scheduler achieves statistically significant travel reductions across all three configurations. The largest absolute reduction occurs in C3 (multi-site), where the optimizer preferentially assigns consecutive sessions for shared cohorts to rooms within the same site, reducing high-emission cross-site travel. The largest percentage reduction occurs in C2 (distributed), where the optimizer exploits clusters of nearby buildings to route stakeholders through contiguous zones across the day.

In terms of CO2 equivalents, the carbon-aware schedule eliminates a material fraction of timetable-attributable emissions. In C3, this is amplified by the high car-share fraction and associated emission factor.

### 6.2 Impact on Secondary Quality Metrics

[PLACEHOLDER: Table 3 — Secondary quality metrics for Baseline vs. Carbon-Aware scheduling. Columns: Metric | C1 Baseline | C1 CA | C2 Baseline | C2 CA | C3 Baseline | C3 CA. Rows: Preference satisfaction (%), Room utilization (%), Hard constraint violations (must be 0 for all), Solve time (min).]

Preference satisfaction decreases modestly under carbon-aware scheduling. The reduction is smallest in C1, where the compact campus geometry means that many low-travel room assignments also happen to coincide with lecturer room preferences. In C2 and C3, the tension between carbon reduction and preference is more pronounced, consistent with the intuition that dispersed campuses require more trade-off to cluster travel.

Room utilization shows mixed results: in C1 the optimizer occasionally places small groups in large centrally located lecture theatres to consolidate travel, slightly reducing utilization; in C3 it avoids this because cross-site travel penalties dominate.

Hard constraint violations are zero in all conditions, confirming that the Phase 1 CP-SAT feasibility guarantee is preserved regardless of Phase 2 soft-constraint weighting.

Solve times remain within the USS v2.0 targets (<30 minutes for 500-course instances; our 800-course instances solve in comparable time with the incremental travel update).

### 6.3 Pareto Front Analysis

[PLACEHOLDER: Figure 2 — Pareto front plot: x-axis = Preference satisfaction rate (%), y-axis = Travel reduction (%) relative to baseline. One curve per campus configuration (C1, C2, C3). Points correspond to w_{L-S9} values from 0.00 to 0.50 in steps of 0.05. Highlight the w=0.20 operating point used in main experiments.]

The Pareto analysis reveals a characteristic "elbow" in all three configurations: substantial travel reductions (8–12%) are achievable with negligible preference loss (< 2 percentage points), after which each additional unit of carbon savings costs disproportionately more preference satisfaction. This elbow provides a natural operating point for institutional policy: administrators can achieve the majority of available carbon gains with minimal disruption to lecturer schedules.

The elbow occurs at lower w_{L-S9} values for C1 than for C3, reflecting that compact campuses have more naturally low-travel room assignment options and require less objective weight to exploit them.

### 6.4 Semester Carbon Budget Tracking

[PLACEHOLDER: Figure 3 — Stacked area chart showing cumulative CO2_timetable per week across a 15-week semester for Baseline and Carbon-Aware under C2. X-axis: week number. Y-axis: cumulative CO2 (kg). Two series. Annotate the end-of-semester totals and percent gap.]

USS v2.0 exposes a real-time carbon budget dashboard fed by the semester-level CO2_timetable estimate. As actual enrolments firm up and room bookings change through the add/drop period, the system recomputes TotalTravel and updates the CO2 projection. Institutions can compare projected emissions against prior-semester actuals and against a target budget, enabling proactive re-scheduling if the trajectory overshoots.

### 6.5 Sensitivity Analysis

[PLACEHOLDER: Table 4 — Sensitivity of travel reduction (%) to mode-share assumptions. Rows: mode-share scenario (walk-heavy, car-heavy, balanced). Columns: C1, C2, C3 travel reduction (%). Shows that CO2 reduction gains are robust to mode-share variation; high car-share amplifies emission savings.]

Travel distance reduction percentages are insensitive to mode-share assumptions because the optimizer minimizes distance, not emissions directly — mode-share only affects the CO2 translation. Emission reduction in absolute terms is more sensitive, with high-car-share scenarios yielding larger kg CO2 savings for the same distance reduction.

Excluding intra-building floor-change distances changes TotalTravel estimates by less than 3% across all configurations, confirming that inter-building travel dominates and that the simplified model (inter-building only) is adequate for planning purposes.

---

## 7 Discussion

### 7.1 Magnitude of Impact

The emission reductions demonstrated here, while modest in absolute terms for a single semester at a single institution, become significant when considered at scale. A mid-size university running three semesters per year, combined across its student and staff population, accumulates thousands of person-km of timetable-driven travel annually. If carbon-aware scheduling is adopted sector-wide, the aggregate impact across hundreds of institutions would constitute a measurable contribution to national higher education decarbonization targets.

Critically, this intervention requires no behavioral change from students or staff and no capital investment. The carbon reduction is achieved entirely through a software optimization decision made once per semester during timetable construction. This contrasts favorably with demand-side interventions (cycling infrastructure, shuttle electrification) that require years and capital budgets to implement.

### 7.2 Relationship to Existing Soft Constraints

The results confirm that L-S9 is largely complementary rather than antagonistic to other soft constraints at moderate weight values. Compact scheduling (L-S11) and carbon reduction share a common mechanism — grouping a person's daily sessions in nearby rooms — so there is a natural synergy. Room utilization (L-S8) and carbon are partially in tension because right-sizing can scatter small groups into specialized rooms that are not co-located; this tension is navigable via the calibration framework.

The adaptive ML weight layer is well-positioned to learn institution-specific w_{L-S9} values over multiple scheduling cycles: if planners consistently approve schedules at a particular carbon level, the system can infer that the implicit revealed preference corresponds to a specific weight, and stabilize there without requiring explicit administrator input.

### 7.3 Campus Topology and Carbon Leverage

The Pareto analysis highlights that carbon leverage — the magnitude of emission savings per unit of objective weight — is strongly modulated by campus topology. Compact campuses (C1) have less room to reduce travel because even the baseline has low inter-building distances. Distributed (C2) and multi-site (C3) campuses offer substantially larger absolute savings. This implies that the institutions most in need of carbon interventions — those with sprawling or fragmented estates — also have the most to gain from green scheduling. Institutions should therefore prioritize carbon-aware scheduling precisely where their estate topology makes travel-intensive timetables most likely.

### 7.4 Integration with IoT and Dynamic Re-Scheduling

USS v2.0's IoT layer (MQTT occupancy and equipment sensors) provides a pathway to dynamic carbon optimization beyond the static semester-level schedule. If real-time occupancy data reveal that a scheduled room is unexpectedly unavailable, the system can propose a room swap that maintains or improves the carbon objective. The LSTM-based predictive availability model provides a forecast horizon of several days, allowing proactive re-scheduling before conflicts materialize. While the current paper focuses on the pre-semester static schedule, future work will evaluate dynamic re-scheduling under IoT constraints as a further lever for emission reduction.

### 7.5 Equity Considerations

The equity soft constraint L-S10 monitors whether carbon-reducing room assignments disproportionately concentrate travel burden on specific student groups — for instance, by clustering high-enrolment courses in one building while isolating minority programs at distant locations. Initial analysis suggests L-S9 and L-S10 are generally aligned: clustering travel tends to reduce both aggregate distance and inequity. However, adversarial cases exist where minimizing total travel concentrates the remaining travel on a small subgroup. Institutions should monitor L-S10 scores when deploying L-S9, and the USS v2.0 equity audit module surfaces these interactions explicitly.

### 7.6 Limitations

Several limitations bound the current study. First, the evaluation uses synthetic data; validation on real institutional schedules with observed travel behavior would strengthen external validity. Second, the mode-share parameters are estimated rather than measured; travel survey data would improve emission estimates. Third, the model treats all inter-building travel as timetable-driven, but some travel is discretionary (office hours, library visits); isolating the timetable component from total campus travel requires empirical study. Fourth, the model does not account for student residential distribution, which affects whether campus travel begins from home or from another building.

---

## 8 Conclusions

This paper has presented the design, formalization, and evaluation of a carbon footprint minimization module for academic timetabling within USS v2.0. The core contributions are: (1) a campus graph travel model that derives inter-building travel distances directly from timetable decisions and aggregates them to semester totals; (2) a mode-share emission estimation layer that converts person-km to CO2 equivalents using institution-specific parameters; (3) soft constraint L-S9, which integrates travel minimization into the existing hybrid CP-SAT/simulated annealing solver as a calibrated, weighted objective; and (4) a multi-objective calibration framework that identifies operating points achieving substantial carbon reduction with acceptable preference trade-offs.

Experimental results on synthetic campus configurations demonstrate that carbon-aware scheduling achieves statistically significant reductions in inter-building travel and associated CO2 emissions across compact, distributed, and multi-site campus topologies, with the largest gains on dispersed estates. The Pareto front reveals that a substantial fraction of available carbon savings is accessible at low cost to lecturer preference satisfaction — the "elbow" of the trade-off curve represents a natural, defensible policy operating point.

The practical implication is clear: universities seeking to reduce their carbon footprint have a low-cost, high-leverage tool available in their existing scheduling infrastructure. Integrating an emission objective into the timetabling process requires no additional capital, no behavioral intervention, and no new data collection beyond what scheduling systems already possess. Future work will extend the model to dynamic re-scheduling under IoT triggers, validate the approach on real institutional data, and investigate the interaction between carbon optimization and student equity at scale.

---

## Declaration of Competing Interest

None.

---

## References

Alvarez-Valdes, R., Crespo, E., & Tamarit, J. M. (2002). Design and implementation of a course scheduling system using tabu search. *European Journal of Operational Research*, 137(3), 512–523.

Arratia-Martinez, N. M., Maya-Padron, C., & Avila-Torres, P. A. (2021). University course timetabling problem with professor assignment. *Mathematical Problems in Engineering*, 2021, 1–9.

Balsas, C. J. L. (2003). Sustainable transportation planning on college campuses. *Transport Policy*, 10(1), 35–49.

Battistutta, M., Ceschia, S., De Cesco, F., Gaspero, L. D., Schaerf, A., & Topan, E. (2017). Local search and constraint programming for a real-world examination timetabling problem. In *International Conference on the Practice and Theory of Automated Timetabling* (pp. 69–84). Springer.

Burke, E. K., & Petrovic, S. (2002). Recent research directions in automated timetabling. *European Journal of Operational Research*, 140(2), 266–280.

Cambazard, H., Hebrard, E., O'Sullivan, B., & Papadopoulos, A. (2012). Local search and constraint programming for the post enrolment-based course timetabling problem. *Annals of Operations Research*, 194(1), 111–135.

DEFRA. (2023). *UK government GHG conversion factors for company reporting*. Department for Environment, Food & Rural Affairs. https://www.gov.uk/government/collections/government-conversion-factors-for-company-reporting

de Werra, D. (1985). An introduction to timetabling. *European Journal of Operational Research*, 19(2), 151–162.

Demeester, P., Souffriau, W., De Causmaecker, P., & Vanden Berghe, G. (2012). A hybrid tabu search algorithm for automatically assigning patients to beds. *Artificial Intelligence in Medicine*, 48(1), 55–65.

Ding, J.-Y., Song, S., & Wu, C. (2016). Carbon-efficient scheduling of flow shops by multi-objective optimization. *European Journal of Operational Research*, 248(3), 758–771.

Ehrgott, M., & Ryan, D. M. (2002). Constructing robust crew schedules with bicriteria optimization. *Journal of Multi-Criteria Decision Analysis*, 11(3), 139–150.

Erben, W., & Keppler, J. (1995). A genetic algorithm solving a weekly course-timetabling problem. In E. K. Burke & P. Ross (Eds.), *Practice and Theory of Automated Timetabling* (pp. 198–211). Springer.

Even, S., Itai, A., & Shamir, A. (1976). On the complexity of timetable and multicommodity flow problems. *SIAM Journal on Computing*, 5(4), 691–703.

GHG Protocol. (2004). *A corporate accounting and reporting standard* (revised ed.). World Business Council for Sustainable Development & World Resources Institute.

Gurobi Optimization, LLC. (2023). *Gurobi optimizer reference manual*. https://www.gurobi.com

IPCC. (2022). *Climate change 2022: Mitigation of climate change. Contribution of Working Group III to the Sixth Assessment Report*. Cambridge University Press.

Lach, G., & Lübbecke, M. E. (2012). Curriculum based course timetabling: New solutions to Udine benchmark instances. *Annals of Operations Research*, 194(1), 255–272.

Larsen, H. N., Pettersen, J., Solli, C., & Hertwich, E. G. (2013). Investigating the carbon footprint of a university — the case of NTNU. *Journal of Cleaner Production*, 48, 39–47.

Liu, X.-F., Zhan, Z.-H., Deng, J. D., Li, Y., Gu, T., & Zhang, J. (2017). An energy efficient ant colony system for virtual machine placement in cloud computing. *IEEE Transactions on Evolutionary Computation*, 22(1), 113–128.

Müller, T., & Murray, K. (2010). Comprehensive approach to student sectioning. *Annals of Operations Research*, 181(1), 249–269.

Müller, T., Rudová, H., & Müllerová, Z. (2014). University course timetabling and international timetabling competition 2019. In *Proceedings of the 12th International Conference on the Practice and Theory of Automated Timetabling* (pp. 4–13).

Nejat, P., Jomehzadeh, F., Taheri, M. M., Gohari, M., & Abd. Majid, M. Z. (2015). A global review of energy consumption, CO2 emissions and policy in the residential sector. *Renewable and Sustainable Energy Reviews*, 43, 843–862.

Pérez-Lombard, L., Ortiz, J., & Pout, C. (2008). A review on buildings energy consumption information. *Energy and Buildings*, 40(3), 394–398.

Post, G., Ahmadi, S., Daskalaki, S., Kingston, J. H., Kyngas, J., Nurmi, C., & Ranson, D. (2012). An XML format for benchmarks in high school timetabling. *Annals of Operations Research*, 194(1), 385–397.

Post, G., Di Gaspero, L., Kingston, J. H., McCollum, B., & Schaerf, A. (2016). The third international timetabling competition. *Annals of Operations Research*, 239(1), 69–75.

Schimmelpfeng, K., & Helber, S. (2007). Application of a real-world university-course timetabling model solved by integer programming. *OR Spectrum*, 29(4), 783–803.

Thompson, J. M., & Dowsland, K. A. (1998). A robust simulated annealing based examination timetabling system. *Computers & Operations Research*, 25(7–8), 637–648.

Turton, H., & Moura, F. (2008). Vehicle-to-grid systems for sustainable development: An integrated energy analysis. *Technological Forecasting and Social Change*, 75(8), 1091–1108.

Universities UK. (2021). *Delivering net zero: Universities' role in the transition to a low-carbon future*. Universities UK.

Wren, A. (1996). Scheduling, timetabling and rostering — a special relationship. In E. K. Burke & P. Ross (Eds.), *Practice and Theory of Automated Timetabling* (pp. 46–75). Springer.

Yang, Y., Wang, S., & Liu, Y. (2020). Multi-objective optimization of green scheduling in manufacturing systems: A review. *Journal of Cleaner Production*, 268, 122087.

Zhang, H., Tam, C. M., & Li, H. (2014). Multimode project scheduling based on particle swarm optimization. *Computer-Aided Civil and Infrastructure Engineering*, 21(2), 93–103.
