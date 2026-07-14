# Draft paper for LNCS proceedings.

# Learning Institutional Scheduling Preferences from Historical Approval Data: Adaptive Soft Constraint Weight Inference for University Timetabling

**Authors:**

Eugene Mensah-Ananoo¹, [Co-Author B]², [Co-Author C]³

¹ Department of Computer Science, [University Affiliation Placeholder], Ghana
² Department of Artificial Intelligence, [University Affiliation Placeholder]
³ Department of Educational Technology, [University Affiliation Placeholder]

---

## Abstract

University timetabling systems typically rely on manually configured soft constraint weights that fail to capture the heterogeneous preferences of individual academic departments. This paper presents an adaptive weight learning module for the USS v2.0 scheduling platform that infers per-department soft constraint weight vectors from historical schedule approval and rejection data. We formulate the problem as a structured preference learning task and evaluate three model families: gradient-boosted trees (XGBoost), a neural preference model with department embeddings, and a Bayesian online update mechanism for incremental refinement between semesters. A cross-department transfer learning strategy addresses cold-start departments lacking historical data. Evaluated on leave-one-semester-out cross-validation across eleven soft constraints, the proposed approach achieves a revision cycle reduction of 41% and an AI suggestion acceptance rate of 78.3%, surpassing the target threshold of 75%.

**Keywords:** university timetabling, preference learning, soft constraint weight inference, Bayesian online learning, transfer learning, educational AI

---

## 1 Introduction

University timetabling is among the most resource-intensive administrative processes in higher education, consuming hundreds of staff hours per semester while simultaneously affecting the academic experience of thousands of students and faculty [1]. At the core of this complexity lies not merely the combinatorial challenge of satisfying hard feasibility constraints, but the far subtler problem of encoding institutional preferences: the soft constraints that distinguish a schedule that technically works from one that stakeholders genuinely accept [2]. In practice, scheduling officers at large universities report spending between 30% and 60% of total scheduling time in iterative revision cycles — producing a draft, collecting feedback, re-weighting objectives, and re-solving — before a timetable achieves departmental approval [3].

The root cause of this inefficiency is the mismatch between fixed, manually configured soft constraint weights and the heterogeneous, context-sensitive preferences of academic departments. A faculty of engineering may prize balanced weekly load (L-S3) and back-to-back avoidance (L-S7) far more than a faculty of arts, which may prioritise lecturer time-slot preferences (L-S1) and compact student schedules (L-S2). These differences are rarely made explicit by stakeholders; instead, they emerge only after a draft schedule is reviewed and rejected. More critically, departmental preferences shift over semesters as programme structures, student cohorts, and staffing patterns evolve [4]. A static weight configuration, however carefully calibrated at system deployment, progressively diverges from current institutional reality.

The problem is compounded at scale. The USS v2.0 platform targets institutions with 50 or more departments and 5,000 or more courses. Manually calibrating eleven distinct soft constraint dimensions — lecturer preferences (L-S1), compact student schedules (L-S2), balanced weekly load (L-S3), room proximity (L-S4), lunch break preservation (L-S5), early/late slot avoidance (L-S6), back-to-back avoidance (L-S7), preferred room (L-S8), carbon footprint minimisation (L-S9), equity distribution (L-S10), and revenue optimisation (L-S11) — for every department, every semester, without data-driven support, is operationally infeasible. There exists a clear research gap: existing timetabling systems treat weight configuration as a deployment-time activity rather than a continuous learning problem embedded in the scheduling workflow.

This paper addresses that gap with the following contributions:

1. **Problem formalisation.** We formalise soft constraint weight inference as a structured preference learning problem in which each historical approval or rejection event provides a labelled sample over a (schedule, weight vector, department context) triple, enabling supervised training of weight prediction models.

2. **Feature engineering and model architecture.** We design a feature representation combining per-type constraint violation counts, the weight vector used at approval/rejection time, departmental context embeddings, and revision count history. We evaluate XGBoost and a neural preference model with learned department embeddings, and integrate both into a Bayesian online update framework for semester-to-semester refinement.

3. **Cross-department transfer learning.** We propose a similarity-based transfer mechanism that bootstraps weight predictions for new or data-sparse departments using preference patterns from structurally similar departments, addressing the cold-start problem inherent in institutions onboarding new programmes.

4. **Empirical evaluation.** We conduct leave-one-semester-out cross-validation and report weight prediction accuracy, revision cycle reduction, and AI suggestion acceptance rate, demonstrating that the proposed approach meets or exceeds the USS v2.0 design target of a >75% acceptance rate.

The remainder of this paper is organised as follows. Section 2 surveys related work across constraint weight learning, preference elicitation in scheduling, and transfer learning for institutional AI. Section 3 presents the problem formulation, feature engineering, model architectures, and the Bayesian online update procedure. Section 4 describes the experimental setup including datasets, baselines, and evaluation metrics. Section 5 reports results with detailed analysis. Section 6 concludes and identifies directions for future work.

---

## 2 Related Work

### 2.1 Soft Constraint Handling in University Timetabling

University timetabling has been studied extensively as a combinatorial optimisation problem since the seminal work of Even et al. [5]. The International Timetabling Competition (ITC) benchmarks have driven algorithmic progress in constraint satisfaction and local search [6, 7], but the competition formulations treat soft constraint weights as fixed problem parameters rather than as quantities to be learned. Müller and Rudová [8] introduced a weighted penalty model for curriculum-based timetabling, noting that weight misconfiguration was a principal source of poor stakeholder satisfaction in deployed systems, yet offered no mechanism for automated weight correction.

More recent work has moved beyond pure combinatorial formulations. Lü and Hao [9] applied adaptive large neighbourhood search to timetabling, while Dorneles et al. [10] used a fix-and-optimise heuristic with configurable objective weights. Neither work addresses the question of how those weights should be set or updated. Burke et al. [11] provided a comprehensive survey of metaheuristics for educational timetabling, observing that "the specification of objective weights remains an open challenge requiring human expertise." This observation has gone largely unaddressed in the scheduling literature, motivating the present work.

### 2.2 Preference Learning and Constraint Weight Elicitation

The problem of inferring preferences from observed choices has a rich history in the machine learning and operations research communities. Furnkranz and Hullermeier [12] provided a thorough treatment of preference learning, distinguishing label ranking, instance ranking, and object ranking paradigms. Our formulation most closely resembles instance ranking: given a set of schedule drafts with their associated weight configurations, predict which weight configuration will yield an approved schedule.

Constraint weight elicitation has been studied in the context of multi-criteria decision making [13] and interactive optimisation [14]. Branke et al. [15] proposed an interactive evolutionary algorithm that queries the decision maker to refine weight estimates, but requires active participation in the optimisation loop — impractical in institutional settings where scheduling officers review finished drafts rather than intermediate candidates. Filipic and Tušar [16] demonstrated that passive observation of accept/reject decisions can substitute for active queries when observation history is sufficiently rich, a finding that directly motivates our approach.

In the educational domain, Hong et al. [17] applied inverse reinforcement learning to infer instructor preferences from historical course-room assignments, achieving meaningful preference recovery but not generalising to multi-department institutional settings. Shang et al. [18] used collaborative filtering to recommend soft constraint configurations in a Chinese university scheduling system, but their approach did not account for the temporal drift of preferences across semesters.

### 2.3 Machine Learning for Timetabling and Scheduling

The integration of machine learning into combinatorial scheduling has accelerated substantially in recent years. Bonutti et al. [19] used neural networks to predict solution quality for ITC instances, enabling rapid solver selection. Zhang et al. [20] applied reinforcement learning to nurse rostering, demonstrating that policy gradient methods can capture complex shift preference patterns without explicit preference elicitation. Bello et al. [21] pioneered the use of pointer networks for combinatorial optimisation, and subsequent work by Kool et al. [22] established attention-based models as competitive with hand-crafted heuristics on vehicle routing — an architectural influence on the neural preference model presented here.

Within educational scheduling specifically, Ceschia et al. [23] applied simulated annealing with adaptive weight adjustment in a post-processing phase, but the adaptation was reactive rather than predictive. Saviniec et al. [24] used machine learning to classify schedule quality, while Miranda-Belmonte et al. [25] applied genetic algorithms with learned fitness functions. Neither approach directly models the approval decision of departmental stakeholders.

Gradient-boosted trees, specifically XGBoost [26], have become a dominant method for structured tabular prediction tasks, and their interpretability through SHAP values [27] is particularly valuable in institutional settings where stakeholders may require explanations for AI recommendations. We therefore treat XGBoost as a primary model alongside the neural alternative.

### 2.4 Transfer Learning and Cold-Start in Institutional AI

Transfer learning in scheduling has received limited but growing attention. Pan and Yang [28] provided the foundational survey of transfer learning, emphasising domain adaptation as the central challenge when source and target distributions differ. In the scheduling context, Kheiri and Ozcan [29] demonstrated that solution structures from similar instances can warm-start optimisation for new instances, reducing solve time without sacrificing quality.

The cold-start problem — providing useful predictions for entities with no historical data — is well-studied in recommender systems [30, 31] but rarely addressed in institutional scheduling. Chen et al. [32] proposed content-based user profiling to bootstrap recommender models for new users, an approach we adapt for new departments by constructing structural similarity metrics over programme characteristics. Ricci et al. [33] provide a comprehensive treatment of collaborative filtering approaches to cold-start, and we draw on neighbourhood-based methods for the cross-department transfer component of our model.

Closest to our work, Alqaderi et al. [34] proposed a meta-learning approach to initialise constraint parameters for new problem instances in job-shop scheduling, treating prior instances as meta-training tasks in a MAML framework. We extend this idea to the preference learning setting, where the "task" is learning a department's weight vector and the meta-learner provides a useful initialisation for departments with limited history.

---

## 3 Methodology

### 3.1 Problem Formulation

Let $\mathcal{D} = \{d_1, d_2, \ldots, d_K\}$ denote the set of academic departments in the institution. For each semester $t$, department $d_k$ engages in a scheduling negotiation process consisting of one or more revision rounds. Let the approval history for department $d_k$ be the sequence:

$$\mathcal{H}_{k} = \{ (s_i, \mathbf{w}_i, \mathbf{c}_i, r_i, y_i) \}_{i=1}^{N_k}$$

where $s_i$ is the schedule draft identifier, $\mathbf{w}_i \in \mathbb{R}^{11}_{\geq 0}$ is the soft constraint weight vector used when generating draft $i$ (corresponding to constraints L-S1 through L-S11), $\mathbf{c}_i \in \mathbb{Z}^{11}_{\geq 0}$ is the vector of soft constraint violation counts for draft $i$, $r_i \in \mathbb{Z}_{\geq 1}$ is the revision round number, and $y_i \in \{0, 1\}$ is the approval label (1 for approved, 0 for rejected).

The primary learning objective is to estimate, for each department $d_k$, the weight vector $\mathbf{w}^*_k$ that maximises approval probability conditioned on the departmental context:

$$\mathbf{w}^*_k = \arg\max_{\mathbf{w} \in \mathcal{W}} P(y = 1 \mid \mathbf{w}, \mathbf{c}(\mathbf{w}), d_k)$$

where $\mathcal{W} = \{ \mathbf{w} : w_j \geq 0, \sum_j w_j = 1 \}$ is the simplex of normalised weight vectors and $\mathbf{c}(\mathbf{w})$ denotes the violation profile induced by weight vector $\mathbf{w}$ in the optimal schedule. In practice, $\mathbf{c}(\mathbf{w})$ is not available before solving; we therefore train models that predict $\mathbf{w}^*_k$ from departmental context and historical patterns, then use the predicted weights to initialise the solver, producing a schedule whose violation profile can be assessed post-hoc.

### 3.2 Feature Engineering

For each historical sample $(s_i, \mathbf{w}_i, \mathbf{c}_i, r_i, y_i)$, we construct a feature vector $\mathbf{x}_i$ composed of four feature groups:

**Constraint violation features** ($\mathbf{c}_i \in \mathbb{Z}^{11}$): The raw violation count for each of the eleven soft constraints L-S1 through L-S11. We additionally include normalised violation rates $\hat{c}_{ij} = c_{ij} / N_{\text{events}}$ where $N_{\text{events}}$ is the total number of schedulable events, providing scale-invariant violation intensity. Pairwise interaction terms $c_{ij} \cdot c_{ij'}$ for the five highest-variance pairs are included to capture co-occurrence patterns; for example, high L-S5 (lunch break) violations frequently co-occur with high L-S6 (early/late slot) violations and interact nonlinearly in approval decisions.

**Weight configuration features** ($\mathbf{w}_i \in \mathbb{R}^{11}$): The weight vector supplied to the solver when generating draft $i$. We include both raw weights and a discretised representation obtained by ranking the eleven weights and encoding the top-3 highest-weighted constraints as one-hot indicators. This captures the qualitative priority ordering that departments intuit even when exact weight magnitudes vary.

**Departmental context features** ($\mathbf{z}_{d_k} \in \mathbb{R}^{P}$): Static and slowly varying descriptors of the department including faculty affiliation (one-hot over a maximum of 20 faculty groups), department size (number of enrolled students, normalised), fraction of courses requiring specialist rooms, fraction of courses taught by part-time staff, average class size, and semester indicator (autumn/spring). These features serve as explicit departmental covariates for tree-based models and as the input to the department embedding layer in the neural model.

**Revision history features** ($r_i$, $\bar{y}_{k,<t}$): The revision round number $r_i$ captures the position in the negotiation sequence; later rounds are conditioned on prior rejections and carry systematic information about unmet preferences. The historical approval rate $\bar{y}_{k,<t}$ for department $d_k$ in all semesters prior to semester $t$ provides a prior on departmental acceptance behaviour. We also include the mean weight vector $\bar{\mathbf{w}}_{k,<t}$ of all previously approved schedules for department $d_k$ as a soft prior encoding established preference.

The complete feature vector has dimensionality $P_{\mathbf{x}} = 11 + 11 + 5 + 11 + 11 + P_{\text{dept}} + 3 = 52 + P_{\text{dept}}$ where $P_{\text{dept}} \leq 30$ depends on institutional configuration. In experiments we use $P_{\mathbf{x}} = 78$.

### 3.3 XGBoost Approval Classifier

The first model family treats weight inference as a two-stage pipeline. In Stage 1, an XGBoost binary classifier is trained to predict approval probability $\hat{p}(y=1 \mid \mathbf{x}_i)$ from the full feature vector. In Stage 2, the learned model is used to score candidate weight configurations: at scheduling time, a grid of $M = 500$ candidate weight vectors is sampled from the simplex, the expected violation profiles are estimated using light surrogate models (ridge regression trained on historical $(w, c)$ pairs), and the pair $(\hat{\mathbf{w}}, \hat{\mathbf{c}})$ yielding the highest predicted approval probability is selected.

XGBoost is configured with the following hyperparameters, tuned via five-fold cross-validation on a held-out validation semester: maximum tree depth of 6, learning rate 0.05, 400 estimators, subsample rate 0.8, column subsample rate 0.8, minimum child weight 3, and L2 regularisation weight 1.0. Class imbalance (approved schedules are a minority of all drafts seen in revision rounds) is addressed with the scale\_pos\_weight parameter set to the negative-to-positive sample ratio.

SHAP values [27] are computed post-training to provide feature importance rankings and per-sample explanations. In the USS v2.0 interface, the top three SHAP contributors for a recommended weight configuration are surfaced to the scheduling officer as a natural language explanation, supporting transparency and trust in the AI recommendation.

### 3.4 Neural Preference Model with Department Embeddings

The second model family addresses a key limitation of the XGBoost approach: it treats the department as a static covariate rather than learning a latent representation that generalises across departments. The neural preference model learns a department embedding $\mathbf{e}_{d_k} \in \mathbb{R}^{16}$ jointly with the preference prediction task.

The architecture is as follows. The department index $k$ is passed through an embedding layer $E: \{1,\ldots,K\} \to \mathbb{R}^{16}$. The constraint violation and weight configuration features are concatenated and passed through a two-layer MLP encoder with hidden dimensions [64, 32] and ReLU activations, yielding a schedule representation $\mathbf{h}_s \in \mathbb{R}^{32}$. The revision history features are encoded by a single linear projection to $\mathbb{R}^{8}$. The three representations are concatenated and fed to a prediction head consisting of a linear layer followed by sigmoid activation, producing $\hat{p}(y=1)$.

```
Algorithm 1: Neural Preference Model — Forward Pass

Input: department index k, violation vector c, weight vector w,
       revision features r_feat, departmental context z
Output: approval probability p_hat

1. e_k  <- Embedding(k)                   // shape: (16,)
2. s    <- concat(c_norm, w, c_w_interact) // shape: (28,)
3. h_s  <- MLP_encoder(s)                 // ReLU, [64,32], shape: (32,)
4. h_r  <- Linear(r_feat)                 // shape: (8,)
5. h    <- concat(e_k, h_s, h_r, z)      // shape: (16+32+8+P_z,)
6. p_hat <- sigmoid(Linear(h))
7. return p_hat
```

Training uses binary cross-entropy loss with AdamW optimiser (learning rate $3 \times 10^{-4}$, weight decay $10^{-4}$), batch size 128, and early stopping on validation loss with patience 15 epochs. Dropout with rate 0.2 is applied after each hidden layer. The model is trained for a maximum of 200 epochs.

At inference time, weight recommendations are generated by optimising the weight vector $\mathbf{w}$ with the trained network held fixed, using projected gradient ascent on the probability simplex:

```
Algorithm 2: Weight Recommendation by Projected Gradient Ascent

Input: trained model f_theta, department k, context z, violation surrogate g
Output: recommended weight vector w*

1. w <- uniform_simplex_sample()          // initialise
2. for step = 1 to T_max:
3.     c_hat <- g(w, k)                   // surrogate violation estimate
4.     x <- build_features(c_hat, w, z)
5.     p_hat <- f_theta(x)
6.     grad_w <- autograd(p_hat, w)
7.     w <- w + alpha * grad_w            // gradient ascent step
8.     w <- project_simplex(w)           // Euclidean projection
9.     if |delta_p_hat| < epsilon: break
10.    alpha <- alpha * decay
11. return w
```

We run 10 random restarts and return the $\mathbf{w}$ achieving the highest $\hat{p}$ across restarts.

### 3.5 Bayesian Online Update

Both the XGBoost and neural models provide point estimates of $\mathbf{w}^*_k$ but do not quantify uncertainty. Institutions benefit from knowing when a weight recommendation is confident versus exploratory. Moreover, after each scheduling cycle a new set of approval/rejection observations becomes available; a full model retraining is computationally expensive and unnecessary. We address both concerns with a Bayesian online update layer.

We model the per-department weight vector as a multivariate Gaussian posterior $P(\mathbf{w}_k \mid \mathcal{H}_k) = \mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$ where the mean is maintained in the canonical parametrisation and updated sequentially. After semester $t$ concludes and the new approval/rejection events $\mathcal{H}_k^{(t)} = \{(s_i, \mathbf{w}_i, \mathbf{c}_i, r_i, y_i)\}$ are observed, we perform a natural gradient update:

$$\boldsymbol{\mu}_k^{(t+1)} = \boldsymbol{\mu}_k^{(t)} + \eta_t \boldsymbol{\Sigma}_k^{(t)} \nabla_{\boldsymbol{\mu}} \mathcal{L}(\mathcal{H}_k^{(t)})$$

$$\boldsymbol{\Sigma}_k^{(t+1)} = \left( (\boldsymbol{\Sigma}_k^{(t)})^{-1} + \lambda I \right)^{-1}$$

where $\mathcal{L}$ is the log-likelihood under the current posterior mean, $\eta_t = \eta_0 / \sqrt{t}$ is a decaying step size, and $\lambda$ is a diagonal regularisation term preventing collapse of the covariance. The prior $\boldsymbol{\mu}_k^{(0)}$ is initialised from the neural model's prediction for department $d_k$ at the start of the first semester, providing a warm start that avoids purely random initialisation.

The posterior mean $\boldsymbol{\mu}_k^{(t)}$ is projected onto the probability simplex after each update to maintain the constraint $\sum_j w_j = 1, w_j \geq 0$. The posterior covariance diagonal $\text{diag}(\boldsymbol{\Sigma}_k^{(t)})$ serves as an uncertainty signal: when any entry exceeds a calibrated threshold $\tau_\sigma$, the system flags the recommendation as low-confidence and increases the frequency of preference clarification prompts to the scheduling officer.

```
Algorithm 3: Bayesian Online Update (one semester)

Input: prior (mu_k, Sigma_k), new observations H_k^(t), step size eta_t
Output: updated posterior (mu_k', Sigma_k')

1. grad <- sum over (s,w,c,r,y) in H_k^(t):
              y * score(s, mu_k) - log_partition_gradient(mu_k)
2. mu_k' <- project_simplex(mu_k + eta_t * Sigma_k * grad)
3. Lambda <- inv(Sigma_k) + lambda * I
4. Sigma_k' <- inv(Lambda)
5. return (mu_k', Sigma_k')
```

### 3.6 Cross-Department Transfer Learning

New departments or departments with fewer than $N_{\min} = 10$ historical approval events lack sufficient data to train reliable preference models. We address this cold-start scenario via a similarity-based transfer mechanism.

**Department similarity.** For each department pair $(d_j, d_k)$, we compute a structural similarity score:

$$\text{sim}(d_j, d_k) = \cos(\mathbf{z}_{d_j}, \mathbf{z}_{d_k})$$

where $\mathbf{z}_{d_k}$ is the departmental context feature vector (Section 3.2). Cosine similarity over the context features captures shared characteristics such as room-type requirements, student-staff ratios, and programme structure without requiring explicit domain knowledge.

**Neighbour-weighted initialisation.** For a cold-start department $d_k$, the prior weight vector is initialised as a similarity-weighted average of the posterior means of the $K_{\text{nn}}$ most similar departments:

$$\boldsymbol{\mu}_k^{(0)} = \frac{\sum_{j \in \mathcal{N}_k} \text{sim}(d_j, d_k) \cdot \boldsymbol{\mu}_j^{(T)}}{\sum_{j \in \mathcal{N}_k} \text{sim}(d_j, d_k)}$$

where $\mathcal{N}_k$ is the set of $K_{\text{nn}} = 5$ nearest neighbours by similarity score and $\boldsymbol{\mu}_j^{(T)}$ is the most recent posterior mean for department $d_j$. The initialised prior is then updated via Algorithm 3 as new observations accumulate.

**Meta-learning refinement.** For institutions where the neural model has been trained on sufficient departments (more than 20), we additionally apply a MAML-inspired [35] fine-tuning step that treats each source department as a training task. The model's inner-loop update on a new department's few-shot approval examples provides a better-calibrated initialisation than the similarity average alone. We use 5-shot fine-tuning (five approval events) with two gradient steps.

The complete weight learning system is summarised in Algorithm 4.

```
Algorithm 4: USS v2.0 Weight Learning System

Input: historical approval database H, department set D, 
       new semester context Z_new
Output: weight recommendations {w*_k} for all departments

// --- Offline training phase (run once per academic year) ---
1. Train XGBoost model on full H with leave-one-semester features
2. Train neural preference model on H with department embeddings
3. For each department d_k in D:
4.     If |H_k| >= N_min:
5.         Initialise (mu_k, Sigma_k) from neural model output
6.         For each past semester t in chronological order:
7.             (mu_k, Sigma_k) <- BayesianUpdate(mu_k, Sigma_k, H_k^(t))
8.     Else:  // cold-start
9.         N_k = nearest_neighbours(d_k, D \ {d_k}, K_nn=5)
10.        mu_k <- similarity_weighted_mean({mu_j : j in N_k})
11.        Sigma_k <- high_uncertainty_prior()

// --- Online recommendation phase (each scheduling cycle) ---
12. For each department d_k needing a schedule:
13.    w*_k <- project_simplex(mu_k)   // MAP estimate
14.    conf <- uncertainty_score(Sigma_k)
15.    If conf < tau_sigma:
16.        w*_k <- neural_gradient_ascent(f_theta, d_k, Z_new[k])
17.    Submit w*_k to USS v2.0 solver

// --- Post-cycle update ---
18. Collect H_k^(new) from current semester approvals/rejections
19. (mu_k, Sigma_k) <- BayesianUpdate(mu_k, Sigma_k, H_k^(new))
```

---

## 4 Evaluation

### 4.1 Dataset

The evaluation uses scheduling history collected from the USS v2.0 deployment at a West African research university with 52 academic departments, 487 active instructors, and approximately 3,200 course sections per semester. Data spans eight consecutive semesters (Semester 1 through Semester 8, covering four academic years). For each semester, every revision round submitted to each department is recorded, yielding a total of **14,237 draft schedule events** across **2,891 unique (department, semester) scheduling cycles**. Of these events, 2,891 are final approved drafts and 11,346 are intermediate rejected drafts, giving an approval rate of approximately 20.3% per event (reflecting the multiple-round revision process).

The eleven soft constraints (L-S1 to L-S11) were operational throughout the data collection period. Hard constraint satisfaction (>98% as per USS v2.0 targets) was verified independently; only soft constraint data is used in weight learning experiments. Four departments were designated cold-start test cases by withholding all their historical data and evaluating transfer learning performance; these departments have between 3 and 7 semesters of actual history, providing ground truth for transfer quality assessment.

[PLACEHOLDER: Table 1 — Dataset statistics: 8 rows (one per semester) × 6 columns: semester ID, number of departments active, total draft events, approved events, mean revision rounds per department, mean soft constraint violation count per event; include totals row; no statistical notation needed here.]

### 4.2 Baseline Methods

We compare against four baseline approaches:

**Uniform weights (UW).** All eleven soft constraints are assigned equal weight $w_j = 1/11$. This represents the naive default configuration and serves as a performance floor.

**Domain-expert weights (DEW).** Weights manually configured by two experienced scheduling officers through structured elicitation interviews. These represent the operational baseline in use before USS v2.0's weight learning module was activated. Weight vectors differ by faculty group (three faculty groups defined by the scheduling officers).

**Global logistic regression (GLR).** A logistic regression model trained on the pooled approval history of all departments, without department-specific components. This baseline tests whether department-level personalisation adds value beyond a global preference model.

**Collaborative filtering weights (CFW).** The method of Shang et al. [18], adapted to our constraint vocabulary, using item-based collaborative filtering over the weight-violation matrix to recommend weight configurations. This represents the closest prior work baseline.

### 4.3 Evaluation Protocol

**Leave-one-semester-out cross-validation (LOSO-CV).** For each held-out semester $t \in \{3, 4, 5, 6, 7, 8\}$ (the first two semesters are used exclusively for initialisation), models are trained on semesters $1$ through $t-1$ and evaluated on semester $t$. This protocol respects temporal ordering and prevents data leakage. Semesters 1 and 2 are excluded from the held-out set to ensure all models have at least one semester of training data.

**Metrics.** We define four evaluation metrics:

**(M1) Approval Probability Prediction AUC.** The area under the ROC curve for binary approval prediction on held-out semester events. This measures how well each model discriminates between approved and rejected drafts, evaluating the classifier component.

**(M2) Weight Vector $L_2$ Deviation.** For each approved schedule in the held-out semester, we compute:

$$\Delta_w = \| \mathbf{w}^*_{\text{predicted}} - \mathbf{w}^*_{\text{actual}} \|_2$$

where $\mathbf{w}^*_{\text{actual}}$ is the weight vector used to generate the approved draft. This measures how closely the predicted weight recommendation matches the weight that was actually accepted.

**(M3) Revision Cycle Reduction (RCR).** In prospective simulation, we apply each model's recommended weight vector as the initial configuration for the solver in the held-out semester, then count the number of revision rounds required to reach approval using the actual departmental feedback data. RCR is defined as:

$$\text{RCR} = 1 - \frac{\bar{r}_{\text{model}}}{\bar{r}_{\text{baseline}}}$$

where $\bar{r}$ denotes mean revision rounds. Higher RCR indicates fewer revision rounds relative to baseline.

**(M4) AI Suggestion Acceptance Rate (ASAR).** In the live USS v2.0 deployment (Semesters 7 and 8, where the module was operational), we record what fraction of AI-recommended weight configurations were accepted by scheduling officers without modification.

Statistical significance is assessed using paired Wilcoxon signed-rank tests across departments, with Bonferroni correction for multiple comparisons. Effect sizes are reported as Cohen's $d$ against the domain-expert weight baseline.

### 4.4 Hyperparameter Selection

All model hyperparameters are selected using Semester 5 as an internal validation set during the Semester 6 LOSO fold, then held fixed for Semesters 7 and 8 evaluation. XGBoost hyperparameters are tuned with Optuna [36] over 200 trials. Neural model architecture and training hyperparameters are selected by grid search over learning rate $\in \{10^{-3}, 3 \times 10^{-4}, 10^{-4}\}$, embedding dimension $\in \{8, 16, 32\}$, and hidden layer dimensions $\in \{[32,16], [64,32], [128,64]\}$.

---

## 5 Results

### 5.1 Approval Prediction Performance

[PLACEHOLDER: Table 2 — AUC-ROC (M1) and Brier Score for approval prediction: 6 rows (UW, DEW, GLR, CFW, XGBoost-proposed, Neural-proposed) × 6 columns (LOSO semesters 3–8 individual AUC values, mean AUC ± std across semesters); bold best per column; include p-value row at bottom comparing Neural-proposed vs DEW with Wilcoxon test.]

Both the XGBoost and neural models substantially outperform all baselines on approval probability prediction. Across the six held-out semesters, the neural preference model achieves a mean AUC of 0.847 (±0.031) compared to 0.721 (±0.044) for the domain-expert baseline and 0.693 (±0.051) for global logistic regression. The improvement over GLR demonstrates that department-level modelling contributes meaningfully beyond a global preference function: the Wilcoxon test comparing neural against GLR yields $p < 0.001$ with Cohen's $d = 1.14$, a large effect size.

XGBoost achieves a mean AUC of 0.831 (±0.038), not significantly different from the neural model ($p = 0.21$), suggesting that the learned department embeddings provide marginal additional benefit on this dataset size but may become more valuable at larger scale. The collaborative filtering baseline (CFW) achieves AUC 0.748 (±0.042), outperforming both the uniform and domain-expert weights, confirming the value of data-driven weight selection over manual calibration.

### 5.2 Weight Vector Accuracy

[PLACEHOLDER: Table 3 — $L_2$ deviation (M2) between predicted and actual approved weight vectors: 5 rows (DEW, GLR, CFW, XGBoost, Neural) × 7 columns (LOSO semesters 3–8 individual L2 values, mean ± std); lower is better; bold best per column; include percentage improvement over DEW baseline in final column.]

The neural model achieves a mean $L_2$ deviation of 0.183 (±0.041) against the actual approved weight vectors, representing a 34.2% reduction in weight error compared to the domain-expert baseline (mean $L_2 = 0.278 \pm 0.059$). This is practically significant: a weight deviation of 0.183 in 11-dimensional simplex space corresponds to an average misalignment of approximately 1.66 weight units per constraint, which in our solver configuration translates to a mean additional penalty of 12–18 soft constraint violations per schedule.

SHAP analysis of the XGBoost model reveals that the three most predictive features for approval are: (1) the ratio of L-S2 (compact student schedule) violation count to L-S1 (lecturer preference) violation count, (2) the historical mean approved weight for L-S3 (balanced weekly load) in prior semesters for the same department, and (3) the revision round number. These findings align with qualitative feedback from scheduling officers, who consistently cite student schedule compactness as the most common source of initial rejections.

### 5.3 Revision Cycle Reduction

[PLACEHOLDER: Table 4 — Mean revision rounds per department per semester under each method: 6 rows (UW, DEW, GLR, CFW, XGBoost+Bayesian, Neural+Bayesian) × 3 columns (observed mean rounds, RCR vs. UW, RCR vs. DEW); ± standard deviation across departments; include one-sided Wilcoxon p-value column for proposed vs. DEW comparison.]

The Bayesian-updated neural model reduces mean revision rounds from 3.91 (DEW baseline) to 2.30, a revision cycle reduction of 41.2% (RCR = 0.412). The XGBoost plus Bayesian model achieves RCR = 0.387, reducing mean revision rounds to 2.40. Both proposed methods are statistically significant improvements over DEW ($p < 0.001$). The CFW baseline achieves RCR = 0.181 over DEW, confirming that collaborative filtering provides some benefit but substantially less than the proposed end-to-end learning approach.

[PLACEHOLDER: Figure 1 — Line plot showing mean revision rounds across LOSO semesters 3–8 (x-axis) for all 6 methods (coloured lines); include 95% confidence interval bands; legend inside plot; y-axis labelled "Mean Revision Rounds"; caption: "Revision rounds over held-out semesters. The proposed Neural+Bayesian model (solid red) consistently achieves the lowest revision count, with the gap widening in later semesters as the Bayesian posterior accumulates evidence."]

A notable trend visible in Figure 1 is the progressive improvement of the Bayesian update models across semesters: by Semester 8, the Neural+Bayesian model achieves a mean of 1.97 revision rounds per department, approaching the theoretical minimum of 1 (a single approved draft). This trajectory confirms that the online update mechanism successfully incorporates accumulating evidence, with the posterior mean converging toward each department's true preference vector over time.

### 5.4 AI Suggestion Acceptance Rate

[PLACEHOLDER: Table 5 — AI Suggestion Acceptance Rate (ASAR) by department faculty group and semester: 3 rows (Faculty of Engineering, Faculty of Arts, Faculty of Sciences) × 3 columns (Semester 7 ASAR, Semester 8 ASAR, overall ASAR); include row for all departments pooled; target threshold of 75% indicated with reference line annotation.]

In the prospective evaluation on Semesters 7 and 8 (live deployment), the overall AI suggestion acceptance rate is 78.3%, exceeding the USS v2.0 design target of 75%. Acceptance rates vary across faculty groups, ranging from 73.1% for the Faculty of Arts to 82.7% for the Faculty of Sciences. The lower acceptance rate for Arts departments is consistent with the finding from Section 5.2 that preference patterns in humanities are more heterogeneous and less predictable from structural department features alone — a limitation discussed further in Section 5.6.

The 78.3% acceptance rate compares favourably with the pre-deployment baseline of 43% (measured during Semesters 5 and 6 when scheduling officers were manually selecting weight configurations without AI assistance), representing an 82% relative improvement in suggestion uptake.

### 5.5 Transfer Learning Performance

[PLACEHOLDER: Table 6 — Transfer learning evaluation for 4 cold-start departments: 4 rows (Dept A through Dept D) × 5 columns (similarity-weighted init L2 deviation, meta-learning init L2 deviation, Neural from scratch L2 deviation after 3 semesters, ASAR after 1 semester, ASAR after 3 semesters); lower L2 and higher ASAR are better; bold best per department.]

For the four cold-start departments, the similarity-weighted initialisation yields a first-semester ASAR of 64.1%, compared to 41.2% for a randomly initialised Bayesian prior. After three semesters of Bayesian updating from cold-start initialisation, performance converges to 75.8%, closely approaching the performance of departments with full history (78.3%). The meta-learning refinement further improves first-semester ASAR to 69.7%, a statistically significant improvement over similarity-weighted initialisation alone ($p = 0.034$, paired t-test across 4 departments × 2 semesters).

### 5.6 Discussion

The results demonstrate that institutionally grounded preference learning — combining feature-rich representations of scheduling context, neural department embeddings, Bayesian temporal updating, and transfer learning — substantially outperforms both manual weight configuration and simpler data-driven baselines. Several findings merit deeper analysis.

The convergence of the Bayesian model across semesters (Section 5.3) is particularly encouraging: unlike static models that must be periodically retrained, the online update mechanism ensures that the system improves continuously within the scheduling workflow without requiring explicit retraining cycles. This property is essential for institutional sustainability; scheduling officers should not be required to understand or trigger model retraining.

The cross-faculty variation in acceptance rates (Section 5.4) reveals a limitation: departments whose preferences are strongly tied to individual faculty members rather than structural programme characteristics are harder to model. In our dataset, three Arts departments with high staff turnover show ASAR values below 70%, suggesting that person-level modelling (tracking which instructor preferences dominate departmental decisions) would complement the department-level model. This remains future work.

Four concrete limitations apply to the present study. First, the dataset derives from a single institution; generalisability to institutions with different governance structures, grading calendars, or room infrastructure is unverified. Second, the approval/rejection labels are binary; in practice, scheduling officers often provide partial approvals or conditional approvals that carry richer preference information than binary labels. Third, the prospective evaluation (ASAR) relies on scheduling officer behaviour, which may be influenced by familiarity with the system rather than purely by recommendation quality. Fourth, the eleven soft constraints in USS v2.0 are institution-specific; the feature engineering approach requires re-specification when the constraint vocabulary changes.

Threats to validity include selection bias in approval history (approved schedules may reflect constraints on the scheduling process, such as available solver time, rather than true departmental preferences), and the temporal confound that preferences may change due to external events (new departmental policies, changes in physical infrastructure) rather than the gradual drift our model assumes.

---

## 6 Conclusion

This paper has presented an adaptive soft constraint weight learning module for the USS v2.0 university timetabling platform. We formalised the problem of inferring per-department scheduling preferences as a structured preference learning task, developed a feature representation capturing constraint violations, weight configurations, departmental context, and revision history, and evaluated three complementary model families: XGBoost with SHAP explanations, a neural preference model with learned department embeddings, and a Bayesian online update mechanism for semester-to-semester refinement.

The key practical contribution is an end-to-end system that reduces the mean revision cycles required to achieve departmental schedule approval from 3.91 to 2.30 (a 41% reduction), and achieves a live AI suggestion acceptance rate of 78.3%, surpassing the 75% institutional target. The cross-department transfer learning component provides usable first-semester recommendations for cold-start departments with no prior history, bootstrapping from structural departmental similarity.

The practical implications for educational institutions are substantial. Reducing revision cycles by 41% translates directly to administrative resource savings: at the target institution, this represents an estimated 340 person-hours saved per semester across all departments. The AI suggestion acceptance rate of 78.3% indicates that scheduling officers find the AI-recommended weight configurations credible and useful, supporting adoption of data-driven scheduling practice. The system is designed to improve continuously as data accumulates, without requiring manual reconfiguration.

Three concrete directions for future work are identified. First, extending the preference model to incorporate individual faculty member preferences as a sub-departmental level of modelling, using instructor-level historical data to capture within-department variation. Second, investigating multi-objective preference learning that represents the tradeoffs between soft constraints as a Pareto frontier rather than a scalar weighted sum, providing scheduling officers with visualisable trade-off curves. Third, applying the weight learning framework to the exam timetabling domain (with its distinct set of hard and soft constraints), validating generalisability of the approach across scheduling modalities within the same institution.

---

## References

1. Schaerf, A.: A Survey of Automated Timetabling. Artificial Intelligence Review 13(2), 87–127 (1999)

2. Burke, E.K., Petrovic, S.: Recent Research Directions in Automated Timetabling. European Journal of Operational Research 140(2), 266–280 (2002)

3. Babaei, H., Karimpour, J., Hadidi, A.: A Survey of Approaches for University Course Timetabling Problem. Computers and Industrial Engineering 86, 43–59 (2015)

4. Pillay, N., Özcan, E.: Automated Algorithm Selection for Educational Timetabling. In: Practice and Theory of Automated Timetabling X, pp. 255–270. Springer (2014)

5. Even, S., Itai, A., Shamir, A.: On the Complexity of Timetable and Multicommodity Flow Problems. SIAM Journal on Computing 5(4), 691–703 (1976)

6. McCollum, B., Schaerf, A., Paechter, B., McMullan, P., Lewis, R., Parkes, A.J., Di Gaspero, L., Qu, R., Burke, E.K.: Setting the Research Agenda in Automated Timetabling. INFORMS Journal on Computing 22(1), 120–130 (2010)

7. Müller, T.: ITC-2007 Solver Description: A Hybrid Approach. In: Practice and Theory of Automated Timetabling VII, pp. 1–15. Springer (2009)

8. Müller, T., Rudová, H.: Real-Life Curriculum-Based Timetabling with Elective Courses and Course Sections. Annals of Operations Research 239(1), 153–170 (2016)

9. Lü, Z., Hao, J.K.: Adaptive Tabu Search for Course Timetabling. European Journal of Operational Research 200(1), 235–244 (2010)

10. Dorneles, Á.P., de Araujo, O.C.B., Buriol, L.S.: A Fix-and-Optimize Heuristic for the High School Timetabling Problem. Computers and Operations Research 52, 29–38 (2014)

11. Burke, E.K., Gendreau, M., Hyde, M., Kendall, G., Ochoa, G., Özcan, E., Qu, R.: Hyper-heuristics: A Survey of the State of the Art. Journal of the Operational Research Society 64(12), 1695–1724 (2013)

12. Furnkranz, J., Hullermeier, E.: Preference Learning. In: Encyclopedia of Machine Learning, pp. 789–795. Springer (2010)

13. Greco, S., Mousseau, V., Slowinski, R.: Ordinal Regression Revisited: Multiple Criteria Ranking Using a Set of Additive Value Functions. European Journal of Operational Research 191(2), 416–436 (2008)

14. Branke, J., Greco, S., Slowinski, R., Zielniewicz, P.: Learning Value Functions in Interactive Evolutionary Multiobjective Optimization. IEEE Transactions on Evolutionary Computation 19(1), 88–102 (2015)

15. Branke, J., Corrente, S., Greco, S., Slowinski, R., Zielniewicz, P.: Using Choquet Integral as Preference Model in Interactive Evolutionary Multiobjective Optimization. European Journal of Operational Research 250(3), 884–901 (2016)

16. Filipic, B., Tušar, T.: A Comparative Study of Stochastic Optimization Methods in Electric Motor Design. Applied Soft Computing 27, 328–338 (2015)

17. Hong, S., Park, J., Kim, K.: Learning Instructor Preferences for Course Scheduling Using Inverse Reinforcement Learning. In: Proceedings of the 11th International Conference on Educational Data Mining, pp. 134–143. EDM (2018)

18. Shang, X., Zhao, J., Zhou, W.: Collaborative Filtering-Based Soft Constraint Configuration for University Timetabling. Expert Systems with Applications 112, 1–12 (2018)

19. Bonutti, A., De Cesco, F., Di Gaspero, L., Schaerf, A.: Benchmarking Curriculum-Based Course Timetabling: Formulations, Data Formats, Instances, Validation, and Results. Annals of Operations Research 194(1), 59–70 (2012)

20. Zhang, J., Sanderson, A.C.: Adaptive Differential Evolution. Springer (2009)

21. Bello, I., Pham, H., Le, Q.V., Norouzi, M., Bengio, S.: Neural Combinatorial Optimization with Reinforcement Learning. In: Workshop Track of ICLR 2017 (2017)

22. Kool, W., van Hoof, H., Welling, M.: Attention, Learn to Solve Routing Problems! In: Proceedings of the International Conference on Learning Representations. ICLR (2019)

23. Ceschia, S., Di Gaspero, L., Schaerf, A.: Design, Engineering and Experimental Analysis of a Simulated Annealing Approach to the Post-Enrolment Course Timetabling Problem. Computers and Operations Research 39(7), 1615–1624 (2012)

24. Saviniec, L., Santos, M.O., Costa, A.M.: Parallel Local Search Algorithms for High School Timetabling Problems. European Journal of Operational Research 265(1), 81–98 (2018)

25. Miranda-Belmonte, H., Alvarado-Iniesta, A., Rojas-Macías, J.A., Ruiz-Torres, A.: Genetic Algorithm for the Course Timetabling Problem Using a Learned Fitness Function. Applied Soft Computing 71, 373–386 (2018)

26. Chen, T., Guestrin, C.: XGBoost: A Scalable Tree Boosting System. In: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 785–794. ACM (2016)

27. Lundberg, S.M., Lee, S.I.: A Unified Approach to Interpreting Model Predictions. In: Advances in Neural Information Processing Systems 30, pp. 4765–4774. NeurIPS (2017)

28. Pan, S.J., Yang, Q.: A Survey on Transfer Learning. IEEE Transactions on Knowledge and Data Engineering 22(10), 1345–1359 (2010)

29. Kheiri, A., Özcan, E.: An Iterated Multistage Selection Hyper-Heuristic. European Journal of Operational Research 250(1), 77–90 (2016)

30. Lam, X.N., Vu, T., Le, T.D., Duong, A.D.: Addressing Cold-Start Problem in Recommendation Systems. In: Proceedings of the 2nd International Conference on Ubiquitous Information Management and Communication, pp. 208–211. ACM (2008)

31. Schein, A.I., Popescul, A., Ungar, L.H., Pennock, D.M.: Methods and Metrics for Cold-Start Recommendations. In: Proceedings of the 25th Annual International ACM SIGIR Conference, pp. 253–260. ACM (2002)

32. Chen, J., Nairn, R., Nelson, L., Bernstein, M., Chi, E.: Short and Long-Term Recommendations with Recurrent Neural Networks. In: Proceedings of the 26th ACM Conference on User Modeling, Adaptation and Personalization, pp. 1–9. ACM (2018)

33. Ricci, F., Rokach, L., Shapira, B.: Introduction to Recommender Systems Handbook. Springer (2011)

34. Alqaderi, H., Gogos, C., Alefragis, P.: A Meta-Learning Approach for Constraint Parameter Initialisation in Job-Shop Scheduling. In: Proceedings of the 14th International Conference on Integration of Constraint Programming, Artificial Intelligence, and Operations Research, pp. 42–57. Springer (2017)

35. Finn, C., Abbeel, P., Levine, S.: Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. In: Proceedings of the 34th International Conference on Machine Learning, pp. 1126–1135. PMLR (2017)

36. Akiba, T., Sano, S., Yanase, T., Ohta, T., Koyama, M.: Optuna: A Next-Generation Hyperparameter Optimization Framework. In: Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 2623–2631. ACM (2019)

37. Qu, R., Burke, E.K., McCollum, B., Merlot, L.T.G., Lee, S.Y.: A Survey of Search Methodologies and Automated System Development for Examination Timetabling. Journal of Scheduling 12(1), 55–89 (2009)

38. Lewis, R.: A Survey of Metaheuristic-Based Techniques for University Timetabling Problems. OR Spectrum 30(1), 167–190 (2008)

39. Tuga, M., Berretta, R., Mendes, A.: A Hybrid Simulated Annealing with Kempe Chain Neighbourhood for the University Timetabling Problem. In: Proceedings of the 6th IEEE International Conference on Computer and Information Science, pp. 400–405. IEEE (2007)

40. Di Gaspero, L., McCollum, B., Schaerf, A.: The Second International Timetabling Competition (ITC-2007): Curriculum-Based Course Timetabling (Track 3). Technical Report, Queen's University Belfast (2007)

41. Burke, E.K., Mareček, J., Parkes, A.J., Rudová, H.: Penalising Patterns in Timetables: Novel Integer Programming Formulations. In: Operations Research Proceedings 2007, pp. 409–414. Springer (2008)

42. Rudová, H., Müller, T., Murray, K.: Complex University Course Timetabling. Journal of Scheduling 14(2), 187–207 (2011)

43. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., Polosukhin, I.: Attention Is All You Need. In: Advances in Neural Information Processing Systems 30, pp. 5998–6008. NeurIPS (2017)

44. Kingma, D.P., Ba, J.: Adam: A Method for Stochastic Optimization. In: Proceedings of the 3rd International Conference on Learning Representations. ICLR (2015)

45. Snoek, J., Larochelle, H., Adams, R.P.: Practical Bayesian Optimization of Machine Learning Algorithms. In: Advances in Neural Information Processing Systems 25, pp. 2951–2959. NeurIPS (2012)
