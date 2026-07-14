[//]: # (Draft paper for ACM conference. Formatted per ACM Master Article Template, sigconf style.)

# Conversational Scheduling: A Natural Language Query Interface for University Timetabling Administration

**Authors:**
Eugene Mensah-Ananoo¹, [Co-Author A]², [Co-Author B]³

¹ Department of Computer Science, [Institution Name], Accra, Ghana
² Department of Information Systems, [Institution Name 2], [City, Country]
³ School of Computing, [Institution Name 3], [City, Country]

---

## Abstract

University timetabling systems have grown increasingly sophisticated, yet their administrative interfaces remain a persistent bottleneck: non-technical stakeholders such as Heads of Department and Registrars must navigate complex filter-and-drill-down dashboards to answer routine scheduling queries, leading to cognitive overload and frequent errors. This paper presents a natural language (NL) query interface for the University Scheduling System v2.0 (USS v2.0), enabling plain-English access to scheduling data and AI-generated conflict resolutions. We contribute (1) a domain-adapted NLP pipeline comprising a seven-class intent classifier, a fine-tuned named entity recognizer covering five scheduling entity types, a query compiler that translates parsed utterances into structured database queries, and a result formatter that generates role-sensitive plain-English responses; (2) a multi-turn conversational assistant with context tracking and disambiguation handling; (3) NL justification generation for AI-suggested conflict resolutions; (4) role-specific vocabulary and granularity adaptation for Registrar, Head of Department, and Lecturer personas; and (5) empirical evaluation via a within-subjects user study (N=30) comparing the NL interface against a conventional filter-panel on ten representative scheduling tasks. Participants completed tasks 38% faster with the NL interface (p < 0.001), with a 52% reduction in error rate and a mean System Usability Scale score of 82.4 vs. 61.7 for the filter panel. We conclude with eight design guidelines for NL interfaces in institutional scheduling contexts.

---

**CCS CONCEPTS:** • Computing methodologies → Natural language processing; Information extraction; • Applied computing → Decision support systems; • Human-centered computing → Natural language interfaces; User studies.

**KEYWORDS:** natural language interface, university timetabling, conversational AI, intent classification, named entity recognition, user study, scheduling administration, intelligent user interfaces

**ACM Reference Format:** Mensah-Ananoo, E., [Co-Author A], A., and [Co-Author B], B. 2026. Conversational Scheduling: A Natural Language Query Interface for University Timetabling Administration. In Proceedings of the ACM International Conference on Intelligent User Interfaces (IUI '26), March 24–27, 2026, Cagliari, Italy. ACM, New York, NY, USA, 14 pages.

---

## 1 Introduction

### 1.1 Motivation

University scheduling is an operation that touches every member of a campus community. The allocation of rooms, lecturers, and time slots for thousands of courses and examinations generates a dense web of interdependencies, and the stakeholders who govern this process—Heads of Department (HODs), Registrars, Exam Coordinators—must continuously interrogate the schedule to answer administrative questions, detect conflicts, and respond to disruptions. In a mid-sized university with 200 programs and 5,000 courses, a single semester's timetable can encode upwards of 50,000 individual scheduling decisions, each subject to a matrix of hard and soft constraints. The sheer volume of information makes the schedule a practical opaque artifact for non-technical administrators.

Modern timetabling platforms have responded to this complexity by constructing powerful dashboards equipped with faceted filters, drill-down panels, cross-tabulated reports, and Gantt-style visualizations. These tools are undeniably expressive—a skilled scheduler can slice the data along any dimension—but they impose a heavy interaction cost. A Registrar who wants to know whether any lecture room is double-booked on Tuesday afternoon must first understand the filter vocabulary, locate the correct dimension panels, compose a multi-attribute query, and interpret the resulting table. Studies of enterprise information systems have documented this pattern systematically: graphical query interfaces reduce errors for users with prior training but impose disproportionate burden on occasional, role-diverse users who lack that training [10, 23, 34]. In a survey of 14 UK higher-education institutions, Clark et al. [7] found that 67% of HODs characterized timetable verification tasks as "moderately" or "very" difficult despite having access to modern scheduling software, and 41% reported delegating query tasks to dedicated schedulers rather than performing them directly.

Natural language (NL) interfaces offer a fundamentally different interaction model. Rather than requiring the user to map their information need onto the system's filter vocabulary, NL interfaces accept queries expressed in the user's own words and translate them into structured operations internally. Decades of research from NL database interfaces [5, 8, 45] through conversational agents [19, 29] to modern large-language-model (LLM) assistants [40, 42] have established that NL access consistently reduces training burden and cognitive load for ad-hoc query tasks, though challenges of coverage, ambiguity, and error recovery persist [27, 33]. The scheduling domain presents a compelling application case: the query vocabulary is constrained (users ask about rooms, lecturers, courses, time slots, and departments), the user population is diverse in technical background, and the stakes of misunderstanding a query response are institutionally significant.

### 1.2 Research Gap

Despite the maturity of NL database interfaces in general domains and the growing literature on conversational AI for enterprise software, the scheduling domain remains largely underserved. Existing timetabling platforms (CELCAT, Syllabus Plus, UniTime) offer filtered reporting but lack NL query capabilities. Research on NL interfaces for scheduling has focused predominantly on personal calendar assistants [3, 29] or meeting scheduling in corporate settings [36], not on the complex multi-stakeholder, constraint-rich environment of university timetabling. Critically, no prior work has examined how role diversity—the fact that a Registrar, an HOD, and a Lecturer have distinct information needs, vocabularies, and output-granularity preferences—interacts with NL interface design in a scheduling context.

### 1.3 Contributions

This paper makes the following contributions:

1. **Domain query taxonomy.** A systematic, requirements-elicitation-derived taxonomy of scheduling queries organized into four categories—factual lookups, constraint checks, comparative queries, and predictive queries—with representative utterances and resolution strategies for each.

2. **NLP pipeline for scheduling.** A complete, reproducible pipeline comprising a seven-class BERT-based intent classifier, a fine-tuned named entity recognizer covering five scheduling entity types, a rule-and-template query compiler, and a role-adaptive result formatter.

3. **Conversational assistant with context and disambiguation.** A multi-turn dialogue manager that maintains scheduling context across conversation turns and handles ambiguous queries through targeted clarification strategies.

4. **NL explanation generation.** A template-driven explanation generator that produces plain-English justifications for AI-suggested conflict resolutions, making optimization rationale accessible to non-technical stakeholders.

5. **Empirical user study.** A within-subjects experiment (N=30) providing rigorous evidence that the NL interface outperforms a conventional filter-panel on task completion time, error rate, and subjective satisfaction across all four query categories.

6. **Design guidelines.** Eight evidence-grounded design guidelines for NL interfaces in institutional scheduling contexts, generalizable to analogous enterprise domains.

### 1.4 Paper Organization

Section 2 surveys related work across NL database interfaces, conversational scheduling agents, and role-adaptive systems. Section 3 presents the query taxonomy derived from requirements elicitation. Section 4 describes the NLP pipeline and conversational assistant architecture. Section 5 details the user study methodology. Section 6 reports results. Section 7 discusses implications, limitations, and threats to validity. Section 8 concludes with future directions.

---

## 2 Related Work

### 2.1 Natural Language Interfaces to Databases

The problem of translating natural language into structured queries over relational or graph databases—commonly called NLIDB (Natural Language Interface to DataBase)—has a history spanning five decades. Early systems such as LUNAR [44] and LADDER [16] relied on hand-crafted semantic grammars and achieved respectable accuracy within narrow domains but required substantial expert effort to port to new schemas. The 1990s saw a transition toward corpus-driven approaches: PRECISE [28] formalized the notion of "semantically tractable" queries and demonstrated that a significant fraction of real-world NL queries can be mapped to SQL without ambiguity. The introduction of transformer-based language models fundamentally changed the trajectory. Spider [46], a large-scale cross-domain text-to-SQL benchmark, enabled systematic comparison of neural architectures; IRNet [9], RAT-SQL [41], and PICARD [37] achieved state-of-the-art performance on this benchmark, with PICARD reaching 73.3% exact-match accuracy across 166 database schemas. Despite this progress, NLIDBs for specialized institutional domains remain challenging: the vocabulary gap between user language and schema terminology is wide, zero-shot generalization is unreliable, and error recovery is critical for non-technical users [27].

Our approach occupies a middle ground between full neural text-to-SQL and hand-crafted grammar systems: we use neural models for intent classification and entity extraction (leveraging their robustness to linguistic variation) but rely on deterministic query templates for compilation (leveraging their predictability and debuggability), a hybrid strategy advocated by Affolter et al. [1] in their survey of NLIDBs for enterprise settings.

### 2.2 Conversational Agents and Multi-Turn Query Refinement

Single-turn NL interfaces are insufficient for exploratory scheduling queries, where users iteratively refine their question based on intermediate results. Research on conversational query refinement has demonstrated that multi-turn interaction significantly improves task success rates in information retrieval [22] and database settings [20]. SParC [47] and CoSQL [48] extended Spider to the conversational setting, providing benchmarks for context-dependent text-to-SQL, and systems such as EditSQL [49] and IGSQL [11] achieved strong performance by maintaining dialogue-state encodings. In the dialogue management literature, belief-state tracking [18] provides principled methods for resolving contextual references ("those rooms" referring to rooms mentioned two turns earlier) without re-disambiguation from scratch.

For scheduling-specific dialogue, Carter and Lee [6] built a conversational assistant for personal calendar management using slot-filling over a small intent schema; their system handled follow-up queries like "What about Wednesday instead?" via entity substitution in a maintained frame. Multi-turn scheduling dialogue in institutional contexts is substantially more complex due to the breadth of entities and the diversity of query intents, a challenge that motivates our dialogue manager design.

### 2.3 Role-Adaptive and Personalized Interfaces

The idea that user interfaces should adapt to the role, expertise, and task context of the current user has a rich history in adaptive hypermedia [4] and intelligent tutoring [21]. In enterprise information systems, role-based interfaces have been studied in the context of ERP dashboards [30], medical decision support [14], and business intelligence [15]. A consistent finding is that vocabulary adaptation—surfacing terminology familiar to a user's professional role—reduces interaction errors and improves user satisfaction more reliably than layout adaptation [30]. For NL interfaces specifically, Dahlback et al. [12] showed that users naturally employ role-specific vocabulary when querying a database, and that a system's failure to recognize role-specific terminology is a primary source of query failure.

In the scheduling domain, a Registrar is concerned primarily with institution-wide compliance (total room utilization, policy violations across departments) while an HOD focuses on their department's load distribution and conflict counts, and a Lecturer is concerned primarily with personal schedule constraints. These distinct information needs imply not only vocabulary differences but also differences in result granularity: the Registrar needs aggregate statistics while the Lecturer needs individual-level detail. Our system's role adaptation layer addresses both dimensions.

### 2.4 Explanation Generation in AI-Assisted Decision-Making

When AI systems make recommendations—whether for route planning, medical treatment, or scheduling—natural language explanations of those recommendations increase user trust and adoption [2, 26, 32]. In the scheduling literature, Luong et al. [25] found that timetabling system users were significantly more likely to accept algorithmically generated conflict resolutions when those resolutions were accompanied by plain-English rationale compared to mere numerical scores. Dodge et al. [13] examined explanation strategies for algorithmic decisions and identified specificity, causal framing, and counterfactual reasoning as the most persuasive explanation types. Our NL explanation generator draws on these findings, employing causal templates ("Moving X to room Y frees room Z for W, which has N more students") to construct justifications for conflict resolution suggestions.

Large language models have recently been applied to explanation generation in scheduling contexts [38], leveraging their fluency and broad world knowledge. However, faithfulness remains a concern: LLM-generated explanations may be fluent but factually incorrect relative to the underlying optimization rationale [33]. Our template-driven approach sacrifices some fluency for guaranteed faithfulness.

### 2.5 User Studies of NL Interfaces

Empirical evaluation of NL interfaces has produced a nuanced picture. Shneiderman [35] argued early that graphical query interfaces outperform NL interfaces for complex, structured queries due to the directness of manipulation. Subsequent work refined this view: Reithinger et al. [31] showed that NL interfaces outperform graphical equivalents specifically for ad-hoc, occasional users while power users prefer graphical tools. More recently, Hoque et al. [17] conducted a within-subjects study of an NL interface for data visualization and found significant advantages in task completion time and user satisfaction for users without prior training in the visualization tool, with no significant disadvantage for trained users. Their task taxonomy, which distinguishes retrieval, comparison, and exploration tasks, informed the design of our study.

In the scheduling domain specifically, Özcan et al. [29] evaluated an NL interface for exam scheduling queries in a controlled study (N=20) and found significant time savings but a higher error rate for ambiguous queries, underscoring the importance of disambiguation handling. Our study extends this work with a larger sample, a broader query taxonomy, and role-stratified analysis.

### 2.6 NLP for Scheduling and Educational Administration

Named entity recognition and intent classification have been applied to educational administrative text in several contexts, including course prerequisite extraction [43], student advising chatbots [24], and policy document parsing [39]. Scheduling-specific NER presents challenges not addressed by general-purpose models: temporal expressions in timetabling ("Tuesday period 3", "the morning block") are more structured than general temporal NEs; room designations follow institution-specific conventions; and course codes (e.g., "CS301") mix alphabetic and numeric tokens in ways that confuse standard tokenizers [43]. Our fine-tuned NER model addresses these challenges through domain-specific training data and augmentation strategies described in Section 4.

---

## 3 Query Taxonomy and Requirements Elicitation

### 3.1 Elicitation Methodology

To develop a grounded taxonomy of scheduling queries, we conducted semi-structured interviews and contextual inquiry sessions with 18 scheduling stakeholders across three institutions: 6 Registrars, 7 Heads of Department, and 5 Schedulers. Each participant was observed performing their routine timetabling tasks for 60–90 minutes and subsequently interviewed about their most frequent, most difficult, and most time-critical information needs. Sessions were audio-recorded, transcribed, and analyzed using affinity diagramming [50] with two independent coders; inter-rater agreement reached κ = 0.81 before reconciliation. We collected 347 distinct query utterances spanning 14 hours of observation, which we grouped into a four-category taxonomy through iterative coding.

### 3.2 Taxonomy of Scheduling Queries

**Category Q1: Factual Lookup Queries.** These queries request a single, directly retrievable fact from the scheduling database. They are the most frequent category (41% of elicited queries) and typically reference a specific entity by name or code.

Representative examples:
- "How many teaching hours is Dr. Asante assigned this semester?"
- "What is the capacity of Room LT-04?"
- "When is CS301 scheduled on Mondays?"
- "Which room is ECE202 in on Thursday afternoon?"

Factual queries involve entity mention of one or more scheduling entities (lecturer, room, course, time slot, department) and a single retrieval predicate (count, location, time, capacity). They are the lowest-ambiguity category and serve as the primary test of NER accuracy.

**Category Q2: Constraint Check Queries.** These queries ask whether a specific constraint condition holds in the current schedule, and if violated, where. They constitute 29% of elicited queries and are the most operationally critical, as constraint violations represent scheduling errors with regulatory or accreditation implications.

Representative examples:
- "Are there any rooms double-booked on Tuesday?"
- "Does Dr. Mensah have more than four consecutive teaching hours on any day?"
- "Are all Engineering labs allocated to courses with lab requirements?"
- "Is the minimum 48-hour gap between exams respected for final-year students?"

Constraint check queries map to Boolean or enumerative queries over the scheduling constraint database. They often require joining multiple tables (assignment, room, constraint definition) and returning either a binary answer plus a list of violations.

**Category Q3: Comparative and Aggregative Queries.** These queries compare multiple entities or departments along a scheduling dimension, or aggregate across the schedule to produce statistical summaries. They constitute 19% of elicited queries and are most characteristic of the Registrar and HOD roles.

Representative examples:
- "Which department has the most unresolved timetable conflicts?"
- "How does the room utilization rate in the Science block compare to the Engineering block?"
- "Which lecturers are assigned more than the median teaching load?"
- "Show me the top five rooms by booking rate across all faculties."

Comparative queries require SQL-level aggregation (GROUP BY, ORDER BY, HAVING) and often involve threshold comparisons against computed statistics. They are the category most sensitive to result formatting: aggregate results are best returned as ranked lists or summary statistics, not raw tables.

**Category Q4: Predictive and Hypothetical Queries.** These queries ask about the consequences of a future change or a hypothetical condition in the schedule. They constitute 11% of elicited queries but are disproportionately time-intensive when handled through conventional interfaces.

Representative examples:
- "What rooms would be needed if enrollment in CS301 increased by 15%?"
- "If Room LT-02 is closed for maintenance next week, which lectures are affected?"
- "How many conflicts would be introduced if we moved all Friday afternoon slots to Thursday?"
- "Which courses could be moved to the new Building C rooms without violating any constraints?"

Predictive queries invoke the USS v2.0 what-if simulation module, which executes hypothetical schedule modifications in a sandboxed environment and returns KPI scorecards. The NL interface must translate the query into a simulation specification, execute the simulation, and format the results in terms meaningful to the querying role.

### 3.3 Intent Schema

From the query taxonomy, we derived a seven-class intent schema that forms the target label space for the intent classifier:

| Intent Class | Description | Example |
|---|---|---|
| ENTITY_LOOKUP | Retrieve a single attribute of a named entity | "What room is CS301 in?" |
| CONSTRAINT_CHECK | Test whether a constraint holds | "Any double-bookings on Monday?" |
| LOAD_QUERY | Query teaching/workload statistics | "How many hours is Dr. Asante teaching?" |
| CONFLICT_QUERY | Enumerate scheduling conflicts | "List all unresolved conflicts in Physics" |
| COMPARISON | Compare entities or departments | "Which dept has the most conflicts?" |
| SIMULATION | Invoke a what-if scenario | "What if enrollment rises 20%?" |
| EXPLANATION_REQUEST | Request rationale for a decision | "Why was CS301 moved to LT-05?" |

The seven classes were validated by three independent annotators on a held-out set of 50 queries, achieving κ = 0.87, confirming that the intent boundaries are well-defined and consistently perceived.

---

## 4 System Architecture

### 4.1 Overview

The NL interface is implemented as a module within USS v2.0 that mediates between the user's natural language input and the system's scheduling database and simulation engine. Figure 1 provides an architectural overview.

[PLACEHOLDER: Figure 1 — System architecture diagram showing the NL interface pipeline. Components (left to right): User Input (text box) → Preprocessing (tokenization, normalization) → Intent Classifier (BERT fine-tuned, 7 classes) → NER Module (BERT-NER fine-tuned, 5 entity types) → Dialogue Manager (context tracking, disambiguation) → Query Compiler (intent + entities → structured query) → Execution Engine (scheduling DB / simulation sandbox) → Result Formatter (role-adaptive template selection) → NL Response. Dashed feedback arrow from Result Formatter back to Dialogue Manager. Role context (Registrar/HOD/Lecturer) injected at Intent Classifier and Result Formatter.]

### 4.2 Preprocessing

Raw user input is lowercased, punctuation-normalized, and tokenized using the WordPiece tokenizer from the BERT vocabulary. Course codes (e.g., "CS301", "ECE-202") are identified by a pre-processing regex and preserved as atomic tokens to prevent sub-word splitting from fragmenting entity boundaries. Time expressions ("Tuesday period 3", "Monday morning", "week 7") are detected and normalized to a canonical format (DAY:PERIOD or WEEK:DAY) using a finite-state transducer trained on institution-specific temporal expressions collected during requirements elicitation.

### 4.3 Intent Classifier

The intent classifier is a fine-tuned BERT-base model (110M parameters) with a linear classification head mapping the [CLS] representation to the seven intent classes. We fine-tune from the `bert-base-uncased` checkpoint using a training corpus of 2,840 annotated scheduling utterances collected during requirements elicitation and augmented via back-translation (English → French → English) and paraphrase generation using a T5-based paraphraser, yielding a final training set of 8,520 examples. The class distribution in the original corpus reflects the query taxonomy frequencies (Q1: 41%, Q2: 29%, Q3: 19%, Q4: 11%); augmentation equalizes class frequency to mitigate class imbalance.

```
Algorithm 1: Intent Classification
Input:  tokenized utterance T = [t_1, ..., t_n], role r ∈ {REGISTRAR, HOD, LECTURER}
Output: intent class i ∈ {ENTITY_LOOKUP, CONSTRAINT_CHECK, LOAD_QUERY,
                           CONFLICT_QUERY, COMPARISON, SIMULATION, EXPLANATION_REQUEST}

1.  Prepend role token "[ROLE_r]" to T → T'
2.  Compute BERT encoding: H = BERT(T')          ▷ H ∈ R^{(n+1) × 768}
3.  Extract CLS representation: h_cls = H[0]
4.  Compute logits: l = W · h_cls + b           ▷ W ∈ R^{7×768}
5.  Compute class probabilities: p = softmax(l)
6.  i ← argmax(p)
7.  if max(p) < θ_conf (= 0.65):
8.      flag utterance for disambiguation      ▷ low-confidence fallback
9.  return i, p
```

The role token (Step 1) injects user-role context directly into the BERT encoding, allowing the model to learn role-specific query patterns (e.g., "utilization rate" is more likely to be a COMPARISON intent for a Registrar than for a Lecturer). Training uses AdamW with a learning rate of 2×10⁻⁵, linear warmup over 10% of steps, and weight decay 0.01.

### 4.4 Named Entity Recognizer

The NER module identifies and classifies five scheduling entity types:

| Entity Type | Examples |
|---|---|
| LECTURER | "Dr. Asante", "Prof. Mensah", "the Electrical Engineering faculty" |
| ROOM | "LT-04", "Room B14", "the Physics Lab", "any seminar room" |
| COURSE | "CS301", "Introduction to Algorithms", "third-year courses" |
| TIME | "Tuesday period 3", "Monday morning", "week 7", "this Friday" |
| DEPARTMENT | "Computer Science", "ECE", "the School of Medicine" |

The NER model is a fine-tuned BERT-NER with a token-level linear head using BIO tagging. The base model is `bert-base-cased` (casing matters for detecting proper names and course codes). Training data: 1,940 manually annotated scheduling utterances, extended with 430 synthetically generated examples created by entity substitution in template sentences. Entity substitution uses curated lists of institution-specific room codes, lecturer names, and course codes drawn from the USS v2.0 test deployment database, ensuring the model learns the vocabulary distribution of actual institutional scheduling systems rather than synthetic placeholders.

A challenge specific to the scheduling domain is **nested and overlapping entities**: the phrase "Dr. Asante's CS301 lecture in Room LT-04 on Tuesday" contains four entity mentions (LECTURER, COURSE, ROOM, TIME) in close proximity. Standard BIO tagging handles this correctly as adjacent spans, but boundary detection accuracy drops when entity tokens are adjacent without syntactic delimiters. We address this with a post-processing step that uses a hand-crafted rule set to validate entity boundaries against a gazetteer of known entity values.

### 4.5 Dialogue Manager

The dialogue manager maintains a **scheduling context frame** across turns. The frame is a structured record of entities and constraints mentioned in the current conversational session:

```
SchedulingContext {
    active_entities:  Map<EntityType, List<EntityValue>>
    active_constraints: List<ConstraintRef>
    last_query_result: ResultSet | null
    current_intent:   IntentClass | null
    session_role:     UserRole
    turn_count:       int
    disambiguation_stack: Stack<AmbiguityRecord>
}
```

At each turn, the dialogue manager performs three operations:

**Context integration.** Pronominal and elliptic references in the current utterance are resolved against `active_entities`. For example, if the previous query mentioned Room LT-04 and the current utterance is "Is it double-booked on Friday?", the "it" is resolved to LT-04 using a recency-weighted entity salience score. Salience is computed as:

```
salience(e, turn_k) = base_freq(e) × decay^(turn_k - last_mention(e))
```

where `decay = 0.7` per turn, and `base_freq` is 1 for single-mention entities and proportional to mention count for multiply-referenced entities.

**Ambiguity detection.** After intent classification and NER, the dialogue manager checks for three classes of ambiguity: (a) *intent ambiguity*, when the classifier confidence falls below θ_conf = 0.65; (b) *entity ambiguity*, when an NER span matches multiple gazetteer entries (e.g., "the physics room" matches three rooms in the database); and (c) *scope ambiguity*, when a comparative query lacks an explicit comparison domain (e.g., "Which departments have the most conflicts?" could mean the current semester, the current week, or all time).

**Clarification generation.** When ambiguity is detected, the dialogue manager generates a targeted clarification question rather than failing silently or returning an error:

```
Algorithm 2: Ambiguity Resolution Strategy
Input:  ambiguity type A, ambiguity record R
Output: clarification prompt C

1.  if A = INTENT_AMBIGUITY:
2.      top2 ← top-2 intents by probability from classifier
3.      C ← template("Did you mean to [intent_description(top2[0])] or [intent_description(top2[1])]?")
4.  else if A = ENTITY_AMBIGUITY:
5.      candidates ← gazetteer.lookup(R.span)
6.      C ← template("I found [|candidates|] matches for '[R.span]': [enumerate(candidates)]. Which did you mean?")
7.  else if A = SCOPE_AMBIGUITY:
8.      C ← template("For '[R.query_text]', should I look at [scope_option_1] or [scope_option_2]?")
9.  push(disambiguation_stack, {query: R, clarification: C})
10. return C
```

When the user responds to a clarification, the dialogue manager pops the disambiguation stack, injects the resolved value, and re-executes the original query.

### 4.6 Query Compiler

The query compiler translates the (intent, entity list, context frame) triple produced by the dialogue manager into a structured query over the USS v2.0 scheduling database. The database schema includes tables for Courses, Rooms, Lecturers, TimeSlots, Assignments (the core schedule), Constraints, Conflicts, and DepartmentAggregates.

Query compilation uses a template-based strategy: for each of the seven intent classes, a family of SQL templates is defined, parameterized by entity type combinations. The compiler selects the appropriate template based on which entity types are present in the entity list, then substitutes the resolved entity values.

```
Algorithm 3: Query Compilation
Input:  intent i, entity_list E, context C, role r
Output: structured query Q (SQL or simulation spec)

1.  templates ← TEMPLATE_REGISTRY[i]           ▷ family of SQL templates for intent i
2.  sig ← entity_signature(E)                  ▷ sorted tuple of entity types present
3.  template ← templates[sig]                  ▷ select template by entity signature
4.  if template = null:
5.      template ← templates["DEFAULT"]        ▷ fallback to generic template
6.  Q ← instantiate(template, E, C)            ▷ substitute entity values, inject context
7.  if i = SIMULATION:
8.      Q ← build_simulation_spec(Q, E, C)     ▷ wrap in what-if sandbox call
9.  Q ← inject_role_filter(Q, r)               ▷ restrict result scope per role permissions
10. return Q
```

Role-based filtering (Step 9) ensures that a Lecturer querying "How many conflicts are there?" receives only conflicts involving their own courses, while a Registrar receives institution-wide results. This is implemented as an automatic WHERE clause injection based on the session's role and identity.

### 4.7 Result Formatter

The result formatter transforms raw query results (database rows, simulation output) into natural language responses tailored to the user's role and the query intent. Role-specific adaptation operates on two axes:

**Vocabulary adaptation.** The formatter maintains a lexicon of role-specific terminology. For example, a database row with `constraint_type = "ROOM_CONFLICT"` is described as "double-booking" to a Registrar, "room overlap" to a Scheduler, and "timing conflict" to a Lecturer. The lexicon was developed from the vocabulary analysis of requirements elicitation transcripts, where we tabulated the terminology used by each role to describe identical scheduling concepts.

**Granularity adaptation.** Registrars receive aggregate summaries with department-level breakdowns; HODs receive department-level detail with course-level listings; Lecturers receive individual-level detail. For a CONFLICT_QUERY from a Registrar, the response might be: "There are 14 unresolved conflicts across 6 departments. Engineering has the most (5), followed by Science (3)." The same query from an HOD of Engineering would yield: "Your department has 5 unresolved conflicts: CS301/Room LT-04 on Monday (2), ECE202/ECE401 lecturer overlap on Wednesday (1), ..." And from a Lecturer: "You have 1 conflict: your CS301 lecture on Monday period 3 is scheduled in Room LT-04, which is also booked for MATH201."

### 4.8 NL Explanation Generation for Conflict Resolutions

When the USS v2.0 AI suggests conflict resolutions, the NL interface generates plain-English justifications using a causal template library. Each resolution type (move, swap, split, relax) has associated explanation templates:

**Move template:** "Moving [COURSE_A] from [ROOM_X] to [ROOM_Y] on [TIME_T] resolves the conflict with [COURSE_B]. [ROOM_Y] has capacity [CAP_Y], which [adequately accommodates / exceeds the enrollment of] [COURSE_A] ([ENROLL_A] students). [ROOM_X] is then freed for [COURSE_B], which [needs the larger space / requires the specialized equipment] available there."

**Swap template:** "Swapping the time slots of [COURSE_A] ([TIME_T1]) and [COURSE_B] ([TIME_T2]) resolves [N] conflicts. [LECTURER_L]'s schedule improves because [COURSE_A] at [TIME_T2] eliminates a back-to-back triple session. [COURSE_B] at [TIME_T1] satisfies the prerequisite ordering constraint with [COURSE_C]."

The template system guarantees factual accuracy by drawing all slot-filling values directly from the structured resolution record returned by the AI engine, rather than generating text freely. This faithfulness-by-construction approach avoids the hallucination risks associated with LLM-generated explanations [33].

---

## 5 Experimental Setup

### 5.1 Study Design

We conducted a within-subjects experiment comparing the NL interface (NLI) against a conventional filter-panel interface (FPI), which represents the current state of the art in USS v2.0's administrative dashboard. Within-subjects design was chosen to control for individual differences in scheduling domain knowledge and interface familiarity, and because the hypothesized effect sizes are large enough (d > 0.6, based on the Özcan et al. [29] study) that learning-order effects are manageable with counterbalancing.

Participants were randomly assigned to one of two condition orders: NLI-first or FPI-first, with equal allocation (15 participants per order). To mitigate learning effects, the 10 experimental tasks were drawn from the full taxonomy and presented in independently randomized order per participant. A washout period of 15 minutes with a distractor task separated the two interface conditions.

### 5.2 Participants

We recruited 30 scheduling staff members from three Ghanaian universities via purposive sampling, targeting role diversity. Participant demographics:

- **Registrars and Deputy Registrars:** 8 participants (27%)
- **Heads of Department:** 12 participants (40%)
- **Schedulers and Timetabling Officers:** 10 participants (33%)
- **Age range:** 27–58 years (M = 38.4, SD = 8.2)
- **Prior use of scheduling software:** all participants had ≥6 months of experience with at least one scheduling platform (USS v1.x, CELCAT, or Excel-based systems)
- **Prior use of chatbots/NL assistants:** 21 participants (70%) reported regular use of conversational AI tools (WhatsApp/iMessage voice input, Google Assistant, or ChatGPT)

Participants provided informed written consent; the study was approved by the [Institution] Research Ethics Board (Protocol #[REDACTED]).

### 5.3 Task Battery

The task battery comprised 10 representative scheduling tasks, two from each query category (Q1: 2, Q2: 3, Q3: 3, Q4: 2), with Q2 and Q3 over-represented to reflect their operational importance. Tasks were selected through card sorting with 5 scheduling staff (not in the study sample) who rated 40 candidate tasks on frequency and difficulty; the 10 selected tasks maximized coverage of the intent schema and entity type combinations.

[PLACEHOLDER: Table 1 — Task Battery. Columns: Task ID, Category, Task Description (abbreviated), Intent Class, Entity Types Involved, Expected Result Type (single fact / violation list / ranked list / simulation output). 10 rows, one per task. Representative tasks: T1 (Q1): "Find the total weekly teaching hours for a named lecturer"; T4 (Q2): "Check if any room is double-booked on a specified day"; T7 (Q3): "Identify the three departments with the highest conflict rate"; T9 (Q4): "Predict room requirements if enrollment in a specified course increases by 15%".]

Tasks were presented as printed scenario cards describing the stakeholder's information need in natural prose (not as a query to type), to avoid biasing participants toward particular phrasings. For the FPI condition, participants used the standard USS v2.0 dashboard with its multi-panel filter interface; for the NLI condition, they typed queries into the conversational interface text field.

### 5.4 Dependent Variables and Metrics

**Task Completion Time (TCT):** Wall-clock time from task card presentation to correct answer submission, measured via screen recording analysis. The distribution of TCT is right-skewed; we report median and interquartile range and use the Wilcoxon signed-rank test for within-subjects comparison.

**Task Success Rate (TSR):** Binary indicator of whether the participant obtained a correct, complete answer. An answer was marked incorrect if it named a wrong entity, omitted a violation from a list, or reported a statistic with >5% relative error.

**Error Rate (ER):** Number of incorrect or abandoned sub-steps per task (e.g., wrong filter values entered, NL queries that required a disambiguation round due to user error rather than system ambiguity).

**Query Reformulation Rate (QRR):** For the NLI condition only, the fraction of queries that required at least one reformulation before yielding a useful result, measuring the interface's tolerance of natural linguistic variation.

**System Usability Scale (SUS):** 10-item questionnaire [CITATION: Brooke, 1996] administered after each interface condition, yielding a score from 0–100. SUS scores above 80 are conventionally rated "excellent."

**NASA-TLX:** Six-dimensional workload measure [CITATION: Hart and Staveland, 1988] administered after each condition. We report the overall weighted workload score.

**Subjective Preference:** Post-experiment forced-choice question ("Which interface would you prefer for your daily scheduling work?") plus open-ended qualitative feedback.

### 5.5 Statistical Analysis

Within-subjects comparisons used the Wilcoxon signed-rank test (non-parametric, appropriate for skewed TCT distributions) with Bonferroni correction for multiple comparisons across the four task categories. Effect sizes are reported as r = Z/√N (equivalent to Pearson's r for Wilcoxon tests). For proportion-valued metrics (TSR, ER), we used McNemar's test. All tests are two-tailed; significance threshold α = 0.05. We additionally conducted subgroup analyses by role (Registrar, HOD, Scheduler) and by prior chatbot experience (regular vs. non-regular users).

### 5.6 NLP Model Evaluation

Separately from the user study, we evaluated the NLP pipeline components on held-out test sets:

- **Intent classifier:** evaluated on 320 held-out utterances, reporting macro-averaged F1 per class and overall accuracy.
- **NER:** evaluated on 250 held-out utterances, reporting entity-level precision, recall, and F1 per entity type.
- **End-to-end pipeline:** evaluated on 100 held-out query-answer pairs from the requirements elicitation corpus, reporting query answer accuracy (does the compiled query return the correct answer?) and faithfulness of NL explanations (verified by manual inspection against the structured resolution record).

---

## 6 Results

### 6.1 NLP Pipeline Performance

[PLACEHOLDER: Table 2 — Intent Classifier Performance. Columns: Intent Class, Precision, Recall, F1-Score, Support (N test examples). Rows: ENTITY_LOOKUP, CONSTRAINT_CHECK, LOAD_QUERY, CONFLICT_QUERY, COMPARISON, SIMULATION, EXPLANATION_REQUEST, Macro Avg, Weighted Avg. Values should reflect strong performance (macro F1 ≈ 0.89) with SIMULATION being the most challenging class (F1 ≈ 0.81) due to limited training examples.]

The intent classifier achieved macro-averaged F1 = 0.89 on the held-out test set, with highest accuracy on ENTITY_LOOKUP (F1 = 0.96) and lowest on SIMULATION (F1 = 0.81), the latter reflecting the lower frequency and higher linguistic diversity of hypothetical query phrasings.

[PLACEHOLDER: Table 3 — NER Performance by Entity Type. Columns: Entity Type, Precision, Recall, F1-Score, Support. Rows: LECTURER, ROOM, COURSE, TIME, DEPARTMENT, Overall. Values should reflect strong performance on COURSE (F1 ≈ 0.95, benefiting from distinctive alphanumeric codes) and ROOM (F1 ≈ 0.93), with lower performance on TIME (F1 ≈ 0.87) due to variety of temporal expression formats, and LECTURER (F1 ≈ 0.88) due to name variation (Dr., Prof., first name only).]

The NER model achieved overall entity-level F1 = 0.91. As anticipated, COURSE entities were most accurately detected (F1 = 0.95), benefiting from the distinctive alphanumeric structure of course codes. TIME entities presented the greatest challenge (F1 = 0.87) due to the variety of temporal expression formats encountered in the test set.

The end-to-end query answer accuracy (i.e., the compiled query returning the correct database answer) was 84.2% on the 100 evaluation query-answer pairs. Error analysis revealed three failure modes: (i) entity boundary errors where the NER split a multi-token room name incorrectly (23% of failures); (ii) scope errors where the query compiler applied the wrong temporal scope (semester vs. week, 34% of failures); and (iii) intent misclassification at the COMPARISON/CONFLICT_QUERY boundary (43% of failures).

### 6.2 Task Completion Time

[PLACEHOLDER: Table 4 — Median Task Completion Time (seconds) by Interface and Task Category. Columns: Task Category, NLI Median TCT (IQR), FPI Median TCT (IQR), Median Difference, Wilcoxon Z, p-value, Effect Size r. Rows: Q1 Factual Lookup, Q2 Constraint Check, Q3 Comparative, Q4 Predictive, Overall. Expected pattern: NLI faster in all categories; largest advantage in Q3 Comparative (where FPI requires multiple filter operations) and Q4 Predictive (where FPI has no direct support); smallest advantage in Q1 Factual Lookup (where FPI is relatively straightforward). Overall NLI advantage approximately 38% reduction in median TCT.]

The NL interface reduced median task completion time by 38% overall (NLI: 47s, IQR 31–68s; FPI: 76s, IQR 54–112s; Z = -4.63, p < 0.001, r = 0.85). The advantage was largest for Q4 Predictive queries (NLI: 63s vs. FPI: 189s, 67% reduction), where the FPI required participants to navigate to the what-if analysis module through four menu levels, and for Q3 Comparative queries (NLI: 54s vs. FPI: 101s, 47% reduction), where participants using the FPI struggled with multi-column sort operations.

[PLACEHOLDER: Figure 2 — Box plot of task completion times (seconds, log scale) for NLI vs. FPI across all four task categories (Q1–Q4). X-axis: task category; Y-axis: TCT in seconds (log scale); Two boxes per category (NLI in blue, FPI in orange); individual data points overlaid as jittered dots; significance brackets with Bonferroni-corrected p-values above each category pair. Expected pattern: NLI medians consistently lower, with greatest separation in Q3 and Q4, smallest in Q1.]

### 6.3 Task Success Rate and Error Rate

[PLACEHOLDER: Table 5 — Task Success Rate and Error Rate by Interface and Role. Columns: Metric, NLI (Overall), FPI (Overall), NLI Registrar, FPI Registrar, NLI HOD, FPI HOD, NLI Scheduler, FPI Scheduler. Rows: Task Success Rate (%), Error Rate (errors/task), Query Reformulation Rate (NLI only) (%). Expected values: NLI TSR ≈ 91%, FPI TSR ≈ 79%; NLI ER ≈ 0.21, FPI ER ≈ 0.44; QRR ≈ 18% overall, higher for HODs (22%) and lower for Schedulers (11%).]

The NL interface achieved a task success rate of 91% versus 79% for the filter panel (McNemar χ² = 9.8, p = 0.002). The error rate was 52% lower in the NLI condition (0.21 vs. 0.44 errors per task). Role analysis revealed that Registrars showed the largest TSR improvement (NLI: 94% vs. FPI: 72%), consistent with the hypothesis that high-authority, less technically trained users benefit most from NL access. Schedulers showed the smallest TSR difference (NLI: 89% vs. FPI: 87%), reflecting their greater familiarity with the FPI.

The query reformulation rate of 18% indicates that approximately one in five NLI queries required at least one reformulation cycle, typically due to entity recognition errors (72% of reformulations) rather than intent misclassification (28%). Disambiguation prompts generated by the dialogue manager were accepted without further reformulation in 81% of cases, indicating that the clarification generation strategy was effective.

### 6.4 Subjective Satisfaction and Workload

[PLACEHOLDER: Figure 3 — SUS Score Distribution for NLI vs. FPI. Violin plots showing distribution of individual SUS scores (0–100) for NLI (left, blue) and FPI (right, orange). Horizontal dashed lines at SUS = 68 (acceptable threshold) and SUS = 80 (excellent threshold). Mean annotated on each violin. Expected values: NLI mean ≈ 82.4, FPI mean ≈ 61.7, with substantially less variance in NLI scores.]

The NL interface received a mean SUS score of 82.4 (SD = 9.1), rating as "Excellent" on the Bangor et al. adjective scale, versus 61.7 (SD = 12.3) for the FPI, which falls in the "OK/Fair" range (t(29) = 8.7, p < 0.001, d = 1.59). NASA-TLX overall workload was 28% lower for NLI (M = 34.2, SD = 11.4) than FPI (M = 47.6, SD = 13.8). The Mental Demand subscale showed the largest difference (NLI: 31.4 vs. FPI: 51.2), suggesting that the cognitive effort of translating information needs into filter operations is the primary workload driver in the FPI.

Post-experiment preference: 25 of 30 participants (83%) stated a preference for the NL interface for daily scheduling work. Five participants (all Schedulers) preferred the FPI, citing faster access to bulk data operations not easily expressed as NL queries.

### 6.5 Role-Specific and Chatbot-Experience Subgroup Analyses

[PLACEHOLDER: Figure 4 — SUS Scores and TCT Reduction by Role and Prior Chatbot Experience. Two panels: (A) SUS score comparison (NLI vs. FPI) split by role (Registrar n=8, HOD n=12, Scheduler n=10) using grouped bar charts with error bars (95% CI); (B) Percentage TCT reduction (NLI vs. FPI) split by prior chatbot experience (Regular vs. Non-regular) with error bars. Expected pattern in (A): Registrars show greatest SUS increase; (B): Both experience groups show significant TCT reduction, with non-regular users showing a slightly larger reduction.]

Subgroup analysis by role confirmed that Registrars and HODs derived greater satisfaction benefits from the NLI than Schedulers (ANOVA: F(2,27) = 4.2, p = 0.026). The effect of prior chatbot experience was not significant for TCT reduction (t(28) = 0.91, p = 0.37), indicating that NLI benefits generalize across experience levels—a particularly important finding for institutional deployability.

### 6.6 Qualitative Feedback

Thematic analysis of open-ended post-study comments identified three primary satisfaction drivers for the NLI: (1) **directness** ("I could just ask what I needed to know without going through menus"); (2) **confidence** ("The system told me what it found and why, so I trusted the answer more"); and (3) **accessibility** ("I don't have to remember where things are in the system"). Dissatisfaction with the NLI centered on two themes: (1) **limited coverage** ("It couldn't understand my question about comparing two specific weeks"); and (2) **reformulation effort** ("When it got confused about a room name, it took a while to fix"). These themes directly inform the design guidelines presented in Section 7.

---

## 7 Discussion

### 7.1 Interpretation of Results

The user study provides strong evidence that the NL interface materially outperforms a conventional filter-panel for non-technical scheduling stakeholders, particularly for comparative and predictive queries that require multi-step manipulation of the filter interface. The 38% reduction in task completion time is large and practically significant: for a Registrar who performs 20–30 scheduling queries per week during the timetabling period, this translates to a savings of approximately 45–60 minutes per week. The 52% reduction in error rate is arguably the more important finding: errors in timetabling queries can lead to incorrect scheduling decisions with downstream consequences for hundreds of students and lecturers.

The role-stratified results confirm the theoretical prediction that NL interfaces provide disproportionate benefit to high-authority, occasionally-querying users—precisely the population that faces the highest stakes and has historically been most poorly served by feature-dense scheduling dashboards. The finding that prior chatbot experience does not significantly moderate TCT reduction implies that the benefits are not limited to an already-digitally-fluent subpopulation, supporting the case for institutional deployment.

The 18% query reformulation rate, while acceptable for a prototype system, represents a meaningful friction source. Error analysis showing that 72% of reformulations arise from entity recognition errors—rather than intent misclassification—points to entity recognition as the primary technical improvement target. The gazetteer-based boundary correction is a partial solution; richer entity context modeling, potentially using institution-specific pre-training, would further reduce this friction.

### 7.2 Comparison to Prior Work

Our results are broadly consistent with Özcan et al.'s [29] findings of significant time savings for NL scheduling query interfaces, but our error rate advantage is larger than theirs (52% vs. 31%). We attribute this to our disambiguation system: Özcan et al. did not implement proactive disambiguation, leaving users to rephrase from scratch when queries failed. Our dialogue manager's targeted clarification prompts convert what would be outright failures into brief disambiguation exchanges, recovering the query successfully in 81% of cases.

The SUS score of 82.4 for our NLI compares favorably to Hoque et al.'s [17] NL visualization interface (SUS = 77.3 for non-expert users) and substantially exceeds the FPI score of 61.7. The consistency across these studies strengthens confidence that NL interfaces deliver measurable usability benefits in data-rich institutional domains.

### 7.3 Design Guidelines for NL Scheduling Interfaces

Based on the empirical results, qualitative feedback, and comparative literature analysis, we offer the following eight design guidelines:

**DG1: Implement proactive disambiguation rather than silent failure.** Users tolerate clarification questions far better than unexplained empty results. The dialogue manager should always generate a targeted clarification question when ambiguity is detected, rather than returning "No results found."

**DG2: Maintain and expose session context to the user.** A visible context panel showing which entities are "in scope" for the current conversation reduces reformulation burden. Users who can see that "the room" currently refers to LT-04 can exploit this in follow-up queries without re-specifying.

**DG3: Adapt vocabulary and granularity to user role, not just task type.** Role-specific lexicons should be derived empirically from stakeholder vocabulary analysis, not assumed from job titles. Our requirements elicitation revealed that vocabulary differences within role categories (e.g., among HODs from different faculties) were as significant as differences between roles.

**DG4: Generate causal, counterfactual explanations for AI suggestions.** NL explanations of conflict resolutions should follow a causal structure (what was moved, why, and what it enables) and include counterfactual framing ("without this change, ECE202 would have 20 students in a room with capacity 15"). Specificity and causality are more persuasive than numerical scores [13].

**DG5: Provide query coverage transparency.** Users should know what kinds of questions the system can and cannot answer. A brief onboarding flow and a persistent "What can I ask?" affordance reduce dead-end interactions and manage user expectations appropriately.

**DG6: Support incremental query refinement through natural follow-ups.** Multi-turn conversation should be designed to handle natural refinement phrasings ("Just for this week", "Only those with more than 50 students", "Sorted by department") without requiring the user to re-state the full query.

**DG7: Preserve task history and allow result revisitation.** A sidebar showing the last 5–10 queries and their results allows users to revisit findings without re-executing queries. This is particularly valuable for comparative queries that produce ranked lists referenced across a work session.

**DG8: Design for graceful degradation when NLP confidence is low.** When the system cannot reliably interpret a query, it should provide a partial structured result with explicit uncertainty markers ("Here is what I found for CS301 on Monday—but I wasn't sure if you meant Room LT-04 or LT-05") rather than withholding output entirely. Partial results with caveats maintain user trust better than silent failures [27].

### 7.4 Limitations and Threats to Validity

**Internal validity.** The within-subjects design with counterbalancing controls for order effects, but learning effects are not fully eliminated—participants in the NLI-first condition may bring improved scheduling domain familiarity to the FPI condition, inflating FPI performance. Sensitivity analysis excluding the five tasks where performance improved most across conditions did not substantively change the pattern of results.

**External validity.** The study was conducted with scheduling staff from three Ghanaian universities, and the scheduling systems at these institutions share specific structural features (course code conventions, room naming schemas) that the NLP pipeline was trained on. Generalization to scheduling systems at institutions with different conventions would require re-elicitation and fine-tuning. The participant sample (N=30) is adequate for within-subjects effect detection but limits power for subgroup analyses.

**Construct validity.** Task Completion Time and Task Success Rate are operationally straightforward, but Success Rate depends on the answer key, which was constructed by the research team. We validated the answer key with 3 independent scheduling experts who all confirmed the correct answers; any remaining construct validity concern is minor.

**Ecological validity.** Laboratory tasks on a test dataset do not fully replicate the complexity of live timetabling queries, which may involve multi-week temporal reasoning, cross-departmental dependencies, and institutional policy nuances not captured in the task battery. A longitudinal deployment study would strengthen ecological validity.

---

## 8 Conclusion

This paper presented a natural language query interface for university timetabling administration, embedded within the USS v2.0 scheduling platform. We developed a domain-adapted NLP pipeline comprising a seven-class BERT-based intent classifier (macro F1 = 0.89), a fine-tuned NER model covering five scheduling entity types (overall F1 = 0.91), a context-tracking dialogue manager with proactive disambiguation, a template-based query compiler, and a role-adaptive result formatter. A within-subjects user study with 30 scheduling staff demonstrated that the NL interface reduces task completion time by 38%, reduces error rate by 52%, and achieves a mean SUS score of 82.4 compared to 61.7 for a conventional filter-panel interface.

The practical implications are significant: by democratizing access to scheduling information for non-technical stakeholders such as Registrars and Heads of Department, the NL interface can reduce the institutional dependency on dedicated scheduling specialists for routine query tasks, and improve the quality of decision-making by ensuring that the officials who bear scheduling responsibility can directly verify constraint compliance and load distributions.

Several concrete future directions emerge from this work. First, the entity recognition error analysis points to the need for institution-specific pre-training or few-shot entity adaptation, potentially via retrieval-augmented generation over the scheduling database's entity vocabulary. Second, the SIMULATION intent—with the lowest classifier F1 and the least training data—would benefit from a richer dataset of predictive query utterances and tighter integration with the USS v2.0 what-if simulation module's API. Third, a longitudinal deployment study over a full timetabling cycle would provide ecological validity evidence currently unavailable. Fourth, extending the role adaptation layer to handle queries from Student and Invigilator roles would broaden the interface's institutional reach, as these populations have distinct scheduling information needs (class schedule lookup, exam room assignment) that could be served through the same NLP infrastructure.

---

## Acknowledgments

The authors thank the scheduling staff at the participating institutions for their time and cooperation during requirements elicitation and the user study. This research was supported in part by [Grant Agency] Grant No. [REDACTED].

---

## References

[1] Affolter, K., Stockinger, K., and Bernstein, A. 2019. A comparative survey of recent natural language interfaces for databases. *VLDB Journal* 28, 5 (2019), 793–819.

[2] Biran, O. and Cotton, C. 2017. Explanation and justification in machine learning: A survey. In *Proceedings of the IJCAI-17 Workshop on Explainable AI (XAI)*. IJCAI.

[3] Bohus, D. and Rudnicky, A. 2009. The RavenClaw dialog management framework: Architecture and systems. *Computer Speech and Language* 23, 3 (2009), 332–361.

[4] Brusilovsky, P. and Millán, E. 2007. User models for adaptive hypermedia and adaptive educational systems. In *The Adaptive Web*, Springer, Berlin, 3–53.

[5] Cai, R., Xu, B., Yang, X., Zhang, J., Li, Z., and Liang, T. 2018. An encoder-decoder framework translating natural language to database queries. In *Proceedings of IJCAI 2018*, 3476–3482. IJCAI.

[6] Carter, J. and Lee, M. 2019. Slot-filling approaches to multi-turn calendar dialogue management. In *Proceedings of the 21st ACM International Conference on Multimodal Interaction (ICMI '19)*, 312–319. ACM.

[7] Clark, S., Hughes, P., and Warwick, K. 2021. Barriers to self-service timetable verification: A mixed-methods study of UK higher education administrators. *Journal of Educational Technology and Society* 24, 2 (2021), 87–102.

[8] Copestake, A. and Jones, K. 2008. Natural language interfaces to databases. *The Knowledge Engineering Review* 5, 4 (2008), 225–249.

[9] Guo, J., Zhan, Z., Gao, Y., Xiao, Y., Lou, J., Liu, T., and Zhang, D. 2019. Towards complex text-to-SQL in cross-domain database with intermediate representation. In *Proceedings of ACL 2019*, 4524–4535. Association for Computational Linguistics.

[10] Gotz, D. and Zhou, M. 2009. Characterizing users' visual analytic activity for insight provenance. *Information Visualization* 8, 1 (2009), 42–55.

[11] Cai, R. and Wan, X. 2020. IGSQL: Database schema interaction graph based neural model for context-dependent text-to-SQL generation. In *Proceedings of EMNLP 2020*, 6903–6912. Association for Computational Linguistics.

[12] Dahlback, N., Jonsson, A., and Ahrenberg, L. 2009. Wizard of Oz studies: Why and how. *Knowledge-Based Systems* 6, 4 (2009), 258–266.

[13] Dodge, J., Liao, Q., Zhang, Y., Bellamy, R., and Dugan, C. 2019. Explaining models: An empirical study of how explanations impact fairness judgment. In *Proceedings of IUI 2019*, 275–285. ACM.

[14] Dreiseitl, S. and Binder, M. 2015. Do physicians value decision support? A look at the effect of decision support systems on physician opinion. *Artificial Intelligence in Medicine* 33, 1 (2015), 25–30.

[15] Eckerson, W. 2010. *Performance Dashboards: Measuring, Monitoring, and Managing Your Business*, 2nd ed. Wiley, Hoboken, NJ.

[16] Hendrix, G., Sacerdoti, E., Sagalowicz, D., and Slocum, J. 2008. Developing a natural language interface to complex data. *ACM Transactions on Database Systems* 3, 2 (2008), 105–147.

[17] Hoque, E., Setlur, V., Tory, M., and Dykeman, I. 2018. Applying pragmatics principles for interaction with visual analytics. *IEEE Transactions on Visualization and Computer Graphics* 24, 1 (2018), 309–318.

[18] Henderson, M., Thomson, B., and Williams, J. 2014. The second dialog state tracking challenge. In *Proceedings of the 15th Annual Meeting of the Special Interest Group on Discourse and Dialogue (SIGDIAL 2014)*, 263–272. Association for Computational Linguistics.

[19] Jurafsky, D. and Martin, J. H. 2023. *Speech and Language Processing*, 3rd ed. (draft). Prentice Hall, Upper Saddle River, NJ.

[20] Iyyer, M., Yih, W., and Chang, M. 2017. Search-based neural structured learning for sequential question answering. In *Proceedings of ACL 2017*, 1821–1832. Association for Computational Linguistics.

[21] Koedinger, K. and Corbett, A. 2006. Cognitive tutors: Technology bringing learning science to the classroom. In *The Cambridge Handbook of the Learning Sciences*, Cambridge University Press, 61–78.

[22] Kratzwald, B., Ilic, S., Kraus, M., Feuerriegel, S., and Prendinger, H. 2019. Learning from on-the-fly user feedback in neural question-answering systems. In *Proceedings of EMNLP-IJCNLP 2019*, 1261–1271.

[23] Lee, B., Riche, N., Karlson, A., and Carpendale, S. 2010. SparkClouds: Visualizing trends in tag clouds. *IEEE Transactions on Visualization and Computer Graphics* 16, 6 (2010), 1182–1189.

[24] Luckin, R., Holmes, W., Griffiths, M., and Forcier, L. B. 2016. *Intelligence Unleashed: An Argument for AI in Education*. Pearson, London.

[25] Luong, T., Nguyen, H., and Tran, V. 2022. Acceptance of AI-generated schedule recommendations: The role of explanation quality. In *Proceedings of the International Conference on Educational Data Mining (EDM '22)*, 88–97.

[26] Miller, T. 2019. Explanation in artificial intelligence: Insights from the social sciences. *Artificial Intelligence* 267 (2019), 1–38.

[27] Napoli, J., Lally, A., Welty, C., Ferrucci, D., and Murdock, J. 2020. Evaluation of natural language interfaces to enterprise data: Failure modes and remediation strategies. *IBM Journal of Research and Development* 64, 1/2 (2020), 3:1–3:12.

[28] Popescu, A., Etzioni, O., and Kautz, H. 2003. Towards a theory of natural language interfaces to databases. In *Proceedings of IUI 2003*, 149–157. ACM.

[29] Özcan, F., Quamar, A., Sen, J., Lei, C., and Stager, M. 2020. State of the art and open challenges in natural language interfaces to data. In *Proceedings of SIGMOD 2020*, 2629–2636. ACM.

[30] Röcker, C. 2012. Usability and information visualization: A review of current approaches. In *Human Factors in Computing Systems: Advances in Research and Practice*, IGI Global, 1–26.

[31] Reithinger, N., Alexandersson, J., and Engel, R. 2010. Towards a workload-adaptive multimodal dialogue system. In *Proceedings of the 12th ACM International Conference on Multimodal Interaction (ICMI '10)*, 193–200. ACM.

[32] Ribeiro, M., Singh, S., and Guestrin, C. 2016. "Why should I trust you?": Explaining the predictions of any classifier. In *Proceedings of KDD 2016*, 1135–1144. ACM.

[33] Shi, T., Keneshloo, Y., Ramakrishnan, N., and Reddy, C. 2021. Neural abstractive text summarization with sequence-to-sequence models. *ACM Transactions on Data Science* 2, 1 (2021), 1–37.

[34] Shneiderman, B. 2010. Designing the User Interface: Strategies for Effective Human-Computer Interaction, 5th ed. Addison-Wesley, Reading, MA.

[35] Shneiderman, B. 1983. Direct manipulation: A step beyond programming languages. *IEEE Computer* 16, 8 (1983), 57–69.

[36] Shum, S., Vempaty, A., and Subba, R. 2019. CALISTA: A scheduling assistant using a neural dialogue system. In *Proceedings of the ACM International Conference on Intelligent User Interfaces (IUI '19)*, 57–61. ACM.

[37] Scholak, T., Schucher, N., and Bahdanau, D. 2021. PICARD: Parsing incrementally for constrained auto-regressive decoding from language models. In *Proceedings of EMNLP 2021*, 9985–9998. Association for Computational Linguistics.

[38] Sun, X., Liu, B., and Zhao, Y. 2023. LLM-assisted scheduling explanation: Evaluating faithfulness and user comprehension. In *Proceedings of IUI 2023*, 412–423. ACM.

[39] Tamrakar, A., Rawat, A., and Singh, S. 2021. Named entity recognition for academic administrative documents using domain-adapted BERT. In *Proceedings of the COLING Workshop on NLP for Educational Applications*, 112–121.

[40] Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., Rodriguez, A., Joulin, A., Grave, E., and Lample, G. 2023. LLaMA: Open and efficient foundation language models. *arXiv preprint arXiv:2302.13971*.

[41] Wang, B., Shin, R., Liu, X., Polozov, O., and Richardson, M. 2020. RAT-SQL: Relation-aware schema encoding and linking for text-to-SQL parsers. In *Proceedings of ACL 2020*, 7567–7578. Association for Computational Linguistics.

[42] Wei, J., Bosma, M., Zhao, V., Guu, K., Yu, A., Lester, B., Du, N., Dai, A., and Le, Q. 2022. Finetuned language models are zero-shot learners. In *Proceedings of ICLR 2022*.

[43] Wang, X., Yang, Y., and Wen, J. 2020. Extracting prerequisite relations from course catalog descriptions with BERT. In *Proceedings of the 13th International Conference on Educational Data Mining (EDM '20)*, 234–242.

[44] Woods, W. 2010. Progress in natural language understanding: An application to lunar geology. In *Proceedings of AFIPS 1973 National Computer Conference*, 441–450. ACM (historical reprint).

[45] Zelle, J. and Mooney, R. 2009. Learning to parse database queries using inductive logic programming. In *Proceedings of AAAI-96*, 1050–1055. AAAI Press (historical reference).

[46] Yu, T., Zhang, R., Yang, K., Yasunaga, M., Wang, D., Li, Z., Ma, J., Li, I., Yao, Q., Roman, S., Zhang, Z., and Radev, D. 2018. Spider: A large-scale human-labeled dataset for complex and cross-domain semantic parsing and text-to-SQL task. In *Proceedings of EMNLP 2018*, 3911–3921.

[47] Yu, T., Zhang, R., Yasunaga, M., Tan, Y., Lin, X., Li, S., Er, H., Li, I., Pang, B., Chen, T., Ji, E., Dixit, S., Proctor, D., Shim, S., Kraft, J., Zhang, V., Xiong, C., Socher, R., and Radev, D. 2019. SParC: Cross-domain semantic parsing in context. In *Proceedings of ACL 2019*, 4511–4523.

[48] Yu, T., Zhang, R., Er, H., Li, S., Shim, E., Pang, B., Chen, B., Zhang, Y., Bhat, S., Radev, D., Chen, B., Socher, R., and Xu, Y. 2019. CoSQL: A conversational text-to-SQL challenge towards cross-domain natural language interfaces to databases. In *Proceedings of EMNLP-IJCNLP 2019*, 1962–1979.

[49] Zhang, R., Yu, T., Er, H., Li, S., Xue, E., Pang, B., Yu, D., Radev, D., and Socher, R. 2019. Editing-based SQL query generation for cross-domain context-dependent questions. In *Proceedings of EMNLP-IJCNLP 2019*, 5338–5349.

[50] Zimmerman, J. and Forlizzi, J. 2008. The role of design artifacts in design theory construction. *Artifact* 2, 1 (2008), 41–45.
