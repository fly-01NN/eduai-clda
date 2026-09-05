# DE-004 frozen protocol

## Frozen specifications

- Protocol version: `2026-08-31.5`
- Collection specification: `2026-08-31.2`
- Deterministic parser and paired-analysis hardening: `2026-08-31.5`
- Snapshot date: `2026-08-31`
- Unit: public Hugging Face Space
- Design: bounded global cross-sectional audit
- Manual relevance labels: none
- Manual dependency, namespace, clone, and history labels: none
- Human participants: none

These are versioned frozen specifications, not a claim of prospective preregistration. Version `.5` retains the `.2` search responses and fixed repository revisions. It adds deterministic safeguards identified during pre-submission review, including runtime-only manifest attribution, a separate machine-candidate observability frame, and CamelCase boundary normalization for functional phrases such as `StudyAssistant` and `LessonPlanGenerator`.

## Discovery

For each of 15 functional queries, request at most 100 results sorted by likes and 100 sorted by descending creation date. De-duplicate the union by Space ID. The terms are frozen in `src/protocol.py` and cover tutors, education chatbots, quiz/course/lesson generation, teaching and study assistants, homework, grading, learning companions, and language tutors.

The discovery union is not a probability sample. Results beyond each cutoff, private projects, projects on other hosts, and applications described outside the query vocabulary are outside the estimand.

## Inclusion

Education relevance and dependency observability are recorded separately. The strict education phrase frame requires only a deterministic strict functional phrase in the Space slug, title, description, tags, or README. Author and organization namespaces cannot trigger relevance.

The dependency-observable broad frame requires:

1. a deterministic broad education phrase; and
2. at least one linked model, identifiable inference-service or user-managed-runtime signal, or frozen machine-extracted service candidate.

The primary project frame is the intersection of the strict education phrase frame and the dependency-observable frame. Provider HHI uses only identifiable provider edges; unresolved candidates enter only the explicit unknown-service bounds. The final service-versus-model contrast further conditions on projects exposing both identifiable layers. Generic references to students, schools, courses, or assessment do not satisfy the strict rule. `Tutor` and `tutorial` are explicitly separated. All three counts are retained so relevance, dependency observability, and provider identifiability cannot be conflated. No manual adjudication can add or remove a project.

## Dependency evidence

Provider signals are accepted only from structured dependency manifests or executable-code contexts. Comments and Python docstrings are removed before matching, while runtime string literals remain available. Test, checkpoint, generated, vendored, and package-cache paths are excluded mechanically.

Only runtime/project dependency groups can create provider edges. `requirements-dev` files, Python optional or dependency groups, Poetry development groups, and npm development, peer, or optional dependencies remain machine-readable QA candidates but cannot enter the primary provider taxonomy.

- exact mapped package names;
- provider-specific imports;
- credential-variable names;
- provider API endpoints;
- official Hub application-to-model links; or
- model identifiers in bounded loading contexts.

README prose can support discovery and GitHub-link extraction but cannot create a provider edge. Official Hub model links, manifests, and executable signatures are high confidence. `.env.example` and code-extracted model IDs are medium confidence.

Unrecognized manifest packages, credential names, and API-like domains are retained in a separate candidate table. They do not become provider edges. Service concentration is additionally bracketed by pooling these candidates, separating their machine identifiers, and assigning project-unique unknown categories.

The `openai` and `langchain-openai` packages and imports are not provider attribution because both clients accept third-party base URLs. OpenAI is recorded only for its official endpoint, a provider-specific credential without a base-URL override, or an unoverridden `OpenAI`/`AsyncOpenAI` constructor. Groq, OpenRouter, Azure, DeepSeek, or other explicit base URLs suppress a simultaneous OpenAI edge. An otherwise unresolved compatible client remains a machine candidate.

Model provider attribution prefers a recognized declared base-model namespace, then a recognized linked-model namespace. Unmapped namespaces remain `namespace:<public-name>`. Hosting region is not a provider attribution and never becomes developer geography.

## Concentration

For project `i` and provider set `P_i`, every detected provider receives `1 / |P_i|`. Each eligible project therefore contributes one total unit. Report:

- fractional provider shares;
- HHI and inverse-HHI effective count;
- top-one and top-three shares;
- Gini as a secondary distribution statistic;
- top-category share and Shannon effective category count as metric-sensitivity summaries; and
- 2,000 project-bootstrap draws with seed `20260831` for project-equal variants.

Primary layers are inference service, user-managed runtime, and model dependency. Hub SDK and generic cloud-package signals are retained as evidence where present but are not silently treated as active inference providers.

The primary layer contrast is repeated on the identical set of projects exposing both an inference-service and a model-dependency signal. The same project indices are used for both layers in each bootstrap draw. The estimand is the service-minus-model HHI difference in this matched project set; the bootstrap dominance fraction is a descriptive stability measure, not a p value.

Within the matched set, each project's fractional service vector is multiplied by its fractional model vector to form a project-equal co-declaration matrix. Mutual information summarizes association between the two declared layers. Ten thousand model-row permutations preserve both marginal distributions while breaking project pairing. This analysis measures structured co-declaration, not routing, API traffic, or a service-to-model technical link.

## Sensitivity analyses

- broad inclusion;
- liked-results arm;
- newest-results arm;
- likes-plus-one project weighting;
- high-confidence evidence only;
- official-Hub-linked models only;
- exact multi-file source-cluster weighting;
- provider-only concentration excluding the non-company user-managed-runtime category;
- unresolved model namespaces disaggregated, pooled into one category, and excluded;
- rank cutoffs of 25, 50, 75, and 100 within every frozen query arm;
- leave-one-query-out re-estimation for all 15 discovery terms;
- automatic near-duplicate source weighting at Jaccard thresholds 0.85, 0.90, and 0.95;
- author-cluster weighting and immediate public model-namespace attribution; and
- leave-one-service-provider-out re-estimation that removes every project declaring the focal provider from both layers.
- same-revision service detection with source-file caps of 10, 15, and 20.

Every search-depth, leave-one-query-out, evidence, weighting, source-reuse, and namespace sensitivity is repeated on the same service-minus-model HHI estimand. Each row reports its own matched-project count and paired bootstrap interval.

The provider-omission analysis asks whether one service ecosystem mechanically creates the cross-layer difference. Each variant removes all projects declaring one focal inference-service provider before reconstructing both layer matrices. Removing only the provider edge while retaining the same project in the model layer is not permitted.

The source-cap sensitivity keeps repository revisions and the model layer fixed while widening only the bounded service-source scan. It therefore isolates missed service declarations from calendar drift or a simultaneous change in model attribution.

## Independent measurement validation

A deterministic, blinded review pack is sampled after the frozen analysis and cannot change its projects or edges. It contains four relevance strata from the complete query union, five dependency-evidence strata spanning high- and medium-confidence rules, and a negative audit of strict projects without an identifiable service edge. Sampling probabilities and automatic decisions remain in separate key files until two reviewers label the public evidence independently. This validation evaluates measurement precision and bounded false negatives; it is not manual adjudication of the analytical frame.

Exact source clusters require at least two identical non-README, non-licence file digests. Single-file bundles remain unique. This check does not detect modified forks or tutorial ancestry.

The near-duplicate extension uses cached executable files only. String literals, comments, letter case, and numeric values are normalized mechanically; seven-token shingles are hashed; projects with at least 50 unique shingles are compared by Jaccard similarity; and threshold-qualified pairs form deterministic single-linkage clusters. The 0.90 threshold is the primary extension and 0.85/0.95 bracket it. These are algorithmic similarity clusters, not human-validated clone labels.

The release contains only irreversible 64-bit shingle hashes and coverage metadata, not source bodies. These inputs reproduce the clustering offline.

Rank-cutoff and leave-one-query-out analyses reuse inclusion and dependency rules already applied to the full frozen candidate union. They evaluate dependence on returned rank depth and individual query terms inside the frozen design. They do not correct search ranking or estimate population coverage.

## Version-history extension

For every strict-sample project, the public commit list is truncated at the end of the snapshot date. Starting from the oldest commit, the collector selects the first revision containing at least one machine-selected executable file or dependency manifest under the same bounded repository-file rules. That revision is compared with the frozen current revision for source-derived inference-service and user-managed-runtime signals. No commit, file, or transition is manually classified.

Version transitions are `unchanged_same`, `unchanged_no_signal`, `added`, `removed`, or `changed` by exact provider-set comparison. Service HHI change is estimated only among projects with a service signal in both versions, using identical project resamples. Sensitivity subsets require minimum observed repository-history spans of 0, 1, 7, 30, 90, or 180 days.

This is a version-paired audit conditioned on projects visible in the final strict sample. Project ages differ, deleted projects are absent, and most histories are short. It is not a common-calendar panel, an upstream-event study, or a survival analysis.

## Licence rules

Application licences come from card metadata or high-precision licence-text fingerprints; model licences come from model cards. Automated statuses are review triage only. They cannot establish infringement, compatibility, enforceability, or legal compliance. Missing is never recoded as proprietary.

## Evidence gates

- Developer-region comparison requires explicit-country coverage of at least 20%.
- Language-orientation comparison requires declared-language coverage of at least 20%.
- Hosting region, names, cities, and language cues cannot fill missing geography.
- Upstream-event survival analysis requires repeated dependency snapshots and dated upstream events.

Failed gates must be reported as `NOT_ESTIMABLE` or `NOT_TESTED`; they cannot be converted into negative findings.

## Supplementary GitHub connector

Every syntactically valid GitHub repository link extracted from primary-sample READMEs is followed automatically. Root and one `src/` level are parsed with the same static rules. Because a README link may denote a mirror, dependency, example, template, or placeholder, these edges remain in a separate supplementary table unless a future validated relation classifier is introduced.
