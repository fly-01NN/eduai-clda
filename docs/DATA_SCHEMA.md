# Data contracts

## Frozen analytical inputs

The public inputs under `data/processed/` retain the minimum fields required to
audit the frozen frame and rerun the analyses.

| File | Unit | Main role |
|---|---|---|
| `search_rankings.csv` | query–arm–rank–project | Reconstructs the bounded discovery union and rank-cutoff analyses |
| `education_phrase_frame.csv` | project | Records strict and broad educational-function classification before the dependency gate |
| `space_frame.csv` | project | Stores the final rule flags, metadata, observability fields, and documentation outcomes |
| `dependency_edges.csv` | project–provider evidence | Represents declared service, local-runtime, and model-reference evidence |
| `model_frame.csv` | model identifier | Stores public model metadata, publisher or namespace mapping, family, and licence field |
| `source_clone_clusters.csv` | project | Supplies exact source-bundle cluster membership and project weights |
| `source_similarity_clusters.csv` | threshold–project | Supplies near-duplicate cluster membership and weights |
| `source_shingles.json.gz` | project | Stores irreversible token-shingle hashes for source-similarity reconstruction |
| `historical_dependency_edges.csv` | project–revision–provider | Records earliest-analyzable service and runtime evidence |
| `space_github_links.csv` | project–repository | Records public README-linked GitHub repositories |
| `github_repository_frame.csv` | linked repository | Summarizes the supplementary linked-repository collection |
| `github_dependency_edges.csv` | project–repository–provider | Stores supplementary linked-repository evidence excluded from the primary estimand |

The principal edge key is the combination of `space_id`, `provider`, `layer`,
`evidence_type`, `evidence_value`, and `source_file`. The `layer` field separates
`inference_service`, `local_runtime`, and `model_dependency`. Project-equal
fractional weights are computed during analysis rather than stored as repeated
source occurrences.

## Independent-review data

`analysis_results/measurement_validation_deidentified.csv` contains one row per
reviewed item and only the following analytical fields:

| Field | Meaning |
|---|---|
| `task` | Education relevance, positive dependency evidence, or negative service audit |
| `sample_id` | Opaque task-specific identifier |
| `validation_stratum` | Prespecified sampling stratum |
| `inclusion_probability` | Probability used for inverse-probability weighting |
| `automatic_positive` | Frozen rule output relevant to the task |
| `reviewer1_label`, `reviewer2_label` | Independently assigned blinded labels |
| `third_review_label` | Third label used where the first two did not resolve the item |
| `resolved_label` | Final yes/no analytical label |
| `candidate_boundary` | Whether a negative-audit item was already represented by the machine candidate boundary; blank for other tasks |

The file contains no project identifier, repository URL, reviewer identity, or
source excerpt. `measurement_validation.json` and
`measurement_validation_by_stratum.csv` contain the corresponding estimates
and resolved counts.

## QA and provenance records

Files under `data/qa/` record the public revision, file path, status, byte count,
retrieval time, and SHA-256 digest needed to audit collection and parsing without
redistributing source bodies. They also retain the candidate-selection audit,
source-similarity coverage, collection gates, transport-status counts, and the
68-check final statistical audit.

## Numerical results and figure sources

`analysis_results/` contains the paper-facing concentration summaries,
provider rankings, same-project bootstrap draws, co-declaration matrix and
permutation draws, robustness variants, version-history outputs, licence
summaries, and validation estimates. Random seeds and draw counts are stored in
the corresponding CSV or JSON files.

`data/figure_source/` contains the author-created long-form data used by each
published figure. Complete final vectors are stored in `figures/`.

| Manuscript figure | Released source table |
|---|---|
| Figure 1 | `figure1_study_design_source_data.csv` |
| Figure 2 | `figure2_signal_intersections_source_data.csv` |
| Figure 3 | `figure3_service_model_composition_source_data.csv` |
| Figure 4 | `figure4_cross_layer_association_source_data.csv` |
| Figure 5 | `figure5_matched_robustness_source_data.csv` |

The plotting command writes descriptively named working exports under
`outputs/`; the checked-in publication vectors use `Figure_1` through
`Figure_5`.
