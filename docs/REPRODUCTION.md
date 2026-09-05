# Reproduction guide

## Environment and integrity

The public workflow requires Python 3.11 or later. From the repository root:

```bash
uv sync --frozen --group dev
sha256sum -c MANIFEST.sha256
uv run pytest -q
```

The release contains 76 deterministic tests. Tests use synthetic or temporary
inputs and do not contact a live service.

## Offline result reproduction

The frozen processed data and irreversible shingle hashes are sufficient for
the complete derived-data analysis:

```bash
uv run python src/analyze_dependencies.py --root .
```

This command rewrites the numerical files in `analysis_results/` from the
released inputs. It does not fetch source files and does not generate a
manuscript-side TeX file unless the legacy `--write-tex` option is requested.

The independent-review estimates can be reproduced separately from the
de-identified public labels:

```bash
uv run python src/reproduce_validation.py --root .
```

The command recalculates weighted precision and omission estimates, bootstrap
intervals, agreement statistics, and the candidate-boundary check. It exits
with an error if the recomputed values differ from the released reference.

Finally, run the structural and numerical audit:

```bash
uv run python src/audit_results.py --root .
```

A successful run reports `"status": "PASS"` and verifies 68 hard checks,
including the matched denominators, concentration reconstruction, sensitivity
families, evidence gates, validation completion, and scope boundaries.

## Figure reproduction

```bash
uv run python src/make_figures.py --root .
```

The plotting script uses `analysis_results/`, `data/processed/`, and `data/qa/`
and writes fresh exports for Figures 2--5 to `outputs/`. Figure 1 is supplied as
an editable direct SVG rather than a code-generated chart. Final paper-facing
PDF and SVG files are retained in `figures/`, while the exact input records used
by each figure are in `data/figure_source/`.

Figure 4 reports Shannon effective counts, not reciprocals of HHI, and
mutual information uses natural logarithms (nats). The HHI intervals for each
layer are marginal percentile intervals from the paired project draws.
In Figure 5, points carry paired 95% intervals from resampling projects; capped lines
without points summarize ranges of point estimates across related settings.
The two unknown-service rows use `machine_identifier_categories` and
`project_unique_unknown_bound` from `unknown_service_boundary.csv`. The row
pooling unmapped model namespaces uses `pooled_unmapped_namespaces` from
`matched_robustness.csv`, retaining all 99 primary matched projects.

The final row excludes every project declaring Hugging Face as an inference
service from both layers; it does not merely remove that service's edges.

## Manuscript methods and implementation

The manuscript describes the measurement rules in Methods. The corresponding
implementation is included here for inspection and reproduction:

- `src/protocol.py` defines the exact search queries, provider rules, and
  publisher and namespace mappings for frozen protocol `2026-08-31.5`.
- `src/license_rules.py` normalizes license metadata and assigns the ordered,
  mutually exclusive documentation review outcomes. Missing application terms
  take precedence over an absence of model references. The remaining rules
  distinguish unresolved upstream terms, terms requiring review, and records
  with no automated flag; they do not determine legal compatibility.

The `.env.example` filename in the manuscript identifies a type of observed
configuration evidence, not an implementation file in this repository.

## Collection-to-analysis reconstruction

The full scientific workflow follows this dependency order:

1. collect the two ranked search arms for each frozen query;
2. apply the strict educational-function and dependency-observability rules;
3. retrieve bounded files at pinned public revisions and derive dependency
   evidence without retaining source bodies in the public release;
4. collect model metadata, linked-repository evidence, and version histories;
5. derive source-reuse clusters, same-project matrices, concentration measures,
   co-declaration statistics, and sensitivity variants;
6. complete the independent blinded validation outside the public repository;
7. release only the de-identified validation panel; and
8. run the final audit and regenerate the figures.

The relevant collection scripts are named by these operations in `src/`. Before
running any network stage, inspect its `--help` output, check current provider
terms, and set a monitored research user agent:

```bash
export EDUAI_CLDA_USER_AGENT="EduAI-CLDA/0.1.0 (research; contact: you@example.org)"
```

The archived manuscript results use protocol version `2026-08-31.5` and the
2026-08-31 snapshot. A refreshed collection must be versioned separately.
