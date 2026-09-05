"""Run structural and numerical checks on the frozen DE-004 result package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from protocol import PROTOCOL_VERSION


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_audit(root: Path) -> dict:
    processed = root / "data/processed"
    qa = root / "data/qa"
    results = root / "analysis_results"
    required = [
        processed / "space_frame.csv",
        processed / "dependency_edges.csv",
        processed / "model_frame.csv",
        processed / "source_clone_clusters.csv",
        processed / "source_similarity_clusters.csv",
        processed / "source_shingles.json.gz",
        processed / "search_rankings.csv",
        processed / "education_phrase_frame.csv",
        processed / "historical_dependency_edges.csv",
        qa / "candidate_selection_audit.csv",
        qa / "collection_manifest.json",
        qa / "source_file_manifest.csv",
        qa / "source_similarity_coverage.csv",
        qa / "source_similarity_pairs.csv",
        qa / "space_commit_history_audit.csv",
        qa / "data_capture_audit.json",
        qa / "unmapped_dependency_candidates.csv",
        qa / "file_cap_source_manifest.csv",
        qa / "file_cap_additional_service_edges.csv",
        qa / "measurement_validation_status.json",
        results / "analysis_manifest.json",
        results / "provider_rankings.csv",
        results / "concentration_summary.csv",
        results / "robustness_matrix.csv",
        results / "license_summary.csv",
        results / "evidence_source_audit.json",
        results / "decision_gate.json",
        results / "regional_analysis_gate.json",
        results / "paired_layer_comparison.json",
        results / "paired_layer_bootstrap.csv",
        results / "cross_layer_codeclaration_summary.json",
        results / "cross_layer_codeclaration.csv",
        results / "cross_layer_service_model_diversity.csv",
        results / "cross_layer_codeclaration_null.csv",
        results / "model_namespace_sensitivity.csv",
        results / "search_rank_cutoff_sensitivity.csv",
        results / "search_query_ablation.csv",
        results / "search_arm_overlap.json",
        results / "source_similarity_sensitivity.csv",
        results / "matched_robustness.csv",
        results / "service_provider_omission.csv",
        results / "matched_sample_composition.csv",
        results / "unknown_service_boundary.csv",
        results / "file_cap_sensitivity.csv",
        results / "selection_funnel.csv",
        results / "version_history_summary.json",
        results / "version_dependency_transitions.csv",
        results / "version_service_hhi_bootstrap.csv",
        results / "version_span_sensitivity.csv",
        results / "measurement_validation.json",
        results / "measurement_validation_by_stratum.csv",
        results / "measurement_validation_deidentified.csv",
    ]
    missing = [path.relative_to(root.parent).as_posix() for path in required if not path.exists()]
    if missing:
        return {
            "status": "FAIL",
            "missing_required_files": missing,
            "hard_checks": {"required_files_present": False},
        }

    spaces = pd.read_csv(processed / "space_frame.csv")
    edges = pd.read_csv(processed / "dependency_edges.csv")
    models = pd.read_csv(processed / "model_frame.csv")
    clusters = pd.read_csv(processed / "source_clone_clusters.csv")
    similarity_clusters = pd.read_csv(processed / "source_similarity_clusters.csv")
    historical_edges = pd.read_csv(processed / "historical_dependency_edges.csv")
    candidates = pd.read_csv(qa / "candidate_selection_audit.csv")
    collection = read_json(qa / "collection_manifest.json")
    analysis = read_json(results / "analysis_manifest.json")
    rankings = pd.read_csv(results / "provider_rankings.csv")
    concentration = pd.read_csv(results / "concentration_summary.csv")
    decision = read_json(results / "decision_gate.json")
    region = read_json(results / "regional_analysis_gate.json")
    paired = read_json(results / "paired_layer_comparison.json")
    paired_draws = pd.read_csv(results / "paired_layer_bootstrap.csv")
    codeclaration = read_json(results / "cross_layer_codeclaration_summary.json")
    codeclaration_joint = pd.read_csv(results / "cross_layer_codeclaration.csv")
    codeclaration_service = pd.read_csv(
        results / "cross_layer_service_model_diversity.csv"
    )
    codeclaration_null = pd.read_csv(results / "cross_layer_codeclaration_null.csv")
    namespaces = pd.read_csv(results / "model_namespace_sensitivity.csv")
    cutoffs = pd.read_csv(results / "search_rank_cutoff_sensitivity.csv")
    ablations = pd.read_csv(results / "search_query_ablation.csv")
    similarity_results = pd.read_csv(results / "source_similarity_sensitivity.csv")
    history = read_json(results / "version_history_summary.json")
    history_audit = pd.read_csv(qa / "space_commit_history_audit.csv")
    transitions = pd.read_csv(results / "version_dependency_transitions.csv")
    history_draws = pd.read_csv(results / "version_service_hhi_bootstrap.csv")
    span_sensitivity = pd.read_csv(results / "version_span_sensitivity.csv")
    data_capture = read_json(qa / "data_capture_audit.json")
    matched_robustness = pd.read_csv(results / "matched_robustness.csv")
    provider_omission = pd.read_csv(results / "service_provider_omission.csv")
    matched_composition = pd.read_csv(results / "matched_sample_composition.csv")
    unknown_service = pd.read_csv(results / "unknown_service_boundary.csv")
    file_cap = pd.read_csv(results / "file_cap_sensitivity.csv")
    license_summary = pd.read_csv(results / "license_summary.csv")
    evidence_source = read_json(results / "evidence_source_audit.json")
    phrase_frame = pd.read_csv(processed / "education_phrase_frame.csv")
    unmapped_candidates = pd.read_csv(qa / "unmapped_dependency_candidates.csv")
    selection_funnel = pd.read_csv(results / "selection_funnel.csv")
    measurement_validation = read_json(results / "measurement_validation.json")
    measurement_validation_status = read_json(
        qa / "measurement_validation_status.json"
    )
    validation_deidentified = pd.read_csv(
        results / "measurement_validation_deidentified.csv"
    )

    main = spaces[spaces["included_strict"].eq(True)]
    main_ids = set(main["space_id"])
    ranking_sums = rankings.groupby("analysis")["fractional_share"].sum()
    summary_hhi = concentration.set_index("analysis")["hhi"]
    recomputed_hhi = rankings.groupby("analysis")["fractional_share"].apply(
        lambda values: float(np.square(values).sum())
    )
    hhi_aligned = all(
        np.isclose(recomputed_hhi[name], summary_hhi[name], atol=1e-12)
        for name in recomputed_hhi.index
    )
    missing_locations = main["author_location"].fillna("").astype(str).eq("")
    inferred_regions_from_missing = main.loc[
        missing_locations, "author_region_class"
    ].isin(["asia", "outside_asia"]).any()
    linked_edges = edges[edges["evidence_type"].eq("hf_linked_model")]
    code_model_edges = edges[edges["evidence_type"].eq("code_model_id")]
    environment_edges = edges[
        edges["layer"].eq("inference_service")
        & edges["source_file"].fillna("").astype(str).str.casefold().str.endswith(
            ".env.example"
        )
    ]
    combined_top = rankings[rankings["analysis"].eq("combined_primary")].iloc[0]["provider"]
    high_top = rankings[rankings["analysis"].eq("combined_high_confidence")].iloc[0]["provider"]
    clone_top = rankings[rankings["analysis"].eq("combined_clone_adjusted")].iloc[0]["provider"]
    near_top = rankings[
        rankings["analysis"].eq("combined_near_duplicate_090")
    ].iloc[0]["provider"]
    matched_service_ids = set(
        edges.loc[
            edges["space_id"].isin(main_ids)
            & edges["layer"].eq("inference_service"),
            "space_id",
        ]
    )
    matched_model_ids = set(
        edges.loc[
            edges["space_id"].isin(main_ids)
            & edges["layer"].eq("model_dependency"),
            "space_id",
        ]
    )
    paired_difference = float(paired["service_hhi"]) - float(paired["model_hhi"])
    cutoff_combined = cutoffs[cutoffs["analysis"].eq("combined")]
    ablation_combined = ablations[ablations["analysis"].eq("combined")]
    primary_similarity = similarity_results[
        np.isclose(similarity_results["threshold"], 0.90)
    ].iloc[0]
    history_change = float(
        history["paired_service_concentration"]["hhi_change_current_minus_initial"]
    )
    cutoff_matched = cutoffs[cutoffs["analysis"].eq("matched_service_minus_model")].set_index(
        "rank_cutoff"
    )["hhi"]
    robustness_cutoff = matched_robustness[
        matched_robustness["robustness_family"].eq("rank_cutoff")
    ].assign(variant=lambda frame: frame["variant"].astype(int)).set_index("variant")[
        "hhi_difference_service_minus_model"
    ]
    ablation_matched = ablations[
        ablations["analysis"].eq("matched_service_minus_model")
    ].set_index("excluded_query")["hhi"]
    robustness_ablation = matched_robustness[
        matched_robustness["robustness_family"].eq("leave_one_query_out")
    ].set_index("variant")["hhi_difference_service_minus_model"]
    service_candidate_ids = set(
        unmapped_candidates.loc[
            unmapped_candidates["candidate_type"].isin(
                {
                    "unmapped_credential",
                    "unmapped_api_domain",
                    "openai_compatible_provider_unresolved",
                }
            ),
            "space_id",
        ].astype(str)
    )
    funnel_expected = {
        "query_union": len(candidates),
        "strict_education_phrase": len(phrase_frame),
        "dependency_observable": len(main),
        "identifiable_service_model_matched": int(paired["matched_projects"]),
    }
    funnel_observed = dict(
        zip(selection_funnel["stage"], selection_funnel["projects"])
    )

    checks = {
        "required_files_present": True,
        "protocol_version_matches": collection.get("protocol_version") == PROTOCOL_VERSION,
        "data_capture_gate_passes": data_capture.get("gate_status") == "PASS",
        "candidate_ids_unique": bool(candidates["space_id"].is_unique),
        "selected_space_ids_unique": bool(spaces["space_id"].is_unique),
        "strict_sample_count_matches_collection": len(main)
        == int(collection["strict_education_ai_spaces"]),
        "strict_sample_count_matches_analysis": len(main)
        == int(analysis["strict_primary_projects"]),
        "strict_phrase_frame_matches_collection": len(phrase_frame)
        == int(collection["strict_education_phrase_spaces"]),
        "all_edges_reference_selected_spaces": set(edges["space_id"]) <= set(spaces["space_id"]),
        "dependency_frame_has_edge_or_machine_service_candidate": main_ids
        <= (set(edges["space_id"].astype(str)) | service_candidate_ids),
        "candidate_only_selection_is_explicit": int(
            analysis["machine_candidate_only_projects"]
        )
        == int(
            (
                ~main["known_dependency_observable"].fillna(False).astype(bool)
                & main["machine_service_candidate_observable"]
                .fillna(False)
                .astype(bool)
            ).sum()
        ),
        "selection_funnel_has_three_separate_construct_frames": bool(
            selection_funnel["stage"].tolist()
            == [
                "query_union",
                "strict_education_phrase",
                "dependency_observable",
                "identifiable_service_model_matched",
            ]
            and funnel_observed == funnel_expected
        ),
        "dependency_rows_unique": not edges.duplicated().any(),
        "model_ids_unique": bool(models["model_id"].is_unique),
        "clone_assignments_unique": bool(clusters["space_id"].is_unique),
        "clone_weights_positive_and_bounded": bool(
            clusters["source_cluster_weight"].gt(0).all()
            and clusters["source_cluster_weight"].le(1).all()
        ),
        "similarity_assignments_cover_each_threshold": bool(
            set(np.round(similarity_clusters["threshold"], 2)) == {0.85, 0.90, 0.95}
            and similarity_clusters.groupby("threshold")["space_id"].nunique().eq(len(main)).all()
            and set(similarity_clusters["space_id"]) == main_ids
        ),
        "similarity_weights_positive_and_bounded": bool(
            similarity_clusters["source_similarity_cluster_weight"].gt(0).all()
            and similarity_clusters["source_similarity_cluster_weight"].le(1).all()
        ),
        "near_duplicate_count_matches_manifest": int(
            analysis["near_duplicate_clustered_spaces_090"]
        ) == int(primary_similarity["clustered_projects"]),
        "official_linked_models_high_confidence": bool(
            linked_edges.empty or linked_edges["confidence"].eq("high").all()
        ),
        "code_model_ids_are_explicit_medium_evidence": bool(
            code_model_edges.empty or code_model_edges["confidence"].eq("medium").all()
        ),
        "environment_examples_are_explicit_medium_evidence": bool(
            environment_edges.empty or environment_edges["confidence"].eq("medium").all()
        ),
        "evidence_source_audit_reconstructs": bool(
            int(evidence_source["official_linked_model_edge_rows"])
            == len(linked_edges[linked_edges["space_id"].isin(main_ids)])
            and int(evidence_source["code_model_id_edge_rows"])
            == len(code_model_edges[code_model_edges["space_id"].isin(main_ids)])
            and int(evidence_source["environment_example_service_edge_rows"])
            == len(environment_edges[environment_edges["space_id"].isin(main_ids)])
        ),
        "model_license_denominators_match_evidence_sets": bool(
            int(
                license_summary.loc[
                    license_summary["measure"].eq(
                        "model_reference_license_disclosure"
                    ),
                    "denominator",
                ].iloc[0]
            )
            == edges.loc[
                edges["space_id"].isin(main_ids)
                & edges["layer"].eq("model_dependency"),
                "evidence_value",
            ].nunique()
            and int(
                license_summary.loc[
                    license_summary["measure"].eq(
                        "official_linked_model_license_disclosure"
                    ),
                    "denominator",
                ].iloc[0]
            )
            == linked_edges.loc[
                linked_edges["space_id"].isin(main_ids), "evidence_value"
            ].nunique()
        ),
        "no_model_dependency_rights_status_is_literal": bool(
            int(
                license_summary.loc[
                    license_summary["measure"].eq("rights_review_status")
                    & license_summary["category"].eq(
                        "no_observed_model_dependency"
                    ),
                    "count",
                ].iloc[0]
            )
            == int(
                (
                    main["app_license"].fillna("missing").ne("missing")
                    & ~main["space_id"].isin(matched_model_ids)
                ).sum()
            )
        ),
        "no_geography_inferred_from_missing_location": not bool(inferred_regions_from_missing),
        "ranking_shares_sum_to_one": bool(np.allclose(ranking_sums.to_numpy(), 1.0, atol=1e-12)),
        "hhi_in_unit_interval": bool(concentration["hhi"].between(0, 1).all()),
        "hhi_matches_provider_rankings": hhi_aligned,
        "bootstrap_intervals_ordered": bool(
            (
                concentration.dropna(subset=["hhi_ci_low", "hhi_ci_high"])["hhi_ci_low"]
                <= concentration.dropna(subset=["hhi_ci_low", "hhi_ci_high"])["hhi"]
            ).all()
            and (
                concentration.dropna(subset=["hhi_ci_low", "hhi_ci_high"])["hhi"]
                <= concentration.dropna(subset=["hhi_ci_low", "hhi_ci_high"])["hhi_ci_high"]
            ).all()
        ),
        "top_provider_stable_high_confidence": combined_top == high_top,
        "top_provider_stable_clone_adjustment": combined_top == clone_top,
        "top_provider_stable_near_duplicate_adjustment": combined_top == near_top,
        "provider_only_variant_excludes_runtime_category": (
            "local_runtime"
            not in str(
                concentration.set_index("analysis").loc[
                    "provider_only_primary", "layers"
                ]
            )
        ),
        "paired_project_count_matches_layer_intersection": int(
            paired["matched_projects"]
        ) == len(matched_service_ids & matched_model_ids),
        "paired_hhi_difference_reconstructs": bool(
            np.isclose(
                paired_difference,
                float(paired["hhi_difference_service_minus_model"]),
                atol=1e-12,
            )
        ),
        "paired_bootstrap_has_fixed_draw_count": len(paired_draws)
        == int(paired["bootstrap_draws"])
        == 2_000,
        "paired_difference_interval_ordered": bool(
            float(paired["hhi_difference_bootstrap_ci"][0])
            <= paired_difference
            <= float(paired["hhi_difference_bootstrap_ci"][1])
        ),
        "paired_alternative_metrics_preserve_layer_ordering": bool(
            float(paired["top_share_difference_bootstrap_ci"][0]) > 0
            and float(paired["shannon_entropy_difference_bootstrap_ci"][0]) > 0
            and float(paired["service_top_share"]) > float(paired["model_top_share"])
            and float(paired["service_shannon_effective_categories"])
            < float(paired["model_shannon_effective_categories"])
        ),
        "cross_layer_codeclaration_reconstructs": bool(
            int(codeclaration["matched_projects"]) == int(paired["matched_projects"])
            and np.isclose(
                codeclaration_joint["joint_fractional_share"].sum(), 1.0, atol=1e-12
            )
            and np.isclose(
                codeclaration_service["service_fractional_share"].sum(),
                1.0,
                atol=1e-12,
            )
            and len(codeclaration_null) == int(codeclaration["permutations"]) == 10_000
            and float(codeclaration["mutual_information"])
            > float(codeclaration["permutation_null_interval"][1])
        ),
        "matched_robustness_primary_matches_central": bool(
            np.isclose(
                matched_robustness.loc[
                    matched_robustness["robustness_family"].eq("primary"),
                    "hhi_difference_service_minus_model",
                ].iloc[0],
                paired_difference,
                atol=1e-12,
            )
        ),
        "matched_robustness_uses_positive_paired_differences": bool(
            matched_robustness["hhi_difference_ci_low"].gt(0).all()
        ),
        "matched_robustness_contains_requested_families": bool(
            {
                "primary",
                "rank_cutoff",
                "leave_one_query_out",
                "evidence",
                "weighting",
                "search_arm",
                "source_reuse",
                "model_taxonomy",
                "leave_one_service_provider_out",
            }
            <= set(matched_robustness["robustness_family"])
            and {
                "near_duplicate_jaccard_085",
                "near_duplicate_jaccard_090",
                "near_duplicate_jaccard_095",
                "exact_multifile_cluster",
                "author_cluster",
            }
            <= set(matched_robustness["variant"])
            and {
                "immediate_public_namespace",
                "disaggregated_unmapped_namespaces",
                "pooled_unmapped_namespaces",
                "mapped_publishers_only",
            }
            <= set(matched_robustness["variant"])
        ),
        "service_provider_omission_preserves_layer_ordering": bool(
            not provider_omission.empty
            and provider_omission["hhi_difference_ci_low"].gt(0).all()
            and set(provider_omission["variant"])
            == set(
                edges.loc[
                    edges["space_id"].isin(main_ids)
                    & edges["layer"].eq("inference_service")
                    & edges["provider"].fillna("").astype(str).ne(""),
                    "provider",
                ].astype(str)
            )
        ),
        "matched_rank_cutoffs_reconstruct": bool(
            cutoff_matched.index.equals(robustness_cutoff.index)
            and np.allclose(cutoff_matched, robustness_cutoff, atol=1e-12)
        ),
        "matched_query_ablations_reconstruct": bool(
            set(ablation_matched.index) == set(robustness_ablation.index)
            and np.allclose(
                ablation_matched.sort_index(),
                robustness_ablation.sort_index(),
                atol=1e-12,
            )
        ),
        "matched_composition_uses_central_group_sizes": bool(
            matched_composition["matched_n"].eq(int(paired["matched_projects"])).all()
            and matched_composition["nonmatched_n"].eq(len(main) - int(paired["matched_projects"])).all()
        ),
        "unknown_service_bounds_are_explicit_and_direction_preserving": bool(
            set(unknown_service["unknown_treatment"])
            == {
                "known_services_only",
                "pooled_unmapped_candidates",
                "machine_identifier_categories",
                "project_unique_unknown_bound",
            }
            and unknown_service["hhi_difference_ci_low"].gt(0).all()
        ),
        "file_cap_sensitivity_reconstructs_and_preserves_ordering": bool(
            file_cap["max_source_files"].tolist() == [10, 15, 20]
            and file_cap["selected_files"].is_monotonic_increasing
            and np.isclose(
                file_cap.loc[
                    file_cap["max_source_files"].eq(10),
                    "hhi_difference_service_minus_model",
                ].iloc[0],
                paired_difference,
                atol=1e-12,
            )
            and file_cap["hhi_difference_ci_low"].gt(0).all()
        ),
        "namespace_variants_preserve_layer_ordering": bool(
            namespaces["service_hhi"].gt(namespaces["model_hhi"]).all()
            and namespaces["namespace_variant"].nunique() == 3
        ),
        "search_cutoffs_complete_and_monotone": bool(
            cutoff_combined["rank_cutoff"].tolist() == [25, 50, 75, 100]
            and cutoff_combined["strict_projects_in_frame"].is_monotonic_increasing
        ),
        "search_cutoff_leader_stable": bool(
            cutoff_combined["top_provider"].eq(combined_top).all()
        ),
        "all_query_ablation_frames_present": bool(
            ablation_combined["excluded_query"].nunique() == 15
            and len(ablation_combined) == 15
        ),
        "query_ablation_leader_stable": bool(
            ablation_combined["top_provider"].eq(combined_top).all()
        ),
        "history_audit_covers_strict_sample": bool(
            history_audit["space_id"].is_unique
            and set(history_audit["space_id"]) == main_ids
            and history_audit["history_status_code"].eq(200).all()
            and history_audit["initial_state_status"].eq("RESOLVED").all()
        ),
        "historical_edges_reference_strict_sample": bool(
            set(historical_edges["space_id"]) <= main_ids
            and historical_edges["revision_role"].eq("earliest_analyzable").all()
        ),
        "version_transitions_cover_strict_sample": bool(
            transitions["space_id"].is_unique
            and set(transitions["space_id"]) == main_ids
        ),
        "zero_manual_annotation_reported": bool(
            decision["annotation_regime"]["manual_annotation_rows"] == 0
            and history["manual_annotation_rows"] == 0
        ),
        "version_bootstrap_has_fixed_draw_count": len(history_draws)
        == int(history["paired_service_concentration"]["bootstrap_draws"])
        == 2_000,
        "version_hhi_change_interval_ordered": bool(
            float(
                history["paired_service_concentration"]["hhi_change_bootstrap_ci"][0]
            )
            <= history_change
            <= float(
                history["paired_service_concentration"]["hhi_change_bootstrap_ci"][1]
            )
        ),
        "version_change_not_directionally_resolved": bool(
            float(
                history["paired_service_concentration"]["hhi_change_bootstrap_ci"][0]
            )
            <= 0
            <= float(
                history["paired_service_concentration"]["hhi_change_bootstrap_ci"][1]
            )
        ),
        "history_span_thresholds_complete_and_monotone": bool(
            span_sensitivity["minimum_history_span_days"].tolist()
            == [0, 1, 7, 30, 90, 180]
            and span_sensitivity["projects"].is_monotonic_decreasing
            and span_sensitivity["paired_service_projects"].is_monotonic_decreasing
        ),
        "regional_gate_respects_coverage": (
            region["author_region_comparison_status"] == "NOT_ESTIMABLE"
            if float(region["author_location_coverage"]) < 0.20
            else region["author_region_comparison_status"] == "PASS"
        ),
        "longitudinal_claim_not_tested": decision["longitudinal_disruption_analysis"]["status"]
        == "NOT_TESTED",
        "measurement_validation_complete": bool(
            measurement_validation.get("status") == "COMPLETE"
            and measurement_validation_status.get("status") == "COMPLETE"
            and all(
                measurement_validation[task]["unresolved_records"] == 0
                for task in (
                    "education_relevance",
                    "dependency_evidence",
                    "service_negative_audit",
                )
            )
        ),
        "measurement_validation_sample_sizes_match": bool(
            measurement_validation["education_relevance"]["records"] == 178
            and measurement_validation["dependency_evidence"]["records"] == 160
            and measurement_validation["service_negative_audit"]["records"] == 80
            and len(validation_deidentified) == 418
        ),
        "confirmed_service_omissions_are_candidate_bounded": bool(
            measurement_validation["service_negative_audit"]
            ["all_missed_service_projects_in_candidate_boundary"]
            and measurement_validation["service_negative_audit"]
            ["missed_service_records"]
            == measurement_validation["service_negative_audit"]
            ["missed_service_records_in_candidate_boundary"]
            == 15
        ),
        "released_validation_labels_are_deidentified": bool(
            set(validation_deidentified.columns)
            == {
                "task",
                "sample_id",
                "validation_stratum",
                "inclusion_probability",
                "automatic_positive",
                "reviewer1_label",
                "reviewer2_label",
                "third_review_label",
                "resolved_label",
                "candidate_boundary",
            }
            and not validation_deidentified.duplicated(["task", "sample_id"]).any()
            and validation_deidentified.loc[
                validation_deidentified["task"].ne("service_negative"),
                "candidate_boundary",
            ].isna().all()
            and validation_deidentified.loc[
                validation_deidentified["task"].eq("service_negative"),
                "candidate_boundary",
            ].notna().all()
        ),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    return {
        "status": status,
        "hard_checks": checks,
        "strict_projects": len(main),
        "dependency_edges": len(edges[edges["space_id"].isin(main_ids)]),
        "official_linked_model_edges": len(linked_edges[linked_edges["space_id"].isin(main_ids)]),
        "not_estimable": {
            "author_region_comparison": region["author_region_comparison_status"],
            "language_orientation_comparison": region["language_orientation_comparison_status"],
            "upstream_event_survival": decision["longitudinal_disruption_analysis"]["status"],
        },
        "note": "NOT_ESTIMABLE and NOT_TESTED are scope boundaries, not failed numerical checks.",
    }


def markdown(report: dict) -> str:
    lines = [
        "# Final statistical package audit",
        "",
        f"Status: **{report.get('status', 'FAIL')}**",
        "",
        "## Hard checks",
        "",
    ]
    for name, value in report.get("hard_checks", {}).items():
        lines.append(f"- `{name}`: {'PASS' if value else 'FAIL'}")
    lines.extend(
        [
            "",
            "## Deliberate scope boundaries",
            "",
            "```json",
            json.dumps(report.get("not_estimable", {}), ensure_ascii=False, indent=2),
            "```",
            "",
            report.get("note", ""),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    report = build_audit(root)
    output = root / "data/qa/final_statistical_audit.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "PASS":
        raise RuntimeError("final statistical package audit failed")


if __name__ == "__main__":
    main()
