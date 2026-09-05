# Final statistical package audit

Status: **PASS**

## Hard checks

- `required_files_present`: PASS
- `protocol_version_matches`: PASS
- `data_capture_gate_passes`: PASS
- `candidate_ids_unique`: PASS
- `selected_space_ids_unique`: PASS
- `strict_sample_count_matches_collection`: PASS
- `strict_sample_count_matches_analysis`: PASS
- `strict_phrase_frame_matches_collection`: PASS
- `all_edges_reference_selected_spaces`: PASS
- `dependency_frame_has_edge_or_machine_service_candidate`: PASS
- `candidate_only_selection_is_explicit`: PASS
- `selection_funnel_has_three_separate_construct_frames`: PASS
- `dependency_rows_unique`: PASS
- `model_ids_unique`: PASS
- `clone_assignments_unique`: PASS
- `clone_weights_positive_and_bounded`: PASS
- `similarity_assignments_cover_each_threshold`: PASS
- `similarity_weights_positive_and_bounded`: PASS
- `near_duplicate_count_matches_manifest`: PASS
- `official_linked_models_high_confidence`: PASS
- `code_model_ids_are_explicit_medium_evidence`: PASS
- `environment_examples_are_explicit_medium_evidence`: PASS
- `evidence_source_audit_reconstructs`: PASS
- `model_license_denominators_match_evidence_sets`: PASS
- `no_model_dependency_rights_status_is_literal`: PASS
- `no_geography_inferred_from_missing_location`: PASS
- `ranking_shares_sum_to_one`: PASS
- `hhi_in_unit_interval`: PASS
- `hhi_matches_provider_rankings`: PASS
- `bootstrap_intervals_ordered`: PASS
- `top_provider_stable_high_confidence`: PASS
- `top_provider_stable_clone_adjustment`: PASS
- `top_provider_stable_near_duplicate_adjustment`: PASS
- `provider_only_variant_excludes_runtime_category`: PASS
- `paired_project_count_matches_layer_intersection`: PASS
- `paired_hhi_difference_reconstructs`: PASS
- `paired_bootstrap_has_fixed_draw_count`: PASS
- `paired_difference_interval_ordered`: PASS
- `paired_alternative_metrics_preserve_layer_ordering`: PASS
- `cross_layer_codeclaration_reconstructs`: PASS
- `matched_robustness_primary_matches_central`: PASS
- `matched_robustness_uses_positive_paired_differences`: PASS
- `matched_robustness_contains_requested_families`: PASS
- `service_provider_omission_preserves_layer_ordering`: PASS
- `matched_rank_cutoffs_reconstruct`: PASS
- `matched_query_ablations_reconstruct`: PASS
- `matched_composition_uses_central_group_sizes`: PASS
- `unknown_service_bounds_are_explicit_and_direction_preserving`: PASS
- `file_cap_sensitivity_reconstructs_and_preserves_ordering`: PASS
- `namespace_variants_preserve_layer_ordering`: PASS
- `search_cutoffs_complete_and_monotone`: PASS
- `search_cutoff_leader_stable`: PASS
- `all_query_ablation_frames_present`: PASS
- `query_ablation_leader_stable`: PASS
- `history_audit_covers_strict_sample`: PASS
- `historical_edges_reference_strict_sample`: PASS
- `version_transitions_cover_strict_sample`: PASS
- `zero_manual_annotation_reported`: PASS
- `version_bootstrap_has_fixed_draw_count`: PASS
- `version_hhi_change_interval_ordered`: PASS
- `version_change_not_directionally_resolved`: PASS
- `history_span_thresholds_complete_and_monotone`: PASS
- `regional_gate_respects_coverage`: PASS
- `longitudinal_claim_not_tested`: PASS
- `measurement_validation_complete`: PASS
- `measurement_validation_sample_sizes_match`: PASS
- `confirmed_service_omissions_are_candidate_bounded`: PASS
- `released_validation_labels_are_deidentified`: PASS

## Deliberate scope boundaries

```json
{
  "author_region_comparison": "NOT_ESTIMABLE",
  "language_orientation_comparison": "NOT_ESTIMABLE",
  "upstream_event_survival": "NOT_TESTED"
}
```

NOT_ESTIMABLE and NOT_TESTED are scope boundaries, not failed numerical checks.
