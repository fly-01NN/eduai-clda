from pathlib import Path

import pandas as pd
import pytest

from analyze_dependencies import (
    bootstrap_hhi,
    concentration_variant_rows,
    cross_layer_codeclaration,
    fractional_rankings,
    leave_one_service_provider_out,
    model_namespace_variants,
    load_released_project_shingles,
    load_search_rankings,
    paired_layer_bootstrap,
    search_union,
    similarity_cluster_assignments,
    source_cluster_assignments,
    source_token_shingles,
    unknown_service_boundary_table,
    write_project_source_shingles,
)


def test_fractional_rankings_allocate_one_unit_per_project() -> None:
    spaces = pd.DataFrame({"space_id": ["a", "b"]})
    edges = pd.DataFrame(
        {
            "space_id": ["a", "a", "b"],
            "layer": ["model_dependency"] * 3,
            "provider": ["P1", "P2", "P1"],
        }
    )
    ranking, summary = fractional_rankings(
        edges, spaces, layers=("model_dependency",)
    )
    weights = ranking.set_index("provider")["fractional_weight"].to_dict()
    assert weights == {"P1": 1.5, "P2": 0.5}
    assert ranking["fractional_share"].sum() == 1.0
    assert summary["hhi"] == 0.625


def test_fractional_rankings_exclude_edges_outside_supplied_space_frame() -> None:
    spaces = pd.DataFrame({"space_id": ["matched"]})
    edges = pd.DataFrame(
        {
            "space_id": ["matched", "outside", "outside"],
            "layer": ["inference_service", "inference_service", "model_dependency"],
            "provider": ["S1", "S2", "M2"],
        }
    )
    ranking, summary = fractional_rankings(edges, spaces, layers=("inference_service",))
    assert ranking["provider"].tolist() == ["S1"]
    assert summary["projects"] == 1
    assert summary["hhi"] == 1.0


def test_matched_sensitivity_row_ignores_nonmatched_edges() -> None:
    spaces = pd.DataFrame({"space_id": ["a", "b"]})
    edges = pd.DataFrame(
        {
            "space_id": ["a", "a", "b"],
            "layer": ["inference_service", "model_dependency", "inference_service"],
            "provider": ["S1", "M1", "S2"],
        }
    )
    rows = concentration_variant_rows(edges, spaces, descriptor={})
    matched = next(row for row in rows if row["analysis"] == "matched_service_minus_model")
    assert matched["projects"] == 1
    assert matched["hhi"] == 0.0


def test_bootstrap_hhi_is_deterministic() -> None:
    spaces = pd.DataFrame({"space_id": ["a", "b", "c"]})
    edges = pd.DataFrame(
        {
            "space_id": ["a", "b", "c"],
            "layer": ["inference_service"] * 3,
            "provider": ["P1", "P1", "P2"],
        }
    )
    first = bootstrap_hhi(edges, spaces, layers=("inference_service",), draws=100, seed=7)
    second = bootstrap_hhi(edges, spaces, layers=("inference_service",), draws=100, seed=7)
    assert first == second
    assert 0 <= first[0] <= first[1] <= 1


def test_source_clusters_require_two_exact_file_hashes() -> None:
    manifest = pd.DataFrame(
        {
            "space_id": ["a", "a", "b", "b", "c"],
            "file_path": ["app.py", "requirements.txt", "main.py", "requirements.txt", "app.py"],
            "sha256": ["x", "y", "x", "y", "x"],
            "status_code": [200] * 5,
        }
    )
    clusters = source_cluster_assignments(manifest).set_index("space_id")
    assert clusters.loc["a", "source_cluster_id"] == clusters.loc["b", "source_cluster_id"]
    assert clusters.loc["a", "source_cluster_size"] == 2
    assert clusters.loc["c", "source_cluster_size"] == 1


def test_paired_layer_bootstrap_uses_identical_projects() -> None:
    edges = pd.DataFrame(
        {
            "space_id": ["a", "b", "c", "d", "a", "b", "c", "d"],
            "layer": ["inference_service"] * 4 + ["model_dependency"] * 4,
            "provider": ["S"] * 4 + ["M1", "M2", "M3", "M4"],
        }
    )
    first, draws_first = paired_layer_bootstrap(
        edges, ["a", "b", "c", "d"], draws=50, seed=11
    )
    second, draws_second = paired_layer_bootstrap(
        edges, ["a", "b", "c", "d"], draws=50, seed=11
    )
    assert first == second
    pd.testing.assert_frame_equal(draws_first, draws_second)
    assert first["matched_projects"] == 4
    assert first["service_hhi"] == 1.0
    assert first["model_hhi"] == 0.25
    assert first["hhi_difference_service_minus_model"] == 0.75
    assert first["service_top_share"] == 1.0
    assert first["model_top_share"] == 0.25
    assert first["top_share_difference_service_minus_model"] == 0.75
    assert first["service_shannon_effective_categories"] == 1.0
    assert first["model_shannon_effective_categories"] == pytest.approx(4.0)


def test_leave_one_provider_out_removes_projects_from_both_layers() -> None:
    spaces = pd.DataFrame({"space_id": ["a", "b", "c", "d"]})
    edges = pd.DataFrame(
        {
            "space_id": ["a", "b", "c", "d", "a", "b", "c", "d"],
            "layer": ["inference_service"] * 4 + ["model_dependency"] * 4,
            "provider": ["S1", "S1", "S2", "S2", "M1", "M2", "M3", "M4"],
        }
    )
    rows = leave_one_service_provider_out(edges, spaces, draws=20, seed=7)
    omitted_s1 = rows.set_index("variant").loc["S1"]
    assert omitted_s1["omitted_projects"] == 2
    assert omitted_s1["strict_projects_in_frame"] == 2
    assert omitted_s1["matched_projects"] == 2
    assert omitted_s1["service_hhi"] == 1.0
    assert omitted_s1["model_hhi"] == 0.5


def test_cross_layer_codeclaration_is_project_equal_and_deterministic() -> None:
    edges = pd.DataFrame(
        {
            "space_id": ["a", "b", "a", "b"],
            "layer": [
                "inference_service",
                "inference_service",
                "model_dependency",
                "model_dependency",
            ],
            "provider": ["S1", "S2", "M1", "M2"],
        }
    )
    first = cross_layer_codeclaration(edges, ["a", "b"], permutations=20, seed=3)
    second = cross_layer_codeclaration(edges, ["a", "b"], permutations=20, seed=3)
    assert first[0] == second[0]
    pd.testing.assert_frame_equal(first[1], second[1])
    pd.testing.assert_frame_equal(first[3], second[3])
    assert first[0]["joint_fractional_share_sum"] == pytest.approx(1.0)
    assert first[1]["joint_fractional_share"].sum() == pytest.approx(1.0)


def test_model_namespace_variants_pool_or_exclude_only_unmapped_edges() -> None:
    edges = pd.DataFrame(
        {
            "space_id": ["a", "b", "c"],
            "layer": ["model_dependency"] * 3,
            "provider": ["Known", "namespace:x", "namespace:y"],
            "provider_basis": [
                "model_namespace",
                "unmapped_public_namespace",
                "unmapped_public_namespace",
            ],
        }
    )
    variants = model_namespace_variants(edges)
    pooled = variants["pooled_unmapped_namespaces"]
    assert pooled["provider"].tolist() == [
        "Known",
        "Unmapped namespace",
        "Unmapped namespace",
    ]
    assert variants["mapped_publishers_only"]["space_id"].tolist() == ["a"]


def test_search_union_respects_rank_cutoff_arm_and_query_ablation() -> None:
    rankings = {
        ("q1", "likes"): ["a", "b", "c"],
        ("q1", "createdAt"): ["d", "e", "f"],
        ("q2", "likes"): ["a", "g", "h"],
        ("q2", "createdAt"): ["i", "j", "k"],
    }
    assert search_union(rankings, cutoff=2) == {"a", "b", "d", "e", "g", "i", "j"}
    assert search_union(rankings, cutoff=2, arm_filter="likes") == {"a", "b", "g"}
    assert search_union(rankings, cutoff=2, excluded_term="q1") == {"a", "g", "i", "j"}


def test_source_shingles_ignore_comments_and_literal_values() -> None:
    left = """
    # one comment
    def answer(value):
        message = "alpha"
        return value + 123 if value >= 0 else message
    """
    right = """
    # another comment
    def answer(value):
        message = "beta"
        return value + 999 if value >= 0 else message
    """
    assert source_token_shingles(left) == source_token_shingles(right)


def test_similarity_clusters_are_thresholded_single_linkage() -> None:
    project_shingles = {
        "a": {f"s{index}" for index in range(100)},
        "b": {f"s{index}" for index in range(95)}
        | {f"x{index}" for index in range(5)},
        "c": {f"z{index}" for index in range(100)},
    }
    assignments, pairs = similarity_cluster_assignments(
        project_shingles,
        ["a", "b", "c"],
        thresholds=(0.90, 0.95),
    )
    assert pairs.iloc[0]["jaccard_similarity"] == pytest.approx(95 / 105)
    at_090 = assignments[assignments["threshold"].eq(0.90)].set_index("space_id")
    at_095 = assignments[assignments["threshold"].eq(0.95)].set_index("space_id")
    assert at_090.loc["a", "source_similarity_cluster_size"] == 2
    assert at_090.loc["b", "source_similarity_cluster_size"] == 2
    assert at_095.loc["a", "source_similarity_cluster_size"] == 1


def test_offline_search_rankings_fall_back_to_compact_release_table(
    tmp_path: Path,
) -> None:
    from protocol import SEARCH_ARMS, SEARCH_TERMS

    processed = tmp_path / "data/processed"
    processed.mkdir(parents=True)
    rows = [
        {"query": term, "arm": arm, "rank": 1, "space_id": f"{term}/{arm}"}
        for term in SEARCH_TERMS
        for arm in SEARCH_ARMS
    ]
    pd.DataFrame(rows).to_csv(processed / "search_rankings.csv", index=False)
    raw_dir = tmp_path / "data/raw/2026-08-31/search_expanded"
    rankings = load_search_rankings(raw_dir)
    assert len(rankings) == len(SEARCH_TERMS) * len(SEARCH_ARMS)
    assert rankings[(SEARCH_TERMS[0], SEARCH_ARMS[0])] == [
        f"{SEARCH_TERMS[0]}/{SEARCH_ARMS[0]}"
    ]


def test_offline_source_similarity_uses_irreversible_shingle_archive(
    tmp_path: Path,
) -> None:
    processed = tmp_path / "data/processed"
    qa = tmp_path / "data/qa"
    qa.mkdir(parents=True)
    shingles = {"a": {f"x{index}" for index in range(50)}}
    write_project_source_shingles(processed / "source_shingles.json.gz", shingles)
    pd.DataFrame(
        {
            "space_id": ["a", "b"],
            "code_files": [1, 0],
            "source_shingles": [50, 0],
            "similarity_eligible": [True, False],
            "missing_cache_files": [0, 0],
        }
    ).to_csv(qa / "source_similarity_coverage.csv", index=False)
    loaded, coverage = load_released_project_shingles(tmp_path, {"a", "b"})
    assert loaded == shingles
    assert set(coverage["space_id"]) == {"a", "b"}


def test_openai_compatible_candidate_enters_unknown_bound_not_primary_service() -> None:
    spaces = pd.DataFrame({"space_id": ["known", "unresolved"]})
    edges = pd.DataFrame(
        {
            "space_id": ["known", "known", "unresolved"],
            "layer": [
                "inference_service",
                "model_dependency",
                "model_dependency",
            ],
            "provider": ["Known service", "Model A", "Model B"],
        }
    )
    candidates = pd.DataFrame(
        {
            "space_id": ["unresolved"],
            "candidate_type": ["openai_compatible_provider_unresolved"],
            "identifier": ["openai-compatible-client"],
            "source_file": ["requirements.txt"],
        }
    )

    result = unknown_service_boundary_table(
        spaces,
        edges,
        candidates,
        strict_phrase_projects=2,
    ).set_index("unknown_treatment")

    assert result.loc["known_services_only", "matched_projects"] == 1
    assert result.loc["pooled_unmapped_candidates", "matched_projects"] == 2
    assert result.loc["machine_identifier_categories", "matched_projects"] == 2
    assert result.loc["project_unique_unknown_bound", "matched_projects"] == 2
