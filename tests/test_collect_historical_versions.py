from pathlib import Path

import pandas as pd

from collect_historical_versions import (
    classify_transition,
    commits_within_snapshot,
    fractional_hhi,
    is_analysis_file,
    paired_version_bootstrap,
    recompute_version_history_outputs,
)


def test_commits_are_sorted_and_truncated_at_snapshot() -> None:
    commits = [
        {"id": "later", "date": "2026-09-01T00:00:00Z"},
        {"id": "second", "date": "2026-08-01T00:00:00Z"},
        {"id": "first", "date": "2025-01-01T00:00:00Z"},
    ]
    output = commits_within_snapshot(commits, snapshot_date="2026-08-31")
    assert [row["id"] for row in output] == ["first", "second"]


def test_analysis_file_rule_is_machine_deterministic() -> None:
    assert is_analysis_file("src/app.py")
    assert is_analysis_file("requirements-dev.txt")
    assert is_analysis_file("pyproject.toml")
    assert not is_analysis_file("README.md")
    assert not is_analysis_file("LICENSE")
    assert not is_analysis_file("image.png")


def test_transition_categories_cover_set_changes() -> None:
    assert classify_transition(set(), set()) == "unchanged_no_signal"
    assert classify_transition({"A"}, {"A"}) == "unchanged_same"
    assert classify_transition(set(), {"A"}) == "added"
    assert classify_transition({"A"}, set()) == "removed"
    assert classify_transition({"A"}, {"B"}) == "changed"


def test_fractional_hhi_allocates_one_unit_per_eligible_project() -> None:
    assert fractional_hhi({"a": {"P1", "P2"}, "b": {"P1"}}) == 0.625


def test_version_bootstrap_is_paired_and_reproducible() -> None:
    initial = {"a": {"P1"}, "b": {"P2"}, "c": set()}
    current = {"a": {"P1"}, "b": {"P1"}, "c": {"P1"}}
    first, first_draws = paired_version_bootstrap(initial, current, draws=40, seed=3)
    second, second_draws = paired_version_bootstrap(initial, current, draws=40, seed=3)
    assert first == second
    pd.testing.assert_frame_equal(first_draws, second_draws)
    assert first["paired_projects"] == 2
    assert first["initial_service_hhi"] == 0.5
    assert first["current_service_hhi"] == 1.0


def test_release_tables_recompute_all_history_outputs_without_raw(
    tmp_path: Path,
) -> None:
    processed = tmp_path / "data/processed"
    qa = tmp_path / "data/qa"
    processed.mkdir(parents=True)
    qa.mkdir(parents=True)
    pd.DataFrame(
        [
            {"space_id": "a", "included_strict": True},
            {"space_id": "b", "included_strict": True},
            {"space_id": "c", "included_strict": True},
            {"space_id": "outside", "included_strict": False},
        ]
    ).to_csv(processed / "space_frame.csv", index=False)
    pd.DataFrame(
        [
            {"space_id": "a", "provider": "P1", "layer": "inference_service"},
            {"space_id": "a", "provider": "GPU", "layer": "local_runtime"},
            {"space_id": "b", "provider": "P1", "layer": "inference_service"},
            {"space_id": "c", "provider": "P1", "layer": "inference_service"},
            {
                "space_id": "outside",
                "provider": "P3",
                "layer": "inference_service",
            },
        ]
    ).to_csv(processed / "dependency_edges.csv", index=False)
    pd.DataFrame(
        [
            {"space_id": "a", "provider": "P1", "layer": "inference_service"},
            {"space_id": "a", "provider": "CPU", "layer": "local_runtime"},
            {"space_id": "b", "provider": "P2", "layer": "inference_service"},
            {
                "space_id": "outside",
                "provider": "P3",
                "layer": "inference_service",
            },
        ]
    ).to_csv(processed / "historical_dependency_edges.csv", index=False)
    pd.DataFrame(
        [
            {
                "space_id": "a",
                "history_status_code": 200,
                "commit_count_through_snapshot": 3,
                "initial_state_status": "RESOLVED",
                "history_span_days": 40,
            },
            {
                "space_id": "b",
                "history_status_code": 200,
                "commit_count_through_snapshot": 2,
                "initial_state_status": "RESOLVED",
                "history_span_days": 10,
            },
            {
                "space_id": "c",
                "history_status_code": 200,
                "commit_count_through_snapshot": 4,
                "initial_state_status": "RESOLVED",
                "history_span_days": 40,
            },
            {
                "space_id": "outside",
                "history_status_code": 200,
                "commit_count_through_snapshot": 9,
                "initial_state_status": "RESOLVED",
                "history_span_days": 90,
            },
        ]
    ).to_csv(qa / "space_commit_history_audit.csv", index=False)

    assert not (tmp_path / "data/raw").exists()
    first = recompute_version_history_outputs(
        tmp_path,
        draws=40,
        seed=7,
        span_thresholds=(0, 30),
    )
    output_names = (
        "version_history_summary.json",
        "version_dependency_transitions.csv",
        "version_service_hhi_bootstrap.csv",
        "version_span_sensitivity.csv",
    )
    first_bytes = {
        name: (tmp_path / "analysis_results" / name).read_bytes()
        for name in output_names
    }
    second = recompute_version_history_outputs(
        tmp_path,
        draws=40,
        seed=7,
        span_thresholds=(0, 30),
    )

    assert first == second
    assert first["strict_projects"] == 3
    assert first["earliest_analyzable_state_resolved"] == 3
    assert first["paired_service_concentration"]["paired_projects"] == 2
    assert first["service_transition_counts"] == {
        "unchanged_same": 1,
        "changed": 1,
        "added": 1,
    }
    assert not (tmp_path / "data/raw").exists()
    assert first_bytes == {
        name: (tmp_path / "analysis_results" / name).read_bytes()
        for name in output_names
    }
