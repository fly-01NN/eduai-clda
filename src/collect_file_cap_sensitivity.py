"""Re-scan pinned repositories with wider source-file caps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from analyze_dependencies import paired_layer_bootstrap
from collect_hf_spaces import Collector, read_json
from dependency_parser import provider_signals, select_repository_files


FILE_CAPS = (10, 15, 20)


def load_frozen_items(raw: Path, strict_ids: set[str]) -> dict[str, dict]:
    items: dict[str, dict] = {}
    for path in sorted((raw / "search_expanded").glob("*.json")):
        payload = read_json(path)
        if not isinstance(payload, list):
            continue
        for item in payload:
            if isinstance(item, dict) and str(item.get("id")) in strict_ids:
                items[str(item["id"])] = item
    missing = strict_ids - set(items)
    if missing:
        raise RuntimeError(f"frozen search payloads omit {len(missing)} strict projects")
    return items


def service_provider_sets(edges: pd.DataFrame) -> dict[str, frozenset[str]]:
    service = edges[edges["layer"].eq("inference_service")]
    return {
        str(space_id): frozenset(group["provider"].dropna().astype(str))
        for space_id, group in service.groupby("space_id")
    }


def run(root: Path, *, workers: int) -> pd.DataFrame:
    processed = root / "data/processed"
    results = root / "analysis_results"
    qa = root / "data/qa"
    spaces = pd.read_csv(processed / "space_frame.csv")
    spaces = spaces[spaces["included_strict"].eq(True)].copy()
    strict_ids = set(spaces["space_id"].astype(str))
    original_edges = pd.read_csv(processed / "dependency_edges.csv")
    model_edges = original_edges[original_edges["layer"].eq("model_dependency")].copy()
    original_service = original_edges[
        original_edges["layer"].eq("inference_service")
        & original_edges["space_id"].astype(str).isin(strict_ids)
    ].copy()
    baseline_sets = service_provider_sets(original_service)
    items = load_frozen_items(root / "data/raw/2026-08-31", strict_ids)

    collector = Collector(root, refresh=False, workers=max(1, workers))
    files_by_project: dict[str, dict[str, str]] = {}
    selected_by_cap: dict[int, dict[str, list[str]]] = {
        cap: {} for cap in FILE_CAPS
    }
    try:
        for space_id in sorted(strict_ids, key=str.casefold):
            item = items[space_id]
            siblings = collector.sibling_names(item)
            app_file = (item.get("cardData") or {}).get("app_file")
            for cap in FILE_CAPS:
                selected_by_cap[cap][space_id] = select_repository_files(
                    siblings,
                    str(app_file) if app_file else None,
                    max_files=cap,
                )
            files_by_project[space_id] = collector.fetch_files(
                item, selected_by_cap[max(FILE_CAPS)][space_id]
            )
    finally:
        collector.close()

    cap_edges: dict[int, pd.DataFrame] = {10: original_service}
    for cap in FILE_CAPS[1:]:
        rows: list[dict[str, str]] = []
        for space_id in sorted(strict_ids, key=str.casefold):
            paths = set(selected_by_cap[cap][space_id])
            files = {
                path: text
                for path, text in files_by_project[space_id].items()
                if path in paths
            }
            for signal in provider_signals(files):
                if signal.layer == "inference_service":
                    rows.append({"space_id": space_id, **signal.to_dict()})
        cap_edges[cap] = pd.DataFrame(rows).drop_duplicates()

    rows: list[dict[str, object]] = []
    for cap in FILE_CAPS:
        service_edges = cap_edges[cap]
        combined = pd.concat([service_edges, model_edges], ignore_index=True, sort=False)
        summary, _ = paired_layer_bootstrap(combined, strict_ids)
        current_sets = service_provider_sets(service_edges)
        changed = sum(
            current_sets.get(space_id, frozenset())
            != baseline_sets.get(space_id, frozenset())
            for space_id in strict_ids
        )
        newly_observable = sum(
            not baseline_sets.get(space_id, frozenset())
            and bool(current_sets.get(space_id, frozenset()))
            for space_id in strict_ids
        )
        rows.append(
            {
                "max_source_files": cap,
                "strict_projects": len(strict_ids),
                "selected_files": sum(len(value) for value in selected_by_cap[cap].values()),
                "projects_with_service": len(current_sets),
                "projects_with_changed_service_set_vs_cap10": changed,
                "newly_service_observable_projects_vs_cap10": newly_observable,
                "matched_projects": summary["matched_projects"],
                "service_hhi": summary["service_hhi"],
                "model_hhi": summary["model_hhi"],
                "hhi_difference_service_minus_model": summary[
                    "hhi_difference_service_minus_model"
                ],
                "hhi_difference_ci_low": summary["hhi_difference_bootstrap_ci"][0],
                "hhi_difference_ci_high": summary["hhi_difference_bootstrap_ci"][1],
                "service_providers": summary["service_providers"],
                "model_publishers_or_namespaces": summary[
                    "model_publishers_or_namespaces"
                ],
                "boundary": (
                    "service-source cap varies on the same frozen revisions; "
                    "the referenced-model layer remains fixed"
                ),
            }
        )
    output = pd.DataFrame(rows)
    output.to_csv(results / "file_cap_sensitivity.csv", index=False)

    original_keys = set(
        original_service[
            ["space_id", "provider", "layer", "evidence_type", "evidence_value", "source_file"]
        ].itertuples(index=False, name=None)
    )
    cap20 = cap_edges[20].copy()
    cap20["edge_key"] = list(
        cap20[
            ["space_id", "provider", "layer", "evidence_type", "evidence_value", "source_file"]
        ].itertuples(index=False, name=None)
    )
    cap20[~cap20["edge_key"].isin(original_keys)].drop(columns="edge_key").to_csv(
        qa / "file_cap_additional_service_edges.csv", index=False
    )
    pd.DataFrame(collector.file_manifest).drop_duplicates(
        ["space_id", "revision", "file_path"]
    ).to_csv(qa / "file_cap_source_manifest.csv", index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    print(run(args.root.resolve(), workers=args.workers).to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
