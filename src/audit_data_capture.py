"""Hard gate for frozen input completeness and transport-level collection failures."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd

from protocol import PER_QUERY_LIMIT, SEARCH_ARMS, SEARCH_TERMS, SNAPSHOT_DATE
from source_encoding import cached_text_matches


def safe_slug(value: str) -> str:
    import re

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-").casefold()
    return cleaned or hashlib.sha256(value.encode()).hexdigest()[:12]


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def audit_capture(root: Path) -> dict[str, object]:
    root = root.resolve()
    raw = root / "data/raw" / SNAPSHOT_DATE
    processed = root / "data/processed"
    qa = root / "data/qa"
    transport_summary_path = qa / "transport_status_summary.json"
    frozen_transport: dict[str, object] = {}
    if transport_summary_path.exists():
        payload = read_json(transport_summary_path)
        if isinstance(payload, dict):
            frozen_transport = payload
    failures: list[str] = []
    warnings: list[str] = []

    candidates = pd.read_csv(qa / "candidate_selection_audit.csv")
    spaces = pd.read_csv(processed / "space_frame.csv")
    source_manifest = pd.read_csv(qa / "source_file_manifest.csv")

    search_lengths: dict[str, int] = {}
    search_ids: set[str] = set()
    capped_searches: list[str] = []
    search_dir = raw / "search_expanded"
    if search_dir.exists():
        for term in SEARCH_TERMS:
            for arm in SEARCH_ARMS:
                name = f"{safe_slug(term)}__{safe_slug(arm)}.json"
                path = search_dir / name
                if not path.exists():
                    failures.append(f"missing frozen search response: {name}")
                    continue
                try:
                    payload = read_json(path)
                except (OSError, json.JSONDecodeError) as exc:
                    failures.append(f"invalid frozen search response {name}: {exc}")
                    continue
                if not isinstance(payload, list):
                    failures.append(f"non-list frozen search response: {name}")
                    continue
                if len(payload) > PER_QUERY_LIMIT:
                    failures.append(f"search response exceeds protocol limit: {name}")
                search_lengths[name] = len(payload)
                if len(payload) == PER_QUERY_LIMIT:
                    capped_searches.append(name)
                search_ids.update(
                    str(item["id"])
                    for item in payload
                    if isinstance(item, dict) and item.get("id")
                )
    else:
        compact = processed / "search_rankings.csv"
        if not compact.exists():
            failures.append("neither raw search responses nor compact rankings are present")
        else:
            ranking = pd.read_csv(compact)
            expected = {(term, arm) for term in SEARCH_TERMS for arm in SEARCH_ARMS}
            observed = set(zip(ranking["query"].astype(str), ranking["arm"].astype(str)))
            if observed != expected:
                failures.append("compact rankings do not cover every frozen query arm")
            for (term, arm), group in ranking.groupby(["query", "arm"]):
                name = f"{safe_slug(str(term))}__{safe_slug(str(arm))}"
                ranked = group[
                    group["rank"].gt(0)
                    & group["space_id"].fillna("").astype(str).ne("")
                ]
                search_lengths[name] = len(ranked)
                search_ids.update(ranked["space_id"].astype(str))
                if len(ranked) == PER_QUERY_LIMIT:
                    capped_searches.append(f"{name}.json")

    cap_order = [
        f"{safe_slug(term)}__{safe_slug(arm)}.json"
        for term in SEARCH_TERMS
        for arm in SEARCH_ARMS
    ]
    capped_set = set(capped_searches)
    capped_searches = [name for name in cap_order if name in capped_set]

    candidate_ids = set(candidates["space_id"].astype(str))
    if search_ids and search_ids != candidate_ids:
        failures.append(
            f"search union and candidate audit differ by {len(search_ids ^ candidate_ids)} IDs"
        )
    if capped_searches:
        warnings.append(
            f"{len(capped_searches)} query arms reach the frozen rank cap; this is bounded-design truncation, not a failed response"
        )

    source_status = source_manifest["status_code"].value_counts().sort_index().to_dict()
    transport_failures = source_manifest[
        source_manifest["status_code"].eq(429)
        | source_manifest["status_code"].ge(500)
    ]
    if not transport_failures.empty:
        failures.append(f"{len(transport_failures)} source fetches have retryable failure status")
    duplicate_files = int(
        source_manifest.duplicated(["space_id", "revision", "file_path"]).sum()
    )
    if duplicate_files:
        failures.append(f"{duplicate_files} duplicate source-manifest keys")

    cache_root = root / "data/raw/file_cache"
    missing_caches = 0
    invalid_caches = 0
    content_mismatches = 0
    metadata_mismatches = 0
    if cache_root.exists():
        for row in source_manifest.itertuples(index=False):
            key = f"{row.space_id}\n{row.revision}\n{row.file_path}".encode("utf-8")
            path = cache_root / f"{hashlib.sha256(key).hexdigest()}.json.gz"
            if not path.exists():
                missing_caches += 1
                continue
            try:
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, EOFError, json.JSONDecodeError):
                invalid_caches += 1
                continue
            if not cached_text_matches(payload):
                content_mismatches += 1
            for field in ("space_id", "revision", "file_path", "sha256", "status_code", "bytes"):
                if str(payload.get(field)) != str(getattr(row, field)):
                    metadata_mismatches += 1
    else:
        if not (processed / "source_shingles.json.gz").exists():
            failures.append("source cache and non-invertible released shingle archive are both absent")
    if missing_caches:
        failures.append(f"{missing_caches} source cache files are missing")
    if invalid_caches:
        failures.append(f"{invalid_caches} source cache files are invalid")
    if content_mismatches:
        failures.append(
            f"{content_mismatches} source cache texts do not reconstruct recorded response bytes"
        )
    if metadata_mismatches:
        failures.append(f"{metadata_mismatches} source cache metadata fields disagree with manifest")

    empty_required = {
        column: int(spaces[column].fillna("").astype(str).eq("").sum())
        for column in ("space_id", "revision", "created_at", "last_modified")
    }
    if any(empty_required.values()):
        failures.append(f"required Space metadata are missing: {empty_required}")

    history_path = qa / "space_commit_history_audit.csv"
    history_summary: dict[str, object] = {"present": history_path.exists()}
    if history_path.exists():
        history = pd.read_csv(history_path)
        expected_history = int(spaces["included_strict"].eq(True).sum())
        history_summary.update(
            {
                "rows": len(history),
                "expected_strict_rows": expected_history,
                "status_codes": history["history_status_code"].value_counts().sort_index().to_dict(),
                "initial_states": history["initial_state_status"].value_counts().to_dict(),
            }
        )
        if len(history) != expected_history:
            failures.append("commit-history audit does not cover every strict project")
        if not history["history_status_code"].eq(200).all():
            failures.append("one or more strict commit histories failed to resolve")
        if not history["initial_state_status"].eq("RESOLVED").all():
            failures.append("one or more earliest analyzable states failed to resolve")

    history_file_cache = root / "data/raw/history_cache/files"
    history_cache_files = 0
    history_cache_invalid = 0
    history_cache_content_mismatches = 0
    if history_file_cache.exists():
        for path in history_file_cache.glob("*.json.gz"):
            history_cache_files += 1
            try:
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, EOFError, json.JSONDecodeError):
                history_cache_invalid += 1
                continue
            if not cached_text_matches(payload):
                history_cache_content_mismatches += 1
        history_summary.update(
            {
                "source_cache_files": history_cache_files,
                "invalid_source_caches": history_cache_invalid,
                "source_cache_content_mismatches": history_cache_content_mismatches,
            }
        )
        if history_cache_invalid:
            failures.append(
                f"{history_cache_invalid} historical source cache files are invalid"
            )
        if history_cache_content_mismatches:
            failures.append(
                f"{history_cache_content_mismatches} historical cache texts do not reconstruct recorded response bytes"
            )
    elif frozen_transport:
        frozen_history_integrity = frozen_transport.get(
            "historical_source_cache_integrity", {}
        )
        if isinstance(frozen_history_integrity, dict):
            history_cache_files = int(frozen_history_integrity.get("files", 0))
            history_cache_invalid = int(frozen_history_integrity.get("invalid", 0))
            history_cache_content_mismatches = int(
                frozen_history_integrity.get("content_mismatches", 0)
            )
            history_summary.update(
                {
                    "source_cache_files": history_cache_files,
                    "invalid_source_caches": history_cache_invalid,
                    "source_cache_content_mismatches": history_cache_content_mismatches,
                }
            )

    model_status: dict[str, int] = {}
    model_frame = pd.read_csv(processed / "model_frame.csv")
    active_model_status: dict[str, int] = {
        str(key): int(value)
        for key, value in model_frame["resolution_status"].value_counts().sort_index().items()
    }
    model_raw = raw / "models"
    if model_raw.exists():
        for path in model_raw.glob("*.json"):
            payload = read_json(path)
            status = (
                payload.get("_collection_status", 200)
                if isinstance(payload, dict)
                else "invalid_payload"
            )
            model_status[str(status)] = model_status.get(str(status), 0) + 1
        bad_transport = sum(
            count
            for status, count in model_status.items()
            if status == "429" or status.startswith("5") or status == "invalid_payload"
        )
        if bad_transport:
            failures.append(f"{bad_transport} model metadata requests have transport/payload failures")
    elif transport_summary_path.exists():
        if not isinstance(frozen_transport, dict):
            failures.append("released transport-status summary is not a JSON object")
        elif frozen_transport.get("snapshot_date") != SNAPSHOT_DATE:
            failures.append("released transport-status summary snapshot does not match protocol")
        else:
            frozen_statuses = frozen_transport.get("model_resolution_statuses", {})
            if not isinstance(frozen_statuses, dict):
                failures.append("released model transport-status summary is invalid")
            else:
                model_status = {
                    str(status): int(count)
                    for status, count in frozen_statuses.items()
                }
    else:
        failures.append(
            "model raw responses and released transport-status summary are both absent"
        )
    unresolved = sum(
        count
        for status, count in active_model_status.items()
        if status in {"401", "403", "404", "410"}
    )
    if unresolved:
        warnings.append(
            f"{unresolved} active exposed model identifiers are not publicly resolvable Hub artifacts; they remain explicitly unresolved"
        )

    github_status: dict[str, int] = {}
    github_frame_path = processed / "github_repository_frame.csv"
    if github_frame_path.exists():
        github_frame = pd.read_csv(github_frame_path)
        github_status = {
            str(key): int(value)
            for key, value in github_frame["collection_status"].value_counts().sort_index().items()
        }
        unresolved = int(github_frame["collection_status"].ne(200).sum())
        if unresolved:
            warnings.append(
                f"{unresolved} README-linked GitHub identifiers are unresolved or placeholders; this connector is supplementary"
            )

    if cache_root.exists() or history_file_cache.exists() or model_raw.exists():
        write_json(
            transport_summary_path,
            {
                "snapshot_date": SNAPSHOT_DATE,
                "model_resolution_statuses": model_status,
                "selected_source_cache_integrity": {
                    "manifest_rows": len(source_manifest),
                    "missing": missing_caches,
                    "invalid": invalid_caches,
                    "content_mismatches": content_mismatches,
                    "metadata_mismatches": metadata_mismatches,
                },
                "historical_source_cache_integrity": {
                    "files": history_cache_files,
                    "invalid": history_cache_invalid,
                    "content_mismatches": history_cache_content_mismatches,
                },
            },
        )

    result: dict[str, object] = {
        "gate_status": "PASS" if not failures else "FAIL",
        "snapshot_date": SNAPSHOT_DATE,
        "search_response_files": len(search_lengths),
        "search_length_distribution": {
            str(key): int(value)
            for key, value in pd.Series(list(search_lengths.values())).value_counts().sort_index().items()
        },
        "search_union_projects": len(search_ids),
        "candidate_audit_projects": len(candidates),
        "capped_query_arms": capped_searches,
        "source_manifest_rows": len(source_manifest),
        "source_status_codes": {str(key): int(value) for key, value in source_status.items()},
        "missing_source_caches": missing_caches,
        "invalid_source_caches": invalid_caches,
        "source_cache_content_mismatches": content_mismatches,
        "source_cache_metadata_mismatches": metadata_mismatches,
        "required_space_metadata_missing": empty_required,
        "model_resolution_statuses": model_status,
        "active_model_resolution_statuses": active_model_status,
        "github_resolution_statuses": github_status,
        "history": history_summary,
        "failures": failures,
        "warnings": warnings,
        "boundary": (
            "HTTP 401/403/404 identifiers and rank-cap warnings are recorded, not silently recoded as transport failures; "
            "only complete public primary inputs pass the gate"
        ),
    }
    write_json(qa / "data_capture_audit.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = audit_capture(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["gate_status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
