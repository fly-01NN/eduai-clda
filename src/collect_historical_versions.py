"""Collect and analyze earliest analyzable public Space revisions.

This extension is fully deterministic and uses no manual annotations. It
conditions on the frozen strict sample and compares each repository's earliest
analyzable source state with its source-derived dependencies at the frozen
snapshot. The result is a version-paired audit, not a population survival
panel or a common-calendar cohort.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping
from urllib.parse import quote

import httpx
import numpy as np
import pandas as pd

from collect_hf_spaces import HF_BASE, HubClient, utc_now
from dependency_parser import provider_signals, select_repository_files
from protocol import (
    CODE_SUFFIXES,
    MAX_SOURCE_BYTES,
    SNAPSHOT_DATE,
    TEXT_FILE_BASENAMES,
)
from source_encoding import cached_text_matches, decode_source_bytes


HISTORY_BOOTSTRAP_DRAWS = 2_000
HISTORY_BOOTSTRAP_SEED = 20260832
HISTORY_PAGE_LIMIT = 500
HISTORY_MAX_PAGES = 100
HISTORY_SPAN_THRESHOLDS = (0, 1, 7, 30, 90, 180)
HISTORY_TRANSITION_COLUMNS = (
    "space_id",
    "initial_service_providers",
    "current_service_providers",
    "service_transition",
    "initial_runtime_categories",
    "current_runtime_categories",
    "runtime_transition",
)
HISTORY_BOOTSTRAP_COLUMNS = (
    "draw",
    "initial_service_hhi",
    "current_service_hhi",
    "hhi_change_current_minus_initial",
)
HISTORY_SPAN_COLUMNS = (
    "minimum_history_span_days",
    "projects",
    "paired_service_projects",
    "initial_service_hhi",
    "current_service_hhi",
    "hhi_change_current_minus_initial",
    "hhi_change_ci_low",
    "hhi_change_ci_high",
    "bootstrap_fraction_change_gt_zero",
    "unchanged_same",
    "unchanged_no_signal",
    "added",
    "changed",
    "removed",
)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def cache_key(*values: str) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def read_gzip_json(path: Path) -> object:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def write_gzip_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as handle:
            handle.write(serialized)


def commit_sort_key(commit: Mapping[str, object]) -> tuple[str, str]:
    return str(commit.get("date") or ""), str(commit.get("id") or "")


def commits_within_snapshot(
    commits: Iterable[Mapping[str, object]],
    *,
    snapshot_date: str = SNAPSHOT_DATE,
) -> list[dict[str, object]]:
    cutoff = pd.Timestamp(snapshot_date, tz="UTC") + pd.Timedelta(days=1)
    rows: list[dict[str, object]] = []
    for commit in commits:
        created = pd.to_datetime(commit.get("date"), utc=True, errors="coerce")
        if pd.notna(created) and created < cutoff and commit.get("id"):
            rows.append(dict(commit))
    return sorted(rows, key=commit_sort_key)


def is_analysis_file(path: str) -> bool:
    name = PurePosixPath(path).name.casefold()
    if PurePosixPath(path).suffix.casefold() in CODE_SUFFIXES:
        return True
    if name.startswith("requirements") and name.endswith(".txt"):
        return True
    return name in TEXT_FILE_BASENAMES - {
        "readme.md",
        "license",
        "license.txt",
    }


def classify_transition(initial: set[str], current: set[str]) -> str:
    if initial == current:
        return "unchanged_same" if initial else "unchanged_no_signal"
    if not initial and current:
        return "added"
    if initial and not current:
        return "removed"
    return "changed"


def fractional_hhi(provider_sets: Mapping[str, set[str]]) -> float:
    eligible = {project: values for project, values in provider_sets.items() if values}
    if not eligible:
        return math.nan
    totals: Counter[str] = Counter()
    for providers in eligible.values():
        for provider in providers:
            totals[provider] += 1.0 / len(providers)
    shares = np.array(list(totals.values()), dtype=float) / len(eligible)
    return float(np.square(shares).sum())


def paired_version_bootstrap(
    initial_sets: Mapping[str, set[str]],
    current_sets: Mapping[str, set[str]],
    *,
    draws: int = HISTORY_BOOTSTRAP_DRAWS,
    seed: int = HISTORY_BOOTSTRAP_SEED,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Bootstrap current-minus-initial HHI on projects observed in both states."""

    projects = sorted(
        {
            project
            for project in set(initial_sets) & set(current_sets)
            if initial_sets[project] and current_sets[project]
        },
        key=str.casefold,
    )
    if not projects:
        return {
            "paired_projects": 0,
            "status": "NOT_ESTIMABLE",
            "reason": "no projects expose service signals in both versions",
        }, pd.DataFrame(columns=HISTORY_BOOTSTRAP_COLUMNS)
    labels = sorted(
        set().union(*(initial_sets[project] | current_sets[project] for project in projects)),
        key=str.casefold,
    )
    label_index = {label: index for index, label in enumerate(labels)}
    initial_matrix = np.zeros((len(projects), len(labels)), dtype=float)
    current_matrix = np.zeros_like(initial_matrix)
    for row_index, project in enumerate(projects):
        for provider in initial_sets[project]:
            initial_matrix[row_index, label_index[provider]] = 1.0 / len(
                initial_sets[project]
            )
        for provider in current_sets[project]:
            current_matrix[row_index, label_index[provider]] = 1.0 / len(
                current_sets[project]
            )
    initial_hhi = float(np.square(initial_matrix.mean(axis=0)).sum())
    current_hhi = float(np.square(current_matrix.mean(axis=0)).sum())
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    for draw in range(draws):
        indices = rng.integers(0, len(projects), size=len(projects))
        initial_value = float(np.square(initial_matrix[indices].mean(axis=0)).sum())
        current_value = float(np.square(current_matrix[indices].mean(axis=0)).sum())
        rows.append(
            {
                "draw": draw + 1,
                "initial_service_hhi": initial_value,
                "current_service_hhi": current_value,
                "hhi_change_current_minus_initial": current_value - initial_value,
            }
        )
    bootstrap = pd.DataFrame(rows)
    summary: dict[str, object] = {
        "status": "ESTIMATED",
        "paired_projects": len(projects),
        "provider_categories_across_versions": len(labels),
        "initial_service_hhi": initial_hhi,
        "current_service_hhi": current_hhi,
        "hhi_change_current_minus_initial": current_hhi - initial_hhi,
        "hhi_change_bootstrap_ci": [
            float(bootstrap["hhi_change_current_minus_initial"].quantile(0.025)),
            float(bootstrap["hhi_change_current_minus_initial"].quantile(0.975)),
        ],
        "bootstrap_fraction_change_gt_zero": float(
            bootstrap["hhi_change_current_minus_initial"].gt(0).mean()
        ),
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
        "boundary": (
            "version-paired descriptive bootstrap among projects with service signals in both "
            "states; not a p value, common-calendar panel, or survival estimate"
        ),
    }
    return summary, bootstrap


def _strict_project_ids(space_frame: pd.DataFrame) -> set[str]:
    strict_values = space_frame["included_strict"]
    strict_mask = strict_values.eq(True) | strict_values.astype(str).str.casefold().eq(
        "true"
    )
    return set(space_frame.loc[strict_mask, "space_id"].astype(str))


def _dependency_sets(
    edges: pd.DataFrame,
    project_ids: set[str],
    layer: str,
) -> dict[str, set[str]]:
    if edges.empty:
        return {}
    frame = edges.copy()
    frame["space_id"] = frame["space_id"].astype(str)
    frame = frame[
        frame["space_id"].isin(project_ids)
        & frame["layer"].eq(layer)
        & frame["provider"].notna()
        & frame["provider"].astype(str).ne("")
    ]
    return {
        str(space_id): set(group["provider"].astype(str))
        for space_id, group in frame.groupby("space_id", sort=True)
    }


def build_version_history_outputs(
    space_frame: pd.DataFrame,
    current_edges: pd.DataFrame,
    history_audit: pd.DataFrame,
    historical_edges: pd.DataFrame,
    *,
    draws: int = HISTORY_BOOTSTRAP_DRAWS,
    seed: int = HISTORY_BOOTSTRAP_SEED,
    span_thresholds: Iterable[int] = HISTORY_SPAN_THRESHOLDS,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute all version-history results from release-safe tabular evidence.

    The function performs no network or filesystem access. Raw commit payloads and
    source bodies are acquisition evidence only; the four published history outputs
    are deterministic functions of the processed edge tables, current Space frame,
    and machine-generated commit-history audit.
    """

    strict_ids = _strict_project_ids(space_frame)
    history_frame = history_audit.copy()
    history_frame["space_id"] = history_frame["space_id"].astype(str)
    history_frame = history_frame[history_frame["space_id"].isin(strict_ids)].copy()
    if history_frame["space_id"].duplicated().any():
        duplicates = sorted(
            history_frame.loc[
                history_frame["space_id"].duplicated(keep=False), "space_id"
            ].unique(),
            key=str.casefold,
        )
        raise ValueError(f"duplicate history-audit rows: {duplicates[:5]}")

    resolved_ids = set(
        history_frame.loc[
            history_frame["initial_state_status"].eq("RESOLVED"), "space_id"
        ]
    )
    initial_service = _dependency_sets(
        historical_edges, resolved_ids, "inference_service"
    )
    initial_runtime = _dependency_sets(historical_edges, resolved_ids, "local_runtime")
    current_service = _dependency_sets(current_edges, resolved_ids, "inference_service")
    current_runtime = _dependency_sets(current_edges, resolved_ids, "local_runtime")

    transition_rows: list[dict[str, object]] = []
    for space_id in sorted(resolved_ids, key=str.casefold):
        initial_service_set = initial_service.get(space_id, set())
        current_service_set = current_service.get(space_id, set())
        initial_runtime_set = initial_runtime.get(space_id, set())
        current_runtime_set = current_runtime.get(space_id, set())
        transition_rows.append(
            {
                "space_id": space_id,
                "initial_service_providers": ";".join(sorted(initial_service_set)),
                "current_service_providers": ";".join(sorted(current_service_set)),
                "service_transition": classify_transition(
                    initial_service_set, current_service_set
                ),
                "initial_runtime_categories": ";".join(sorted(initial_runtime_set)),
                "current_runtime_categories": ";".join(sorted(current_runtime_set)),
                "runtime_transition": classify_transition(
                    initial_runtime_set, current_runtime_set
                ),
            }
        )
    transitions = pd.DataFrame(transition_rows, columns=HISTORY_TRANSITION_COLUMNS)

    initial_service_complete = {
        space_id: initial_service.get(space_id, set()) for space_id in resolved_ids
    }
    current_service_complete = {
        space_id: current_service.get(space_id, set()) for space_id in resolved_ids
    }
    paired_summary, paired_draws = paired_version_bootstrap(
        initial_service_complete,
        current_service_complete,
        draws=draws,
        seed=seed,
    )

    history_spans = history_frame[["space_id", "history_span_days"]].copy()
    history_spans["history_span_days"] = pd.to_numeric(
        history_spans["history_span_days"], errors="coerce"
    )
    transition_with_span = transitions.merge(
        history_spans,
        on="space_id",
        how="left",
        validate="one_to_one",
    )
    thresholds = tuple(span_thresholds)
    span_rows: list[dict[str, object]] = []
    for minimum_days in thresholds:
        span_ids = set(
            transition_with_span.loc[
                transition_with_span["history_span_days"].ge(minimum_days),
                "space_id",
            ]
        )
        initial_subset = {
            space_id: initial_service_complete.get(space_id, set())
            for space_id in span_ids
        }
        current_subset = {
            space_id: current_service_complete.get(space_id, set())
            for space_id in span_ids
        }
        span_summary, _ = paired_version_bootstrap(
            initial_subset,
            current_subset,
            draws=draws,
            seed=seed + minimum_days,
        )
        transition_counts = transition_with_span.loc[
            transition_with_span["space_id"].isin(span_ids),
            "service_transition",
        ].value_counts()
        confidence_interval = span_summary.get(
            "hhi_change_bootstrap_ci", [None, None]
        )
        span_rows.append(
            {
                "minimum_history_span_days": minimum_days,
                "projects": len(span_ids),
                "paired_service_projects": span_summary.get("paired_projects", 0),
                "initial_service_hhi": span_summary.get("initial_service_hhi"),
                "current_service_hhi": span_summary.get("current_service_hhi"),
                "hhi_change_current_minus_initial": span_summary.get(
                    "hhi_change_current_minus_initial"
                ),
                "hhi_change_ci_low": confidence_interval[0],
                "hhi_change_ci_high": confidence_interval[1],
                "bootstrap_fraction_change_gt_zero": span_summary.get(
                    "bootstrap_fraction_change_gt_zero"
                ),
                "unchanged_same": int(transition_counts.get("unchanged_same", 0)),
                "unchanged_no_signal": int(
                    transition_counts.get("unchanged_no_signal", 0)
                ),
                "added": int(transition_counts.get("added", 0)),
                "changed": int(transition_counts.get("changed", 0)),
                "removed": int(transition_counts.get("removed", 0)),
            }
        )
    span_sensitivity = pd.DataFrame(span_rows, columns=HISTORY_SPAN_COLUMNS)

    span = pd.to_numeric(history_frame["history_span_days"], errors="coerce")
    commits = pd.to_numeric(
        history_frame["commit_count_through_snapshot"], errors="coerce"
    )
    strict_projects = len(strict_ids)
    result: dict[str, object] = {
        "snapshot_date": SNAPSHOT_DATE,
        "strict_projects": strict_projects,
        "commit_history_api_success": int(
            pd.to_numeric(
                history_frame["history_status_code"], errors="coerce"
            ).eq(200).sum()
        ),
        "projects_with_multiple_commits": int(commits.ge(2).sum()),
        "earliest_analyzable_state_resolved": len(resolved_ids),
        "earliest_state_resolution_share": (
            len(resolved_ids) / strict_projects if strict_projects else None
        ),
        "median_commits": float(commits.median()) if commits.notna().any() else None,
        "median_history_span_days_among_defined": (
            float(span.dropna().median()) if span.notna().any() else None
        ),
        "service_transition_counts": transitions[
            "service_transition"
        ].value_counts().to_dict(),
        "runtime_transition_counts": transitions[
            "runtime_transition"
        ].value_counts().to_dict(),
        "paired_service_concentration": paired_summary,
        "history_span_sensitivity_thresholds_days": list(thresholds),
        "manual_annotation_rows": 0,
        "selection_rule": (
            "earliest commit on or before snapshot with at least one bounded machine-selected "
            "executable or dependency-manifest file"
        ),
        "interpretive_boundary": (
            "conditions on projects visible in the 2026-08-31 strict sample; version ages vary, "
            "deleted projects are absent, and commit activity is not project survival"
        ),
    }
    return result, transitions, paired_draws, span_sensitivity


def recompute_version_history_outputs(
    root: Path,
    *,
    draws: int = HISTORY_BOOTSTRAP_DRAWS,
    seed: int = HISTORY_BOOTSTRAP_SEED,
    span_thresholds: Iterable[int] = HISTORY_SPAN_THRESHOLDS,
) -> dict[str, object]:
    """Rebuild the four published history outputs without consulting raw data."""

    root = root.resolve()
    processed = root / "data/processed"
    qa = root / "data/qa"
    results = root / "analysis_results"
    outputs = build_version_history_outputs(
        pd.read_csv(processed / "space_frame.csv"),
        pd.read_csv(processed / "dependency_edges.csv"),
        pd.read_csv(qa / "space_commit_history_audit.csv"),
        pd.read_csv(processed / "historical_dependency_edges.csv"),
        draws=draws,
        seed=seed,
        span_thresholds=span_thresholds,
    )
    summary, transitions, paired_draws, span_sensitivity = outputs
    results.mkdir(parents=True, exist_ok=True)
    transitions.to_csv(results / "version_dependency_transitions.csv", index=False)
    paired_draws.to_csv(results / "version_service_hhi_bootstrap.csv", index=False)
    span_sensitivity.to_csv(results / "version_span_sensitivity.csv", index=False)
    write_json(results / "version_history_summary.json", summary)
    return summary


class HistoricalCollector:
    def __init__(self, root: Path, *, refresh: bool, workers: int) -> None:
        self.root = root.resolve()
        self.refresh = refresh
        self.workers = workers
        self.cache = self.root / "data/raw/history_cache"
        self.processed = self.root / "data/processed"
        self.qa = self.root / "data/qa"
        self.results = self.root / "analysis_results"
        # Anonymous Hub traffic may route over a separately rate-limited IPv6
        # address. A single IPv4 connection keeps the resumable collector on a
        # stable public route without raising concurrency.
        self.client = HubClient(timeout=60.0, local_address="0.0.0.0")

    def close(self) -> None:
        self.client.close()

    def _cache_path(self, kind: str, *values: str) -> Path:
        return self.cache / kind / f"{cache_key(*values)}.json.gz"

    def list_commits(self, space_id: str) -> tuple[int, list[dict[str, object]], int]:
        path = self._cache_path("commits", space_id, SNAPSHOT_DATE)
        if path.exists() and not self.refresh:
            payload = read_gzip_json(path)
            return int(payload["status_code"]), list(payload["items"]), int(payload["pages"])
        encoded_space = quote(space_id, safe="/")
        url = f"{HF_BASE}/api/spaces/{encoded_space}/commits/main"
        items: list[dict[str, object]] = []
        status_code = 200
        pages = 0
        for page in range(HISTORY_MAX_PAGES):
            response = self.client.get(
                url,
                params={"p": page, "limit": HISTORY_PAGE_LIMIT},
            )
            status_code = response.status_code
            pages += 1
            if status_code != 200:
                break
            page_items = response.json()
            if not isinstance(page_items, list):
                raise TypeError(f"unexpected commit payload for {space_id}")
            items.extend(page_items)
            if len(page_items) < HISTORY_PAGE_LIMIT:
                break
        payload = {
            "space_id": space_id,
            "status_code": status_code,
            "pages": pages,
            "captured_at": utc_now(),
            "items": items,
        }
        write_gzip_json(path, payload)
        return status_code, items, pages

    def list_tree(self, space_id: str, revision: str) -> tuple[int, list[str]]:
        path = self._cache_path("trees", space_id, revision)
        if path.exists() and not self.refresh:
            payload = read_gzip_json(path)
            return int(payload["status_code"]), list(payload["files"])
        encoded_space = quote(space_id, safe="/")
        encoded_revision = quote(revision, safe="")
        url = f"{HF_BASE}/api/spaces/{encoded_space}/tree/{encoded_revision}"
        response = self.client.get(
            url,
            params={"recursive": "true", "expand": "false", "limit": 1000},
        )
        files: list[str] = []
        if response.status_code == 200:
            payload = response.json()
            if not isinstance(payload, list):
                raise TypeError(f"unexpected tree payload for {space_id}@{revision}")
            files = sorted(
                {
                    str(item["path"])
                    for item in payload
                    if isinstance(item, dict)
                    and item.get("type") == "file"
                    and item.get("path")
                },
                key=str.casefold,
            )
        write_gzip_json(
            path,
            {
                "space_id": space_id,
                "revision": revision,
                "status_code": response.status_code,
                "captured_at": utc_now(),
                "files": files,
            },
        )
        return response.status_code, files

    def fetch_file(self, space_id: str, revision: str, file_path: str) -> tuple[int, str | None]:
        path = self._cache_path("files", space_id, revision, file_path)
        stale_payload: dict[str, object] | None = None
        if path.exists() and not self.refresh:
            payload = read_gzip_json(path)
            if cached_text_matches(payload):
                return int(payload["status_code"]), payload.get("text")
            stale_payload = payload
        encoded_space = quote(space_id, safe="/")
        encoded_revision = quote(revision, safe="")
        encoded_path = quote(file_path, safe="/")
        url = f"{HF_BASE}/spaces/{encoded_space}/resolve/{encoded_revision}/{encoded_path}"
        response = self.client.get(url)
        response_sha = hashlib.sha256(response.content).hexdigest()
        if stale_payload is not None and (
            response.status_code != int(stale_payload["status_code"])
            or len(response.content) != int(stale_payload["bytes"])
            or response_sha != str(stale_payload["sha256"])
        ):
            raise RuntimeError(
                f"pinned historical source changed while repairing cache: {space_id}@{revision}/{file_path}"
            )
        text: str | None = None
        source_encoding: str | None = None
        if response.status_code == 200 and len(response.content) <= MAX_SOURCE_BYTES:
            text, source_encoding = decode_source_bytes(response.content)
        write_gzip_json(
            path,
            {
                "space_id": space_id,
                "revision": revision,
                "file_path": file_path,
                "status_code": response.status_code,
                "bytes": len(response.content),
                "sha256": response_sha,
                "captured_at": utc_now(),
                "decoder_version": "lossless-v2",
                "source_encoding": source_encoding,
                "text": text,
            },
        )
        return response.status_code, text

    def collect_project(self, space_id: str, current_revision: str) -> tuple[dict[str, object], list[dict[str, object]]]:
        status_code, raw_commits, pages = self.list_commits(space_id)
        commits = commits_within_snapshot(raw_commits)
        summary: dict[str, object] = {
            "space_id": space_id,
            "current_revision": current_revision,
            "history_status_code": status_code,
            "history_pages": pages,
            "commit_count_through_snapshot": len(commits),
            "first_commit_date": commits[0].get("date") if commits else None,
            "last_commit_date": commits[-1].get("date") if commits else None,
            "initial_revision": None,
            "initial_revision_date": None,
            "initial_revision_rank": None,
            "initial_tree_status_code": None,
            "initial_selected_file_count": 0,
            "initial_dependency_signal_count": 0,
            "initial_state_status": "NO_COMMIT_HISTORY" if not commits else "NO_ANALYZABLE_REVISION",
            "current_revision_in_history": current_revision in {
                str(commit.get("id")) for commit in commits
            },
        }
        if len(commits) >= 2:
            first = pd.to_datetime(commits[0].get("date"), utc=True, errors="coerce")
            last = pd.to_datetime(commits[-1].get("date"), utc=True, errors="coerce")
            if pd.notna(first) and pd.notna(last):
                summary["history_span_days"] = (last - first).total_seconds() / 86_400
        edges: list[dict[str, object]] = []
        for rank, commit in enumerate(commits, start=1):
            revision = str(commit["id"])
            tree_status, tree_files = self.list_tree(space_id, revision)
            selected = select_repository_files(tree_files, None)
            if not any(is_analysis_file(path) for path in selected):
                continue
            files: dict[str, str] = {}
            for file_path in selected:
                _, text = self.fetch_file(space_id, revision, file_path)
                if text is not None:
                    files[file_path] = text
            if not any(is_analysis_file(path) for path in files):
                continue
            signals = [
                signal
                for signal in provider_signals(files)
                if signal.layer in {"inference_service", "local_runtime"}
            ]
            edges = [
                {
                    "space_id": space_id,
                    "revision_role": "earliest_analyzable",
                    "revision": revision,
                    "revision_date": commit.get("date"),
                    **signal.to_dict(),
                }
                for signal in signals
            ]
            summary.update(
                {
                    "initial_revision": revision,
                    "initial_revision_date": commit.get("date"),
                    "initial_revision_rank": rank,
                    "initial_tree_status_code": tree_status,
                    "initial_selected_file_count": len(files),
                    "initial_dependency_signal_count": len(signals),
                    "initial_state_status": "RESOLVED",
                }
            )
            break
        return summary, edges

    def run(self) -> dict[str, object]:
        spaces = pd.read_csv(self.processed / "space_frame.csv")
        strict = spaces[spaces["included_strict"].eq(True)].copy()
        summaries: list[dict[str, object]] = []
        historical_edges: list[dict[str, object]] = []
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {
                executor.submit(
                    self.collect_project,
                    str(row.space_id),
                    str(row.revision),
                ): str(row.space_id)
                for row in strict.itertuples(index=False)
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                space_id = futures[future]
                try:
                    summary, edges = future.result()
                except (httpx.HTTPError, RuntimeError, ValueError, TypeError, OSError) as exc:
                    summary = {
                        "space_id": space_id,
                        "initial_state_status": "ERROR",
                        "error_type": type(exc).__name__,
                    }
                    edges = []
                summaries.append(summary)
                historical_edges.extend(edges)
                if completed % 25 == 0 or completed == len(futures):
                    print(f"historical versions: {completed}/{len(futures)}", flush=True)

        history_frame = pd.DataFrame(summaries).sort_values("space_id")
        edge_columns = [
            "space_id",
            "revision_role",
            "revision",
            "revision_date",
            "provider",
            "layer",
            "evidence_type",
            "evidence_value",
            "source_file",
            "confidence",
        ]
        historical_edge_frame = pd.DataFrame(historical_edges, columns=edge_columns)
        history_frame.to_csv(self.qa / "space_commit_history_audit.csv", index=False)
        historical_edge_frame.to_csv(
            self.processed / "historical_dependency_edges.csv", index=False
        )
        return recompute_version_history_outputs(self.root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collector = HistoricalCollector(
        args.root,
        refresh=args.refresh,
        workers=max(1, args.workers),
    )
    try:
        result = collector.run()
    finally:
        collector.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
