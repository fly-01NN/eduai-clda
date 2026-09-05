"""Collect and parse public GitHub repositories linked from Space READMEs.

README links can denote a project mirror, dependency, example, or template. The
collector therefore writes a supplementary audit and never merges these edges
into the primary concentration analysis without a validated relation label.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any
from urllib.parse import quote

import httpx
import pandas as pd

from dependency_parser import (
    extract_code_model_ids,
    provider_signals,
    select_repository_files,
)
from protocol import MAX_SOURCE_BYTES, SNAPSHOT_DATE, model_family, model_provider


GITHUB_API = "https://api.github.com"
SUPPLEMENTARY_BOUNDARY = (
    "README links may identify mirrors, dependencies, examples, or templates; "
    "derived edges are not part of the primary concentration estimand"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def linked_repository_map(spaces: pd.DataFrame) -> dict[str, list[str]]:
    """Return case-insensitively de-duplicated repository-to-Space links."""

    repositories: dict[str, dict[str, object]] = {}
    for row in spaces.itertuples(index=False):
        for raw in str(getattr(row, "github_repositories", "") or "").split(";"):
            repository = raw.strip().strip("/")
            if repository.casefold() in {"", "nan"} or repository.count("/") != 1:
                continue
            key = repository.casefold()
            item = repositories.setdefault(key, {"repository": repository, "spaces": set()})
            item["spaces"].add(str(row.space_id))
    return {
        str(item["repository"]): sorted(item["spaces"], key=str.casefold)
        for item in repositories.values()
    }


class GitHubClient:
    def __init__(self, token: str | None = None) -> None:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": os.environ.get(
                "EDUAI_CLDA_USER_AGENT", "EduAI-CLDA/0.1.0"
            ),
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.client = httpx.Client(timeout=60, follow_redirects=True, headers=headers)
        self.requests = 0
        self.retries = 0

    def close(self) -> None:
        self.client.close()

    def get(self, url: str) -> httpx.Response:
        for attempt in range(5):
            self.requests += 1
            response = self.client.get(url)
            if response.status_code in {429, 500, 502, 503, 504}:
                self.retries += 1
                time.sleep(min(8.0, 0.5 * (2**attempt)))
                continue
            return response
        return response


class GitHubCollector:
    def __init__(self, root: Path, *, refresh: bool, token: str | None) -> None:
        self.root = root.resolve()
        self.refresh = refresh
        self.raw = self.root / "data/raw/github" / SNAPSHOT_DATE
        self.file_cache = self.root / "data/raw/github_file_cache"
        self.processed = self.root / "data/processed"
        self.qa = self.root / "data/qa"
        self.client = GitHubClient(token)
        self.file_manifest: list[dict[str, object]] = []

    def close(self) -> None:
        self.client.close()

    def cached_listing(self, repository: str, subpath: str = "") -> Any:
        slug = hashlib.sha256(f"{repository}\n{subpath}".encode()).hexdigest()[:20]
        path = self.raw / f"contents__{slug}.json"
        if path.exists() and not self.refresh:
            return json.loads(path.read_text(encoding="utf-8"))
        suffix = f"/contents/{quote(subpath, safe='/')}" if subpath else "/contents"
        response = self.client.get(f"{GITHUB_API}/repos/{repository}{suffix}")
        if response.status_code == 200:
            payload: Any = response.json()
        else:
            payload = {
                "_collection_status": response.status_code,
                "_message": response.text[:500],
            }
        write_json(path, payload)
        return payload

    def fetch_file(self, repository: str, item: dict[str, Any]) -> tuple[str, str | None]:
        file_path = str(item.get("path") or item.get("name") or "")
        sha = str(item.get("sha") or "unresolved")
        cache_key = hashlib.sha256(f"{repository}\n{sha}\n{file_path}".encode()).hexdigest()
        cache_path = self.file_cache / f"{cache_key}.json.gz"
        if cache_path.exists() and not self.refresh:
            with gzip.open(cache_path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.file_manifest.append({key: value for key, value in payload.items() if key != "text"})
            return file_path, payload.get("text")

        download_url = item.get("download_url")
        text: str | None = None
        content = b""
        status_code = 0
        if download_url:
            response = self.client.get(str(download_url))
            status_code = response.status_code
            if response.status_code == 200:
                content = response.content
        elif item.get("url"):
            response = self.client.get(str(item["url"]))
            status_code = response.status_code
            if response.status_code == 200:
                encoded = response.json().get("content") or ""
                content = base64.b64decode(encoded)
        if len(content) <= MAX_SOURCE_BYTES and status_code == 200:
            text = content.decode("utf-8", errors="replace")
        payload = {
            "github_repository": repository,
            "file_path": file_path,
            "git_blob_sha": sha,
            "public_url": item.get("html_url") or download_url or item.get("url") or "",
            "status_code": status_code,
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "captured_at": utc_now(),
            "text": text,
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        with cache_path.open("wb") as raw_handle:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as handle:
                handle.write(serialized)
        self.file_manifest.append({key: value for key, value in payload.items() if key != "text"})
        return file_path, text

    def repository_items(self, repository: str) -> tuple[list[dict[str, Any]], int]:
        root = self.cached_listing(repository)
        if not isinstance(root, list):
            return [], int(root.get("_collection_status", 0)) if isinstance(root, dict) else 0
        items = [item for item in root if isinstance(item, dict) and item.get("type") == "file"]
        # One bounded nested lookup captures common source entry points while
        # keeping unauthenticated API use below its documented hourly limit.
        src = next(
            (
                item
                for item in root
                if isinstance(item, dict)
                and item.get("type") == "dir"
                and str(item.get("name", "")).casefold() == "src"
            ),
            None,
        )
        if src:
            nested = self.cached_listing(repository, str(src.get("path") or "src"))
            if isinstance(nested, list):
                items.extend(
                    item
                    for item in nested
                    if isinstance(item, dict) and item.get("type") == "file"
                )
        return items, 200

    def run(self) -> dict[str, object]:
        spaces = pd.read_csv(self.processed / "space_frame.csv")
        spaces = spaces[spaces["included_strict"].eq(True)]
        repository_map = linked_repository_map(spaces)
        repository_rows: list[dict[str, object]] = []
        relation_rows: list[dict[str, str]] = []
        edge_rows: list[dict[str, object]] = []

        for repository in sorted(repository_map, key=str.casefold):
            source_spaces = repository_map[repository]
            relation_rows.extend(
                {"space_id": space_id, "github_repository": repository}
                for space_id in source_spaces
            )
            items, status = self.repository_items(repository)
            by_path = {str(item.get("path") or item.get("name")): item for item in items}
            selected_paths = select_repository_files(by_path, None)
            files: dict[str, str] = {}
            for path in selected_paths:
                returned_path, text = self.fetch_file(repository, by_path[path])
                if text is not None:
                    files[returned_path] = text
            signals = provider_signals(files)
            model_ids = extract_code_model_ids(files)
            repository_rows.append(
                {
                    "github_repository": repository,
                    "collection_status": status,
                    "linked_space_count": len(source_spaces),
                    "listed_files": len(items),
                    "selected_files": len(selected_paths),
                    "retrieved_files": len(files),
                    "provider_signals": len(signals),
                    "code_model_ids": len(model_ids),
                    "interpretive_boundary": SUPPLEMENTARY_BOUNDARY,
                }
            )
            for space_id in source_spaces:
                for signal in signals:
                    edge_rows.append(
                        {
                            "space_id": space_id,
                            "github_repository": repository,
                            **signal.to_dict(),
                            "model_family": "",
                            "provider_basis": signal.evidence_type,
                        }
                    )
                for model_id in model_ids:
                    provider, basis = model_provider(model_id)
                    edge_rows.append(
                        {
                            "space_id": space_id,
                            "github_repository": repository,
                            "provider": provider,
                            "layer": "model_dependency",
                            "evidence_type": "github_code_model_id",
                            "evidence_value": model_id,
                            "source_file": "selected_github_source_files",
                            "confidence": "medium",
                            "model_family": model_family(model_id),
                            "provider_basis": basis,
                        }
                    )

        repository_frame = pd.DataFrame(repository_rows)
        relation_frame = pd.DataFrame(relation_rows)
        edge_frame = pd.DataFrame(edge_rows)
        repository_frame.to_csv(self.processed / "github_repository_frame.csv", index=False)
        relation_frame.to_csv(self.processed / "space_github_links.csv", index=False)
        edge_frame.to_csv(self.processed / "github_dependency_edges.csv", index=False)
        pd.DataFrame(self.file_manifest).drop_duplicates(
            subset=["github_repository", "git_blob_sha", "file_path"]
        ).sort_values(["github_repository", "file_path"]).to_csv(
            self.qa / "github_file_manifest.csv", index=False
        )
        manifest = {
            "snapshot_date": SNAPSHOT_DATE,
            "strict_spaces_with_github_link": int(relation_frame["space_id"].nunique()),
            "unique_readme_linked_repositories": len(repository_map),
            "repositories_resolved": int(repository_frame["collection_status"].eq(200).sum()),
            "repositories_unresolved": int(repository_frame["collection_status"].ne(200).sum()),
            "retrieved_files": int(repository_frame["retrieved_files"].sum()),
            "supplementary_dependency_edges": len(edge_frame),
            "http_requests_this_run": self.client.requests,
            "http_retries_this_run": self.client.retries,
            "raw_source_cache_released": False,
            "interpretive_boundary": SUPPLEMENTARY_BOUNDARY,
        }
        write_json(self.qa / "github_collection_manifest.json", manifest)
        return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--github-token-env",
        default="GITHUB_TOKEN",
        help="environment-variable name containing an optional GitHub token",
    )
    args = parser.parse_args()
    collector = GitHubCollector(
        args.root,
        refresh=args.refresh,
        token=os.environ.get(args.github_token_env),
    )
    try:
        manifest = collector.run()
    finally:
        collector.close()
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
