"""Collect a frozen, query-defined sample of public educational AI Spaces.

The collector caches public source files locally for reproducibility but only
releases checksums and derived signals by default. It uses no manual relevance
labels and never treats a Space hosting-region tag as developer geography.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Iterable
from urllib.parse import quote

import httpx
import pandas as pd

from dependency_parser import (
    extract_code_model_ids,
    extract_github_repositories,
    has_machine_service_candidate,
    provider_signals,
    select_repository_files,
    unmapped_dependency_candidates,
)
from geography import classify_location, language_orientation
from license_rules import (
    detect_license_from_text,
    normalize_license,
    rights_review_flag,
)
from protocol import (
    MAX_SOURCE_BYTES,
    PER_QUERY_LIMIT,
    PROTOCOL_VERSION,
    SEARCH_ARMS,
    SEARCH_TERMS,
    SNAPSHOT_DATE,
    match_education_constructs,
    model_family,
    model_provider,
)
from source_encoding import cached_text_matches, decode_source_bytes


HF_BASE = "https://huggingface.co"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-").casefold()
    return cleaned or hashlib.sha256(value.encode()).hexdigest()[:12]


def education_metadata_fields(item: dict, readme: str) -> list[str]:
    """Return independent relevance fields without the author namespace."""

    card = item.get("cardData") or {}
    space_slug = str(item.get("id") or "").split("/", 1)[-1]
    return [
        space_slug,
        str(card.get("title") or ""),
        str(card.get("short_description") or ""),
        " ".join(str(value) for value in (item.get("tags") or [])),
        readme,
    ]


def education_metadata_text(item: dict, readme: str) -> str:
    """Build display-only relevance text from the independent fields."""

    return " ".join(education_metadata_fields(item, readme))


def education_metadata_constructs(
    item: dict,
    readme: str,
    *,
    broad: bool = False,
) -> list[str]:
    """Match each metadata field independently, then take a deterministic union."""

    return list(
        dict.fromkeys(
            construct
            for field in education_metadata_fields(item, readme)
            for construct in match_education_constructs(field, broad=broad)
        )
    )


class HubClient:
    """Small retrying client with request accounting."""

    def __init__(
        self,
        timeout: float = 45.0,
        *,
        local_address: str | None = None,
    ) -> None:
        transport = (
            httpx.HTTPTransport(local_address=local_address)
            if local_address is not None
            else None
        )
        self.client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={
                "User-Agent": os.environ.get(
                    "EDUAI_CLDA_USER_AGENT", "EduAI-CLDA/0.1.0"
                )
            },
            transport=transport,
        )
        self.requests = 0
        self.retries = 0

    def close(self) -> None:
        self.client.close()

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        last: Exception | None = None
        for attempt in range(5):
            try:
                self.requests += 1
                response = self.client.get(url, **kwargs)
                if response.status_code == 429:
                    self.retries += 1
                    retry_after = response.headers.get("retry-after")
                    delay = min(30.0, float(retry_after) if retry_after else 2.0 ** attempt)
                    time.sleep(delay)
                    continue
                if response.status_code >= 500:
                    self.retries += 1
                    time.sleep(min(8.0, 0.5 * 2.0 ** attempt))
                    continue
                return response
            except httpx.HTTPError as exc:
                last = exc
                self.retries += 1
                time.sleep(min(8.0, 0.5 * 2.0 ** attempt))
        if last:
            raise last
        raise RuntimeError(f"request failed after retries: {url}")


class Collector:
    def __init__(self, root: Path, *, refresh: bool, workers: int) -> None:
        self.root = root.resolve()
        self.refresh = refresh
        self.workers = workers
        self.raw = self.root / "data/raw" / SNAPSHOT_DATE
        self.cache = self.root / "data/raw/file_cache"
        self.processed = self.root / "data/processed"
        self.qa = self.root / "data/qa"
        self.client = HubClient()
        self.file_manifest: list[dict[str, object]] = []

    def close(self) -> None:
        self.client.close()

    def cached_api_json(self, relative: Path, url: str, params: Any = None) -> Any:
        path = self.raw / relative
        if path.exists() and not self.refresh:
            return read_json(path)
        response = self.client.get(url, params=params)
        if response.status_code in {400, 401, 403, 404, 410}:
            payload: Any = {"_collection_status": response.status_code}
        else:
            response.raise_for_status()
            payload = response.json()
        write_json(path, payload)
        return payload

    def search(self) -> tuple[dict[str, dict], dict[str, set[str]]]:
        spaces: dict[str, dict] = {}
        hits: dict[str, set[str]] = {}
        for term in SEARCH_TERMS:
            for arm in SEARCH_ARMS:
                relative = Path("search_expanded") / f"{safe_slug(term)}__{safe_slug(arm)}.json"
                params: list[tuple[str, object]] = [
                    ("search", term),
                    ("limit", PER_QUERY_LIMIT),
                    ("sort", arm),
                    ("direction", -1),
                ]
                for field in (
                    "author", "cardData", "createdAt", "datasets", "disabled",
                    "lastModified", "likes", "models", "runtime", "sdk",
                    "sha", "siblings", "tags",
                ):
                    params.append(("expand", field))
                payload = self.cached_api_json(
                    relative,
                    f"{HF_BASE}/api/spaces",
                    params,
                )
                if not isinstance(payload, list):
                    raise TypeError(f"unexpected search payload for {term}/{arm}")
                for item in payload:
                    space_id = item.get("id")
                    if not space_id:
                        continue
                    spaces[space_id] = item
                    hits.setdefault(space_id, set()).add(f"{term}|{arm}")
        return spaces, hits

    def _file_cache_path(self, space_id: str, revision: str, file_path: str) -> Path:
        key = f"{space_id}\n{revision}\n{file_path}".encode("utf-8")
        return self.cache / f"{hashlib.sha256(key).hexdigest()}.json.gz"

    def fetch_file(self, space_id: str, revision: str, file_path: str) -> tuple[str, str | None]:
        cache_path = self._file_cache_path(space_id, revision, file_path)
        stale_payload: dict[str, object] | None = None
        if cache_path.exists() and not self.refresh:
            with gzip.open(cache_path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
            if cached_text_matches(payload):
                self.file_manifest.append({key: value for key, value in payload.items() if key != "text"})
                return file_path, payload.get("text")
            stale_payload = payload
        encoded_space = quote(space_id, safe="/")
        encoded_revision = quote(revision or "main", safe="")
        encoded_path = quote(file_path, safe="/")
        url = f"{HF_BASE}/spaces/{encoded_space}/resolve/{encoded_revision}/{encoded_path}"
        response = self.client.get(url)
        response_sha = sha256_bytes(response.content)
        if stale_payload is not None and (
            response.status_code != int(stale_payload["status_code"])
            or len(response.content) != int(stale_payload["bytes"])
            or response_sha != str(stale_payload["sha256"])
        ):
            raise RuntimeError(
                f"pinned source changed while repairing cache: {space_id}@{revision}/{file_path}"
            )
        if response.status_code == 404:
            content = b""
            text = None
            source_encoding = None
        else:
            response.raise_for_status()
            content = response.content[: MAX_SOURCE_BYTES + 1]
            if len(content) > MAX_SOURCE_BYTES:
                text = None
                source_encoding = None
            else:
                text, source_encoding = decode_source_bytes(content)
        payload = {
            "space_id": space_id,
            "revision": revision,
            "file_path": file_path,
            "public_url": url,
            "status_code": response.status_code,
            "bytes": len(response.content),
            "sha256": response_sha,
            "captured_at": utc_now(),
            "decoder_version": "lossless-v2",
            "source_encoding": source_encoding,
            "text": text,
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        with cache_path.open("wb") as raw_handle:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as handle:
                handle.write(serialized)
        self.file_manifest.append({key: value for key, value in payload.items() if key != "text"})
        return file_path, text

    @staticmethod
    def sibling_names(item: dict) -> list[str]:
        output = []
        for sibling in item.get("siblings") or []:
            if isinstance(sibling, dict) and sibling.get("rfilename"):
                output.append(str(sibling["rfilename"]))
            elif isinstance(sibling, str):
                output.append(sibling)
        return sorted(set(output))

    def fetch_files(self, item: dict, paths: Iterable[str]) -> dict[str, str]:
        space_id = item["id"]
        revision = item.get("sha") or "main"
        path_list = sorted(set(paths))
        files: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {
                executor.submit(self.fetch_file, space_id, revision, path): path
                for path in path_list
            }
            for future in as_completed(futures):
                path, text = future.result()
                if text is not None:
                    files[path] = text
        return files

    def collect_candidate(
        self, item: dict, discovery_hits: set[str]
    ) -> tuple[dict, list[dict], list[dict]]:
        siblings = self.sibling_names(item)
        readme_name = next((name for name in siblings if name.casefold() == "readme.md"), "README.md")
        readme_files = self.fetch_files(item, [readme_name])
        readme = readme_files.get(readme_name, "")
        card = item.get("cardData") or {}
        strict_constructs = education_metadata_constructs(item, readme)
        broad_constructs = education_metadata_constructs(item, readme, broad=True)
        if not broad_constructs:
            return (
                self._audit_row(
                    item,
                    discovery_hits,
                    [],
                    [],
                    False,
                    False,
                    "no_education_function_phrase",
                ),
                [],
                [],
            )

        app_file = card.get("app_file")
        selected = select_repository_files(siblings, str(app_file) if app_file else None)
        files = dict(readme_files)
        remaining = [path for path in selected if path not in files]
        files.update(self.fetch_files(item, remaining))
        signals = provider_signals(files)
        unmapped_candidates = unmapped_dependency_candidates(files)
        linked_models = sorted(
            {
                str(model) for model in (item.get("models") or []) if str(model).strip()
            }
            | set(extract_code_model_ids(files)),
            key=str.casefold,
        )
        ai_signal_layers = {"inference_service", "local_runtime"}
        has_identifiable_ai_edge = bool(linked_models) or any(
            signal.layer in ai_signal_layers for signal in signals
        )
        has_candidate_service = has_machine_service_candidate(unmapped_candidates)
        has_ai_evidence = has_identifiable_ai_edge or has_candidate_service
        has_known_service = any(signal.layer == "inference_service" for signal in signals)
        service_candidate_count = sum(
            candidate.candidate_type != "unmapped_manifest_package"
            for candidate in unmapped_candidates
        )
        included_broad = bool(broad_constructs and has_ai_evidence)
        included_strict = bool(strict_constructs and has_ai_evidence)
        if not has_ai_evidence:
            reason = "no_model_or_inference_dependency"
        elif not strict_constructs:
            reason = "broad_only"
        else:
            reason = "included_strict"

        card_license = normalize_license(card.get("license"))
        detected_license = card_license
        license_basis = "card_metadata" if card_license != "missing" else "missing"
        if card_license == "missing":
            for path, text in files.items():
                if Path(path).name.casefold() in {"license", "license.txt"}:
                    detected = detect_license_from_text(text)
                    if detected != "missing":
                        detected_license = detected
                        license_basis = "license_text_fingerprint"
                        break
        github_repositories = extract_github_repositories(readme)
        language_class, language_codes = language_orientation(
            [str(value) for value in (item.get("tags") or [])], card.get("language")
        )
        row = self._audit_row(
            item,
            discovery_hits,
            strict_constructs,
            broad_constructs,
            included_strict,
            included_broad,
            reason,
        )
        row.update(
            {
                "app_license": detected_license,
                "app_license_basis": license_basis,
                "linked_model_ids": ";".join(linked_models),
                "linked_model_count": len(linked_models),
                "provider_signal_count": len(signals),
                "dependency_observable": has_ai_evidence,
                "known_dependency_observable": has_identifiable_ai_edge,
                "identifiable_dependency_observable": has_identifiable_ai_edge,
                "known_service_observable": has_known_service,
                "machine_service_candidate_observable": has_candidate_service,
                "unknown_service_candidate_count": service_candidate_count,
                "unmapped_manifest_package_count": sum(
                    candidate.candidate_type == "unmapped_manifest_package"
                    for candidate in unmapped_candidates
                ),
                "github_repositories": ";".join(github_repositories),
                "github_repository_count": len(github_repositories),
                "declared_language_class": language_class,
                "declared_language_codes": language_codes,
            }
        )
        signal_rows = [
            {"space_id": item["id"], **signal.to_dict()} for signal in signals
        ]
        for model_id in linked_models:
            signal_rows.append(
                {
                    "space_id": item["id"],
                    "provider": "",
                    "layer": "model_dependency",
                    "evidence_type": "hf_linked_model" if model_id in (item.get("models") or []) else "code_model_id",
                    "evidence_value": model_id,
                    "source_file": "hub_metadata" if model_id in (item.get("models") or []) else "selected_source_files",
                    "confidence": "high" if model_id in (item.get("models") or []) else "medium",
                }
            )
        candidate_rows = [
            {"space_id": item["id"], **candidate.to_dict()}
            for candidate in unmapped_candidates
        ]
        return row, signal_rows, candidate_rows

    @staticmethod
    def _audit_row(
        item: dict,
        discovery_hits: set[str],
        strict_constructs: list[str],
        broad_constructs: list[str],
        included_strict: bool,
        included_broad: bool,
        reason: str,
    ) -> dict:
        card = item.get("cardData") or {}
        runtime = item.get("runtime") or {}
        return {
            "space_id": item.get("id"),
            "author": item.get("author") or str(item.get("id", "")).split("/", 1)[0],
            "title": card.get("title") or "",
            "created_at": item.get("createdAt") or "",
            "last_modified": item.get("lastModified") or "",
            "likes": item.get("likes") or 0,
            "sdk": item.get("sdk") or card.get("sdk") or "",
            "runtime_stage": runtime.get("stage") or "unreported",
            "disabled": bool(item.get("disabled", False)),
            "hosting_region": item.get("region") or "",
            "revision": item.get("sha") or "",
            "discovery_hits": ";".join(sorted(discovery_hits)),
            "discovery_queries": ";".join(sorted({hit.split("|", 1)[0] for hit in discovery_hits})),
            "discovery_arms": ";".join(sorted({hit.split("|", 1)[1] for hit in discovery_hits})),
            "strict_constructs": ";".join(strict_constructs),
            "broad_constructs": ";".join(broad_constructs),
            "education_strict_match": bool(strict_constructs),
            "education_broad_match": bool(broad_constructs),
            "included_strict": included_strict,
            "included_broad": included_broad,
            "selection_status": reason,
        }

    def author_profile(self, author: str) -> dict:
        relative_user = Path("profiles/users") / f"{safe_slug(author)}.json"
        payload = self.cached_api_json(
            relative_user,
            f"{HF_BASE}/api/users/{quote(author, safe='')}/overview",
        )
        profile_type = "user"
        if isinstance(payload, dict) and payload.get("_collection_status"):
            relative_org = Path("profiles/organizations") / f"{safe_slug(author)}.json"
            payload = self.cached_api_json(
                relative_org,
                f"{HF_BASE}/api/organizations/{quote(author, safe='')}/overview",
            )
            profile_type = "organization"
        if not isinstance(payload, dict) or payload.get("_collection_status"):
            return {"profile_type": "unresolved", "location": "", "fullname": ""}
        return {
            "profile_type": profile_type,
            "location": payload.get("location") or "",
            "fullname": payload.get("fullname") or payload.get("name") or "",
        }

    def model_info(self, model_id: str) -> dict:
        key = hashlib.sha256(model_id.encode()).hexdigest()[:16]
        payload = self.cached_api_json(
            Path("models") / f"{safe_slug(model_id)[:80]}__{key}.json",
            f"{HF_BASE}/api/models/{quote(model_id, safe='/')}",
        )
        if not isinstance(payload, dict) or payload.get("_collection_status"):
            return {
                "model_id": model_id,
                "resolved": False,
                "resolution_status": (
                    payload.get("_collection_status") if isinstance(payload, dict) else "invalid_payload"
                ),
            }
        card = payload.get("cardData") or {}
        base_models = normalize_base_models(card.get("base_model"), payload.get("tags") or [])
        return {
            "model_id": model_id,
            "resolved": True,
            "resolution_status": 200,
            "author": payload.get("author") or model_id.split("/", 1)[0],
            "base_models": ";".join(base_models),
            "license": normalize_license(card.get("license")),
            "gated": payload.get("gated", False),
            "downloads": payload.get("downloads") or 0,
            "likes": payload.get("likes") or 0,
            "created_at": payload.get("createdAt") or "",
            "last_modified": payload.get("lastModified") or "",
            "pipeline_tag": payload.get("pipeline_tag") or "",
        }

    def run(self) -> dict:
        started = utc_now()
        spaces, hits = self.search()
        candidate_rows: list[dict] = []
        signal_rows: list[dict] = []
        unmapped_candidate_rows: list[dict] = []
        # Repository-level fetches are parallelized by file, while candidate
        # order stays deterministic for audit output.
        for index, space_id in enumerate(sorted(spaces, key=str.casefold), start=1):
            row, signals, unmapped = self.collect_candidate(
                spaces[space_id], hits.get(space_id, set())
            )
            candidate_rows.append(row)
            signal_rows.extend(signals)
            unmapped_candidate_rows.extend(unmapped)
            if index % 100 == 0:
                print(f"processed {index}/{len(spaces)} discovery candidates", flush=True)

        candidates = pd.DataFrame(candidate_rows).sort_values("space_id", key=lambda s: s.str.casefold())
        self.qa.mkdir(parents=True, exist_ok=True)
        self.processed.mkdir(parents=True, exist_ok=True)
        candidates.to_csv(self.qa / "candidate_selection_audit.csv", index=False)
        candidates[candidates["education_strict_match"].eq(True)].to_csv(
            self.processed / "education_phrase_frame.csv", index=False
        )
        unmapped_frame = pd.DataFrame(
            unmapped_candidate_rows,
            columns=["space_id", "candidate_type", "identifier", "source_file"],
        ).drop_duplicates()
        unmapped_frame.sort_values(
            ["space_id", "candidate_type", "identifier", "source_file"],
            key=lambda series: series.astype(str).str.casefold(),
        ).to_csv(self.qa / "unmapped_dependency_candidates.csv", index=False)
        selected = candidates[candidates["included_broad"].eq(True)].copy()

        profiles = {
            author: self.author_profile(author)
            for author in sorted(set(selected["author"].astype(str)), key=str.casefold)
        }
        selected["author_profile_type"] = selected["author"].map(lambda value: profiles[str(value)]["profile_type"])
        selected["author_location"] = selected["author"].map(lambda value: profiles[str(value)]["location"])
        location_values = selected["author_location"].map(classify_location)
        selected["author_region_class"] = location_values.map(lambda value: value[0])
        selected["author_country_code"] = location_values.map(lambda value: value[1])

        model_ids = sorted(
            {
                model_id
                for value in selected["linked_model_ids"].fillna("")
                for model_id in str(value).split(";")
                if model_id
            },
            key=str.casefold,
        )
        model_rows = [self.model_info(model_id) for model_id in model_ids]
        models = pd.DataFrame(model_rows)
        if models.empty:
            models = pd.DataFrame(
                columns=["model_id", "resolved", "base_models", "license"]
            )
        models["base_models"] = models.get("base_models", pd.Series(dtype=str)).fillna("")
        models["license"] = models.get("license", pd.Series(dtype=str)).fillna("missing")
        models["provider"] = models.apply(
            lambda row: model_provider(
                row["model_id"], [value for value in str(row["base_models"]).split(";") if value]
            )[0],
            axis=1,
        )
        models["provider_basis"] = models.apply(
            lambda row: model_provider(
                row["model_id"], [value for value in str(row["base_models"]).split(";") if value]
            )[1],
            axis=1,
        )
        models["model_family"] = models.apply(
            lambda row: model_family(
                row["model_id"], [value for value in str(row["base_models"]).split(";") if value]
            ),
            axis=1,
        )
        model_lookup = models.set_index("model_id").to_dict("index") if not models.empty else {}

        for row in signal_rows:
            if row["layer"] == "model_dependency":
                info = model_lookup.get(row["evidence_value"], {})
                row["provider"] = info.get("provider") or model_provider(row["evidence_value"])[0]
                row["model_family"] = info.get("model_family") or model_family(row["evidence_value"])
                row["upstream_license"] = info.get("license", "missing")
                row["provider_basis"] = info.get("provider_basis", "unmapped_public_namespace")
            else:
                row["model_family"] = ""
                row["upstream_license"] = "not_applicable"
                row["provider_basis"] = row["evidence_type"]
        edges = pd.DataFrame(signal_rows)
        if not edges.empty:
            edges = edges[edges["space_id"].isin(set(selected["space_id"]))]
            edges = edges.drop_duplicates().sort_values(
                ["space_id", "layer", "provider", "evidence_value"],
                key=lambda series: series.astype(str).str.casefold(),
            )

        def upstream_licenses_for_space(space_id: str) -> list[str]:
            if edges.empty:
                return []
            values = edges.loc[
                edges["space_id"].eq(space_id) & edges["layer"].eq("model_dependency"),
                "upstream_license",
            ]
            return [str(value) for value in values]

        selected["rights_review_status"] = selected.apply(
            lambda row: rights_review_flag(
                row["app_license"], upstream_licenses_for_space(row["space_id"])
            ),
            axis=1,
        )
        self.processed.mkdir(parents=True, exist_ok=True)
        selected.sort_values("space_id", key=lambda s: s.str.casefold()).to_csv(
            self.processed / "space_frame.csv", index=False
        )
        edges.to_csv(self.processed / "dependency_edges.csv", index=False)
        models.sort_values("model_id", key=lambda s: s.str.casefold()).to_csv(
            self.processed / "model_frame.csv", index=False
        )
        pd.DataFrame(self.file_manifest).drop_duplicates(
            subset=["space_id", "revision", "file_path"]
        ).sort_values(["space_id", "file_path"]).to_csv(
            self.qa / "source_file_manifest.csv", index=False
        )

        manifest = {
            "protocol_version": PROTOCOL_VERSION,
            "snapshot_date": SNAPSHOT_DATE,
            "started_at": started,
            "completed_at": utc_now(),
            "search_terms": list(SEARCH_TERMS),
            "search_arms": list(SEARCH_ARMS),
            "per_query_limit": PER_QUERY_LIMIT,
            "discovery_candidates": len(candidates),
            "strict_education_phrase_spaces": int(candidates["education_strict_match"].sum()),
            "broad_education_ai_spaces": int(candidates["included_broad"].sum()),
            "strict_education_ai_spaces": int(candidates["included_strict"].sum()),
            "strict_dependency_observable_spaces": int(candidates["included_strict"].sum()),
            "strict_identifiable_dependency_spaces": int(
                candidates.loc[
                    candidates["education_strict_match"].eq(True),
                    "identifiable_dependency_observable",
                ].fillna(False).sum()
            ),
            "strict_spaces_with_unmapped_service_candidates": int(
                candidates.loc[
                    candidates["education_strict_match"].eq(True),
                    "unknown_service_candidate_count",
                ].fillna(0).gt(0).sum()
            ),
            "unique_linked_models": len(models),
            "dependency_signal_rows": len(edges),
            "http_requests_this_run": self.client.requests,
            "http_retries_this_run": self.client.retries,
            "raw_source_cache_released": False,
            "geography_rule": "explicit public author-profile country names only; hosting region excluded",
            "interpretive_boundary": "bounded query-defined public Hugging Face Spaces sample, not a platform census",
        }
        write_json(self.qa / "collection_manifest.json", manifest)
        return manifest


def normalize_base_models(value: object, tags: Iterable[object]) -> list[str]:
    output: set[str] = set()
    if isinstance(value, str):
        output.add(value)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            if isinstance(item, str):
                output.add(item)
            elif isinstance(item, dict):
                candidate = item.get("model") or item.get("name")
                if candidate:
                    output.add(str(candidate))
    elif isinstance(value, dict):
        candidate = value.get("model") or value.get("name")
        if candidate:
            output.add(str(candidate))
    for tag in tags:
        text = str(tag)
        if text.startswith("base_model:"):
            candidate = text.split(":")[-1]
            if "/" in candidate:
                output.add(candidate)
    return sorted((item for item in output if "/" in item), key=str.casefold)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collector = Collector(args.root, refresh=args.refresh, workers=max(1, args.workers))
    try:
        manifest = collector.run()
    finally:
        collector.close()
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
