"""Build a deterministic, blinded validation pack for independent review."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re

import pandas as pd


VALIDATION_SEED = 20260903


def stable_rank(value: str, *, salt: str) -> str:
    payload = f"{VALIDATION_SEED}|{salt}|{value}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_sample(
    frame: pd.DataFrame,
    count: int,
    *,
    salt: str,
    identity_columns: tuple[str, ...] = ("space_id",),
) -> pd.DataFrame:
    ranked = frame.copy()
    ranked["_sample_rank"] = ranked.apply(
        lambda row: stable_rank(
            "|".join(str(row[column]) for column in identity_columns), salt=salt
        ),
        axis=1,
    )
    return ranked.sort_values("_sample_rank").head(min(count, len(ranked))).drop(
        columns="_sample_rank"
    )


def provider_balanced_sample(
    frame: pd.DataFrame,
    count: int,
    *,
    salt: str,
    minimum_per_provider: int,
) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    selected_indices: set[int] = set()
    for provider, group in frame.groupby("provider", sort=True):
        portion = stable_sample(
            group,
            minimum_per_provider,
            salt=f"{salt}|provider|{provider}",
            identity_columns=("space_id", "provider", "source_file", "evidence_value"),
        )
        selected.append(portion)
        selected_indices.update(portion.index)
    initial = pd.concat(selected, ignore_index=False) if selected else frame.head(0)
    if len(initial) >= count:
        return stable_sample(
            initial,
            count,
            salt=f"{salt}|trim",
            identity_columns=("space_id", "provider", "source_file", "evidence_value"),
        )
    remainder = frame.loc[~frame.index.isin(selected_indices)]
    fill = stable_sample(
        remainder,
        count - len(initial),
        salt=f"{salt}|fill",
        identity_columns=("space_id", "provider", "source_file", "evidence_value"),
    )
    return pd.concat([initial, fill], ignore_index=False)


def cache_path(cache: Path, space_id: str, revision: str, file_path: str) -> Path:
    key = f"{space_id}\n{revision}\n{file_path}".encode("utf-8")
    return cache / f"{hashlib.sha256(key).hexdigest()}.json.gz"


def cached_payload(
    cache: Path,
    space_id: str,
    revision: str,
    file_path: str,
) -> dict[str, object] | None:
    path = cache_path(cache, space_id, revision, file_path)
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


def compact_text(value: object, limit: int = 1_200) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else f"{text[: limit - 1]}…"


def evidence_excerpt(text: str, provider: str, evidence_value: str) -> str:
    lines = text.splitlines()
    tokens = {
        token.casefold()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9.-]{2,}", f"{provider} {evidence_value}")
        if token.casefold() not in {"official", "signature", "provider", "service"}
    }
    matched = [
        index
        for index, line in enumerate(lines)
        if any(token in line.casefold() for token in tokens)
    ]
    if not matched:
        return compact_text(text)
    context: list[str] = []
    seen: set[int] = set()
    for index in matched[:6]:
        for line_index in range(max(0, index - 1), min(len(lines), index + 2)):
            if line_index not in seen:
                context.append(f"L{line_index + 1}: {lines[line_index]}")
                seen.add(line_index)
    return compact_text(" | ".join(context))


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def build(root: Path) -> dict[str, int]:
    processed = root / "data/processed"
    qa = root / "data/qa"
    cache = root / "data/raw/file_cache"
    output = root / "data/validation_private"
    output.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(qa / "candidate_selection_audit.csv")
    spaces = pd.read_csv(processed / "space_frame.csv")
    edges = pd.read_csv(processed / "dependency_edges.csv")
    manifest = pd.read_csv(qa / "source_file_manifest.csv")

    strict = candidates["education_strict_match"].fillna(False).astype(bool)
    broad = candidates["education_broad_match"].fillna(False).astype(bool)
    included = candidates["included_strict"].fillna(False).astype(bool)
    candidates = candidates.copy()
    candidates["validation_stratum"] = ""
    candidates.loc[included, "validation_stratum"] = "strict_dependency_observable"
    candidates.loc[strict & ~included, "validation_stratum"] = "strict_not_dependency_observable"
    candidates.loc[broad & ~strict, "validation_stratum"] = "broad_only"
    candidates.loc[~broad, "validation_stratum"] = "no_detected_construct"
    education_targets = {
        "strict_dependency_observable": 60,
        "strict_not_dependency_observable": 60,
        "broad_only": 17,
        "no_detected_construct": 41,
    }
    education_parts: list[pd.DataFrame] = []
    for stratum, target in education_targets.items():
        population = candidates[candidates["validation_stratum"].eq(stratum)]
        sampled = stable_sample(population, target, salt=f"education|{stratum}")
        sampled = sampled.copy()
        sampled["stratum_population"] = len(population)
        sampled["inclusion_probability"] = len(sampled) / len(population)
        education_parts.append(sampled)
    education = pd.concat(education_parts, ignore_index=True)
    education["_blind_rank"] = education["space_id"].map(
        lambda value: stable_rank(str(value), salt="education|blind-order")
    )
    education = education.sort_values("_blind_rank").reset_index(drop=True)
    education["sample_id"] = [f"E{index:03d}" for index in range(1, len(education) + 1)]

    def readme_excerpt(row: pd.Series) -> str:
        payload = cached_payload(
            cache,
            str(row["space_id"]),
            str(row["revision"]),
            "README.md",
        )
        return compact_text(payload.get("text", "") if payload else "")

    education_blind = pd.DataFrame(
        {
            "sample_id": education["sample_id"],
            "title": education["title"],
            "project_url": education["space_id"].map(
                lambda value: f"https://huggingface.co/spaces/{value}"
            ),
            "readme_excerpt": education.apply(readme_excerpt, axis=1),
            "reviewer1_education_relevant": "",
            "reviewer1_notes": "",
            "reviewer2_education_relevant": "",
            "reviewer2_notes": "",
            "adjudicated_education_relevant": "",
            "adjudication_notes": "",
        }
    )
    education_key = education[
        [
            "sample_id",
            "space_id",
            "validation_stratum",
            "stratum_population",
            "inclusion_probability",
            "education_strict_match",
            "education_broad_match",
            "included_strict",
            "strict_constructs",
            "broad_constructs",
        ]
    ].copy()

    edge_frame = edges[
        edges["layer"].isin(["inference_service", "model_dependency"])
    ].drop_duplicates(
        ["space_id", "provider", "layer", "evidence_type", "evidence_value", "source_file"]
    )
    edge_frame = edge_frame.copy()
    edge_frame["validation_stratum"] = (
        edge_frame["layer"].astype(str)
        + "|"
        + edge_frame["evidence_type"].astype(str)
        + "|"
        + edge_frame["confidence"].astype(str)
    )
    edge_targets = {
        "inference_service|configuration_signature|medium": 17,
        "inference_service|code_signature|high": 40,
        "inference_service|package|high": 35,
        "model_dependency|code_model_id|medium": 24,
        "model_dependency|hf_linked_model|high": 44,
    }
    edge_parts: list[pd.DataFrame] = []
    for stratum, target in edge_targets.items():
        population = edge_frame[edge_frame["validation_stratum"].eq(stratum)]
        if stratum.startswith("inference_service"):
            sampled = provider_balanced_sample(
                population,
                target,
                salt=f"dependency|{stratum}",
                minimum_per_provider=2 if "code_signature" in stratum else 1,
            )
        else:
            sampled = stable_sample(
                population,
                target,
                salt=f"dependency|{stratum}",
                identity_columns=("space_id", "provider", "source_file", "evidence_value"),
            )
        sampled = sampled.copy()
        sampled["stratum_population"] = len(population)
        sampled["inclusion_probability"] = len(sampled) / len(population)
        edge_parts.append(sampled)
    dependency = pd.concat(edge_parts, ignore_index=True)
    dependency["_blind_rank"] = dependency.apply(
        lambda row: stable_rank(
            f"{row['space_id']}|{row['provider']}|{row['source_file']}|{row['evidence_value']}",
            salt="dependency|blind-order",
        ),
        axis=1,
    )
    dependency = dependency.sort_values("_blind_rank").reset_index(drop=True)
    dependency["sample_id"] = [f"D{index:03d}" for index in range(1, len(dependency) + 1)]
    revisions = spaces.drop_duplicates("space_id").set_index("space_id")["revision"]

    def dependency_source(row: pd.Series) -> tuple[str, str]:
        space_id = str(row["space_id"])
        if row["source_file"] == "hub_metadata":
            return (
                f"https://huggingface.co/spaces/{space_id}",
                f"Public Space metadata declares linked model identifier {row['evidence_value']}.",
            )
        revision = str(revisions.get(space_id, ""))
        if row["source_file"] == "selected_source_files":
            selected = manifest[
                manifest["space_id"].astype(str).eq(space_id)
                & manifest["revision"].astype(str).eq(revision)
                & manifest["status_code"].eq(200)
            ]
            matches: list[tuple[str, str]] = []
            target = str(row["evidence_value"]).casefold()
            for file_path in selected["file_path"].dropna().astype(str):
                payload = cached_payload(cache, space_id, revision, file_path)
                if not payload:
                    continue
                text = str(payload.get("text") or "")
                if target not in text.casefold():
                    continue
                matches.append(
                    (
                        str(payload.get("public_url") or ""),
                        f"[{file_path}] {evidence_excerpt(text, str(row['provider']), str(row['evidence_value']))}",
                    )
                )
            if matches:
                return (
                    " | ".join(url for url, _ in matches if url),
                    " || ".join(excerpt for _, excerpt in matches),
                )
        payload = cached_payload(cache, space_id, revision, str(row["source_file"]))
        if not payload:
            return (f"https://huggingface.co/spaces/{space_id}", "")
        text = str(payload.get("text") or "")
        return (
            str(payload.get("public_url") or f"https://huggingface.co/spaces/{space_id}"),
            evidence_excerpt(text, str(row["provider"]), str(row["evidence_value"])),
        )

    dependency_sources = dependency.apply(dependency_source, axis=1)
    dependency_blind = pd.DataFrame(
        {
            "sample_id": dependency["sample_id"],
            "project_url": dependency["space_id"].map(
                lambda value: f"https://huggingface.co/spaces/{value}"
            ),
            "source_file": dependency["source_file"],
            "source_url": [value[0] for value in dependency_sources],
            "source_excerpt": [value[1] for value in dependency_sources],
            "reviewer1_dependency_supported": "",
            "reviewer1_layer": "",
            "reviewer1_provider_or_model": "",
            "reviewer1_notes": "",
            "reviewer2_dependency_supported": "",
            "reviewer2_layer": "",
            "reviewer2_provider_or_model": "",
            "reviewer2_notes": "",
            "adjudicated_dependency_supported": "",
            "adjudicated_layer": "",
            "adjudicated_provider_or_model": "",
            "adjudication_notes": "",
        }
    )
    dependency_key = dependency[
        [
            "sample_id",
            "space_id",
            "provider",
            "layer",
            "evidence_type",
            "evidence_value",
            "source_file",
            "confidence",
            "validation_stratum",
            "stratum_population",
            "inclusion_probability",
        ]
    ].copy()

    strict_spaces = spaces[spaces["included_strict"].eq(True)].copy()
    service_ids = set(
        edges.loc[edges["layer"].eq("inference_service"), "space_id"].astype(str)
    )
    service_negatives = strict_spaces[
        ~strict_spaces["space_id"].astype(str).isin(service_ids)
    ].copy()
    service_negatives["validation_stratum"] = service_negatives["sdk"].fillna("missing")
    negative_parts: list[pd.DataFrame] = []
    for sdk, group in service_negatives.groupby("validation_stratum", sort=True):
        target = max(1, round(80 * len(group) / len(service_negatives)))
        negative_parts.append(
            stable_sample(group, target, salt=f"service-negative|{sdk}")
        )
    negative = pd.concat(negative_parts, ignore_index=True).drop_duplicates("space_id")
    if len(negative) > 80:
        negative = stable_sample(negative, 80, salt="service-negative|trim")
    elif len(negative) < min(80, len(service_negatives)):
        fill = stable_sample(
            service_negatives[~service_negatives["space_id"].isin(negative["space_id"])],
            80 - len(negative),
            salt="service-negative|fill",
        )
        negative = pd.concat([negative, fill], ignore_index=True)
    negative["_blind_rank"] = negative["space_id"].map(
        lambda value: stable_rank(str(value), salt="service-negative|blind-order")
    )
    negative = negative.sort_values("_blind_rank").reset_index(drop=True)
    negative["sample_id"] = [f"N{index:03d}" for index in range(1, len(negative) + 1)]

    def source_urls(row: pd.Series) -> str:
        rows = manifest[
            manifest["space_id"].astype(str).eq(str(row["space_id"]))
            & manifest["revision"].astype(str).eq(str(row["revision"]))
            & manifest["status_code"].eq(200)
        ]
        return " | ".join(rows["public_url"].dropna().astype(str).tolist())

    negative_blind = pd.DataFrame(
        {
            "sample_id": negative["sample_id"],
            "title": negative["title"],
            "project_url": negative["space_id"].map(
                lambda value: f"https://huggingface.co/spaces/{value}"
            ),
            "selected_source_urls": negative.apply(source_urls, axis=1),
            "reviewer1_named_service_present": "",
            "reviewer1_provider": "",
            "reviewer1_notes": "",
            "reviewer2_named_service_present": "",
            "reviewer2_provider": "",
            "reviewer2_notes": "",
            "adjudicated_named_service_present": "",
            "adjudicated_provider": "",
            "adjudication_notes": "",
        }
    )
    negative_key = negative[
        ["sample_id", "space_id", "sdk", "validation_stratum"]
    ].copy()
    negative_key["stratum_population"] = negative_key["validation_stratum"].map(
        service_negatives["validation_stratum"].value_counts()
    )
    negative_key["stratum_sample"] = negative_key["validation_stratum"].map(
        negative_key["validation_stratum"].value_counts()
    )
    negative_key["inclusion_probability"] = (
        negative_key["stratum_sample"] / negative_key["stratum_population"]
    )

    write_csv(education_blind, output / "education_relevance_blinded.csv")
    write_csv(education_key, output / "education_relevance_key.csv")
    write_csv(dependency_blind, output / "dependency_evidence_blinded.csv")
    write_csv(dependency_key, output / "dependency_evidence_key.csv")
    write_csv(negative_blind, output / "service_negative_blinded.csv")
    write_csv(negative_key, output / "service_negative_key.csv")

    readme = f"""# Independent validation pack

This pack was generated deterministically with seed `{VALIDATION_SEED}`. It evaluates the frozen rules and must not be used to add or remove projects or edges from the main analysis.

## Blinding

Give reviewers only the three files ending in `_blinded.csv`. Keep the corresponding `_key.csv` files separate until both reviewers have completed their labels. Project order is independently randomized in each task.

## Label rules

1. `education_relevance_blinded.csv`: label `yes`, `no`, or `uncertain` according to whether the project has an educational function, not merely an educational word in incidental prose.
2. `dependency_evidence_blinded.csv`: decide whether the shown public source supports a declared external inference-service or model dependency. Record the layer and provider/model without consulting the key.
3. `service_negative_blinded.csv`: inspect the pinned selected-source URLs and record whether a named external inference service was missed by the parser. Generic local runtimes and model identifiers are not services.

Reviewers should work independently. Disagreements may be adjudicated only after initial agreement and Cohen's kappa are calculated. Source excerpts are review aids; the pinned public source URL is authoritative. The pack is private because excerpts reproduce third-party source text. Aggregate validation results and de-identified labels can be released later.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    return {
        "education_records": len(education_blind),
        "dependency_records": len(dependency_blind),
        "service_negative_records": len(negative_blind),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = build(args.root.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
