"""Analyze concentration, disclosure, and activity in the frozen Space frame."""

from __future__ import annotations

import argparse
from datetime import date
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Iterable

import numpy as np
import pandas as pd

from collect_historical_versions import recompute_version_history_outputs
from protocol import (
    ASIA_MODEL_PROVIDERS,
    CODE_SUFFIXES,
    SEARCH_ARMS,
    SEARCH_TERMS,
    SNAPSHOT_DATE,
)


BOOTSTRAP_SEED = 20260831
BOOTSTRAP_DRAWS = 2_000
CODECLARATION_PERMUTATIONS = 10_000
SOURCE_EXCLUSIONS = {"readme.md", "license", "license.txt", ".gitattributes", ".gitignore"}
SEARCH_CUTOFFS = (25, 50, 75, 100)
SIMILARITY_THRESHOLDS = (0.85, 0.90, 0.95)
PRIMARY_SIMILARITY_THRESHOLD = 0.90
SOURCE_SHINGLE_SIZE = 7
MIN_SOURCE_SHINGLES = 50

_STRING_LITERAL_RE = re.compile(
    r'''(?s)(?:"{3}.*?"{3}|'{3}.*?'{3}|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')'''
)
_BLOCK_COMMENT_RE = re.compile(r"(?s)/\*.*?\*/|<!--.*?-->")
_LINE_COMMENT_RE = re.compile(r"(?m)(?<!:)//[^\n]*|#[^\n]*")
_SOURCE_TOKEN_RE = re.compile(
    r"[A-Za-z_$][A-Za-z0-9_$]*|\d+(?:\.\d+)?|==|!=|<=|>=|=>|->|::|[^\s]"
)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def source_cluster_assignments(manifest: pd.DataFrame) -> pd.DataFrame:
    """Group exact multi-file source bundles; single-file bundles stay unique."""

    frame = manifest.copy()
    frame = frame[frame["status_code"].eq(200)]
    frame = frame[
        ~frame["file_path"].astype(str).map(
            lambda value: PurePosixPath(value).name.casefold() in SOURCE_EXCLUSIONS
        )
    ]
    rows: list[dict[str, object]] = []
    for space_id, group in frame.groupby("space_id", sort=True):
        digests = sorted(set(group["sha256"].dropna().astype(str)))
        if len(digests) >= 2:
            signature = hashlib.sha256("\n".join(digests).encode()).hexdigest()
            basis = "exact_multifile_bundle"
        else:
            signature = f"unique:{space_id}"
            basis = "insufficient_files_for_clustering"
        rows.append(
            {
                "space_id": space_id,
                "source_cluster_id": signature,
                "source_cluster_basis": basis,
                "source_file_digests": len(digests),
            }
        )
    assignments = pd.DataFrame(rows)
    sizes = assignments.groupby("source_cluster_id").size().rename("source_cluster_size")
    assignments = assignments.join(sizes, on="source_cluster_id")
    assignments["source_cluster_weight"] = 1.0 / assignments["source_cluster_size"]
    return assignments.sort_values("space_id")


def fractional_rankings(
    edges: pd.DataFrame,
    spaces: pd.DataFrame,
    *,
    layers: Iterable[str],
    label_column: str = "provider",
    base_weight_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Allocate each project's weight equally across its detected upstreams."""

    project_ids = set(spaces["space_id"].astype(str))
    subset = edges[
        edges["space_id"].astype(str).isin(project_ids)
        & edges["layer"].isin(set(layers))
    ].copy()
    subset = subset[subset[label_column].fillna("").astype(str).ne("")]
    subset = subset[["space_id", label_column]].drop_duplicates()
    if subset.empty:
        return pd.DataFrame(), {
            "projects": 0,
            "providers": 0,
            "hhi": math.nan,
            "effective_number": math.nan,
            "top1_share": math.nan,
            "top3_share": math.nan,
            "gini": math.nan,
        }
    weights = spaces.set_index("space_id")
    if base_weight_column:
        subset["base_weight"] = subset["space_id"].map(weights[base_weight_column]).astype(float)
    else:
        subset["base_weight"] = 1.0
    counts = subset.groupby("space_id")[label_column].transform("nunique")
    subset["fractional_weight"] = subset["base_weight"] / counts
    ranking = (
        subset.groupby(label_column, as_index=False)
        .agg(
            project_count=("space_id", "nunique"),
            fractional_weight=("fractional_weight", "sum"),
        )
        .sort_values(["fractional_weight", "project_count", label_column], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    denominator = subset[["space_id", "base_weight"]].drop_duplicates()["base_weight"].sum()
    ranking["fractional_share"] = ranking["fractional_weight"] / denominator
    shares = ranking["fractional_share"].to_numpy(float)
    hhi = float(np.square(shares).sum())
    summary = {
        "projects": int(subset["space_id"].nunique()),
        "providers": int(ranking[label_column].nunique()),
        "hhi": hhi,
        "effective_number": float(1.0 / hhi) if hhi else math.nan,
        "top1_share": float(shares[0]),
        "top3_share": float(shares[:3].sum()),
        "gini": gini(ranking["fractional_weight"].to_numpy(float)),
    }
    return ranking, summary


def gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or values.sum() <= 0:
        return math.nan
    differences = np.abs(values[:, None] - values[None, :]).sum()
    return float(differences / (2.0 * values.size * values.sum()))


def shannon_entropy(shares: np.ndarray) -> float:
    positive = shares[shares > 0]
    return float(-(positive * np.log(positive)).sum())


def bootstrap_hhi(
    edges: pd.DataFrame,
    spaces: pd.DataFrame,
    *,
    layers: Iterable[str],
    label_column: str = "provider",
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    subset = edges[edges["layer"].isin(set(layers))]
    subset = subset[subset[label_column].fillna("").astype(str).ne("")]
    subset = subset[["space_id", label_column]].drop_duplicates()
    eligible = sorted(set(spaces["space_id"]) & set(subset["space_id"]))
    labels = sorted(subset[label_column].unique(), key=str.casefold)
    if not eligible or not labels:
        return math.nan, math.nan
    row_index = {space_id: index for index, space_id in enumerate(eligible)}
    col_index = {label: index for index, label in enumerate(labels)}
    matrix = np.zeros((len(eligible), len(labels)), dtype=float)
    provider_counts = subset.groupby("space_id")[label_column].nunique()
    for row in subset.itertuples(index=False):
        if row.space_id in row_index:
            matrix[row_index[row.space_id], col_index[getattr(row, label_column)]] = (
                1.0 / provider_counts[row.space_id]
            )
    rng = np.random.default_rng(seed)
    values = np.empty(draws, dtype=float)
    for draw in range(draws):
        indices = rng.integers(0, len(eligible), size=len(eligible))
        shares = matrix[indices].sum(axis=0) / len(eligible)
        values[draw] = np.square(shares).sum()
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def fractional_matrix(
    edges: pd.DataFrame,
    project_ids: Iterable[str],
    *,
    layers: Iterable[str],
    label_column: str = "provider",
) -> tuple[np.ndarray, list[str], list[str]]:
    """Return one unit of fractional dependency weight per project row."""

    projects = sorted(set(project_ids), key=str.casefold)
    subset = edges[
        edges["space_id"].isin(projects) & edges["layer"].isin(set(layers))
    ].copy()
    subset = subset[subset[label_column].fillna("").astype(str).ne("")]
    subset = subset[["space_id", label_column]].drop_duplicates()
    labels = sorted(subset[label_column].astype(str).unique(), key=str.casefold)
    matrix = np.zeros((len(projects), len(labels)), dtype=float)
    if not projects or not labels:
        return matrix, projects, labels
    row_index = {space_id: index for index, space_id in enumerate(projects)}
    col_index = {label: index for index, label in enumerate(labels)}
    counts = subset.groupby("space_id")[label_column].nunique()
    for row in subset.itertuples(index=False):
        matrix[row_index[row.space_id], col_index[str(getattr(row, label_column))]] = (
            1.0 / counts[row.space_id]
        )
    return matrix, projects, labels


def paired_layer_bootstrap(
    edges: pd.DataFrame,
    project_ids: Iterable[str],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Compare service and model concentration on identical project resamples."""

    project_set = set(project_ids)
    service_ids = set(
        edges.loc[edges["layer"].eq("inference_service"), "space_id"].astype(str)
    )
    model_ids = set(
        edges.loc[edges["layer"].eq("model_dependency"), "space_id"].astype(str)
    )
    matched_ids = sorted(project_set & service_ids & model_ids, key=str.casefold)
    service_matrix, _, service_labels = fractional_matrix(
        edges,
        matched_ids,
        layers=("inference_service",),
    )
    model_matrix, _, model_labels = fractional_matrix(
        edges,
        matched_ids,
        layers=("model_dependency",),
    )
    if not matched_ids:
        raise ValueError("no projects have both service and model dependency signals")
    if not np.allclose(service_matrix.sum(axis=1), 1.0):
        raise ValueError("service matrix does not allocate one unit per matched project")
    if not np.allclose(model_matrix.sum(axis=1), 1.0):
        raise ValueError("model matrix does not allocate one unit per matched project")

    service_shares = service_matrix.mean(axis=0)
    model_shares = model_matrix.mean(axis=0)
    service_hhi = float(np.square(service_shares).sum())
    model_hhi = float(np.square(model_shares).sum())
    service_top_share = float(service_shares.max())
    model_top_share = float(model_shares.max())
    service_entropy = shannon_entropy(service_shares)
    model_entropy = shannon_entropy(model_shares)
    rng = np.random.default_rng(seed)
    bootstrap_rows: list[dict[str, float | int]] = []
    for draw in range(draws):
        indices = rng.integers(0, len(matched_ids), size=len(matched_ids))
        service_draw_shares = service_matrix[indices].mean(axis=0)
        model_draw_shares = model_matrix[indices].mean(axis=0)
        service_value = float(np.square(service_draw_shares).sum())
        model_value = float(np.square(model_draw_shares).sum())
        service_draw_entropy = shannon_entropy(service_draw_shares)
        model_draw_entropy = shannon_entropy(model_draw_shares)
        bootstrap_rows.append(
            {
                "draw": draw + 1,
                "service_hhi": service_value,
                "model_hhi": model_value,
                "hhi_difference": service_value - model_value,
                "hhi_ratio": service_value / model_value,
                "service_top_share": float(service_draw_shares.max()),
                "model_top_share": float(model_draw_shares.max()),
                "top_share_difference": float(
                    service_draw_shares.max() - model_draw_shares.max()
                ),
                "service_shannon_entropy": service_draw_entropy,
                "model_shannon_entropy": model_draw_entropy,
                "shannon_entropy_difference_model_minus_service": (
                    model_draw_entropy - service_draw_entropy
                ),
            }
        )
    bootstrap = pd.DataFrame(bootstrap_rows)

    def interval(column: str) -> list[float]:
        return [
            float(bootstrap[column].quantile(0.025)),
            float(bootstrap[column].quantile(0.975)),
        ]

    summary: dict[str, object] = {
        "independent_unit": "project",
        "matched_projects": len(matched_ids),
        "service_providers": len(service_labels),
        "model_publishers_or_namespaces": len(model_labels),
        "service_hhi": service_hhi,
        "model_hhi": model_hhi,
        "hhi_difference_service_minus_model": service_hhi - model_hhi,
        "hhi_ratio_service_over_model": service_hhi / model_hhi,
        "service_top_share": service_top_share,
        "model_top_share": model_top_share,
        "top_share_difference_service_minus_model": service_top_share - model_top_share,
        "service_shannon_entropy": service_entropy,
        "model_shannon_entropy": model_entropy,
        "shannon_entropy_difference_model_minus_service": model_entropy - service_entropy,
        "service_shannon_effective_categories": float(np.exp(service_entropy)),
        "model_shannon_effective_categories": float(np.exp(model_entropy)),
        "service_hhi_bootstrap_ci": interval("service_hhi"),
        "model_hhi_bootstrap_ci": interval("model_hhi"),
        "hhi_difference_bootstrap_ci": interval("hhi_difference"),
        "hhi_ratio_bootstrap_ci": interval("hhi_ratio"),
        "top_share_difference_bootstrap_ci": interval("top_share_difference"),
        "shannon_entropy_difference_bootstrap_ci": interval(
            "shannon_entropy_difference_model_minus_service"
        ),
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
        "bootstrap_fraction_service_hhi_gt_model_hhi": float(
            bootstrap["hhi_difference"].gt(0).mean()
        ),
        "interpretation": (
            "paired descriptive bootstrap stability; the dominance fraction is not a p value"
        ),
    }
    return summary, bootstrap


def paired_layer_estimate(
    edges: pd.DataFrame,
    spaces: pd.DataFrame,
    *,
    base_weight_column: str | None = None,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Return one consistently paired estimand for robustness comparisons."""

    project_ids = set(spaces["space_id"].astype(str))
    valid = edges[edges["provider"].fillna("").astype(str).ne("")]
    service_ids = set(
        valid.loc[valid["layer"].eq("inference_service"), "space_id"].astype(str)
    )
    model_ids = set(
        valid.loc[valid["layer"].eq("model_dependency"), "space_id"].astype(str)
    )
    matched_ids = sorted(project_ids & service_ids & model_ids, key=str.casefold)
    if not matched_ids:
        raise ValueError("no projects have both service and model dependency signals")
    service_matrix, _, service_labels = fractional_matrix(
        valid, matched_ids, layers=("inference_service",)
    )
    model_matrix, _, model_labels = fractional_matrix(
        valid, matched_ids, layers=("model_dependency",)
    )
    if not np.allclose(service_matrix.sum(axis=1), 1.0):
        raise ValueError("service matrix does not allocate one unit per matched project")
    if not np.allclose(model_matrix.sum(axis=1), 1.0):
        raise ValueError("model matrix does not allocate one unit per matched project")

    indexed = spaces.drop_duplicates("space_id").set_index("space_id")
    if base_weight_column:
        weights = (
            indexed.reindex(matched_ids)[base_weight_column]
            .fillna(1.0)
            .astype(float)
            .to_numpy()
        )
    else:
        weights = np.ones(len(matched_ids), dtype=float)
    if np.any(weights < 0) or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError(f"invalid project weights: {base_weight_column}")

    def weighted_hhi(matrix: np.ndarray, row_weights: np.ndarray) -> float:
        shares = (matrix * row_weights[:, None]).sum(axis=0) / row_weights.sum()
        return float(np.square(shares).sum())

    service_hhi = weighted_hhi(service_matrix, weights)
    model_hhi = weighted_hhi(model_matrix, weights)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty((draws, 3), dtype=float)
    for draw in range(draws):
        indices = rng.integers(0, len(matched_ids), size=len(matched_ids))
        sampled_weights = weights[indices]
        service_value = weighted_hhi(service_matrix[indices], sampled_weights)
        model_value = weighted_hhi(model_matrix[indices], sampled_weights)
        bootstrap[draw] = (service_value, model_value, service_value - model_value)

    return {
        "matched_projects": len(matched_ids),
        "service_categories": len(service_labels),
        "model_categories": len(model_labels),
        "weighting": base_weight_column or "project_equal",
        "service_hhi": service_hhi,
        "model_hhi": model_hhi,
        "hhi_difference_service_minus_model": service_hhi - model_hhi,
        "service_hhi_ci_low": float(np.quantile(bootstrap[:, 0], 0.025)),
        "service_hhi_ci_high": float(np.quantile(bootstrap[:, 0], 0.975)),
        "model_hhi_ci_low": float(np.quantile(bootstrap[:, 1], 0.025)),
        "model_hhi_ci_high": float(np.quantile(bootstrap[:, 1], 0.975)),
        "hhi_difference_ci_low": float(np.quantile(bootstrap[:, 2], 0.025)),
        "hhi_difference_ci_high": float(np.quantile(bootstrap[:, 2], 0.975)),
        "bootstrap_fraction_service_hhi_gt_model_hhi": float(
            np.mean(bootstrap[:, 2] > 0)
        ),
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
    }


def cross_layer_codeclaration(
    edges: pd.DataFrame,
    project_ids: Iterable[str],
    *,
    permutations: int = CODECLARATION_PERMUTATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Measure project-level service--model co-declaration without inferring routing."""

    project_set = set(str(value) for value in project_ids)
    service_ids = set(
        edges.loc[edges["layer"].eq("inference_service"), "space_id"].astype(str)
    )
    model_ids = set(
        edges.loc[edges["layer"].eq("model_dependency"), "space_id"].astype(str)
    )
    matched_ids = sorted(project_set & service_ids & model_ids, key=str.casefold)
    service_matrix, _, service_labels = fractional_matrix(
        edges, matched_ids, layers=("inference_service",)
    )
    model_matrix, _, model_labels = fractional_matrix(
        edges, matched_ids, layers=("model_dependency",)
    )
    joint = service_matrix.T @ model_matrix / len(matched_ids)
    service_marginal = joint.sum(axis=1)
    model_marginal = joint.sum(axis=0)

    def mutual_information(matrix: np.ndarray) -> float:
        return (
            shannon_entropy(matrix.sum(axis=1))
            + shannon_entropy(matrix.sum(axis=0))
            - shannon_entropy(matrix.ravel())
        )

    observed_mi = mutual_information(joint)
    rng = np.random.default_rng(seed)
    null_values = np.empty(permutations, dtype=float)
    for draw in range(permutations):
        permuted_joint = (
            service_matrix.T @ model_matrix[rng.permutation(len(matched_ids))]
            / len(matched_ids)
        )
        null_values[draw] = mutual_information(permuted_joint)

    joint_rows: list[dict[str, object]] = []
    for service_index, service_provider in enumerate(service_labels):
        for model_index, model_provider in enumerate(model_labels):
            weight = float(joint[service_index, model_index])
            if weight <= 0:
                continue
            joint_rows.append(
                {
                    "service_provider": service_provider,
                    "model_publisher_or_namespace": model_provider,
                    "joint_fractional_share": weight,
                    "model_share_within_service": (
                        weight / service_marginal[service_index]
                    ),
                    "service_share_within_model": weight / model_marginal[model_index],
                }
            )
    joint_frame = pd.DataFrame(joint_rows)

    service_rows: list[dict[str, object]] = []
    for index, provider in enumerate(service_labels):
        conditional = joint[index] / service_marginal[index]
        top_index = int(np.argmax(conditional))
        service_rows.append(
            {
                "service_provider": provider,
                "projects_declaring_service": int(
                    np.count_nonzero(service_matrix[:, index] > 0)
                ),
                "service_fractional_share": float(service_marginal[index]),
                "conditional_model_shannon_effective_categories": float(
                    np.exp(shannon_entropy(conditional))
                ),
                "top_codeclared_model_publisher_or_namespace": model_labels[top_index],
                "top_codeclared_model_share_within_service": float(
                    conditional[top_index]
                ),
            }
        )
    service_frame = pd.DataFrame(service_rows).sort_values(
        "service_fractional_share", ascending=False
    )
    null_frame = pd.DataFrame(
        {"draw": np.arange(1, permutations + 1), "mutual_information": null_values}
    )
    summary: dict[str, object] = {
        "matched_projects": len(matched_ids),
        "joint_fractional_share_sum": float(joint.sum()),
        "mutual_information": observed_mi,
        "normalized_mutual_information_sqrt": float(
            observed_mi
            / np.sqrt(
                shannon_entropy(service_marginal) * shannon_entropy(model_marginal)
            )
        ),
        "permutation_null_mean": float(null_values.mean()),
        "permutation_null_interval": [
            float(np.quantile(null_values, 0.025)),
            float(np.quantile(null_values, 0.975)),
        ],
        "randomization_p": float(
            (1 + np.count_nonzero(null_values >= observed_mi)) / (permutations + 1)
        ),
        "permutations": permutations,
        "seed": seed,
        "interpretation": (
            "within-project co-declaration association; not evidence that a "
            "service routes traffic to a co-declared model"
        ),
    }
    return summary, joint_frame, service_frame, null_frame


def matched_project_ids(edges: pd.DataFrame, project_ids: Iterable[str]) -> set[str]:
    project_set = set(str(value) for value in project_ids)
    service_ids = set(
        edges.loc[
            edges["layer"].eq("inference_service")
            & edges["provider"].fillna("").astype(str).ne(""),
            "space_id",
        ].astype(str)
    )
    model_ids = set(
        edges.loc[
            edges["layer"].eq("model_dependency")
            & edges["provider"].fillna("").astype(str).ne(""),
            "space_id",
        ].astype(str)
    )
    return project_set & service_ids & model_ids


def matched_sample_composition(
    spaces: pd.DataFrame,
    edges: pd.DataFrame,
) -> pd.DataFrame:
    """Compare matched and non-matched projects without human classifications."""

    frame = spaces.copy()
    matched_ids = matched_project_ids(edges, frame["space_id"])
    frame["matched_layers"] = frame["space_id"].isin(matched_ids)
    snapshot = pd.Timestamp(SNAPSHOT_DATE, tz="UTC")
    frame["created_age_days"] = (
        snapshot - pd.to_datetime(frame["created_at"], utc=True, errors="coerce")
    ).dt.total_seconds() / 86_400
    frame["modified_within_180_days"] = (
        snapshot - pd.to_datetime(frame["last_modified"], utc=True, errors="coerce")
    ).dt.total_seconds().le(180 * 86_400)
    frame["log1p_likes"] = np.log1p(frame["likes"].fillna(0).astype(float))
    frame["runtime_running"] = frame["runtime_stage"].eq("RUNNING")
    frame["likes_ge_1"] = frame["likes"].fillna(0).astype(float).ge(1)

    rows: list[dict[str, object]] = []
    matched = frame[frame["matched_layers"]]
    other = frame[~frame["matched_layers"]]

    def add_continuous(metric: str, column: str) -> None:
        left = matched[column].dropna().astype(float)
        right = other[column].dropna().astype(float)
        pooled = math.sqrt((left.var(ddof=1) + right.var(ddof=1)) / 2.0)
        rows.append(
            {
                "metric": metric,
                "metric_type": "continuous_mean",
                "matched_value": float(left.mean()),
                "nonmatched_value": float(right.mean()),
                "difference": float(left.mean() - right.mean()),
                "standardized_difference": (
                    float((left.mean() - right.mean()) / pooled) if pooled else math.nan
                ),
                "matched_n": len(left),
                "nonmatched_n": len(right),
            }
        )

    def add_binary(metric: str, values: pd.Series) -> None:
        left = values[frame["matched_layers"]].astype(bool)
        right = values[~frame["matched_layers"]].astype(bool)
        p_left = float(left.mean())
        p_right = float(right.mean())
        pooled = math.sqrt(
            (p_left * (1 - p_left) + p_right * (1 - p_right)) / 2.0
        )
        rows.append(
            {
                "metric": metric,
                "metric_type": "binary_share",
                "matched_value": p_left,
                "nonmatched_value": p_right,
                "difference": p_left - p_right,
                "standardized_difference": (
                    (p_left - p_right) / pooled if pooled else math.nan
                ),
                "matched_n": len(left),
                "nonmatched_n": len(right),
            }
        )

    add_continuous("created_age_days", "created_age_days")
    add_continuous("log1p_likes", "log1p_likes")
    for metric in ("modified_within_180_days", "runtime_running", "likes_ge_1"):
        add_binary(metric, frame[metric])
    for sdk in sorted(frame["sdk"].fillna("unreported").astype(str).unique(), key=str.casefold):
        add_binary(f"sdk={sdk}", frame["sdk"].fillna("unreported").astype(str).eq(sdk))
    constructs = sorted(
        {
            construct
            for value in frame["strict_constructs"].fillna("")
            for construct in str(value).split(";")
            if construct
        }
    )
    for construct in constructs:
        add_binary(
            f"construct={construct}",
            frame["strict_constructs"].fillna("").map(
                lambda value: construct in str(value).split(";")
            ),
        )
    return pd.DataFrame(rows)


def unknown_service_boundary_table(
    spaces: pd.DataFrame,
    edges: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    strict_phrase_projects: int,
) -> pd.DataFrame:
    """Bound service concentration under transparent, automatic unknown handling."""

    service_candidates = candidates[
        candidates["candidate_type"].isin(
            {
                "unmapped_credential",
                "unmapped_api_domain",
                "openai_compatible_provider_unresolved",
            }
        )
    ].drop_duplicates(["space_id", "candidate_type", "identifier"])
    project_ids = set(spaces["space_id"].astype(str))
    known_service_ids = set(
        edges.loc[
            edges["layer"].eq("inference_service"), "space_id"
        ].astype(str)
    ) & project_ids
    candidate_ids = set(service_candidates["space_id"].astype(str)) & project_ids
    common = {
        "strict_phrase_projects": strict_phrase_projects,
        "strict_dependency_observable_projects": len(spaces),
        "known_service_projects": len(known_service_ids),
        "unmapped_candidate_projects": len(candidate_ids),
        "unmapped_candidate_only_projects": len(candidate_ids - known_service_ids),
        "known_or_candidate_service_projects": len(known_service_ids | candidate_ids),
    }
    rows: list[dict[str, object]] = []
    rows.append(
        {
            "unknown_treatment": "known_services_only",
            **common,
            **paired_layer_estimate(edges, spaces),
        }
    )
    variants: dict[str, pd.Series] = {
        "pooled_unmapped_candidates": pd.Series(
            "Unmapped service candidate", index=service_candidates.index
        ),
        "machine_identifier_categories": service_candidates.apply(
            lambda row: f"unmapped:{row['candidate_type']}:{row['identifier']}",
            axis=1,
        ),
        "project_unique_unknown_bound": service_candidates["space_id"].map(
            lambda value: f"unmapped_project:{value}"
        ),
    }
    for name, providers in variants.items():
        added = pd.DataFrame(
            {
                "space_id": service_candidates["space_id"].astype(str),
                "provider": providers.astype(str),
                "layer": "inference_service",
                "evidence_type": service_candidates["candidate_type"].astype(str),
                "evidence_value": service_candidates["identifier"].astype(str),
                "confidence": "candidate",
            }
        ).drop_duplicates(["space_id", "provider", "layer"])
        expanded = pd.concat([edges, added], ignore_index=True, sort=False)
        rows.append(
            {
                "unknown_treatment": name,
                **common,
                **paired_layer_estimate(expanded, spaces),
            }
        )
    return pd.DataFrame(rows)


def model_namespace_variants(edges: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return auditable model-provider mappings without manual attribution."""

    model_edges = edges[edges["layer"].eq("model_dependency")].copy()
    pooled = model_edges.copy()
    pooled.loc[
        pooled["provider_basis"].eq("unmapped_public_namespace"), "provider"
    ] = "Unmapped namespace"
    mapped = model_edges[
        ~model_edges["provider_basis"].eq("unmapped_public_namespace")
    ].copy()
    return {
        "disaggregated_unmapped_namespaces": model_edges,
        "pooled_unmapped_namespaces": pooled,
        "mapped_publishers_only": mapped,
    }


def namespace_sensitivity(
    edges: pd.DataFrame,
    spaces: pd.DataFrame,
) -> pd.DataFrame:
    """Quantify how unresolved model-namespace treatment changes layer HHI."""

    rows: list[dict[str, object]] = []
    service_ids = set(
        edges.loc[edges["layer"].eq("inference_service"), "space_id"].astype(str)
    )
    all_ids = set(spaces["space_id"].astype(str))
    for variant, model_edges in model_namespace_variants(edges).items():
        model_ids = set(model_edges["space_id"].astype(str)) & all_ids
        matched_ids = model_ids & service_ids
        matched_spaces = spaces[spaces["space_id"].isin(matched_ids)]
        service_ranking, service_summary = fractional_rankings(
            edges[edges["space_id"].isin(matched_ids)],
            matched_spaces,
            layers=("inference_service",),
        )
        model_ranking, model_summary = fractional_rankings(
            model_edges[model_edges["space_id"].isin(matched_ids)],
            matched_spaces,
            layers=("model_dependency",),
        )
        rows.append(
            {
                "namespace_variant": variant,
                "matched_projects": len(matched_ids),
                "service_hhi": service_summary["hhi"],
                "model_hhi": model_summary["hhi"],
                "hhi_difference_service_minus_model": (
                    service_summary["hhi"] - model_summary["hhi"]
                ),
                "service_top_provider": (
                    str(service_ranking.iloc[0]["provider"])
                    if not service_ranking.empty
                    else "unresolved"
                ),
                "model_top_provider_or_namespace": (
                    str(model_ranking.iloc[0]["provider"])
                    if not model_ranking.empty
                    else "unresolved"
                ),
                "model_provider_categories": model_summary["providers"],
            }
        )
    return pd.DataFrame(rows)


def safe_slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-").casefold()
    return cleaned or hashlib.sha256(value.encode()).hexdigest()[:12]


def load_search_rankings(raw_dir: Path) -> dict[tuple[str, str], list[str]]:
    """Load the frozen order returned for every query and ranking arm."""

    rankings: dict[tuple[str, str], list[str]] = {}
    compact_path = raw_dir.parents[2] / "processed" / "search_rankings.csv"
    expected_raw = [
        raw_dir / f"{safe_slug(term)}__{safe_slug(arm)}.json"
        for term in SEARCH_TERMS
        for arm in SEARCH_ARMS
    ]
    if not all(path.exists() for path in expected_raw):
        if not compact_path.exists():
            missing = [str(path) for path in expected_raw if not path.exists()]
            raise FileNotFoundError(
                "frozen search inputs are unavailable; missing raw responses and "
                f"compact ranking table: {missing[:3]}"
            )
        compact = pd.read_csv(compact_path).sort_values(
            ["query", "arm", "rank"]
        )
        for (term, arm), group in compact.groupby(["query", "arm"], sort=False):
            ranked = group[
                group["rank"].gt(0) & group["space_id"].fillna("").astype(str).ne("")
            ]
            rankings[(str(term), str(arm))] = ranked["space_id"].astype(str).tolist()
        expected_keys = {(term, arm) for term in SEARCH_TERMS for arm in SEARCH_ARMS}
        if set(rankings) != expected_keys:
            raise ValueError("compact search ranking table does not cover the frozen design")
        return rankings
    for term in SEARCH_TERMS:
        for arm in SEARCH_ARMS:
            path = raw_dir / f"{safe_slug(term)}__{safe_slug(arm)}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise TypeError(f"unexpected frozen search payload: {path.name}")
            rankings[(term, arm)] = [
                str(item["id"])
                for item in payload
                if isinstance(item, dict) and item.get("id")
            ]
    return rankings


def search_rankings_frame(
    rankings: dict[tuple[str, str], list[str]],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"query": term, "arm": arm, "rank": rank, "space_id": space_id}
            for (term, arm), ids in rankings.items()
            for rank, space_id in (
                list(enumerate(ids, start=1)) if ids else [(0, "")]
            )
        ]
    ).sort_values(["query", "arm", "rank"])


def search_union(
    rankings: dict[tuple[str, str], list[str]],
    *,
    cutoff: int,
    excluded_term: str | None = None,
    arm_filter: str | None = None,
) -> set[str]:
    output: set[str] = set()
    for (term, arm), ordered_ids in rankings.items():
        if term == excluded_term or (arm_filter is not None and arm != arm_filter):
            continue
        output.update(ordered_ids[:cutoff])
    return output


def concentration_variant_rows(
    edges: pd.DataFrame,
    spaces: pd.DataFrame,
    *,
    descriptor: dict[str, object],
) -> list[dict[str, object]]:
    specs = {
        "combined": ("inference_service", "local_runtime", "model_dependency"),
        "provider_only": ("inference_service", "model_dependency"),
        "inference_service": ("inference_service",),
        "model_provider": ("model_dependency",),
    }
    rows: list[dict[str, object]] = []
    edge_frame = edges[edges["space_id"].isin(set(spaces["space_id"]))]
    for analysis, layers in specs.items():
        ranking, summary = fractional_rankings(edge_frame, spaces, layers=layers)
        rows.append(
            {
                **descriptor,
                "analysis": analysis,
                "strict_projects_in_frame": len(spaces),
                **summary,
                "top_provider": (
                    str(ranking.iloc[0]["provider"]) if not ranking.empty else "unresolved"
                ),
            }
        )
    service_ids = set(
        edge_frame.loc[edge_frame["layer"].eq("inference_service"), "space_id"]
    )
    model_ids = set(
        edge_frame.loc[edge_frame["layer"].eq("model_dependency"), "space_id"]
    )
    matched_ids = service_ids & model_ids & set(spaces["space_id"])
    matched_spaces = spaces[spaces["space_id"].isin(matched_ids)]
    matched_edges = edge_frame[edge_frame["space_id"].isin(matched_ids)]
    _, service_summary = fractional_rankings(
        matched_edges,
        matched_spaces,
        layers=("inference_service",),
    )
    _, model_summary = fractional_rankings(
        matched_edges,
        matched_spaces,
        layers=("model_dependency",),
    )
    rows.append(
        {
            **descriptor,
            "analysis": "matched_service_minus_model",
            "strict_projects_in_frame": len(spaces),
            "projects": len(matched_ids),
            "providers": math.nan,
            "hhi": service_summary["hhi"] - model_summary["hhi"],
            "effective_number": math.nan,
            "top1_share": math.nan,
            "top3_share": math.nan,
            "gini": math.nan,
            "top_provider": "not_applicable",
        }
    )
    return rows


def search_design_sensitivity(
    raw_dir: Path,
    candidates: pd.DataFrame,
    spaces: pd.DataFrame,
    edges: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Re-estimate results across frozen rank cutoffs and query ablations."""

    rankings = load_search_rankings(raw_dir)
    strict_ids = set(spaces.loc[spaces["included_strict"].eq(True), "space_id"])
    cutoff_rows: list[dict[str, object]] = []
    for cutoff in SEARCH_CUTOFFS:
        candidate_ids = search_union(rankings, cutoff=cutoff)
        frame_ids = strict_ids & candidate_ids
        frame = spaces[spaces["space_id"].isin(frame_ids)]
        cutoff_rows.extend(
            concentration_variant_rows(
                edges,
                frame,
                descriptor={
                    "rank_cutoff": cutoff,
                    "candidate_union": len(candidate_ids),
                },
            )
        )

    query_rows: list[dict[str, object]] = []
    for term in SEARCH_TERMS:
        candidate_ids = search_union(rankings, cutoff=max(SEARCH_CUTOFFS), excluded_term=term)
        frame_ids = strict_ids & candidate_ids
        frame = spaces[spaces["space_id"].isin(frame_ids)]
        query_rows.extend(
            concentration_variant_rows(
                edges,
                frame,
                descriptor={
                    "excluded_query": term,
                    "candidate_union": len(candidate_ids),
                },
            )
        )

    arm_ids = {
        arm: search_union(rankings, cutoff=max(SEARCH_CUTOFFS), arm_filter=arm)
        for arm in SEARCH_ARMS
    }
    strict_arm_ids = {arm: ids & strict_ids for arm, ids in arm_ids.items()}
    left, right = SEARCH_ARMS
    overlap = strict_arm_ids[left] & strict_arm_ids[right]
    union = strict_arm_ids[left] | strict_arm_ids[right]
    arm_summary: dict[str, object] = {
        "candidate_audit_rows": len(candidates),
        "full_strict_sample": len(strict_ids),
        "arms": {
            arm: {
                "candidate_union": len(arm_ids[arm]),
                "strict_projects": len(strict_arm_ids[arm]),
            }
            for arm in SEARCH_ARMS
        },
        "strict_overlap_projects": len(overlap),
        "strict_union_projects": len(union),
        "strict_arm_jaccard": len(overlap) / len(union) if union else None,
        "boundary": (
            "sensitivity within the frozen query vocabulary and returned ranks; "
            "not a representativeness correction"
        ),
    }
    return pd.DataFrame(cutoff_rows), pd.DataFrame(query_rows), arm_summary


def matched_robustness_table(
    raw_dir: Path,
    spaces: pd.DataFrame,
    edges: pd.DataFrame,
    similarity_assignments: pd.DataFrame,
) -> pd.DataFrame:
    """Apply every robustness choice to the same paired layer estimand."""

    rankings = load_search_rankings(raw_dir)
    strict_ids = set(spaces["space_id"].astype(str))
    rows: list[dict[str, object]] = []

    def record(
        family: str,
        variant: str,
        frame: pd.DataFrame,
        *,
        edge_frame: pd.DataFrame = edges,
        base_weight_column: str | None = None,
        boundary: str = "",
    ) -> None:
        summary = paired_layer_estimate(
            edge_frame,
            frame,
            base_weight_column=base_weight_column,
        )
        rows.append(
            {
                "robustness_family": family,
                "variant": variant,
                "strict_projects_in_frame": len(frame),
                "boundary": boundary,
                **summary,
            }
        )

    record("primary", "project_equal", spaces)
    for row in leave_one_service_provider_out(edges, spaces).to_dict("records"):
        rows.append(row)
    for cutoff in SEARCH_CUTOFFS:
        frame_ids = strict_ids & search_union(rankings, cutoff=cutoff)
        record(
            "rank_cutoff",
            str(cutoff),
            spaces[spaces["space_id"].isin(frame_ids)],
            boundary="frozen query vocabulary and returned ranks",
        )
    for term in SEARCH_TERMS:
        frame_ids = strict_ids & search_union(
            rankings,
            cutoff=max(SEARCH_CUTOFFS),
            excluded_term=term,
        )
        record(
            "leave_one_query_out",
            term,
            spaces[spaces["space_id"].isin(frame_ids)],
            boundary="frozen query vocabulary and returned ranks",
        )

    evidence_variants = {
        "high_confidence_both_layers": edges[edges["confidence"].eq("high")],
        "hub_linked_model_only": edges[
            edges["layer"].eq("inference_service")
            | (
                edges["layer"].eq("model_dependency")
                & edges["evidence_type"].eq("hf_linked_model")
            )
        ],
        "service_code_signature_only": edges[
            edges["layer"].eq("model_dependency")
            | (
                edges["layer"].eq("inference_service")
                & edges["evidence_type"].eq("code_signature")
            )
        ],
        "service_package_only": edges[
            edges["layer"].eq("model_dependency")
            | (
                edges["layer"].eq("inference_service")
                & edges["evidence_type"].eq("package")
            )
        ],
    }
    for variant, edge_frame in evidence_variants.items():
        record("evidence", variant, spaces, edge_frame=edge_frame)

    record("weighting", "likes_plus_one", spaces, base_weight_column="likes_weight")
    for arm in SEARCH_ARMS:
        frame = spaces[
            spaces["discovery_arms"].fillna("").str.contains(arm, regex=False)
        ]
        record("search_arm", arm, frame)
    record(
        "source_reuse",
        "exact_multifile_cluster",
        spaces,
        base_weight_column="source_cluster_weight",
    )
    for threshold in SIMILARITY_THRESHOLDS:
        threshold_frame = similarity_assignments[
            np.isclose(similarity_assignments["threshold"], threshold)
        ].set_index("space_id")
        weighted = spaces.copy()
        weight_column = f"source_similarity_weight_{int(round(threshold * 100)):03d}"
        weighted[weight_column] = (
            weighted["space_id"]
            .map(threshold_frame["source_similarity_cluster_weight"])
            .fillna(1.0)
        )
        record(
            "source_reuse",
            f"near_duplicate_jaccard_{int(round(threshold * 100)):03d}",
            weighted,
            base_weight_column=weight_column,
        )

    author_weighted = spaces.copy()
    author_sizes = author_weighted.groupby("author")["space_id"].transform("size")
    author_weighted["author_cluster_weight"] = 1.0 / author_sizes
    record(
        "source_reuse",
        "author_cluster",
        author_weighted,
        base_weight_column="author_cluster_weight",
    )

    immediate = edges.copy()
    model_mask = immediate["layer"].eq("model_dependency")
    immediate.loc[model_mask, "provider"] = immediate.loc[
        model_mask, "evidence_value"
    ].fillna("unscoped").astype(str).map(
        lambda value: f"namespace:{value.split('/', 1)[0]}" if "/" in value else "namespace:unscoped"
    )
    record(
        "model_taxonomy",
        "immediate_public_namespace",
        spaces,
        edge_frame=immediate,
        boundary="namespace is an observable artifact owner, not verified corporate control",
    )
    non_model_edges = edges[~edges["layer"].eq("model_dependency")]
    for variant, model_edges in model_namespace_variants(edges).items():
        variant_edges = pd.concat(
            [non_model_edges, model_edges], ignore_index=True, sort=False
        )
        record(
            "model_taxonomy",
            variant,
            spaces,
            edge_frame=variant_edges,
            boundary=(
                "model publisher mapping varies only by the deterministic treatment "
                "of unresolved public namespaces"
            ),
        )
    return pd.DataFrame(rows)


def leave_one_service_provider_out(
    edges: pd.DataFrame,
    spaces: pd.DataFrame,
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Re-estimate the paired contrast after omitting each provider's projects.

    The project, rather than only the named edge, is removed from both layers. This
    prevents a focal service provider from changing the service denominator while
    leaving the same projects in the model layer.
    """

    space_ids = set(spaces["space_id"].astype(str))
    service_edges = edges[
        edges["space_id"].astype(str).isin(space_ids)
        & edges["layer"].eq("inference_service")
        & edges["provider"].fillna("").astype(str).ne("")
    ]
    rows: list[dict[str, object]] = []
    for provider in sorted(service_edges["provider"].astype(str).unique(), key=str.casefold):
        exposed_ids = set(
            service_edges.loc[
                service_edges["provider"].astype(str).eq(provider), "space_id"
            ].astype(str)
        )
        frame = spaces[~spaces["space_id"].astype(str).isin(exposed_ids)]
        summary = paired_layer_estimate(
            edges,
            frame,
            draws=draws,
            seed=seed,
        )
        rows.append(
            {
                "robustness_family": "leave_one_service_provider_out",
                "variant": provider,
                "strict_projects_in_frame": len(frame),
                "omitted_projects": len(exposed_ids),
                "boundary": (
                    "omits every project declaring the focal inference-service "
                    "provider from both layers"
                ),
                **summary,
            }
        )
    return pd.DataFrame(rows)


def source_token_shingles(
    text: str,
    *,
    shingle_size: int = SOURCE_SHINGLE_SIZE,
) -> set[str]:
    """Create language-agnostic lexical shingles without human labels."""

    normalized = _STRING_LITERAL_RE.sub(" STR ", text)
    normalized = _BLOCK_COMMENT_RE.sub(" ", normalized)
    normalized = _LINE_COMMENT_RE.sub(" ", normalized)
    tokens = [
        "NUM" if token[0].isdigit() else token.casefold()
        for token in _SOURCE_TOKEN_RE.findall(normalized)
    ]
    return {
        hashlib.blake2b(
            "\x1f".join(tokens[index : index + shingle_size]).encode("utf-8"),
            digest_size=8,
        ).hexdigest()
        for index in range(max(0, len(tokens) - shingle_size + 1))
    }


def source_cache_path(root: Path, space_id: str, revision: str, file_path: str) -> Path:
    key = f"{space_id}\n{revision}\n{file_path}".encode("utf-8")
    return root / "data/raw/file_cache" / f"{hashlib.sha256(key).hexdigest()}.json.gz"


def write_project_source_shingles(
    path: Path, project_shingles: dict[str, set[str]]
) -> None:
    """Release irreversible shingle hashes, never third-party source bodies."""

    payload = {
        space_id: sorted(shingles)
        for space_id, shingles in sorted(project_shingles.items(), key=lambda item: item[0].casefold())
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as handle:
            handle.write(serialized)


def load_released_project_shingles(
    root: Path, project_ids: set[str]
) -> tuple[dict[str, set[str]], pd.DataFrame]:
    archive = root / "data/processed/source_shingles.json.gz"
    coverage_path = root / "data/qa/source_similarity_coverage.csv"
    if not archive.exists() or not coverage_path.exists():
        raise FileNotFoundError(
            "source caches are absent and released shingle inputs are incomplete"
        )
    with gzip.open(archive, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError("released source-shingle archive must contain a JSON object")
    shingles = {
        str(space_id): {str(value) for value in values}
        for space_id, values in payload.items()
        if str(space_id) in project_ids
    }
    coverage = pd.read_csv(coverage_path)
    coverage = coverage[coverage["space_id"].astype(str).isin(project_ids)].copy()
    if set(coverage["space_id"].astype(str)) != project_ids:
        raise ValueError("released source-similarity coverage omits strict projects")
    eligible = set(
        coverage.loc[coverage["similarity_eligible"].eq(True), "space_id"].astype(str)
    )
    if set(shingles) != eligible:
        raise ValueError("released shingle archive disagrees with coverage eligibility")
    return shingles, coverage


def load_project_source_shingles(
    root: Path,
    manifest: pd.DataFrame,
    project_ids: Iterable[str],
) -> tuple[dict[str, set[str]], pd.DataFrame]:
    """Load cached executable files and construct one shingle set per project."""

    project_set = set(project_ids)
    cache_root = root / "data/raw/file_cache"
    if not cache_root.exists():
        return load_released_project_shingles(root, project_set)
    frame = manifest[
        manifest["space_id"].isin(project_set) & manifest["status_code"].eq(200)
    ]
    project_shingles: dict[str, set[str]] = {}
    rows: list[dict[str, object]] = []
    for space_id in sorted(project_set, key=str.casefold):
        group = frame[frame["space_id"].eq(space_id)]
        shingles: set[str] = set()
        code_files = 0
        missing_cache_files = 0
        for row in group.itertuples(index=False):
            if PurePosixPath(str(row.file_path)).suffix.casefold() not in CODE_SUFFIXES:
                continue
            path = source_cache_path(
                root,
                str(row.space_id),
                str(row.revision),
                str(row.file_path),
            )
            if not path.exists():
                missing_cache_files += 1
                continue
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
            text = payload.get("text")
            if not isinstance(text, str):
                continue
            code_files += 1
            shingles.update(source_token_shingles(text))
        eligible = len(shingles) >= MIN_SOURCE_SHINGLES
        if eligible:
            project_shingles[space_id] = shingles
        rows.append(
            {
                "space_id": space_id,
                "code_files": code_files,
                "source_shingles": len(shingles),
                "similarity_eligible": eligible,
                "missing_cache_files": missing_cache_files,
            }
        )
    return project_shingles, pd.DataFrame(rows)


class DisjointSet:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            smaller, larger = sorted((left_root, right_root), key=str.casefold)
            self.parent[larger] = smaller


def similarity_cluster_assignments(
    project_shingles: dict[str, set[str]],
    project_ids: Iterable[str],
    *,
    thresholds: Iterable[float] = SIMILARITY_THRESHOLDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster high-Jaccard source bundles by deterministic single linkage."""

    project_list = sorted(set(project_ids), key=str.casefold)
    eligible_ids = sorted(project_shingles, key=str.casefold)
    minimum_threshold = min(thresholds)
    pair_rows: list[dict[str, object]] = []
    for index, left in enumerate(eligible_ids):
        left_shingles = project_shingles[left]
        for right in eligible_ids[index + 1 :]:
            right_shingles = project_shingles[right]
            intersection = len(left_shingles & right_shingles)
            union = len(left_shingles | right_shingles)
            similarity = intersection / union if union else 0.0
            if similarity >= minimum_threshold:
                pair_rows.append(
                    {
                        "left_space_id": left,
                        "right_space_id": right,
                        "jaccard_similarity": similarity,
                        "intersection_shingles": intersection,
                        "union_shingles": union,
                    }
                )
    pairs = pd.DataFrame(pair_rows)

    assignment_rows: list[dict[str, object]] = []
    for threshold in sorted(set(thresholds)):
        disjoint = DisjointSet(project_list)
        for row in pair_rows:
            if float(row["jaccard_similarity"]) >= threshold:
                disjoint.union(str(row["left_space_id"]), str(row["right_space_id"]))
        components: dict[str, list[str]] = {}
        for project_id in project_list:
            components.setdefault(disjoint.find(project_id), []).append(project_id)
        for members in components.values():
            ordered_members = sorted(members, key=str.casefold)
            if len(ordered_members) > 1:
                cluster_id = hashlib.sha256(
                    "\n".join(ordered_members).encode("utf-8")
                ).hexdigest()
            else:
                cluster_id = f"unique:{ordered_members[0]}"
            for project_id in ordered_members:
                assignment_rows.append(
                    {
                        "threshold": threshold,
                        "space_id": project_id,
                        "source_similarity_cluster_id": cluster_id,
                        "source_similarity_cluster_size": len(ordered_members),
                        "source_similarity_cluster_weight": 1.0 / len(ordered_members),
                        "source_shingles": len(project_shingles.get(project_id, set())),
                        "similarity_eligible": project_id in project_shingles,
                        "source_similarity_basis": (
                            "token_jaccard_single_linkage"
                            if project_id in project_shingles
                            else "insufficient_source_shingles"
                        ),
                    }
                )
    return pd.DataFrame(assignment_rows), pairs


def source_similarity_sensitivity(
    assignments: pd.DataFrame,
    spaces: pd.DataFrame,
    edges: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for threshold, group in assignments.groupby("threshold", sort=True):
        frame = spaces.drop(
            columns=["source_similarity_cluster_weight"], errors="ignore"
        ).merge(
            group[["space_id", "source_similarity_cluster_weight"]],
            on="space_id",
            how="left",
            validate="one_to_one",
        )
        frame["source_similarity_cluster_weight"] = frame[
            "source_similarity_cluster_weight"
        ].fillna(1.0)
        frame_edges = edges[edges["space_id"].isin(set(frame["space_id"]))]
        combined_ranking, combined = fractional_rankings(
            frame_edges,
            frame,
            layers=("inference_service", "local_runtime", "model_dependency"),
            base_weight_column="source_similarity_cluster_weight",
        )
        _, provider_only = fractional_rankings(
            frame_edges,
            frame,
            layers=("inference_service", "model_dependency"),
            base_weight_column="source_similarity_cluster_weight",
        )
        duplicated = group[group["source_similarity_cluster_size"].gt(1)]
        rows.append(
            {
                "threshold": threshold,
                "shingle_size": SOURCE_SHINGLE_SIZE,
                "minimum_shingles": MIN_SOURCE_SHINGLES,
                "eligible_projects": int(group["similarity_eligible"].sum()),
                "clustered_projects": len(duplicated),
                "non_singleton_clusters": int(
                    duplicated["source_similarity_cluster_id"].nunique()
                ),
                "largest_cluster": int(group["source_similarity_cluster_size"].max()),
                "effective_project_clusters": float(
                    group["source_similarity_cluster_weight"].sum()
                ),
                "combined_hhi": combined["hhi"],
                "provider_only_hhi": provider_only["hhi"],
                "combined_top_provider": (
                    str(combined_ranking.iloc[0]["provider"])
                    if not combined_ranking.empty
                    else "unresolved"
                ),
                "combined_top_share": combined["top1_share"],
            }
        )
    return pd.DataFrame(rows)


def top_provider(summary_table: pd.DataFrame, analysis: str) -> str:
    subset = summary_table[summary_table["analysis"] == analysis]
    return str(subset.iloc[0]["provider"]) if not subset.empty else "unresolved"


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}"


def tex_escape(value: str) -> str:
    return value.replace("&", r"\&").replace("_", r"\_")


def build_generated_results(path: Path | None, macros: dict[str, object]) -> None:
    if path is None:
        return
    lines = ["% Generated by pilot/src/analyze_dependencies.py; do not edit manually."]
    for name, value in macros.items():
        lines.append(rf"\newcommand{{\{name}}}{{{value}}}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(root: Path, *, write_tex: bool = False) -> dict:
    processed = root / "data/processed"
    qa = root / "data/qa"
    results = root / "analysis_results"
    results.mkdir(parents=True, exist_ok=True)
    spaces_all = pd.read_csv(processed / "space_frame.csv")
    edges_all = pd.read_csv(processed / "dependency_edges.csv")
    models = pd.read_csv(processed / "model_frame.csv")
    files = pd.read_csv(qa / "source_file_manifest.csv")
    candidates = pd.read_csv(qa / "candidate_selection_audit.csv")

    clusters = source_cluster_assignments(files)
    clusters.to_csv(processed / "source_clone_clusters.csv", index=False)
    spaces_all = spaces_all.merge(
        clusters[["space_id", "source_cluster_id", "source_cluster_size", "source_cluster_weight"]],
        on="space_id",
        how="left",
        validate="one_to_one",
    )
    spaces_all["source_cluster_size"] = spaces_all["source_cluster_size"].fillna(1)
    spaces_all["source_cluster_weight"] = spaces_all["source_cluster_weight"].fillna(1.0)
    spaces_all["likes_weight"] = spaces_all["likes"].fillna(0).astype(float) + 1.0
    main = spaces_all[spaces_all["included_strict"].eq(True)].copy()
    broad = spaces_all[spaces_all["included_broad"].eq(True)].copy()
    candidate_only_projects = int(
        (
            ~main["known_dependency_observable"].fillna(False).astype(bool)
            & main["machine_service_candidate_observable"].fillna(False).astype(bool)
        ).sum()
    )

    project_shingles, similarity_coverage = load_project_source_shingles(
        root,
        files,
        main["space_id"],
    )
    write_project_source_shingles(
        processed / "source_shingles.json.gz", project_shingles
    )
    similarity_assignments, similarity_pairs = similarity_cluster_assignments(
        project_shingles,
        main["space_id"],
    )
    similarity_coverage.to_csv(qa / "source_similarity_coverage.csv", index=False)
    similarity_assignments.to_csv(
        processed / "source_similarity_clusters.csv", index=False
    )
    similarity_pairs.to_csv(qa / "source_similarity_pairs.csv", index=False)
    primary_similarity = similarity_assignments[
        np.isclose(
            similarity_assignments["threshold"], PRIMARY_SIMILARITY_THRESHOLD
        )
    ]
    main = main.merge(
        primary_similarity[["space_id", "source_similarity_cluster_weight"]],
        on="space_id",
        how="left",
        validate="one_to_one",
    )
    main["source_similarity_cluster_weight"] = main[
        "source_similarity_cluster_weight"
    ].fillna(1.0)

    analysis_specs = {
        "combined_primary": (main, ("inference_service", "local_runtime", "model_dependency"), None),
        "provider_only_primary": (main, ("inference_service", "model_dependency"), None),
        "inference_service_primary": (main, ("inference_service",), None),
        "model_provider_primary": (main, ("model_dependency",), None),
        "combined_broad": (broad, ("inference_service", "local_runtime", "model_dependency"), None),
        "combined_clone_adjusted": (main, ("inference_service", "local_runtime", "model_dependency"), "source_cluster_weight"),
        "combined_near_duplicate_090": (
            main,
            ("inference_service", "local_runtime", "model_dependency"),
            "source_similarity_cluster_weight",
        ),
        "combined_likes_weighted": (main, ("inference_service", "local_runtime", "model_dependency"), "likes_weight"),
        "combined_likes_arm": (
            main[main["discovery_arms"].fillna("").str.contains("likes", regex=False)],
            ("inference_service", "local_runtime", "model_dependency"),
            None,
        ),
        "combined_recent_arm": (
            main[main["discovery_arms"].fillna("").str.contains("createdAt", regex=False)],
            ("inference_service", "local_runtime", "model_dependency"),
            None,
        ),
    }
    ranking_rows: list[pd.DataFrame] = []
    concentration_rows: list[dict[str, object]] = []
    for name, (frame, layers, weight_column) in analysis_specs.items():
        frame_edges = edges_all[edges_all["space_id"].isin(set(frame["space_id"]))]
        ranking, summary = fractional_rankings(
            frame_edges,
            frame,
            layers=layers,
            base_weight_column=weight_column,
        )
        if not ranking.empty:
            ranking = ranking.rename(columns={"provider": "provider"})
            ranking.insert(0, "analysis", name)
            ranking_rows.append(ranking)
        if weight_column is None:
            ci_low, ci_high = bootstrap_hhi(frame_edges, frame, layers=layers)
        else:
            ci_low, ci_high = math.nan, math.nan
        concentration_rows.append(
            {
                "analysis": name,
                "layers": ";".join(layers),
                "weighting": weight_column or "project_equal",
                **summary,
                "hhi_ci_low": ci_low,
                "hhi_ci_high": ci_high,
            }
        )
    edge_sensitivity_specs = {
        "combined_high_confidence": edges_all[edges_all["confidence"].eq("high")],
        "model_provider_hf_linked_only": edges_all[
            edges_all["layer"].eq("model_dependency")
            & edges_all["evidence_type"].eq("hf_linked_model")
        ],
    }
    for name, sensitivity_edges in edge_sensitivity_specs.items():
        layers = (
            ("model_dependency",)
            if name == "model_provider_hf_linked_only"
            else ("inference_service", "local_runtime", "model_dependency")
        )
        frame_edges = sensitivity_edges[
            sensitivity_edges["space_id"].isin(set(main["space_id"]))
        ]
        ranking, summary = fractional_rankings(frame_edges, main, layers=layers)
        if not ranking.empty:
            ranking.insert(0, "analysis", name)
            ranking_rows.append(ranking)
        ci_low, ci_high = bootstrap_hhi(frame_edges, main, layers=layers)
        concentration_rows.append(
            {
                "analysis": name,
                "layers": ";".join(layers),
                "weighting": "project_equal",
                **summary,
                "hhi_ci_low": ci_low,
                "hhi_ci_high": ci_high,
            }
        )
    provider_rankings = pd.concat(ranking_rows, ignore_index=True)
    concentration = pd.DataFrame(concentration_rows)
    provider_rankings.to_csv(results / "provider_rankings.csv", index=False)
    concentration.to_csv(results / "concentration_summary.csv", index=False)
    concentration.to_csv(results / "robustness_matrix.csv", index=False)

    paired_summary, paired_bootstrap = paired_layer_bootstrap(
        edges_all,
        main["space_id"],
    )
    write_json(results / "paired_layer_comparison.json", paired_summary)
    paired_bootstrap.to_csv(results / "paired_layer_bootstrap.csv", index=False)
    (
        codeclaration_summary,
        codeclaration_joint,
        codeclaration_service,
        codeclaration_null,
    ) = cross_layer_codeclaration(edges_all, main["space_id"])
    write_json(
        results / "cross_layer_codeclaration_summary.json",
        codeclaration_summary,
    )
    codeclaration_joint.to_csv(
        results / "cross_layer_codeclaration.csv", index=False
    )
    codeclaration_service.to_csv(
        results / "cross_layer_service_model_diversity.csv", index=False
    )
    codeclaration_null.to_csv(
        results / "cross_layer_codeclaration_null.csv", index=False
    )
    namespace_results = namespace_sensitivity(edges_all, main)
    namespace_results.to_csv(results / "model_namespace_sensitivity.csv", index=False)

    search_raw_dir = root / "data/raw" / SNAPSHOT_DATE / "search_expanded"
    search_rankings_frame(load_search_rankings(search_raw_dir)).to_csv(
        processed / "search_rankings.csv", index=False
    )
    cutoff_results, query_ablation, arm_overlap = search_design_sensitivity(
        search_raw_dir,
        candidates,
        spaces_all,
        edges_all,
    )
    cutoff_results.to_csv(results / "search_rank_cutoff_sensitivity.csv", index=False)
    query_ablation.to_csv(results / "search_query_ablation.csv", index=False)
    write_json(results / "search_arm_overlap.json", arm_overlap)
    matched_robustness = matched_robustness_table(
        search_raw_dir,
        main,
        edges_all,
        similarity_assignments,
    )
    matched_robustness.to_csv(results / "matched_robustness.csv", index=False)
    provider_omission = matched_robustness[
        matched_robustness["robustness_family"].eq(
            "leave_one_service_provider_out"
        )
    ].copy()
    provider_omission.to_csv(
        results / "service_provider_omission.csv", index=False
    )
    matched_composition = matched_sample_composition(main, edges_all)
    matched_composition.to_csv(
        results / "matched_sample_composition.csv", index=False
    )
    unmapped_path = qa / "unmapped_dependency_candidates.csv"
    unmapped_candidates = (
        pd.read_csv(unmapped_path)
        if unmapped_path.exists()
        else pd.DataFrame(
            columns=["space_id", "candidate_type", "identifier", "source_file"]
        )
    )
    strict_phrase_projects = int(
        candidates.get("education_strict_match", candidates["included_strict"])
        .fillna(False)
        .astype(bool)
        .sum()
    )
    unknown_service_boundaries = unknown_service_boundary_table(
        main,
        edges_all,
        unmapped_candidates,
        strict_phrase_projects=strict_phrase_projects,
    )
    unknown_service_boundaries.to_csv(
        results / "unknown_service_boundary.csv", index=False
    )

    source_similarity_results = source_similarity_sensitivity(
        similarity_assignments,
        main,
        edges_all,
    )
    source_similarity_results.to_csv(
        results / "source_similarity_sensitivity.csv", index=False
    )
    history_input_paths = (
        processed / "space_frame.csv",
        processed / "dependency_edges.csv",
        processed / "historical_dependency_edges.csv",
        qa / "space_commit_history_audit.csv",
    )
    history_summary = (
        recompute_version_history_outputs(root)
        if all(path.exists() for path in history_input_paths)
        else None
    )
    history_span_results = (
        pd.read_csv(results / "version_span_sensitivity.csv")
        if (results / "version_span_sensitivity.csv").exists()
        else pd.DataFrame()
    )

    model_edges = edges_all[
        edges_all["space_id"].isin(set(main["space_id"]))
        & edges_all["layer"].eq("model_dependency")
    ]
    official_model_edges = model_edges[
        model_edges["evidence_type"].eq("hf_linked_model")
    ]
    code_model_edges = model_edges[model_edges["evidence_type"].eq("code_model_id")]
    official_model_projects = set(official_model_edges["space_id"].astype(str))
    code_model_projects = set(code_model_edges["space_id"].astype(str))
    service_project_ids = set(
        edges_all.loc[
            edges_all["space_id"].isin(set(main["space_id"]))
            & edges_all["layer"].eq("inference_service"),
            "space_id",
        ].astype(str)
    )
    environment_service_edges = edges_all[
        edges_all["space_id"].isin(set(main["space_id"]))
        & edges_all["layer"].eq("inference_service")
        & edges_all["source_file"].fillna("").astype(str).map(
            lambda value: PurePosixPath(value).name.casefold() == ".env.example"
        )
    ]
    other_service_projects = set(
        edges_all.loc[
            edges_all["space_id"].isin(set(main["space_id"]))
            & edges_all["layer"].eq("inference_service")
            & ~edges_all["source_file"].fillna("").astype(str).map(
                lambda value: PurePosixPath(value).name.casefold() == ".env.example"
            ),
            "space_id",
        ].astype(str)
    )
    evidence_source_audit = {
        "primary_model_edge_rows": len(model_edges),
        "official_linked_model_edge_rows": len(official_model_edges),
        "official_linked_model_projects": len(official_model_projects),
        "official_linked_unique_model_ids": int(
            official_model_edges["evidence_value"].nunique()
        ),
        "code_model_id_edge_rows": len(code_model_edges),
        "code_model_id_projects": len(code_model_projects),
        "code_only_model_projects": len(code_model_projects - official_model_projects),
        "matched_projects_with_any_code_model_id": len(
            code_model_projects & service_project_ids
        ),
        "matched_projects_with_only_code_model_id": len(
            (code_model_projects - official_model_projects) & service_project_ids
        ),
        "environment_example_service_edge_rows": len(environment_service_edges),
        "environment_example_service_projects": int(
            environment_service_edges["space_id"].nunique()
        ),
        "environment_example_only_service_projects": len(
            set(environment_service_edges["space_id"].astype(str))
            - other_service_projects
        ),
        "boundary": (
            "primary model evidence includes exact public identifiers from Hub metadata and "
            "bounded model-loading code; .env.example is medium-confidence machine-readable "
            "configuration evidence, with high-confidence-only variants reported separately"
        ),
    }
    write_json(results / "evidence_source_audit.json", evidence_source_audit)
    family_ranking, family_summary = fractional_rankings(
        model_edges,
        main,
        layers=("model_dependency",),
        label_column="model_family",
    )
    family_ranking.to_csv(results / "model_family_rankings.csv", index=False)
    write_json(results / "model_family_summary.json", family_summary)

    selection_funnel = pd.DataFrame(
        [
            {"stage": "query_union", "projects": len(candidates)},
            {
                "stage": "strict_education_phrase",
                "projects": strict_phrase_projects,
            },
            {"stage": "dependency_observable", "projects": len(main)},
            {
                "stage": "identifiable_service_model_matched",
                "projects": int(paired_summary["matched_projects"]),
            },
        ]
    )
    selection_funnel.to_csv(results / "selection_funnel.csv", index=False)

    main_model_ids = set(
        model_edges["evidence_value"].dropna().astype(str)
    )
    main_models = models[models["model_id"].isin(main_model_ids)].copy()
    official_model_ids = set(
        model_edges.loc[
            model_edges["evidence_type"].eq("hf_linked_model"), "evidence_value"
        ].dropna().astype(str)
    )
    official_models = models[models["model_id"].isin(official_model_ids)].copy()
    model_license_resolved = float(main_models["license"].fillna("missing").ne("missing").mean()) if len(main_models) else math.nan
    official_model_license_resolved = (
        float(official_models["license"].fillna("missing").ne("missing").mean())
        if len(official_models)
        else math.nan
    )
    license_rows = []
    for status, count in main["rights_review_status"].value_counts().items():
        license_rows.append(
            {
                "measure": "rights_review_status",
                "category": status,
                "count": int(count),
                "denominator": len(main),
                "share": float(count / len(main)),
            }
        )
    app_license_disclosed = int(main["app_license"].fillna("missing").ne("missing").sum())
    license_rows.extend(
        [
            {
                "measure": "app_license_disclosure",
                "category": "disclosed",
                "count": app_license_disclosed,
                "denominator": len(main),
                "share": app_license_disclosed / len(main),
            },
            {
                "measure": "model_reference_license_disclosure",
                "category": "disclosed",
                "count": int(main_models["license"].fillna("missing").ne("missing").sum()),
                "denominator": len(main_models),
                "share": model_license_resolved,
            },
            {
                "measure": "official_linked_model_license_disclosure",
                "category": "disclosed",
                "count": int(
                    official_models["license"].fillna("missing").ne("missing").sum()
                ),
                "denominator": len(official_models),
                "share": official_model_license_resolved,
            },
        ]
    )
    license_summary = pd.DataFrame(license_rows)
    license_summary.to_csv(results / "license_summary.csv", index=False)

    snapshot = pd.Timestamp(SNAPSHOT_DATE, tz="UTC")
    modified = pd.to_datetime(main["last_modified"], utc=True, errors="coerce")
    age_days = (snapshot - modified).dt.total_seconds() / 86_400
    main["last_modified_age_days"] = age_days
    runtime_reported = main["runtime_stage"].ne("unreported")
    runtime_running = main["runtime_stage"].eq("RUNNING")
    active_summary = {
        "projects": len(main),
        "runtime_status_reported": int(runtime_reported.sum()),
        "runtime_status_coverage": float(runtime_reported.mean()),
        "runtime_running": int(runtime_running.sum()),
        "runtime_running_share_among_reported": (
            float(runtime_running[runtime_reported].mean()) if runtime_reported.any() else None
        ),
        "disabled": int(main["disabled"].eq(True).sum()),
        "modified_within_180_days": int(age_days.le(180).sum()),
        "modified_within_180_days_share": float(age_days.le(180).mean()),
        "not_modified_for_365_days": int(age_days.gt(365).sum()),
        "not_modified_for_365_days_share": float(age_days.gt(365).mean()),
        "interpretive_boundary": "snapshot activity indicators; not survival analysis or evidence of model-event effects",
    }
    write_json(results / "activity_summary.json", active_summary)
    main[["space_id", "runtime_stage", "disabled", "last_modified", "last_modified_age_days"]].to_csv(
        results / "project_activity.csv", index=False
    )

    explicit_regions = main["author_region_class"].isin(["asia", "outside_asia"])
    region_coverage = float(explicit_regions.mean())
    declared_languages = main["declared_language_class"].ne("undeclared")
    language_coverage = float(declared_languages.mean())
    asia_model_spaces = set(
        model_edges.loc[model_edges["provider"].isin(ASIA_MODEL_PROVIDERS), "space_id"]
    )
    model_space_ids = set(model_edges["space_id"])
    regional_gate = {
        "author_location_coverage": region_coverage,
        "author_location_projects": int(explicit_regions.sum()),
        "declared_language_coverage": language_coverage,
        "declared_language_projects": int(declared_languages.sum()),
        "author_region_comparison_status": "PASS" if region_coverage >= 0.20 else "NOT_ESTIMABLE",
        "language_orientation_comparison_status": "PASS" if language_coverage >= 0.20 else "NOT_ESTIMABLE",
        "spaces_with_linked_models": len(model_space_ids),
        "spaces_with_asia_provider_model": len(asia_model_spaces),
        "asia_provider_model_share_among_model_spaces": (
            len(asia_model_spaces) / len(model_space_ids) if model_space_ids else None
        ),
        "boundary": "model-provider origin is not developer geography or data-sovereignty compliance",
    }
    write_json(results / "regional_analysis_gate.json", regional_gate)

    main_combined = concentration.set_index("analysis").loc["combined_primary"]
    clone_combined = concentration.set_index("analysis").loc["combined_clone_adjusted"]
    near_combined = concentration.set_index("analysis").loc[
        "combined_near_duplicate_090"
    ]
    provider_only_row = concentration.set_index("analysis").loc[
        "provider_only_primary"
    ]
    main_ranking = provider_rankings[provider_rankings["analysis"].eq("combined_primary")]
    clone_ranking = provider_rankings[provider_rankings["analysis"].eq("combined_clone_adjusted")]
    near_ranking = provider_rankings[
        provider_rankings["analysis"].eq("combined_near_duplicate_090")
    ]
    cutoff_combined = cutoff_results[cutoff_results["analysis"].eq("combined")]
    ablation_combined = query_ablation[query_ablation["analysis"].eq("combined")]
    primary_similarity_result = source_similarity_results[
        np.isclose(
            source_similarity_results["threshold"], PRIMARY_SIMILARITY_THRESHOLD
        )
    ].iloc[0]
    file_cap_results = (
        pd.read_csv(results / "file_cap_sensitivity.csv")
        if (results / "file_cap_sensitivity.csv").exists()
        else pd.DataFrame()
    )
    decision = {
        "annotation_regime": {
            "status": "PASS",
            "manual_annotation_rows": 0,
            "rule": "all inclusion, dependency, namespace, and similarity outputs are deterministic",
        },
        "bounded_sample_size": {
            "status": "PASS" if len(main) >= 100 else "FAIL",
            "value": len(main),
            "threshold": 100,
        },
        "upstream_detection_coverage": {
            "status": "PASS" if int(main_combined["projects"]) / len(main) >= 0.90 else "FAIL",
            "value": int(main_combined["projects"]) / len(main),
            "threshold": 0.90,
        },
        "clone_sensitivity": {
            "status": "PASS" if (
                top_provider(main_ranking, "combined_primary")
                == top_provider(clone_ranking, "combined_clone_adjusted")
                == top_provider(near_ranking, "combined_near_duplicate_090")
            ) else "FAIL",
            "primary_top_provider": top_provider(main_ranking, "combined_primary"),
            "clone_adjusted_top_provider": top_provider(clone_ranking, "combined_clone_adjusted"),
            "near_duplicate_adjusted_top_provider": top_provider(
                near_ranking, "combined_near_duplicate_090"
            ),
            "primary_hhi": float(main_combined["hhi"]),
            "clone_adjusted_hhi": float(clone_combined["hhi"]),
            "near_duplicate_adjusted_hhi": float(near_combined["hhi"]),
        },
        "search_design_sensitivity": {
            "status": "PASS" if (
                cutoff_combined["top_provider"].nunique() == 1
                and ablation_combined["top_provider"].nunique() == 1
                and cutoff_combined.iloc[0]["top_provider"]
                == main_ranking.iloc[0]["provider"]
            ) else "FAIL",
            "rank_cutoff_hhi_range": [
                float(cutoff_combined["hhi"].min()),
                float(cutoff_combined["hhi"].max()),
            ],
            "leave_one_query_out_hhi_range": [
                float(ablation_combined["hhi"].min()),
                float(ablation_combined["hhi"].max()),
            ],
            "boundary": "frozen returned ranks only; not a population-coverage test",
        },
        "matched_layer_contrast": {
            "status": "ESTIMATED",
            "matched_projects": paired_summary["matched_projects"],
            "service_minus_model_hhi": paired_summary[
                "hhi_difference_service_minus_model"
            ],
            "paired_bootstrap_ci": paired_summary[
                "hhi_difference_bootstrap_ci"
            ],
            "interpretation": paired_summary["interpretation"],
        },
        "alternative_concentration_metrics": {
            "status": (
                "PASS"
                if paired_summary["top_share_difference_bootstrap_ci"][0] > 0
                and paired_summary["shannon_entropy_difference_bootstrap_ci"][0] > 0
                else "FAIL"
            ),
            "service_top_share": paired_summary["service_top_share"],
            "model_top_share": paired_summary["model_top_share"],
            "top_share_difference_interval": paired_summary[
                "top_share_difference_bootstrap_ci"
            ],
            "service_shannon_effective_categories": paired_summary[
                "service_shannon_effective_categories"
            ],
            "model_shannon_effective_categories": paired_summary[
                "model_shannon_effective_categories"
            ],
            "shannon_entropy_difference_interval": paired_summary[
                "shannon_entropy_difference_bootstrap_ci"
            ],
        },
        "cross_layer_codeclaration": {
            "status": "ESTIMATED",
            "matched_projects": codeclaration_summary["matched_projects"],
            "normalized_mutual_information_sqrt": codeclaration_summary[
                "normalized_mutual_information_sqrt"
            ],
            "randomization_p": codeclaration_summary["randomization_p"],
            "boundary": codeclaration_summary["interpretation"],
        },
        "service_provider_omission": {
            "status": (
                "PASS"
                if provider_omission["hhi_difference_ci_low"].gt(0).all()
                else "FAIL"
            ),
            "providers_omitted_individually": len(provider_omission),
            "minimum_hhi_difference": float(
                provider_omission["hhi_difference_service_minus_model"].min()
            ),
            "minimum_interval_lower_bound": float(
                provider_omission["hhi_difference_ci_low"].min()
            ),
            "boundary": (
                "each variant omits all projects declaring one service provider "
                "from both layers"
            ),
        },
        "source_file_cap_sensitivity": {
            "status": (
                "PASS"
                if not file_cap_results.empty
                and file_cap_results["hhi_difference_ci_low"].gt(0).all()
                else "NOT_TESTED"
            ),
            "caps": (
                file_cap_results["max_source_files"].astype(int).tolist()
                if not file_cap_results.empty
                else []
            ),
            "changed_service_sets_at_cap20": (
                int(
                    file_cap_results.set_index("max_source_files").loc[
                        20, "projects_with_changed_service_set_vs_cap10"
                    ]
                )
                if not file_cap_results.empty
                else None
            ),
            "boundary": (
                "same frozen revisions; service-source cap varies while the "
                "referenced-model layer remains fixed"
            ),
        },
        "version_history_audit": {
            "status": "ESTIMATED" if history_summary else "NOT_TESTED",
            "resolved_projects": (
                history_summary["earliest_analyzable_state_resolved"]
                if history_summary
                else 0
            ),
            "paired_service_projects": (
                history_summary["paired_service_concentration"]["paired_projects"]
                if history_summary
                else 0
            ),
            "boundary": (
                history_summary["interpretive_boundary"]
                if history_summary
                else "historical collector has not been run"
            ),
        },
        "regional_comparison": {
            "status": regional_gate["author_region_comparison_status"],
            "coverage": region_coverage,
        },
        "longitudinal_disruption_analysis": {
            "status": "NOT_TESTED",
            "reason": "one frozen snapshot cannot identify upstream events or downstream survival responses",
        },
        "route": "bounded_global_cross_sectional_audit",
    }
    write_json(results / "decision_gate.json", decision)

    duplicate_spaces = int(
        clusters.loc[clusters["source_cluster_size"].gt(1), "space_id"].isin(set(main["space_id"])).sum()
    )
    app_disclosure_share = app_license_disclosed / len(main)
    primary_top = main_ranking.iloc[0]
    service_row = concentration.set_index("analysis").loc["inference_service_primary"]
    model_row = concentration.set_index("analysis").loc["model_provider_primary"]
    manifest = {
        "snapshot_date": SNAPSHOT_DATE,
        "discovery_candidates": len(candidates),
        "strict_education_phrase_projects": strict_phrase_projects,
        "dependency_observable_projects": len(main),
        "identifiable_service_model_matched_projects": int(
            paired_summary["matched_projects"]
        ),
        "strict_primary_projects": len(main),
        "machine_candidate_only_projects": candidate_only_projects,
        "unique_authors": int(main["author"].nunique()),
        "dependency_edges": int(edges_all[edges_all["space_id"].isin(set(main["space_id"]))].shape[0]),
        "exact_source_duplicate_spaces": duplicate_spaces,
        "near_duplicate_clustered_spaces_090": int(
            primary_similarity_result["clustered_projects"]
        ),
        "primary_combined_hhi": float(main_combined["hhi"]),
        "primary_combined_hhi_ci": [float(main_combined["hhi_ci_low"]), float(main_combined["hhi_ci_high"])],
        "primary_combined_effective_providers": float(main_combined["effective_number"]),
        "primary_top_provider": str(primary_top["provider"]),
        "primary_top_provider_share": float(primary_top["fractional_share"]),
        "provider_only_hhi": float(provider_only_row["hhi"]),
        "service_hhi": float(service_row["hhi"]),
        "model_provider_hhi": float(model_row["hhi"]),
        "matched_layer_comparison": paired_summary,
        "cross_layer_codeclaration": codeclaration_summary,
        "service_provider_omission": {
            "providers_omitted_individually": len(provider_omission),
            "minimum_hhi_difference": float(
                provider_omission["hhi_difference_service_minus_model"].min()
            ),
            "minimum_interval_lower_bound": float(
                provider_omission["hhi_difference_ci_low"].min()
            ),
        },
        "source_file_cap_sensitivity": (
            file_cap_results.to_dict("records") if not file_cap_results.empty else []
        ),
        "version_history_audit": history_summary,
        "near_duplicate_adjusted_hhi_090": float(near_combined["hhi"]),
        "search_rank_cutoff_combined_hhi_range": [
            float(cutoff_combined["hhi"].min()),
            float(cutoff_combined["hhi"].max()),
        ],
        "search_query_ablation_combined_hhi_range": [
            float(ablation_combined["hhi"].min()),
            float(ablation_combined["hhi"].max()),
        ],
        "app_license_disclosure_share": app_disclosure_share,
        "model_license_disclosure_share": model_license_resolved,
        "official_linked_model_license_disclosure_share": official_model_license_resolved,
        "author_location_coverage": region_coverage,
        "asia_provider_model_share_among_model_spaces": regional_gate["asia_provider_model_share_among_model_spaces"],
        "interpretation": "descriptive estimates for a frozen query-defined public sample",
    }
    write_json(results / "analysis_manifest.json", manifest)

    history_paired = (
        history_summary["paired_service_concentration"]
        if history_summary
        else {}
    )
    hugging_face_omission = provider_omission.set_index("variant").loc[
        "Hugging Face"
    ]
    top_service_diversity = codeclaration_service.iloc[0]
    file_cap_20 = (
        file_cap_results.set_index("max_source_files").loc[20]
        if not file_cap_results.empty
        else None
    )
    composition_by_metric = matched_composition.set_index("metric")
    study_support_composition = composition_by_metric.loc[
        "construct=study_support"
    ]
    age_composition = composition_by_metric.loc["created_age_days"]
    likes_composition = composition_by_metric.loc["log1p_likes"]
    history_30 = (
        history_span_results[
            history_span_results["minimum_history_span_days"].eq(30)
        ].iloc[0]
        if not history_span_results.empty
        and history_span_results["minimum_history_span_days"].eq(30).any()
        else None
    )
    project_unique_unknown = unknown_service_boundaries.set_index(
        "unknown_treatment"
    ).loc["project_unique_unknown_bound"]

    paper_dir = root.parent / "paper"
    build_generated_results(
        paper_dir / "generated_results.tex" if write_tex else None,
        {
            "DiscoveryN": len(candidates),
            "StrictPhraseN": int(
                candidates["education_strict_match"].fillna(False).astype(bool).sum()
            ),
            "StrictN": len(main),
            "SourceFileN": len(files),
            "CandidateOnlyN": candidate_only_projects,
            "AuthorN": int(main["author"].nunique()),
            "ModelN": len(main_models),
            "OfficialModelN": len(official_models),
            "CombinedHHI": f"{float(main_combined['hhi']):.3f}",
            "CombinedEffectiveN": f"{float(main_combined['effective_number']):.1f}",
            "TopProvider": tex_escape(str(primary_top["provider"])),
            "TopProviderShare": pct(float(primary_top["fractional_share"])),
            "ProviderOnlyHHI": f"{float(provider_only_row['hhi']):.3f}",
            "ServiceHHI": f"{float(service_row['hhi']):.3f}",
            "ServiceProjectN": int(service_row["projects"]),
            "ModelProviderHHI": f"{float(model_row['hhi']):.3f}",
            "ModelProjectN": int(model_row["projects"]),
            "MatchedN": int(paired_summary["matched_projects"]),
            "MatchedServiceHHI": f"{float(paired_summary['service_hhi']):.3f}",
            "MatchedModelHHI": f"{float(paired_summary['model_hhi']):.3f}",
            "MatchedHHIDifference": f"{float(paired_summary['hhi_difference_service_minus_model']):.3f}",
            "MatchedHHIDifferenceLow": f"{float(paired_summary['hhi_difference_bootstrap_ci'][0]):.3f}",
            "MatchedHHIDifferenceHigh": f"{float(paired_summary['hhi_difference_bootstrap_ci'][1]):.3f}",
            "MatchedDominanceShare": pct(
                float(paired_summary["bootstrap_fraction_service_hhi_gt_model_hhi"])
            ),
            "MatchedServiceTopShare": pct(float(paired_summary["service_top_share"])),
            "MatchedModelTopShare": pct(float(paired_summary["model_top_share"])),
            "MatchedTopShareDifferenceLow": pct(
                float(paired_summary["top_share_difference_bootstrap_ci"][0])
            ),
            "MatchedTopShareDifferenceHigh": pct(
                float(paired_summary["top_share_difference_bootstrap_ci"][1])
            ),
            "MatchedServiceShannonEffectiveN": f"{float(paired_summary['service_shannon_effective_categories']):.1f}",
            "MatchedModelShannonEffectiveN": f"{float(paired_summary['model_shannon_effective_categories']):.1f}",
            "CodeclarationNMI": f"{float(codeclaration_summary['normalized_mutual_information_sqrt']):.2f}",
            "CodeclarationMI": f"{float(codeclaration_summary['mutual_information']):.3f}",
            "CodeclarationNullLow": f"{float(codeclaration_summary['permutation_null_interval'][0]):.3f}",
            "CodeclarationNullHigh": f"{float(codeclaration_summary['permutation_null_interval'][1]):.3f}",
            "CodeclarationP": f"{float(codeclaration_summary['randomization_p']):.4f}",
            "TopServiceCodeclaredModelEffectiveN": f"{float(top_service_diversity['conditional_model_shannon_effective_categories']):.1f}",
            "TopServiceCodeclaredModelTopShare": pct(
                float(top_service_diversity["top_codeclared_model_share_within_service"])
            ),
            "HFOmissionN": int(hugging_face_omission["matched_projects"]),
            "HFOmissionDifference": f"{float(hugging_face_omission['hhi_difference_service_minus_model']):.3f}",
            "HFOmissionDifferenceLow": f"{float(hugging_face_omission['hhi_difference_ci_low']):.3f}",
            "HFOmissionDifferenceHigh": f"{float(hugging_face_omission['hhi_difference_ci_high']):.3f}",
            "FileCapTwentyChangedN": (
                int(file_cap_20["projects_with_changed_service_set_vs_cap10"])
                if file_cap_20 is not None
                else 0
            ),
            "FileCapTwentyMatchedN": (
                int(file_cap_20["matched_projects"])
                if file_cap_20 is not None
                else 0
            ),
            "FileCapTwentyDifference": (
                f"{float(file_cap_20['hhi_difference_service_minus_model']):.3f}"
                if file_cap_20 is not None
                else "NA"
            ),
            "FileCapTwentyDifferenceLow": (
                f"{float(file_cap_20['hhi_difference_ci_low']):.3f}"
                if file_cap_20 is not None
                else "NA"
            ),
            "FileCapTwentyDifferenceHigh": (
                f"{float(file_cap_20['hhi_difference_ci_high']):.3f}"
                if file_cap_20 is not None
                else "NA"
            ),
            "ProjectUniqueUnknownDifference": f"{float(project_unique_unknown['hhi_difference_service_minus_model']):.3f}",
            "ProjectUniqueUnknownDifferenceLow": f"{float(project_unique_unknown['hhi_difference_ci_low']):.3f}",
            "ProjectUniqueUnknownDifferenceHigh": f"{float(project_unique_unknown['hhi_difference_ci_high']):.3f}",
            "MatchedStudySupportShare": pct(
                float(study_support_composition["matched_value"])
            ),
            "NonmatchedStudySupportShare": pct(
                float(study_support_composition["nonmatched_value"])
            ),
            "StudySupportStandardizedDifference": f"{float(study_support_composition['standardized_difference']):.2f}",
            "MatchedAgeStandardizedDifferenceAbs": f"{abs(float(age_composition['standardized_difference'])):.2f}",
            "MatchedLikesStandardizedDifferenceAbs": f"{abs(float(likes_composition['standardized_difference'])):.2f}",
            "NearDuplicateN": int(primary_similarity_result["clustered_projects"]),
            "NearDuplicateHHI": f"{float(near_combined['hhi']):.3f}",
            "SearchCutoffMinN": int(cutoff_combined["strict_projects_in_frame"].min()),
            "SearchCutoffHHILow": f"{float(cutoff_combined['hhi'].min()):.3f}",
            "SearchCutoffHHIHigh": f"{float(cutoff_combined['hhi'].max()):.3f}",
            "QueryAblationHHILow": f"{float(ablation_combined['hhi'].min()):.3f}",
            "QueryAblationHHIHigh": f"{float(ablation_combined['hhi'].max()):.3f}",
            "HistoryResolvedN": int(
                history_summary["earliest_analyzable_state_resolved"]
                if history_summary
                else 0
            ),
            "HistoryPairedN": int(history_paired.get("paired_projects", 0)),
            "HistoryInitialServiceHHI": f"{float(history_paired.get('initial_service_hhi', math.nan)):.3f}",
            "HistoryCurrentServiceHHI": f"{float(history_paired.get('current_service_hhi', math.nan)):.3f}",
            "HistoryHHIChange": f"{float(history_paired.get('hhi_change_current_minus_initial', math.nan)):.3f}",
            "HistoryHHIChangeLow": f"{float(history_paired.get('hhi_change_bootstrap_ci', [math.nan, math.nan])[0]):.3f}",
            "HistoryHHIChangeHigh": f"{float(history_paired.get('hhi_change_bootstrap_ci', [math.nan, math.nan])[1]):.3f}",
            "HistoryMedianSpan": f"{float(history_summary['median_history_span_days_among_defined'] if history_summary else math.nan):.2f}",
            "HistoryThirtyDayN": int(history_30["projects"] if history_30 is not None else 0),
            "HistoryThirtyDayPairedN": int(
                history_30["paired_service_projects"] if history_30 is not None else 0
            ),
            "HistoryThirtyDayHHIChange": f"{float(history_30['hhi_change_current_minus_initial'] if history_30 is not None else math.nan):.3f}",
            "HistoryThirtyDayHHIChangeLow": f"{float(history_30['hhi_change_ci_low'] if history_30 is not None else math.nan):.3f}",
            "HistoryThirtyDayHHIChangeHigh": f"{float(history_30['hhi_change_ci_high'] if history_30 is not None else math.nan):.3f}",
            "AppLicenseDisclosure": pct(app_disclosure_share),
            "ModelLicenseDisclosure": pct(model_license_resolved),
            "OfficialModelLicenseDisclosure": pct(
                official_model_license_resolved
            ),
            "LocationCoverage": pct(region_coverage),
            "AsiaModelShare": pct(regional_gate["asia_provider_model_share_among_model_spaces"] or 0.0),
            "RecentShare": pct(active_summary["modified_within_180_days_share"]),
            "DormantShare": pct(active_summary["not_modified_for_365_days_share"]),
            "DuplicateN": duplicate_spaces,
        },
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--write-tex",
        action="store_true",
        help="also write the legacy generated-results TeX macro file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_analysis(args.root.resolve(), write_tex=args.write_tex)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
