"""Generate the submission figures for the refined DE-004 manuscript."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


INK = "#111111"
LIGHT = "#E2E5E8"
PALE = "#F5F6F7"
BLUE = "#2A6F97"
BLUE_LIGHT = "#A9C5DF"
ORANGE = "#D97732"
TEAL = "#2A9D8F"
VIOLET = "#725A9A"
GOLD = "#C7952D"
RED = "#B7534F"


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 9.2,
        "axes.labelsize": 9.4,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "legend.fontsize": 9.0,
        "axes.linewidth": 0.65,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "de004-clda-20260831",
        "savefig.facecolor": "white",
    }
)


def boolean(series: pd.Series) -> pd.Series:
    if str(series.dtype) in {"bool", "boolean"}:
        return series.fillna(False).astype(bool)
    return series.fillna("").astype(str).str.casefold().eq("true")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def panel_title(ax: plt.Axes, label: str, title: str) -> None:
    prefix = f"{label}  " if label else ""
    ax.set_title(f"{prefix}{title}", loc="left", fontweight="bold", pad=8)


def clean_axes(ax: plt.Axes, *, grid_axis: str | None = None) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=LIGHT, linewidth=0.55, zorder=0)


def save_bundle(
    fig: plt.Figure,
    project: Path,
    stem: str,
    source: pd.DataFrame,
) -> dict[str, object]:
    release_layout = (project / "CITATION.cff").exists()
    if release_layout:
        paper_figure_dir = project / "outputs" / "figures"
        output_dir = paper_figure_dir
        source_dir = project / "outputs" / "figure_source_data"
    else:
        paper_figure_dir = project / "paper" / "figures"
        output_dir = project / "researchwrite" / "figure_outputs"
        source_dir = project / "researchwrite" / "figure_source_data"
    paper_figure_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    pdf = paper_figure_dir / f"{stem}.pdf"
    paper_svg = paper_figure_dir / f"{stem}.svg"
    svg = output_dir / f"{stem}.svg"
    png = output_dir / f"{stem}.png"
    tiff = output_dir / f"{stem}.tiff"
    source_path = source_dir / f"{stem}_source_data.csv"

    fixed_date = datetime(2026, 8, 31, tzinfo=timezone.utc)
    fig.savefig(
        pdf,
        bbox_inches="tight",
        pad_inches=0.08,
        metadata={
            "Title": stem,
            "Author": "Hui Shi, Zhuangzhuang Pan, Xiaoyu Meng, and Yan Xia",
            "Creator": "CLDA deterministic figure generator",
            "CreationDate": fixed_date,
            "ModDate": fixed_date,
        },
    )
    fig.savefig(
        paper_svg,
        bbox_inches="tight",
        pad_inches=0.08,
        metadata={
            "Title": stem,
            "Date": "2026-08-31",
            "Creator": "CLDA deterministic figure generator",
        },
    )
    if svg != paper_svg:
        fig.savefig(
            svg,
            bbox_inches="tight",
            pad_inches=0.08,
            metadata={
                "Title": stem,
                "Date": "2026-08-31",
                "Creator": "CLDA deterministic figure generator",
            },
        )
    fig.savefig(png, dpi=400, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(tiff, dpi=600, bbox_inches="tight", pad_inches=0.08)
    source.to_csv(source_path, index=False)
    width, height = fig.get_size_inches()
    plt.close(fig)
    return {
        "stem": stem,
        "paper_pdf": pdf.relative_to(project).as_posix(),
        "paper_svg": paper_svg.relative_to(project).as_posix(),
        "editable_svg": svg.relative_to(project).as_posix(),
        "review_png": png.relative_to(project).as_posix(),
        "submission_tiff": tiff.relative_to(project).as_posix(),
        "source_data": source_path.relative_to(project).as_posix(),
        "size_inches": [float(width), float(height)],
        "pdf_sha256": sha256(pdf),
    }


def load_core(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    spaces = pd.read_csv(root / "data" / "processed" / "space_frame.csv")
    edges = pd.read_csv(root / "data" / "processed" / "dependency_edges.csv")
    candidates = pd.read_csv(root / "data" / "qa" / "candidate_selection_audit.csv")
    unknown = pd.read_csv(root / "data" / "qa" / "unmapped_dependency_candidates.csv")
    return spaces, edges, candidates, unknown


def figure_signal_intersections(root: Path, project: Path) -> dict[str, object]:
    spaces, edges, candidates, unknown = load_core(root)
    main = spaces[boolean(spaces["included_strict"])].copy()
    main_ids = set(main["space_id"].astype(str))
    main_edges = edges[edges["space_id"].astype(str).isin(main_ids)]
    service_ids = set(
        main_edges.loc[main_edges["layer"].eq("inference_service"), "space_id"].astype(str)
    )
    model_ids = set(
        main_edges.loc[main_edges["layer"].eq("model_dependency"), "space_id"].astype(str)
    )
    runtime_ids = set(
        main_edges.loc[main_edges["layer"].eq("local_runtime"), "space_id"].astype(str)
    )
    matched_ids = service_ids & model_ids
    candidate_types = {
        "unmapped_credential",
        "unmapped_api_domain",
        "openai_compatible_provider_unresolved",
    }
    unknown_ids = set(
        unknown.loc[unknown["candidate_type"].isin(candidate_types), "space_id"].astype(str)
    ) & main_ids

    membership_sets = {
        "Service": service_ids,
        "Model": model_ids,
        "Runtime": runtime_ids,
        "Candidate": unknown_ids,
    }
    intersections: list[dict[str, object]] = []
    for project_id in sorted(main_ids):
        state = tuple(project_id in membership_sets[name] for name in membership_sets)
        if not any(state):
            raise ValueError(f"Dependency-observable project {project_id} has no signal set")
        intersections.append(
            {
                "service": state[0],
                "model": state[1],
                "runtime": state[2],
                "candidate": state[3],
            }
        )
    intersections_frame = (
        pd.DataFrame(intersections)
        .value_counts(sort=True)
        .rename("projects")
        .reset_index()
        .sort_values(
            ["projects", "service", "model", "runtime", "candidate"],
            ascending=[False, False, False, False, False],
        )
        .reset_index(drop=True)
    )
    intersections_frame["matched"] = (
        intersections_frame["service"] & intersections_frame["model"]
    )
    if int(intersections_frame.loc[intersections_frame["matched"], "projects"].sum()) != len(
        matched_ids
    ):
        raise ValueError("Intersection counts do not recover the matched set")

    source = intersections_frame.assign(
        category="signal intersection", value=lambda x: x["projects"]
    )[
        [
            "category",
            "value",
            "service",
            "model",
            "runtime",
            "candidate",
            "matched",
        ]
    ]

    fig = plt.figure(figsize=(6.5, 3.25))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.38, 0.72], hspace=0.04)
    ax_bar = fig.add_subplot(grid[0])
    x = np.arange(len(intersections_frame))
    bar_colors = np.where(intersections_frame["matched"], BLUE, "#B9C0C5")
    bars = ax_bar.bar(x, intersections_frame["projects"], color=bar_colors, width=0.72, zorder=2)
    for bar, count in zip(bars, intersections_frame["projects"], strict=True):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 4,
            str(int(count)),
            ha="center",
            va="bottom",
            fontsize=8.0,
        )
    ax_bar.set_xlim(-0.75, len(intersections_frame) - 0.25)
    ax_bar.set_ylim(0, float(intersections_frame["projects"].max()) * 1.18)
    ax_bar.set_ylabel("Projects")
    clean_axes(ax_bar, grid_axis="y")
    ax_bar.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_bar.legend(
        handles=[
            Patch(facecolor=BLUE, edgecolor="none", label=f"Service + model matched (n={len(matched_ids)})"),
            Patch(facecolor="#B9C0C5", edgecolor="none", label="Other signal intersections"),
        ],
        loc="upper right",
        frameon=False,
        ncol=2,
        borderaxespad=0.1,
        columnspacing=1.2,
        handlelength=1.3,
        handletextpad=0.5,
    )

    ax_matrix = fig.add_subplot(grid[1], sharex=ax_bar)
    set_names = list(membership_sets)
    set_sizes = [len(membership_sets[name]) for name in set_names]
    row_keys = [name.casefold() for name in set_names]
    y_positions = np.arange(len(set_names))[::-1]
    for y in y_positions:
        ax_matrix.axhline(y, color="#EDF0F2", linewidth=0.55, zorder=0)
    for col, row in intersections_frame.iterrows():
        active = [bool(row[key]) for key in row_keys]
        active_y = [y_positions[i] for i, value in enumerate(active) if value]
        if len(active_y) > 1:
            ax_matrix.plot([col, col], [min(active_y), max(active_y)], color=INK, linewidth=1.05, zorder=1)
        for row_index, y in enumerate(y_positions):
            is_active = active[row_index]
            color = BLUE if is_active and bool(row["matched"]) else INK if is_active else "#D9DEE2"
            size = 25 if is_active else 15
            ax_matrix.scatter(col, y, s=size, color=color, edgecolor="none", zorder=2)
    ax_matrix.set_yticks(
        y_positions,
        [f"{name}  (n={size})" for name, size in zip(set_names, set_sizes, strict=True)],
    )
    ax_matrix.set_ylim(-0.55, len(set_names) - 0.45)
    ax_matrix.set_xlabel("Observed signal intersections, ordered by project count")
    ax_matrix.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_matrix.tick_params(axis="y", length=0, pad=6)
    for spine in ax_matrix.spines.values():
        spine.set_visible(False)

    fig.subplots_adjust(left=0.20, right=0.985, top=0.965, bottom=0.10)
    return save_bundle(fig, project, "figure2_signal_intersections", source)


def figures_cross_layer(root: Path, project: Path) -> list[dict[str, object]]:
    """Separate descriptive composition from inferential cross-layer evidence."""

    joint = pd.read_csv(root / "analysis_results" / "cross_layer_codeclaration.csv")
    diversity = pd.read_csv(
        root / "analysis_results" / "cross_layer_service_model_diversity.csv"
    ).sort_values("service_fractional_share", ascending=False)
    null = pd.read_csv(root / "analysis_results" / "cross_layer_codeclaration_null.csv")
    summary = json.loads(
        (root / "analysis_results" / "cross_layer_codeclaration_summary.json").read_text(
            encoding="utf-8"
        )
    )
    paired = json.loads(
        (root / "analysis_results" / "paired_layer_comparison.json").read_text(
            encoding="utf-8"
        )
    )

    # Aggregate only for display. All reported statistics retain the complete
    # nine-service by 26-model-category matrices.
    top_services = diversity.head(4)["service_provider"].astype(str).tolist()
    global_models = (
        joint.groupby("model_publisher_or_namespace", as_index=False)[
            "joint_fractional_share"
        ]
        .sum()
        .sort_values("joint_fractional_share", ascending=False)
    )
    top_models = global_models.head(6)["model_publisher_or_namespace"].astype(str).tolist()
    service_other = f"Other services ({joint['service_provider'].nunique() - len(top_services)})"
    model_other = (
        f"Other model categories "
        f"({joint['model_publisher_or_namespace'].nunique() - len(top_models)})"
    )
    service_order = top_services + [service_other]
    model_order = top_models + [model_other]
    grouped = joint.assign(
        service_group=lambda frame: frame["service_provider"].where(
            frame["service_provider"].isin(top_services), service_other
        ),
        model_group=lambda frame: frame["model_publisher_or_namespace"].where(
            frame["model_publisher_or_namespace"].isin(top_models), model_other
        ),
    )
    mosaic = (
        grouped.groupby(["service_group", "model_group"], as_index=False)[
            "joint_fractional_share"
        ]
        .sum()
        .set_index(["service_group", "model_group"])
        .reindex(
            pd.MultiIndex.from_product(
                [service_order, model_order], names=["service_group", "model_group"]
            ),
            fill_value=0.0,
        )
        .reset_index()
    )
    service_marginals = (
        mosaic.groupby("service_group")["joint_fractional_share"]
        .sum()
        .reindex(service_order)
    )
    mosaic["service_fractional_share"] = mosaic["service_group"].map(service_marginals)
    mosaic["model_share_within_service"] = (
        mosaic["joint_fractional_share"] / mosaic["service_fractional_share"]
    )

    hhi = pd.DataFrame(
        [
            {
                "layer": "Declared services",
                "hhi": paired["service_hhi"],
                "low": paired["service_hhi_bootstrap_ci"][0],
                "high": paired["service_hhi_bootstrap_ci"][1],
                "top_share": paired["service_top_share"],
                "shannon_effective_categories": paired[
                    "service_shannon_effective_categories"
                ],
            },
            {
                "layer": "Referenced models",
                "hhi": paired["model_hhi"],
                "low": paired["model_hhi_bootstrap_ci"][0],
                "high": paired["model_hhi_bootstrap_ci"][1],
                "top_share": paired["model_top_share"],
                "shannon_effective_categories": paired[
                    "model_shannon_effective_categories"
                ],
            },
        ]
    )
    composition_source = mosaic.assign(panel="composition")
    association_source = pd.concat(
        [hhi.assign(panel="a"), null.assign(panel="b")],
        ignore_index=True,
        sort=False,
    )

    fig = plt.figure(figsize=(6.5, 3.15))
    ax = fig.add_subplot(111)
    model_colors = ["#4477AA", "#66CCEE", "#228833", "#CCBB44", "#EE6677", "#AA3377", "#B7B7B7"]
    model_hatches = ["", "///", "\\\\", "...", "xx", "--", "++"]
    short_model_labels = {
        "namespace:sentence-transformers": "sentence-transformers (namespace)",
        model_other: model_other,
    }
    y_positions = np.arange(len(service_order))[::-1]
    for service_index, service in enumerate(service_order):
        service_share = float(service_marginals.loc[service])
        x_left = 0.0
        for model_index, model in enumerate(model_order):
            joint_share = float(
                mosaic.loc[
                    mosaic["service_group"].eq(service)
                    & mosaic["model_group"].eq(model),
                    "joint_fractional_share",
                ].iloc[0]
            )
            if joint_share <= 0:
                continue
            within_share = joint_share / service_share
            ax.barh(
                y_positions[service_index],
                within_share,
                left=x_left,
                height=0.62,
                color=model_colors[model_index],
                edgecolor="white",
                linewidth=0.8,
                hatch=model_hatches[model_index],
                zorder=2,
            )
            if within_share >= 0.065:
                luminance = matplotlib.colors.to_rgb(model_colors[model_index])
                perceived = 0.299 * luminance[0] + 0.587 * luminance[1] + 0.114 * luminance[2]
                text_color = "white" if perceived < 0.52 else INK
                ax.text(
                    x_left + within_share / 2,
                    y_positions[service_index],
                    f"{100 * within_share:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=9.0,
                    fontweight="bold",
                    color=text_color,
                    zorder=3,
                )
            x_left += within_share
    service_labels = [
        ("Other" if value == service_other else value)
        + f"\n{100 * service_marginals.loc[value]:.1f}%"
        for value in service_order
    ]
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.55, len(service_order) - 0.45)
    ax.set_xticks([0, 0.25, 0.50, 0.75, 1.0])
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_yticks(y_positions, service_labels)
    ax.set_xlabel("Model composition within service")
    ax.tick_params(axis="y", length=0, pad=6)
    clean_axes(ax, grid_axis="x")
    model_handles = [
        Patch(
            facecolor=model_colors[index],
            edgecolor=INK,
            linewidth=0.4,
            hatch=model_hatches[index],
            label=short_model_labels.get(model, model),
        )
        for index, model in enumerate(model_order)
    ]
    ax.legend(
        handles=model_handles,
        loc="lower center",
        bbox_to_anchor=(0.54, 1.01),
        ncol=4,
        frameon=False,
        columnspacing=0.9,
        handlelength=1.25,
        handletextpad=0.45,
        fontsize=8.1,
    )
    fig.subplots_adjust(left=0.16, right=0.985, top=0.78, bottom=0.17)
    composition_record = save_bundle(
        fig,
        project,
        "figure3_service_model_composition",
        composition_source,
    )

    fig = plt.figure(figsize=(6.5, 2.52))
    grid = fig.add_gridspec(1, 2, wspace=0.34)
    ax = fig.add_subplot(grid[0, 0])
    layer_colors = [BLUE, ORANGE]
    layer_markers = ["o", "s"]
    y = np.array([1.55, 1.02])
    for index, row in hhi.iterrows():
        ax.errorbar(
            row["hhi"],
            y[index],
            xerr=np.array([[row["hhi"] - row["low"]], [row["high"] - row["hhi"]]]),
            fmt=layer_markers[index],
            color=layer_colors[index],
            ecolor=layer_colors[index],
            elinewidth=1.25,
            capsize=2.5,
            markersize=5.2,
            zorder=3,
        )
        ax.text(row["high"] + 0.012, y[index], f"{row['hhi']:.3f}", va="center", fontsize=8.2)
    ax.set_yticks(y, ["Services", "Models"])
    ax.set_ylim(-0.55, 2.00)
    ax.set_xlim(0, 0.50)
    ax.set_xlabel("HHI (higher = more concentrated)")
    panel_title(ax, "a", "Paired concentration")
    clean_axes(ax, grid_axis="x")
    ax.tick_params(axis="y", length=0)
    ax.add_patch(
        Rectangle(
            (0.0, 0.0),
            1.0,
            0.38,
            transform=ax.transAxes,
            facecolor="white",
            edgecolor="none",
            zorder=2,
        )
    )
    ax.plot([0, 1], [0.36, 0.36], color=LIGHT, linewidth=0.65, transform=ax.transAxes, zorder=3)
    ax.text(0.76, 0.29, "Services", transform=ax.transAxes, fontweight="bold", color=BLUE, ha="right", va="center", fontsize=8.0, zorder=4)
    ax.text(1.00, 0.29, "Models", transform=ax.transAxes, fontweight="bold", color=ORANGE, ha="right", va="center", fontsize=8.0, zorder=4)
    metric_rows = [
        ("Top category share", paired["service_top_share"], paired["model_top_share"], "percent"),
        (
            "Shannon effective count",
            paired["service_shannon_effective_categories"],
            paired["model_shannon_effective_categories"],
            "number",
        ),
    ]
    for row_index, (label, service_value, model_value, kind) in enumerate(metric_rows):
        y_value = 0.18 - row_index * 0.12
        ax.text(0.00, y_value, label, transform=ax.transAxes, ha="left", va="center", fontsize=8.1, zorder=4)
        if kind == "percent":
            service_text = f"{100 * service_value:.1f}%"
            model_text = f"{100 * model_value:.1f}%"
        else:
            service_text = f"{service_value:.1f}"
            model_text = f"{model_value:.1f}"
        ax.text(0.76, y_value, service_text, transform=ax.transAxes, ha="right", va="center", fontsize=8.0, zorder=4)
        ax.text(1.00, y_value, model_text, transform=ax.transAxes, ha="right", va="center", fontsize=8.0, zorder=4)

    ax = fig.add_subplot(grid[0, 1])
    null_values = null["mutual_information"].astype(float)
    ax.hist(null_values, bins=36, color=BLUE_LIGHT, edgecolor="white", linewidth=0.35)
    observed = float(summary["mutual_information"])
    low, high = [float(value) for value in summary["permutation_null_interval"]]
    ax.axvspan(low, high, color=LIGHT, alpha=0.55, zorder=0)
    ax.axvline(observed, color=RED, linewidth=1.8, zorder=3)
    ax.text(
        observed - 0.008,
        0.88,
        f"Observed {observed:.3f}\n$p<.001$",
        transform=ax.get_xaxis_transform(),
        ha="right",
        va="top",
        color=RED,
        fontweight="bold",
        fontsize=8.7,
    )
    ax.text(
        (low + high) / 2,
        0.96,
        "95% null interval",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=8.1,
    )
    ax.set_xlabel("Mutual information (nats)")
    ax.set_ylabel("Permutations")
    panel_title(ax, "b", "Association within projects")
    clean_axes(ax, grid_axis="y")
    fig.subplots_adjust(left=0.11, right=0.985, top=0.87, bottom=0.17)
    association_record = save_bundle(
        fig,
        project,
        "figure4_cross_layer_association",
        association_source,
    )
    return [composition_record, association_record]


def _single_variant(
    frame: pd.DataFrame,
    family: str,
    variant: str,
    group: str,
    label: str,
) -> dict[str, object]:
    row = frame[
        frame["robustness_family"].eq(family)
        & frame["variant"].astype(str).eq(variant)
    ]
    if len(row) != 1:
        raise ValueError(f"Expected one robustness row for {family}/{variant}, found {len(row)}")
    item = row.iloc[0]
    return {
        "group": group,
        "label": label,
        "estimate": float(item["hhi_difference_service_minus_model"]),
        "low": float(item["hhi_difference_ci_low"]),
        "high": float(item["hhi_difference_ci_high"]),
        "matched_n": str(int(item["matched_projects"])),
        "interval_kind": "paired percentile interval",
    }


def _variant_range(
    frame: pd.DataFrame,
    family: str,
    group: str,
    label: str,
) -> dict[str, object]:
    subset = frame[frame["robustness_family"].eq(family)]
    if subset.empty:
        raise ValueError(f"No robustness rows found for {family}")
    values = subset["hhi_difference_service_minus_model"].astype(float)
    ns = subset["matched_projects"].astype(int)
    n_label = str(int(ns.min())) if int(ns.min()) == int(ns.max()) else f"{ns.min()}–{ns.max()}"
    return {
        "group": group,
        "label": label,
        "estimate": float(values.median()),
        "low": float(values.min()),
        "high": float(values.max()),
        "matched_n": n_label,
        "interval_kind": "range across variants",
    }


def figure_robustness(root: Path, project: Path) -> dict[str, object]:
    matrix = pd.read_csv(root / "analysis_results" / "matched_robustness.csv")
    unknown = pd.read_csv(root / "analysis_results" / "unknown_service_boundary.csv")
    provider_omission = pd.read_csv(
        root / "analysis_results" / "service_provider_omission.csv"
    )
    file_caps = pd.read_csv(root / "analysis_results" / "file_cap_sensitivity.csv")
    rows: list[dict[str, object]] = []
    rows.append(_single_variant(matrix, "primary", "project_equal", "Primary", "Equal project weights"))
    rows.extend(
        [
            _variant_range(matrix, "rank_cutoff", "Search design", "Rank cutoffs 25–100"),
            _variant_range(matrix, "leave_one_query_out", "Search design", "Omit each query in turn"),
            _variant_range(matrix, "search_arm", "Search design", "Likes vs newest"),
        ]
    )
    rows.extend(
        [
            _single_variant(matrix, "evidence", "high_confidence_both_layers", "Evidence", "High-confidence edges"),
            _single_variant(matrix, "evidence", "hub_linked_model_only", "Evidence", "Official model links only"),
        ]
    )
    cap20 = file_caps[file_caps["max_source_files"].eq(20)]
    if len(cap20) != 1:
        raise ValueError("Expected one 20-file source-cap row")
    cap20_row = cap20.iloc[0]
    rows.append(
        {
            "group": "Evidence",
            "label": "Source file cap 20",
            "estimate": float(cap20_row["hhi_difference_service_minus_model"]),
            "low": float(cap20_row["hhi_difference_ci_low"]),
            "high": float(cap20_row["hhi_difference_ci_high"]),
            "matched_n": str(int(cap20_row["matched_projects"])),
            "interval_kind": "paired percentile interval",
        }
    )
    rows.extend(
        [
            _single_variant(matrix, "source_reuse", "exact_multifile_cluster", "Weight and reuse", "Exact source clusters"),
            _variant_range(matrix[matrix["variant"].astype(str).str.startswith("near_duplicate")], "source_reuse", "Weight and reuse", "Jaccard 0.85–0.95"),
            _single_variant(matrix, "source_reuse", "author_cluster", "Weight and reuse", "Author clusters"),
        ]
    )
    rows.extend(
        [
            _single_variant(matrix, "model_taxonomy", "immediate_public_namespace", "Category mapping", "Immediate public namespace"),
            _single_variant(matrix, "model_taxonomy", "pooled_unmapped_namespaces", "Category mapping", "Unmapped namespaces pooled"),
            _single_variant(matrix, "model_taxonomy", "mapped_publishers_only", "Category mapping", "Mapped publishers only"),
        ]
    )
    for treatment, label in [
        ("machine_identifier_categories", "Unknowns: type and identifier"),
        ("project_unique_unknown_bound", "Unknowns: one category per project"),
    ]:
        selected = unknown[unknown["unknown_treatment"].eq(treatment)]
        if len(selected) != 1:
            raise ValueError(f"Expected one unknown-service row for {treatment}")
        item = selected.iloc[0]
        rows.append(
            {
                "group": "Service boundary",
                "label": label,
                "estimate": float(item["hhi_difference_service_minus_model"]),
                "low": float(item["hhi_difference_ci_low"]),
                "high": float(item["hhi_difference_ci_high"]),
                "matched_n": str(int(item["matched_projects"])),
                "interval_kind": "paired percentile interval",
            }
        )
    omitted_hf = provider_omission[
        provider_omission["variant"].astype(str).eq("Hugging Face")
    ]
    if len(omitted_hf) != 1:
        raise ValueError("Expected one Hugging Face omission row")
    item = omitted_hf.iloc[0]
    rows.append(
        {
            "group": "Dependence on leading service",
            "label": "Exclude projects declaring Hugging Face",
            "estimate": float(item["hhi_difference_service_minus_model"]),
            "low": float(item["hhi_difference_ci_low"]),
            "high": float(item["hhi_difference_ci_high"]),
            "matched_n": str(int(item["matched_projects"])),
            "interval_kind": "paired percentile interval",
        }
    )
    source = pd.DataFrame(rows)
    if (source["estimate"] <= 0).any():
        failed = source.loc[source["estimate"] <= 0, "label"].tolist()
        raise ValueError(f"Non-positive robustness estimates require manuscript review: {failed}")

    group_order = list(dict.fromkeys(source["group"]))
    y_values: list[float] = []
    group_bounds: dict[str, tuple[float, float]] = {}
    cursor = 0.0
    for group in group_order:
        indices = source.index[source["group"].eq(group)].tolist()
        start = cursor
        for _ in indices:
            y_values.append(cursor)
            cursor += 1.0
        group_bounds[group] = (start - 0.45, cursor - 0.55)
        cursor += 0.62
    source["y"] = y_values

    fig, ax = plt.subplots(figsize=(6.5, 5.1))
    group_colors = {
        "Primary": INK,
        "Search design": BLUE,
        "Evidence": TEAL,
        "Weight and reuse": VIOLET,
        "Category mapping": GOLD,
        "Service boundary": ORANGE,
        "Dependence on leading service": RED,
    }
    for index, group in enumerate(group_order):
        low_y, high_y = group_bounds[group]
        if index % 2:
            ax.axhspan(low_y, high_y, color=PALE, zorder=0)
        ax.text(
            0.005,
            low_y + 0.05,
            group,
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color=group_colors[group],
        )

    for _, row in source.iterrows():
        color = group_colors[row["group"]]
        if row["interval_kind"] == "range across variants":
            ax.hlines(row["y"], row["low"], row["high"], color=color, linewidth=1.35, zorder=3)
            ax.vlines(
                [row["low"], row["high"]],
                row["y"] - 0.10,
                row["y"] + 0.10,
                color=color,
                linewidth=1.0,
                zorder=3,
            )
        else:
            ax.errorbar(
                row["estimate"],
                row["y"],
                xerr=np.array([[row["estimate"] - row["low"]], [row["high"] - row["estimate"]]]),
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=1.1,
                capsize=2.5,
                markersize=4.5,
                markeredgecolor="white",
                markeredgewidth=0.45,
                zorder=3,
            )
        ax.text(
            0.99,
            row["y"],
            f"n={row['matched_n']}",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=8.1,
            color=INK,
        )
    ax.set_yticks(source["y"], source["label"])
    ax.set_ylim(source["y"].max() + 0.65, -0.8)
    upper = max(0.52, float(source["high"].max()) + 0.04)
    ax.set_xlim(0, upper)
    ax.set_xlabel("Service HHI − model HHI")
    clean_axes(ax, grid_axis="x")
    ax.tick_params(axis="y", length=0, pad=5)
    legend = [
        Line2D([0], [0], marker="o", color=INK, linewidth=1.1, markersize=4.5,
               label="Estimate with paired 95% interval"),
        Line2D([0], [0], color=INK, linewidth=1.35,
               label="Range across related variants"),
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.67, 0.985),
        frameon=False,
        ncol=2,
        columnspacing=1.2,
        handletextpad=0.6,
    )
    fig.subplots_adjust(left=0.36, right=0.98, top=0.90, bottom=0.10)
    return save_bundle(fig, project, "figure5_matched_robustness", source)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Pilot directory containing analysis_results and data",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    project = root if (root / "CITATION.cff").exists() else root.parent
    release_layout = (project / "CITATION.cff").exists()
    figure_dir = project / "outputs" / "figures" if release_layout else project / "paper" / "figures"
    source_dir = (
        project / "outputs" / "figure_source_data"
        if release_layout
        else project / "researchwrite" / "figure_source_data"
    )
    records: list[dict[str, object]] = []
    manual_svg = figure_dir / "figure1_study_design.svg"
    manual_pdf = figure_dir / "figure1_study_design.pdf"
    manual_source = source_dir / "figure1_study_design_source_data.csv"
    if manual_svg.exists() and manual_pdf.exists():
        records.append(
            {
                "stem": "figure1_study_design",
                "drawing_source": "direct SVG",
                "paper_pdf": manual_pdf.relative_to(project).as_posix(),
                "paper_svg": manual_svg.relative_to(project).as_posix(),
                "source_data": (
                    manual_source.relative_to(project).as_posix()
                    if manual_source.exists()
                    else None
                ),
                "size_inches": [7.0, 3.0333],
                "pdf_sha256": sha256(manual_pdf),
            }
        )
    records.append(figure_signal_intersections(root, project))
    records.extend(figures_cross_layer(root, project))
    records.append(figure_robustness(root, project))
    manifest = {
        "generator": "direct SVG schematic plus pilot/src/make_figures.py",
        "figure_count": len(records),
        "style_reference": "01-ai-crawler-governance final figures",
        "font_floor_points": 8.0,
        "manual_labels": 0,
        "records": records,
    }
    qa_path = (
        project / "outputs" / "figure_qa.json"
        if (project / "CITATION.cff").exists()
        else project / "researchwrite" / "qa_logs" / "figure_qa.json"
    )
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    qa_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
