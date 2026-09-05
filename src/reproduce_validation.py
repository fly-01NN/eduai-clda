"""Reproduce validation estimates from the released de-identified labels."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from analyze_validation_labels import (
    cohen_kappa,
    education_metrics,
    missed_service,
    positive_precision,
    stratified_bootstrap,
)


def as_boolean(series: pd.Series) -> pd.Series:
    if str(series.dtype) in {"bool", "boolean"}:
        return series.fillna(False).astype(bool)
    return series.fillna("").astype(str).str.casefold().eq("true")


def reproduce(root: Path) -> dict[str, object]:
    results = root / "analysis_results"
    labels = pd.read_csv(
        results / "measurement_validation_deidentified.csv",
        keep_default_na=False,
    )
    labels["resolved"] = labels["resolved_label"].astype(str).str.casefold()

    education = labels[labels["task"].eq("education_relevance")].copy()
    dependency = labels[labels["task"].eq("dependency_evidence")].copy()
    service = labels[labels["task"].eq("service_negative")].copy()
    education["education_strict_match"] = as_boolean(
        education["automatic_positive"]
    )

    education_result = education_metrics(education)
    education_result["bootstrap_intervals"] = stratified_bootstrap(
        education, education_metrics
    )
    dependency_result = positive_precision(dependency)
    dependency_result["bootstrap_intervals"] = stratified_bootstrap(
        dependency, positive_precision
    )
    service_result = missed_service(service)
    service_result["bootstrap_intervals"] = stratified_bootstrap(
        service, missed_service
    )
    service_result["missed_service_records"] = int(
        service["resolved"].eq("yes").sum()
    )
    service_result["all_missed_service_records_in_candidate_boundary"] = bool(
        as_boolean(service.loc[service["resolved"].eq("yes"), "candidate_boundary"])
        .all()
    )

    agreements = {
        task: cohen_kappa(group["reviewer1_label"], group["reviewer2_label"])
        for task, group in labels.groupby("task")
    }
    computed = {
        "education_rule": education_result,
        "dependency_edge_rule": dependency_result,
        "service_negative_audit": service_result,
        "initial_interreviewer_agreement": agreements,
    }

    reference = json.loads(
        (results / "measurement_validation.json").read_text(encoding="utf-8")
    )
    checks = {
        "education_precision": math.isclose(
            float(education_result["precision"]),
            float(reference["education_rule"]["precision"]),
            abs_tol=1e-12,
        ),
        "dependency_precision": math.isclose(
            float(dependency_result["precision_overall"]),
            float(reference["dependency_edge_rule"]["precision_overall"]),
            abs_tol=1e-12,
        ),
        "missed_service_share": math.isclose(
            float(service_result["missed_service_share"]),
            float(reference["service_negative_audit"]["missed_service_share"]),
            abs_tol=1e-12,
        ),
        "bootstrap_intervals": (
            education_result["bootstrap_intervals"]
            == reference["education_rule"]["bootstrap_intervals"]
            and dependency_result["bootstrap_intervals"]
            == reference["dependency_edge_rule"]["bootstrap_intervals"]
            and service_result["bootstrap_intervals"]
            == reference["service_negative_audit"]["bootstrap_intervals"]
        ),
        "missed_service_count": (
            service_result["missed_service_records"]
            == reference["service_negative_audit"]["missed_service_records"]
        ),
        "candidate_boundary": service_result[
            "all_missed_service_records_in_candidate_boundary"
        ],
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "computed": computed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    report = reproduce(args.root.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise RuntimeError("released validation estimates did not reproduce")


if __name__ == "__main__":
    main()
