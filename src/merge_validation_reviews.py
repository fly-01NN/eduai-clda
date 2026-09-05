"""Merge blinded reviewer files and third-reviewer adjudications by sample ID."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


VALID_LABELS = {"yes", "no", "uncertain"}


def normalized(value: object) -> str:
    return str(value or "").strip().casefold()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, encoding="utf-8-sig")


def assert_labels(frame: pd.DataFrame, column: str, path: Path) -> None:
    invalid = sorted(set(frame[column].map(normalized)) - VALID_LABELS)
    if invalid:
        raise ValueError(f"Invalid labels in {path}: {invalid}")
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate sample IDs in {path}")


def unresolved_ids(
    first: pd.DataFrame, second: pd.DataFrame, label: str
) -> set[str]:
    paired = first[["sample_id", label]].merge(
        second[["sample_id", label]],
        on="sample_id",
        suffixes=("_first", "_second"),
        validate="one_to_one",
    )
    left = paired[f"{label}_first"].map(normalized)
    right = paired[f"{label}_second"].map(normalized)
    resolved = left.eq(right) & left.isin({"yes", "no"})
    return set(paired.loc[~resolved, "sample_id"].astype(str))


def merge_task(
    *,
    master_path: Path,
    reviewer_a_path: Path,
    reviewer_b_path: Path,
    reviewer_c_path: Path,
    label: str,
    mappings: dict[str, tuple[str, str, str]],
) -> int:
    master = read_csv(master_path)
    reviewer_a = read_csv(reviewer_a_path)
    reviewer_b = read_csv(reviewer_b_path)
    reviewer_c = read_csv(reviewer_c_path)

    for frame, path in (
        (reviewer_a, reviewer_a_path),
        (reviewer_b, reviewer_b_path),
        (reviewer_c, reviewer_c_path),
    ):
        assert_labels(frame, label, path)
    expected_ids = set(master["sample_id"].astype(str))
    if set(reviewer_a["sample_id"].astype(str)) != expected_ids:
        raise ValueError(f"Reviewer A sample IDs do not match {master_path.name}")
    if set(reviewer_b["sample_id"].astype(str)) != expected_ids:
        raise ValueError(f"Reviewer B sample IDs do not match {master_path.name}")

    unresolved = unresolved_ids(reviewer_a, reviewer_b, label)
    if set(reviewer_c["sample_id"].astype(str)) != unresolved:
        missing = sorted(unresolved - set(reviewer_c["sample_id"].astype(str)))
        extra = sorted(set(reviewer_c["sample_id"].astype(str)) - unresolved)
        raise ValueError(f"Third-review coverage mismatch; missing={missing}, extra={extra}")
    if not reviewer_c[label].map(normalized).isin({"yes", "no"}).all():
        pending = reviewer_c.loc[
            ~reviewer_c[label].map(normalized).isin({"yes", "no"}), "sample_id"
        ].tolist()
        raise ValueError(f"Third-review labels remain unresolved: {pending}")

    indexed = {
        "a": reviewer_a.set_index("sample_id"),
        "b": reviewer_b.set_index("sample_id"),
        "c": reviewer_c.set_index("sample_id"),
    }
    for source_column, (first_column, second_column, final_column) in mappings.items():
        master[first_column] = master["sample_id"].map(indexed["a"][source_column])
        master[second_column] = master["sample_id"].map(indexed["b"][source_column])
        master[final_column] = master["sample_id"].map(indexed["c"][source_column]).fillna("")

    master.to_csv(master_path, index=False)
    return len(unresolved)


def merge(root: Path) -> dict[str, int]:
    private = root / "data" / "validation_private"
    packages = private / "reviewer_packages"
    a = packages / "reviewer_A"
    b = packages / "reviewer_B"
    c = packages / "reviewer_C"

    education = merge_task(
        master_path=private / "education_relevance_blinded.csv",
        reviewer_a_path=a / "01_education_relevance.csv",
        reviewer_b_path=b / "01_education_relevance.csv",
        reviewer_c_path=c / "01_education_relevance.csv",
        label="education_relevant",
        mappings={
            "education_relevant": (
                "reviewer1_education_relevant",
                "reviewer2_education_relevant",
                "adjudicated_education_relevant",
            ),
            "notes": ("reviewer1_notes", "reviewer2_notes", "adjudication_notes"),
        },
    )
    dependency = merge_task(
        master_path=private / "dependency_evidence_blinded.csv",
        reviewer_a_path=a / "02_dependency_evidence.csv",
        reviewer_b_path=b / "02_dependency_evidence.csv",
        reviewer_c_path=c / "02_dependency_evidence.csv",
        label="dependency_supported",
        mappings={
            "dependency_supported": (
                "reviewer1_dependency_supported",
                "reviewer2_dependency_supported",
                "adjudicated_dependency_supported",
            ),
            "layer": ("reviewer1_layer", "reviewer2_layer", "adjudicated_layer"),
            "provider_or_model": (
                "reviewer1_provider_or_model",
                "reviewer2_provider_or_model",
                "adjudicated_provider_or_model",
            ),
            "notes": ("reviewer1_notes", "reviewer2_notes", "adjudication_notes"),
        },
    )
    service = merge_task(
        master_path=private / "service_negative_blinded.csv",
        reviewer_a_path=a / "03_service_false_negative_audit.csv",
        reviewer_b_path=b / "03_service_false_negative_audit.csv",
        reviewer_c_path=c / "03_service_false_negative_audit.csv",
        label="named_service_present",
        mappings={
            "named_service_present": (
                "reviewer1_named_service_present",
                "reviewer2_named_service_present",
                "adjudicated_named_service_present",
            ),
            "provider_if_present": (
                "reviewer1_provider",
                "reviewer2_provider",
                "adjudicated_provider",
            ),
            "notes": ("reviewer1_notes", "reviewer2_notes", "adjudication_notes"),
        },
    )
    return {
        "education_adjudicated": education,
        "dependency_adjudicated": dependency,
        "service_negative_adjudicated": service,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    for name, count in merge(args.root.resolve()).items():
        print(f"{name}: {count}")


if __name__ == "__main__":
    main()
