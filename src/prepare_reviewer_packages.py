"""Prepare two answer-free, independently ordered human-validation packages."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import zipfile

import pandas as pd


PACKAGE_SEED = 20260903


TASKS = {
    "01_education_relevance.csv": {
        "source": "education_relevance_blinded.csv",
        "evidence": ["sample_id", "title", "project_url", "readme_excerpt"],
        "labels": ["education_relevant", "notes"],
    },
    "02_dependency_evidence.csv": {
        "source": "dependency_evidence_blinded.csv",
        "evidence": [
            "sample_id",
            "project_url",
            "source_file",
            "source_url",
            "source_excerpt",
        ],
        "labels": ["dependency_supported", "layer", "provider_or_model", "notes"],
    },
    "03_service_false_negative_audit.csv": {
        "source": "service_negative_blinded.csv",
        "evidence": ["sample_id", "title", "project_url", "selected_source_urls"],
        "labels": ["named_service_present", "provider_if_present", "notes"],
    },
}


README = "请在每个 CSV 的空白列填写：主要判断填 yes、no 或 uncertain；若填 yes，再填写相应的层级或服务商/模型名称，必要时可在 notes 备注，完成后将三个 CSV 原样发回即可。\n"


def stable_order(sample_id: str, *, coder: str, task: str) -> str:
    payload = f"{PACKAGE_SEED}|{coder}|{task}|{sample_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_zip(source_dir: Path, destination: Path) -> None:
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(source_dir.iterdir()):
            if not path.is_file():
                continue
            info = zipfile.ZipInfo(path.name, date_time=(2026, 9, 3, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())


def prepare(root: Path) -> list[Path]:
    private = root / "data" / "validation_private"
    output = private / "reviewer_packages"
    output.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    for coder in ("A", "B"):
        coder_dir = output / f"reviewer_{coder}"
        coder_dir.mkdir(parents=True, exist_ok=True)
        for old_file in coder_dir.iterdir():
            if old_file.is_file():
                old_file.unlink()
        (coder_dir / "填写说明.txt").write_text(README, encoding="utf-8")

        for filename, specification in TASKS.items():
            source = pd.read_csv(
                private / specification["source"], keep_default_na=False
            )
            frame = source[specification["evidence"]].copy()
            for column in specification["labels"]:
                frame[column] = ""
            frame["_order"] = frame["sample_id"].map(
                lambda value: stable_order(
                    str(value), coder=coder, task=filename
                )
            )
            frame = frame.sort_values("_order").drop(columns="_order")
            frame.to_csv(coder_dir / filename, index=False, encoding="utf-8-sig")

        zip_path = output / f"reviewer_{coder}_validation_pack.zip"
        write_zip(coder_dir, zip_path)
        generated.append(zip_path)
    return generated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    for path in prepare(args.root.resolve()):
        print(path)


if __name__ == "__main__":
    main()
