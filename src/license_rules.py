"""Conservative license-disclosure and rights-asymmetry rules.

These rules identify documentation conditions that merit review. They do not
make legal compatibility determinations.
"""

from __future__ import annotations

import re
from typing import Iterable


PERMISSIVE = {
    "apache-2.0", "mit", "bsd", "bsd-2-clause", "bsd-3-clause", "isc",
    "cc-by-4.0", "cc-by-3.0", "cc0-1.0", "unlicense", "wtfpl",
}
COPYLEFT = {
    "gpl", "gpl-2.0", "gpl-3.0", "agpl-3.0", "lgpl", "lgpl-2.1",
    "lgpl-3.0", "mpl-2.0", "cc-by-sa-4.0",
}
RESTRICTED = {
    "cc-by-nc-4.0", "cc-by-nc-sa-4.0", "cc-by-nd-4.0", "cc-by-nc-nd-4.0",
    "research-only", "non-commercial", "noncommercial", "proprietary",
}
MODEL_SPECIFIC_PREFIXES = (
    "llama", "gemma", "qwen", "openrail", "bigscience-openrail",
    "creativeml-openrail", "deepseek", "cohere", "nvidia-open-model",
)


def normalize_license(value: object) -> str:
    if value is None:
        return "missing"
    if isinstance(value, (list, tuple, set)):
        parts = [normalize_license(item) for item in value]
        parts = [part for part in parts if part != "missing"]
        return "+".join(sorted(set(parts))) if parts else "missing"
    text = str(value).strip().casefold()
    if not text or text in {"none", "null", "unknown", "other"}:
        return "missing" if text != "other" else "other"
    text = text.replace("license:", "").replace("_", "-").replace(" ", "-")
    text = re.sub(r"-+", "-", text)
    aliases = {
        "apache2": "apache-2.0",
        "apache-2": "apache-2.0",
        "apache-license-2.0": "apache-2.0",
        "mit-license": "mit",
        "gpl3": "gpl-3.0",
        "agpl3": "agpl-3.0",
    }
    return aliases.get(text, text)


def license_class(value: object) -> str:
    normalized = normalize_license(value)
    parts = normalized.split("+")
    classes = {_single_license_class(part) for part in parts}
    precedence = ("restricted", "model_specific", "copyleft", "permissive", "other", "missing")
    return next(label for label in precedence if label in classes)


def _single_license_class(normalized: str) -> str:
    if normalized == "missing":
        return "missing"
    if normalized in PERMISSIVE:
        return "permissive"
    if normalized in COPYLEFT:
        return "copyleft"
    if normalized in RESTRICTED or any(token in normalized for token in ("-nc", "research-only", "non-commercial")):
        return "restricted"
    if normalized.startswith(MODEL_SPECIFIC_PREFIXES) or "openrail" in normalized:
        return "model_specific"
    return "other"


def rights_review_flag(app_license: object, upstream_licenses: Iterable[object]) -> str:
    """Return a documentation-review status, not a legal conclusion."""

    app_class = license_class(app_license)
    upstream_classes = [license_class(value) for value in upstream_licenses]
    if app_class == "missing":
        return "app_license_missing"
    if not upstream_classes:
        return "no_observed_model_dependency"
    if all(value == "missing" for value in upstream_classes):
        return "upstream_license_unresolved"
    if app_class in {"permissive", "copyleft"} and any(
        value in {"restricted", "model_specific"} for value in upstream_classes
    ):
        return "review_rights_asymmetry"
    if app_class in {"permissive", "copyleft"} and any(
        value in {"missing", "other"} for value in upstream_classes
    ):
        return "review_incomplete_upstream_terms"
    if app_class in {"permissive", "copyleft"} and all(
        value in {"permissive", "copyleft"} for value in upstream_classes
    ):
        return "no_automated_flag"
    return "manual_review"


def detect_license_from_text(text: str) -> str:
    """Identify only high-precision license-text fingerprints."""

    normalized = re.sub(r"\s+", " ", text.casefold())
    fingerprints = (
        ("mit", "permission is hereby granted, free of charge, to any person obtaining a copy"),
        ("apache-2.0", "apache license version 2.0"),
        ("agpl-3.0", "gnu affero general public license"),
        ("lgpl-3.0", "gnu lesser general public license"),
        ("gpl-3.0", "gnu general public license"),
        ("mpl-2.0", "mozilla public license version 2.0"),
        ("bsd-3-clause", "neither the name of the copyright holder nor the names of its contributors"),
    )
    for label, fingerprint in fingerprints:
        if fingerprint in normalized:
            return label
    return "missing"
