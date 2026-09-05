"""Parse public Space files into auditable dependency signals."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import io
import ipaddress
import json
from pathlib import PurePosixPath
import re
import tokenize
import tomllib
from typing import Iterable, Mapping

from protocol import (
    CODE_PROVIDER_PATTERNS,
    CODE_SUFFIXES,
    MAX_SOURCE_FILES,
    PACKAGE_PROVIDER_RULES,
    TEXT_FILE_BASENAMES,
)


@dataclass(frozen=True)
class DependencySignal:
    provider: str
    layer: str
    evidence_type: str
    evidence_value: str
    source_file: str
    confidence: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class UnmappedDependencyCandidate:
    """Machine-extracted signal excluded from the primary provider taxonomy."""

    candidate_type: str
    identifier: str
    source_file: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


_GENERATED_OR_TEST_PARTS = {
    ".ipynb_checkpoints",
    ".next",
    "__pycache__",
    "build",
    "checkpoint",
    "checkpoints",
    "coverage",
    "dist",
    "node_modules",
    "site-packages",
    "test",
    "tests",
    "vendor",
}
_TEST_FILE_RE = re.compile(
    r"^(?:conftest|run_tests?|test_.+|.+_(?:spec|test))\.(?:py|js|jsx|ts|tsx)$",
    flags=re.IGNORECASE,
)
_BLOCK_COMMENT_RE = re.compile(r"(?s)/\*.*?\*/|<!--.*?-->")
_LINE_COMMENT_RE = re.compile(r"(?m)(?<!:)//[^\n]*")
_URL_RE = re.compile(r"https?://[A-Za-z0-9.-]+(?::\d+)?(?:/[^\s\"'`<>)]*)?", re.IGNORECASE)
_CREDENTIAL_RE = re.compile(
    r"\b[A-Z][A-Z0-9_]{2,}_(?:API_KEY|API_TOKEN|ACCESS_TOKEN|ENDPOINT)\b"
)
_OPENAI_CLIENT_RE = re.compile(
    r"(?:from\s+openai\s+import|import\s+openai\b|from\s+langchain_openai\s+import|\b(?:Async)?OpenAI\s*\()",
    flags=re.IGNORECASE,
)
_EXPLICIT_BASE_URL_RE = re.compile(
    r"(?:base_?url|baseURL|api_?base|openai_api_base|endpoint)\s*[:=]\s*[rubf]*[\"']"
    r"(https?://[^\"']+)[\"']",
    flags=re.IGNORECASE,
)
_BASE_URL_OVERRIDE_RE = re.compile(
    r"(?:base_?url|baseURL|api_?base|openai_api_base|OPENAI_BASE_URL)\s*[:=]",
    flags=re.IGNORECASE,
)
_OPENAI_CONSTRUCTOR_RE = re.compile(r"\b(?:Async)?OpenAI\s*\(", re.IGNORECASE)
_OPENAI_CLIENT_PACKAGES = {"openai", "langchain-openai"}
SERVICE_CANDIDATE_TYPES = {
    "unmapped_credential",
    "unmapped_api_domain",
    "openai_compatible_provider_unresolved",
}

_NONRUNTIME_REQUIREMENT_MARKERS = {
    "dev",
    "develop",
    "development",
    "doc",
    "docs",
    "lint",
    "test",
    "tests",
    "typing",
}


def is_analysis_source(path: str) -> bool:
    """Exclude generated, vendored, checkpoint, and test files mechanically."""

    normalized = PurePosixPath(path)
    parts = {part.casefold() for part in normalized.parts[:-1]}
    name = normalized.name.casefold()
    return not (parts & _GENERATED_OR_TEST_PARTS or _TEST_FILE_RE.match(name))


def _blank_spans(text: str, spans: Iterable[tuple[int, int]]) -> str:
    """Blank character spans while retaining line breaks and source positions."""

    output = list(text)
    for start, end in spans:
        for index in range(max(0, start), min(len(output), end)):
            if output[index] not in {"\n", "\r"}:
                output[index] = " "
    return "".join(output)


def _line_offsets(text: str) -> list[int]:
    offsets = [0]
    for match in re.finditer("\n", text):
        offsets.append(match.end())
    return offsets


def _absolute_offset(offsets: list[int], position: tuple[int, int], text_length: int) -> int:
    line, column = position
    if line <= 0 or line > len(offsets):
        return text_length
    return min(text_length, offsets[line - 1] + column)


def _python_non_executable_spans(text: str) -> list[tuple[int, int]]:
    offsets = _line_offsets(text)
    spans: list[tuple[int, int]] = []
    try:
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        for token in tokens:
            if token.type == tokenize.COMMENT:
                spans.append(
                    (
                        _absolute_offset(offsets, token.start, len(text)),
                        _absolute_offset(offsets, token.end, len(text)),
                    )
                )
    except (IndentationError, tokenize.TokenError):
        pass

    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return spans
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        body = getattr(node, "body", ())
        if not body:
            continue
        first = body[0]
        if not (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
            and hasattr(first, "end_lineno")
        ):
            continue
        spans.append(
            (
                _absolute_offset(offsets, (first.lineno, first.col_offset), len(text)),
                _absolute_offset(
                    offsets,
                    (first.end_lineno or first.lineno, first.end_col_offset or first.col_offset),
                    len(text),
                ),
            )
        )
    return spans


def executable_text(path: str, text: str) -> str:
    """Remove comments and Python docstrings without deleting runtime strings."""

    text = text.lstrip("\ufeff")
    suffix = PurePosixPath(path).suffix.casefold()
    if suffix == ".py":
        return _blank_spans(text, _python_non_executable_spans(text))
    if suffix in {".js", ".jsx", ".ts", ".tsx", ".html"}:
        return _LINE_COMMENT_RE.sub(" ", _BLOCK_COMMENT_RE.sub(" ", text))
    if suffix == ".ini" or PurePosixPath(path).name == ".env.example":
        return re.sub(r"(?m)^\s*[#;].*$", "", text)
    return text


def _openai_client_context(files: Mapping[str, str]) -> dict[str, object]:
    code_texts: dict[str, str] = {}
    package_sources: list[str] = []
    for path, text in files.items():
        if not is_analysis_source(path):
            continue
        if extract_packages(path, text) & _OPENAI_CLIENT_PACKAGES:
            package_sources.append(path)
        suffix = PurePosixPath(path).suffix.casefold()
        if suffix in CODE_SUFFIXES or PurePosixPath(path).name == ".env.example":
            code_texts[path] = executable_text(path, text)
    joined = "\n".join(code_texts.values())
    explicit_urls = _EXPLICIT_BASE_URL_RE.findall(joined)
    alternative_urls = [
        url for url in explicit_urls if "api.openai.com" not in url.casefold()
    ]
    official_endpoint = bool(re.search(r"api\.openai\.com", joined, re.IGNORECASE))
    base_url_override = bool(_BASE_URL_OVERRIDE_RE.search(joined)) and not official_endpoint
    official_credential = bool(re.search(r"\bOPENAI_API_KEY\b", joined))
    official_constructor = bool(_OPENAI_CONSTRUCTOR_RE.search(joined))
    code_sources = [
        path for path, text in code_texts.items() if _OPENAI_CLIENT_RE.search(text)
    ]
    other_provider_explicit = any(
        rule.provider != "OpenAI" and re.search(pattern, joined, re.IGNORECASE)
        for rule, _, pattern in CODE_PROVIDER_PATTERNS
    )
    client_present = bool(code_sources or package_sources)
    official = bool(
        official_endpoint
        or (
            (official_credential or official_constructor)
            and not alternative_urls
            and not base_url_override
        )
    )
    unresolved = bool(
        client_present
        and not official
        and not alternative_urls
        and (code_sources or not other_provider_explicit)
    )
    return {
        "alternative_urls": alternative_urls,
        "official_endpoint": official_endpoint,
        "official_credential": official_credential,
        "official_constructor": official_constructor,
        "base_url_override": base_url_override,
        "official": official,
        "unresolved": unresolved,
        "code_sources": code_sources,
        "package_sources": package_sources,
    }


def normalize_package(value: str) -> str:
    value = re.split(r"[<>=!~;\s\[]", value.strip(), maxsplit=1)[0]
    return re.sub(r"[_.]+", "-", value.casefold())


def _requirement_packages(text: str) -> set[str]:
    packages: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip().lstrip("\ufeff")
        if not line or line.startswith(("#", "-r", "--", "git+", "http:" , "https:")):
            continue
        package = normalize_package(line)
        if package:
            packages.add(package)
    return packages


def _pyproject_package_groups(text: str) -> tuple[set[str], set[str]]:
    try:
        payload = tomllib.loads(text.lstrip("\ufeff"))
    except (tomllib.TOMLDecodeError, ValueError):
        return set(), set()
    runtime_values: list[str] = []
    nonruntime_values: list[str] = []
    project = payload.get("project") or {}
    runtime_values.extend(project.get("dependencies") or [])
    for deps in (project.get("optional-dependencies") or {}).values():
        nonruntime_values.extend(deps or [])
    poetry = (((payload.get("tool") or {}).get("poetry") or {}).get("dependencies") or {})
    runtime_values.extend(str(name) for name in poetry if name.casefold() != "python")
    poetry_groups = (((payload.get("tool") or {}).get("poetry") or {}).get("group") or {})
    for group in poetry_groups.values():
        dependencies = (group or {}).get("dependencies") or {}
        nonruntime_values.extend(
            str(name) for name in dependencies if name.casefold() != "python"
        )
    groups = (payload.get("dependency-groups") or {})
    for deps in groups.values():
        nonruntime_values.extend(item for item in (deps or []) if isinstance(item, str))
    runtime = {
        normalize_package(value)
        for value in runtime_values
        if normalize_package(value)
    }
    nonruntime = {
        normalize_package(value)
        for value in nonruntime_values
        if normalize_package(value)
    }
    return runtime, nonruntime


def _package_json_package_groups(text: str) -> tuple[set[str], set[str]]:
    try:
        payload = json.loads(text.lstrip("\ufeff"))
    except (json.JSONDecodeError, TypeError):
        return set(), set()
    runtime = {
        normalize_package(name) for name in (payload.get("dependencies") or {})
    }
    nonruntime = {
        normalize_package(name)
        for key in ("devDependencies", "peerDependencies", "optionalDependencies")
        for name in (payload.get(key) or {})
    }
    return (
        {package for package in runtime if package},
        {package for package in nonruntime if package},
    )


def _requirements_file_is_nonruntime(path: str) -> bool:
    name = PurePosixPath(path).name.casefold()
    stem = name.removesuffix(".txt")
    parts = {part for part in re.split(r"[-_.]+", stem) if part}
    return name != "requirements.txt" and bool(
        parts & _NONRUNTIME_REQUIREMENT_MARKERS
    )


def extract_packages(path: str, text: str) -> set[str]:
    """Extract normalized runtime packages from a supported manifest.

    Development, test, peer, and optional dependency groups are deliberately
    excluded from primary provider edges. They remain in the QA inventory via
    :func:`extract_nonruntime_packages`.
    """

    name = PurePosixPath(path).name.casefold()
    if name.startswith("requirements") and name.endswith(".txt"):
        if _requirements_file_is_nonruntime(path):
            return set()
        return _requirement_packages(text)
    if name == "pyproject.toml":
        return _pyproject_package_groups(text)[0]
    if name == "package.json":
        return _package_json_package_groups(text)[0]
    if name in {"environment.yml", "environment.yaml", "pipfile", "setup.py", "setup.cfg"}:
        # Conservative line parser: only exact known package names are later
        # accepted, so incidental prose cannot create a provider edge.
        candidates = set(re.findall(r"[A-Za-z][A-Za-z0-9_.-]{1,60}", text))
        return {normalize_package(value) for value in candidates}
    return set()


def extract_nonruntime_packages(path: str, text: str) -> set[str]:
    """Extract packages declared only in dev, test, peer, or optional groups."""

    name = PurePosixPath(path).name.casefold()
    if name.startswith("requirements") and name.endswith(".txt"):
        return (
            _requirement_packages(text)
            if _requirements_file_is_nonruntime(path)
            else set()
        )
    if name == "pyproject.toml":
        return _pyproject_package_groups(text)[1]
    if name == "package.json":
        return _package_json_package_groups(text)[1]
    return set()


def _placeholder_api_domain(domain: str) -> bool:
    """Reject documentation and loopback endpoints as external services."""

    value = domain.casefold().strip("[]")
    if value in {"localhost", "api.example.com"} or value.endswith(
        (".example", ".example.com", ".example.org", ".example.net")
    ):
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def has_machine_service_candidate(
    candidates: Iterable[UnmappedDependencyCandidate],
) -> bool:
    """Return whether frozen machine signals support dependency observability."""

    return any(
        candidate.candidate_type in SERVICE_CANDIDATE_TYPES
        for candidate in candidates
    )


def select_repository_files(
    siblings: Iterable[str],
    app_file: str | None,
    *,
    max_files: int = MAX_SOURCE_FILES,
) -> list[str]:
    """Choose a bounded, reproducible set of manifests and execution files."""

    names = sorted({str(name) for name in siblings if is_analysis_source(str(name))})
    selected: list[str] = []
    # The declared execution entry point is the highest-priority bounded file;
    # otherwise a repository with many manifests can silently push it past the
    # fixed file cap.
    if app_file and app_file in names and is_analysis_source(app_file):
        selected.append(app_file)
    manifest_names = [
        name for name in names if PurePosixPath(name).name.casefold() in TEXT_FILE_BASENAMES
        or PurePosixPath(name).name.casefold().startswith("requirements")
        and name.casefold().endswith(".txt")
    ]
    selected.extend(manifest_names)
    preferred = (
        "app.py", "main.py", "index.py", "server.py", "app.js", "index.js",
        "src/app.py", "src/main.py", "src/index.js", "index.html",
    )
    selected.extend(name for name in preferred if name in names)
    if len(set(selected)) < max_files:
        roots = [
            name for name in names
            if len(PurePosixPath(name).parts) <= 2
            and PurePosixPath(name).suffix.casefold() in CODE_SUFFIXES
        ]
        selected.extend(roots)
    return list(dict.fromkeys(selected))[:max_files]


def provider_signals(files: Mapping[str, str]) -> list[DependencySignal]:
    """Detect unique provider signals from manifests and executable code."""

    signals: list[DependencySignal] = []
    openai_context = _openai_client_context(files)
    for path, text in sorted(files.items()):
        if not is_analysis_source(path):
            continue
        for package in sorted(extract_packages(path, text)):
            rule = PACKAGE_PROVIDER_RULES.get(package)
            if rule:
                signals.append(
                    DependencySignal(
                        provider=rule.provider,
                        layer=rule.layer,
                        evidence_type="package",
                        evidence_value=package,
                        source_file=path,
                        confidence="high",
                    )
                )
        suffix = PurePosixPath(path).suffix.casefold()
        if suffix not in CODE_SUFFIXES and PurePosixPath(path).name != ".env.example":
            continue
        analysis_text = executable_text(path, text)
        for rule, label, pattern in CODE_PROVIDER_PATTERNS:
            matched = bool(re.search(pattern, analysis_text, flags=re.IGNORECASE))
            if rule.provider == "OpenAI":
                matched = bool(
                    re.search(r"api\.openai\.com", analysis_text, re.IGNORECASE)
                    or (
                        (
                            re.search(r"\bOPENAI_API_KEY\b", analysis_text)
                            or _OPENAI_CONSTRUCTOR_RE.search(analysis_text)
                        )
                        and not openai_context["alternative_urls"]
                        and not openai_context["base_url_override"]
                    )
                )
            if matched:
                is_configuration = (
                    PurePosixPath(path).name.casefold() == ".env.example"
                )
                confidence = "medium" if is_configuration else "high"
                signals.append(
                    DependencySignal(
                        provider=rule.provider,
                        layer=rule.layer,
                        evidence_type=(
                            "configuration_signature"
                            if is_configuration
                            else "code_signature"
                        ),
                        evidence_value=label,
                        source_file=path,
                        confidence=confidence,
                    )
                )
    unique: dict[tuple[str, str, str, str, str], DependencySignal] = {}
    for signal in signals:
        key = (
            signal.provider,
            signal.layer,
            signal.evidence_type,
            signal.evidence_value,
            signal.source_file,
        )
        unique[key] = signal
    return sorted(
        unique.values(),
        key=lambda item: (
            item.provider, item.layer, item.evidence_type, item.source_file, item.evidence_value
        ),
    )


def extract_github_repositories(text: str) -> list[str]:
    """Extract normalized public GitHub owner/repository links from prose."""

    pattern = r"https?://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)"
    values = []
    for owner, repository in re.findall(pattern, text, flags=re.IGNORECASE):
        repository = repository.removesuffix(".git").rstrip(".,);]}")
        if repository:
            values.append(f"{owner}/{repository}")
    return sorted(set(values), key=str.casefold)


def extract_code_model_ids(files: Mapping[str, str]) -> list[str]:
    """Extract model IDs only from common model-loading argument contexts."""

    patterns = (
        r"from_pretrained\(\s*[rubf]*[\"']([\w.-]+/[\w.:-]+)[\"']",
        r"(?:model_id|model_name|repo_id|model)\s*=\s*[rubf]*[\"']([\w.-]+/[\w.:-]+)[\"']",
    )
    found: set[str] = set()
    for path, text in files.items():
        if (
            PurePosixPath(path).suffix.casefold() not in CODE_SUFFIXES
            or not is_analysis_source(path)
        ):
            continue
        analysis_text = executable_text(path, text)
        for pattern in patterns:
            found.update(re.findall(pattern, analysis_text, flags=re.IGNORECASE))
    return sorted(found, key=str.casefold)


def unmapped_dependency_candidates(
    files: Mapping[str, str],
) -> list[UnmappedDependencyCandidate]:
    """Inventory unmapped machine signals without treating them as provider edges.

    Manifest packages are reported separately because an unrecognized package is
    not necessarily an inference service. Credential names and API-like domains
    are stronger service candidates, but remain outside the primary taxonomy.
    """

    candidates: dict[tuple[str, str, str], UnmappedDependencyCandidate] = {}
    openai_context = _openai_client_context(files)
    for path, text in sorted(files.items()):
        if not is_analysis_source(path):
            continue
        for package in sorted(extract_nonruntime_packages(path, text)):
            candidate = UnmappedDependencyCandidate(
                candidate_type="nonruntime_manifest_package",
                identifier=package,
                source_file=path,
            )
            candidates[(candidate.candidate_type, package, path)] = candidate
        for package in sorted(extract_packages(path, text)):
            if package not in PACKAGE_PROVIDER_RULES and package not in _OPENAI_CLIENT_PACKAGES:
                candidate = UnmappedDependencyCandidate(
                    candidate_type="unmapped_manifest_package",
                    identifier=package,
                    source_file=path,
                )
                candidates[(candidate.candidate_type, package, path)] = candidate

        suffix = PurePosixPath(path).suffix.casefold()
        if suffix not in CODE_SUFFIXES and PurePosixPath(path).name != ".env.example":
            continue
        analysis_text = executable_text(path, text)
        for credential in sorted(set(_CREDENTIAL_RE.findall(analysis_text))):
            if any(
                re.search(pattern, credential, flags=re.IGNORECASE)
                for _, _, pattern in CODE_PROVIDER_PATTERNS
            ):
                continue
            candidate = UnmappedDependencyCandidate(
                candidate_type="unmapped_credential",
                identifier=credential,
                source_file=path,
            )
            candidates[(candidate.candidate_type, credential, path)] = candidate
        for url in sorted(set(_URL_RE.findall(analysis_text)), key=str.casefold):
            domain_match = re.match(r"https?://([^/:]+)", url, flags=re.IGNORECASE)
            if not domain_match:
                continue
            domain = domain_match.group(1).casefold()
            if _placeholder_api_domain(domain):
                continue
            api_like = (
                domain.startswith("api.")
                or any(
                    marker in url.casefold()
                    for marker in ("/v1", "/chat", "/complet", "/generat", "/infer", "/models")
                )
            )
            if not api_like or any(
                re.search(pattern, url, flags=re.IGNORECASE)
                for _, _, pattern in CODE_PROVIDER_PATTERNS
            ):
                continue
            candidate = UnmappedDependencyCandidate(
                candidate_type="unmapped_api_domain",
                identifier=domain,
                source_file=path,
            )
            candidates[(candidate.candidate_type, domain, path)] = candidate
    if openai_context["unresolved"]:
        sources = list(openai_context["code_sources"]) or list(
            openai_context["package_sources"]
        )
        for path in sorted(set(str(value) for value in sources)):
            candidate = UnmappedDependencyCandidate(
                candidate_type="openai_compatible_provider_unresolved",
                identifier="openai-compatible-client",
                source_file=path,
            )
            candidates[(candidate.candidate_type, candidate.identifier, path)] = candidate
    return sorted(
        candidates.values(),
        key=lambda item: (item.candidate_type, item.identifier.casefold(), item.source_file),
    )
