"""Verify manuscript references against Crossref, arXiv, or the cited URL."""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import time
import unicodedata
from urllib.parse import quote
import xml.etree.ElementTree as ET

import httpx


ENTRY_RE = re.compile(
    r"^@(?P<kind>\w+)\{(?P<key>[^,]+),\s*(?P<body>.*?)(?=^@\w+\{|\Z)",
    flags=re.MULTILINE | re.DOTALL,
)
FIELD_RE = re.compile(
    r"^\s*(?P<name>[A-Za-z][\w-]*)\s*=\s*\{(?P<value>.*)\},?\s*$",
    flags=re.MULTILINE,
)
ATOM = {"a": "http://www.w3.org/2005/Atom"}


def parse_bibtex(text: str) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for match in ENTRY_RE.finditer(text):
        fields = {
            field.group("name").casefold(): field.group("value").strip()
            for field in FIELD_RE.finditer(match.group("body"))
        }
        entries.append(
            {
                "entry_type": match.group("kind").casefold(),
                "key": match.group("key").strip(),
                "fields": fields,
            }
        )
    return entries


def normalized(value: object) -> str:
    text = str(value or "")
    # Common BibTeX accent forms are reduced only for metadata comparison.
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"\\[\"'`^~=.uvHckbdtr]\s*\{?([A-Za-z])\}?", r"\1", text)
    text = re.sub(r"\\[A-Za-z]+", " ", text)
    text = text.replace("--", " ").replace("&", " and ")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9]+", " ", text.casefold())
    return re.sub(r"\s+", " ", text).strip()


def title_similarity(left: object, right: object) -> float:
    return SequenceMatcher(None, normalized(left), normalized(right)).ratio()


def bib_family_names(author_field: str) -> list[str]:
    values: list[str] = []
    for raw in re.split(r"\s+and\s+", author_field):
        name = raw.strip().strip("{}")
        family = name.split(",", 1)[0] if "," in name else name.split()[-1]
        if family:
            values.append(normalized(family))
    return values


def author_token(value: object) -> str:
    """Return a comparison token robust to Crossref name-particle handling."""

    parts = normalized(value).split()
    return parts[-1] if parts else ""


def crossref_years(message: dict) -> set[int]:
    output: set[int] = set()
    for key in ("issued", "published", "published-print", "published-online", "created"):
        date_parts = (message.get(key) or {}).get("date-parts") or []
        if date_parts and date_parts[0]:
            output.add(int(date_parts[0][0]))
    return output


def crossref_title(message: dict) -> str:
    """Reconstruct a work title when Crossref stores its subtitle separately."""

    title = str((message.get("title") or [""])[0]).strip()
    subtitle = str((message.get("subtitle") or [""])[0]).strip()
    if not subtitle:
        return title
    if normalized(subtitle) in normalized(title):
        return title
    return f"{title}: {subtitle}"


def verify_doi(client: httpx.Client, key: str, fields: dict[str, str]) -> dict[str, object]:
    doi = fields["doi"]
    response = client.get(f"https://api.crossref.org/works/{quote(doi, safe='')}")
    response.raise_for_status()
    message = response.json()["message"]
    remote_title = crossref_title(message)
    similarity = title_similarity(fields.get("title"), remote_title)
    local_authors = {author_token(value) for value in bib_family_names(fields.get("author", ""))}
    remote_authors = {
        author_token(author.get("family", ""))
        for author in message.get("author") or []
        if author.get("family")
    }
    author_coverage = (
        len(local_authors & remote_authors) / len(local_authors) if local_authors else 1.0
    )
    local_year = int(fields["year"]) if fields.get("year", "").isdigit() else None
    years = crossref_years(message)
    year_match = local_year in years if local_year is not None else True
    passed = similarity >= 0.88 and author_coverage >= 0.80 and year_match
    return {
        "key": key,
        "route": "crossref_doi",
        "identifier": doi,
        "status": "PASS" if passed else "REVIEW",
        "title_similarity": round(similarity, 4),
        "author_coverage": round(author_coverage, 4),
        "year_match": year_match,
        "remote_years": sorted(years),
        "remote_title": remote_title,
        "source": f"https://api.crossref.org/works/{quote(doi, safe='')}",
    }


def arxiv_records(client: httpx.Client, identifiers: list[str]) -> dict[str, dict[str, object]]:
    if not identifiers:
        return {}
    response = client.get(
        "https://export.arxiv.org/api/query",
        params={"id_list": ",".join(identifiers), "max_results": len(identifiers)},
        timeout=90,
    )
    response.raise_for_status()
    root = ET.fromstring(response.text)
    records: dict[str, dict[str, object]] = {}
    for entry in root.findall("a:entry", ATOM):
        identifier = (entry.findtext("a:id", default="", namespaces=ATOM).rsplit("/", 1)[-1])
        identifier = identifier.split("v", 1)[0]
        records[identifier] = {
            "title": entry.findtext("a:title", default="", namespaces=ATOM),
            "authors": [
                node.findtext("a:name", default="", namespaces=ATOM)
                for node in entry.findall("a:author", ATOM)
            ],
            "published": entry.findtext("a:published", default="", namespaces=ATOM),
            "updated": entry.findtext("a:updated", default="", namespaces=ATOM),
        }
    return records


def verify_arxiv(
    key: str,
    fields: dict[str, str],
    records: dict[str, dict[str, object]],
) -> dict[str, object]:
    identifier = fields["eprint"]
    remote = records.get(identifier)
    if not remote:
        return {
            "key": key,
            "route": "arxiv",
            "identifier": identifier,
            "status": "REVIEW",
            "reason": "identifier absent from arXiv response",
        }
    similarity = title_similarity(fields.get("title"), remote["title"])
    local_authors = {author_token(value) for value in bib_family_names(fields.get("author", ""))}
    remote_authors = {author_token(name) for name in remote["authors"]}
    author_coverage = len(local_authors & remote_authors) / len(local_authors)
    local_year = fields.get("year")
    year_match = str(remote["published"]).startswith(str(local_year))
    passed = similarity >= 0.88 and author_coverage >= 0.80 and year_match
    return {
        "key": key,
        "route": "arxiv",
        "identifier": identifier,
        "status": "PASS" if passed else "REVIEW",
        "title_similarity": round(similarity, 4),
        "author_coverage": round(author_coverage, 4),
        "year_match": year_match,
        "published": remote["published"],
        "updated": remote["updated"],
        "source": f"https://export.arxiv.org/api/query?id_list={identifier}",
    }


def verify_url(client: httpx.Client, key: str, fields: dict[str, str]) -> dict[str, object]:
    url = fields["url"]
    try:
        response = client.get(url, timeout=60)
        passed = 200 <= response.status_code < 400
        return {
            "key": key,
            "route": "authoritative_url",
            "identifier": url,
            "status": "PASS" if passed else "REVIEW",
            "http_status": response.status_code,
            "resolved_url": str(response.url),
            "source": url,
        }
    except httpx.HTTPError as exc:
        return {
            "key": key,
            "route": "authoritative_url",
            "identifier": url,
            "status": "REVIEW",
            "reason": type(exc).__name__,
            "source": url,
        }


def write_report(output: Path, entries: list[dict[str, object]]) -> None:
    status = "PASS" if all(row["status"] == "PASS" for row in entries) else "REVIEW"
    payload = {
        "status": status,
        "references": len(entries),
        "pass": sum(row["status"] == "PASS" for row in entries),
        "review": sum(row["status"] != "PASS" for row in entries),
        "checks": entries,
        "note": "Metadata agreement verifies citation identity; it does not validate every manuscript claim.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Reference verification",
        "",
        f"- Overall status: **{status}**",
        f"- References checked: {len(entries)}",
        f"- Passed: {payload['pass']}",
        f"- Review: {payload['review']}",
        "",
        "| Key | Route | Status | Identifier |",
        "|---|---|---:|---|",
    ]
    for row in entries:
        lines.append(
            f"| `{row['key']}` | {row['route']} | {row['status']} | {row['identifier']} |"
        )
    lines.extend(
        [
            "",
            "Metadata agreement verifies citation identity; claim-level support is tracked in the research evidence table.",
        ]
    )
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bib",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "paper/references.bib",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/qa/reference_verification.json",
    )
    args = parser.parse_args()
    bib_entries = parse_bibtex(args.bib.read_text(encoding="utf-8"))
    arxiv_ids = [
        str(entry["fields"]["eprint"])
        for entry in bib_entries
        if "eprint" in entry["fields"] and "doi" not in entry["fields"]
    ]
    client = httpx.Client(
        follow_redirects=True,
        headers={"User-Agent": "DE-004-citation-verifier/0.1 (mailto:23078403@siswa.um.edu.my)"},
    )
    try:
        records = arxiv_records(client, arxiv_ids)
        checks: list[dict[str, object]] = []
        for index, entry in enumerate(bib_entries):
            fields = entry["fields"]
            key = str(entry["key"])
            if "doi" in fields:
                check = verify_doi(client, key, fields)
            elif "eprint" in fields:
                check = verify_arxiv(key, fields, records)
            elif "url" in fields:
                check = verify_url(client, key, fields)
            else:
                check = {
                    "key": key,
                    "route": "none",
                    "identifier": "",
                    "status": "REVIEW",
                    "reason": "no DOI, arXiv identifier, or URL",
                }
            checks.append(check)
            if index and index % 5 == 0:
                time.sleep(0.1)
    finally:
        client.close()
    write_report(args.output, checks)
    print(json.dumps({"output": str(args.output), "status": "PASS" if all(row["status"] == "PASS" for row in checks) else "REVIEW", "references": len(checks)}, indent=2))


if __name__ == "__main__":
    main()
