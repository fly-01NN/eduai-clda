from verify_references import (
    bib_family_names,
    crossref_title,
    normalized,
    parse_bibtex,
    title_similarity,
)


def test_parse_bibtex_entries_and_fields() -> None:
    entries = parse_bibtex(
        """@article{Example2026,
  title = {A {Test} Title},
  author = {Doe, Jane and Roe, Richard},
  year = {2026},
  doi = {10.1000/example}
}
"""
    )
    assert len(entries) == 1
    assert entries[0]["key"] == "Example2026"
    assert entries[0]["fields"]["doi"] == "10.1000/example"


def test_normalization_ignores_bibtex_case_braces_and_punctuation() -> None:
    assert title_similarity("Open {AI}: A Study", "Open AI - A Study") == 1.0
    assert normalized("Mar{\\'i}n") == "marin"


def test_bib_family_names_support_comma_form() -> None:
    assert bib_family_names("Doe, Jane and van Dijck, Jos{\\'e}") == ["doe", "van dijck"]


def test_crossref_title_reconstructs_separate_subtitle() -> None:
    message = {
        "title": ["SourcererCC"],
        "subtitle": ["scaling code clone detection to big-code"],
    }
    assert crossref_title(message) == (
        "SourcererCC: scaling code clone detection to big-code"
    )


def test_crossref_title_does_not_duplicate_embedded_subtitle() -> None:
    message = {
        "title": ["A study: evidence from public repositories"],
        "subtitle": ["Evidence from public repositories"],
    }
    assert crossref_title(message) == "A study: evidence from public repositories"
