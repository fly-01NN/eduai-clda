"""Conservative geography parsing for explicit public profile locations."""

from __future__ import annotations

import re

import pycountry

from protocol import ASIA_ALPHA2, ASIA_LANGUAGE_CODES


ALIASES = {
    "usa": "US",
    "u s a": "US",
    "united states of america": "US",
    "uk": "GB",
    "u k": "GB",
    "south korea": "KR",
    "north korea": "KP",
    "russia": "RU",
    "vietnam": "VN",
    "laos": "LA",
    "iran": "IR",
    "syria": "SY",
    "turkey": "TR",
    "uae": "AE",
    "u a e": "AE",
    "hong kong": "HK",
    "macao": "MO",
    "macau": "MO",
    "taiwan": "TW",
}

# HK and MO are not ISO entries in the protocol's UN-country set but are
# retained as Asian locations when explicitly named in public profiles.
ASIA_LOCATION_CODES = ASIA_ALPHA2 | {"HK", "MO"}


def _normalized(value: str) -> str:
    value = value.casefold().replace(".", " ").replace("_", " ")
    return re.sub(r"\s+", " ", re.sub(r"[^\w ]+", " ", value)).strip()


COUNTRY_NAMES: tuple[tuple[str, str], ...] = tuple(
    sorted(
        {
            (_normalized(country.name), country.alpha_2)
            for country in pycountry.countries
        }
        | {
            (_normalized(getattr(country, "official_name")), country.alpha_2)
            for country in pycountry.countries
            if hasattr(country, "official_name")
        },
        key=lambda item: (-len(item[0]), item[0]),
    )
)


def classify_location(value: object) -> tuple[str, str]:
    """Return ``(region_class, country_code)`` from an explicit country name.

    City-only strings stay unresolved. This intentionally trades recall for
    avoiding inferred developer geography.
    """

    if value is None or not str(value).strip():
        return "missing", ""
    text = _normalized(str(value))
    for alias, code in ALIASES.items():
        if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", text):
            return ("asia" if code in ASIA_LOCATION_CODES else "outside_asia"), code
    for name, code in COUNTRY_NAMES:
        if len(name) < 4:
            continue
        if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text):
            return ("asia" if code in ASIA_LOCATION_CODES else "outside_asia"), code
    return "unresolved", ""


def language_orientation(tags: list[str], card_language: object = None) -> tuple[str, str]:
    """Classify declared content languages without treating them as geography."""

    values: set[str] = set()
    for tag in tags:
        if str(tag).casefold().startswith("language:"):
            values.add(str(tag).split(":", 1)[1].casefold().split("-", 1)[0])
    if isinstance(card_language, str):
        values.add(card_language.casefold().split("-", 1)[0])
    elif isinstance(card_language, (list, tuple, set)):
        values.update(str(item).casefold().split("-", 1)[0] for item in card_language)
    values.discard("")
    if not values:
        return "undeclared", ""
    if values <= ASIA_LANGUAGE_CODES:
        label = "asia_language_only"
    elif values & ASIA_LANGUAGE_CODES:
        label = "includes_asia_language"
    else:
        label = "other_declared_language"
    return label, ";".join(sorted(values))

