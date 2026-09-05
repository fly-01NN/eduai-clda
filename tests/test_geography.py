from geography import classify_location, language_orientation


def test_explicit_country_locations_only() -> None:
    assert classify_location("Shanghai, China") == ("asia", "CN")
    assert classify_location("Paris, France") == ("outside_asia", "FR")
    assert classify_location("San Francisco") == ("unresolved", "")
    assert classify_location(None) == ("missing", "")


def test_language_orientation_is_not_geography() -> None:
    assert language_orientation(["language:zh"])[0] == "asia_language_only"
    assert language_orientation(["language:zh", "language:en"])[0] == "includes_asia_language"
    assert language_orientation(["region:us"])[0] == "undeclared"

