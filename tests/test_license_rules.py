from license_rules import (
    detect_license_from_text,
    license_class,
    normalize_license,
    rights_review_flag,
)


def test_license_normalization_and_classes() -> None:
    assert normalize_license("Apache 2") == "apache-2.0"
    assert license_class("MIT") == "permissive"
    assert license_class("cc-by-nc-4.0") == "restricted"
    assert license_class("llama3.1") == "model_specific"


def test_rights_review_flag_is_not_legal_compatibility() -> None:
    assert rights_review_flag("mit", ["llama3.1"]) == "review_rights_asymmetry"
    assert rights_review_flag("mit", ["apache-2.0"]) == "no_automated_flag"
    assert rights_review_flag(None, ["apache-2.0"]) == "app_license_missing"
    assert rights_review_flag("mit", []) == "no_observed_model_dependency"


def test_detect_license_fingerprint() -> None:
    assert (
        detect_license_from_text(
            "Permission is hereby granted, free of charge, to any person obtaining a copy"
        )
        == "mit"
    )
    assert detect_license_from_text("Copyright only") == "missing"
