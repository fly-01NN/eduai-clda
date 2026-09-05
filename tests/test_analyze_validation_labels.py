import pandas as pd

from analyze_validation_labels import cohen_kappa, resolved_label


def test_resolved_label_requires_agreement_or_adjudication() -> None:
    assert resolved_label("yes", "yes", "") == "yes"
    assert resolved_label("yes", "no", "no") == "no"
    assert resolved_label("yes", "no", "") == ""
    assert resolved_label("uncertain", "uncertain", "yes") == "yes"


def test_cohen_kappa_reports_perfect_agreement() -> None:
    labels = pd.Series(["yes", "no", "uncertain"])
    result = cohen_kappa(labels, labels)
    assert result["n"] == 3
    assert result["agreement"] == 1.0
    assert result["kappa"] == 1.0
