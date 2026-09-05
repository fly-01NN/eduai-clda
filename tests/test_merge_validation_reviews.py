import pandas as pd

from merge_validation_reviews import unresolved_ids


def test_unresolved_ids_include_disagreements_and_uncertain_pairs():
    first = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "label": ["yes", "yes", "uncertain", "no"],
        }
    )
    second = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "label": ["yes", "no", "uncertain", "no"],
        }
    )

    assert unresolved_ids(first, second, "label") == {"b", "c"}
