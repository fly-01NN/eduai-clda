import pandas as pd

from collect_file_cap_sensitivity import service_provider_sets


def test_service_provider_sets_collapse_duplicate_evidence() -> None:
    edges = pd.DataFrame(
        {
            "space_id": ["a", "a", "a", "b"],
            "layer": ["inference_service", "inference_service", "model_dependency", "inference_service"],
            "provider": ["S1", "S1", "M1", "S2"],
        }
    )
    assert service_provider_sets(edges) == {
        "a": frozenset({"S1"}),
        "b": frozenset({"S2"}),
    }
