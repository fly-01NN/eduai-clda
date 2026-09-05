import pandas as pd

from build_validation_pack import provider_balanced_sample, stable_sample


def test_stable_sample_is_deterministic_and_bounded() -> None:
    frame = pd.DataFrame({"space_id": ["d", "c", "b", "a"]})
    first = stable_sample(frame, 3, salt="test")
    second = stable_sample(frame.sample(frac=1, random_state=9), 3, salt="test")
    assert first["space_id"].tolist() == second["space_id"].tolist()
    assert len(first) == 3


def test_provider_balanced_sample_retains_rare_provider() -> None:
    frame = pd.DataFrame(
        {
            "space_id": ["a", "b", "c", "d"],
            "provider": ["common", "common", "common", "rare"],
            "source_file": ["app.py"] * 4,
            "evidence_value": ["x", "y", "z", "r"],
        }
    )
    sampled = provider_balanced_sample(
        frame,
        3,
        salt="test",
        minimum_per_provider=1,
    )
    assert set(sampled["provider"]) == {"common", "rare"}
    assert len(sampled) == 3
