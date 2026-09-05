from collect_hf_spaces import (
    education_metadata_constructs,
    education_metadata_text,
    normalize_base_models,
)
from protocol import match_education_constructs


def test_normalize_base_models_from_card_and_tags() -> None:
    values = normalize_base_models(
        [{"model": "meta-llama/Llama-3"}],
        ["base_model:finetune:Qwen/Qwen3-8B", "text-generation"],
    )
    assert values == ["meta-llama/Llama-3", "Qwen/Qwen3-8B"]


def test_author_namespace_cannot_trigger_education_inclusion() -> None:
    item = {
        "id": "AI-Tutor-Team/Travel-Agent",
        "cardData": {"title": "Travel Agent", "short_description": "Plan a trip"},
        "tags": ["agent"],
    }
    text = education_metadata_text(item, "A general itinerary planner.")
    assert match_education_constructs(text) == []


def test_education_phrases_cannot_form_across_metadata_field_boundaries() -> None:
    item = {
        "id": "team/study",
        "cardData": {"title": "Assistant", "short_description": "Plan a trip"},
        "tags": ["agent"],
    }
    assert "study_support" in match_education_constructs(
        education_metadata_text(item, "A planner")
    )
    assert education_metadata_constructs(item, "A planner") == []
