from protocol import match_education_constructs, model_family, model_provider


def test_tutor_matches_but_tutorial_does_not() -> None:
    assert "tutor" in match_education_constructs("An AI tutor for algebra")
    assert match_education_constructs("A machine-learning tutorial") == []


def test_strict_education_function_phrases() -> None:
    assert "quiz_generation" in match_education_constructs("AI quiz generator")
    assert "study_support" in match_education_constructs("Learning companion")
    assert "language_learning" in match_education_constructs("Japanese language tutor")


def test_camel_case_function_phrases_are_normalized() -> None:
    assert "study_support" in match_education_constructs("StudyAssistant")
    assert "lesson_planning" in match_education_constructs("LessonPlanGenerator")
    assert "education_app" in match_education_constructs("EducationalAIChatbot")


def test_generic_student_requires_broad_rule() -> None:
    assert match_education_constructs("student placement prediction") == []
    assert "generic_school" in match_education_constructs(
        "student placement prediction", broad=True
    )


def test_non_education_assessment_is_not_strict() -> None:
    assert match_education_constructs("Environmental impact assessment generator") == []


def test_model_provider_prefers_known_base_model() -> None:
    provider, basis = model_provider(
        "individual/my-finetune", ["meta-llama/Llama-3.1-8B-Instruct"]
    )
    assert provider == "Meta"
    assert basis == "base_model_namespace"


def test_unmapped_model_namespace_is_preserved() -> None:
    provider, basis = model_provider("researcher/custom-model")
    assert provider == "namespace:researcher"
    assert basis == "unmapped_public_namespace"


def test_model_family_uses_base_model() -> None:
    assert model_family("person/custom", ["Qwen/Qwen3-8B"]) == "Qwen"
