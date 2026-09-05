from dependency_parser import (
    extract_code_model_ids,
    extract_github_repositories,
    extract_nonruntime_packages,
    extract_packages,
    has_machine_service_candidate,
    provider_signals,
    select_repository_files,
    unmapped_dependency_candidates,
)


def test_extract_requirements_packages() -> None:
    packages = extract_packages(
        "requirements.txt",
        "openai>=1.0\nlangchain_openai[extras]==0.2\n# anthropic\n-r base.txt\n",
    )
    assert packages == {"openai", "langchain-openai"}


def test_repeated_bom_cannot_contaminate_first_requirement() -> None:
    assert extract_packages(
        "requirements.txt", "\ufeff\ufefflangchain-core\nopenai\n"
    ) == {"langchain-core", "openai"}


def test_extract_pyproject_pep621_and_poetry() -> None:
    packages = extract_packages(
        "pyproject.toml",
        """
[project]
dependencies = ["anthropic>=0.40", "transformers"]
[tool.poetry.dependencies]
python = "^3.11"
google-generativeai = "*"
""",
    )
    assert packages == {"anthropic", "transformers", "google-generativeai"}


def test_extract_package_json() -> None:
    packages = extract_packages(
        "package.json",
        '{"dependencies":{"openai":"^4"},"devDependencies":{"typescript":"^5"}}',
    )
    assert packages == {"openai"}
    assert extract_nonruntime_packages(
        "package.json",
        '{"dependencies":{"openai":"^4"},"devDependencies":{"typescript":"^5"}}',
    ) == {"typescript"}


def test_dev_requirement_is_qa_only_and_cannot_create_provider_edge() -> None:
    files = {"requirements-dev.txt": "anthropic\nopenai\n"}
    assert extract_packages("requirements-dev.txt", files["requirements-dev.txt"]) == set()
    assert extract_nonruntime_packages(
        "requirements-dev.txt", files["requirements-dev.txt"]
    ) == {"anthropic", "openai"}
    assert provider_signals(files) == []
    candidates = unmapped_dependency_candidates(files)
    assert {
        (candidate.candidate_type, candidate.identifier) for candidate in candidates
    } == {
        ("nonruntime_manifest_package", "anthropic"),
        ("nonruntime_manifest_package", "openai"),
    }
    assert not has_machine_service_candidate(candidates)


def test_pyproject_optional_and_dependency_groups_are_qa_only() -> None:
    text = '''
[project]
dependencies = ["anthropic>=0.40"]
[project.optional-dependencies]
ai = ["openai>=1"]
[dependency-groups]
dev = ["groq>=0.1"]
'''
    files = {"pyproject.toml": text}
    assert extract_packages("pyproject.toml", text) == {"anthropic"}
    assert extract_nonruntime_packages("pyproject.toml", text) == {"openai", "groq"}
    assert {signal.provider for signal in provider_signals(files)} == {"Anthropic"}
    assert {
        candidate.identifier
        for candidate in unmapped_dependency_candidates(files)
        if candidate.candidate_type == "nonruntime_manifest_package"
    } == {"openai", "groq"}


def test_package_json_dev_peer_and_optional_are_qa_only() -> None:
    text = '''{
      "dependencies": {"cohere": "^5"},
      "devDependencies": {"openai": "^4"},
      "peerDependencies": {"anthropic": "^1"},
      "optionalDependencies": {"groq": "^1"}
    }'''
    files = {"package.json": text}
    assert extract_packages("package.json", text) == {"cohere"}
    assert extract_nonruntime_packages("package.json", text) == {
        "openai", "anthropic", "groq"
    }
    assert {signal.provider for signal in provider_signals(files)} == {"Cohere"}


def test_provider_signals_require_dependency_context() -> None:
    signals = provider_signals(
        {
            "requirements.txt": "cohere>=5\n",
            "README.md": "This compares OpenAI and Anthropic in prose.",
            "app.py": "from anthropic import Anthropic\nkey = ANTHROPIC_API_KEY",
        }
    )
    providers = {(signal.provider, signal.layer) for signal in signals}
    assert ("Cohere", "inference_service") in providers
    assert ("Anthropic", "inference_service") in providers
    assert not [signal for signal in signals if signal.provider == "OpenAI"]


def test_openai_package_only_is_unresolved_compatible_candidate() -> None:
    files = {"requirements.txt": "openai>=1\nlangchain-openai\n"}
    assert provider_signals(files) == []
    candidates = unmapped_dependency_candidates(files)
    assert {
        (item.candidate_type, item.identifier) for item in candidates
    } == {
        ("openai_compatible_provider_unresolved", "openai-compatible-client")
    }
    assert has_machine_service_candidate(candidates)


def test_openai_client_with_third_party_base_url_is_not_openai_service() -> None:
    files = {
        "requirements.txt": "openai>=1\n",
        "app.py": '''
from openai import OpenAI
client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)
''',
    }
    providers = {signal.provider for signal in provider_signals(files)}
    assert providers == {"OpenRouter"}
    candidate_types = {
        item.candidate_type for item in unmapped_dependency_candidates(files)
    }
    assert "openai_compatible_provider_unresolved" not in candidate_types


def test_openai_official_endpoint_resolves_openai_service() -> None:
    files = {
        "requirements.txt": "openai>=1\n",
        "app.py": '''
from openai import OpenAI
client = OpenAI(base_url="https://api.openai.com/v1")
''',
    }
    providers = {signal.provider for signal in provider_signals(files)}
    assert providers == {"OpenAI"}
    candidate_types = {
        item.candidate_type for item in unmapped_dependency_candidates(files)
    }
    assert "openai_compatible_provider_unresolved" not in candidate_types


def test_environment_example_is_medium_configuration_evidence() -> None:
    signals = provider_signals({"config/.env.example": "GROQ_API_KEY=replace-me\n"})
    assert len(signals) == 1
    assert signals[0].provider == "Groq"
    assert signals[0].evidence_type == "configuration_signature"
    assert signals[0].confidence == "medium"


def test_unoverridden_openai_constructor_resolves_official_service() -> None:
    files = {
        "app.py": '''
from openai import OpenAI
client = OpenAI()
'''
    }
    assert {signal.provider for signal in provider_signals(files)} == {"OpenAI"}


def test_openai_constructor_with_indirect_base_url_stays_unresolved() -> None:
    files = {
        "app.py": '''
from openai import OpenAI
client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL"))
'''
    }
    assert provider_signals(files) == []
    assert {
        item.candidate_type for item in unmapped_dependency_candidates(files)
    } == {"openai_compatible_provider_unresolved"}


def test_select_repository_files_is_bounded_and_prefers_app() -> None:
    siblings = ["README.md", "requirements.txt", "src/main.py", "app.py"] + [
        f"src/module_{index}.py" for index in range(30)
    ]
    selected = select_repository_files(siblings, "src/main.py")
    assert "requirements.txt" in selected
    assert "src/main.py" in selected
    assert "app.py" in selected
    assert len(selected) <= 10


def test_declared_app_cannot_be_displaced_by_manifest_cap() -> None:
    siblings = [f"requirements-{index}.txt" for index in range(12)] + [
        "declared_app.py"
    ]
    selected = select_repository_files(siblings, "declared_app.py")
    assert selected[0] == "declared_app.py"
    assert len(selected) == 10


def test_repository_file_cap_can_be_increased_for_sensitivity_analysis() -> None:
    siblings = [f"module_{index}.py" for index in range(30)]
    selected = select_repository_files(siblings, None, max_files=20)
    assert len(selected) == 20


def test_extract_github_repositories() -> None:
    text = "Source: https://github.com/Example/TeacherBot.git and https://github.com/x/y)."
    assert extract_github_repositories(text) == ["Example/TeacherBot", "x/y"]


def test_extract_model_ids_only_in_loading_contexts() -> None:
    values = extract_code_model_ids(
        {
            "app.py": 'model = "Qwen/Qwen3-8B"\nAutoModel.from_pretrained("meta-llama/Llama-3")',
            "README.md": "google/gemma-3-4b",
        }
    )
    assert values == ["meta-llama/Llama-3", "Qwen/Qwen3-8B"]


def test_comments_docstrings_tests_and_checkpoints_cannot_create_signals() -> None:
    files = {
        "app.py": '''
"""from openai import OpenAI; model = "fake/docstring-model"""
# from anthropic import Anthropic
endpoint = "https://api.groq.com/openai/v1/chat/completions"
''',
        "tests/test_provider.py": "from openai import OpenAI\nmodel='fake/test-model'",
        ".ipynb_checkpoints/app-checkpoint.py": "from google import genai",
    }
    providers = {signal.provider for signal in provider_signals(files)}
    assert providers == {"Groq"}
    assert extract_code_model_ids(files) == []


def test_runtime_string_model_id_survives_docstring_filter() -> None:
    values = extract_code_model_ids(
        {
            "app.py": '''
def load():
    """Example: model = "fake/example"""
    return AutoModel.from_pretrained("meta-llama/Llama-3")
'''
        }
    )
    assert values == ["meta-llama/Llama-3"]


def test_unmapped_candidates_remain_separate_from_primary_edges() -> None:
    candidates = unmapped_dependency_candidates(
        {
            "requirements.txt": "openai\nunknown-ai-client\n",
            "app.py": '''
ACME_API_KEY = "placeholder"
base_url = "https://api.acme.ai/v1/chat/completions"
''',
        }
    )
    values = {(item.candidate_type, item.identifier) for item in candidates}
    assert ("unmapped_manifest_package", "unknown-ai-client") in values
    assert ("unmapped_credential", "ACME_API_KEY") in values
    assert ("unmapped_api_domain", "api.acme.ai") in values
    assert all(item.identifier != "openai" for item in candidates)


def test_placeholder_and_loopback_api_domains_are_not_service_candidates() -> None:
    candidates = unmapped_dependency_candidates(
        {
            "app.py": '''
docs = "https://api.example.com/v1/chat/completions"
local = "http://127.0.0.1:8000/v1/models"
'''
        }
    )
    assert not has_machine_service_candidate(candidates)
