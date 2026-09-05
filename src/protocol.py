"""Frozen discovery and classification rules for DE-004.

The rules in this module are intentionally deterministic. They identify a
bounded, query-defined public sample; they do not estimate the population of
all educational AI projects.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


PROTOCOL_VERSION = "2026-08-31.5"
SNAPSHOT_DATE = "2026-08-31"

# Two sorted arms are retained for every query: established/high-engagement
# projects and newly created projects. The union is de-duplicated by Space ID.
SEARCH_TERMS = (
    "ai tutor",
    "socratic tutor",
    "tutor chatbot",
    "education chatbot",
    "educational chatbot",
    "educational ai",
    "quiz generator",
    "course generator",
    "teaching assistant",
    "study assistant",
    "learning companion",
    "homework assistant",
    "lesson plan generator",
    "grading assistant",
    "language tutor",
)
SEARCH_ARMS = ("likes", "createdAt")
PER_QUERY_LIMIT = 100


@dataclass(frozen=True)
class PhraseRule:
    construct: str
    pattern: str


# "Tutor" is deliberately not expanded to "tutorial". Generic mentions of a
# student, course, or assessment are not sufficient for the strict frame.
STRICT_EDUCATION_RULES = (
    PhraseRule("tutor", r"\b(?:ai[- ]?)?tutors?\b"),
    PhraseRule("socratic_tutor", r"\bsocratic(?:[- ]+(?:ai[- ]?)?tutors?)?\b"),
    PhraseRule(
        "teaching_assistant",
        r"\b(?:ai[- ]?|virtual[- ]?)?(?:teaching|teacher|classroom)[- ]+assistants?\b",
    ),
    PhraseRule(
        "study_support",
        r"\b(?:ai[- ]?)?(?:study|learning)[- ]+(?:assistants?|budd(?:y|ies)|coaches?|companions?|partners?)\b",
    ),
    PhraseRule(
        "education_app",
        r"\b(?:educational?|education)[- ]+(?:ai|apps?|assistants?|budd(?:y|ies)|chatbots?|companions?|platforms?|tools?)\b",
    ),
    PhraseRule(
        "quiz_generation",
        r"\b(?:ai[- ]?)?(?:quiz|question)[- ]+(?:generators?|makers?|assistants?|tutors?|chatbots?)\b",
    ),
    PhraseRule(
        "course_generation",
        r"\b(?:ai[- ]?)?(?:course|curriculum|syllabus)[- ]+(?:creators?|generators?|assistants?|planners?|tutors?)\b",
    ),
    PhraseRule(
        "lesson_planning",
        r"\b(?:ai[- ]?)?(?:lesson[- ]+plans?|lesson[- ]+planning)[- ]+(?:assistants?|generators?|tools?)\b",
    ),
    PhraseRule(
        "homework_support",
        r"\b(?:ai[- ]?)?homework[- ]+(?:assistants?|helpers?|tutors?)\b",
    ),
    PhraseRule(
        "grading_support",
        r"\b(?:ai[- ]?)?(?:grading|essay[- ]+grading|assessment)[- ]+(?:assistants?|graders?|feedback|generators?)\b",
    ),
    PhraseRule(
        "study_material_generation",
        r"\b(?:ai[- ]?)?(?:flashcard|worksheet|study[- ]+guide)s?[- ]+(?:creators?|generators?|makers?)\b",
    ),
    PhraseRule(
        "exam_support",
        r"\b(?:exam|test)[- ]+(?:prep(?:aration)?|assistants?|tutors?|coaches?)\b",
    ),
    PhraseRule(
        "language_learning",
        r"\b(?:language|english|spanish|french|german|chinese|japanese|korean)[- ]+(?:learning[- ]+)?(?:tutors?|coaches?|partners?|companions?)\b",
    ),
)

BROAD_EDUCATION_RULES = STRICT_EDUCATION_RULES + (
    PhraseRule("generic_education", r"\b(?:education|educational|edtech)\b"),
    PhraseRule("generic_quiz", r"\bquiz(?:zes|zer)?\b"),
    PhraseRule("generic_school", r"\b(?:school|classroom|teacher|student|learner)s?\b"),
)

# Terms that can create false positives even inside targeted discovery queries.
# They do not exclude an item if another strict educational construct matches.
NON_EDUCATION_ASSESSMENT = (
    r"\benvironmental[- ]+impact[- ]+assessment\b",
    r"\bhealth[- ]+(?:risk[- ]+)?assessment\b",
    r"\bsecurity[- ]+assessment\b",
    r"\bfinancial[- ]+assessment\b",
)

TEXT_FILE_BASENAMES = {
    "requirements.txt",
    "requirements-dev.txt",
    "pyproject.toml",
    "package.json",
    "environment.yml",
    "environment.yaml",
    "pipfile",
    "setup.py",
    "setup.cfg",
    ".env.example",
    "readme.md",
    "license",
    "license.txt",
}
CODE_SUFFIXES = {".py", ".js", ".jsx", ".ts", ".tsx", ".html", ".ini"}
MAX_SOURCE_FILES = 10
MAX_SOURCE_BYTES = 512_000


@dataclass(frozen=True)
class ProviderRule:
    provider: str
    layer: str


PACKAGE_PROVIDER_RULES: dict[str, ProviderRule] = {
    # The OpenAI Python and LangChain clients support third-party ``base_url``
    # endpoints. They are inventoried as unresolved OpenAI-compatible clients
    # unless provider-specific code evidence resolves the service.
    "anthropic": ProviderRule("Anthropic", "inference_service"),
    "langchain-anthropic": ProviderRule("Anthropic", "inference_service"),
    "google-generativeai": ProviderRule("Google", "inference_service"),
    "google-genai": ProviderRule("Google", "inference_service"),
    "langchain-google-genai": ProviderRule("Google", "inference_service"),
    "mistralai": ProviderRule("Mistral AI", "inference_service"),
    "cohere": ProviderRule("Cohere", "inference_service"),
    "groq": ProviderRule("Groq", "inference_service"),
    "together": ProviderRule("Together AI", "inference_service"),
    "together-python": ProviderRule("Together AI", "inference_service"),
    "replicate": ProviderRule("Replicate", "inference_service"),
    "openrouter": ProviderRule("OpenRouter", "inference_service"),
    "fireworks-ai": ProviderRule("Fireworks AI", "inference_service"),
    "dashscope": ProviderRule("Alibaba Cloud", "inference_service"),
    "zhipuai": ProviderRule("Zhipu AI", "inference_service"),
    "qianfan": ProviderRule("Baidu", "inference_service"),
    "boto3": ProviderRule("Amazon Web Services", "cloud_sdk"),
    "azure-ai-inference": ProviderRule("Microsoft Azure", "inference_service"),
    "ollama": ProviderRule("User-managed runtime", "local_runtime"),
    "llama-cpp-python": ProviderRule("User-managed runtime", "local_runtime"),
    "transformers": ProviderRule("User-managed runtime", "local_runtime"),
    "vllm": ProviderRule("User-managed runtime", "local_runtime"),
    "huggingface-hub": ProviderRule("Hugging Face", "hub_sdk"),
}

# Code signatures require an import, credential name, or endpoint. Plain prose
# mentions are not counted as dependencies.
CODE_PROVIDER_PATTERNS: tuple[tuple[ProviderRule, str, str], ...] = (
    (ProviderRule("OpenAI", "inference_service"), "openai_official", r"\bOPENAI_API_KEY\b|api\.openai\.com"),
    (ProviderRule("Anthropic", "inference_service"), "anthropic_import", r"(?:from|import)\s+anthropic\b|\bANTHROPIC_API_KEY\b|api\.anthropic\.com"),
    (ProviderRule("Google", "inference_service"), "gemini_sdk", r"google\.generativeai|google\.genai|\bGEMINI_API_KEY\b|generativelanguage\.googleapis\.com"),
    (ProviderRule("Mistral AI", "inference_service"), "mistral_sdk", r"(?:from|import)\s+mistralai\b|\bMISTRAL_API_KEY\b"),
    (ProviderRule("Cohere", "inference_service"), "cohere_sdk", r"(?:from|import)\s+cohere\b|\bCOHERE_API_KEY\b"),
    (ProviderRule("Groq", "inference_service"), "groq_sdk", r"(?:from|import)\s+groq\b|\bGROQ_API_KEY\b|api\.groq\.com"),
    (ProviderRule("Together AI", "inference_service"), "together_sdk", r"(?:from|import)\s+together\b|\bTOGETHER_API_KEY\b|api\.together\.xyz"),
    (ProviderRule("Replicate", "inference_service"), "replicate_sdk", r"(?:from|import)\s+replicate\b|\bREPLICATE_API_TOKEN\b"),
    (ProviderRule("OpenRouter", "inference_service"), "openrouter_endpoint", r"\bOPENROUTER_API_KEY\b|openrouter\.ai/api"),
    (ProviderRule("Fireworks AI", "inference_service"), "fireworks_endpoint", r"\bFIREWORKS_API_KEY\b|api\.fireworks\.ai"),
    (ProviderRule("Alibaba Cloud", "inference_service"), "dashscope_sdk", r"(?:from|import)\s+dashscope\b|\bDASHSCOPE_API_KEY\b"),
    (ProviderRule("DeepSeek", "inference_service"), "deepseek_endpoint", r"\bDEEPSEEK_API_KEY\b|api\.deepseek\.com"),
    (ProviderRule("Zhipu AI", "inference_service"), "zhipu_sdk", r"(?:from|import)\s+zhipuai\b|\bZHIPUAI_API_KEY\b"),
    (ProviderRule("Baidu", "inference_service"), "qianfan_sdk", r"(?:from|import)\s+qianfan\b|\bQIANFAN_(?:AK|SK)\b"),
    (ProviderRule("Microsoft Azure", "inference_service"), "azure_openai", r"\bAZURE_OPENAI_(?:API_KEY|ENDPOINT)\b|\.openai\.azure\.com"),
    (ProviderRule("Hugging Face", "inference_service"), "hf_inference", r"\bInferenceClient\b|api-inference\.huggingface\.co|router\.huggingface\.co"),
    (ProviderRule("User-managed runtime", "local_runtime"), "transformers_runtime", r"(?:from|import)\s+transformers\b|\btransformers\.pipeline\b"),
    (ProviderRule("User-managed runtime", "local_runtime"), "ollama_runtime", r"(?:from|import)\s+ollama\b|localhost:11434"),
)


# Ordered rules map public model namespaces to the organization that publishes
# the upstream family. Unmapped namespaces remain visible rather than being
# silently assigned to a company.
MODEL_NAMESPACE_PROVIDERS: tuple[tuple[str, str], ...] = (
    ("meta-llama", "Meta"),
    ("google", "Google"),
    ("mistralai", "Mistral AI"),
    ("microsoft", "Microsoft"),
    ("openai", "OpenAI"),
    ("anthropic", "Anthropic"),
    ("qwen", "Alibaba"),
    ("deepseek-ai", "DeepSeek"),
    ("zai-org", "Z.ai"),
    ("thudm", "Tsinghua THUDM"),
    ("baai", "BAAI"),
    ("internlm", "Shanghai AI Laboratory"),
    ("01-ai", "01.AI"),
    ("baichuan-inc", "Baichuan"),
    ("moonshotai", "Moonshot AI"),
    ("tencent", "Tencent"),
    ("bytedance", "ByteDance"),
    ("cohereforai", "Cohere"),
    ("tiiuae", "TII"),
    ("bigscience", "BigScience"),
    ("huggingfaceh4", "Hugging Face"),
    ("eleutherai", "EleutherAI"),
    ("stabilityai", "Stability AI"),
    ("black-forest-labs", "Black Forest Labs"),
    ("naver", "NAVER"),
    ("kakaobrain", "Kakao Brain"),
    ("lgai-exaone", "LG AI Research"),
    ("sakanaai", "Sakana AI"),
    ("rinna", "rinna"),
    ("cyberagent", "CyberAgent"),
)

ASIA_MODEL_PROVIDERS = {
    "Alibaba",
    "DeepSeek",
    "Z.ai",
    "Tsinghua THUDM",
    "BAAI",
    "Shanghai AI Laboratory",
    "01.AI",
    "Baichuan",
    "Moonshot AI",
    "Tencent",
    "ByteDance",
    "NAVER",
    "Kakao Brain",
    "LG AI Research",
    "Sakana AI",
    "rinna",
    "CyberAgent",
}

ASIA_LANGUAGE_CODES = {
    "ar", "bn", "fa", "hi", "id", "ja", "jv", "km", "ko", "lo",
    "ms", "my", "ne", "pa", "si", "ta", "te", "th", "tl", "tr",
    "ur", "uz", "vi", "zh",
}

# ISO alpha-2 codes for countries/areas conventionally grouped in Asia by UN
# M49, retained locally so no location is inferred from a hosting-region tag.
ASIA_ALPHA2 = {
    "AF", "AM", "AZ", "BH", "BD", "BT", "BN", "KH", "CN", "CY", "GE",
    "IN", "ID", "IR", "IQ", "IL", "JP", "JO", "KZ", "KW", "KG", "LA",
    "LB", "MY", "MV", "MN", "MM", "NP", "KP", "OM", "PK", "PS", "PH",
    "QA", "SA", "SG", "KR", "LK", "SY", "TW", "TJ", "TH", "TL", "TR",
    "TM", "AE", "UZ", "VN", "YE",
}


def normalize_search_text(value: str) -> str:
    """Normalize repository prose for deterministic phrase matching."""

    # Repository slugs and card titles often concatenate functional phrases
    # (for example, ``StudyAssistant`` or ``LessonPlanGenerator``).  Split
    # lower-to-upper and acronym-to-word boundaries before case folding so the
    # same phrase rules apply to CamelCase and space-delimited metadata.
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(value))
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", text)
    text = text.casefold().replace("_", " ").replace("/", " ")
    text = re.sub(r"[^\w\-+. ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def match_education_constructs(value: str, *, broad: bool = False) -> list[str]:
    """Return matched educational-function constructs."""

    text = normalize_search_text(value)
    rules = BROAD_EDUCATION_RULES if broad else STRICT_EDUCATION_RULES
    matches = [rule.construct for rule in rules if re.search(rule.pattern, text)]
    if (
        matches
        and set(matches) <= {"grading_support"}
        and any(re.search(pattern, text) for pattern in NON_EDUCATION_ASSESSMENT)
    ):
        return []
    return sorted(set(matches))


def model_namespace(model_id: str) -> str:
    return model_id.split("/", 1)[0].strip() if "/" in model_id else "unscoped"


def model_provider(model_id: str, base_models: Iterable[str] = ()) -> tuple[str, str]:
    """Resolve a provider from a linked model or a declared base model.

    Returns ``(provider, basis)``. The fallback preserves the public namespace
    and is not treated as a manually verified corporate attribution.
    """

    candidates = [*base_models, model_id]
    for candidate in candidates:
        namespace = model_namespace(str(candidate)).casefold()
        for expected, provider in MODEL_NAMESPACE_PROVIDERS:
            if namespace == expected:
                basis = "base_model_namespace" if candidate != model_id else "model_namespace"
                return provider, basis
    namespace = model_namespace(model_id)
    return f"namespace:{namespace}", "unmapped_public_namespace"


def model_family(model_id: str, base_models: Iterable[str] = ()) -> str:
    """Map common public model names to transparent family labels."""

    joined = " ".join([*map(str, base_models), model_id]).casefold()
    rules = (
        (r"\bllama[- _]?\d*|meta-llama", "Llama"),
        (r"\bqwen(?:\d|[- _])", "Qwen"),
        (r"\bgemma(?:\d|[- _])", "Gemma"),
        (r"\b(?:mixtral|mistral)(?:[- _]|\d)", "Mistral/Mixtral"),
        (r"\bphi[- _]?\d", "Phi"),
        (r"\bdeepseek", "DeepSeek"),
        (r"\b(?:flan[- _]?)?t5\b", "T5/FLAN-T5"),
        (r"\broberta\b", "RoBERTa"),
        (r"\bbert\b", "BERT"),
        (r"\bbloom", "BLOOM"),
        (r"\bfalcon", "Falcon"),
        (r"\bwhisper", "Whisper"),
        (r"\bstable[- _]diffusion|\bsd(?:xl|3)[- _]", "Stable Diffusion"),
    )
    for pattern, family in rules:
        if re.search(pattern, joined):
            return family
    chosen = next(iter(base_models), model_id)
    name = str(chosen).split("/", 1)[-1]
    return re.split(r"[-_]", name, maxsplit=1)[0] or "unresolved"
