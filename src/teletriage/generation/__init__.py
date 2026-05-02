"""Generation backends (Phase 3): unified interface over Groq, Gemini, and local Qwen."""
from teletriage.generation.llm_client import (
    AllBackendsFailedError,
    BackendError,
    BackendUnavailableError,
    FallbackClient,
    GeminiClient,
    GenerationOutput,
    GroqClient,
    LLMClient,
    LocalQwenClient,
    build_user_prompt,
    make_llm_client,
)

__all__ = [
    "LLMClient",
    "GenerationOutput",
    "GroqClient",
    "GeminiClient",
    "LocalQwenClient",
    "FallbackClient",
    "make_llm_client",
    "build_user_prompt",
    "BackendError",
    "BackendUnavailableError",
    "AllBackendsFailedError",
]
