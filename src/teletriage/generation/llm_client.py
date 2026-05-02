"""
Unified LLM client interface for the Generative Tier.

━━━ DESIGN DECISIONS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

(1) Should the generative tier accept retrieved context from RAG as grounding
    even when retrieval confidence was BELOW threshold — and why?

    YES.  The reranker threshold gates whether RAG can answer ALONE, not
    whether its candidates carry any useful signal.  A retrieval score of
    0.45 (below the 0.5 threshold) still means the cross-encoder thinks that
    document is plausibly related — it just isn't confident enough to surface
    it as the authoritative answer.  Passing that document as grounding context
    to the LLM is strictly better than generating from the LLM's training data
    alone, because:

      a) It anchors the LLM to the corpus's actual telecom terminology and
         vendor-specific procedures rather than generic internet knowledge.
      b) It dramatically reduces hallucinated specific commands or parameters,
         since the model sees a real example of the correct domain language.
      c) Cost is near-zero: the retrieval already ran, so we're just re-using
         the candidates that were computed but didn't cross the bar.

    The implementation contract: callers populate Query.metadata["retrieved_context"]
    with a list of {"question": str, "answer": str} dicts before calling
    GenerativeTier.answer().  The router (Phase 4+) should wire this up.
    When the field is absent the tier generates from scratch — graceful degradation.

    Token budget:  context is truncated to MAX_CONTEXT_CHARS characters
    (roughly 500 tokens) so it never crowds out the user query or the
    model's instruction following.

(2) What confidence value should we report for generative output, and what
    are the tradeoffs of each approach?

    We use FIXED 0.60 for Phase 3.  The options and their tradeoffs:

    Option A — Fixed value (0.60, what we implement):
      Honest: "I generated this from scratch; I don't know if it's right."
      Tells the router the answer exists but shouldn't be treated as ground truth.
      Simple.  Easy to calibrate later by comparing to human-verified answers.
      Downside: no discrimination between a well-grounded answer (LLM had good
      retrieved context) and a hallucination (LLM was flying blind).

    Option B — Token log-probabilities:
      Groq and Gemini expose per-token logprobs.  Average them and exponentiate.
      Problem: high logprob means the model is confident in its generation,
      NOT that the answer is correct.  A confidently wrong answer (e.g. a
      plausible but fabricated vendor CLI command) scores just as high as a
      correct one.  Misleading — worse than a fixed value.

    Option C — Self-consistency (generate N times, measure agreement):
      Generate 3-5 responses, embed them, measure pairwise cosine similarity.
      High agreement → higher confidence.  Works well empirically.
      Downside: 3-5× API cost and latency.  Not acceptable for interactive use.
      Candidate for Phase 4 offline evaluation, not Phase 3 real-time.

    Option D — LLM-as-judge:
      Second model call: "Rate the correctness of this answer 1-5."
      Surprisingly effective proxy for accuracy.  Adds ~200 ms latency.
      Susceptible to the judge model's own biases.  Viable in Phase 4.

    Upgrade path: confidence can be updated post-generation in details dict;
    Phase 4 evaluation harness can learn a calibration mapping from fixed
    0.60 to per-query estimates using BERTScore against a reference set.

━━━ CONCEPTS IN THIS FILE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Temperature and why 0.2 for fault triage:
  Temperature scales the logit distribution before sampling:
      p(token) ∝ exp(logit / T)
  T=1.0 → unmodified distribution (default).
  T→0   → deterministic argmax (always the most probable token).
  T→∞   → uniform distribution (random token).
  For troubleshooting, we want CONSISTENT, REPRODUCIBLE advice.
  An engineer asking the same question twice should get the same steps.
  T=0.2 sharpens the distribution without fully eliminating variance
  (T=0 can trap the model in repetitive loops on some architectures).

System vs user prompts:
  System prompt sets ROLE and FORMAT — it's the permanent context.
  User prompt carries the QUERY — it changes every call.
  Modern instruction-tuned models (Llama, Gemini) were trained with
  this structure; respecting it gives better instruction-following.

Retry logic — exponential backoff:
  Transient failures (5xx, connection resets) are common on free API tiers.
  Naive immediate retry hammers the server and burns rate-limit quota.
  Exponential backoff: wait 1s, 2s, 4s... caps at 10s.  Jitter prevents
  thundering-herd when multiple clients retry simultaneously.

Backend fallback chain:
  Groq → Gemini → LocalQwen.  FallbackClient tries each in order.
  Any exception from a backend → log it and try the next one.
  LocalQwen always succeeds (if the model file is downloaded), so it
  acts as the ultimate safety net.  The tradeoff: 1.5B parameter model
  vs 70B — quality will be lower, but an answer is better than a crash.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# Maximum characters of retrieved context to embed in the user prompt.
# ~500 tokens at ~4 chars/token.  Leaves ~1500 tokens for the model's
# own reasoning in a 2048-token context window.
MAX_CONTEXT_CHARS = 2000

# Fixed confidence for all generative output in Phase 3.
# See design decision (2) above for tradeoffs.
GENERATIVE_CONFIDENCE = 0.60


# ─── Output type ────────────────────────────────────────────────────────────


@dataclass
class GenerationOutput:
    """
    Structured output from any LLM backend.

    Keeping this as a plain dataclass (not Pydantic) because it's an
    internal value object — it never crosses a serialisation boundary.
    """
    text: str
    tokens_used: int
    finish_reason: str      # "stop", "length", or backend-specific value
    model: str              # model name as reported by the backend
    backend: str            # "groq", "gemini", or "local"
    raw_response: Any = field(default=None, repr=False)  # for debugging


# ─── Protocol ────────────────────────────────────────────────────────────────


@runtime_checkable
class LLMClient(Protocol):
    """
    Minimal interface every backend must satisfy.

    Runtime-checkable so we can assert isinstance(client, LLMClient) in tests
    without coupling to a concrete class.
    """

    def generate(
        self,
        system: str,
        user: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> GenerationOutput: ...


# ─── Exceptions ──────────────────────────────────────────────────────────────


class BackendError(Exception):
    """Base for all LLM backend failures."""


class BackendUnavailableError(BackendError):
    """Raised when a backend cannot be reached or has no valid credentials."""


class AllBackendsFailedError(BackendError):
    """Raised when every backend in the fallback chain has been exhausted."""


# ─── Groq backend ────────────────────────────────────────────────────────────


class GroqClient:
    """
    Groq Cloud client — Llama 3.3 70B on dedicated inference hardware.

    Free tier: 30 req/min, 500k tokens/day on llama-3.3-70b-versatile.
    Fastest inference of the three backends (~200–400 ms for 512 tokens).

    Retry policy:
      - InternalServerError / APIConnectionError / APITimeoutError:
        transient — retry up to MAX_RETRIES times with exponential backoff.
      - RateLimitError: not transient — raise immediately so FallbackClient
        can switch to the next backend.
      - AuthenticationError: misconfigured key — raise immediately.
    """

    MAX_RETRIES = 3
    BASE_WAIT_SEC = 1.0

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile") -> None:
        if not api_key:
            raise BackendUnavailableError("GroqClient: api_key is empty")
        from groq import Groq
        self._client = Groq(api_key=api_key)
        self._model = model

    def generate(
        self,
        system: str,
        user: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> GenerationOutput:
        from groq import (
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            RateLimitError,
        )

        last_exc: Exception | None = None
        for attempt in range(self.MAX_RETRIES):
            try:
                completion = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return GenerationOutput(
                    text=completion.choices[0].message.content or "",
                    tokens_used=completion.usage.total_tokens if completion.usage else 0,
                    finish_reason=completion.choices[0].finish_reason or "unknown",
                    model=self._model,
                    backend="groq",
                    raw_response=completion,
                )
            except RateLimitError as exc:
                # Rate limit → don't retry this backend, let FallbackClient switch
                raise BackendUnavailableError(f"Groq rate limit: {exc}") from exc
            except (InternalServerError, APIConnectionError, APITimeoutError) as exc:
                last_exc = exc
                wait = self.BASE_WAIT_SEC * (2 ** attempt)
                logger.warning(
                    "Groq transient error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, self.MAX_RETRIES, wait, exc,
                )
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(wait)

        raise BackendUnavailableError(
            f"Groq failed after {self.MAX_RETRIES} attempts: {last_exc}"
        ) from last_exc


# ─── Gemini backend ───────────────────────────────────────────────────────────


class GeminiClient:
    """
    Google Gemini 2.0 Flash via the google-genai SDK (v1+).

    Free tier: 15 req/min, 1M tokens/day on gemini-2.0-flash-exp.
    Uses the new google.genai package (google-generativeai is deprecated).

    Retry policy: ServerError (5xx) → retry with backoff.
    ClientError (4xx including quota) → raise immediately for fallback.
    """

    MAX_RETRIES = 3
    BASE_WAIT_SEC = 1.0

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-exp") -> None:
        if not api_key:
            raise BackendUnavailableError("GeminiClient: api_key is empty")
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def generate(
        self,
        system: str,
        user: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> GenerationOutput:
        from google import genai as _genai
        from google.genai import errors as gerrors
        from google.genai import types as gtypes

        del _genai  # only imported to satisfy type checker; client already built

        last_exc: Exception | None = None
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=user,
                    config=gtypes.GenerateContentConfig(
                        system_instruction=system,
                        max_output_tokens=max_tokens,
                        temperature=temperature,
                    ),
                )
                text = response.text or ""
                tokens = (
                    response.usage_metadata.total_token_count
                    if response.usage_metadata
                    else 0
                )
                finish = (
                    response.candidates[0].finish_reason.name
                    if response.candidates
                    else "unknown"
                )
                return GenerationOutput(
                    text=text,
                    tokens_used=tokens,
                    finish_reason=finish,
                    model=self._model,
                    backend="gemini",
                    raw_response=response,
                )
            except gerrors.ClientError as exc:
                # 4xx — quota, auth, bad request — don't retry this backend
                raise BackendUnavailableError(f"Gemini client error: {exc}") from exc
            except gerrors.ServerError as exc:
                last_exc = exc
                wait = self.BASE_WAIT_SEC * (2 ** attempt)
                logger.warning(
                    "Gemini server error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, self.MAX_RETRIES, wait, exc,
                )
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(wait)

        raise BackendUnavailableError(
            f"Gemini failed after {self.MAX_RETRIES} attempts: {last_exc}"
        ) from last_exc


# ─── Local Qwen backend ───────────────────────────────────────────────────────


class LocalQwenClient:
    """
    Local Qwen2.5-1.5B-Instruct via HuggingFace transformers.

    No API key required — runs entirely on-device.  Model weights (~3 GB)
    are downloaded from HuggingFace Hub on first use and cached locally.

    The model is loaded LAZILY (not at __init__ time) to avoid a 3-second
    startup cost when the cloud backends are available.  Once loaded, the
    pipeline object is kept alive for the duration of the process.

    Quality note:
      1.5B parameters vs 70B (Groq Llama) — noticeably lower quality for
      complex multi-step troubleshooting.  This tier exists for offline use
      or when all cloud backends fail, not for primary production traffic.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct") -> None:
        self._model_name = model_name
        self._pipeline: Any = None  # loaded on first call

    def _load_pipeline(self) -> None:
        """Download (if needed) and load the model into a text-generation pipeline."""
        import torch
        from transformers import pipeline

        logger.info("Loading local model %s (may take a few seconds)…", self._model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._pipeline = pipeline(
            "text-generation",
            model=self._model_name,
            device_map=device,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        logger.info("Local model loaded on %s.", device)

    def generate(
        self,
        system: str,
        user: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> GenerationOutput:
        if self._pipeline is None:
            self._load_pipeline()

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        # Qwen2.5-Instruct supports the chat template natively via pipeline
        result = self._pipeline(
            messages,
            max_new_tokens=max_tokens,
            temperature=max(temperature, 1e-4),  # pipeline rejects exactly 0
            do_sample=temperature > 0.05,
        )

        # pipeline returns: [{"generated_text": [{"role": ..., "content": ...}, ...]}]
        generated = result[0]["generated_text"]
        # The last message is the assistant's response
        assistant_turn = generated[-1]
        text = (
            assistant_turn["content"]
            if isinstance(assistant_turn, dict)
            else str(assistant_turn)
        )

        return GenerationOutput(
            text=text,
            tokens_used=len(text.split()),  # approximation — no token count from pipeline
            finish_reason="stop",
            model=self._model_name,
            backend="local",
        )


# ─── Fallback orchestrator ────────────────────────────────────────────────────


class FallbackClient:
    """
    Tries a prioritised list of LLM backends in order.

    If a backend raises any exception (including BackendUnavailableError),
    the error is logged and the next backend is attempted.  If all backends
    fail, AllBackendsFailedError is raised with the full failure log.

    This is the ONLY client that GenerativeTier should instantiate directly.
    Individual backend clients are used for unit testing in isolation.
    """

    def __init__(self, clients: list[LLMClient]) -> None:
        if not clients:
            raise ValueError("FallbackClient requires at least one backend client.")
        self._clients = clients

    def generate(
        self,
        system: str,
        user: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> GenerationOutput:
        failures: list[str] = []

        for client in self._clients:
            try:
                output = client.generate(
                    system=system,
                    user=user,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                if failures:
                    # We succeeded on a fallback — log so ops knows the primary failed
                    logger.warning(
                        "Generative tier: primary backend(s) failed (%s); "
                        "answered via %s.",
                        "; ".join(failures),
                        output.backend,
                    )
                return output
            except Exception as exc:
                msg = f"{type(client).__name__}: {exc}"
                failures.append(msg)
                logger.warning("Backend failed, trying next: %s", msg)

        raise AllBackendsFailedError(
            f"All {len(self._clients)} LLM backend(s) failed:\n"
            + "\n".join(f"  {f}" for f in failures)
        )

    @property
    def backend_names(self) -> list[str]:
        return [type(c).__name__ for c in self._clients]


# ─── Factory ──────────────────────────────────────────────────────────────────


def make_llm_client(
    groq_api_key: str = "",
    google_api_key: str = "",
    groq_model: str = "llama-3.3-70b-versatile",
    gemini_model: str = "gemini-2.0-flash-exp",
    local_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
) -> FallbackClient:
    """
    Build a FallbackClient with all available backends in priority order:
      Groq (if groq_api_key set) → Gemini (if google_api_key set) → Local Qwen.

    LocalQwenClient is always included as the last-resort fallback.
    Cloud backends are only added when their API key is non-empty.

    Usage::

        from teletriage.config import get_config
        cfg = get_config()
        client = make_llm_client(
            groq_api_key=cfg.secrets.groq_api_key,
            google_api_key=cfg.secrets.google_api_key,
            groq_model=cfg.generative_tier.groq_model,
            gemini_model=cfg.generative_tier.gemini_model,
            local_model=cfg.generative_tier.local_model,
        )
    """
    clients: list[LLMClient] = []

    if groq_api_key:
        try:
            clients.append(GroqClient(api_key=groq_api_key, model=groq_model))
            logger.debug("Groq backend registered (model: %s).", groq_model)
        except Exception as exc:
            logger.warning("Could not initialise Groq backend: %s", exc)

    if google_api_key:
        try:
            clients.append(GeminiClient(api_key=google_api_key, model=gemini_model))
            logger.debug("Gemini backend registered (model: %s).", gemini_model)
        except Exception as exc:
            logger.warning("Could not initialise Gemini backend: %s", exc)

    # LocalQwen is always appended — no key required, just disk space
    clients.append(LocalQwenClient(model_name=local_model))
    logger.debug("LocalQwen backend registered (model: %s).", local_model)

    return FallbackClient(clients)


# ─── Context helpers ─────────────────────────────────────────────────────────


def build_user_prompt(query_text: str, retrieved_context: list[dict]) -> str:
    """
    Construct the user-turn prompt, optionally grounding it with retrieved docs.

    retrieved_context: list of {"question": str, "answer": str} dicts from
    RAG candidates that did not cross the confidence threshold but still carry
    relevant signal.  See design decision (1) above.

    Context is truncated to MAX_CONTEXT_CHARS to respect token budgets.
    """
    if not retrieved_context:
        return query_text

    context_parts: list[str] = []
    total_chars = 0
    for doc in retrieved_context:
        snippet = f"Q: {doc.get('question', '')}\nA: {doc.get('answer', '')}"
        if total_chars + len(snippet) > MAX_CONTEXT_CHARS:
            break
        context_parts.append(snippet)
        total_chars += len(snippet)

    if not context_parts:
        return query_text

    context_block = "\n\n---\n".join(context_parts)
    return (
        f"{query_text}\n\n"
        f"[Potentially relevant context — use as grounding, not as authoritative answer]\n"
        f"{context_block}"
    )
