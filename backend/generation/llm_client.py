"""
Unified LLM client interface for the Generative Tier.

Design notes:
  - Below-threshold RAG context is still passed to the generative tier as
    grounding (Query.metadata["retrieved_context"]) — a reranker score
    too low to answer alone can still anchor the LLM to real corpus
    terminology and reduce hallucinated specifics, at near-zero extra
    cost since retrieval already ran. See build_user_prompt().
  - Confidence is a fixed 0.60 for all generative output: an honest
    "generated, not verified" signal. Token logprobs measure confidence,
    not correctness; self-consistency/LLM-as-judge approaches cost 3-5x
    latency, too slow for interactive use.
  - Temperature 0.2 (not 0): sharpens the output distribution for
    consistent, reproducible troubleshooting steps without the
    repetitive-loop failure mode T=0 can trigger on some models.
  - GroqClient and GeminiClient share the same retry shape: MAX_RETRIES=3,
    wait = BASE_WAIT_SEC * 2**attempt (1s, then 2s). The final attempt
    never sleeps before raising — no retry follows it. No jitter, no cap.
  - Backend fallback order is per-tier (not one fixed chain) — see
    make_llm_client()'s `backend_order` param.
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

# Fixed confidence for all generative output — see module docstring.
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
    # Retries this call needed before succeeding (0 = first attempt).
    # Always 0 for Gemini/local — only GroqClient's retry loop counts it.
    retry_count: int = 0
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

    Free tier: 30 req/min, 500k tokens/day, 12,000 TPM on
    llama-3.3-70b-versatile. Fastest inference of the three backends
    (~200-400ms for 512 tokens).

    Retry policy:
      - InternalServerError / APIConnectionError / APITimeoutError:
        transient — retry up to MAX_RETRIES times with exponential backoff.
      - RateLimitError: not transient — raise immediately so FallbackClient
        can switch to the next backend.
      - AuthenticationError: misconfigured key — raise immediately.

    Constructed with max_retries=0 (below) to disable the `groq` SDK's
    own hidden transport-level retry, which otherwise silently retries
    429s before this class's retry loop ever sees, logs, or counts them —
    every retry that happens is this class's own, in generate() below.
    """

    MAX_RETRIES = 3
    BASE_WAIT_SEC = 1.0

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile") -> None:
        if not api_key:
            raise BackendUnavailableError("GroqClient: api_key is empty")
        from groq import Groq
        self._client = Groq(api_key=api_key, max_retries=0)  # see class docstring
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
                if attempt > 0:
                    logger.warning(
                        "Groq call succeeded after %d retry/retries (model=%s).",
                        attempt, self._model,
                    )
                return GenerationOutput(
                    text=completion.choices[0].message.content or "",
                    tokens_used=completion.usage.total_tokens if completion.usage else 0,
                    finish_reason=completion.choices[0].finish_reason or "unknown",
                    model=self._model,
                    backend="groq",
                    retry_count=attempt,
                    raw_response=completion,
                )
            except RateLimitError as exc:
                # Rate limit → not transient for THIS backend, let
                # FallbackClient switch instead of retrying in place.
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
    Google Gemini (flash tier) via the google-genai SDK (v1+).

    Uses gemini-flash-latest — a Google-managed alias, not a pinned
    version. Pinned Gemini model names on free-tier accounts have gone
    stale (404) more than once; the alias avoids re-hitting that at the
    cost of behavior possibly drifting if Google repoints it.

    Retry policy: ServerError (5xx) → retry with backoff.
    ClientError (4xx including quota) → raise immediately for fallback.
    """

    MAX_RETRIES = 3
    BASE_WAIT_SEC = 1.0

    def __init__(self, api_key: str, model: str = "gemini-flash-latest") -> None:
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
                        # Newer Gemini models spend part of max_output_tokens
                        # on internal "thinking" before the visible answer —
                        # at default thinking level this can consume the
                        # whole budget and truncate the real answer to
                        # nothing. MINIMAL keeps enough budget free to
                        # actually answer.
                        thinking_config=gtypes.ThinkingConfig(
                            thinking_level=gtypes.ThinkingLevel.MINIMAL
                        ),
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

# Generous timeout: CPU inference plus a cold model load every call (see
# LocalQwenClient docstring) can genuinely take tens of seconds.
# Configurable per-instance via LocalQwenClient(timeout_sec=...).
_LOCAL_QWEN_TIMEOUT_SEC = 90.0

# Inline script run via `sys.executable -c` in an isolated subprocess —
# same stdin-JSON-in/stdout-JSON-out shape as evaluator.py's
# _bertscore_subprocess(), the one other place this codebase isolates
# model inference this way.
_LOCAL_QWEN_WORKER_SCRIPT = """
import json
import sys

import torch
torch.multiprocessing.set_sharing_strategy("file_system")
from transformers import pipeline

d = json.loads(sys.stdin.read())

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipeline(
    "text-generation",
    model=d["model_name"],
    device_map=device,
    dtype=torch.float16 if device == "cuda" else torch.float32,
)

messages = [
    {"role": "system", "content": d["system"]},
    {"role": "user", "content": d["user"]},
]

temperature = d["temperature"]
result = pipe(
    messages,
    max_new_tokens=d["max_tokens"],
    temperature=max(temperature, 1e-4),  # pipeline rejects exactly 0
    do_sample=temperature > 0.05,
)

# pipeline returns: [{"generated_text": [{"role": ..., "content": ...}, ...]}]
generated = result[0]["generated_text"]
assistant_turn = generated[-1]
text = assistant_turn["content"] if isinstance(assistant_turn, dict) else str(assistant_turn)

print(json.dumps({"text": text}))
"""


class LocalQwenClient:
    """
    Local Qwen2.5-1.5B-Instruct via HuggingFace transformers, run in an
    ISOLATED SUBPROCESS for every call — never loaded in this process.

    Why: this backend is only reached after Groq AND Gemini have both
    failed, which is exactly when tier 2's BGE-small embedder and
    bge-reranker-base cross-encoder are typically already loaded in this
    same process. A third PyTorch model coexisting with them there has
    caused a real SIGSEGV crash — the same class of conflict
    evaluator.py's _bertscore_subprocess() isolates against. Subprocess
    isolation is the fix; don't revert to a persistent in-process
    pipeline without re-solving that crash first.

    Tradeoff: every call pays real process-spawn + cold-model-load
    overhead (~1-3s) since nothing persists between calls — acceptable
    because this only fires as a last resort, not on the hot path.

    No API key required — runs entirely on-device. Model weights (~3 GB)
    download from HuggingFace Hub on first use and cache locally.

    Quality note: 1.5B params vs 70B (Groq) — noticeably weaker on
    complex multi-step troubleshooting. Last-resort fallback, not
    primary traffic.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        timeout_sec: float = _LOCAL_QWEN_TIMEOUT_SEC,
    ) -> None:
        self._model_name = model_name
        self._timeout_sec = timeout_sec

    def generate(
        self,
        system: str,
        user: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> GenerationOutput:
        import json
        import subprocess
        import sys

        data = {
            "model_name": self._model_name,
            "system": system,
            "user": user,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Mapped to BackendUnavailableError (same as Groq/Gemini) so the
        # subprocess boundary is invisible to FallbackClient.
        try:
            result = subprocess.run(
                [sys.executable, "-c", _LOCAL_QWEN_WORKER_SCRIPT],
                input=json.dumps(data),
                capture_output=True,
                text=True,
                timeout=self._timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            raise BackendUnavailableError(
                f"LocalQwen: subprocess timed out after {self._timeout_sec}s"
            ) from exc

        if result.returncode != 0:
            raise BackendUnavailableError(
                f"LocalQwen: subprocess exited {result.returncode}: "
                f"{result.stderr.strip()[-2000:]}"
            )

        try:
            parsed = json.loads(result.stdout.strip())
            text = parsed["text"]
        except (json.JSONDecodeError, KeyError) as exc:
            raise BackendUnavailableError(
                f"LocalQwen: subprocess produced malformed output: {exc}"
            ) from exc

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

    This is used by BOTH tier 2 (retrieval_tier.py's RAG synthesis) and
    tier 3 (generative_tier.py) — each with its own instance and its own
    backend_order (see make_llm_client()'s docstring for why they're kept
    separate). Individual backend clients are used for unit testing in
    isolation.
    """

    def __init__(self, clients: list[LLMClient], name: str = "llm") -> None:
        if not clients:
            raise ValueError("FallbackClient requires at least one backend client.")
        self._clients = clients
        # Identifies which tier's chain this is in logs, e.g.
        # "tier2-synthesis" vs "tier3-generative".
        self._name = name

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
                        "%s: primary backend(s) failed (%s); answered via %s.",
                        self._name,
                        "; ".join(failures),
                        output.backend,
                    )
                return output
            except Exception as exc:
                msg = f"{type(client).__name__}: {exc}"
                failures.append(msg)
                logger.warning("%s: backend failed, trying next: %s", self._name, msg)

        raise AllBackendsFailedError(
            f"All {len(self._clients)} LLM backend(s) failed ({self._name}):\n"
            + "\n".join(f"  {f}" for f in failures)
        )

    @property
    def backend_names(self) -> list[str]:
        return [type(c).__name__ for c in self._clients]


# ─── Factory ──────────────────────────────────────────────────────────────────


_DEFAULT_BACKEND_ORDER = ("groq", "gemini", "local")


def make_llm_client(
    groq_api_key: str = "",
    google_api_key: str = "",
    groq_model: str = "llama-3.3-70b-versatile",
    gemini_model: str = "gemini-flash-latest",
    local_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    backend_order: list[str] | tuple[str, ...] = _DEFAULT_BACKEND_ORDER,
    client_name: str = "llm",
) -> FallbackClient:
    """
    Build a FallbackClient trying backends in `backend_order`, left to right.

    `backend_order` is authoritative for both which backends are included
    and the order they're tried — a name absent from the list is excluded
    even if its API key is set. Cloud backends are additionally skipped
    if their key is empty or construction fails; "local" needs no key and
    is only skipped by omitting it from `backend_order`.

    Tier 2 and tier 3 each call this with their OWN Groq API key
    (groq_api_key vs groq_api_key_tier3 — selected by the caller, not
    this function) so they draw from independent TPM pools instead of
    sharing one. See GenerativeTier.__init__ for the key-selection logic
    and config.yaml's backend_order comments for why this exists.

    client_name: identifies this chain in logs, e.g. "tier2-synthesis" vs
    "tier3-generative" (see FallbackClient).

    Usage (tier 3)::

        from backend.config import get_config
        cfg = get_config()
        client = make_llm_client(
            groq_api_key=cfg.secrets.groq_api_key_tier3 or cfg.secrets.groq_api_key,
            google_api_key=cfg.secrets.google_api_key,
            groq_model=cfg.generative_tier.groq_model,
            gemini_model=cfg.generative_tier.gemini_model,
            local_model=cfg.generative_tier.local_model,
            backend_order=cfg.generative_tier.backend_order,
            client_name="tier3-generative",
        )
    """
    available: dict[str, LLMClient] = {}

    if groq_api_key:
        try:
            available["groq"] = GroqClient(api_key=groq_api_key, model=groq_model)
        except Exception as exc:
            logger.warning("Could not initialise Groq backend: %s", exc)

    if google_api_key:
        try:
            available["gemini"] = GeminiClient(api_key=google_api_key, model=gemini_model)
        except Exception as exc:
            logger.warning("Could not initialise Gemini backend: %s", exc)

    # LocalQwen needs no key — only ever skipped by omitting it from
    # backend_order (see below).
    available["local"] = LocalQwenClient(model_name=local_model)

    clients: list[LLMClient] = []
    for backend_name in backend_order:
        client = available.pop(backend_name, None)
        if client is None:
            logger.warning(
                "%s: backend_order specifies %r but it's not available "
                "(missing API key, or already listed twice) — skipping.",
                client_name, backend_name,
            )
            continue
        clients.append(client)
        logger.debug(
            "%s: %s backend registered (position %d in backend_order).",
            client_name, backend_name, len(clients),
        )

    if not clients:
        raise ValueError(
            f"make_llm_client({client_name!r}): backend_order {list(backend_order)!r} "
            f"produced no usable backends — check API keys are set for at least one of them."
        )

    return FallbackClient(clients, name=client_name)


# ─── Context helpers ─────────────────────────────────────────────────────────


def build_user_prompt(query_text: str, retrieved_context: list[dict]) -> str:
    """
    Construct the user-turn prompt, optionally grounding it with retrieved docs.

    retrieved_context: list of dicts from RAG candidates that did not cross
    the confidence threshold but still carry relevant signal (see module
    docstring) — only "question"/"answer" are read; "doc_id"/"logit" are
    for eval/observability, not prompt construction.

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
