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

    WIRED (as of Phase 4): the implementation contract is callers populate
    Query.metadata["retrieved_context"] with a list of dicts (at minimum
    "question"/"answer" — build_user_prompt() reads only those two keys)
    before calling GenerativeTier.answer(). When the field is absent the
    tier generates from scratch — graceful degradation.

    How it's actually wired, end to end:
      1. backend/tiers/retrieval_tier.py reranks with top_k=3 (not just
         the top_1 it needs to answer with) and packages all 3 into
         TierResult.details["candidates"] as {"doc_id", "question",
         "answer", "logit"} dicts, per-answer capped at 800 chars — on
         BOTH the low-confidence delegation path and the confident-answer
         path (the latter purely for eval/observability; the router never
         gets a chance to forward it downstream in that case, since the
         cascade already stopped).
      2. backend/router.py's route() loop reads
         result.details.get("candidates") after each tier that delegates,
         accumulates it, and — because Query is a frozen Pydantic model
         and can't be mutated mid-cascade — constructs a NEW Query with
         metadata={**metadata, "retrieved_context": candidates} before
         calling the next tier. The ORIGINAL Query object (whatever the
         caller submitted) is preserved separately and is what ends up on
         the final Response — the decorated, mid-cascade version never
         escapes the router.
      3. GenerativeTier.answer() reads query.metadata.get("retrieved_context", [])
         (unchanged — this file's contract was already correct; only the
         producer side was missing) and passes it to build_user_prompt().

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
  Actual behaviour (GroqClient, GeminiClient — same MAX_RETRIES/
  BASE_WAIT_SEC pattern in both): MAX_RETRIES=3 attempts total, waiting
  BASE_WAIT_SEC * 2**attempt between them — 1s after the first failure,
  2s after the second. The sleep is guarded (`if attempt < MAX_RETRIES -
  1`) so the third, final attempt never sleeps before giving up — there's
  nothing left to retry into once the loop is about to exit, so a 4s wait
  there would just add latency with no retry to show for it. No jitter,
  no explicit time cap: the ceiling is MAX_RETRIES itself, not a duration.

Backend fallback chain:
  FallbackClient tries backends in whatever order make_llm_client()'s
  `backend_order` param specifies — see that function's docstring.
  NOT a single fixed Groq → Gemini → LocalQwen chain: tier 2 (RAG
  synthesis) and tier 3 (closed-book fallback) are configured with
  DIFFERENT primaries (config.yaml's retrieval_tier.backend_order vs
  generative_tier.backend_order) specifically so they draw from separate
  provider quotas instead of silently competing for one shared budget.
  Any exception from a backend → log it and try the next one in that
  tier's order. LocalQwen always succeeds (if the model file is
  downloaded), so as long as it's included in the order it acts as the
  ultimate safety net.  The tradeoff: 1.5B parameter model vs 70B —
  quality will be lower, but an answer is better than a crash.
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
    # Number of GroqClient-level retries this call needed before
    # succeeding (0 = succeeded on the first attempt). See GroqClient's
    # docstring — this used to be unmeasurable: the groq SDK's OWN
    # internal retry (its default max_retries=2) absorbed 429s silently,
    # so a call could take 20+ seconds with zero signal anywhere that a
    # retry happened. Always 0 for Gemini/local for now — only GroqClient
    # has ever exhibited this specific silent-retry behaviour.
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

    Free tier: 30 req/min, 500k tokens/day on llama-3.3-70b-versatile.
    Fastest inference of the three backends (~200–400 ms for 512 tokens).
    NOTE: other models on this same account can have MUCH lower limits —
    llama-3.1-8b-instant (tier 2's synthesis model) was measured at just
    6000 TPM (tokens per minute) via the x-ratelimit-limit-tokens response
    header. That's roughly 2-3 of tier 2's synthesis calls before hitting
    the ceiling, not the 30 req/min figure above (which is a per-model,
    per-request limit, not a token budget, and doesn't apply here).

    Retry policy:
      - InternalServerError / APIConnectionError / APITimeoutError:
        transient — retry up to MAX_RETRIES times with exponential backoff.
      - RateLimitError: not transient — raise immediately so FallbackClient
        can switch to the next backend.
      - AuthenticationError: misconfigured key — raise immediately.

    CONFIRMED BUG (fixed here): the `groq` SDK constructs its underlying
    HTTP client with max_retries=2 BY DEFAULT. That's a SEPARATE retry
    layer, below and invisible to the retry loop in generate() below — a
    429 hit inside the SDK's own transport gets silently retried there
    (honouring the server's Retry-After header) before EVER raising an
    exception this class could see, log, or count. Measured effect: an
    identical call repeated 4x took 442ms / 374ms / 8458ms / 21533ms, with
    NO warning logged anywhere, and GenerationOutput indistinguishable
    from a normal fast call — meaning TierResult.latency_sec in
    production could be silently inflated by 20-50x with zero diagnostic
    signal. Passing max_retries=0 below disables the SDK's hidden retry
    entirely, so every retry that happens is the one in generate() below —
    logged via logger.warning AND counted in GenerationOutput.retry_count.
    """

    MAX_RETRIES = 3
    BASE_WAIT_SEC = 1.0

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile") -> None:
        if not api_key:
            raise BackendUnavailableError("GroqClient: api_key is empty")
        from groq import Groq
        # max_retries=0: see "CONFIRMED BUG" above. Without this, a 429
        # can be silently retried by the SDK's own transport layer before
        # this class ever gets a chance to see, log, or count it.
        self._client = Groq(api_key=api_key, max_retries=0)
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
                # Rate limit → don't retry this backend, let FallbackClient
                # switch. (exc carries a retry-after hint from Groq, e.g.
                # "Rate limit reached... limit 6000... try again in 12s" —
                # not consumed here; a smarter same-backend wait-and-retry
                # using that value would be a reasonable future change,
                # but isn't what was asked for — this preserves the
                # existing fallback-on-rate-limit behaviour unchanged.)
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

    Uses the new google.genai package (google-generativeai is deprecated).

    NOTE: this previously defaulted to gemini-2.0-flash, which was
    retired — confirmed live (2026-08-11): the API returns 404 "This
    model models/gemini-2.0-flash is no longer available." The obvious
    fix (pin a current dated model like gemini-2.5-flash) was ALSO
    confirmed broken for this account specifically: 404 "no longer
    available to new users" — client.models.list() listing a model does
    not mean generate_content will accept it for a given API key/account
    tier. Version-pinned model names on free-tier Gemini are not
    reliably stable; gemini-flash-latest (a Google-managed alias that
    always points at their current flash model) is used instead
    specifically to avoid re-hitting this exact class of bug. Tradeoff:
    behavior can drift between demo runs if Google repoints the alias —
    accepted here in favor of not silently 404ing again later.

    Newer Gemini models default to spending some of max_output_tokens on
    internal "thinking" before the visible answer — see the
    thinking_config comment in generate() below; this cost real answer
    truncation before it was set explicitly.

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
                        # Gemini's newer models "think" before answering,
                        # and thinking tokens are drawn from the SAME
                        # max_output_tokens budget as the visible answer
                        # (no separate allowance) — confirmed live
                        # (2026-08-11): at the default model + this file's
                        # 512-token production budget, thinking consumed
                        # the entire budget and the visible answer came
                        # back truncated to ~80 characters with
                        # finish_reason=MAX_TOKENS, not an error, just a
                        # silently useless response. MINIMAL keeps enough
                        # budget free for a real answer at 512 tokens.
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

# Default subprocess timeout for local inference. Generous because CPU
# inference (the common case — most dev/deploy environments here have no
# GPU) can genuinely take tens of seconds, on top of a cold model load
# every call (see LocalQwenClient's docstring for why nothing persists
# between calls). Configurable per-instance via LocalQwenClient(timeout_sec=...).
_LOCAL_QWEN_TIMEOUT_SEC = 90.0

# Runs in an ISOLATED subprocess (via `sys.executable -c _LOCAL_QWEN_WORKER_SCRIPT`)
# — same invocation shape as backend/evaluation/evaluator.py's
# _bertscore_subprocess(): stdin JSON in, stdout JSON out, nothing else on
# stdout. Kept as one big inline script (not a separate worker .py file) to
# match that existing precedent exactly — this codebase should have ONE
# pattern for process-isolated model inference, not two.
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

    WHY SUBPROCESS ISOLATION (not lazy-loading in-process, memory limits,
    or torch.multiprocessing sharing tricks): this is the last-resort
    backend, reached only after Groq AND Gemini have both failed — which
    is exactly when tier 2's BGE-small embedder and bge-reranker-base
    cross-encoder are typically ALREADY loaded in this same process (RAG
    synthesis loads both before ever calling an LLM). Loading a THIRD
    PyTorch model (Qwen) in-process alongside them reproduces the exact
    multi-model coexistence conflict that backend/evaluation/evaluator.py's
    _bertscore_subprocess() already isolates BERTScore against — see that
    method's docstring. This isn't theoretical: the in-process version of
    this class SIGSEGV'd during a real eval run (2026-08-12) when Groq and
    Gemini were both exhausted and Qwen loaded in-process next to the
    already-loaded embedder/reranker. Subprocess isolation is the fix that
    already works in this codebase for this exact failure mode — attempts
    to make three PyTorch models coexist safely in one process are not
    revisited here because that's the thing already proven not to work
    reliably.

    Tradeoff: every call now pays real process-spawn + cold-model-load
    overhead (roughly 1-3s just to start Python and reload the model,
    before inference even begins) since nothing persists between calls —
    that's the whole point: the subprocess's memory, including the loaded
    model, is torn down when it exits, so it can never coexist with
    whatever's loaded in the parent process. Acceptable specifically
    BECAUSE this only fires as a last resort (after Groq AND Gemini both
    fail), not on the hot path. Do not "optimize" this back to a
    persistent in-process pipeline without re-solving the coexistence
    crash first.

    No API key required — runs entirely on-device.  Model weights (~3 GB)
    are downloaded from HuggingFace Hub on first use and cached locally
    (by the subprocess's transformers/huggingface_hub install, same cache
    dir as any other process on the machine — the isolation is about
    process/memory boundaries, not the on-disk model cache).

    Quality note:
      1.5B parameters vs 70B (Groq Llama) — noticeably lower quality for
      complex multi-step troubleshooting.  This tier exists for offline use
      or when all cloud backends fail, not for primary production traffic.
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

        # Any failure here is mapped to BackendUnavailableError — the same
        # exception GroqClient/GeminiClient raise on failure — so
        # FallbackClient's existing "any exception -> try the next
        # backend" handling doesn't need to know a subprocess is involved
        # at all; the boundary is invisible above this method.
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
        # Identifies which tier's chain this is in logs — e.g. "tier2-synthesis"
        # vs "tier3-generative". Previously this warning was hardcoded to say
        # "Generative tier" unconditionally, which became actively misleading
        # once tier 2 also started using FallbackClient: a tier 2 fallback
        # event logged as if it were tier 3's, making it look like tier 3 was
        # still Groq-primary when debugging exactly this kind of contention.
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

    `backend_order` is authoritative for BOTH which backends are included
    AND the order they're tried in — a name absent from the list is
    excluded even if its API key is set (e.g. `["gemini", "local"]`
    deliberately disables Groq for a tier without touching .env). Cloud
    backends (groq/gemini) are additionally skipped if their API key is
    empty, or if construction fails (missing SDK, bad key format, etc.);
    "local" (LocalQwenClient) needs no key and is only skipped if omitted
    from `backend_order`.

    WHY THIS PARAMETER EXISTS (not just per-tier model names): tier 2's
    RAG synthesis (backend/tiers/retrieval_tier.py) and tier 3's
    closed-book fallback (backend/tiers/generative_tier.py) both used to
    default to Groq as primary, authenticated with the SAME key. That
    meant two supposedly-independent cascade stages silently shared ONE
    Groq TPM budget — heavy tier 2 traffic could starve tier 3's fallback
    capacity and vice versa. Not a rate-limit inconvenience; an
    architectural flaw. Confirmed via scripts/run_eval.py (2026-08-11):
    9/20 queries fell back to Gemini in one unpaced run purely from both
    tiers hitting llama-3.3-70b-versatile's 12,000 TPM limit together.

    First fix attempt: give each tier its own `backend_order`, with tier 3
    reordered to Gemini-primary — same key-per-tier idea, but decoupling
    by PROVIDER instead. That measurably reduced Groq-side contention but
    traded it for a worse problem: Gemini's free tier turned out to be
    request-count limited (5 req/min, 20 req/day observed), fine as an
    occasional fallback but exhausted almost immediately once it became a
    tier's PRIMARY under real traffic — partially recreating the
    contention it was meant to fix. Reverted.

    Actual shipped fix: both tiers keep `backend_order` = [groq, gemini,
    local] (same provider order), but authenticate with DIFFERENT Groq
    keys — tier 2 uses Secrets.groq_api_key, tier 3 uses Secrets.
    groq_api_key_tier3 (selected at the GenerativeTier.__init__ call
    site, not inside this function — see its docstring). Two keys under
    Groq draw from independent TPM pools, so tier 2 and tier 3 stop
    competing for one budget without giving up Groq's speed/reliability
    as the primary for both. Gemini and local remain each tier's OWN
    fallback-of-last-resort behind Groq, never a primary. See config.yaml's
    generative_tier.backend_order comment for the full dated history.

    client_name: identifies this chain in logs (e.g. "tier3-generative" vs
    "tier2-synthesis") — see FallbackClient's docstring for why this
    matters: its fallback/failure warnings used to be hardcoded to say
    "Generative tier" unconditionally, which became actively misleading
    once tier 2 also started building a FallbackClient.

    Usage (tier 3 — note groq_api_key_tier3, NOT groq_api_key; see
    GenerativeTier.__init__ for the real fallback-to-shared-key logic
    omitted here for brevity)::

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
    the confidence threshold but still carry relevant signal — only the
    "question" and "answer" keys are read here; the "doc_id"/"logit" keys
    RetrievalTier._build_candidates() also includes are ignored (they're
    for eval/observability, not prompt construction). See design
    decision (1) above.

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
