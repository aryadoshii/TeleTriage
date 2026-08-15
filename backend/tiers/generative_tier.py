"""
Generative Tier.

Last-resort fallback: when CAG misses AND RAG similarity is below
threshold, invoke an LLM to generate an answer from scratch (optionally
grounded by RAG near-misses — see llm_client.py's build_user_prompt()).

BACKENDS (config.generative_tier.backend_order, default order below):
  1. Groq:   Llama 3.3 70B — requires GROQ_API_KEY_TIER3, a key separate
             from tier 2's, so the two tiers don't share one TPM budget.
             Falls back to tier 2's key if unset (logged once — __init__).
  2. Gemini: gemini-flash-latest — fallback-of-last-resort only, not
             primary (its free tier is request-count limited, too tight
             for primary traffic).
  3. Local:  Qwen2.5-1.5B-Instruct — no key, slower, ~3 GB on disk

Backends are registered at __init__ time; no network calls until the
first answer().
"""
from __future__ import annotations

import logging

from backend.config import get_config
from backend.generation.llm_client import (
    GENERATIVE_CONFIDENCE,
    AllBackendsFailedError,
    FallbackClient,
    build_user_prompt,
    make_llm_client,
)
from backend.tiers.base import BaseTier
from backend.types import Query, TierName, TierResult

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior telecom network engineer assisting with fault triage.
Given a fault description, respond with:
1. Most likely root cause (one sentence)
2. Diagnostic steps (numbered list, max 5 steps)
3. Remediation (numbered list, max 3 steps)

Be concise. Use standard telecom terminology (RSRP, SINR, PDN, eNodeB, etc.).
If the query is ambiguous, state what additional info you'd need.
Never fabricate specific vendor commands you're not sure about."""


class GenerativeTier(BaseTier):
    tier_name = TierName.GENERATIVE

    def __init__(self) -> None:
        cfg = get_config()
        # Generative is the LAST tier — it must always produce an answer.
        # min_confidence=0.0 means the router never delegates past this tier.
        self.min_confidence = 0.0
        self._cfg = cfg.generative_tier

        # Tier 3's own Groq key (see Secrets.groq_api_key_tier3), falling
        # back to tier 2's shared key if unset. Warns once here at
        # __init__ (not per-query) rather than spamming logs.
        groq_key = cfg.secrets.groq_api_key_tier3
        if not groq_key and cfg.secrets.groq_api_key:
            groq_key = cfg.secrets.groq_api_key
            log.warning(
                "GROQ_API_KEY_TIER3 not set — tier 3 is falling back to "
                "the SAME Groq key as tier 2 (GROQ_API_KEY). This works "
                "but tier 2 and tier 3 will compete for one 12,000 TPM "
                "pool again. Set GROQ_API_KEY_TIER3 in .env to a separate "
                "key for full isolation."
            )

        self._client: FallbackClient = make_llm_client(
            groq_api_key=groq_key,
            google_api_key=cfg.secrets.google_api_key,
            groq_model=self._cfg.groq_model,
            gemini_model=self._cfg.gemini_model,
            local_model=self._cfg.local_model,
            backend_order=self._cfg.backend_order,
            client_name="tier3-generative",
        )

    def answer(self, query: Query) -> TierResult:
        start = self._now()

        # Near-miss context RAG surfaced but didn't trust enough to
        # answer with directly — see build_user_prompt()'s docstring.
        retrieved_context: list[dict] = query.metadata.get("retrieved_context", [])
        user_prompt = build_user_prompt(query.text, retrieved_context)

        try:
            output = self._client.generate(
                system=SYSTEM_PROMPT,
                user=user_prompt,
                max_tokens=self._cfg.max_tokens,
                temperature=self._cfg.temperature,
            )
        except AllBackendsFailedError as exc:
            # This should almost never happen because LocalQwen is always
            # available — but if it does (e.g. missing model weights), we
            # must still return a TierResult, never raise.
            elapsed = self._now() - start
            return TierResult(
                tier=self.tier_name,
                answer=(
                    "All LLM backends failed.  "
                    "Please check API keys in .env and ensure the local "
                    "Qwen model weights are downloaded."
                ),
                confidence=0.0,
                latency_sec=elapsed,
                details={"error": str(exc), "status": "all_backends_failed"},
            )

        return TierResult(
            tier=self.tier_name,
            answer=output.text,
            # Fixed 0.60 — honest acknowledgement that this is generated,
            # not retrieved from a verified knowledge base.
            confidence=GENERATIVE_CONFIDENCE,
            latency_sec=self._now() - start,
            details={
                "backend": output.backend,
                "model": output.model,
                "tokens_used": output.tokens_used,
                "finish_reason": output.finish_reason,
                "retrieved_context_docs": len(retrieved_context),
                # Non-zero means latency_sec includes at least one retry.
                "retry_count": output.retry_count,
            },
        )
