"""
Generative Tier — Phase 3 implementation.

Last-resort fallback.  When CAG misses AND RAG similarity is below threshold,
we invoke an LLM to generate an answer from scratch (optionally grounded by
the RAG near-misses — see llm_client.py design decision (1)).

BACKENDS (all free, tried in this order):
  1. Groq:   Llama 3.3 70B — fast, 30 req/min, requires GROQ_API_KEY
  2. Gemini: Gemini 2.0 Flash — 15 req/min, requires GOOGLE_API_KEY
  3. Local:  Qwen2.5-1.5B-Instruct — no key, slower, ~3 GB on disk

Startup:
  Backends are registered at __init__ time based on which API keys are
  configured in .env.  No network calls happen until the first answer().
"""
from __future__ import annotations

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

        self._client: FallbackClient = make_llm_client(
            groq_api_key=cfg.secrets.groq_api_key,
            google_api_key=cfg.secrets.google_api_key,
            groq_model=self._cfg.groq_model,
            gemini_model=self._cfg.gemini_model,
            local_model=self._cfg.local_model,
        )

    def answer(self, query: Query) -> TierResult:
        start = self._now()

        # Retrieve any near-miss context that RAG surfaced but didn't
        # trust enough to answer with directly.  See llm_client.py
        # design decision (1) for why we use this even below threshold.
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
            # See llm_client.py design decision (2) for upgrade paths.
            confidence=GENERATIVE_CONFIDENCE,
            latency_sec=self._now() - start,
            details={
                "backend": output.backend,
                "model": output.model,
                "tokens_used": output.tokens_used,
                "finish_reason": output.finish_reason,
                "retrieved_context_docs": len(retrieved_context),
            },
        )
