"""
Generative Tier — PHASE 3 STUB.

Last-resort fallback. When CAG misses AND RAG similarity is below
threshold, we invoke an LLM to generate an answer from scratch.

BACKENDS (all free):
  - groq:   Llama 3.3 70B via Groq Cloud       (PRIMARY - fast, free)
  - gemini: Gemini 2.0 Flash via Google AI     (BACKUP)
  - local:  Qwen2.5-1.5B-Instruct (transformers) (OFFLINE FALLBACK)

PROMPT ENGINEERING NOTES:
  - Keep the system prompt SHORT. Every token eats context.
  - INCLUDE the original user query verbatim (don't rephrase).
  - OPTIONAL: pass top-k retrieved context from RAG as grounding
    even though retrieval similarity was below threshold — partial
    context often improves generation quality meaningfully.
  - Set temperature LOW (0.1-0.3). This is troubleshooting, not
    creative writing. We want consistent, deterministic advice.
  - REQUEST structured output: "Give a concise, numbered list of
    diagnostic steps." Long prose responses are hard to evaluate
    and hard for a network engineer to act on.

CONFIDENCE FOR GENERATIVE OUTPUT:
  This is the hardest part. You CAN'T trust the LLM's self-reported
  confidence. Options (all imperfect):
    1. Fixed low confidence (e.g. 0.60) - honest that we don't know
    2. Log-probability of generated tokens (if API exposes it)
    3. Self-consistency: generate N times, measure agreement
    4. Ask a second LLM to grade the answer (LLM-as-judge)
  Start with option 1. Explore others in Phase 4.

IMPLEMENTATION ORDER:
  1. generation/llm_client.py  - unified interface over groq/gemini/local
  2. Prompt template (below, in SYSTEM_PROMPT)
  3. Wire up in GenerativeTier.answer()
  4. Add retry logic with backend fallback (groq -> gemini -> local)
"""
from __future__ import annotations

from teletriage.config import get_config
from teletriage.tiers.base import BaseTier
from teletriage.types import Query, TierName, TierResult

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
        # Generative is the LAST tier - it must always answer.
        # So min_confidence is effectively 0 (we never delegate).
        self.min_confidence = 0.0
        self._cfg = cfg.generative_tier
        # TODO Phase 3: initialize client
        # self._client = make_llm_client(
        #     backend=self._cfg.backend,
        #     api_key=cfg.secrets.groq_api_key or cfg.secrets.google_api_key,
        # )

    def answer(self, query: Query) -> TierResult:
        start = self._now()

        # TODO Phase 3: implement. Pseudocode:
        #
        # try:
        #     output = self._client.generate(
        #         system=SYSTEM_PROMPT,
        #         user=query.text,
        #         max_tokens=self._cfg.max_tokens,
        #         temperature=self._cfg.temperature,
        #     )
        # except RateLimitError:
        #     # Fall back to secondary backend
        #     output = self._fallback_client.generate(...)
        #
        # return TierResult(
        #     tier=self.tier_name,
        #     answer=output.text,
        #     confidence=0.60,  # honest - we generated from scratch
        #     latency_sec=self._now() - start,
        #     details={
        #         "model": self._cfg.groq_model,
        #         "tokens_used": output.tokens,
        #         "finish_reason": output.finish_reason,
        #     },
        # )

        # Placeholder so the system runs end-to-end today
        return TierResult(
            tier=self.tier_name,
            answer=(
                f"[GENERATIVE TIER NOT YET IMPLEMENTED — Phase 3]\n"
                f"Query was: {query.text!r}\n"
                f"Configure GROQ_API_KEY in .env and implement "
                f"teletriage.generation.llm_client to enable."
            ),
            confidence=0.60,
            latency_sec=self._now() - start,
            details={"status": "stub"},
        )
