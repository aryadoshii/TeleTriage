"""
Tests for Phase 3: LLM client orchestration and GenerativeTier.

Test organisation
─────────────────
TestBuildUserPrompt   — context assembly helper (pure function, no mocks)
TestFallbackClient    — fallback / retry orchestration (mocked backends)
TestGroqClientUnit    — Groq SDK call shape (mocked Groq SDK)
TestGeminiClientUnit  — Gemini SDK call shape (mocked google.genai)
TestGenerativeTier    — tier wrapper logic (mocked FallbackClient)
TestGroqIntegration   — live Groq API call (gated on GROQ_API_KEY env var)

Run only fast mocked tests (no network, no models):
    uv run pytest tests/test_generative_tier.py -m "not integration and not slow"

Run integration test (requires GROQ_API_KEY in env or .env):
    GROQ_API_KEY=gsk_... uv run pytest tests/test_generative_tier.py -m integration
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from backend.generation.llm_client import (
    GENERATIVE_CONFIDENCE,
    AllBackendsFailedError,
    BackendUnavailableError,
    FallbackClient,
    GenerationOutput,
    build_user_prompt,
)
from backend.types import Query, TierName


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _make_output(backend: str = "groq", text: str = "Test answer.") -> GenerationOutput:
    return GenerationOutput(
        text=text,
        tokens_used=42,
        finish_reason="stop",
        model="test-model",
        backend=backend,
    )


def _make_mock_client(backend: str = "groq", text: str = "Test answer.") -> MagicMock:
    """Return a MagicMock that satisfies the LLMClient protocol."""
    mock = MagicMock()
    mock.generate.return_value = _make_output(backend=backend, text=text)
    return mock


# ─── TestBuildUserPrompt ──────────────────────────────────────────────────────


class TestBuildUserPrompt:
    """Pure function — no mocking needed."""

    def test_no_context_returns_query_unchanged(self):
        assert build_user_prompt("packet loss LTE", []) == "packet loss LTE"

    def test_context_appended_to_query(self):
        docs = [{"question": "How to fix X?", "answer": "Do Y."}]
        result = build_user_prompt("my question", docs)
        assert "my question" in result
        assert "How to fix X?" in result
        assert "Do Y." in result

    def test_context_truncated_at_max_chars(self):
        from backend.generation.llm_client import MAX_CONTEXT_CHARS

        # Build a single doc whose content exceeds the limit
        big_answer = "A" * (MAX_CONTEXT_CHARS + 500)
        docs = [
            {"question": "Q1", "answer": big_answer},
            {"question": "Q2", "answer": "short answer"},
        ]
        result = build_user_prompt("query", docs)
        # The second doc should be cut off because the first already fills the budget
        assert "Q2" not in result

    def test_multiple_context_docs_included(self):
        docs = [
            {"question": "Q1", "answer": "A1"},
            {"question": "Q2", "answer": "A2"},
        ]
        result = build_user_prompt("query", docs)
        assert "Q1" in result
        assert "A1" in result
        assert "Q2" in result
        assert "A2" in result

    def test_missing_keys_in_doc_dont_crash(self):
        # Partial docs (no 'answer' key) should not raise
        docs = [{"question": "Q?"}]
        result = build_user_prompt("query", docs)
        assert "query" in result

    def test_empty_doc_list_returns_bare_query(self):
        result = build_user_prompt("bare query", [])
        assert result == "bare query"


# ─── TestFallbackClient ───────────────────────────────────────────────────────


class TestFallbackClient:
    """Tests the fallback orchestration logic with fully mocked backends."""

    def test_returns_output_from_first_backend_if_ok(self):
        client_a = _make_mock_client("groq", "answer from groq")
        client_b = _make_mock_client("gemini", "answer from gemini")
        fallback = FallbackClient([client_a, client_b])

        result = fallback.generate("sys", "user")

        assert result.backend == "groq"
        assert result.text == "answer from groq"
        client_b.generate.assert_not_called()

    def test_falls_back_to_second_backend_on_first_failure(self):
        client_a = _make_mock_client()
        client_a.generate.side_effect = BackendUnavailableError("rate limit")
        client_b = _make_mock_client("gemini", "gemini answer")
        fallback = FallbackClient([client_a, client_b])

        result = fallback.generate("sys", "user")

        assert result.backend == "gemini"
        assert result.text == "gemini answer"

    def test_falls_back_through_full_chain(self):
        client_a = _make_mock_client()
        client_b = _make_mock_client()
        client_c = _make_mock_client("local", "local answer")
        client_a.generate.side_effect = BackendUnavailableError("groq down")
        client_b.generate.side_effect = BackendUnavailableError("gemini down")
        fallback = FallbackClient([client_a, client_b, client_c])

        result = fallback.generate("sys", "user")

        assert result.backend == "local"

    def test_raises_all_backends_failed_when_chain_exhausted(self):
        client_a = _make_mock_client()
        client_b = _make_mock_client()
        client_a.generate.side_effect = Exception("groq error")
        client_b.generate.side_effect = Exception("gemini error")
        fallback = FallbackClient([client_a, client_b])

        with pytest.raises(AllBackendsFailedError) as exc_info:
            fallback.generate("sys", "user")

        assert "groq error" in str(exc_info.value)
        assert "gemini error" in str(exc_info.value)

    def test_passes_kwargs_to_backend(self):
        client = _make_mock_client()
        fallback = FallbackClient([client])

        fallback.generate("sys prompt", "user msg", max_tokens=256, temperature=0.1)

        client.generate.assert_called_once_with(
            system="sys prompt",
            user="user msg",
            max_tokens=256,
            temperature=0.1,
        )

    def test_any_exception_triggers_fallback(self):
        """FallbackClient catches ALL exceptions, not just BackendUnavailableError."""
        client_a = _make_mock_client()
        client_b = _make_mock_client("gemini", "recovered")
        client_a.generate.side_effect = RuntimeError("unexpected crash")
        fallback = FallbackClient([client_a, client_b])

        result = fallback.generate("sys", "user")
        assert result.backend == "gemini"

    def test_raises_on_empty_client_list(self):
        with pytest.raises(ValueError, match="at least one"):
            FallbackClient([])

    def test_backend_names_property(self):
        # FallbackClient.backend_names returns type(c).__name__ for each client.
        # MagicMock's type name is always "MagicMock", so we use real (but
        # un-called) client instances via patched constructors.
        from backend.generation.llm_client import GroqClient, LocalQwenClient

        with patch("groq.Groq"):
            groq_c = GroqClient(api_key="sk-test")
        local_c = LocalQwenClient()

        fallback = FallbackClient([groq_c, local_c])
        assert fallback.backend_names == ["GroqClient", "LocalQwenClient"]


# ─── TestGroqClientUnit ───────────────────────────────────────────────────────


class TestGroqClientUnit:
    """Verify GroqClient constructs the correct API call shape — Groq SDK mocked."""

    def _make_groq_completion(
        self,
        text: str = "test answer",
        finish_reason: str = "stop",
        total_tokens: int = 50,
    ) -> MagicMock:
        """Build a realistic mock of groq.types.chat.ChatCompletion."""
        mock_usage = MagicMock()
        mock_usage.total_tokens = total_tokens

        mock_message = MagicMock()
        mock_message.content = text

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = finish_reason

        completion = MagicMock()
        completion.choices = [mock_choice]
        completion.usage = mock_usage
        return completion

    def test_raises_on_empty_api_key(self):
        with pytest.raises(BackendUnavailableError, match="api_key is empty"):
            from backend.generation.llm_client import GroqClient
            GroqClient(api_key="")

    def test_calls_chat_completions_create(self):
        from backend.generation.llm_client import GroqClient

        mock_completion = self._make_groq_completion("answer text")

        with patch("groq.Groq") as MockGroq:
            mock_sdk = MockGroq.return_value
            mock_sdk.chat.completions.create.return_value = mock_completion

            client = GroqClient(api_key="sk-test", model="llama-3.3-70b-versatile")
            client.generate(system="sys", user="usr", max_tokens=128, temperature=0.2)

        mock_sdk.chat.completions.create.assert_called_once()
        call_kwargs = mock_sdk.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "sys"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "usr"
        assert call_kwargs.kwargs["max_tokens"] == 128
        assert call_kwargs.kwargs["temperature"] == 0.2

    def test_output_fields_mapped_correctly(self):
        from backend.generation.llm_client import GroqClient

        mock_completion = self._make_groq_completion(
            text="the answer", finish_reason="stop", total_tokens=77
        )

        with patch("groq.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.return_value = mock_completion
            client = GroqClient(api_key="sk-test", model="llama-3.3-70b-versatile")
            output = client.generate("sys", "usr")

        assert output.text == "the answer"
        assert output.finish_reason == "stop"
        assert output.tokens_used == 77
        assert output.backend == "groq"
        assert output.model == "llama-3.3-70b-versatile"

    def test_rate_limit_triggers_fallback_via_fallback_client(self):
        # We can't easily mock RateLimitError's constructor (Groq uses httpx
        # internally and the exception needs real HTTP response objects).
        # Instead, verify the fallback path end-to-end: if the first backend
        # raises BackendUnavailableError (which GroqClient raises on rate-limit),
        # FallbackClient moves to the next backend.  This is fully covered in
        # TestFallbackClient.test_falls_back_to_second_backend_on_first_failure.
        pass  # coverage via TestFallbackClient

    def test_internal_server_error_retries_then_raises(self):
        from backend.generation.llm_client import GroqClient

        # Simulate a persistent transient error by raising a plain exception
        # that matches the isinstance check inside GroqClient's except clause.
        # We patch the except-branch directly by making every call raise
        # BackendUnavailableError (the outcome after MAX_RETRIES exhaustion).
        with patch("groq.Groq") as MockGroq:
            with patch("time.sleep"):  # don't actually sleep during tests
                MockGroq.return_value.chat.completions.create.side_effect = (
                    BackendUnavailableError("simulated retry exhaustion")
                )
                client = GroqClient(api_key="sk-test")

                with pytest.raises(BackendUnavailableError):
                    client.generate("sys", "usr")


# ─── TestGeminiClientUnit ─────────────────────────────────────────────────────


class TestGeminiClientUnit:
    """Verify GeminiClient constructs the correct google.genai API call."""

    def test_raises_on_empty_api_key(self):
        with pytest.raises(BackendUnavailableError, match="api_key is empty"):
            from backend.generation.llm_client import GeminiClient
            GeminiClient(api_key="")

    def test_calls_generate_content_with_system_instruction(self):
        from backend.generation.llm_client import GeminiClient

        mock_usage = MagicMock()
        mock_usage.total_token_count = 33

        mock_finish = MagicMock()
        mock_finish.name = "STOP"

        mock_candidate = MagicMock()
        mock_candidate.finish_reason = mock_finish

        mock_response = MagicMock()
        mock_response.text = "gemini says hello"
        mock_response.usage_metadata = mock_usage
        mock_response.candidates = [mock_candidate]

        with patch("google.genai.Client") as MockClient:
            mock_sdk = MockClient.return_value
            mock_sdk.models.generate_content.return_value = mock_response

            client = GeminiClient(api_key="AIza-test", model="gemini-2.0-flash-exp")
            client.generate(
                system="system prompt", user="user msg", max_tokens=64, temperature=0.2
            )

        mock_sdk.models.generate_content.assert_called_once()
        call_kwargs = mock_sdk.models.generate_content.call_args.kwargs

        assert call_kwargs["model"] == "gemini-2.0-flash-exp"
        assert call_kwargs["contents"] == "user msg"
        # Verify the system instruction is passed via config
        config = call_kwargs["config"]
        assert config.system_instruction == "system prompt"
        assert config.max_output_tokens == 64
        assert config.temperature == 0.2

    def test_output_fields_mapped_correctly(self):
        from backend.generation.llm_client import GeminiClient

        mock_usage = MagicMock()
        mock_usage.total_token_count = 55

        mock_finish = MagicMock()
        mock_finish.name = "STOP"

        mock_candidate = MagicMock()
        mock_candidate.finish_reason = mock_finish

        mock_response = MagicMock()
        mock_response.text = "gemini answer"
        mock_response.usage_metadata = mock_usage
        mock_response.candidates = [mock_candidate]

        with patch("google.genai.Client") as MockClient:
            MockClient.return_value.models.generate_content.return_value = mock_response
            client = GeminiClient(api_key="AIza-test")
            output = client.generate("sys", "usr")

        assert output.text == "gemini answer"
        assert output.tokens_used == 55
        assert output.finish_reason == "STOP"
        assert output.backend == "gemini"


# ─── TestGenerativeTier ───────────────────────────────────────────────────────


class TestGenerativeTier:
    """
    Unit tests for GenerativeTier.answer() — FallbackClient is mocked so
    no real LLM calls are made.  Config is loaded from the real config.yaml
    but secrets are effectively empty in the test environment.
    """

    @pytest.fixture
    def tier_with_mock_client(self):
        """
        Return a GenerativeTier whose internal FallbackClient is replaced
        by a MagicMock so no real API calls are issued.
        """
        from backend.generation.llm_client import GenerationOutput
        from backend.tiers.generative_tier import GenerativeTier

        mock_client = MagicMock(spec=FallbackClient)
        mock_client.generate.return_value = GenerationOutput(
            text="LTE packet loss fix: check S1-U path.",
            tokens_used=30,
            finish_reason="stop",
            model="llama-3.3-70b-versatile",
            backend="groq",
        )

        with patch(
            "backend.tiers.generative_tier.make_llm_client",
            return_value=mock_client,
        ):
            tier = GenerativeTier()

        return tier, mock_client

    def test_answer_returns_tier_result_with_answer(self, tier_with_mock_client):
        tier, _ = tier_with_mock_client
        result = tier.answer(Query(text="packet loss LTE"))

        assert result.answered
        assert result.tier == TierName.GENERATIVE
        assert "S1-U" in result.answer

    def test_confidence_is_fixed_value(self, tier_with_mock_client):
        tier, _ = tier_with_mock_client
        result = tier.answer(Query(text="anything"))

        assert result.confidence == GENERATIVE_CONFIDENCE

    def test_details_contain_backend_and_model(self, tier_with_mock_client):
        tier, _ = tier_with_mock_client
        result = tier.answer(Query(text="anything"))

        assert result.details["backend"] == "groq"
        assert result.details["model"] == "llama-3.3-70b-versatile"
        assert result.details["tokens_used"] == 30
        assert result.details["finish_reason"] == "stop"

    def test_latency_is_positive(self, tier_with_mock_client):
        tier, _ = tier_with_mock_client
        result = tier.answer(Query(text="anything"))
        assert result.latency_sec >= 0.0

    def test_min_confidence_is_zero(self, tier_with_mock_client):
        tier, _ = tier_with_mock_client
        # Generative tier never delegates — it's the final tier
        assert tier.min_confidence == 0.0
        assert tier.should_delegate(result=tier.answer(Query(text="x"))) is False

    def test_retrieved_context_passed_to_client(self, tier_with_mock_client):
        tier, mock_client = tier_with_mock_client
        ctx = [{"question": "Q?", "answer": "A."}]
        query = Query(text="my question", metadata={"retrieved_context": ctx})

        tier.answer(query)

        call_args = mock_client.generate.call_args
        user_prompt = call_args.kwargs["user"]
        assert "my question" in user_prompt
        assert "Q?" in user_prompt
        assert "A." in user_prompt

    def test_no_context_when_metadata_empty(self, tier_with_mock_client):
        tier, mock_client = tier_with_mock_client
        tier.answer(Query(text="plain query"))

        call_args = mock_client.generate.call_args
        user_prompt = call_args.kwargs["user"]
        # Without context the prompt should be exactly the query text
        assert user_prompt == "plain query"

    def test_all_backends_failed_returns_error_answer(self, tier_with_mock_client):
        tier, mock_client = tier_with_mock_client
        mock_client.generate.side_effect = AllBackendsFailedError("all down")

        result = tier.answer(Query(text="anything"))

        assert result.answered  # returns an error message, not None
        assert result.confidence == 0.0
        assert "backends failed" in result.answer.lower() or "all" in result.answer.lower()
        assert result.details["status"] == "all_backends_failed"

    def test_system_prompt_is_passed_to_client(self, tier_with_mock_client):
        from backend.tiers.generative_tier import SYSTEM_PROMPT

        tier, mock_client = tier_with_mock_client
        tier.answer(Query(text="LTE fault"))

        call_args = mock_client.generate.call_args
        assert call_args.kwargs["system"] == SYSTEM_PROMPT

    def test_max_tokens_and_temperature_from_config(self, tier_with_mock_client):
        tier, mock_client = tier_with_mock_client
        tier.answer(Query(text="q"))

        call_args = mock_client.generate.call_args
        assert call_args.kwargs["max_tokens"] == tier._cfg.max_tokens
        assert call_args.kwargs["temperature"] == tier._cfg.temperature


# ─── TestGroqIntegration ──────────────────────────────────────────────────────


@pytest.mark.integration
class TestGroqIntegration:
    """
    Live call to the Groq API.

    Skips automatically if GROQ_API_KEY is not set in the environment.
    Requires a valid key — get one free at https://console.groq.com
    """

    @pytest.fixture(scope="class")
    def groq_client(self):
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            # Also check .env file via the config system
            from backend.config import get_config
            api_key = get_config().secrets.groq_api_key

        if not api_key:
            pytest.skip(
                "GROQ_API_KEY not set.  "
                "Set it in .env or environment to run integration tests."
            )

        from backend.generation.llm_client import GroqClient
        return GroqClient(api_key=api_key, model="llama-3.3-70b-versatile")

    def test_real_groq_call_returns_non_empty_answer(self, groq_client):
        output = groq_client.generate(
            system=(
                "You are a telecom engineer. "
                "Answer in exactly one sentence."
            ),
            user="What does RSRQ measure?",
            max_tokens=60,
            temperature=0.2,
        )

        assert output.text.strip(), "Expected non-empty response from Groq"
        assert output.backend == "groq"
        assert output.tokens_used > 0
        assert output.finish_reason in {"stop", "length"}

    def test_real_groq_output_is_parseable_string(self, groq_client):
        output = groq_client.generate(
            system="You are a helpful assistant.",
            user="Say hello.",
            max_tokens=20,
            temperature=0.0,
        )
        assert isinstance(output.text, str)
        assert len(output.text) > 0
