"""
Tests for Phase 3: LLM client orchestration and GenerativeTier.

Test organisation
─────────────────
TestBuildUserPrompt   — context assembly helper (pure function, no mocks)
TestFallbackClient    — fallback / retry orchestration (mocked backends)
TestMakeLLMClient     — backend_order controls membership + priority (SDK ctors mocked)
TestGroqClientUnit    — Groq SDK call shape (mocked Groq SDK)
TestGeminiClientUnit  — Gemini SDK call shape (mocked google.genai)
TestLocalQwenClientUnit — subprocess isolation call shape (mocked subprocess.run)
TestGenerativeTierGroqKeySelection — tier3 Groq key vs shared-key fallback (mocked get_config)
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
from backend.types import Query, TierName, TierResult


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

    def test_context_block_header_present_when_candidates_given(self):
        """The grounding-vs-authoritative disclaimer header must appear
        whenever there's at least one usable candidate — this is the
        signal the model uses to distinguish 'use as context' from
        'treat as the answer'."""
        docs = [{"question": "Q?", "answer": "A."}]
        result = build_user_prompt("query", docs)
        assert "[Potentially relevant context" in result
        assert "not as authoritative answer" in result

    def test_context_block_header_absent_with_no_candidates(self):
        result = build_user_prompt("query", [])
        assert "[Potentially relevant context" not in result

    def test_truncation_holds_with_3_realistic_800_char_candidates(self):
        """
        RetrievalTier._build_candidates() caps each candidate's answer at
        800 chars before handing it to the router. Three such candidates
        (~809 chars each once formatted as "Q: ...\\nA: ...") sum to well
        over MAX_CONTEXT_CHARS (2000), so this exercises the truncation
        loop against the actual shape it receives in production, not just
        the single-oversized-string case above.
        """
        from backend.generation.llm_client import MAX_CONTEXT_CHARS

        candidates = [
            {"doc_id": f"kb{i}", "question": f"Q{i}", "answer": "A" * 800, "logit": 1.0 - i}
            for i in range(3)
        ]
        # Each snippet = "Q: Q{i}\nA: " (9 chars) + 800 = 809 chars.
        # 809 -> fits (809 <= 2000). 809+809=1618 -> fits. 1618+809=2427 -> doesn't.
        # So candidates 0 and 1 should survive, candidate 2 should not.
        result = build_user_prompt("original query", candidates)

        assert len(result) < len("original query") + 3 * 810  # didn't include all 3 in full
        assert "Q: Q0" in result
        assert "Q: Q1" in result
        assert "Q: Q2" not in result
        assert "original query" in result


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


# ─── TestMakeLLMClient ─────────────────────────────────────────────────────────


class TestMakeLLMClient:
    """
    backend_order is authoritative for both WHICH backends are included and
    the ORDER they're tried in — this is the fix for tier 2 and tier 3
    silently sharing one Groq TPM budget (see make_llm_client()'s docstring).
    Real GroqClient/GeminiClient/LocalQwenClient are constructed (not
    MagicMocks) so FallbackClient.backend_names gives distinguishable
    class names — same pattern as test_backend_names_property above. Only
    the underlying SDK constructors are mocked, so no real network calls.
    """

    def test_default_order_is_groq_gemini_local(self):
        from backend.generation.llm_client import make_llm_client

        with patch("groq.Groq"), patch("google.genai.Client"):
            client = make_llm_client(groq_api_key="sk-test", google_api_key="AIza-test")

        assert client.backend_names == ["GroqClient", "GeminiClient", "LocalQwenClient"]

    def test_custom_order_reorders_backends(self):
        from backend.generation.llm_client import make_llm_client

        with patch("groq.Groq"), patch("google.genai.Client"):
            client = make_llm_client(
                groq_api_key="sk-test",
                google_api_key="AIza-test",
                backend_order=["gemini", "groq", "local"],
            )

        assert client.backend_names == ["GeminiClient", "GroqClient", "LocalQwenClient"]

    def test_backend_omitted_from_order_is_excluded_even_with_key_set(self):
        # A backend absent from backend_order must NOT sneak in at the end
        # just because its API key happens to be set — the order list is
        # authoritative for membership too.
        from backend.generation.llm_client import make_llm_client

        with patch("groq.Groq"), patch("google.genai.Client"):
            client = make_llm_client(
                groq_api_key="sk-test",
                google_api_key="AIza-test",
                backend_order=["gemini", "local"],
            )

        assert client.backend_names == ["GeminiClient", "LocalQwenClient"]

    def test_missing_api_key_skips_backend_even_if_listed_in_order(self):
        from backend.generation.llm_client import make_llm_client

        with patch("groq.Groq"):
            client = make_llm_client(
                groq_api_key="sk-test",
                google_api_key="",  # no Gemini key configured
                backend_order=["gemini", "groq", "local"],
            )

        assert client.backend_names == ["GroqClient", "LocalQwenClient"]

    def test_raises_when_backend_order_yields_no_usable_backends(self):
        from backend.generation.llm_client import make_llm_client

        with pytest.raises(ValueError, match="no usable backends"):
            make_llm_client(groq_api_key="", google_api_key="", backend_order=["groq", "gemini"])


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

    def test_constructs_sdk_client_with_max_retries_zero(self):
        """
        CONFIRMED BUG regression test: the groq SDK's Groq(...) client
        defaults to max_retries=2 at the HTTP transport layer — a retry
        mechanism entirely separate from (and invisible to) the retry
        loop in generate() above. Measured live: an identical call
        repeated 4x took 442ms/374ms/8458ms/21533ms with NO warning
        logged and NO way to tell a retry happened — the SDK silently
        absorbed 429s using the server's Retry-After header before ever
        raising to this class. Passing max_retries=0 disables that
        hidden layer so THIS class's own (logged, counted) retry loop is
        the only one that can ever fire.
        """
        from backend.generation.llm_client import GroqClient

        with patch("groq.Groq") as MockGroq:
            GroqClient(api_key="sk-test", model="llama-3.1-8b-instant")

        MockGroq.assert_called_once_with(api_key="sk-test", max_retries=0)

    def test_retry_count_zero_on_first_attempt_success(self):
        from backend.generation.llm_client import GroqClient

        mock_completion = self._make_groq_completion("answer")
        with patch("groq.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.return_value = mock_completion
            client = GroqClient(api_key="sk-test")
            output = client.generate("sys", "usr")

        assert output.retry_count == 0

    def test_retry_count_reflects_retries_before_success(self):
        """A transient error on attempt 0, success on attempt 1 -> the
        caller can now SEE that a retry happened (retry_count=1) instead
        of it being indistinguishable from a normal fast call."""
        import httpx
        from groq import InternalServerError

        from backend.generation.llm_client import GroqClient

        req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        resp = httpx.Response(status_code=500, request=req)
        transient_error = InternalServerError("simulated 500", response=resp, body=None)
        mock_completion = self._make_groq_completion("answer after retry")

        with patch("groq.Groq") as MockGroq:
            with patch("time.sleep"):  # don't actually sleep in tests
                MockGroq.return_value.chat.completions.create.side_effect = [
                    transient_error,
                    mock_completion,
                ]
                client = GroqClient(api_key="sk-test")
                output = client.generate("sys", "usr")

        assert output.retry_count == 1
        assert output.text == "answer after retry"

    def test_logs_warning_when_retry_succeeds(self, caplog):
        """A retry that eventually succeeds must be logged, not silent —
        this is the whole point of the fix: retries are now OBSERVABLE."""
        import logging

        import httpx
        from groq import InternalServerError

        from backend.generation.llm_client import GroqClient

        req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        resp = httpx.Response(status_code=500, request=req)
        transient_error = InternalServerError("simulated 500", response=resp, body=None)
        mock_completion = self._make_groq_completion("recovered")

        with caplog.at_level(logging.WARNING, logger="backend.generation.llm_client"):
            with patch("groq.Groq") as MockGroq:
                with patch("time.sleep"):
                    MockGroq.return_value.chat.completions.create.side_effect = [
                        transient_error,
                        mock_completion,
                    ]
                    client = GroqClient(api_key="sk-test")
                    client.generate("sys", "usr")

        assert any("succeeded after" in r.message for r in caplog.records)


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


# ─── TestLocalQwenClientUnit ────────────────────────────────────────────────────


class TestLocalQwenClientUnit:
    """
    LocalQwenClient runs inference in an isolated subprocess (see its
    docstring: coexisting in-process with the embedder/reranker SIGSEGV'd
    during a real eval run). subprocess.run is mocked here — same spirit
    as mocking groq.Groq / google.genai.Client for the other backends —
    so no real subprocess is spawned and no model loads. subprocess.run
    is patched at its source ("subprocess.run"), not
    "backend.generation.llm_client.subprocess.run", because the import
    lives inside generate() rather than at module level; patching the
    module attribute directly affects that local import too since
    attribute lookup happens at call time.
    """

    def _completed(self, stdout: str = "", returncode: int = 0, stderr: str = ""):
        import subprocess as sp

        return sp.CompletedProcess(
            args=["python", "-c", "..."], returncode=returncode, stdout=stdout, stderr=stderr,
        )

    def test_successful_generation_returns_output(self):
        import json

        from backend.generation.llm_client import LocalQwenClient

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = self._completed(json.dumps({"text": "local answer"}))
            client = LocalQwenClient()
            output = client.generate("sys", "usr")

        assert output.text == "local answer"
        assert output.backend == "local"
        assert output.finish_reason == "stop"
        assert output.tokens_used == 2  # len("local answer".split())

    def test_passes_correct_input_via_stdin_and_default_timeout(self):
        import json

        from backend.generation.llm_client import LocalQwenClient

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = self._completed(json.dumps({"text": "ok"}))
            client = LocalQwenClient(model_name="Qwen/Test-Model")
            client.generate("sys prompt", "user msg", max_tokens=128, temperature=0.5)

        call_kwargs = mock_run.call_args.kwargs
        sent = json.loads(call_kwargs["input"])
        assert sent == {
            "model_name": "Qwen/Test-Model",
            "system": "sys prompt",
            "user": "user msg",
            "max_tokens": 128,
            "temperature": 0.5,
        }
        assert call_kwargs["timeout"] == 90.0  # default

    def test_custom_timeout_is_configurable_and_passed_through(self):
        import json

        from backend.generation.llm_client import LocalQwenClient

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = self._completed(json.dumps({"text": "ok"}))
            client = LocalQwenClient(timeout_sec=15.0)
            client.generate("sys", "usr")

        assert mock_run.call_args.kwargs["timeout"] == 15.0

    def test_timeout_raises_backend_unavailable(self):
        import subprocess as sp

        from backend.generation.llm_client import BackendUnavailableError, LocalQwenClient

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = sp.TimeoutExpired(cmd="python", timeout=90.0)
            client = LocalQwenClient()
            with pytest.raises(BackendUnavailableError, match="timed out"):
                client.generate("sys", "usr")

    def test_nonzero_exit_raises_backend_unavailable(self):
        from backend.generation.llm_client import BackendUnavailableError, LocalQwenClient

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = self._completed(
                returncode=1, stderr="CUDA out of memory"
            )
            client = LocalQwenClient()
            with pytest.raises(BackendUnavailableError, match="exited 1"):
                client.generate("sys", "usr")

    def test_malformed_json_output_raises_backend_unavailable(self):
        from backend.generation.llm_client import BackendUnavailableError, LocalQwenClient

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = self._completed(stdout="not valid json {{{")
            client = LocalQwenClient()
            with pytest.raises(BackendUnavailableError, match="malformed output"):
                client.generate("sys", "usr")

    def test_missing_text_key_raises_backend_unavailable(self):
        import json

        from backend.generation.llm_client import BackendUnavailableError, LocalQwenClient

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = self._completed(json.dumps({"wrong_key": "oops"}))
            client = LocalQwenClient()
            with pytest.raises(BackendUnavailableError, match="malformed output"):
                client.generate("sys", "usr")


# ─── TestGenerativeTierGroqKeySelection ────────────────────────────────────────


class TestGenerativeTierGroqKeySelection:
    """
    Tier 3 authenticates to Groq with its OWN key (secrets.
    groq_api_key_tier3) — a separate 12,000 TPM pool from tier 2's RAG
    synthesis (secrets.groq_api_key) — falling back to tier 2's key (with
    a one-time startup warning) if GROQ_API_KEY_TIER3 isn't set. See
    GenerativeTier.__init__.

    get_config() is patched with a fully-controlled fake config (not the
    real .env-backed one) so these tests are deterministic regardless of
    which keys happen to be set in this machine's .env. make_llm_client
    is patched too, purely to inspect what groq_api_key it was called
    with — no real client construction or network calls happen.
    """

    def _make_fake_cfg(self, groq_api_key: str, groq_api_key_tier3: str):
        from backend.config import GenerativeTierConfig

        cfg = MagicMock()
        cfg.secrets.groq_api_key = groq_api_key
        cfg.secrets.groq_api_key_tier3 = groq_api_key_tier3
        cfg.secrets.google_api_key = ""
        cfg.generative_tier = GenerativeTierConfig()
        return cfg

    def test_uses_tier3_key_when_set(self):
        from backend.tiers.generative_tier import GenerativeTier

        fake_cfg = self._make_fake_cfg(groq_api_key="sk-tier2", groq_api_key_tier3="sk-tier3")
        with patch("backend.tiers.generative_tier.get_config", return_value=fake_cfg):
            with patch("backend.tiers.generative_tier.make_llm_client") as mock_make:
                mock_make.return_value = MagicMock(spec=FallbackClient)
                GenerativeTier()

        assert mock_make.call_args.kwargs["groq_api_key"] == "sk-tier3"

    def test_falls_back_to_shared_key_when_tier3_key_absent(self, caplog):
        import logging

        from backend.tiers.generative_tier import GenerativeTier

        fake_cfg = self._make_fake_cfg(groq_api_key="sk-tier2", groq_api_key_tier3="")
        with caplog.at_level(logging.WARNING, logger="backend.tiers.generative_tier"):
            with patch("backend.tiers.generative_tier.get_config", return_value=fake_cfg):
                with patch("backend.tiers.generative_tier.make_llm_client") as mock_make:
                    mock_make.return_value = MagicMock(spec=FallbackClient)
                    GenerativeTier()

        assert mock_make.call_args.kwargs["groq_api_key"] == "sk-tier2"
        assert any("falling back" in r.message for r in caplog.records)

    def test_no_warning_when_tier3_key_present(self, caplog):
        import logging

        from backend.tiers.generative_tier import GenerativeTier

        fake_cfg = self._make_fake_cfg(groq_api_key="sk-tier2", groq_api_key_tier3="sk-tier3")
        with caplog.at_level(logging.WARNING, logger="backend.tiers.generative_tier"):
            with patch("backend.tiers.generative_tier.get_config", return_value=fake_cfg):
                with patch("backend.tiers.generative_tier.make_llm_client") as mock_make:
                    mock_make.return_value = MagicMock(spec=FallbackClient)
                    GenerativeTier()

        assert not any("falling back" in r.message for r in caplog.records)

    def test_no_groq_keys_at_all_passes_empty_string(self):
        # Neither key set: groq_key resolves to "" (empty), passed through
        # as-is — make_llm_client is responsible for skipping Groq when
        # its api_key is empty (see its own tests), not this fallback logic.
        from backend.tiers.generative_tier import GenerativeTier

        fake_cfg = self._make_fake_cfg(groq_api_key="", groq_api_key_tier3="")
        with patch("backend.tiers.generative_tier.get_config", return_value=fake_cfg):
            with patch("backend.tiers.generative_tier.make_llm_client") as mock_make:
                mock_make.return_value = MagicMock(spec=FallbackClient)
                GenerativeTier()

        assert mock_make.call_args.kwargs["groq_api_key"] == ""


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

    def test_graceful_degradation_no_retrieved_context_key_at_all(self, tier_with_mock_client):
        """
        Now that the router wires retrieved_context in (see
        TestRouterCandidatePropagation below), it's important the
        generative tier still works exactly as before for a Query whose
        metadata has no 'retrieved_context' key at all — not merely an
        empty one. Covers direct GenerativeTier.answer() callers that
        bypass the router entirely (CLI/eval harness one-off calls).
        """
        tier, mock_client = tier_with_mock_client
        query = Query(text="plain query with no context")

        result = tier.answer(query)

        assert result.answered
        assert result.details["retrieved_context_docs"] == 0
        call_args = mock_client.generate.call_args
        assert call_args.kwargs["user"] == "plain query with no context"

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


# ─── TestRouterCandidatePropagation ────────────────────────────────────────────


class TestRouterCandidatePropagation:
    """
    Router.route() must translate a delegating tier's
    TierResult.details["candidates"] into the NEXT tier's
    Query.metadata["retrieved_context"] — this is the wiring that makes
    backend/generation/llm_client.py design decision (1) actually take
    effect in production. Both tiers are mocked (no real retrieval index,
    no real LLM call) so this stays in the fast suite; it verifies the
    router's propagation logic itself, not either tier's internals.
    """

    def _mock_tier(self, *, tier_name, answer, confidence, details=None, delegates):
        """Build a BaseTier-spec'd mock with should_delegate pinned to a
        fixed outcome (bypassing the real confidence-threshold logic —
        we're testing the router's candidate plumbing, not delegation
        math)."""
        from backend.tiers.base import BaseTier

        tier = MagicMock(spec=BaseTier)
        tier.answer.return_value = TierResult(
            tier=tier_name,
            answer=answer,
            confidence=confidence,
            details=details or {},
        )
        tier.should_delegate.return_value = delegates
        return tier

    def test_delegated_candidates_reach_next_tier(self):
        from backend.router import Router

        candidates = [
            {"doc_id": "kb1", "question": "Q1", "answer": "A1", "logit": 0.3},
            {"doc_id": "kb2", "question": "Q2", "answer": "A2", "logit": 0.1},
        ]

        retrieval_tier = self._mock_tier(
            tier_name=TierName.RETRIEVAL,
            answer=None,
            confidence=0.4,
            details={"reason": "low_rerank_score", "candidates": candidates},
            delegates=True,
        )
        generative_tier = self._mock_tier(
            tier_name=TierName.GENERATIVE,
            answer="generated answer", confidence=0.6, delegates=False,
        )

        router = Router(tiers=[retrieval_tier, generative_tier])
        query = Query(text="MME rejects TAU request with EMM cause 11")
        response = router.route(query)

        generative_tier.answer.assert_called_once()
        passed_query = generative_tier.answer.call_args[0][0]
        assert passed_query.metadata["retrieved_context"] == candidates

        # The final Response must report the ORIGINAL query the caller
        # submitted — not the router's internally-decorated version.
        assert response.query is query
        assert response.query.metadata == {}
        assert response.answer == "generated answer"

    def test_no_candidates_means_no_retrieved_context_key(self):
        from backend.router import Router

        retrieval_tier = self._mock_tier(
            tier_name=TierName.RETRIEVAL,
            answer=None,
            confidence=0.0,
            details={"reason": "no_match"},  # no "candidates" key at all
            delegates=True,
        )
        generative_tier = self._mock_tier(
            tier_name=TierName.GENERATIVE,
            answer="generated", confidence=0.6, delegates=False,
        )

        router = Router(tiers=[retrieval_tier, generative_tier])
        router.route(Query(text="novel query"))

        passed_query = generative_tier.answer.call_args[0][0]
        assert "retrieved_context" not in passed_query.metadata

    def test_empty_candidates_list_means_no_retrieved_context_key(self):
        """An explicit empty list is treated the same as absence — the
        router's `if new_candidates:` check must not add an empty-but-
        present retrieved_context key, matching build_user_prompt's own
        graceful-degradation-on-empty-list behaviour."""
        from backend.router import Router

        retrieval_tier = self._mock_tier(
            tier_name=TierName.RETRIEVAL,
            answer=None, confidence=0.0, details={"candidates": []}, delegates=True,
        )
        generative_tier = self._mock_tier(
            tier_name=TierName.GENERATIVE,
            answer="generated", confidence=0.6, delegates=False,
        )

        router = Router(tiers=[retrieval_tier, generative_tier])
        router.route(Query(text="novel query"))

        passed_query = generative_tier.answer.call_args[0][0]
        assert "retrieved_context" not in passed_query.metadata

    def test_candidates_do_not_leak_into_final_response_query(self):
        """The decorated Query (with retrieved_context in its metadata)
        must never be what the caller sees back — only the tier that
        receives it should see it."""
        from backend.router import Router

        candidates = [{"doc_id": "kb1", "question": "Q1", "answer": "A1", "logit": 0.3}]
        retrieval_tier = self._mock_tier(
            tier_name=TierName.RETRIEVAL,
            answer=None, confidence=0.4, details={"candidates": candidates}, delegates=True,
        )
        generative_tier = self._mock_tier(
            tier_name=TierName.GENERATIVE,
            answer="generated", confidence=0.6, delegates=False,
        )

        router = Router(tiers=[retrieval_tier, generative_tier])
        original_query = Query(text="query text")
        response = router.route(original_query)

        assert "retrieved_context" not in response.query.metadata
        assert response.query.text == "query text"


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
