"""Offline checks for the OpenAI-compatible API integration."""

import json

import httpx2
import openai
import pytest

from cs329z_hw1 import llm


@pytest.fixture(autouse=True)
def api_settings(monkeypatch):
    monkeypatch.setattr(llm, "_load_env", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://provider.example/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.delenv("CS329Z_REASONING_EFFORT", raising=False)
    for role in llm.MODEL_ROLES:
        monkeypatch.delenv(f"CS329Z_{role.upper()}_MODEL", raising=False)


@pytest.mark.parametrize("effort", ["", "low", "high", "max", "none"])
def test_api_request_and_cache(monkeypatch, tmp_path, effort):
    monkeypatch.setenv("CS329Z_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("CS329Z_REASONING_EFFORT", effort)
    requests = []

    def respond(request):
        requests.append(request)
        usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        return httpx2.Response(200, json={
            "id": "test", "object": "chat.completion", "created": 0,
            "model": "test-model",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": "Hello", "reasoning_content": "Hidden",
            }}],
            "usage": usage,
        })

    real_client = openai.OpenAI
    with httpx2.Client(transport=httpx2.MockTransport(respond)) as http_client:
        monkeypatch.setattr(openai, "OpenAI", lambda **kw: real_client(http_client=http_client, **kw))
        model = llm.LM()
        messages = [{"role": "user", "content": "Say hello"}]
        assert model(messages) == "Hello"
        assert model(messages) == "Hello"

    assert len(requests) == 1
    assert str(requests[0].url) == "https://provider.example/v1/chat/completions"
    assert requests[0].headers["authorization"] == "Bearer test-key"
    body = json.loads(requests[0].content)
    assert body == {
        "model": "test-model", "messages": messages, "max_tokens": 4096,
        **({"reasoning_effort": effort} if effort else {}),
        **({"temperature": 0.0} if not effort or effort == "none" else {}),
    }
    assert model.cached_calls == 1
    assert model.input_tokens == 100
    assert model.output_tokens == 50
    assert list(tmp_path.glob("*/*.json"))
    assert not list(tmp_path.glob("*.jsonl"))


def test_model_override(monkeypatch):
    monkeypatch.setenv("CS329Z_JUDGE_MODEL", "judge-model")
    assert llm.resolve_model("judge") == "judge-model"
    assert llm.resolve_model("user") == "test-model"
    assert llm.LM("any-provider-model").model == "any-provider-model"


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        llm.LM()._complete([{"role": "user", "content": "Hello"}])


def test_missing_model(monkeypatch):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    with pytest.raises(ValueError, match="OPENAI_MODEL"):
        llm.LM()


def test_endpoint_isolates_cache(monkeypatch):
    messages = [{"role": "user", "content": "Hello"}]
    first = llm.LM()
    monkeypatch.setenv("OPENAI_BASE_URL", "https://another-provider.example/v1")
    second = llm.LM()
    assert first._key(messages) != second._key(messages)
    monkeypatch.delenv("OPENAI_BASE_URL")
    assert llm.LM().base_url == "https://api.openai.com/v1"


def test_truncated_reply_is_not_cached(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "_load_env", lambda: None)
    monkeypatch.setenv("CS329Z_CACHE_DIR", str(tmp_path))
    model = llm.LM("test-model")
    monkeypatch.setattr(model, "_complete", lambda messages: ("Partial", {
        "input": 100, "output": 50,
        "reasoning": 30, "finish_reason": "length",
    }))
    with pytest.raises(llm.OutputTruncated) as error:
        model([{"role": "user", "content": "Hello"}])
    assert error.value.text == "Partial"
    assert error.value.reasoning_tokens == 30
    assert model.output_tokens == 50
    assert not list(tmp_path.glob("*/*.json"))


def test_simulation_usage_and_report(monkeypatch, tmp_path):
    from cs329z_hw1.simulation.evaluate import format_table
    from cs329z_hw1.simulation.record import ConversationRecord
    from cs329z_hw1.simulation.runner import RetryingLM, _usage

    monkeypatch.setenv("CS329Z_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(llm.LM, "_complete", lambda self, messages: ("Answer", {
        "input": 10, "output": self.max_tokens,
        "finish_reason": "length" if self.max_tokens == 2 else "stop",
    }))
    model = RetryingLM("judge", max_tokens=2)
    assert model([{"role": "user", "content": "Hello"}]) == "Answer"
    usage = _usage([model])
    assert usage == {"calls": 2, "cached_calls": 0, "input_tokens": 20, "output_tokens": 6}
    record = ConversationRecord("test", "coding", usage={"judge": usage}, success=True)
    saved = ConversationRecord.from_json(record.to_json())
    assert saved.usage == record.usage
    table = format_table([saved])
    assert "PASS" in table
    assert "Successful conversations: 1 of 1" in table
    assert "USD" not in table and "$" not in table
