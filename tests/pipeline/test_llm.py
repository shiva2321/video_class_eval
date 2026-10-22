import json

import httpx
import pytest

from classeval.pipeline.llm import LLMBadOutput, LLMUnavailable, OllamaClient


def test_chat_json_sends_schema_and_zero_temperature():
    captured_request = {}

    def mock_handler(request: httpx.Request) -> httpx.Response:
        data = json.loads(request.read())
        captured_request.update(data)
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"result": "ok"}),
                }
            },
        )

    transport = httpx.MockTransport(mock_handler)
    client = OllamaClient(
        base_url="http://mock-ollama:11434",
        model="qwen3-vl:8b",
        transport=transport,
    )

    schema = {
        "type": "object",
        "properties": {"result": {"type": "string"}},
        "required": ["result"],
    }
    resp = client.chat_json("You are an evaluator.", "Assess this chunk.", schema=schema)

    assert resp == {"result": "ok"}
    assert captured_request["model"] == "qwen3-vl:8b"
    assert captured_request["format"] == schema
    assert captured_request["think"] is False
    assert captured_request["stream"] is False
    assert captured_request["options"]["temperature"] == 0
    assert captured_request["options"]["seed"] == 7
    assert captured_request["options"]["num_ctx"] == 16384


def test_invalid_json_twice_raises_bad_output():
    call_count = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "not valid json {",
                }
            },
        )

    transport = httpx.MockTransport(mock_handler)
    client = OllamaClient(
        base_url="http://mock-ollama:11434",
        model="qwen3-vl:8b",
        transport=transport,
    )

    with pytest.raises(LLMBadOutput):
        client.chat_json("sys", "user", schema={"type": "object"})

    # Exactly one retry -> 2 total calls
    assert call_count == 2


def test_timeout_raises_unavailable():
    def mock_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("Connection timed out")

    transport = httpx.MockTransport(mock_handler)
    client = OllamaClient(
        base_url="http://mock-ollama:11434",
        model="qwen3-vl:8b",
        transport=transport,
    )

    with pytest.raises(LLMUnavailable):
        client.chat_json("sys", "user", schema={"type": "object"})
