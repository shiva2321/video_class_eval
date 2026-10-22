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
    assert captured_request["options"]["num_ctx"] == 8192  # a chunk and its answer need about 5000 tokens; a smaller window leaves room for parallel calls


def test_invalid_json_on_every_attempt_raises_bad_output():
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

    assert call_count == 3  # the first attempt and two retries


def test_a_retry_changes_seed_and_temperature_and_can_recover():
    seen: list[dict] = []

    def mock_handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        seen.append(_json.loads(request.content)["options"])
        content = "not json {" if len(seen) == 1 else '{"ok": true}'
        return httpx.Response(200, json={"message": {"role": "assistant", "content": content}})

    client = OllamaClient(base_url="http://mock-ollama:11434", model="m", transport=httpx.MockTransport(mock_handler))
    assert client.chat_json("sys", "user", schema={"type": "object"}) == {"ok": True}
    assert (seen[0]["seed"], seen[0]["temperature"]) == (7, 0)
    assert seen[1]["seed"] != seen[0]["seed"] and seen[1]["temperature"] > 0


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


def test_the_context_window_and_the_seed_come_from_the_client_settings():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        seen.update(_json.loads(request.content)["options"])
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "{}"}})

    client = OllamaClient("http://mock", "m", transport=httpx.MockTransport(handler), seed=42, num_ctx=4096)
    client.chat_json("s", "u", schema={"type": "object"})
    assert seen["seed"] == 42 and seen["num_ctx"] == 4096
