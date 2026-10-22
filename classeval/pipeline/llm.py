import base64
import json
import re
from typing import Any

import httpx

MAX_OUTPUT_TOKENS = 1500  # a full checklist answer is a few hundred tokens


class LLMUnavailable(Exception):
    """Raised when Ollama cannot be reached or times out."""


class LLMBadOutput(Exception):
    """Raised when Ollama returns invalid JSON or empty output after retry."""


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_s: float = 180.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.transport = transport

    def chat_json(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        images: list[bytes] | None = None,
    ) -> dict[str, Any]:
        """Send chat request to Ollama and return parsed JSON conforming to schema."""
        user_msg: dict[str, Any] = {"role": "user", "content": user}
        if images:
            user_msg["images"] = [base64.b64encode(img).decode("utf-8") for img in images]

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                user_msg,
            ],
            "format": schema,
            "stream": False,
            "think": False,
            "keep_alive": "10m",  # stay loaded between the chunks of one job
            "options": {
                "temperature": 0,
                "seed": 7,
                "num_ctx": 16384,
                "num_predict": MAX_OUTPUT_TOKENS,  # hard bound: a runaway generation must never stall a job
            },
        }

        # Up to 2 attempts (initial + 1 retry on JSON parse failure)
        max_attempts = 2
        last_error: Exception | None = None

        for attempt in range(max_attempts):
            try:
                with httpx.Client(
                    base_url=self.base_url,
                    timeout=self.timeout_s,
                    transport=self.transport,
                ) as client:
                    resp = client.post("/api/chat", json=payload)
                    resp.raise_for_status()
                    data = resp.json()
            except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as e:
                raise LLMUnavailable(f"Ollama unavailable at {self.base_url}: {e}") from e
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (502, 503, 504):
                    raise LLMUnavailable(f"Ollama service error ({e.response.status_code}): {e}") from e
                raise LLMBadOutput(f"Ollama returned HTTP error {e.response.status_code}: {e}") from e
            except Exception as e:
                raise LLMUnavailable(f"Failed to communicate with Ollama: {e}") from e

            content = data.get("message", {}).get("content", "")
            # Strip any residual thinking tags
            cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            if not cleaned:
                last_error = ValueError("Empty response content from LLM")
                continue

            try:
                return json.loads(cleaned)
            except json.JSONDecodeError as e:
                last_error = e
                continue

        raise LLMBadOutput(f"Failed to parse valid JSON from LLM after {max_attempts} attempts: {last_error}")

    def unload(self) -> None:
        """Release VRAM by requesting Ollama unload the model immediately."""
        try:
            with httpx.Client(
                base_url=self.base_url,
                timeout=10.0,
                transport=self.transport,
            ) as client:
                client.post("/api/chat", json={"model": self.model, "keep_alive": 0})
        except httpx.HTTPError:
            pass
