from typing import Self

import json
from typing import AsyncIterator

import httpx


class GenerationClient:
    """Talks to the generation llama-server (OpenAI-compatible /v1/chat/completions)."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._client = client or httpx.AsyncClient(
            base_url=base_url, timeout=httpx.Timeout(300.0, connect=10.0)
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self._client.aclose()
        return

    async def generate(self, messages: list[dict], **params) -> str:
        """Wait for the whole answer and return it."""
        resp = await self._client.post(
            "/v1/chat/completions",
            json={"messages": messages, "stream": False, **params},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    async def stream(self, messages: list[dict], **params) -> AsyncIterator[str]:
        """Yield the answer piece by piece as llama-server produces it."""
        async with self._client.stream(
            "POST",
            "/v1/chat/completions",
            json={"messages": messages, "stream": True, **params},
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if payload == "[DONE]":
                    break
                choices = json.loads(payload).get("choices") or []
                if choices:
                    token = choices[0].get("delta", {}).get("content")
                    if token:
                        yield token
