"""
CHANGES:
    From embed_text implementing embed_texts --> embed_texts implementing embed_text
"""

from typing import Self

import httpx
import numpy as np

import logging

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Talks to the embedding llama-server (OpenAI-compatible /v1/embeddings)."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8081",
        client: httpx.AsyncClient | None = None,
        batch_size: int = 32,
    ) -> None:
        self._base_url = base_url
        self._client = (
            client
            if client is not None
            else httpx.AsyncClient(base_url=base_url, timeout=30)
        )
        self._batch_size = batch_size

    async def __aenter__(self) -> Self:
        logger.info("Starting Embedding Client")
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self._client.aclose()
        logger.info("Closed Embedding Client")

    async def embed_text(self, text: str) -> np.ndarray:
        """One text -> one unit-length vector, shape (dim,)."""
        logger.info("Embedding Text")
        return (await self.embed_texts([text]))[0]

    async def embed_texts(self, texts: list[str]) -> np.ndarray:
        """N texts -> unit-length vectors, shape (N, dim)."""
        batches = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            resp = await self._client.post("/v1/embeddings", json={"input": batch})
            resp.raise_for_status()
            data = sorted(resp.json()["data"], key=lambda d: d["index"])
            batches.append(np.array([d["embedding"] for d in data], dtype=np.float32))

        vectors = np.vstack(batches)

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)
