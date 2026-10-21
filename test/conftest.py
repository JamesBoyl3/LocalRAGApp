import hashlib

import numpy as np
import pytest

DIM = 16


class FakeEmbedder:
    """Hashes words into a 16-dim bag-of-words vector, so texts sharing words are 'similar'."""

    async def embed_many(self, texts):
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in text.lower().split():
                out[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)

    async def embed(self, text):
        return (await self.embed_many([text]))[0]


class FakeGenerator:
    def __init__(self):
        self.last_messages = None

    async def generate(self, messages, **params):
        self.last_messages = messages
        return "fake answer"

    async def stream(self, messages, **params):
        self.last_messages = messages
        for token in ["fake ", "streamed ", "answer"]:
            yield token


@pytest.fixture
def anyio_backend():
    return "asyncio"
