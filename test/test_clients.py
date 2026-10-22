import json

import httpx
import numpy as np
import pytest

from localrag.Clients import EmbeddingClient, GenerationClient

pytestmark = pytest.mark.anyio


def make_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://test"
    )


async def test_embed_texts_normalises_and_orders_by_index():
    def handler(request):
        assert request.url.path == "/v1/embeddings"
        # deliberately out of order
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.0, 2.0]},
                    {"index": 0, "embedding": [3.0, 4.0]},
                ]
            },
        )

    vectors = await EmbeddingClient(client=make_client(handler)).embed_texts(["a", "b"])
    np.testing.assert_allclose(vectors[0], [0.6, 0.8], rtol=1e-6)
    np.testing.assert_allclose(vectors[1], [0.0, 1.0], rtol=1e-6)


async def test_embed_texts_splits_into_batches():
    calls = []

    def handler(request):
        batch = json.loads(request.content)["input"]
        calls.append(len(batch))
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": i, "embedding": [1.0, 0.0]} for i in range(len(batch))
                ]
            },
        )

    client = EmbeddingClient(client=make_client(handler), batch_size=2)
    vectors = await client.embed_texts(["a", "b", "c", "d", "e"])
    assert calls == [2, 2, 1]
    assert vectors.shape == (5, 2)


async def test_generate_returns_message_content():
    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    assert await GenerationClient(client=make_client(handler)).generate([]) == "hi"


async def test_stream_yields_tokens_until_done():
    sse = (
        'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
        'data: {"choices":[]}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(request):
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, text=sse)

    tokens = [t async for t in GenerationClient(client=make_client(handler)).stream([])]
    assert tokens == ["Hel", "lo"]


async def test_http_error_is_raised():
    client = GenerationClient(client=make_client(lambda r: httpx.Response(500)))
    with pytest.raises(httpx.HTTPStatusError):
        await client.generate([])
