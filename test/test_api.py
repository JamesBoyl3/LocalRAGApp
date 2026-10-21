import json
from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient

from localrag.DataHandling.Documents import SQLManager
from localrag.DataHandling.Embeddings import FaissDB
from localrag.rag_model import RagModel
from localragapi.app import app

from conftest import DIM, FakeEmbedder, FakeGenerator


@pytest.fixture
def client(tmp_path):
    # Swap the real lifespan (needs llama-server) for one that injects fakes. Using
    # `with TestClient` runs it on the same thread as the requests, like uvicorn does;
    # that matters because a sqlite3 connection can't be shared across threads.
    @asynccontextmanager
    async def fake_lifespan(app):
        with SQLManager(str(tmp_path / "t.db")) as store:
            app.state.rag = RagModel(FakeEmbedder(), FakeGenerator(),
                                     FaissDB(DIM, str(tmp_path / "t.faiss")), store, top_k=1)
            yield

    original = app.router.lifespan_context
    app.router.lifespan_context = fake_lifespan
    with TestClient(app) as test_client:
        yield test_client
    app.router.lifespan_context = original


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_add_document_then_query(client):
    r = client.post("/documents", json={"text": "rockets burn fuel to reach orbit", "src": "s.txt", "title": "Space"})
    assert r.status_code == 200 and r.json()["chunks_added"] == 1

    r = client.post("/query", json={"question": "rockets orbit?"})
    assert r.status_code == 200
    assert r.json() == {"answer": "fake answer", "sources": [{"title": "Space", "src": "s.txt", "page_no": 1}]}


def test_query_validates_input(client):
    assert client.post("/query", json={}).status_code == 422


def test_stream_sends_sources_tokens_then_done(client):
    client.post("/documents", json={"text": "rockets burn fuel to reach orbit", "src": "s.txt", "title": "Space"})
    r = client.post("/query/stream", json={"question": "rockets orbit?"})
    assert r.headers["content-type"].startswith("text/event-stream")
    events = [line[6:] for line in r.text.splitlines() if line.startswith("data: ")]
    assert json.loads(events[0])["sources"][0]["title"] == "Space"
    assert [json.loads(e)["token"] for e in events[1:-1]] == ["fake ", "streamed ", "answer"]
    assert events[-1] == "[DONE]"
