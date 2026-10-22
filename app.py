import json
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from localrag.Clients import EmbeddingClient, GenerationClient
from localrag.config import Settings
from localrag.DataHandling.Documents import SQLManager
from localrag.DataHandling.Embeddings import FaissManager
from localrag.rag_model import RagModel


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once at startup (before `yield`) and once at shutdown (after it)."""
    settings = Settings.from_env()

    async with (
        EmbeddingClient(settings.embedding_url) as embedder,
        GenerationClient(settings.generation_url) as generator,
    ):
        # Ask the embedding server for one vector so the FAISS dimension always matches the model.
        dim = (await embedder.embed_text("dimension probe")).shape[0]
        index = FaissManager(dim, settings.faiss_path)

        # SQLManager is a sync context manager, so a plain `with` is correct here.
        with SQLManager(settings.sql_path) as store:
            app.state.rag = RagModel(embedder, generator, index, store, settings.top_k)
            yield


app = FastAPI(title="LocalRAG", lifespan=lifespan)


def get_rag(request: Request) -> RagModel:
    return request.app.state.rag


# ---- request / response shapes (Pydantic validates incoming JSON for us) ----


class QueryIn(BaseModel):
    question: str


class SourceOut(BaseModel):
    title: str
    src: str
    page_no: int


class QueryOut(BaseModel):
    answer: str
    sources: list[SourceOut]


class DocumentIn(BaseModel):
    text: str
    src: str
    title: str = ""
    author: str = ""


def _sources(chunks) -> list[dict]:
    return [{"title": c.title, "src": c.src, "page_no": c.page_no} for c in chunks]


# ---- endpoints -------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryOut)
async def query(body: QueryIn, rag: RagModel = Depends(get_rag)):
    result = await rag.get_query(body.question)
    return {"answer": result.answer, "sources": _sources(result.sources)}


@app.post("/query/stream")
async def query_stream(body: QueryIn, rag: RagModel = Depends(get_rag)):
    chunks, tokens = await rag.stream_query(body.question)

    async def events():
        # Server-Sent Events: each message is `data: <json>` followed by a blank line.
        yield f"data: {json.dumps({'sources': _sources(chunks)})}\n\n"
        async for token in tokens:
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/documents")
async def add_document(body: DocumentIn, rag: RagModel = Depends(get_rag)):
    ids = await rag.add_text(body.text, body.src, body.title, body.author)
    return {"chunks_added": len(ids), "ids": ids}


@app.exception_handler(httpx.HTTPError)
async def llama_server_error(request: Request, exc: httpx.HTTPError):
    # A llama-server being down/slow is a *gateway* problem (502), not the client's fault.
    return JSONResponse(
        status_code=502, content={"detail": "llama-server request failed"}
    )
