from dataclasses import dataclass
from typing import AsyncIterator

from localrag.core import DocumentChunk

from .Clients import EmbeddingClient, GenerationClient
from .DataHandling import FaissManager, SQLManager

SYSTEM_PROMPT = (
    "Answer the question using only the context below. "
    "If the context does not contain the answer, say you don't know. "
    "Cite sources by their number, like [1]."
)


@dataclass
class QueryResult:
    answer: str
    sources: list[DocumentChunk]


class RagModel:
    """Orchestrates the pipeline. It owns no clients or databases itself;
    they are passed in (dependency injection), which makes it easy to test."""

    def __init__(
        self,
        embedder: EmbeddingClient,
        generator: GenerationClient,
        index: FaissManager,
        store: SQLManager,
        top_k: int = 4,
    ) -> None:
        self._embedder = embedder
        self._generator = generator
        self._index = index
        self._store = store
        self._top_k = top_k

    # ---- ingestion -------------------------------------------------------

    async def add_text(
        self, text: str, src: str, title: str = "", author: str = "", page_no: int = 1
    ) -> list[int]:
        pieces = chunk_text(text) """REWRITE THIS!!!!!!"""
        if not pieces:
            return []

        embeddings = await self._embedder.embed_texts(pieces)
        chunks = [
            DocumentChunk(
                src=src,
                title=title,
                author=author,
                content=piece,
                page_no=page_no,
                id=0,
                embedding=emb,
            )
            for piece, emb in zip(pieces, embeddings)
        ]
        ids = self._store.add_documents(chunks)
        for chunk, id_ in zip(chunks, ids):
            chunk.id = id_
        self._index.add_embedding(chunks)
        return ids

    # ---- querying --------------------------------------------------------

    async def retrieve(self, query: str) -> list[DocumentChunk]:
        vector = await self._embedder.embed_text(query)
        ids = self._index.search_db(
            vector.reshape(1, -1), self._top_k
        )  # FAISS wants 2-D
        return self._store.get_documents(ids)

    async def get_query(self, query: str) -> QueryResult:
        chunks = await self.retrieve(query)
        answer = await self._generator.generate(self._build_messages(query, chunks))
        return QueryResult(answer=answer, sources=chunks)

    async def stream_query(
        self, query: str
    ) -> tuple[list[DocumentChunk], AsyncIterator[str]]:
        # Retrieval is awaited *before* streaming starts, so if it fails the API can
        # still return a proper error status instead of a half-sent response.
        chunks = await self.retrieve(query)
        return chunks, self._generator.stream(self._build_messages(query, chunks))

    @staticmethod
    def _build_messages(query: str, chunks: list[DocumentChunk]) -> list[dict]:
        context = (
            "\n\n".join(
                f"[{i}] ({c.title}, p.{c.page_no})\n{c.content}"
                for i, c in enumerate(chunks, start=1)
            )
            or "(no relevant context found)"
        )
        return [
            {"role": "system", "content": f"{SYSTEM_PROMPT}\n\nContext:\n{context}"},
            {"role": "user", "content": query},
        ]
