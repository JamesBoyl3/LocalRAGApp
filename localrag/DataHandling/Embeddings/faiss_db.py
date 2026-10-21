from localrag.core import DocumentChunk

import faiss
import numpy as np
import os

from typing import Protocol

import logging

logger = logging.getLogger(__name__)


class VectorDB(Protocol):
    def add_embedding(self) -> None: ...
    def search_db(self) -> list[int]: ...


class FaissManager(VectorDB):
    def __init__(self, dim: int, path: str | None = None) -> None:
        self._dim = dim
        self._index_path = path if path is not None else "/srv/embeddings.faiss"
        self._index = self._setup_index()
        self._save()

    def add_embedding(self, documents: DocumentChunk | list[DocumentChunk]) -> None:
        if isinstance(documents, DocumentChunk):
            documents = [documents]

        embeddings = np.array(
            [document.embedding for document in documents], np.float32
        )
        ids = np.array([document.id for document in documents], np.int64)

        self._index.add_with_ids(embeddings, ids)
        self._save()

    def search_db(self, query_vector: np.ndarray, top_k: int = 4) -> list[int]:
        logger.debug(query_vector.shape)
        _, ids = self._index.search(query_vector, top_k)
        ids = ids.flatten().tolist()
        logger.debug(ids)
        return [id for id in ids if id != -1]

    def _save(self) -> None:
        faiss.write_index(self._index, self._index_path)
        logger.info("Saved faiss Index")
        return

    def _setup_index(self) -> faiss.IndexIDMap:
        if os.path.exists(self._index_path):
            return self._load_index()

        return self._create_index()

    def _load_index(self) -> faiss.IndexIDMap:
        index = faiss.read_index(self._index_path)

        if not isinstance(index, faiss.IndexIDMap):
            logger.warning(
                f"The index at {self._index_path} is not of the expected type"
            )
            raise TypeError("Exisitng Index is not of the expected type")

        if index.d != self._dim:
            logger.warning(
                f"The index at {self._index_path} does not have the right dims"
            )
            raise TypeError("Existing index does not have the correct stims")

        logger.info(f"Loaded index from {self._index_path}")
        return index

    def _create_index(self) -> faiss.IndexIDMap:
        logger.info(f"Creating index at {self._index_path}")
        return faiss.IndexIDMap(faiss.IndexFlatIP(self._dim))
