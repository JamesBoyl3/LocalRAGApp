import logging
import sqlite3
from typing import Self

from localrag.core.documents import DocumentChunk
from .interfaces import DocumentManager

logger = logging.getLogger(__name__)


class SQLManager(DocumentManager):
    def __init__(self, path: str | None = None) -> None:
        self._db_path = path if path is not None else "/srv/documents.db"

    def __enter__(self) -> Self:
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._cursor = self._conn.cursor()
        self._create_db()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if hasattr(self, "_conn"):
            if exc_type is None:
                self._conn.commit()
            else:
                self._conn.rollback()
            self._conn.close()

    def _create_db(self) -> None:
        self._cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                src TEXT,
                title TEXT,
                author TEXT,
                content TEXT NOT NULL,
                page_no INTEGER,
                visited_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.commit()

    def add_document(self, document: DocumentChunk) -> int:
        return self.add_documents([document])[0]

    def add_documents(self, documents: list[DocumentChunk] | DocumentChunk) -> list[int]:
        if isinstance(documents, DocumentChunk):
            documents = [documents]

        ids: list[int] = []
        with self._conn:  # one transaction: all rows are kept, or none
            for document in documents:
                self._cursor.execute(
                    """
                    INSERT INTO documents (src, title, author, content, page_no)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        document.src,
                        document.title,
                        document.author,
                        document.content,
                        document.page_no,
                    ),
                )
                ids.append(self._cursor.lastrowid)

        logger.info(f"Added {len(ids)} documents")
        return ids

    def get_documents(self, ids: list[int] | int) -> list[DocumentChunk]:
        if isinstance(ids, int):
            ids = [ids]

        if not ids:  # in case an empty list is passed
            return []

        placeholders = ",".join("?" for _ in ids)
        self._cursor.execute(
            f"SELECT * FROM documents WHERE id IN ({placeholders})", ids
        )
        by_id = {row["id"]: row for row in self._cursor.fetchall()}

        # SQL doesn't return rows in the order of the IN list, so re-order
        # them to match `ids` (which is the FAISS ranking)
        rows = [by_id[i] for i in ids if i in by_id]

        return [
            DocumentChunk(
                src=row["src"],
                title=row["title"],
                author=row["author"],
                content=row["content"],
                page_no=row["page_no"],
                id=row["id"],
            )
            for row in rows
        ]
