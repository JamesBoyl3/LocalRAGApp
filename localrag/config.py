import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """All the knobs in one place, read from environment variables.

    Your systemd services already load /etc/default/localrag-server.env,
    so setting these there configures the API too.
    """

    embedding_url: str = "http://127.0.0.1:8081"
    generation_url: str = "http://127.0.0.1:8080"
    sql_path: str | None = None  # None -> SQLManager's default (/srv/documents.db)
    faiss_path: str | None = None  # None -> FaissDB's default (/srv/embeddings.faiss)
    top_k: int = 4

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            embedding_url=os.environ.get("LOCALRAG_EMBEDDING_URL", cls.embedding_url),
            generation_url=os.environ.get(
                "LOCALRAG_GENERATION_URL", cls.generation_url
            ),
            sql_path=os.environ.get("LOCALRAG_SQL_PATH"),
            faiss_path=os.environ.get("LOCALRAG_FAISS_PATH"),
            top_k=int(os.environ.get("LOCALRAG_TOP_K", cls.top_k)),
        )
