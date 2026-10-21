import faiss
import numpy as np
import pytest

from localrag.core import DocumentChunk
from localrag.DataHandling.Embeddings import FaissDB  # adjust to your module path

DIM = 4


def make_chunk(id: int, embedding: np.ndarray) -> DocumentChunk:
    # Adjust to DocumentChunk's real constructor (it may need text, metadata, etc.)
    return DocumentChunk(
            src = "https:example.pdf", 
            title = "Examples",
            author = "Joe Schmitt",
            content = "A place for examples",
            page_no = 1,
            id=id, 
            embedding=embedding
            )


@pytest.fixture
def index_path(tmp_path):
    return str(tmp_path / "test.faiss")


@pytest.fixture
def db(index_path):
    return FaissDB(DIM, index_path)


@pytest.fixture
def basis_chunks():
    """Four chunks whose embeddings are the unit vectors along each axis."""
    eye = np.eye(DIM, dtype=np.float32)
    return [make_chunk(10 * (i + 1), eye[i]) for i in range(DIM)]  # ids 10,20,30,40


# --- creating / loading ---

def test_creates_file_when_missing(index_path):
    import os
    assert not os.path.exists(index_path)
    FaissDB(DIM, index_path)
    assert os.path.exists(index_path)


def test_loads_existing_index(index_path, basis_chunks):
    first = FaissDB(DIM, index_path)
    first.add_embedding(basis_chunks)

    second = FaissDB(DIM, index_path)  # should load, not recreate
    query = np.eye(DIM, dtype=np.float32)[2:3]
    assert second.search_db(query, top_k=1) == [30]


def test_wrong_dim_raises(index_path):
    FaissDB(DIM, index_path)
    with pytest.raises(TypeError):
        FaissDB(DIM + 1, index_path)


def test_wrong_index_type_raises(index_path):
    faiss.write_index(faiss.IndexFlatIP(DIM), index_path)  # not an IndexIDMap
    with pytest.raises(TypeError):
        FaissDB(DIM, index_path)


# --- add_embedding / search_db ---

def test_add_single_chunk(db):
    db.add_embedding(make_chunk(7, np.eye(DIM, dtype=np.float32)[0]))
    query = np.eye(DIM, dtype=np.float32)[0:1]
    assert db.search_db(query, top_k=1) == [7]


def test_search_returns_nearest_first(db, basis_chunks):
    db.add_embedding(basis_chunks)
    # Mostly axis 0, a bit of axis 1 -> id 10 should beat id 20
    query = np.array([[0.9, 0.1, 0.0, 0.0]], dtype=np.float32)
    assert db.search_db(query, top_k=2) == [10, 20]


def test_search_on_empty_db_returns_empty_list(db):
    query = np.ones((1, DIM), dtype=np.float32)
    assert db.search_db(query) == []


def test_search_filters_missing_results(db, basis_chunks):
    db.add_embedding(basis_chunks[:2])
    query = np.ones((1, DIM), dtype=np.float32)
    results = db.search_db(query, top_k=4)  # asks for 4, only 2 exist
    assert sorted(results) == [10, 20]


# --- _save ---

def test_add_embedding_persists_to_disk(index_path, db, basis_chunks):
    db.add_embedding(basis_chunks)
    assert faiss.read_index(index_path).ntotal == len(basis_chunks)


def test_save_recreates_deleted_file(index_path, db):
    import os
    os.remove(index_path)
    db._save()
    assert os.path.exists(index_path)
