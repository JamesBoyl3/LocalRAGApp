import os
import sqlite3

import numpy as np
import pytest

from localrag.core.documents import DocumentChunk
from localrag.DataHandling.Documents.SQL_db import SQLManager  # adjust


def make_doc(content: str | None = "some text", page_no: int = 1, id: int = 0) -> DocumentChunk:
    # Adjust to DocumentChunk's real constructor. The id is ignored by the DB.
    return DocumentChunk(
        src="https://example.pdf",
        title="Example",
        author="Joe Schmitt",
        content=content,
        page_no=page_no,
        id=id,
        embedding=np.zeros(4, dtype=np.float32),
    )


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture
def db(db_path):
    with SQLManager(db_path) as database:
        yield database  # test runs here, then __exit__ runs as teardown


# --- creating / loading / closing ---

def test_creates_file_when_missing(db_path):
    assert not os.path.exists(db_path)
    with SQLManager(db_path):
        pass
    assert os.path.exists(db_path)


def test_creates_documents_table(db_path):
    with SQLManager(db_path):
        pass
    con = sqlite3.connect(db_path)
    tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    con.close()
    assert "documents" in tables


def test_data_survives_reopening(db_path):
    with SQLManager(db_path) as first:
        doc_id = first.add_document(make_doc("persistent text"))

    with SQLManager(db_path) as second:  # should open the existing file, not wipe it
        docs = second.get_documents(doc_id)

    assert docs[0].content == "persistent text"


def test_connection_closed_after_exit(db_path):
    with SQLManager(db_path) as database:
        pass
    with pytest.raises(sqlite3.ProgrammingError):
        database.get_documents([1])


# --- add_document ---

def test_add_document_returns_int_id(db):
    assert isinstance(db.add_document(make_doc()), int)


def test_add_document_ignores_chunks_own_id(db):
    assert db.add_document(make_doc(id=999)) != 999


def test_added_document_can_be_read_back(db):
    doc_id = db.add_document(make_doc("hello", page_no=3))
    doc = db.get_documents(doc_id)[0]
    assert isinstance(doc, DocumentChunk)
    assert doc.id == doc_id
    assert doc.content == "hello"
    assert doc.page_no == 3
    assert doc.title == "Example"
    assert doc.author == "Joe Schmitt"
    assert doc.src == "https://example.pdf"


def test_special_characters_are_stored_safely(db):
    nasty = "Robert'); DROP TABLE documents;--"
    doc_id = db.add_document(make_doc(nasty))
    assert db.get_documents(doc_id)[0].content == nasty


# --- add_documents ---

def test_add_documents_returns_one_unique_id_per_document(db):
    ids = db.add_documents([make_doc("a"), make_doc("b"), make_doc("c")])
    assert len(ids) == 3
    assert len(set(ids)) == 3
    assert all(isinstance(i, int) for i in ids)


def test_add_documents_accepts_single_chunk(db):
    ids = db.add_documents(make_doc("solo"))
    assert len(ids) == 1


def test_add_documents_ids_match_input_order(db):
    ids = db.add_documents([make_doc("a"), make_doc("b"), make_doc("c")])
    assert [d.content for d in db.get_documents(ids)] == ["a", "b", "c"]


def test_add_documents_empty_list_returns_empty_list(db):
    assert db.add_documents([]) == []


def test_ids_keep_increasing_across_calls(db):
    first = db.add_documents([make_doc("a"), make_doc("b")])
    second = db.add_documents([make_doc("c")])
    assert second[0] > max(first)


def test_failed_batch_adds_nothing(db):
    with pytest.raises(sqlite3.IntegrityError):
        db.add_documents([make_doc("good"), make_doc(content=None)])  # NOT NULL violation
    assert db.get_documents([1, 2]) == []  # the good one was rolled back too


# --- get_documents ---

def test_get_documents_preserves_requested_order(db):
    ids = db.add_documents([make_doc("a"), make_doc("b"), make_doc("c")])
    docs = db.get_documents(list(reversed(ids)))
    assert [d.content for d in docs] == ["c", "b", "a"]


def test_get_documents_skips_unknown_ids(db):
    doc_id = db.add_document(make_doc("a"))
    assert [d.id for d in db.get_documents([doc_id, 12345])] == [doc_id]


def test_get_documents_empty_list(db):
    assert db.get_documents([]) == []
