import pytest

from localrag.core.chunking import chunk_text


def test_short_text_is_one_chunk():
    assert chunk_text("hello world") == ["hello world"]


def test_empty_text_gives_no_chunks():
    assert chunk_text("   ") == []


def test_chunks_respect_size_and_overlap_content():
    text = " ".join(f"word{i}" for i in range(200))
    chunks = chunk_text(text, size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)
    # the end of one chunk reappears at the start of the next
    assert chunks[0].split()[-1] in chunks[1]


def test_no_text_is_lost():
    text = " ".join(f"word{i}" for i in range(200))
    joined = " ".join(chunk_text(text, size=100, overlap=20))
    assert all(f"word{i}" in joined for i in range(200))


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("x", size=10, overlap=10)
