import pytest

from localrag.DataHandling.Documents import SQLManager
from localrag.DataHandling.Embeddings import FaissManager
from localrag.rag_model import RagModel

from conftest import DIM, FakeEmbedder, FakeGenerator

pytestmark = pytest.mark.anyio


@pytest.fixture
def generator():
    return FakeGenerator()


@pytest.fixture
def rag(tmp_path, generator):
    # Real SQLite + real FAISS (in a temp dir); only the llama servers are faked.
    with SQLManager(str(tmp_path / "t.db")) as store:
        yield RagModel(
            FakeEmbedder(),
            generator,
            FaissManager(DIM, str(tmp_path / "t.faiss")),
            store,
            top_k=1,
        )


async def test_add_text_returns_ids_and_stores_in_both_databases(rag):
    ids = await rag.add_text("cats purr and nap all day", src="cats.txt", title="Cats")
    assert len(ids) == 1
    assert [c.id for c in rag._store.get_documents(ids)] == ids


async def test_retrieve_finds_the_relevant_chunk(rag):
    await rag.add_text("cats purr and nap all day", src="cats.txt", title="Cats")
    await rag.add_text(
        "rockets burn fuel to reach orbit", src="space.txt", title="Space"
    )
    top = await rag.retrieve("how do rockets reach orbit")
    assert [c.title for c in top] == ["Space"]


async def test_get_query_puts_context_in_prompt_and_returns_sources(rag, generator):
    await rag.add_text(
        "rockets burn fuel to reach orbit", src="space.txt", title="Space"
    )
    result = await rag.get_query("rockets orbit?")
    assert result.answer == "fake answer"
    assert result.sources[0].title == "Space"
    system_prompt = generator.last_messages[0]["content"]
    assert "rockets burn fuel" in system_prompt
    assert generator.last_messages[1] == {"role": "user", "content": "rockets orbit?"}


async def test_query_with_empty_database_still_answers(rag, generator):
    result = await rag.get_query("anything?")
    assert result.sources == []
    assert "no relevant context" in generator.last_messages[0]["content"]


async def test_stream_query_returns_sources_then_tokens(rag):
    await rag.add_text(
        "rockets burn fuel to reach orbit", src="space.txt", title="Space"
    )
    chunks, tokens = await rag.stream_query("rockets orbit?")
    assert chunks[0].title == "Space"
    assert "".join([t async for t in tokens]) == "fake streamed answer"
