import numpy as np
import pytest

from localrag.core import DocumentChunk
from localrag.DataHandling.Documents.citations import (
    APAGenerator,
)


def make_doc(
    author: str = "Smith, J.",
    year: int = 2020,
    title: str = "A Study of Things",
) -> DocumentChunk:
    # Adjust to DocumentChunk's real constructor (it needs a `year` field)
    return DocumentChunk(
        src="https://example.pdf",
        title=title,
        author=author,
        year=year,
        content="some content",
        page_no=1,
        id=1,
        embedding=np.zeros(4, dtype=np.float32),
    )


# --- bibliography ---


def test_bibliography_returns_a_string():
    assert isinstance(APAGenerator.bibliography("answer", [make_doc()]), str)


def test_bibliography_can_be_called_without_an_instance():
    # It's a staticmethod, so no APAGenerator() is needed
    APAGenerator.bibliography("answer", [make_doc()])


def test_bibliography_single_document_exact_output():
    doc = make_doc(author="Smith, J.", year=2020, title="A Study of Things")
    result = APAGenerator.bibliography("The answer.", [doc])
    assert result == "The answer.\n\nSmith, J. (2020) A Study of Things"


def test_bibliography_keeps_original_text_at_the_start():
    result = APAGenerator.bibliography("The answer.", [make_doc()])
    assert result.startswith("The answer.")


def test_bibliography_separates_text_and_references_with_blank_line():
    result = APAGenerator.bibliography("The answer.", [make_doc()])
    text_part, _, reference_part = result.partition("\n\n")
    assert text_part == "The answer."
    assert reference_part != ""


def test_bibliography_puts_each_document_on_its_own_line():
    docs = [
        make_doc(author="Smith, J.", year=2020, title="First"),
        make_doc(author="Jones, A.", year=2018, title="Second"),
        make_doc(author="Lee, K.", year=2022, title="Third"),
    ]
    result = APAGenerator.bibliography("answer", docs)
    references = result.split("\n\n", 1)[1].split("\n")
    assert references == [
        "Smith, J. (2020) First",
        "Jones, A. (2018) Second",
        "Lee, K. (2022) Third",
    ]


@pytest.mark.parametrize("text", ["", "short", "line one\nline two", "Unicode: café ✓"])
def test_bibliography_preserves_any_text(text):
    result = APAGenerator.bibliography(text, [make_doc()])
    assert result.startswith(f"{text}\n\n")


def test_bibliography_does_not_modify_the_documents_list():
    docs = [make_doc(title="First"), make_doc(title="Second")]
    APAGenerator.bibliography("answer", docs)
    assert [d.title for d in docs] == ["First", "Second"]


# --- behaviour I'd expect but the current code doesn't have ---


@pytest.mark.xfail(
    strict=True, reason="empty list currently leaves a trailing blank line"
)
def test_bibliography_with_no_documents_returns_text_unchanged():
    assert APAGenerator.bibliography("answer", []) == "answer"


@pytest.mark.xfail(strict=True, reason="duplicates are not removed yet")
def test_bibliography_lists_a_repeated_document_once():
    doc = make_doc()
    result = APAGenerator.bibliography("answer", [doc, doc])
    assert result.count("A Study of Things") == 1


# --- in_text ---


@pytest.mark.skip(reason="in_text is not implemented yet")
def test_in_text_placeholder(): ...
