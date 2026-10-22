from localrag.core import DocumentChunk

from typing import Protocol


class CitiationGenerator(Protocol):
    @staticmethod
    def in_text(text: str, documents: list[DocumentChunk]) -> str: ...
    @staticmethod
    def bibliography(text: str, documents: list[DocumentChunk]) -> str: ...


class APAGenerator(CitiationGenerator):
    @staticmethod
    def in_text(text: str, documents: list[DocumentChunk]) -> str: ...

    @staticmethod
    def bibliography(text: str, documents: list[DocumentChunk]) -> str:
        answer = text

        bibliography = "\n".join(
            [
                f"{document.author} ({document.year}) {document.title}"
                for document in documents
            ]
        )

        return f"{text}\n\n{bibliography}"
