from dataclasses import dataclass

import numpy as np


@dataclass
class DocumentChunk:
    src: str
    id: int
    content: str
    title: str | None
    author: str | None
    page_no: int | None = None
    year: int | None = None
    embedding: np.ndarray | None = None
