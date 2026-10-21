
from dataclasses import dataclass

import numpy as np

@dataclass
class DocumentChunk:
    src: str
    title: str
    author: str
    content: str
    page_no: int
    id: int
    embedding: np.ndarray | None=None
