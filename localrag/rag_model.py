
from typing import Generator, Self

class RagModel:
	def __init__(self) -> None:
		...

	def create(self) -> Self:
		...

	def _startup(self) -> str:
		...

	def add_src(self, src: str) -> str:
		...

	def get_query(self, query: str) -> str:
		...

	def stream_query(self, query: str) -> Generator:
		...
