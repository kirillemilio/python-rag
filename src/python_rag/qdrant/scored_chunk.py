"""Contains implementation of scored chunk struct."""

from __future__ import annotations

from typing import Generic, Literal, TypeVar

from ..dto.document import IChunk

T = TypeVar("T", bound=IChunk)


class ScoredChunk(Generic[T]):
    """Scored chunk struct.

    Attributes
    ----------
    score : float
        chunk score after retrieval.
    chunk : IChunk
        underlying chunk.
    """

    score: float
    chunk: T
    distance: Literal["euclidean", "cosine", "dot"]

    def __init__(
        self,
        chunk: T,
        score: float,
        distance: Literal["euclidean", "cosine", "dot"],
    ) -> None:
        self.chunk = chunk
        self.score = score
        self.distance = distance

    def get_distance(self) -> Literal["euclidean", "cosine", "dot"]:
        """Get distance method.

        Returns
        -------
        Literal["euclidean", "cosine", "dot"]
            distance method.
        """
        return self.distance

    def get_score(self) -> float:
        """Get score.

        Returns
        -------
        float
            chunk's score after retrieval.
        """
        return self.score

    def get_chunk(self) -> T:
        """Get underlying chunk.

        Returns
        -------
        IChunk
            underlying chunk.
        """
        return self.chunk
