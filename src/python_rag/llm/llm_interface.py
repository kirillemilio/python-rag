"""Contains implementation of llm interface model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from ..dto import ChatHistory, LLMResponse


class ILLM(ABC):
    """Large Language Model interface class implementation."""

    @abstractmethod
    def get_response_on_query(self, query: str, temperature: float | None = None) -> LLMResponse:
        raise NotImplementedError()

    @abstractmethod
    def stream_response_on_query(
        self, query: str, temperature: float | None = None
    ) -> Iterator[str]:
        raise NotImplementedError()

    @abstractmethod
    def get_response_on_chat(
        self, chat_history: ChatHistory, temperature: float | None = None
    ) -> LLMResponse:
        raise NotImplementedError()

    @abstractmethod
    def stream_response_on_chat(
        self, chat_history: ChatHistory, temperature: float | None = None
    ) -> Iterator[str]:
        raise NotImplementedError()
