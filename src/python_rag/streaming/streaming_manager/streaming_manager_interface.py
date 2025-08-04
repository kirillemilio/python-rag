"""Contains implementation of streaming manager interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from ...dto import ChatHistory, LLMStreamItem


class IStreamingManager(ABC):
    @abstractmethod
    def schedule_to_worker(self, chat_history: ChatHistory) -> None:
        """Schedule chat history to worker.

        Parameters
        ----------
        chat_history : ChatHistory
            chat history to process with worker.
        """
        raise NotImplementedError()

    @abstractmethod
    def read_stream_from_worker(self) -> Iterator[LLMStreamItem]:
        """Generate items from worker stream."""
