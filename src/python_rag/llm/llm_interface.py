"""Contains implementation of llm interface model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from ..dto import ChatHistory, LLMResponse, LLMStreamItem
from ..dto.chat_message import RolesMappingTypedDict


class ILLM(ABC):
    """Large Language Model interface class implementation."""

    @abstractmethod
    def get_model_name(self) -> str:
        """Get large language model name.

        Returns
        -------
        str
            large language model name.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_response_on_query(self, query: str, temperature: float | None = None) -> LLMResponse:
        """Get response on input formatted query.

        Parameters
        ----------
        query : str
            formatted query for llm.
        temperature
        : float | None
            optional temperature for sampling.
            Default is None meaning that argmax sampling
            strategy will be used.
        """
        raise NotImplementedError()

    @abstractmethod
    def stream_response_on_query(
        self, query: str, temperature: float | None = None
    ) -> Iterator[LLMStreamItem]:
        """Stream llm response.

        Yields
        ------
        LLMStreamItem
            llm stream item containing
            text per chunk and number of tokens in chunk.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_response_on_chat(
        self,
        chat_history: ChatHistory,
        temperature: float | None = None,
        roles_mapping: RolesMappingTypedDict | None = None,
    ) -> LLMResponse:
        """Get response on input chat history.

        Parameters
        ----------
        chat_history : ChatHistory
            chat history instance containing
            chat history messages.
        temperature : float | None
            temperature of sampling.
            Default is None meaning that argmax sampling
            strategy will be used.
        roles_mapping : RolesMappingTypedDict | None
            roles mapping typed dictionary defines mapping
            for standard roles like 'user', 'assistant', 'system'.
            Default is None meaning default roles will be used.

        Returns
        -------
        LLMResponse
            llm response on input query.
        """
        raise NotImplementedError()

    @abstractmethod
    def stream_response_on_chat(
        self,
        chat_history: ChatHistory,
        temperature: float | None = None,
        roles_mapping: RolesMappingTypedDict | None = None,
    ) -> Iterator[LLMStreamItem]:
        """Stream llm response on chat history.

        Parameters
        ----------
        chat_history : ChatHistory
            chat history instance containing
            chat history messages.
        temperature : float | None
            temperature of sampling.
            Default is None meaning that argmax sampling
            strategy will be used.
        roles_mapping : RolesMappingTypedDict | None
            roles mapping typed dictionary defines mapping
            for standard roles like 'user', 'assistant', 'system'.
            Default is None meaning default roles will be used.

        Yields
        ------
        LLMStreamItem
            llm stream item containing text per chunk and
            number of tokens in chunk.
        """
        raise NotImplementedError()
