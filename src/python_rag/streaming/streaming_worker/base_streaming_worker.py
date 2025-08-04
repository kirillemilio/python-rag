"""Contains implementation of base streaming worker."""

from __future__ import annotations

import logging

from ...dto import ChatHistory, LLMStreamItem
from ...llm import ILLM
from ..streaming_backend import (
    BaseStreamConsumer,
    BaseStreamProducer,
    EConsumerConfirmationState,
    IStreamingBackend,
)
from .streaming_worker_interface import IStreamingWorker

logger = logging.getLogger(__name__)


class BaseStreamingWorker(IStreamingWorker):
    """Base streaming worker implementation.

    Attributes
    ----------
    sampling_temperature : float | None
        sampling temperature for llm.
    llm : ILLM
        llm instance that will be used for response generation.
    streaming_backend : IStreamingBackend
        streaming backend.
    """

    sampling_temperature: float | None
    llm: ILLM

    streaming_backend: IStreamingBackend

    def __init__(
        self,
        llm: ILLM,
        streaming_backend: IStreamingBackend,
        sampling_temperature: float | None = None,
    ) -> None:
        self.sampling_temperature = sampling_temperature
        self.streaming_backend = streaming_backend
        self.llm = llm

    def get_streaming_backend(self) -> IStreamingBackend:
        """Get streaming backend associated with streaming worker.

        Redis streaming worker uses redis for stats of streaming
        update, while streaming backend can be different(e.g. Kafka).

        Returns
        -------
        IStreamingBackend
            streaming backen instance associated with redis streaming
            backend.
        """
        return self.streaming_backend

    def get_llm(self) -> ILLM:
        """Get worker underlying llm instance.

        Returns
        -------
        ILLM
            worker's underlying llm.
        """
        return self.llm

    def get_temperature(self) -> float | None:
        """Get sampling temperature for llm.

        Returns
        -------
        float | None
            sampling temperature for llm.
        """
        return self.sampling_temperature

    def get_consumer(self) -> BaseStreamConsumer[ChatHistory]:
        """Get input consumer.

        Returns
        -------
        BaseStreamConsumer[ChatHistory]
            stream consumer for given item type.
        """
        return self.streaming_backend.get_consumer(
            item_builder=ChatHistory,
            stream_name=f'input_stream:{self.llm.get_model_name()}',
            consumer_group=f'model_consumer:{self.llm.get_model_name()}',
        )

    def get_producer(self, chat_history: ChatHistory) -> BaseStreamProducer[LLMStreamItem]:
        """Get output producer.

        Parameters
        ----------
        chat_history : ChatHistory
            chat history used to get output stream producer.

        Returns
        -------
        BaseStreamProducer[LLMSTreamItem]
            stream producer for given item type.
        """
        chat_id = chat_history.get_chat_id()
        return self.get_streaming_backend().get_producer(
            item_builder=LLMStreamItem, stream_name=f'llm:chat:{chat_id}:stream'
        )

    def run(self) -> None:
        """Run llop of streaming worker."""
        stream_consumer = self.get_consumer()
        llm = self.get_llm()
        while True:
            for chat_history in stream_consumer:
                stream_producer = self.get_producer(chat_history=chat_history)

                total_tokens, total_chunks = 0, 0
                for response_item in llm.stream_response_on_chat(
                    chat_history=chat_history, temperature=self.get_temperature()
                ):
                    total_tokens += response_item.get_num_tokens()
                    total_chunks += 1

                    response_item >> stream_producer

                logger.info(f'Processed input: {chat_history}')
                EConsumerConfirmationState.ACK >> stream_consumer
