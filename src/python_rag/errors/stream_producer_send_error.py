"""Contains implementation of stream producer send error."""

from __future__ import annotations

from .base_error import BaseError


class StreamProducerSendError(BaseError):
    """Contains implementation of stream producer send error."""

    def __init__(self, message: str = '') -> None:
        """Initialize stream producer send error with message.

        Parameters
        ----------
        message : str
            error message that will be used
            to initializer stream producer send error.
            Default is empty string.
        """
        super().__init__(subsystem='stream_producer', method='send_raw', message=message)
