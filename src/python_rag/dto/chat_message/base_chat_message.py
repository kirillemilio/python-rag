"""Contains implementation of base chat message class."""

from __future__ import annotations

from typing import Literal, TypeVar

from pydantic import BaseModel

from .chat_message_interface import IChatMessage

T = TypeVar('T', bound='BaseChatMessage')


class BaseChatMessage(BaseModel, IChatMessage):
    """Base chat message implementation.

    Attributes
    ----------
    timestamp : float
        timestamp of message.
    text : str
        content of message.
    message_type : Literal["user", "assistant", "system"]
        message type that act as a primary discriminator field.
    """

    timestamp: float
    text: str
    message_type: Literal['user', 'assistant', 'system']

    def get_text(self) -> str:
        """Get get chat message text.

        Returns
        -------
        str
            chat message text.
        """
        return self.text

    def get_message_type(self) -> Literal['user', 'assistant', 'system']:
        """Get message type.

        Returns
        -------
        str
            message type.
        """
        return self.message_type

    def get_timestamp(self) -> float:
        """Get timestamp of message.

        Returns
        -------
        float
            utc based timestamp of message.
        """
        return self.timestamp

    def clone(self: T) -> T:
        """Get copy of origina chat message.

        Returns
        -------
        BaseChatMessage
            copy of base chat message.
        """
        return self.model_copy()
