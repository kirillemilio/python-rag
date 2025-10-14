"""Contains implementation of base chat message class."""

from __future__ import annotations

from typing import Literal, TypeVar, TypedDict, Required

from pydantic import BaseModel

from .chat_message_interface import IChatMessage

T = TypeVar('T', bound='BaseChatMessage')


class BaseChatMessageTypedDict(TypedDict):
    """Base chat message typed dictinoary definition.
    
    Attributes
    ----------
    timestamp : Required[float]
        timestamp of message.
    text : Required[str]
        content of message.
    message_type : Required[Literal["user", "assistant", "system"]]
        message type that act as a primary discriminator field.
    message_id : Required[str]
        message id.
    chat_id : Required[str]
        chat id.
    """

    timestamp: Required[float]
    text: Required[str]
    message_type: Required[Literal["user", "assistant", "system"]]
    message_id: Required[str]
    chat_id: Required[str]


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
    message_id : str
        message id.
    chat_id : str
        chat id.
    """

    timestamp: float
    text: str
    message_type: Literal['user', 'assistant', 'system']
    message_id: str
    chat_id: str

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

    def get_message_id(self) -> str:
        """Get message id.

        Returns
        -------
        str
            message id.
        """
        return self.message_id

    def get_chat_id(self) -> str:
        """Get chat id.

        Returns
        -------
        str
            chat id.
        """
        return self.chat_id

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
