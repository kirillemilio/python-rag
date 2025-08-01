"""Contains implementation of chat history structure."""

from __future__ import annotations

from typing import Sequence

from pydantic import BaseModel, Field

from .chat_message import ChatMessageUnion, RolesMappingTypedDict


class ChatHistory(BaseModel):
    """Chat history datastructure implementation.

    Attributes
    ----------
    messages : list[ChatMessageUnion]
        messages of chat history.
        Default is empty list.
    chat_id : str
        chat id associated with chat history.
    user_id : str
        user id associated with chat history.
    """

    messages: list[ChatMessageUnion] = Field(default_factory=list)
    chat_id: str
    user_id: str

    def get_chat_id(self) -> str:
        """Get chat id associated with chat history.

        Returns
        -------
        str
            chat id associated with chat history.
        """
        return self.chat_id

    def get_user_id(self) -> str:
        """Get user id associated with chat history.

        Returns
        -------
        str
            user id associated with chat history.
        """
        return self.user_id

    def get_messages(self) -> Sequence[ChatMessageUnion]:
        """Get sequence of chat messages.

        Returns
        -------
        Sequence[ChatMessageUnion]
            sequence of chat messages.
        """
        return self.messages

    def format_messages(self, roles_mapping: RolesMappingTypedDict | None = None) -> list[str]:
        """Format all underlying history messages.

        Parameters
        ----------
        roles_mapping : RolesMappingTypedDict | None
            roles mapping dictionary.
            Default is None meaning that default roles names
            will be used.

        Returns
        -------
        list[str]
            list of formatted messages.
        """
        return [
            message.format_message(roles_mapping=roles_mapping) for message in self.get_messages()
        ]
