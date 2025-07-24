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
    """

    messages: list[ChatMessageUnion] = Field(default_factory=list)

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
