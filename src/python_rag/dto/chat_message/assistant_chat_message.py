"""Contains implementation of assistant chat message."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .base_chat_message import BaseChatMessage
from .roles_mapping import RolesMappingTypedDict


class AssistantChatMessage(BaseChatMessage):
    """Contains implementation of assistant chat message."""

    message_type: Literal['assistant'] = Field(default='assistant')
    model_name: str = Field(default='default')
    rating: float = Field(default=1.0)

    def format_message(self, roles_mapping: RolesMappingTypedDict | None = None) -> str:
        """Format assistant chat message.

        Parameters
        ----------
        roles_mapping : RolesMappingTypedDict | None
            roles mapping dictionary.
            Default is None meaning that default roles names will be used.

        Returns
        -------
        str
            formatted assistant's message according to roles mapping.
        """
        role: str = self.get_message_type()
        if roles_mapping is not None and 'assistant' in roles_mapping:
            role = roles_mapping['assistant']
        return f'<|{role}|> {self.get_text()}'
