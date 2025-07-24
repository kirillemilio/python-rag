"""Contains implementation of system chat message."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .base_chat_message import BaseChatMessage
from .roles_mapping import RolesMappingTypedDict


class SystemChatMessage(BaseChatMessage):
    """Contains implementation of system chat message."""

    message_type: Literal['system'] = Field(default='system')

    def format_message(self, roles_mapping: RolesMappingTypedDict | None = None) -> str:
        """Format system chat message.

        Parameters
        ----------
        roles_mapping : RolesMappingTypedDict | None
            roles mapping dictionary.
            Default is None meaning that default roles names will be used.

        Returns
        -------
        str
            formatted system's message according to roles mapping.
        """
        role: str = self.get_message_type()
        if roles_mapping is not None and 'system' in roles_mapping:
            role = roles_mapping['system']
        return f'<|{role}|> {self.get_text()}'
