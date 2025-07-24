"""Contains implementation of user chat message."""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import Field

from .base_chat_message import BaseChatMessage
from .roles_mapping import RolesMappingTypedDict


class UserChatMessage(BaseChatMessage):
    """Contains implementation of user chat message."""

    message_type: Literal['user'] = Field(default='user')
    user_id: str = Field(default_factory=lambda: uuid.uuid4().hex)

    def format_message(self, roles_mapping: RolesMappingTypedDict | None = None) -> str:
        """Format user chat message.

        Parameters
        ----------
        roles_mapping : RolesMappingTypedDict | None
            roels mapping dictionary.
            Default is None meaning that default roles names
            will be used.

        Returns
        -------
        str
            formatted user message according to roles mapping.
        """
        role: str = self.get_message_type()
        if roles_mapping is not None and 'user' in roles_mapping:
            role = roles_mapping['user']
        return f'<|{role}|> {self.get_text()}'
