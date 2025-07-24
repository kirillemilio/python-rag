"""Contains implementation of chat message factory."""

from __future__ import annotations

from typing import Any

from .assistant_chat_message import AssistantChatMessage
from .base_chat_message import BaseChatMessage
from .system_chat_message import SystemChatMessage
from .user_chat_message import UserChatMessage


class ChatMessageFactory:
    """Chat message factory class."""

    @classmethod
    def create_message(cls, message_dict: dict[str, Any]) -> BaseChatMessage:
        """Create chat message.

        Parameters
        ----------
        message_dict : dict[str, Any]
            chat message dictionary.

        Returns
        -------
        BaseChatMessage
            chat message instance.
        """
        match message_dict.get('message_type'):
            case 'user':
                return UserChatMessage.model_validate(message_dict)
            case 'assistant':
                return AssistantChatMessage.model_validate(message_dict)
            case 'system':
                return SystemChatMessage.model_validate(message_dict)
            case _:
                raise ValueError(f"Can't parse chat message: {message_dict}")
