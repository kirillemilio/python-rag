"""Contains implementation of chat message factory."""

from __future__ import annotations


from .assistant_chat_message import AssistantChatMessage
from .base_chat_message import BaseChatMessage, BaseChatMessageTypedDict
from .system_chat_message import SystemChatMessage
from .user_chat_message import UserChatMessage


class ChatMessageFactory:
    """Chat message factory class."""

    @classmethod
    def create_message(cls, message_dict: BaseChatMessageTypedDict) -> BaseChatMessage:
        """Create chat message.

        Parameters
        ----------
        message_dict : BaseChatMessageTypedDict
            chat message dictionary.
            Must contain at least following fields:
                - timestamp: float
                - text: str
                - message_type: str
                - message_id: str
                - chat_id: str

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


ChatMessageUnion = UserChatMessage | AssistantChatMessage | SystemChatMessage