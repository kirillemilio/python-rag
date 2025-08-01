"""Contains implementation of chat message interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

from .roles_mapping import RolesMappingTypedDict

T = TypeVar('T', bound='IChatMessage')


class IChatMessage(ABC):
    """Chat message interface."""

    @abstractmethod
    def format_message(self, roles_mapping: RolesMappingTypedDict | None = None) -> str:
        """Format chat message.

        Parameters
        ----------
        roles_mapping : RolesMappingTypedDict | None
            roles mapping dictionary.
            Default is None meaning that default roles names
            will be used.

        Returns
        -------
        str
            formatted message according to roles mapping.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_text(self) -> str:
        """Get chat message text.

        Returns
        -------
        str
            chat message text.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_message_type(self) -> str:
        """Get message type.

        Returns
        -------
        str
            message type.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_message_id(self) -> str:
        """Get message id.

        Returns
        -------
        str
            message id.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_chat_id(self) -> str:
        """Get chat id.

        Returns
        -------
        str
            chat id.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_timestamp(self) -> float:
        """Get timestamp of chat message.

        Returns
        -------
        float
            chat message timestamp.
        """
        raise NotImplementedError()

    @abstractmethod
    def clone(self: T) -> T:
        """Clone chat message.

        Returns
        -------
        Self
            cloned instance of chat message.
        """
        raise NotImplementedError()
