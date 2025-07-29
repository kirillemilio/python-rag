"""Contains implementation of generic item with ack status."""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)


class ItemWithAckStatus(Generic[T]):
    """Item with ack status implementation.

    Attributes
    ----------
    item: T
        generic item.
    ack_status : bool
        ack status, bool.
    """

    item: T
    ack_status: bool

    def __init__(self, item: T, ack_status: bool) -> None:
        """Initialize item with ack status.

        Parameters
        ----------
        item : T
            generic item used to initialize struct.
        ack_status : bool
            ack status of the item.
        """
        self.item = item
        self.ack_status = ack_status

    def get_item(self) -> T:
        """Get underlying item.

        Returns
        -------
        T
            generic item, which is successor
            of pydantic BaseModel.
        """
        return self.item

    def get_ack_status(self) -> bool:
        """Get ack status.

        Returns
        -------
        bool
            ack status assoiciated with item.
        """
        return self.ack_status
