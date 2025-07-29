"""Contains implementation of confirmation signals enums."""

from __future__ import annotations

from enum import Enum


class EConsumerConfirmationState(Enum):
    """Consumer confirmation state."""

    ACK = 0
    FAILED = 1
