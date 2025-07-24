"""Contains roles mapping dictionary implementation."""

from __future__ import annotations

from typing import NotRequired, TypedDict


class RolesMappingTypedDict(TypedDict):
    """Roles mapping typed dictionary.

    Attributes
    ----------
    user : NotRequired[str]
        user role mapping, optional.
    assistant : NotRequired[str]
        assistant role mapping, optional.
    system : NotRequired[str]
        system role mapping, optional.
    """

    user: NotRequired[str]
    assistant: NotRequired[str]
    system: NotRequired[str]
