"""Contains implementatin of llm stream item class."""

from __future__ import annotations

from pydantic import BaseModel


class LLMStreamItem(BaseModel):
    """Implements llm stream item."""

    content: str
    num_tokens: int
    is_final: bool

    def is_done(self) -> bool:
        """Check whether this content if final.

        Returns
        -------
        bool
            True if content is final False otherwise.
        """
        return self.is_final

    def get_content(self) -> str:
        """Get content.

        Returns
        -------
        str
            content of stream item.
        """
        return self.content

    def get_num_tokens(self) -> int:
        """Get number of generated tokens.

        Returns
        -------
        int
            number of generated tokens.
        """
        return self.num_tokens
