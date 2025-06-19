"""Contains implementation of size struct."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Size(BaseModel):
    """Base class representing size of the image."""

    w: int = Field(ge=0)
    h: int = Field(ge=0)

    def rescale(self, scale_x: float, scale_y: float, inplace: bool = False) -> Size:
        """Rescale image size."""
        if inplace:
            self.w = int(self.w * scale_x)
            self.h = int(self.h * scale_y)
            return self
        return Size(w=int(self.w * scale_x), h=int(self.h * scale_y))
