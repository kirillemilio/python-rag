"""Contains implementation of class representing 2D shift vector."""

from __future__ import annotations

from typing import Dict, List, Literal, Tuple

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from .size import Size


class Shift(BaseModel):
    """
    Represent a 2D shift vector with horizontal and vertical components.

    This class provides methods for rescaling, resizing, normalization,
    and various arithmetic operations on the shift vector. Can be used to
    express movement, displacement, or vector deltas in coordinate space.

    Attributes
    ----------
    x : float
        Horizontal component of the shift (Δx).
    y : float
        Vertical component of the shift (Δy).
    """

    x: float
    y: float

    def rescale(self, scale_x: float, scale_y: float, inplace: bool = False) -> Shift:
        """
        Scale the shift vector independently along x and y axes.

        Parameters
        ----------
        scale_x : float
            Scaling factor for x component.
        scale_y : float
            Scaling factor for y component.
        inplace : bool, optional
            If True, modify this instance in-place. Default is False.

        Returns
        -------
        Shift
            The rescaled shift vector (new instance or self).
        """
        if inplace:
            self.x *= scale_x
            self.y *= scale_y
            return self
        return Shift(x=self.x * scale_x, y=self.y * scale_y)

    def resize(self, from_size: Size, to_size: Size, inplace: bool = False) -> Shift:
        """
        Resize the shift vector according to new canvas dimensions.

        Parameters
        ----------
        from_size : Size
            Original size (used as denominator).
        to_size : Size
            Target size to scale shift toward.
        inplace : bool, optional
            If True, modify this instance in-place. Default is False.

        Returns
        -------
        Shift
            The resized shift vector.
        """
        scale_x = to_size.w / from_size.w
        scale_y = to_size.h / from_size.h
        return self.rescale(scale_x, scale_y, inplace)

    def norm(self) -> Shift:
        """
        Normalize this shift vector to unit length.

        Returns
        -------
        Shift
            A new normalized shift vector.
        """
        magnitude = np.sqrt(self.x**2 + self.y**2)
        if magnitude == 0.0:
            return Shift(x=0.0, y=0.0)
        return Shift(x=self.x / magnitude, y=self.y / magnitude)

    def get_norm(self) -> float:
        """
        Compute the magnitude (Euclidean norm) of this shift.

        Returns
        -------
        float
            Magnitude of the shift vector.
        """
        return np.sqrt(self.x**2 + self.y**2)

    def get_normal(self, orient: Literal["clockwise", "counterclockwise"] = "clockwise") -> Shift:
        """
        Get a perpendicular (normal) vector to this shift.

        Parameters
        ----------
        orient : {"clockwise", "counterclockwise"}, optional
            Direction of rotation for normal. Default is "clockwise".

        Returns
        -------
        Shift
            Perpendicular shift vector.
        """
        return Shift(x=self.y, y=-self.x) if orient == "clockwise" else Shift(x=-self.y, y=self.x)

    def to_numpy(self, fmt: Literal["xy", "yx"] = "xy") -> NDArray[np.float32]:
        """
        Convert shift to NumPy array.

        Parameters
        ----------
        fmt : {"xy", "yx"}, optional
            Coordinate order. Default is "xy".

        Returns
        -------
        NDArray[np.float32]
            NumPy array representation.
        """
        return np.array(self.to_list(fmt), dtype=np.float32)

    def to_tuple(self, fmt: Literal["xy", "yx"] = "xy") -> Tuple[float, ...]:
        """
        Convert shift to tuple.

        Parameters
        ----------
        fmt : {"xy", "yx"}, optional
            Coordinate order. Default is "xy".

        Returns
        -------
        tuple of float
            Tuple representation.
        """
        return tuple(self.to_list(fmt))

    def to_list(self, fmt: Literal["xy", "yx"] = "xy") -> List[float]:
        """
        Convert shift to list.

        Parameters
        ----------
        fmt : {"xy", "yx"}, optional
            Coordinate order. Default is "xy".

        Returns
        -------
        list of float
            List representation.
        """
        return [self.x, self.y] if fmt == "xy" else [self.y, self.x]

    def to_dict(self) -> Dict[str, float]:
        """
        Convert shift to dictionary format.

        Returns
        -------
        dict
            Dictionary with keys 'x' and 'y'.
        """
        return {"x": self.x, "y": self.y}

    # Arithmetic operations
    def add(self, other: Shift) -> Shift:
        """Return elementwise sum of two shift vectors."""
        return Shift(x=self.x + other.x, y=self.y + other.y)

    def sub(self, other: Shift) -> Shift:
        """Return elementwise difference of two shift vectors."""
        return Shift(x=self.x - other.x, y=self.y - other.y)

    def mul(self, factor: float) -> Shift:
        """Return shift scaled by a scalar."""
        return Shift(x=self.x * factor, y=self.y * factor)

    def div(self, factor: float) -> Shift:
        """Return shift divided by a scalar."""
        return self.mul(1.0 / factor)

    def dot(self, other: Shift) -> float:
        """Return dot product of this and another shift."""
        return self.x * other.x + self.y * other.y

    def abs(self) -> Shift:
        """Return shift with absolute value of each component."""
        return Shift(x=abs(self.x), y=abs(self.y))

    def neg(self) -> Shift:
        """Return shift with negated components."""
        return Shift(x=-self.x, y=-self.y)

    # Python magic methods
    def __neg__(self) -> Shift:
        return self.neg()

    def __add__(self, other: Shift) -> Shift:
        return self.add(other)

    def __sub__(self, other: Shift) -> Shift:
        return self.sub(other)

    def __mul__(self, factor: float) -> Shift:
        return self.mul(factor)

    def __truediv__(self, factor: float) -> Shift:
        return self.div(factor)

    def __matmul__(self, other: Shift) -> float:
        return self.dot(other)

    # In-place ops (PyTorch-style)
    def add_(self, other: Shift) -> Shift:
        """In-place addition."""
        self.x += other.x
        self.y += other.y
        return self

    def sub_(self, other: Shift) -> Shift:
        """In-place subtraction."""
        self.x -= other.x
        self.y -= other.y
        return self

    def mul_(self, factor: float) -> Shift:
        """In-place multiplication by scalar."""
        self.x *= factor
        self.y *= factor
        return self

    def div_(self, factor: float) -> Shift:
        """In-place division by scalar."""
        self.x /= factor
        self.y /= factor
        return self

    def abs_(self) -> Shift:
        """In-place absolute value of components."""
        self.x = abs(self.x)
        self.y = abs(self.y)
        return self

    def __iadd__(self, other: Shift) -> Shift:
        return self.add_(other)

    def __isub__(self, other: Shift) -> Shift:
        return self.sub_(other)

    def __imul__(self, factor: float) -> Shift:
        return self.mul_(factor)

    def __itruediv__(self, factor: float) -> Shift:
        return self.div_(factor)

    def clone(self) -> Shift:
        """Create a deep copy of the current shift vector."""
        return Shift(x=self.x, y=self.y)

    def __str__(self) -> str:
        """Return short string representation of the shift."""
        return f"Shift(x={self.x:.2f}, y={self.y:.2f})"

    def __repr__(self) -> str:
        """Return detailed string representation of the shift."""
        return self.__str__()
