"""Contains implementation of base Point class."""

from __future__ import annotations

from typing import Dict, List, Literal, Tuple

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from .shift import Shift
from .size import Size


class Point(BaseModel):
    """Point class representing x, y coordinates."""

    x: float
    y: float

    def rescale(self, scale_x: float, scale_y: float, inplace: bool = False) -> Point:
        """Rescale point by given scale factors along x and y axes.

        Parameters
        ----------
        scale_x : float
            Scale factor along the x-axis.
        scale_y : float
            Scale factor along the y-axis.
        inplace : bool
            If True, modifies the point in-place.

        Returns
        -------
        Point
            The rescaled point (might be self if inplace is True).
        """
        if inplace:
            self.x = self.x * scale_x
            self.y = self.y * scale_y
            return self
        return Point(x=self.x * scale_x, y=self.y * scale_y)

    def distance(self, other: Point) -> float:
        """Calculate Euclidean distance between two points.

        Parameters
        ----------
        other : Point
            The other point to calculate the distance to.

        Returns
        -------
        float
            Euclidean distance between the points.
        """
        dx = (self.x - other.x) ** 2
        dy = (self.y - other.y) ** 2
        return np.sqrt(dx + dy)

    def move(self, v: Shift, dt: float = 1.0, inplace: bool = False) -> Point:
        """Move point using velocity.

        Parameters
        ----------
        v : Shift
            shift vector that will be used for moving.
        dt : float
            timedelta to move by velocity. Default is 1.0.
        inplace : bool
            Whether to move point inplace(update) or create
            new point. Default is False meaning that
            new point will be created.

        Returns
        -------
        Point
        """
        if inplace:
            self.x = self.x + v.x * dt
            self.y = self.y + v.y * dt
            return self
        return Point(x=self.x + v.x * dt, y=self.y + v.y * dt)

    def resize(self, from_size: Size, to_size: Size, inplace: bool = False) -> Point:
        """Resize point from one resolution size to another.

        Parameters
        ----------
        from_size : Size
            Original size.
        to_size : Size
            Target size.
        inplace : bool
            If True, resizes the point in-place.

        Returns
        -------
        Point
            The resized point.
        """
        scale_x, scale_y = to_size.w / from_size.w, to_size.w / from_size.w
        return self.rescale(scale_x=scale_x, scale_y=scale_y, inplace=inplace)

    def to_numpy(self, fmt: Literal["xy", "yx"] = "xy") -> NDArray[np.float32]:
        """Convert point coordinates to numpy array.

        Parameters
        ----------
        fmt : Literal["xy", "yx"]
            format to use.

        Returns
        -------
        NDArray[np.float32]
            numpy array of dtype np.float32 and shape (2, )
            representing point coordinates in xy or yx order.
        """
        return np.array(self.to_list(fmt=fmt), dtype=np.float32)

    def to_tuple(self, fmt: Literal["xy", "yx"] = "xy") -> Tuple[float, ...]:
        """Convert point coordinates to tuple of floats.

        Parameters
        ----------
        fmt : Literal["xy", "yx"]
            format to use.

        Returns
        -------
        Tuple[float, float]
            tuple of point coordinates in xy or yx order.
        """
        return tuple(self.to_list(fmt=fmt))

    def to_list(self, fmt: Literal["xy", "yx"] = "xy") -> List[float]:
        """Convert point coordinates to list of floats.

        Parameters
        ----------
        fmt : Literal["xy", "yx"]
            format to use.

        Returns
        -------
        List[float]
            list of point coordinates in xy or yx order.
        """
        return [self.x, self.y] if fmt == "xy" else [self.y, self.x]

    def to_dict(self) -> Dict[str, float]:
        """Convert point coordinates to dict.

        Dictionary keys are 'x' and 'y'.

        Returns
        -------
        Dict[str, float]
            dictionary with keys 'x' and 'y'
            and values corresponding to point's
            coordinates.
        """
        return {"x": self.x, "y": self.y}

    def clone(self) -> Point:
        """Create a deep copy of this point.

        Returns
        -------
        Point
            A new instance of Point with the same coordinates.
        """
        return Point(x=self.x, y=self.y)
