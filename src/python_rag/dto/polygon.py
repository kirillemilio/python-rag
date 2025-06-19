"""Contains implementation of polygon and bounding box classes implementation."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, TypedDict, Union, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel
from shapely import Point as ShapelyPoint  # type: ignore
from shapely import Polygon as ShapelyPolygon
from shapely.affinity import translate  # type: ignore

from .point import Point
from .shift import Shift
from .size import Size

LOGGER = logging.getLogger(__name__)


class InvalidBoxError(Exception):
    """Invalid bounding box exception.

    Typically raise if bounding box
    has invalid boundaries.
    """

    pass


class Region(ABC):
    """Abstract base class representing a geometric region within a frame.

    Provides an interface for geometric operations on the region.
    """

    @abstractmethod
    def get_cxcy(self) -> Point:
        """
        Get the center (cx, cy) of the region.

        Returns
        -------
        Point
            The center point of the region.
        """
        raise NotImplementedError()

    @abstractmethod
    def rescale(self, scale_x: float, scale_y: float, inplace: bool) -> Region:
        """
        Rescale the region by specified scale factors along x and y axes.

        Parameters
        ----------
        scale_x : float
            Scale factor along the x-axis.
        scale_y : float
            Scale factor along the y-axis.
        inplace : bool
            If True, modifies the region in-place.

        Returns
        -------
        Region
            The rescaled region (might be self if inplace is True).
        """
        raise NotImplementedError()

    def resize(self, from_size: Size, to_size: Size, inplace: bool = False) -> Region:
        """
        Resize the region based on the change from one resolution to another.

        Parameters
        ----------
        from_size : Size
            The original size.
        to_size : Size
            The new size.
        inplace : bool, optional
            If True, resizes the region in place. Defaults to False.

        Returns
        -------
        Region
            The resized region.
        """
        raise NotImplementedError()

    @abstractmethod
    def move(self, v: Shift, dt: float = 1.0, inplace: bool = False) -> Region:
        """
        Move the region by a velocity v over time dt.

        Parameters
        ----------
        v : Shift
            The shift at which to move the region.
        dt : float
            The time interval over which to move.
            Default is 1.0.
        inplace : bool
            If True, modifies the region in-place.
            Default is False.

        Returns
        -------
        Region
            The moved region (might be self if inplace is True).
        """
        raise NotImplementedError()

    @abstractmethod
    def get_area(self) -> float:
        """
        Calculate the area of the region.

        Returns
        -------
        float
            The area of the region.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_inter(self, other: Region) -> float:
        """
        Compute the intersection area with another region.

        Parameters
        ----------
        other : Region
            The other region to compute the intersection with.

        Returns
        -------
        float
            The area of intersection.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_union(self, other: Region) -> float:
        """
        Compute the union area with another region.

        Parameters
        ----------
        other : Region
            The other region to compute the union with.

        Returns
        -------
        float
            The area of the union.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_iou(self, other: Region) -> float:
        """
        Compute the Intersection over Union (IoU) with another region.

        Parameters
        ----------
        other : Region
            The other region to compute the IoU with.

        Returns
        -------
        float
            The IoU value.
        """
        raise NotImplementedError()

    @abstractmethod
    def contains(self, other: Union[Region, Point]) -> bool:
        """
        Determine if the region completely contains another region.

        Parameters
        ----------
        other : Union[Region, Point]
            The other region or point to check containment of.

        Returns
        -------
        bool
            True if this region contains the other, False otherwise.
        """
        raise NotImplementedError()

    @abstractmethod
    def clone(self) -> Region:
        """
        Create a deep copy of this region.

        Returns
        -------
        Region
            A deep copy of this region.
        """
        raise NotImplementedError()

    @abstractmethod
    def to_polygon(self) -> Polygon:
        """Get polygon from region.

        Retuns
        ------
        Polygon
            polygon corresponding to region.
        """
        raise NotImplementedError()

    @abstractmethod
    def to_bbox(self) -> BBox:
        """Get bounding box from region.

        Returns
        -------
        BBox
            bbox corresponding to bbox.
        """
        raise NotImplementedError()


class XYXYDict(TypedDict):
    """Dictionary type for representing a bounding box in XYXY format.

    Attributes
    ----------
    x1 : float
        The x-coordinate of the top-left corner.
    y1 : float
        The y-coordinate of the top-left corner.
    x2 : float
        The x-coordinate of the bottom-right corner.
    y2 : float
        The y-coordinate of the bottom-right corner.
    """

    x1: float
    y1: float
    x2: float
    y2: float


class CXYWHDict(TypedDict):
    """
    Dictionary type for representing a bounding box in center x, center y, width, and height format.

    Attributes
    ----------
    cx : float
        The x-coordinate of the center.
    cy : float
        The y-coordinate of the center.
    w : float
        The width of the bounding box.
    h : float
        The height of the bounding box.
    """

    cx: float
    cy: float
    w: float
    h: float


class TLWHDict(TypedDict):
    """Dictionary type for representing a bounding box in top-left width-height format.

    Attributes
    ----------
    x : float
        The x-coordinate of the top-left corner.
    y : float
        The y-coordinate of the top-left corner.
    w : float
        The width of the bounding box.
    h : float
        The height of the bounding box.
    """

    x: float
    y: float
    w: float
    h: float


class CXYAHDict(TypedDict):
    """
    Dictionary type for representing a bounding box in center x, center y, ar, and height format.

    Attributes
    ----------
    cx : float
        The x-coordinate of the center.
    cy : float
        The y-coordinate of the center.
    a : float
        The aspect ratio (width/height).
    h : float
        The height of the bounding box.
    """

    cx: float
    cy: float
    a: float
    h: float


class TLAHDict(TypedDict):
    """
    Dictionary type for representing a bounding box in top-left, aspect ratio, and height format.

    Attributes
    ----------
    x : float
        The x-coordinate of the top-left corner.
    y : float
        The y-coordinate of the top-left corner.
    a : float
        The aspect ratio (width/height).
    h : float
        The height of the bounding box.
    """

    x: float
    y: float
    a: float
    h: float


BBoxDict = Union[XYXYDict, CXYWHDict, TLWHDict, CXYAHDict, TLAHDict]


class Polygon(Region):
    """A class that encapsulates a geometric polygon using the Shapely library.

    It offers methods to manipulate and interact with the polygon, including
    resizing, moving, and performing geometric calculations like intersections
    and unions with other polygons or bounding boxes.

    It provides the following key functionalities:
    - **Conversion** between different geometric formats such as bounding boxes.
    - **Rescaling and resizing** to adjust the size according to specific scale factors
      or in response to changes in resolution.
    - **Movement** via specified velocities to simulate physical translation over time.
    - **Geometric calculations** such as area, intersection, union, and containment checks
      to support spatial analysis and operations.

    Attributes
    ----------
    _polygon : ShapelyPolygon
        The internal representation of the polygon using Shapely's Polygon object.

    Methods
    -------
    __init__(*args, **kwargs)
        Initialize a new Polygon with defined boundary points.
    to_bbox()
        Convert the polygon to a bounding box.
    from_bbox(bbox)
        Create a polygon from a bounding box.
    to_numpy()
        Convert polygon points to a numpy array.
    from_numpy(np_array)
        Create a polygon from a numpy array of point coordinates.
    to_list()
        Convert the polygon points to a list of lists.
    from_list(points_list)
        Create a polygon from a list of point coordinates.
    to_dict()
        Serialize the polygon to a dictionary format.
    from_dict(data)
        Deserialize a polygon from dictionary format.
    rescale(scale_x, scale_y, inplace=False)
        Rescale the polygon by given scale factors.
    resize(from_size, to_size, inplace=False)
        Resize the polygon based on dimension changes.
    move(v, dt, inplace=False)
        Translate the polygon by a specified velocity over a given time interval.
    __eq__(other)
        Check equality with another polygon or bounding box.
    equals(other)
        Determine if exactly equal to another geometric object.
    almost_equals(other, decimal)
        Determine if approximately equal to another geometric object.
    equals_exact(other, tolerance)
        Check if exactly equal within a specified tolerance.
    clone()
        Create a deep copy of the polygon instance.

    Examples
    --------
    Creating a polygon using a list of points:
    >>> poly = Polygon([(0, 0), (1, 1), (1, 0)])

    Creating a polygon using separate coordinate sequences:
    >>> poly = Polygon([(0, 1, 1), (0, 1, 0)])
    """

    _polygon: ShapelyPolygon

    def __init__(self, *args, **kwargs):
        """Initialize a new Polygon instance with the given points that define polygon's boundary.

        Parameters
        ----------
        *args : tuple
            Coordinates that define the polygon. Can be a sequence of (x, y) tuples
            or separate x and y coordinate sequences.
        **kwargs : dict
            Additional keyword arguments to pass to ShapelyPolygon constructor.

        Examples
        --------
        Creating a polygon using a list of points:
        >>> poly = Polygon([(0, 0), (1, 1), (1, 0)])

        Creating a polygon using separate coordinate sequences:
        >>> poly = Polygon([(0, 1, 1), (0, 1, 0)])
        """
        self._polygon = ShapelyPolygon(*args, **kwargs)

    def get_raw_polygon(self) -> ShapelyPolygon:
        """Get raw shaply polygon.

        Returns
        -------
        ShapelyPolygon
            raw polygon.
        """
        return self._polygon

    def get_raw_coords(self) -> NDArray[np.float32]:
        """Get raw coordinates in xy format.

        Returns
        -------
        NDArray[np.float32]
            of shape (n, 2).
        """
        return np.array(self._polygon.exterior.coords.xy, dtype=np.float32).T

    def contains(self, other: Union[Point, Region]) -> bool:
        """Check whether polygon contains point, bbox or polygon.

        Returns
        -------
        bool
            True if polygon contains other.
        """
        if isinstance(other, Point):
            return self.get_raw_polygon().contains(ShapelyPoint(other.x, other.y))
        else:
            return self.get_raw_polygon().contains(other.to_polygon().get_raw_polygon())

    def get_area(self) -> float:
        """Get area of polygon.

        Returns
        -------
        float
            polygon area.
        """
        return self.get_raw_polygon().area

    def get_inter(self, other: Region) -> float:
        """Return intersection area with another BBox or Polygon.

        Parameters
        ----------
        other : Union[BBox, Polygon]
            other polygon or bbox.

        Returns
        -------
        float
            intersection area.
        """
        other_raw = other.to_polygon().get_raw_polygon()
        return self.get_raw_polygon().intersection(other_raw).area

    def get_union(self, other: Region) -> float:
        """Get union area with another BBox or Polygon.

        Parameters
        ----------
        other : Union[BBox, Polygon]
            other polygon or bbox.

        Returns
        -------
        float
            union area.
        """
        other_raw = other.to_polygon().get_raw_polygon()
        return self.get_raw_polygon().union(other_raw).area

    def get_iou(self, other: Region) -> float:
        """Get intersection over union value with another BBox or Polygon.

        Parameters
        ----------
        other : Union[BBox, Polygon]
            other polygon.

        Returns
        -------
        float
            iou value.
        """
        self_raw = self.get_raw_polygon()
        other_raw = other.to_polygon().get_raw_polygon()
        union_area = self_raw.union(other_raw).area
        inter_area = self_raw.intersection(other_raw).area
        return inter_area / (union_area + 1e-12)

    def get_cxcy(self) -> Point:
        """Get cxcy Point.

        x, y coordinates of centroid.

        Returns
        -------
        Point
            centroid.
        """
        raw_polygon = self.get_raw_polygon()
        return Point(x=raw_polygon.centroid.x, y=raw_polygon.centroid.y)

    @property
    def __geo_interface__(self) -> Dict[str, Any]:
        """Geo inteface required by polygon."""
        return self.get_raw_polygon().__geo_interface__

    def to_polygon(self) -> Polygon:
        """Transform to polygon.

        Returns
        -------
        Polygon
            essentially returns self.
        """
        return self

    @classmethod
    def from_polygon(cls, polygon: Polygon) -> Polygon:
        """Create polygon from polygon.

        Essentially it's just a clone method.

        Parameters
        ----------
        polygon : Polygon
            polygon to clone.

        Returns
        -------
        Polygon
            cloned polygon.
        """
        if not isinstance(polygon, Polygon):
            raise ValueError("Input must be an instance of Polygon")

        new_polygon = cls(polygon.get_raw_polygon().exterior.coords)
        return new_polygon

    def to_bbox(self) -> BBox:
        """
        Convert the polygon to a bounding box.

        Returns
        -------
        BBox
            The smallest bounding box that contains the polygon.
        """
        minx, miny, maxx, maxy = self._polygon.bounds
        return BBox(x1=minx, y1=miny, x2=maxx, y2=maxy)

    @classmethod
    def from_bbox(cls, bbox: BBox) -> Polygon:
        """
        Construct a polygon from a bounding box.

        Parameters
        ----------
        bbox : BBox
            The bounding box to convert into a polygon.

        Returns
        -------
        Polygon
            A new polygon that matches the bounding box.
        """
        points = [(bbox.x1, bbox.y1), (bbox.x2, bbox.y1), (bbox.x2, bbox.y2), (bbox.x1, bbox.y2)]
        return cls(points)

    @classmethod
    def from_numpy(cls, np_array: NDArray[np.float32]) -> Polygon:
        """
        Create a polygon from a 2D numpy array representing point coordinates.

        Parameters
        ----------
        np_array : NDArray[np.float32]
            A 2D numpy array where each row represents a point as (x, y).

        Returns
        -------
        Polygon
            The polygon created from the numpy array points.

        Raises
        ------
        ValueError
            If the numpy array is not 2-dimensional or the second dimension is not 2.
        """
        if np_array.ndim != 2 or np_array.shape[1] != 2:
            raise ValueError("NumPy array must be 2-dimensional with shape (n, 2).")
        points: list[tuple[float, ...]] = list(map(tuple, np_array))  # type: ignore
        return cls(points)

    def to_numpy(self) -> NDArray[np.float32]:
        """
        Convert the polygon's points to a numpy array.

        Returns
        -------
        NDArray[np.float32]
            A numpy array of the polygon's points.
        """
        return np.array(list(self._polygon.exterior.coords), dtype=np.float32)

    @classmethod
    def from_list(cls, points_list: List[List[float]]) -> Polygon:
        """
        Create a polygon from a list of point coordinates.

        Parameters
        ----------
        points_list : List[List[float]]
            A list of points, where each point is represented as [x, y].

        Returns
        -------
        Polygon
            The polygon created from the list of points.
        """
        return cls(points_list)

    def to_list(self) -> List[List[float]]:
        """
        Convert the polygon's points to a list.

        Returns
        -------
        List[List[float]]
            A list of the polygon's points.
        """
        return self.to_numpy().tolist()

    @classmethod
    def from_dict(cls, data: Dict) -> Polygon:
        """
        Create a polygon from a dictionary containing points.

        Parameters
        ----------
        data : Dict
            A dictionary with a 'points' key containing a list of [x, y] coordinates.

        Returns
        -------
        Polygon
            The polygon created from the dictionary data.

        Raises
        ------
        ValueError
            If the dictionary does not contain a 'points' key or it is not a list.
        """
        if "points" not in data or not isinstance(data["points"], list):
            raise ValueError("Dictionary must have a 'points' key with a list of tuples.")
        return cls.from_list(data["points"])

    def to_dict(self) -> Dict:
        """
        Convert the polygon to a dictionary format.

        Returns
        -------
        Dict
            A dictionary with 'points' as a key and a list of [x, y] coordinates as the value.
        """
        return {"points": self.to_list()}

    def rescale(self, scale_x: float, scale_y: float, inplace: bool = False) -> Polygon:
        """
        Rescale the polygon by the specified scale factors.

        NOTE new_x = old_x * scale_x and new_y = old_y * scale_y

        Parameters
        ----------
        scale_x : float
            Scale factor for the x-axis.
        scale_y : float
            Scale factor for the y-axis.
        inplace : bool, optional
            If True, modifies the polygon in-place. Defaults to False.

        Returns
        -------
        Polygon
            The rescaled polygon.
        """
        scaled_coords = self.get_raw_coords() * np.array([scale_x, scale_y])
        if inplace:
            self._polygon = ShapelyPolygon(scaled_coords)
            return self
        return Polygon(scaled_coords)

    def resize(self, from_size: Size, to_size: Size, inplace: bool = False) -> Polygon:
        """
        Resize the polygon based on the change from one resolution to another.

        Parameters
        ----------
        from_size : Size
            The original size.
        to_size : Size
            The new size.
        inplace : bool, optional
            If True, resizes the polygon in place. Defaults to False.

        Returns
        -------
        Polygon
            The resized polygon.
        """
        scale_x, scale_y = to_size.w / from_size.w, to_size.w / from_size.w
        return self.rescale(scale_x=scale_x, scale_y=scale_y, inplace=inplace)

    def move(self, v: Shift, dt: float = 1.0, inplace: bool = False) -> Polygon:
        """Move polygon using shift.

        Parameters
        ----------
        v : Shift
            shift vector that will be used for moving.
        dt : float
            timedelta to move by velocity. Default is 1.0.
        inplace : bool
            Whether to move polygon inplace(update) or create
            new polygon. Default is False meaning that
            new polygon will be created.

        Returns
        -------
        Polygon
        """
        dx = v.x * dt
        dy = v.y * dt
        if inplace:
            self._polygon = translate(self._polygon, xoff=dx, yoff=dy)
            return self
        else:
            new_polygon = translate(self._polygon, xoff=dx, yoff=dy)
            return Polygon(new_polygon)

    def __eq__(self, other: object) -> bool:
        """
        Check equality with another polygon or bounding box.

        Parameters
        ----------
        other : object
            The object to compare against.

        Returns
        -------
        bool
            True if equal, False otherwise.
        """
        if other is None:
            return False
        if not isinstance(other, (BBox, Polygon)):
            raise NotImplementedError()
        return self.equals(other)

    def equals(self, other: Union[Polygon, BBox]) -> bool:
        """
        Check if the polygon is equal to another polygon or bounding box.

        Parameters
        ----------
        other : Union[Polygon, BBox]
            The object to compare for equality.

        Returns
        -------
        bool
            True if equal, False otherwise.
        """
        if not isinstance(other, (Polygon, BBox)):
            raise NotImplementedError()

        return other.to_polygon().get_raw_polygon().equals(self._polygon)

    def almost_equals(self, other: Union[Polygon, BBox], decimal: int = 6) -> bool:
        """Check if the polygon is almost equal to another polygon or bounding box.

        Parameters
        ----------
        other : Union[Polygon, BBox]
            The object to compare against.
        decimal : int
            The precision of the comparison.

        Returns
        -------
        bool
            True if almost equal, False otherwise.
        """
        if not isinstance(other, (Polygon, BBox)):
            raise NotImplementedError

        return other.to_polygon().get_raw_polygon().almost_equals(self._polygon, decimal)

    def equals_exact(self, other: Union[Polygon, BBox], tolerance: float) -> bool:
        """Check if the polygon is exactly equal to another polygon or bounding box.

        Parameters
        ----------
        other : Union[Polygon, BBox]
            The object to compare against.
        tolerance : float
            The tolerance for the comparison.

        Returns
        -------
        bool
            True if exactly equal, False otherwise.
        """
        if not isinstance(other, (Polygon, BBox)):
            raise NotImplementedError

        return other.to_polygon().get_raw_polygon().equals_exact(self._polygon, tolerance)

    def clone(self) -> Polygon:
        """
        Create a deep copy of the current Polygon instance.

        Returns
        -------
        Polygon
            A new Polygon instance that is a copy of this instance.
        """
        # Clone the underlying ShapelyPolygon object
        new_shapely_polygon = ShapelyPolygon(self._polygon.exterior.coords[:])

        # Create a new Polygon instance with the cloned ShapelyPolygon
        return Polygon(new_shapely_polygon)


# BBox class
class BBox(BaseModel, Region):
    """Bounding box class."""

    x1: float
    x2: float
    y1: float
    y2: float

    @property
    def x_min(self) -> float:
        """Get x left coordinate of bbox.

        Alias for `x1` attribute.

        Returns
        -------
        float
            x left coordinate of bbox.
        """
        return self.x1

    @x_min.setter
    def x_min(self, value: float):
        """Set x left coordiate of bbox.

        Alias for `x1` attribute.

        Parameters
        ----------
        value: float
            value to set.
        """
        self.x1 = float(value)

    @property
    def x_max(self) -> float:
        """Get x right coordinate of bbox.

        Alias for `x2` attribute.

        Returns
        -------
        float
            x right coordinate of bbox.
        """
        return self.x2

    @x_max.setter
    def x_max(self, value: float):
        """Set x right coordinate of bbox.

        Alias for `x2` attribute.

        Parameters
        ----------
        value: float
            value to set.
        """
        self.x2 = float(value)

    # y coordinates aliases
    @property
    def y_min(self) -> float:
        """Get y lower coordinate of bbox.

        Alias for `y1` attribute.

        Returns
        -------
        float
            y lower coordinate of bbox.
        """
        return self.y1

    @y_min.setter
    def y_min(self, value: float):
        """Set y lower coordinate of bbox.

        Alias for `y1` attribute.

        Parameters
        ----------
        value: float
            value to set.
        """
        self.y1 = float(value)

    @property
    def y_max(self) -> float:
        """Get y upper coordinate of bbox.

        Alias for `y2` attribute.

        Returns
        -------
        float
            y upper coordinate of bbox.
        """
        return self.y2

    @y_max.setter
    def y_max(self, value: float):
        """Set y upper coordinate of bbox.

        Alias for `y2` attribute.

        Parameters
        ----------
        value: float
            value to set.
        """
        self.y2 = float(value)

    @property
    def w(self) -> float:
        """Get width of the bounding box.

        Returns
        -------
        float
            width of the bounding box.
        """
        return self.width

    @property
    def h(self) -> float:
        """Get height of the bounding box.

        Returns
        -------
        float
            height of the bounding box/
        """
        return self.height

    @property
    def width(self) -> float:
        """Get width of the bounding box.

        Returns
        -------
        float
            width of the bounding box.
        """
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        """Get height of the bounding box.

        Returns
        -------
        float
            height of the bounding box/
        """
        return self.y_max - self.y_min

    @property
    def cx(self) -> float:
        """Get x coordinate of the center.

        Returns
        -------
        float
        """
        return self.x_min + (self.x_max - self.x_min) / 2

    @property
    def cy(self) -> float:
        """Get y coordinate of the center.

        Returns
        -------
        float
        """
        return self.y_min + (self.y_max - self.y_min) / 2

    @property
    def cxcy(self) -> Point:
        """Get central point.

        Returns
        -------
        Point
            central point of the box.
        """
        return Point(x=self.cx, y=self.cy)

    def get_cxcy(self) -> Point:
        """Get central point.

        Returns
        -------
        Point
            central point of the box.
        """
        return Point(x=self.cx, y=self.cy)

    def is_valid(self) -> bool:
        """Check whether bounding box is valid.

        Bounding box is considered valid if x1 <= x2 and y1 <= y2.

        Returns
        -------
        is_valid : bool
            whether bounding box is valid or not.
        """
        if self.x1 > self.x2 or self.y1 > self.y2:
            return False
        return True

    def get_area(self) -> float:
        """Get area for bounding box.

        Returns
        -------
        area : float
            area corresponing to bounding box.

        Raises
        ------
        ValueError
            if bounding box is not valid. See method is_valid() for more information.
        """
        if not self.is_valid():
            raise ValueError("Invalid bounding box.")
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def get_inter(self, other: Region) -> float:
        """Return intersection area with another BBox or Polygon.

        Parameters
        ----------
        other : Union[BBox, Polygon]
            other polygon or bbox.

        Returns
        -------
        float
            intersection area.
        """
        if isinstance(other, BBox):
            inter_x1 = max(self.x1, other.x1)
            inter_y1 = max(self.y1, other.y1)
            inter_x2 = min(self.x2, other.x2)
            inter_y2 = min(self.y2, other.y2)

            if inter_x1 <= inter_x2 and inter_y1 <= inter_y2:
                return BBox(x1=inter_x1, y1=inter_y1, x2=inter_x2, y2=inter_y2).get_area()
            else:
                return 0.0

        return self.to_polygon().get_inter(other)

    def get_union(self, other: Region) -> float:
        """Get union area with another BBox or Polygon.

        Parameters
        ----------
        other : Union[BBox, Polygon]
            other polygon or bbox.

        Returns
        -------
        float
            union area.
        """
        """Return the union of this Polygon with another BBox or Polygon."""
        return self.to_polygon().get_union(other)

    def get_iou(self, other: Region) -> float:
        """Get intersection over union value with another BBox or Polygon.

        Parameters
        ----------
        other : Union[BBox, Polygon]
            other polygon.

        Returns
        -------
        float
            iou value.
        """
        return self.to_polygon().get_iou(other)

    @staticmethod
    def _list_to_dict(
        bbox_list: List[float], fmt: Literal["xyxy", "cxywh", "tlwh", "cxyah", "tlah"]
    ) -> XYXYDict:
        if fmt == "xyxy":
            return {"x1": bbox_list[0], "y1": bbox_list[1], "x2": bbox_list[2], "y2": bbox_list[3]}
        elif fmt == "cxywh":
            cx, cy, w, h = bbox_list
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        elif fmt == "cxyah":
            cx, cy, ar, h = bbox_list
            w = h * ar
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        elif fmt == "tlah":
            x, y, ar, h = bbox_list
            w = h * ar
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h
            return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        elif fmt == "tlwh":
            x, y, w, h = bbox_list
            return {"x1": x, "y1": y, "x2": x + w, "y2": y + h}
        else:
            raise ValueError(f"Unsupported BBox format: {fmt}")

    @classmethod
    def from_dict(
        cls,
        bbox_dict: Union[XYXYDict, CXYWHDict, TLWHDict, CXYAHDict, TLAHDict],
        fmt: Literal["xyxy", "cxywh", "tlwh", "cxyah", "tlah"],
    ) -> BBox:
        """Create BBox object from dict.

        Parameters
        ----------
        bbox_dict : Union[XYXYDict, CXYWHDict, TLWHDict, CXYAHDict, TLAHDict]
        fmt : Literal["xyxy", "cxywh", "tlwh", "cxyah", "tlah"]

        Returns
        -------
        BBox
            resulting BBox object.
        """
        if fmt == "xyxy":
            bbox_dict = cast(XYXYDict, bbox_dict)
            return cls(**bbox_dict)

        elif fmt == "cxywh":
            bbox_dict = cast(CXYWHDict, bbox_dict)
            cx, cy, w, h = bbox_dict["cx"], bbox_dict["cy"], bbox_dict["w"], bbox_dict["h"]
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            return cls(x1=x1, y1=y1, x2=x2, y2=y2)

        elif fmt == "cxyah":
            bbox_dict = cast(CXYAHDict, bbox_dict)
            cx, cy, ar, h = bbox_dict["cx"], bbox_dict["cy"], bbox_dict["a"], bbox_dict["h"]
            w = h * ar
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            return cls(x1=x1, y1=y1, x2=x2, y2=y2)

        elif fmt == "tlwh":
            bbox_dict = cast(TLWHDict, bbox_dict)
            x, y, w, h = bbox_dict["x"], bbox_dict["y"], bbox_dict["w"], bbox_dict["h"]
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h
            return cls(x1=x1, y1=y1, x2=x2, y2=y2)

        elif fmt == "tlah":
            bbox_dict = cast(TLAHDict, bbox_dict)
            x, y, ar, h = bbox_dict["x"], bbox_dict["y"], bbox_dict["a"], bbox_dict["h"]
            w = h * ar
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h
            return cls(x1=x1, y1=y1, x2=x2, y2=y2)

        else:
            raise ValueError(f"Unsupported BBox format: {fmt}")

    def to_dict(self, fmt: Literal["xyxy", "cxywh", "tlwh", "cxyah", "tlah"]) -> BBoxDict:
        """Convert bounding box to dict.

        Parameters
        ----------
        fmt : Literal["xyxy", "cxywh", "tlwh", "cxyah", "tlah"]
            format of bounding box.

        Returns
        -------
        BBoxDict
        """
        if fmt == "xyxy":
            return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}

        elif fmt == "cxywh":
            return {"cx": self.cx, "cy": self.cy, "w": self.width, "h": self.height}

        elif fmt == "cxyah":
            a = self.width / self.height if self.height > 0 else self.width / (self.height + 1e-12)
            return {"cx": self.cx, "cy": self.cy, "a": a, "h": self.height}

        elif fmt == "tlwh":
            return {"x": self.x1, "y": self.y1, "w": self.width, "h": self.height}
        elif fmt == "tlah":
            a = self.width / self.height if self.height > 0 else self.width / (self.height + 1e-12)
            return {"x": self.x1, "y": self.y1, "a": a, "h": self.height}

        else:
            raise ValueError(f"Unsupported BBox format: {fmt}")

    @classmethod
    def from_numpy(
        cls, bbox_array: NDArray[np.float32], fmt: Literal["xyxy", "cxywh", "tlwh", "cxyah", "tlah"]
    ) -> BBox:
        """Create BBox object from 1d numpy array.

        Parameters
        ----------
        bbox_array : NDArray[np.float32]
        fmt : Literal["xyxy", "cxywh", "tlwh", "cxyah", "tlah"]
            order of values defined by this parameter.

        Returns
        -------
        BBox
            resulting BBox object.
        """
        if bbox_array.ndim != 1:
            raise ValueError("Array must be one-dimensional and of type float.")

        return cls.from_list(bbox_list=bbox_array.tolist(), fmt=fmt)

    def to_numpy(
        self, fmt: Literal["xyxy", "cxywh", "tlwh", "cxyah", "tlah"]
    ) -> NDArray[np.float32]:
        """Transform bounding box to 1d numpy array.

        Parameters
        ----------
        fmt : Literal["xyxy", "cxywh", "tlwh", "cxyah", "tlah"]
            order of values in the output list
            will defined by this parameter.

        Returns
        -------
        NDArray[np.float32]
            numpy array of size 4
            and dtype=np.float32
            in format defined by fmt parameter.
        """
        return np.array(self.to_list(fmt=fmt), dtype=np.float32)

    @classmethod
    def from_list(
        cls, bbox_list: List[float], fmt: Literal["xyxy", "cxywh", "tlwh", "cxyah", "tlah"]
    ) -> BBox:
        """Create BBox object from list.

        Parameters
        ----------
        bbox_list : List[float]
        fmt : Literal["xyxy", "cxywh", "tlwh", "cxyah". "tlah"]
            order of values in the output list
            will defined by this parameter.

        Returns
        -------
        BBox
            resulting BBox object.
        """
        if len(bbox_list) != 4:
            raise ValueError("List must contain exactly four elements.")

        if not all(isinstance(x, float) for x in bbox_list):
            raise ValueError("All elements in the list must be of type float.")

        bbox_dict = cls._list_to_dict(bbox_list, fmt)
        return cls.from_dict(bbox_dict, fmt="xyxy")

    def to_list(self, fmt: Literal["xyxy", "cxywh", "tlwh", "cxyah", "tlah"]) -> List[float]:
        """Transform bounding box to list.

        Parameters
        ----------
        fmt : Literal["xyxy", "cxywh", "tlwh", "cxyah"]
            order of values defined by this parameter.

        Returns
        -------
        List[float]
            list of size 4 in the following
        """
        bbox_dict = self.to_dict(fmt)
        if fmt == "xyxy":
            bbox_dict = cast(XYXYDict, bbox_dict)
            return [bbox_dict["x1"], bbox_dict["y1"], bbox_dict["x2"], bbox_dict["y2"]]
        elif fmt == "cxywh":
            bbox_dict = cast(CXYWHDict, bbox_dict)
            return [bbox_dict["cx"], bbox_dict["cy"], bbox_dict["w"], bbox_dict["h"]]
        elif fmt == "cxyah":
            bbox_dict = cast(CXYAHDict, bbox_dict)
            return [bbox_dict["cx"], bbox_dict["cy"], bbox_dict["a"], bbox_dict["h"]]
        elif fmt == "tlwh":
            bbox_dict = cast(TLWHDict, bbox_dict)
            return [bbox_dict["x"], bbox_dict["y"], bbox_dict["w"], bbox_dict["h"]]
        elif fmt == "tlah":
            bbox_dict = cast(TLAHDict, bbox_dict)
            return [bbox_dict["x"], bbox_dict["y"], bbox_dict["a"], bbox_dict["h"]]
        else:
            raise NotImplementedError(f"Uknown format: {fmt}")

    @classmethod
    def from_polygon(cls, polygon: Polygon) -> BBox:
        """Create BBox object from polygon.

        Parameters
        ----------
        polygon: Polygon
            polygon from which bounding
            box will be constructed.

        Returns
        -------
        BBox
            created bounding box containing
            input polygon.
        """
        minx, miny, maxx, maxy = polygon.get_raw_polygon().bounds
        return cls(x1=minx, y1=miny, x2=maxx, y2=maxy)

    def to_polygon(self) -> Polygon:
        """Get shapely polygon from bbox.

        Returns
        -------
        Polygon
            shapely polygon corresponding
            to bounding box.
        """
        return Polygon(
            np.array(
                [[self.x1, self.y1], [self.x2, self.y1], [self.x2, self.y2], [self.x1, self.y2]]
            )
        )

    @classmethod
    def from_bbox(cls, bbox: BBox) -> BBox:
        """Create bbox from other bbox.

        Essentially this is just a clone method.

        Parameters
        ----------
        bbox : BBox
            input bbox to clone.

        Returns
        -------
        BBox
            new bbox.
        """
        return BBox(x1=bbox.x1, y1=bbox.y1, x2=bbox.x2, y2=bbox.y2)

    def to_bbox(self) -> BBox:
        """Transform to BBox.

        Returns
        -------
        BBox
            essentially returns self.
        """
        return self

    def crop(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Get crop from provided image.

        Input image must be provided in a form
        of numpy array of 2 or 3 dims and dtype=np.uint8.

        If input image is 2-dim array that 1 will be added
        in the end of the shape (h, w, 1).

        Parameters
        ----------
        image : NDArray[np.uint8]
            source frame image.

        Returns
        -------
        NDArray[np.uint8]
            crop from provided image
            corresponding to detection's
            bounding box.

        Raises
        ------
        ValueError
            if input image is not numpy array
            of 2 or 3 dimension.
        """
        if image.ndim not in (2, 3):
            raise ValueError("Image shape must be 2 or 3 dim")
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        return image[int(self.y1) : int(self.y2), int(self.x1) : int(self.x2), :]

    def contains(self, other: Union[Region, Point]) -> bool:
        """
        Determine if this bounding box contains another region or point.

        Parameters
        ----------
        other : Union[Region, Point]
            The object to check containment against.

        Returns
        -------
        bool
            True if this bounding box contains `other`, False otherwise.
        """
        if isinstance(other, BBox):
            return (
                self.x1 <= other.x1 <= self.x2
                and self.y1 <= other.y1 <= self.y2
                and self.x1 <= other.x2 <= self.x2
                and self.y1 <= other.y2 <= self.y2
            )
        elif isinstance(other, Point):
            return self.x1 <= other.x <= self.x2 and self.y1 <= other.y <= self.y2
        else:
            return self.to_polygon().contains(other)

    def rescale(self, scale_x: float, scale_y: float, inplace: bool = False) -> BBox:
        """
        Rescale the bounding box by the given scale factors.

        Parameters
        ----------
        scale_x : float
            The factor to scale the x-dimension.
        scale_y : float
            The factor to scale the y-dimension.
        inplace : bool, optional
            If True, modifies the bounding box in place.

        Returns
        -------
        BBox
            The rescaled bounding box.
        """
        if inplace:
            self.x1 = self.x1 * scale_x
            self.y1 = self.y1 * scale_y
            self.x2 = self.x2 * scale_x
            self.y2 = self.y2 * scale_y
            return self
        return BBox(
            x1=self.x1 * scale_x, y1=self.y1 * scale_y, x2=self.x2 * scale_x, y2=self.y2 * scale_y
        )

    def resize(self, from_size: Size, to_size: Size, inplace: bool = False) -> BBox:
        """
        Resize the bounding box based on the change from one resolution to another.

        Parameters
        ----------
        from_size : Size
            The initial frame size.
        to_size : Size
            The target frame size.
        inplace : bool, optional
            If True, resizes the bounding box in place.

        Returns
        -------
        BBox
            The resized bounding box.
        """
        scale_x, scale_y = to_size.w / from_size.w, to_size.w / from_size.w
        return self.rescale(scale_x=scale_x, scale_y=scale_y, inplace=inplace)

    def move(self, v: Shift, dt: float = 1.0, inplace: bool = False) -> BBox:
        """Move bbox using velocity.

        Parameters
        ----------
        v : Shift
            shift vector that will be used for moving.
        dt : float
            timedelta to move by velocity. Default is 1.0.
        inplace : bool
            Whether to move bbox inplace(update) or create
            new bbox. Default is False meaning that
            new bbox will be created.

        Returns
        -------
        BBox
            moved bounding box.
        """
        new_x1 = self.x1 + v.x * dt
        new_x2 = self.x2 + v.x * dt

        new_y1 = self.y1 + v.y * dt
        new_y2 = self.y2 + v.y * dt
        if inplace:
            self.x1 = new_x1
            self.x2 = new_x2

            self.y1 = new_y1
            self.y2 = new_y2
            return self
        return BBox(x1=new_x1, x2=new_x2, y1=new_y1, y2=new_y2)

    def __eq__(self, other: object) -> bool:
        """
        Check equality with another bounding box or polygon.

        Parameters
        ----------
        other : object
            The object to compare against.

        Returns
        -------
        bool
            True if equal, False otherwise.
        """
        if other is None:
            return False
        if not isinstance(other, (BBox, Polygon)):
            raise NotImplementedError()
        return self.equals(other)

    def equals(self, other: Union[Polygon, BBox]) -> bool:
        """
        Check if the bounding box is equal to another bounding box or polygon.

        Parameters
        ----------
        other : Union[Polygon, BBox]
            The object to compare for equality.

        Returns
        -------
        bool
            True if equal, False otherwise.
        """
        if not isinstance(other, (Polygon, BBox)):
            raise NotImplementedError()
        if isinstance(other, BBox):
            return (
                self.x1 == other.x1
                and self.x2 == other.x2
                and self.y1 == other.y1
                and self.y2 == other.y2
            )
        return other.equals(self)

    def almost_equals(self, other: Union[Polygon, BBox], decimal: int = 6) -> bool:
        """Check if the bounding box is almost equal to another bounding or polygon.

        Parameters
        ----------
        other : Union[Polygon, BBox]
            The object to compare against.
        decimal : int
            The precision of the comparison.

        Returns
        -------
        bool
            True if almost equal, False otherwise.
        """
        if not isinstance(other, (Polygon, BBox)):
            raise NotImplementedError
        return self.to_polygon().almost_equals(other=other, decimal=decimal)

    def equals_exact(self, other: Union[Polygon, BBox], tolerance: float) -> bool:
        """Check if the bounding box is exactly equal to another bounding box or polygon.

        Parameters
        ----------
        other : Union[Polygon, BBox]
            The object to compare against.
        tolerance : float
            The tolerance for the comparison.

        Returns
        -------
        bool
            True if exactly equal, False otherwise.
        """
        if not isinstance(other, (Polygon, BBox)):
            raise NotImplementedError
        return self.to_polygon().equals_exact(other=other, tolerance=tolerance)

    def clone(self) -> BBox:
        """
        Create a deep copy of this bounding box.

        Returns
        -------
        BBox
            A copy of this bounding box.
        """
        return BBox(x1=self.x1, y1=self.y1, x2=self.x2, y2=self.y2)
