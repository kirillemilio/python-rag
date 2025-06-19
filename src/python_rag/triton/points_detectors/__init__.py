"""Contains imports related to iwpodnet model."""

from .iwpodnet_points_detector import IWpodNetTritonModel
from .points_detector import PointsDetectorTritonModel, PointsOutputDict

__all__ = ["IWpodNetTritonModel", "PointsOutputDict", "PointsDetectorTritonModel"]
