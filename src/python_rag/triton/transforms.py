"""Contains implementation of various usefull functions."""

from typing import Callable, List, Literal, Tuple

import cv2
import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from ..dto import Size


@njit(nogil=True)  # type: ignore
def crop_njit(image, bboxes):
    """Get crops from image.

    Parameters
    ----------
    image : np.ndarray
        source image of shape (m, n, 3)
    bboxes : np.ndarray
        bounding boxes to crop from image repsresented
        by numpy array of shape (n, 4).

    Returns
    -------
    List[np.ndarray]
        list of crops from original image.
    """
    n = bboxes.shape[0]
    crops = []
    for i in prange(n):
        x_min, y_min = bboxes[i, 0], bboxes[i, 1]
        x_max, y_max = bboxes[i, 2], bboxes[i, 3]
        crops.append(image[y_min:y_max, x_min:x_max, :])
    return crops


class CropResize:
    """Utility class for cropping and resizing images based on specified dimensions and mode."""

    size: Size
    mode: Literal["center", "top", "bottom"]

    def __init__(self, size: Size, mode: Literal["center", "top", "bottom"]):
        """Initialize the CropResize object with specified size and crop mode.

        Parameters
        ----------
        size : Size
            Target size for cropping and resizing.
        - mode : Literal['center', 'top', 'bottom']
            Specifies the crop mode, which can be 'center', 'top', or 'bottom'.
        """
        self.mode = mode
        self.size = size

    @classmethod
    def crop(
        cls, image: NDArray[np.float32], mode: Literal["center", "top", "bottom"]
    ) -> NDArray[np.float32]:
        """Crop input image with given crop mode.

        Parameters
        ----------
        image : NDArray[np.float32]
            input image represented by numpy array
            of shape (h, w, 3)
        mode : Literal["center", "top", "right_bot"]
            cropping model. "center" corresponds to center crop,
            "top" corresponds to crop for top left position of image,
            "bottom" corresponds to right bottom position of image.
        """
        h, w = image.shape[:2]
        m = min(h, w)
        top, left = 0, 0
        if mode == "center":
            top, left = (h - m) // 2, (w - m) // 2
        elif mode == "bottom":
            top, left = h - m, w - m
        elif mode != "top":
            raise ValueError(
                f"Invalid value for crop mode: `{mode}`."
                + "Must be one of 'center', 'top', 'bottom"
            )
        return image[top : top + m, left : left + m]

    @classmethod
    def resize(cls, image: NDArray[np.float32], size: Size) -> NDArray[np.float32]:
        """Resize image to given size.

        Parameters
        ----------
        image : NDArray[np.float32]
            image to resize.
        size : Size
            result image size.

        Returns
        -------
        NDArray[np.float32]
            image after crop and resize.
        """
        return cv2.resize(image, (size.w, size.h), interpolation=cv2.INTER_NEAREST)  # type: ignore

    def __call__(self, image: NDArray[np.float32]) -> NDArray[np.float32]:
        """Run crop and resize on image.

        Parameters
        ----------
        image : NDArray[np.float32]

        Returns
        -------
        NDArray[np.float32]
            image after crop resize, having shape equal
            to self.Size.
        """
        return self.resize(self.crop(image, mode=self.mode), size=self.size)


class Normalize:
    """Normalizer callable transform."""

    unify_scale: bool
    mean: NDArray[np.float32]
    std: NDArray[np.float32]

    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        std: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        unify_scale: bool = True,
    ):
        """Initialize normalizer functor.

        NOTE that if unify_scale is set to True
        then normalization applied after
        dividing image by 255.0

        Parameters
        ----------
        mean : Tuple[float, float, float]
            mean for channels of the image.
            Default is (0.0, 0.0, 0.0).
        std : Tuple[float, float, float]
            std deviation for channels of the image.
            Default is (1.0, 1.0, 1.0).
        unify_scale : bool
            whether to unify scale first.
            Scale unification means that
            image channels will be divided
            by 255.0 first before applying
            normalization.
            Default is True.
        """
        self.mean = np.array(mean).reshape(1, 1, -1)
        self.std = np.array(std).reshape(1, 1, -1)
        self.unify_scale = bool(unify_scale)

    def __call__(self, image: NDArray[np.float32]) -> NDArray[np.float32]:
        """Normalize input image.

        Parameters
        ----------
        image : NDArray[np.float32]
            input image in H x W x 3 format.

        Returns
        -------
        NDArray[np.float32]
            image after normalization
            in H x W x 3 format.
        """
        if self.unify_scale:
            x = image / 255.0
        return (x - self.mean) / self.std


class MoveChannelsDim:
    """Transforms the input image by moving the channels dimension to the front."""

    def __call__(self, image: NDArray[np.float32]) -> NDArray[np.float32]:
        """Normalize input image.

        Parameters
        ----------
        image : NDArray[np.float32]
            input image in H x W x 3 format.

        Returns
        -------
        NDArray[np.float32]
            image after normalization
            in H x W x 3 format.
        """
        return np.transpose(image, axes=(2, 0, 1))


class Compose:
    """Compose callable transform class."""

    transforms: List[Callable[[NDArray[np.float32]], NDArray[np.float32]]]

    def __init__(self, transforms: List[Callable[[NDArray[np.float32]], NDArray[np.float32]]]):
        """Create composed transform from list of transforms.

        Composed transform will apply transforms in sequential
        order.

        Parameters
        ----------
        transforms : List[Callable[[NDArray[np.float32]], NDArray[np.float32]]]
            transforms to apply one by one.
        """
        self.transforms = [] if transforms is None else list(transforms)

    def __call__(self, image: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply transforms on image.

        Parameters
        ----------
        image : NDArray[np.float32]
            image on which transforms will be
            applied.

        Returns
        -------
        NDArray[np.float32]
            transformed image.
        """
        x = image
        for transform in self.transforms:
            x = transform(x)
        return x


def get_transform_sequence(
    size: Size = Size(w=224, h=224),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    crop_mode: Literal["top", "center", "bottom"] = "top",
) -> Callable[[NDArray[np.float32]], NDArray[np.float32]]:
    """Get composed transform.

    Parameters
    ----------
    size : int
        target image size for yolo v5 model.
    imagenet_mean : List[float]
        mean values to subtract
    imagenet_std : List[float]
        std values to divide
    crop_mode : str
        selector for crop method
    """
    return Compose(
        [
            CropResize(mode=crop_mode, size=size),
            Normalize(mean=mean, std=std, unify_scale=True),
            MoveChannelsDim(),
        ]
    )
