"""Contains implementation of basic unet segmentator."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from operator import methodcaller
from typing import Dict, List, Literal, Optional

import cv2
import numpy as np
import shapely
import tritonclient.grpc as grpcclient  # type: ignore
from numba import njit
from numpy.typing import NDArray

from ...dto import Polygon, Size
from ..models_factory import TritonModelFactory
from ..triton_model import BaseTritonModel

LOGGER = logging.getLogger(__name__)


@njit(nogil=True)
def compute_hard_mask_njit(soft_mask: NDArray[np.float32]) -> NDArray[np.bool_]:
    """Compute hard binary mask from soft segmentation mask.

    Parameters
    ----------
    soft_mask : NDArray[np.float32]
        soft mask array of shape (n, h, w)
        where n is number of labels, h is height of mask
        and w is width of mask.
        Hard mask is just binary representation of soft mask:
         - having 1 for max prob label at given i, j spatial position
         - having 0 for non max prob label at given i, j spatial position

    Returns
    -------
    NDArray[np.bool_]
        hard binary mask corresponding to provided soft mask.
    """
    n, h, w = soft_mask.shape
    hard_mask: NDArray[np.bool_] = np.zeros((n, h, w), dtype=np.bool_)
    for i in range(h):
        for j in range(w):
            max_label_id, max_label_prob = -1, -1
            for k in range(n):
                if soft_mask[k, i, j] > max_label_prob:
                    max_label_id = k
                    max_label_prob = soft_mask[k, i, j]
            if max_label_id >= 0:
                hard_mask[max_label_id, i, j] = True
    return hard_mask


@njit(nogil=True)
def collapse_label_mask_njit(hard_mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Collapse several labels binary mask into single label in logical or fashion.

    Parameters
    ----------
    hard_mask : NDArray[np.bool_]
        input hard mask binary mask represented
        by boolean numpy array of shape
        (n, h, w) where n is number of labels.

    Returns
    -------
    NDArray[np.bool_]
        numpy array containing collapsed mask
        having shape of (h, w). Each mask pixel
        is just an or operation along all label
        masks.
    """
    n, h, w = hard_mask.shape
    collapsed_mask: NDArray[np.bool_] = np.zeros((h, w), dtype=np.bool_)
    for k in range(n):
        for i in range(h):
            for j in range(w):
                collapsed_mask[i, j] |= hard_mask[k, i, j]
    return collapsed_mask


@njit(nogil=True)
def softmax_1d_njit(x: NDArray[np.float32]) -> NDArray[np.float32]:
    """Compute softmax over 1d numpy array.

    Returns
    -------
    NDArray[np.float32]
        input over which softmax function was applied.
    """
    y = np.exp(x - x.max())
    return y / np.sum(y)


@njit(nogil=True)
def softmax_njit(logits: NDArray[np.float32]) -> NDArray[np.float32]:
    """Compute softmax over first dimension of logits.

    Parameters
    ----------
    logits : NDArray[np.float32]
        logits to compute softmax.

    Returns
    -------
    NDArray[np.float32]
        softmax applied over first dimension of input logits.
    """
    _, h, w = logits.shape
    for i in range(h):
        for j in range(w):
            logits[:, i, j] = softmax_1d_njit(logits[:, i, j])
    return logits


def sigmoid_like_softmax(logits: NDArray[np.float32]) -> NDArray[np.float32]:
    """Apply sigmoid to input to mimic softmax with background class.

    Parameters
    ----------
    logits : NDArray[np.float32]
        logits to compute sigmoid mimicing softmax.

    Returns
    -------
    NDArray[np.float32]
        sigmoid mimicing softmax with background class.
    """
    sigmoid = 1.0 / (np.exp(-logits) + 1.0)
    return np.concatenate((1.0 - sigmoid, sigmoid), axis=0)


@dataclass
class SemanticSegmentationResult:
    """Semantic segmentation result data class.

    Attributes
    ----------
    soft_mask : NDArray[np.float32]
        soft mask represented by float32
        numpy array with probabilities of each label
        at each pixel.
    labels : List[str]
        list of labels corresponding to different
        positions of 0-index of soft-mask.
    dst_size : Size
        destination size.
    """

    soft_mask: NDArray[np.float32]
    labels: List[str]
    dst_size: Size

    def get_dst_size(self) -> Size:
        """Get destination spatial size of mask.

        Returns
        -------
        Size
            destination spatial size of the mask.
        """
        return self.dst_size

    def get_src_size(self) -> Size:
        """Get source spatial size of mask.

        Returns
        -------
        Size
            source spatial size of the mask.
        """
        return Size(w=self.soft_mask.shape[2], h=self.soft_mask.shape[1])

    def get_num_classes(self) -> int:
        """Get number of classes.

        Returns
        -------
        int
            number of classes in mask.
        """
        return self.soft_mask.shape[0]

    def get_label_soft_mask(
        self, label: str, *labels: str, scale: bool = True
    ) -> NDArray[np.float32]:
        """Get summed soft mask of one or more labels.

        Parameters
        ----------
        label : str
            First label to include.
        *labels : str
            Additional labels to include in the result.
        scale : bool
            whether to scale mask to original size or not.
            Default is True.

        Returns
        -------
        NDArray[np.float32]
            2D soft mask of shape (h, w) representing the sum
            of probabilities for the given labels.
        """
        all_labels_set = {label, *labels}
        indices_mask: NDArray[np.bool_] = np.zeros(self.soft_mask.shape[0], dtype=np.bool_)
        for i in range(len(self.labels)):
            if self.labels[i] in all_labels_set:
                indices_mask[i] = True

        res_mask = self.soft_mask[indices_mask, ...].sum(axis=0)
        if not scale:
            return res_mask

        dst_size = self.get_dst_size()
        return cv2.resize(res_mask, (dst_size.w, dst_size.h), interpolation=cv2.INTER_NEAREST)

    def get_label_hard_mask(
        self, label: str, *labels: str, scale: bool = True
    ) -> NDArray[np.bool_]:
        """Get binary hard mask for one or more labels.

        Parameters
        ----------
        label : str
            First label to include.
        *labels : str
            Additional labels to include.
        scale : bool
            whether to scale mask to original size or not.
            Deafult is True.

        Returns
        -------
        NDArray[np.bool_]
            2D boolean array of shape (h, w) where pixels corresponding
            to any of the specified labels are marked as True.
        """
        all_labels_set = {label, *labels}
        hard_mask: NDArray[np.bool_] = compute_hard_mask_njit(soft_mask=self.soft_mask)
        indices_mask = np.zeros(self.soft_mask.shape[0], dtype=np.bool_)
        for i in range(len(self.labels)):
            if self.labels[i] in all_labels_set:
                indices_mask[i] = True

        colapsed_hard_mask = collapse_label_mask_njit(hard_mask[indices_mask, ...])
        if not scale:
            return colapsed_hard_mask

        dst_size = self.get_dst_size()
        return cv2.resize(
            colapsed_hard_mask.astype(np.uint8),
            (dst_size.w, dst_size.h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.uint8)

    def get_label_polygons(
        self,
        label: str,
        *labels: str,
        min_area: float = 0.0,
        top_k: Optional[int] = None,
        scale: bool = True,
    ) -> List[Polygon]:
        """Get list of polygons corresponding to set of labels.

        Parameters
        ----------
        label : str
            First label to include.
        *labels : str
            Additional labels to include.
        min_area : float
            minimal polygon area to consider.
            Default is 0.0.
        top_k : Optional[int]
            take top_k polygons sorted by their area.
            Default is None meaning that all polygons will be returned.
        scale : bool
            whether to scale polygons or not.
            Default is True.

        Returns
        -------
        List[Polygon]
            list of polygons corresponding to provided labels.
        """
        mask: NDArray[np.uint8] = (
            self.get_label_hard_mask(label, *labels, scale=False) * 255
        ).astype(np.uint8)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not len(contours) or hierarchy is None:
            return []
        contours = [
            c
            for c, h in zip(contours, hierarchy[0, ...])
            if cv2.contourArea(c) >= min_area and h[-1] == -1  # type: ignore
        ]
        contours = [c.reshape(-1, 2) for c in contours if c.shape[0] >= 3]

        fixed_raw_polygons: List[NDArray[np.float32]] = []

        # fixed_raw_polygons = contours
        for contour in contours:
            s = shapely.Polygon(contour)
            if not s.is_valid:
                r = s.buffer(0)
                if isinstance(r, shapely.MultiPolygon):
                    fixed_raw_polygons.extend(
                        [np.array(x.exterior.coords, dtype=np.float32) for x in r.geoms]
                    )
                elif isinstance(r, shapely.Polygon):
                    fixed_raw_polygons.append(np.array(r.exterior.coords, dtype=np.float32))
            else:
                fixed_raw_polygons.append(contour)
        polygons = [Polygon.from_numpy(c.reshape(-1, 2)) for c in fixed_raw_polygons]
        polygons = sorted(polygons, key=methodcaller("get_area"), reverse=True)
        if top_k is not None:
            polygons = polygons[:top_k]

        if not scale:
            return polygons
        return [
            polygon.resize(from_size=self.get_src_size(), to_size=self.get_dst_size(), inplace=True)
            for polygon in polygons
        ]


@TritonModelFactory.register_model(model_type="semantic-segmentator", arch_type="unet")
class SemanticSegmentatorTritonModel(
    BaseTritonModel[NDArray[np.float32], SemanticSegmentationResult]
):
    """A specialized triton model class for image semantic segmentation tasks.

    This class handles the preprocessing of images,
    setting up necessary transformations,
    and manages interactions with Triton inference server.

    Default preprocessing config is (0.0, 0.0, 0.0) as mean,
    (1.0, 1.0, 1.0) as std.

    Attributes
    ----------
    input_name : str
        Name of the input tensor expected by triton model.
    transform : Callable
        Transformation function applied on input images.
    input_size : Size
        Size object specifying the dimensions to which input images are resized
        or/and croppped.
    labels : List[str]
        list of labels.
    preprocess_mode : Literal["color", "grey"]
        preprocessing mode. Can be either "color" or "grey".
    original_size_list : List[Size]
        original images height and widths.
    """

    input_size: Size
    input_name: str
    output_name: str
    labels: List[str]

    preprocess_mode: Literal["color", "grey"]

    mean: NDArray[np.float32]
    std: NDArray[np.float32]

    original_size_list: List[Size]

    def __init__(
        self,
        client: grpcclient.InferenceServerClient,
        model_name: str,
        input_name: str,
        output_name: str,
        input_size: Size,
        labels: List[str],
        preprocess_mode: Literal["grey", "color"],
        mean: NDArray[np.float32],
        std: NDArray[np.float32],
        client_timeout: float | None,
        device_id: int = 0,
        use_cushm: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            client=client,
            model_name=model_name,
            inputs={input_name: (3, input_size.h, input_size.w)},
            outputs=[output_name],
            datatype="FP32",
            client_timeout=client_timeout,
            device_id=device_id,
            use_cushm=use_cushm,
            **kwargs,
        )
        self.input_size = input_size
        self.input_name = input_name
        self.output_name = output_name
        self.labels = labels
        self.mean = mean
        self.std = std
        self.original_size_list = []
        self.preprocess_mode = preprocess_mode

    def set_original_size_list(self, inputs: List[NDArray[np.float32]]) -> None:
        """Set original sizes of inputs.

        Parameters
        ----------
        inputs : List[NDArray[np.float32]]
            List of input images to segment.

        Notes
        -----
        This method is crucial for accurate scaling of output masks back
        to original image dimensions.
        """
        for image in inputs:
            h, w, c = image.shape
            self.original_size_list.append(Size(h=h, w=w))

    def clear_original_size_list(self):
        """Clear the list that holds the original sizes of processed images.

        This method should be called after completing the processing of a batch
        to reset the list for the next batch of images.
        """
        self.original_size_list.clear()

    def _preprocess_one_image_grey(self, image: NDArray[np.float32]) -> NDArray[np.float32]:
        """Preprocess one image in grey mode."""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # type: ignore
        image = cv2.resize(image, dsize=(self.input_size.w, self.input_size.h))  # type: ignore
        image = image.astype(np.float32) / 255.0
        image_tiled = np.tile(image[np.newaxis, ...], [3, 1, 1])
        image_tiled = (image_tiled - self.mean[:, np.newaxis, np.newaxis]) / (
            self.std[:, np.newaxis, np.newaxis] + 1e-12
        )
        return image_tiled

    def _preprocess_one_image_color(self, image: NDArray[np.float32]) -> NDArray[np.float32]:
        """Preprocess one image in color mode."""
        image = cv2.resize(
            image,
            dsize=(self.input_size.w, self.input_size.h),
            interpolation=cv2.INTER_LINEAR,  # type: ignore
        )
        image = image.astype(np.float32).transpose(2, 0, 1) / 255.0
        image = (image - self.mean[:, np.newaxis, np.newaxis]) / (
            self.std[:, np.newaxis, np.newaxis] + 1e-12
        )
        return image

    def preprocess(self, inputs: List[NDArray[np.float32]]) -> Dict[str, NDArray[np.float32]]:
        """Preprocess inputs for unet like segmentation.

        Parameters
        ----------
        inputs : List[NDArray[np.float32]]
            list of input raw images, potentially of different sizes.

        Returns
        -------
        Dict[str, NDArray[np.float32]]
            dictionary mapping triton model input name to numpy array
            corresponding to input image input in triton model.
        """
        self.set_original_size_list(inputs=inputs)
        preprocessed_images = []
        for image in inputs:
            preprocessed_image = (
                self._preprocess_one_image_color(image)
                if self.preprocess_mode == "color"
                else self._preprocess_one_image_grey(image)
            )
            preprocessed_images.append(preprocessed_image)
        return {self.input_name: np.stack(preprocessed_images, axis=0).astype(np.float32)}

    def postprocess(
        self, raw_outputs: Dict[str, NDArray[np.float32]]
    ) -> List[SemanticSegmentationResult]:
        """Postprocess for semantic segmentation model.

        Parameters
        ----------
        raw_outputs : Dict[str, NDArray[np.float32]]
            raw outputs from triton model represented by dictionary
            mapping output name to numpy array containing soft mask
            probabilities.

        Returns
        -------
        List[SemanticSegmentationResult]
            list of semantic segmentation results.
            Each item correponds to one batch item.
        """
        probs = raw_outputs[self.output_name]
        res = []
        for i in range(self.get_batch_size(self.input_name)):
            size = self.original_size_list[i]
            mask = probs[i, ...]
            if mask.shape[0] == 1:
                mask = sigmoid_like_softmax(mask)
                res.append(
                    SemanticSegmentationResult(
                        soft_mask=mask,
                        labels=["background", *self.labels],
                        dst_size=size,
                    )
                )
            else:
                mask = softmax_njit(mask)
                res.append(
                    SemanticSegmentationResult(soft_mask=mask, labels=self.labels, dst_size=size)
                )
        self.clear_original_size_list()
        return res
