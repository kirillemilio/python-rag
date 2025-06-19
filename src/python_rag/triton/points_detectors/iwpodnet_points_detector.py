"""Contains implementation of iwpodnet triton model."""

from __future__ import annotations

from typing import Dict, List, Literal, cast

import cv2
import numpy as np
import tritonclient.grpc as grpcclient  # type: ignore
from numpy.typing import NDArray

from ...dto import Size
from ..models_factory import TritonModelFactory
from .points_detector import PointsDetectorTritonModel, PointsOutputDict


def get_rect_pts(tlx: float, tly: float, brx: float, bry: float):
    """Get the corner points of a rectangle given top-left and bottom-right coordinates.

    Parameters
    ----------
    tlx : float
        Top-left x-coordinate.
    tly : float
        Top-left y-coordinate.
    brx : float
        Bottom-right x-coordinate.
    bry : float
        Bottom-right y-coordinate.

    Returns
    -------
    np.matrix
        Matrix of rectangle corner points.
    """
    return np.matrix(
        [[tlx, brx, brx, tlx], [tly, tly, bry, bry], [1.0, 1.0, 1.0, 1.0]], dtype=float
    )


def iou(
    tl1: NDArray[np.float32],
    br1: NDArray[np.float32],
    tl2: NDArray[np.float32],
    br2: NDArray[np.float32],
) -> float:
    """Calculate the Intersection over Union (IoU) between two bounding boxes.

    Parameters
    ----------
    tl1 : np.ndarray
        Top-left coordinates of the first bounding box.
    br1 : np.ndarray
        Bottom-right coordinates of the first bounding box.
    tl2 : np.ndarray
        Top-left coordinates of the second bounding box.
    br2 : np.ndarray
        Bottom-right coordinates of the second bounding box.

    Returns
    -------
    float
        Intersection over Union (IoU) value.
    """
    wh1, wh2 = br1 - tl1, br2 - tl2
    assert (wh1 >= 0.0).all() and (wh2 >= 0.0).all()

    intersection_wh = np.maximum(np.minimum(br1, br2) - np.maximum(tl1, tl2), 0.0)
    intersection_area = np.prod(intersection_wh)
    area1, area2 = (np.prod(wh1), np.prod(wh2))
    union_area = area1 + area2 - intersection_area
    return intersection_area / union_area


def iou_labels(l1: DLabel, l2: DLabel):
    """Calculate IoU between two DLabel objects.

    Parameters
    ----------
    l1 : DLabel
        First DLabel object.
    l2 : DLabel
        Second DLabel object.

    Returns
    -------
    float
        IoU value.
    """
    return iou(l1.tl(), l1.br(), l2.tl(), l2.br())


def nms(labels: List[DLabel], iou_threshold: float = 0.5) -> List[DLabel]:
    """Apply Non-Maximum Suppression (NMS) to filter out overlapping bounding boxes.

    Parameters
    ----------
    labels : List[DLabel]
        List of DLabel objects to apply NMS on.
    iou_threshold : float, optional
        IoU threshold for determining overlaps, by default 0.5.

    Returns
    -------
    List[DLabel]
        List of DLabel objects after applying NMS.
    """
    selected_labels: List[DLabel] = []
    labels.sort(key=lambda l: l.prob(), reverse=True)

    for label in labels:
        non_overlap = True
        for sel_label in selected_labels:
            if iou_labels(label, sel_label) > iou_threshold:
                non_overlap = False
                break

        if non_overlap:
            selected_labels.append(label)

    return selected_labels


def four_point_transform(
    image: NDArray[np.float32], points: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Get plate crop from image using points.

    Parameters
    ----------
    image : NDArray[np.float32]
        input image from which iwpodnet points were extracted.
    points : NDArray[np.float32]
        numpy array of shape of shape (4, 2) and dtype np.float32
        containing 4 border points from iwpodnet.
        The last dimension is in xy coords order.

    Returns
    -------
    NDArray[np.float32]
        transformed and extracted licence plate crop.
    """
    # obtain a consistent order of the points and unpack them
    # individually
    rect = points.astype(np.float32)
    (tl, tr, br, bl) = rect
    # coords are in x, y format
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))  # type: ignore
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))  # type: ignore
    max_width = max(int(width_a), int(width_b))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))  # type: ignore
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))  # type: ignore
    max_height = max(int(height_a), int(height_b))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )
    # compute the perspective transform matrix and then apply it
    transform_matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cast(
        NDArray[np.float32], cv2.warpPerspective(image, transform_matrix, (max_width, max_height))
    )
    # return the warped image
    return warped


def four_point_crop(image: NDArray[np.float32], points: NDArray[np.float32]):
    """Create crop from iamge by four points.

    Parameters
    ----------
    image : NDArray[np.float32]
        image from which crop will be created.
    points : NDArray[np.float32]
        points that will be used to detect cropping points.
    """
    rect = points.astype(int)
    x_min = rect[:, 0].min()
    x_max = rect[:, 0].max()
    y_min = rect[:, 1].min()
    y_max = rect[:, 1].max()
    crop = image[y_min:y_max, x_min:x_max, :]
    return crop


class Label:
    """Represents a bounding box label with class, coordinates, and probability."""

    def __init__(self, cl=-1, tl=np.array([0.0, 0.0]), br=np.array([0.0, 0.0]), prob=None):
        self.__tl = tl
        self.__br = br
        self.__cl = cl
        self.__prob = prob

    def __str__(self) -> str:
        """Get string representation of label.

        Returns
        -------
        str
            string representation of label.
        """
        return "Class: %d, top_left(x:%f,y:%f), bottom_right(x:%f,y:%f)" % (
            self.__cl,
            self.__tl[0],
            self.__tl[1],
            self.__br[0],
            self.__br[1],
        )

    def copy(self):
        """Create a copy of the label.

        Returns
        -------
        Label
            Copy of the label.
        """
        return Label(self.__cl, self.__tl, self.__br)

    def wh(self):
        """Get the width and height of the bounding box.

        Returns
        -------
        np.ndarray
            Width and height of the bounding box.
        """
        return self.__br - self.__tl

    def cc(self):
        """Get the center coordinates of the bounding box.

        Returns
        -------
        np.ndarray
            Center coordinates of the bounding box.
        """
        return self.__tl + self.wh() / 2

    def tl(self):
        """Get the top-left coordinates of the bounding box.

        Returns
        -------
        np.ndarray
            Top-left coordinates of the bounding box.
        """
        return self.__tl

    def br(self):
        """Get the bottom-right coordinates of the bounding box.

        Returns
        -------
        np.ndarray
            Bottom-right coordinates of the bounding box.
        """
        return self.__br

    def tr(self):
        """Get the top-right coordinates of the bounding box.

        Returns
        -------
        np.ndarray
            Top-right coordinates of the bounding box.
        """
        return np.array([self.__br[0], self.__tl[1]])

    def bl(self):
        """Get the bottom-left coordinates of the bounding box.

        Returns
        -------
        np.ndarray
            Bottom-left coordinates of the bounding box.
        """
        return np.array([self.__tl[0], self.__br[1]])

    def cl(self):
        """Get the class of the bounding box.

        Returns
        -------
        int
            Class of the bounding box.
        """
        return self.__cl

    def area(self):
        """Calculate the area of the bounding box.

        Returns
        -------
        float
            Area of the bounding box.
        """
        return np.prod(self.wh())

    def prob(self):
        """Get the probability of the bounding box.

        Returns
        -------
        float
            Probability of the bounding box.
        """
        return self.__prob

    def set_class(self, cl):
        """Set the class of the bounding box.

        Parameters
        ----------
        cl : int
            Class to set.
        """
        self.__cl = cl

    def set_tl(self, tl):
        """Set the top-left coordinates of the bounding box.

        Parameters
        ----------
        tl : np.ndarray
            Top-left coordinates to set.
        """
        self.__tl = tl

    def set_br(self, br):
        """Set the bottom-right coordinates of the bounding box.

        Parameters
        ----------
        br : np.ndarray
            Bottom-right coordinates to set.
        """
        self.__br = br

    def set_wh(self, wh):
        """Set the width and height of the bounding box.

        Parameters
        ----------
        wh : np.ndarray
            Width and height to set.
        """
        cc = self.cc()
        self.__tl = cc - 0.5 * wh
        self.__br = cc + 0.5 * wh

    def set_prob(self, prob):
        """Set the probability of the bounding box.

        Parameters
        ----------
        prob : float
            Probability to set.
        """
        self.__prob = prob


class DLabel(Label):
    """Represents a detected label with points and probability."""

    def __init__(self, cl, pts, prob):
        self.pts = pts
        tl = np.amin(pts, 1)
        br = np.amax(pts, 1)
        super().__init__(cl, tl, br, prob)


@TritonModelFactory.register_model(model_type="points-detector", arch_type="iwpodnet")
class IWpodNetTritonModel(PointsDetectorTritonModel):
    """IWpodNet triton model implementation."""

    iou_threshold: float
    conf_threshold: float

    default_base: NDArray[np.float32]

    preprocessed_size_list: List[Size]

    def __init__(
        self,
        client: grpcclient.InferenceServerClient,
        model_name: str,
        input_name: str,
        input_size: Size,
        output_name: str,
        client_timeout: float | None,
        iou_threshold: float = 0.1,
        conf_threshold: float = 0.35,
        model_version: str = "1",
        device_id: int = 0,
        use_cushm: bool = False,
        **kwargs,
    ):
        super().__init__(
            client=client,
            model_name=model_name,
            input_name=input_name,
            input_size=input_size,
            output_name=output_name,
            iou_threshold=iou_threshold,
            conf_threshold=conf_threshold,
            client_timeout=client_timeout,
            model_version=model_version,
            device_id=device_id,
            use_cushm=use_cushm,
            channels_last=True,
            allow_spatial_adjustment=True,
            **kwargs,
        )
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        self.preprocessed_size_list = []
        vx, vy = 0.5, 0.5
        self.default_base = np.array(
            [[-vx, -vy, 1.0], [vx, -vy, 1.0], [vx, vy, 1.0], [-vx, vy, 1.0]]
        ).T

    def _preprocess_one_image(self, orig_image: NDArray[np.float32]):
        orig_image = cv2.cvtColor(orig_image, cv2.COLOR_RGB2BGR) / 255.0  # type: ignore
        orig_image = orig_image.astype(np.float32)

        orig_size = Size(h=orig_image.shape[0], w=orig_image.shape[1])
        ar = max(1, min(2.75, 1.0 * orig_size.w / orig_size.h))
        max_width = ar * self.input_size.w
        factor = min(1, max_width / orig_size.w)

        net_step = 2**4

        w = int(orig_size.w * factor)
        h = int(orig_size.h * factor)

        if w % net_step != 0:
            w += net_step - w % net_step
        if h % net_step != 0:
            h += net_step - h % net_step

        return cv2.resize(orig_image, (w, h), interpolation=cv2.INTER_CUBIC)

    def set_preprocessed_size_list(self, images: List[NDArray[np.float32]]):
        """Store preprocessed input images sizes for inference.

        Stored sizes will be used for outputs rescale and callibration
        purposes typically inside postprocess method.

        Parameters
        ----------
        images : List[NDArray[np.float32]]
            list of input images whose sizes will
            be saved.
        """
        for image in images:
            self.preprocessed_size_list.append(Size(w=image.shape[1], h=image.shape[0]))

    def clear_preprocessed_size_list(self):
        """Clear the list containing preprocessed images sizes.

        This method must be called after each completion
        of model inference, typically in postprocess method.
        """
        self.preprocessed_size_list.clear()

    def preprocess(self, inputs: List[NDArray[np.float32]]) -> Dict[str, NDArray[np.float32]]:
        """Preprocess input images into feed dict for triton model.

        Parameters
        ----------
        inputs : List[NDArray[np.float32]]
            list containing input images that will
            be processed in a batch fashion.

        Returns
        -------
        Dict[str, NDArray[np.float32]]
            dictionary mapping triton model output name
            into raw output numpy array.
        """
        self.set_original_size_list(images=inputs)
        preprocessed_images = [
            self._preprocess_one_image(orig_image=input_image) for input_image in inputs
        ]
        self.set_preprocessed_size_list(images=preprocessed_images)
        return {
            self.input_name: np.concatenate(
                [preprocessed_image[np.newaxis, ...] for preprocessed_image in preprocessed_images],
                axis=0,
            )
        }

    def _get_valid_points_mask(self, points: NDArray[np.float32]) -> NDArray[np.float32]:
        """Get boolean mask for valid points.

        Parameters
        ----------
        points : NDArray[np.float32]
            points array of shape (n, 4, 2).

        Returns
        -------
        NDArray[np.bool_]
            boolean array of shape (n, )
            indicating whether points are valid.
        """
        x_min, x_max = np.min(points[:, :, 0], axis=1), np.max(points[:, :, 0], axis=1)
        y_min, y_max = np.min(points[:, :, 1], axis=1), np.max(points[:, :, 0], axis=1)
        return ((x_max - x_min) > 0) | ((y_max - y_min) > 0)

    def postprocess(self, raw_outputs: Dict[str, NDArray[np.float32]]) -> List[PointsOutputDict]:
        """Apply postprocessing for raw iwpodnet outputs.

        Parameters
        ----------
        raw_outputs : Dict[str, NDArray[np.float32]]
            mapping from triton models output name to raw numpy array
            associated with this output.

        Returns
        -------
        List[PointsOutputDict]
            list of iwpod output dict.
            Each element of the list corresponds
            to element in batch.
            Each element is a dict of type
            PointsOutputDict
            containing probs of set of points
            and points.
        """
        predictions = raw_outputs[self.output_name]  # (batch_size, h, w, 6) shape

        # Some magic number here
        net_stride = 2**4
        side = ((208.0 + 40.0) / 2.0) / net_stride  # based on rescaling of training data

        probs = predictions[..., 0]
        affines = predictions[..., -6:]  # affine transform params

        b_coords, y_coords, x_coords = np.where(probs > self.conf_threshold)

        results: List[PointsOutputDict] = []
        for batch_index in range(probs.shape[0]):
            original_size = self.original_size_list[batch_index]
            preprocessed_size = self.preprocessed_size_list[batch_index]

            batch_mask = b_coords == batch_index
            bi = b_coords[batch_mask]
            yi = y_coords[batch_mask]
            xi = x_coords[batch_mask]

            affine_ = affines[bi, yi, xi, :]
            prob_ = probs[bi, yi, xi]

            if affine_.size == 0:
                results.append(
                    {
                        "points": np.zeros((0, 4, 2), dtype=np.float32),
                        "probs": np.zeros((0,), dtype=np.float32),
                    }
                )
                continue

            a = affine_.reshape(-1, 2, 3)
            a[:, 0, 0] = np.clip(a[:, 0, 0], a_min=0.0, a_max=None)
            a[:, 1, 1] = np.clip(a[:, 1, 1], a_min=0.0, a_max=None)

            points = a @ self.default_base * side + np.expand_dims(
                np.stack([xi + 0.5, yi + 0.5], axis=1), axis=-1
            )
            points = (
                net_stride
                * points
                / np.array([preprocessed_size.w, preprocessed_size.h], dtype=np.float32).reshape(
                    1, 2, 1
                )
            )

            labels = [DLabel(0, points[j, :], prob_[j]) for j in range(points.shape[0])]
            labels = nms(labels, iou_threshold=0.1)
            labels.sort(key=lambda x: x.prob(), reverse=True)

            result_points, result_probs = [], []
            for label in labels:
                u = label.pts * np.array(
                    [original_size.w, original_size.h], dtype=np.float32
                ).reshape(2, 1)
                u = np.transpose(u, axes=(1, 0))
                result_points.append(u)

                prob = 1.0 if label.prob() is None else float(label.prob())
                result_probs.append(prob)

            result_points_numpy = np.stack(result_points)
            result_probs_numpy = np.array(result_probs, dtype=np.float32)
            result_points_mask = self._get_valid_points_mask(result_points_numpy)
            results.append(
                {
                    "points": result_points_numpy[result_points_mask, ...],
                    "probs": result_probs_numpy[result_points_mask, ...],
                }
            )

        self.clear_original_size_list()
        self.clear_preprocessed_size_list()

        return results

    def get_crops(
        self,
        image: NDArray[np.float32],
        output: PointsOutputDict,
        crop_mode: Literal["perspective", "borders"],
    ) -> List[NDArray[np.float32]]:
        """Get list of crops from image using iwpodnet points.

        Parameters
        ----------
        image : NDArray[np.float32]
            source image from which crops will be extracted.
        output : PointsOutputDict
            detected points dict.
        crop_mode : Literal["perspective", "borders"]
            switch to select iwpodnet postprocessing method

        Returns
        -------
        List[NDArray[np.float32]]
            list of crops from the image.
        """
        points = output["points"]
        num_rects = points.shape[0]
        crops = []
        for i in range(num_rects):
            if crop_mode == "perspective":
                crop = four_point_transform(image, points=points[i, ...])
            else:
                crop = four_point_crop(image, points=points[i, ...])
            crops.append(crop)
        return crops
