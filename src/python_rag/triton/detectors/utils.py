from typing import Literal, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

from ...dto import BBox, Polygon


def get_detections_mask_for_roi(
    contour: Polygon,
    bboxes_xyxy: NDArray[np.float32],
    intersection_threshold: float,
    filter_method: Literal["iou", "point"],
) -> NDArray[np.bool_]:
    """Compute 2 polygon intersection fraction to get special metric like IoU.

    Parameters
    ----------
    contour : Polygon
        array of shape (n, 2) containing polygon coordinates in format (x, y).
    bboxes_xyxy : NDArray[np.float32]
        numpy array of shape (m, 4) containing raw bboxes of detections in xyxy
        format.
    intersection_threshold: float
        confidence threshold to pass object as valid
    filter_method: Literal["iou", "point"]
        selector to filter out detection boxes.
        Can be iou or point.

    Returns
    -------
    NDArray[np.bool_]
        bool array of shape (m, ) representing
        whether corresponding bounding boxes
        is in roi area or not.
    """
    mask = []
    for i in range(bboxes_xyxy.shape[0]):
        bbox = BBox(
            x1=int(bboxes_xyxy[i, 1]),
            y1=int(bboxes_xyxy[i, 2]),
            x2=int(bboxes_xyxy[i, 3]),
            y2=int(bboxes_xyxy[i, 4]),
        )
        cond = False
        if filter_method == "iou":
            cond = (contour.get_inter(bbox) / (bbox.get_area() + 1e-12)) > intersection_threshold
        elif filter_method == "point":
            cond = contour.contains(bbox.get_cxcy())
        else:
            raise NotImplementedError()
        mask.append(cond)
    return mask


def xyxy2tlwh(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x[..., [0, 1]], x[..., [2, 3]] - x[..., [0, 1]]], axis=1)


def box_iou(box1, box2):
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (
        (np.min(box1[:, None, 2:], box2[:, 2:]) - np.max(box1[:, None, :2], box2[:, :2]))
        .clamp(0)
        .prod(2)
    )
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def scale_coords(
    coords: np.ndarray, input_shape: Tuple[int, int], output_shape: Tuple[int, int]
) -> np.ndarray:
    """Scale input bounding boxes coordinates from input shape to output shape.

    First apply rescaling to input bounding boxes coordinates.
    After that clipping to shape of output image is performed.

    Parameters
    ----------
    coords : np.ndarray
        array of shape (n, 4) containing bounding boxes coordinates
        in format (x1, y1, x2, y2).
    input_shape : Tuple[int, int]
        tuple with shape of input image.
    output_shape : Tuple[int, int]
        tuple with shape of output image.

    Returns
    -------
    np.ndarray
        scaled and clipped bounding boxes
        in scale of output image shape.
    """
    coords = coords.copy()
    input_shape_ = (float(input_shape[0]), float(input_shape[1]))
    output_shape_ = (float(output_shape[0]), float(output_shape[1]))
    gain = min(input_shape_[0] / output_shape_[0], input_shape_[1] / output_shape_[1])
    pad = (
        (input_shape_[1] - output_shape_[1] * gain) / 2,
        (input_shape_[0] - output_shape_[0] * gain) / 2,
    )
    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    return clip_coords(coords=coords, image_shape=output_shape, inplace=True)


def clip_coords(
    coords: np.ndarray, image_shape: Tuple[int, int], inplace: bool = True
) -> np.ndarray:
    """Clip bounding boxes coordinates to fit image shape.

    Parameters
    ----------
    coords : np.ndarray
        array of shape (n, 4) containing bounding boxes coordinates
        in format (x1, y1, x2, y2).
    image_shape : Union[Tuple[int, int], int]
        image shape provided by tuple of two int values or single int value
        if sizes along x and y axis are considered to be equal.
    inplace : bool
        whether to compute clipping of bounding boxes values inplace or not.
        Default is True.

    Returns
    -------
    np.ndarray
        array of shape (n, 4) containing clipped bounding boxes values
        in format (x1, y1, x2, y2). If flag inplace is set to True then
        modified input array is returned, otherwise new array for clipped
        bounding boxes is created.
    """
    new_coords = np.stack(
        [
            np.clip(coords[:, 0], 0, image_shape[1]),
            np.clip(coords[:, 1], 0, image_shape[0]),
            np.clip(coords[:, 2], 0, image_shape[1]),
            np.clip(coords[:, 3], 0, image_shape[0]),
        ],
        axis=1,
    )

    if inplace:
        coords[...] = new_coords[...]
        return coords

    return new_coords


def letterbox(
    img,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_NEAREST)  # type: ignore
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(  # type: ignore
        img,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color,  # type: ignore
    )  # add border
    return img, ratio, (dw, dh)


def get_data_size(data: NDArray[np.float32]) -> int:
    """Get size of data in input array.

    Parameters
    ----------
    data: NDArray[np.float]
        input data.

    Returns
    -------
    int
        size of input data
    """
    return data.size * data.itemsize
