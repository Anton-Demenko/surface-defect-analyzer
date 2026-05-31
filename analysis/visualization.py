import cv2
import numpy as np
from numpy.typing import NDArray

from models.analysis_result import DefectInfo


def draw_defects(image: NDArray[np.uint8], defects: list[DefectInfo]) -> NDArray[np.uint8]:
    _validate_image(image)

    marked_image = image.copy()

    for defect in defects:
        _draw_defect(marked_image, defect)

    return marked_image


def _validate_image(image: NDArray[np.uint8]) -> None:
    if image is None:
        raise ValueError("Image is None")

    if image.ndim != 3:
        raise ValueError("Image must be a 3-channel RGB array")

    if image.shape[2] != 3:
        raise ValueError("Image must have exactly 3 channels")

    if image.dtype != np.uint8:
        raise ValueError("Image must have dtype uint8")


def _draw_defect(image: NDArray[np.uint8], defect: DefectInfo) -> None:
    x1 = defect.x
    y1 = defect.y
    x2 = defect.x + defect.width
    y2 = defect.y + defect.height

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        2
    )

    cv2.putText(
        image,
        str(defect.id),
        (x1, max(0, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2,
        cv2.LINE_AA
    )


def mask_to_display(mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
    if mask is None:
        raise ValueError("Mask is None")

    if mask.ndim != 2:
        raise ValueError("Mask must be a single-channel array")

    if mask.dtype != np.uint8:
        raise ValueError("Mask must have dtype uint8")

    return mask