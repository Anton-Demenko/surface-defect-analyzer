import cv2
import numpy as np
from numpy.typing import NDArray

from models.analysis_request import AnalysisSettings
from models.analysis_result import ObjectNotFoundError


def segment_object(image: NDArray[np.uint8], settings: AnalysisSettings) -> NDArray[np.uint8]:
    _validate_image(image)

    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    border_pixels = _get_border_pixels(lab_image, settings.background_border_percent)
    background_color = _estimate_background_color(border_pixels)

    raw_mask = _build_foreground_mask(
        lab_image,
        background_color,
        settings.object_threshold
    )

    cleaned_mask = _clean_object_mask(raw_mask)

    if settings.keep_largest_object:
        cleaned_mask = _keep_largest_component(cleaned_mask)

    filled_mask = _fill_mask_holes(cleaned_mask)
    _validate_object_mask(filled_mask)

    return filled_mask


def _validate_image(image: NDArray[np.uint8]) -> None:
    if image is None:
        raise ValueError("Image is None")

    if image.ndim != 3:
        raise ValueError("Image must be a 3-channel RGB array")

    if image.shape[2] != 3:
        raise ValueError("Image must have exactly 3 channels")

    if image.dtype != np.uint8:
        raise ValueError("Image must have dtype uint8")


def _get_border_pixels(image_lab: NDArray[np.uint8], border_percent: int) -> NDArray[np.uint8]:
    height, width = image_lab.shape[:2]

    border_y = max(1, int(height * border_percent / 100))
    border_x = max(1, int(width * border_percent / 100))

    top = image_lab[:border_y, :, :]
    bottom = image_lab[height - border_y:, :, :]
    left = image_lab[:, :border_x, :]
    right = image_lab[:, width - border_x:, :]

    return np.concatenate([
        top.reshape(-1, 3),
        bottom.reshape(-1, 3),
        left.reshape(-1, 3),
        right.reshape(-1, 3)
    ], axis=0)


def _estimate_background_color(border_pixels: NDArray[np.uint8]) -> NDArray[np.float32]:
    return np.median(border_pixels, axis=0).astype(np.float32)


def _build_foreground_mask(
    image_lab: NDArray[np.uint8],
    background_color: NDArray[np.float32],
    threshold: int
) -> NDArray[np.uint8]:
    image_float = image_lab.astype(np.float32)
    diff = image_float - background_color
    distance = np.sqrt(np.sum(diff * diff, axis=2))

    return np.where(distance > threshold, 255, 0).astype(np.uint8)


def _clean_object_mask(mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

    return closed


def _keep_largest_component(mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if component_count <= 1:
        return np.zeros_like(mask)

    object_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(object_areas)) + 1

    return np.where(labels == largest_label, 255, 0).astype(np.uint8)


def _fill_mask_holes(mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filled = np.zeros_like(mask)

    if contours:
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)

    return filled


def _validate_object_mask(mask: NDArray[np.uint8]) -> None:
    image_area = mask.shape[0] * mask.shape[1]
    object_area = int(np.count_nonzero(mask))

    min_object_area = max(1, int(image_area * 0.005))

    if object_area < min_object_area:
        raise ObjectNotFoundError("Object was not found or object mask is too small")