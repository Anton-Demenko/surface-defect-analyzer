import cv2
import numpy as np
from numpy.typing import NDArray

from models.analysis_request import AnalysisSettings
from models.analysis_result import DefectInfo


def detect_defects(
    image: NDArray[np.uint8],
    object_mask: NDArray[np.uint8],
    settings: AnalysisSettings
) -> tuple[NDArray[np.uint8], list[DefectInfo]]:
    _validate_inputs(image, object_mask)

    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lightness = lab_image[:, :, 0]

    inner_object_mask = _build_inner_object_mask(object_mask)
    local_diff = _build_local_difference_map(lightness, settings.surface_smoothing)
    raw_defect_mask = _build_raw_defect_mask(
        local_diff,
        inner_object_mask,
        settings.defect_sensitivity
    )

    cleaned_mask = _clean_defect_mask(raw_defect_mask)
    final_mask, defects = _filter_defects_by_area(
        cleaned_mask,
        settings.min_defect_area
    )

    return final_mask, defects


def _validate_inputs(image: NDArray[np.uint8], object_mask: NDArray[np.uint8]) -> None:
    if image is None:
        raise ValueError("Image is None")

    if object_mask is None:
        raise ValueError("Object mask is None")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image must be a 3-channel RGB array")

    if object_mask.ndim != 2:
        raise ValueError("Object mask must be a single-channel array")

    if image.shape[:2] != object_mask.shape[:2]:
        raise ValueError("Image and object mask sizes must match")

    if image.dtype != np.uint8:
        raise ValueError("Image must have dtype uint8")

    if object_mask.dtype != np.uint8:
        raise ValueError("Object mask must have dtype uint8")


def _build_inner_object_mask(object_mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
    binary_mask = np.where(object_mask > 0, 1, 0).astype(np.uint8)
    distance_map = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)

    height, width = object_mask.shape[:2]
    edge_margin = max(8, int(min(height, width) * 0.015))

    inner_mask = np.where(distance_map > edge_margin, 255, 0).astype(np.uint8)

    if np.count_nonzero(inner_mask) == 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        inner_mask = cv2.erode(object_mask, kernel, iterations=1)

    if np.count_nonzero(inner_mask) == 0:
        return object_mask.copy()

    return inner_mask


def _build_local_difference_map(
    lightness: NDArray[np.uint8],
    smoothing_kernel_size: int
) -> NDArray[np.uint8]:
    kernel_size = _make_odd_kernel_size(smoothing_kernel_size)

    denoised = cv2.GaussianBlur(lightness, (3, 3), 0)
    smoothed = cv2.GaussianBlur(denoised, (kernel_size, kernel_size), 0)

    return cv2.absdiff(denoised, smoothed)


def _make_odd_kernel_size(value: int) -> int:
    value = max(3, int(value))

    if value % 2 == 0:
        value += 1

    return value


def _build_raw_defect_mask(
    local_diff: NDArray[np.uint8],
    inner_object_mask: NDArray[np.uint8],
    sensitivity: int
) -> NDArray[np.uint8]:
    object_pixels = local_diff[inner_object_mask > 0]

    if object_pixels.size == 0:
        return np.zeros_like(local_diff, dtype=np.uint8)

    mean_value = float(np.mean(object_pixels))
    std_value = float(np.std(object_pixels))

    statistical_threshold = mean_value + _sensitivity_to_k(sensitivity) * max(std_value, 1.0)
    percentile_threshold = float(np.percentile(object_pixels, _sensitivity_to_percentile(sensitivity)))
    threshold_value = max(statistical_threshold, percentile_threshold)

    _, threshold_mask = cv2.threshold(
        local_diff,
        threshold_value,
        255,
        cv2.THRESH_BINARY
    )

    return cv2.bitwise_and(threshold_mask, inner_object_mask)


def _sensitivity_to_k(sensitivity: int) -> float:
    sensitivity = max(1, min(100, int(sensitivity)))
    return 3.4 - ((sensitivity - 1) / 99) * 2.2


def _sensitivity_to_percentile(sensitivity: int) -> float:
    sensitivity = max(1, min(100, int(sensitivity)))
    return 99.7 - ((sensitivity - 1) / 99) * 7.7


def _clean_defect_mask(mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    return closed


def _filter_defects_by_area(
    mask: NDArray[np.uint8],
    min_area: int
) -> tuple[NDArray[np.uint8], list[DefectInfo]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    accepted_contours = []
    defects = []

    for contour in contours:
        area = float(cv2.contourArea(contour))

        if area < min_area:
            continue

        accepted_contours.append(contour)

    accepted_contours.sort(key=cv2.contourArea, reverse=True)

    final_mask = np.zeros_like(mask, dtype=np.uint8)

    for index, contour in enumerate(accepted_contours, start=1):
        x, y, width, height = cv2.boundingRect(contour)
        area = float(cv2.contourArea(contour))

        cv2.drawContours(final_mask, [contour], -1, 255, thickness=cv2.FILLED)

        defects.append(
            DefectInfo(
                id=index,
                x=int(x),
                y=int(y),
                width=int(width),
                height=int(height),
                area=area
            )
        )

    return final_mask, defects