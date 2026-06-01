import cv2
import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors

from analysis.object_segmenter import segment_object
from models.analysis_request import AnalysisSettings
from models.analysis_result import DefectInfo, ObjectNotFoundError


def detect_patch_anomalies(
    image: NDArray[np.uint8],
    object_mask: NDArray[np.uint8],
    normal_images: list[NDArray[np.uint8]],
    settings: AnalysisSettings
) -> tuple[NDArray[np.uint8], list[DefectInfo]]:
    _validate_inputs(image, object_mask, normal_images)

    patch_size = 32
    stride = 16

    normal_features = _build_normal_feature_bank(
        normal_images,
        settings,
        patch_size,
        stride
    )

    if len(normal_features) < 10:
        return np.zeros(object_mask.shape, dtype=np.uint8), []

    test_features, test_positions = _extract_patch_features(
        image,
        object_mask,
        patch_size,
        stride
    )

    if len(test_features) == 0:
        return np.zeros(object_mask.shape, dtype=np.uint8), []

    normal_features = np.asarray(normal_features, dtype=np.float32)
    test_features = np.asarray(test_features, dtype=np.float32)

    normal_mean = np.mean(normal_features, axis=0)
    normal_std = np.std(normal_features, axis=0)
    normal_std = np.where(normal_std < 1e-6, 1.0, normal_std)

    normal_scaled = (normal_features - normal_mean) / normal_std
    test_scaled = (test_features - normal_mean) / normal_std

    neighbor_count = 2 if len(normal_scaled) >= 2 else 1

    model = NearestNeighbors(
        n_neighbors=neighbor_count,
        algorithm="auto",
        metric="euclidean"
    )

    model.fit(normal_scaled)

    test_distances, _ = model.kneighbors(test_scaled, n_neighbors=1)
    test_scores = test_distances[:, 0]

    normal_reference_scores = _calculate_normal_reference_scores(
        model,
        normal_scaled,
        neighbor_count
    )

    threshold = _calculate_patch_threshold(
        test_scores,
        normal_reference_scores,
        settings.defect_sensitivity
    )

    raw_mask = _build_patch_score_mask(
        object_mask.shape,
        test_positions,
        test_scores,
        threshold,
        patch_size
    )

    raw_mask = cv2.bitwise_and(raw_mask, object_mask)
    cleaned_mask = _clean_patch_mask(raw_mask)
    final_mask, defects = _filter_defects_by_area(
        cleaned_mask,
        settings.min_defect_area
    )

    return final_mask, defects


def _validate_inputs(
    image: NDArray[np.uint8],
    object_mask: NDArray[np.uint8],
    normal_images: list[NDArray[np.uint8]]
) -> None:
    if image is None:
        raise ValueError("Image is None")

    if object_mask is None:
        raise ValueError("Object mask is None")

    if not normal_images:
        raise ValueError("Normal images list is empty")

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


def _build_normal_feature_bank(
    normal_images: list[NDArray[np.uint8]],
    settings: AnalysisSettings,
    patch_size: int,
    stride: int
) -> list[list[float]]:
    features = []

    for normal_image in normal_images:
        normal_mask = _build_normal_image_mask(normal_image, settings)

        image_features, _ = _extract_patch_features(
            normal_image,
            normal_mask,
            patch_size,
            stride
        )

        features.extend(image_features)

    max_features = 6000

    if len(features) <= max_features:
        return features

    indices = np.linspace(0, len(features) - 1, max_features).astype(int)
    return [features[index] for index in indices]


def _build_normal_image_mask(
    image: NDArray[np.uint8],
    settings: AnalysisSettings
) -> NDArray[np.uint8]:
    if settings.use_full_image_as_object:
        height, width = image.shape[:2]
        return np.full((height, width), 255, dtype=np.uint8)

    try:
        return segment_object(image, settings)
    except ObjectNotFoundError:
        height, width = image.shape[:2]
        return np.full((height, width), 255, dtype=np.uint8)


def _extract_patch_features(
    image: NDArray[np.uint8],
    mask: NDArray[np.uint8],
    patch_size: int,
    stride: int
) -> tuple[list[list[float]], list[tuple[int, int]]]:
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    gray = lab_image[:, :, 0]

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)

    height, width = gray.shape[:2]

    features = []
    positions = []

    half = patch_size // 2

    for y in range(half, height - half + 1, stride):
        for x in range(half, width - half + 1, stride):
            y1 = y - half
            y2 = y + half
            x1 = x - half
            x2 = x + half

            mask_patch = mask[y1:y2, x1:x2]

            if _mask_coverage(mask_patch) < 0.85:
                continue

            lab_patch = lab_image[y1:y2, x1:x2]
            gray_patch = gray[y1:y2, x1:x2]
            gradient_patch = gradient[y1:y2, x1:x2]

            features.append(_calculate_patch_features(lab_patch, gray_patch, gradient_patch))
            positions.append((x, y))

    return features, positions


def _mask_coverage(mask_patch: NDArray[np.uint8]) -> float:
    return float(np.count_nonzero(mask_patch)) / float(mask_patch.size)


def _calculate_patch_features(
    lab_patch: NDArray[np.uint8],
    gray_patch: NDArray[np.uint8],
    gradient_patch: NDArray[np.float32]
) -> list[float]:
    lab_float = lab_patch.astype(np.float32)
    gray_float = gray_patch.astype(np.float32)

    l_channel = lab_float[:, :, 0]
    a_channel = lab_float[:, :, 1]
    b_channel = lab_float[:, :, 2]

    return [
        float(np.mean(l_channel)),
        float(np.std(l_channel)),
        float(np.mean(a_channel)),
        float(np.std(a_channel)),
        float(np.mean(b_channel)),
        float(np.std(b_channel)),
        float(np.mean(gray_float)),
        float(np.std(gray_float)),
        float(np.mean(gradient_patch)),
        float(np.std(gradient_patch)),
        float(np.percentile(gray_float, 10)),
        float(np.percentile(gray_float, 90))
    ]


def _calculate_normal_reference_scores(
    model: NearestNeighbors,
    normal_scaled: NDArray[np.float32],
    neighbor_count: int
) -> NDArray[np.float32]:
    if len(normal_scaled) < 2 or neighbor_count < 2:
        return np.zeros(len(normal_scaled), dtype=np.float32)

    distances, _ = model.kneighbors(normal_scaled, n_neighbors=2)
    return distances[:, 1].astype(np.float32)


def _calculate_patch_threshold(
    test_scores: NDArray[np.float32],
    normal_reference_scores: NDArray[np.float32],
    sensitivity: int
) -> float:
    sensitivity = max(1, min(100, int(sensitivity)))

    normal_mean = float(np.mean(normal_reference_scores))
    normal_std = float(np.std(normal_reference_scores))

    k = 4.2 - ((sensitivity - 1) / 99) * 2.4
    normal_threshold = normal_mean + k * max(normal_std, 0.05)

    percentile = 99.8 - ((sensitivity - 1) / 99) * 6.8
    test_percentile_threshold = float(np.percentile(test_scores, percentile))

    return max(normal_threshold, test_percentile_threshold)


def _build_patch_score_mask(
    shape: tuple[int, int],
    positions: list[tuple[int, int]],
    scores: NDArray[np.float32],
    threshold: float,
    patch_size: int
) -> NDArray[np.uint8]:
    mask = np.zeros(shape, dtype=np.uint8)

    half = patch_size // 2

    for (x, y), score in zip(positions, scores):
        if score <= threshold:
            continue

        x1 = max(0, x - half)
        x2 = min(shape[1], x + half)
        y1 = max(0, y - half)
        y2 = min(shape[0], y + half)

        mask[y1:y2, x1:x2] = 255

    return mask


def _clean_patch_mask(mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

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