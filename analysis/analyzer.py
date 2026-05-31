import numpy as np

from analysis.defect_detector import detect_defects
from analysis.object_segmenter import segment_object
from analysis.visualization import draw_defects
from models.analysis_request import AnalysisRequest
from models.analysis_result import AnalysisResult, DefectInfo


def analyze_image(request: AnalysisRequest) -> AnalysisResult:
    image = request.image
    settings = request.settings

    if settings.use_full_image_as_object:
        object_mask = _build_full_image_object_mask(image)
    else:
        object_mask = segment_object(image, settings)

    defect_mask, defects = detect_defects(image, object_mask, settings)
    marked_image = draw_defects(image, defects)

    object_area = _calculate_object_area(object_mask)
    total_defect_area = _calculate_total_defect_area(defects)
    defect_ratio = _calculate_defect_ratio(total_defect_area, object_area)
    largest_defect_area = _calculate_largest_defect_area(defects)
    average_defect_area = _calculate_average_defect_area(defects)

    return AnalysisResult(
        original_image=image,
        object_mask=object_mask,
        defect_mask=defect_mask,
        marked_image=marked_image,
        defects=defects,
        object_area=object_area,
        total_defect_area=total_defect_area,
        defect_ratio=defect_ratio,
        largest_defect_area=largest_defect_area,
        average_defect_area=average_defect_area
    )


def _build_full_image_object_mask(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    return np.full((height, width), 255, dtype=np.uint8)


def _calculate_object_area(object_mask: np.ndarray) -> int:
    return int(np.count_nonzero(object_mask))


def _calculate_total_defect_area(defects: list[DefectInfo]) -> float:
    return float(sum(defect.area for defect in defects))


def _calculate_defect_ratio(total_defect_area: float, object_area: int) -> float:
    if object_area <= 0:
        return 0.0

    return float(total_defect_area / object_area * 100)


def _calculate_largest_defect_area(defects: list[DefectInfo]) -> float:
    if not defects:
        return 0.0

    return float(max(defect.area for defect in defects))


def _calculate_average_defect_area(defects: list[DefectInfo]) -> float:
    if not defects:
        return 0.0

    return float(sum(defect.area for defect in defects) / len(defects))