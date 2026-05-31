from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


class ObjectNotFoundError(Exception):
    pass


@dataclass
class DefectInfo:
    id: int
    x: int
    y: int
    width: int
    height: int
    area: float


@dataclass
class AnalysisResult:
    original_image: NDArray[np.uint8]
    object_mask: NDArray[np.uint8]
    defect_mask: NDArray[np.uint8]
    marked_image: NDArray[np.uint8]
    defects: list[DefectInfo]
    object_area: int
    total_defect_area: float
    defect_ratio: float
    largest_defect_area: float
    average_defect_area: float