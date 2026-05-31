from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class AnalysisSettings:
    use_full_image_as_object: bool = False
    object_threshold: int = 35
    background_border_percent: int = 8
    keep_largest_object: bool = True
    defect_sensitivity: int = 45
    min_defect_area: int = 100
    surface_smoothing: int = 21

    def __post_init__(self) -> None:
        self.object_threshold = max(1, min(255, self.object_threshold))
        self.background_border_percent = max(1, min(30, self.background_border_percent))
        self.defect_sensitivity = max(1, min(100, self.defect_sensitivity))
        self.min_defect_area = max(1, self.min_defect_area)
        self.surface_smoothing = max(3, self.surface_smoothing)

        if self.surface_smoothing % 2 == 0:
            self.surface_smoothing += 1


@dataclass
class AnalysisRequest:
    image: NDArray[np.uint8]
    settings: AnalysisSettings