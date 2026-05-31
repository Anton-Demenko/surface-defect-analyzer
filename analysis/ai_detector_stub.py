import numpy as np
from numpy.typing import NDArray

from models.analysis_request import AnalysisSettings


class AiDetectorNotImplementedError(NotImplementedError):
    pass


def detect_defects_ai(
    image: NDArray[np.uint8],
    object_mask: NDArray[np.uint8],
    settings: AnalysisSettings
) -> NDArray[np.uint8]:
    raise AiDetectorNotImplementedError(
        "AI defect detector is not implemented in MVP. "
        "This module is reserved for future neural network based detection."
    )