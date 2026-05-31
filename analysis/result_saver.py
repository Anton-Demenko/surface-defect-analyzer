from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from numpy.typing import NDArray

from models.analysis_result import AnalysisResult


@dataclass
class SavedResultPaths:
    output_dir: Path
    marked_result_path: Path
    object_mask_path: Path
    defect_mask_path: Path
    report_path: Path


def save_analysis_result(
    result: AnalysisResult,
    output_dir: str | Path = "output"
) -> SavedResultPaths:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    marked_result_path = output_path / f"{timestamp}_marked_result.png"
    object_mask_path = output_path / f"{timestamp}_object_mask.png"
    defect_mask_path = output_path / f"{timestamp}_defect_mask.png"
    report_path = output_path / f"{timestamp}_report.txt"

    _save_rgb_image(result.marked_image, marked_result_path)
    _save_mask(result.object_mask, object_mask_path)
    _save_mask(result.defect_mask, defect_mask_path)
    _save_report(result, report_path)

    return SavedResultPaths(
        output_dir=output_path,
        marked_result_path=marked_result_path,
        object_mask_path=object_mask_path,
        defect_mask_path=defect_mask_path,
        report_path=report_path
    )


def _save_rgb_image(image: NDArray[np.uint8], path: Path) -> None:
    Image.fromarray(image).save(path)


def _save_mask(mask: NDArray[np.uint8], path: Path) -> None:
    Image.fromarray(mask).save(path)


def _save_report(result: AnalysisResult, path: Path) -> None:
    path.write_text(_build_report_text(result), encoding="utf-8")


def _build_report_text(result: AnalysisResult) -> str:
    lines = [
        "Surface Defect Analyzer Report",
        "",
        "Summary",
        f"Object area: {result.object_area} px",
        f"Detected defects: {len(result.defects)}",
        f"Total defect area: {result.total_defect_area:.2f} px",
        f"Defect ratio: {result.defect_ratio:.2f}%",
        f"Largest defect area: {result.largest_defect_area:.2f} px",
        f"Average defect area: {result.average_defect_area:.2f} px",
        "",
        "Defects",
    ]

    if not result.defects:
        lines.append("No defects detected.")
        return "\n".join(lines)

    lines.append("ID;X;Y;Width;Height;Area")

    for defect in result.defects:
        lines.append(
            f"{defect.id};"
            f"{defect.x};"
            f"{defect.y};"
            f"{defect.width};"
            f"{defect.height};"
            f"{defect.area:.2f}"
        )

    return "\n".join(lines)