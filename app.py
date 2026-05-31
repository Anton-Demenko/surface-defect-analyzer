from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from analysis.analyzer import analyze_image
from analysis.result_saver import save_analysis_result
from models.analysis_request import AnalysisRequest, AnalysisSettings
from models.analysis_result import AnalysisResult, ObjectNotFoundError


ANALYSIS_MODE_LABELS = {
    "Одно изображение": "single_image",
    "По нормальным примерам": "normal_examples"
}


def main() -> None:
    st.set_page_config(
        page_title="Surface Defect Analyzer",
        layout="wide"
    )

    st.title("Surface Defect Analyzer")
    st.subheader("Система анализа изображений для обнаружения дефектов поверхности объекта")

    st.write(
        "Загрузите изображение объекта на контрастном фоне. "
        "Приложение выделит объект, исключит фон из анализа и найдёт подозрительные визуальные дефекты на поверхности."
    )

    uploaded_file, normal_files = _build_sidebar()

    if uploaded_file is None:
        st.info("Загрузите изображение для начала анализа.")
        return

    image = _load_image(uploaded_file)
    normal_images = _load_images(normal_files)
    settings = _read_settings_from_sidebar()

    if settings.analysis_mode == "normal_examples" and not normal_images:
        st.info(
            "Для режима анализа по нормальным примерам загрузите одно или несколько нормальных изображений "
            "того же типа поверхности."
        )
        st.image(image, caption="Проверяемое изображение", use_container_width=True)
        return

    request = AnalysisRequest(
        image=image,
        settings=settings,
        normal_images=normal_images
    )

    try:
        result = analyze_image(request)
    except ObjectNotFoundError:
        st.error(
            "Объект не найден. Попробуйте изменить чувствительность выделения объекта. "
            "Убедитесь, что фон контрастный и виден по краям изображения."
        )
        st.image(image, caption="Исходное изображение", use_container_width=True)
        return
    except ValueError as error:
        st.error(f"Ошибка обработки изображения: {error}")
        return

    _show_images(result)
    _show_statistics(result)
    _show_defects_table(result)

    if settings.analysis_mode == "normal_examples" and normal_images:
        _show_normal_examples_preview(normal_images)

    _show_save_button(result)


def _build_sidebar() -> tuple[BytesIO | None, list[BytesIO]]:
    st.sidebar.header("Загрузка изображения")

    uploaded_file = st.sidebar.file_uploader(
        "Проверяемое изображение",
        type=["jpg", "jpeg", "png"]
    )

    st.sidebar.header("Режим анализа")

    st.sidebar.selectbox(
        "Режим",
        options=list(ANALYSIS_MODE_LABELS.keys()),
        index=0,
        key="analysis_mode_label"
    )

    analysis_mode = ANALYSIS_MODE_LABELS[st.session_state.analysis_mode_label]

    normal_files = []

    if analysis_mode == "normal_examples":
        normal_files = st.sidebar.file_uploader(
            "Нормальные примеры",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )

    st.sidebar.header("Выделение объекта")

    use_full_image_as_object = st.sidebar.checkbox(
        "Анализировать всё изображение как объект",
        value=False,
        key="use_full_image_as_object"
    )

    st.sidebar.slider(
        "Чувствительность выделения объекта",
        min_value=10,
        max_value=100,
        value=35,
        key="object_threshold",
        disabled=use_full_image_as_object
    )

    st.sidebar.slider(
        "Размер области фона по краям, %",
        min_value=2,
        max_value=20,
        value=8,
        key="background_border_percent",
        disabled=use_full_image_as_object
    )

    st.sidebar.checkbox(
        "Оставлять только крупнейший объект",
        value=True,
        key="keep_largest_object",
        disabled=use_full_image_as_object
    )

    st.sidebar.header("Поиск дефектов")

    st.sidebar.slider(
        "Чувствительность поиска дефектов",
        min_value=1,
        max_value=100,
        value=45,
        key="defect_sensitivity"
    )

    st.sidebar.slider(
        "Минимальная площадь дефекта, px",
        min_value=10,
        max_value=5000,
        value=100,
        key="min_defect_area"
    )

    st.sidebar.slider(
        "Сглаживание поверхности",
        min_value=3,
        max_value=51,
        value=21,
        step=2,
        key="surface_smoothing"
    )

    return uploaded_file, normal_files


def _read_settings_from_sidebar() -> AnalysisSettings:
    return AnalysisSettings(
        analysis_mode=ANALYSIS_MODE_LABELS[st.session_state.analysis_mode_label],
        use_full_image_as_object=st.session_state.use_full_image_as_object,
        object_threshold=st.session_state.object_threshold,
        background_border_percent=st.session_state.background_border_percent,
        keep_largest_object=st.session_state.keep_largest_object,
        defect_sensitivity=st.session_state.defect_sensitivity,
        min_defect_area=st.session_state.min_defect_area,
        surface_smoothing=st.session_state.surface_smoothing
    )


def _load_image(uploaded_file: BytesIO) -> np.ndarray:
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)


def _load_images(uploaded_files: list[BytesIO]) -> list[np.ndarray]:
    return [_load_image(uploaded_file) for uploaded_file in uploaded_files]


def _show_images(result: AnalysisResult) -> None:
    top_left, top_right = st.columns(2)

    with top_left:
        st.image(
            result.original_image,
            caption="Исходное изображение",
            use_container_width=True
        )

    with top_right:
        st.image(
            result.object_mask,
            caption="Маска объекта",
            use_container_width=True
        )

    bottom_left, bottom_right = st.columns(2)

    with bottom_left:
        st.image(
            result.defect_mask,
            caption="Маска дефектов",
            use_container_width=True
        )

    with bottom_right:
        st.image(
            result.marked_image,
            caption="Результат с разметкой",
            use_container_width=True
        )


def _show_statistics(result: AnalysisResult) -> None:
    st.header("Статистика анализа")

    col1, col2, col3 = st.columns(3)

    col1.metric("Площадь объекта", f"{result.object_area} px")
    col2.metric("Найдено дефектов", len(result.defects))
    col3.metric("Доля дефектов", f"{result.defect_ratio:.2f}%")

    col4, col5, col6 = st.columns(3)

    col4.metric("Суммарная площадь дефектов", f"{result.total_defect_area:.1f} px")
    col5.metric("Крупнейший дефект", f"{result.largest_defect_area:.1f} px")
    col6.metric("Средняя площадь дефекта", f"{result.average_defect_area:.1f} px")


def _show_defects_table(result: AnalysisResult) -> None:
    st.header("Найденные дефекты")

    if not result.defects:
        st.info(
            "Дефекты не найдены при текущих параметрах. "
            "Можно повысить чувствительность поиска дефектов или уменьшить минимальную площадь дефекта."
        )
        return

    data = [
        {
            "ID": defect.id,
            "X": defect.x,
            "Y": defect.y,
            "Width": defect.width,
            "Height": defect.height,
            "Area": round(defect.area, 2)
        }
        for defect in result.defects
    ]

    st.dataframe(pd.DataFrame(data), use_container_width=True)


def _show_normal_examples_preview(normal_images: list[np.ndarray]) -> None:
    st.header("Нормальные примеры")

    columns = st.columns(min(4, len(normal_images)))

    for index, image in enumerate(normal_images[:4]):
        with columns[index % len(columns)]:
            st.image(
                image,
                caption=f"Нормальный пример {index + 1}",
                use_container_width=True
            )

    if len(normal_images) > 4:
        st.caption(f"Загружено нормальных примеров: {len(normal_images)}")


def _show_save_button(result: AnalysisResult) -> None:
    st.sidebar.header("Сохранение результата")

    if not st.sidebar.button("Сохранить результат"):
        return

    saved_paths = save_analysis_result(result)

    st.sidebar.success("Результат сохранён.")
    st.sidebar.write(f"Папка: `{saved_paths.output_dir}`")
    st.sidebar.write(f"Разметка: `{saved_paths.marked_result_path.name}`")
    st.sidebar.write(f"Маска объекта: `{saved_paths.object_mask_path.name}`")
    st.sidebar.write(f"Маска дефектов: `{saved_paths.defect_mask_path.name}`")
    st.sidebar.write(f"Отчёт: `{saved_paths.report_path.name}`")


if __name__ == "__main__":
    main()