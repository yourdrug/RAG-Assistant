class PaddleOCR:
    def __init__(self, *, use_angle_cls: bool, lang: str, show_log: bool) -> None: ...
    def ocr(
        self, image: object, *, cls: bool
    ) -> list[list[tuple[list[tuple[float, float]], tuple[str, float]]]] | None: ...

class PaddleOCRVL:
    def __init__(
        self,
        *,
        pipeline_version: str,
        use_doc_orientation_classify: bool,
        use_doc_unwarping: bool,
        layout_shape_mode: str,
    ) -> None: ...
