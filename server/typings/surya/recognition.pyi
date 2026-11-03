from PIL.Image import Image
from surya.detection import DetectionPredictor

class TextLine:
    text: str
    confidence: float | None
    bbox: list[float] | None

class RecognitionResult:
    text_lines: list[TextLine]

class RecognitionPredictor:
    def __init__(self) -> None: ...
    def __call__(
        self, images: list[Image], languages: list[list[str]], detector: DetectionPredictor
    ) -> list[RecognitionResult]: ...
