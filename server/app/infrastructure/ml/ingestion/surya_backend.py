"""Construct Surya predictors after the ingestion runner loads this module lazily."""

from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor


def create_surya_predictors() -> tuple[RecognitionPredictor, DetectionPredictor]:
    return RecognitionPredictor(), DetectionPredictor()
