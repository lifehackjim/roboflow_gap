import logging
import typing as t
from pathlib import Path

import supervision as sv
from inference import get_model
from inference.core.models.base import Model as InferenceModel

from roboflow_gap.models import ImageData


class Analyze:
    def __init__(  # noqa: PLR0913
        self,
        api_key: str,
        model_id: str,
        confidence: float = 0.5,
        iou_threshold: float = 0.5,
        lazy_load: bool = False,
    ):
        self.api_key = api_key
        self.model_id = model_id
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.model: t.Optional[InferenceModel] = None
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

        if not lazy_load:
            self.load_model()

    async def load_model(self) -> InferenceModel:
        """Loads the Roboflow model."""
        self.model: InferenceModel = get_model(model_id=self.model_id, api_key=self.api_key)
        self.logger.debug("Loaded model: %s", self.model)
        return self.model

    async def run_inference(self, image_path: Path) -> t.Dict[str, t.Any]:
        """Runs inference on an image using the loaded model."""
        if not self.model:
            await self.load_model()
        results = self.model.infer(
            image=str(image_path), confidence=self.confidence, iou_threshold=self.iou_threshold
        )
        # NOTE1:
        # there are a lot of other arguments that can be passed to infer, and
        # the args seem to change based on the model type

        # NOTE2:
        # image can be a path or a cv2 image, but the type hinting is not clear on this

        # NOTE3:
        # we already have loaded the image in image_data.original, so we should pass that to
        # the model.infer instead of the path

        return results

    async def analyze_image(self, image_data: ImageData) -> ImageData:
        """Analyzes an image and updates the ImageData instance with analysis results."""
        if not image_data.original:
            return image_data  # Skip if there's no image to analyze

        try:
            image_path = image_data.context["path"]  # Assuming context contains the image path
            results = await self.run_inference(image_path)
            if results:
                detections = sv.Detections.from_inference(results[0])
                image_data.analysis = {
                    "detections": detections.to_dict()
                }  # Convert detections to a serializable format
                # Update date_analyze_done if necessary
            else:
                image_data.error = "No results from inference."
        except Exception as e:
            image_data.error = str(e)
            # Optionally log the error or handle it as needed

        return image_data

    @staticmethod
    def build_detections(inference_results: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
        """Processes inference results into a standardized format."""
        # Placeholder for custom processing logic
        detections: sv.Detections = sv.Detections.from_inference(inference_results[0])
        return {"detections": detections.to_dict()}
