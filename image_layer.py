import os
import pathlib
import typing as t

import cv2
import dotenv
import mss
import numpy as np
import supervision as sv
from inference.core.models.base import Model as BaseInferenceModel
from inference.models.utils import get_roboflow_model
from supervision.annotators.base import BaseAnnotator

dotenv.load_dotenv()

# Image path
IMAGE_PATH: t.Union[str, pathlib.Path] = "image_single.jpg"

# Roboflow model
MODEL_NAME: str = "face-detection-mik1i"
MODEL_VERSION: str = "18"
MODEL_ID: str = f"{MODEL_NAME}/{MODEL_VERSION}"

# Roboflow API key
API_KEY: str = os.environ.get("ROBOFLOW_API_KEY")

ANNOTATORS: t.Tuple[t.Union[str, BaseAnnotator, sv.LabelAnnotator]] = (
    sv.BoundingBoxAnnotator(),
    sv.LabelAnnotator(),
)


def pathify(
    path: t.Union[str, pathlib.Path],
    resolve: bool = True,
    absolute: bool = True,
    as_file: t.Optional[bool] = None,
) -> pathlib.Path:
    """Get path object from string or path."""
    path = pathlib.Path(path)

    if resolve:
        path = path.resolve()

    if absolute:
        path = path.absolute()

    if as_file is True and not path.is_file():
        raise FileNotFoundError(f"File not found at: {path}")

    return path


def load_image_path(path: t.Union[str, pathlib.Path]) -> cv2.typing.MatLike:
    """Load image frame from path with opencv."""
    path = pathify(path, as_file=True)

    try:
        image = cv2.imread(str(path))
    except Exception as exc:
        raise ValueError(f"Error loading image from file at: {path}\n{exc}") from exc

    return image


def get_cap(camera: int = 0) -> cv2.VideoCapture:
    """Get video capture object."""
    cap = cv2.VideoCapture(camera)

    if not cap.isOpened():
        raise IOError("Error: Could not open camera.")

    return cap


def show_image_keypress(image: cv2.typing.MatLike, title: str = "Image Frame") -> None:
    """Show image with opencv."""
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_image_plot(image: cv2.typing.MatLike) -> None:
    """Show image with matplotlib."""
    sv.plot_image(image)


def get_valid_annotators() -> t.List[str]:
    """Get valid annotators."""
    return [x.__name__ for x in BaseAnnotator.__subclasses__()] + [sv.LabelAnnotator.__name__]


def get_desktop_image() -> cv2.typing.MatLike:
    """Get image from the desktop."""
    # Capture the screen using mss
    screenshot = mss.mss().grab(mss.mss().monitors[1])

    # Convert the screenshot to a numpy array format
    image = np.array(screenshot)

    # Convert the image from RGB (which pyautogui uses) to BGR (which OpenCV uses)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image


class Layer:
    """Layer for adding annotations to an image."""

    EMPTY_RESULTS_TEXT: t.ClassVar[bool] = True
    AANOTATORS: t.ClassVar[t.List[t.Union[BaseAnnotator, sv.LabelAnnotator]]] = None
    MODEL: t.ClassVar[BaseInferenceModel] = None
    CONFIDENCE: float = 0.5
    IOU_THRESHOLD: float = 0.5

    def __init__(
        self,
        model_id: str,
        api_key: t.Optional[str] = None,
        annotators: t.Iterable[t.Union[str, BaseAnnotator, sv.LabelAnnotator]] = ANNOTATORS,
    ):
        self.add_annotators(annotators)
        self.MODEL: BaseInferenceModel = get_roboflow_model(model_id=model_id, api_key=api_key)

    def annotate(
        self,
        image: cv2.typing.MatLike,
        show_keypress: bool = False,
    ) -> t.Optional[t.List[sv.Detections]]:
        """Add annotations to an image."""
        if not self.ANNOTATORS:
            raise ValueError("No annotators defined.")

        results = self.MODEL.infer(
            image=image, confidence=self.CONFIDENCE, iou_threshold=self.IOU_THRESHOLD
        )

        if results:
            detections: sv.Detections = sv.Detections.from_inference(results[0])
            for annotator in self.ANNOTATORS:
                image = annotator.annotate(scene=image, detections=detections)
        elif self.EMPTY_RESULTS_TEXT:
            image = sv.draw_text(
                scene=image,
                text="No results returned",
                # need to figure out how to center it
                text_anchor=sv.Point(x=0, y=0),
                text_color=sv.Color.RED,
                text_scale=1.0,
                text_thickness=2,
                text_padding=0,
                text_font=cv2.FONT_HERSHEY_SIMPLEX,
                background_color=sv.Color.BLACK,
            )

        if show_keypress:
            show_image_keypress(image)
        return image

    def annotate_from_path(self, path: t.Union[str, pathlib.Path], **kwargs) -> cv2.typing.MatLike:
        """Add annotations to an image from a path."""
        image = load_image_path(path)
        image = self.annotate(image=image, **kwargs)
        return image

    def annotate_from_url(self, url: str, **kwargs) -> cv2.typing.MatLike:
        """Add annotations to an image from a url."""
        # turn this into a function that checks the url and does exception
        # handling for bad url's/bad image's
        image = sv.load_image(url)
        image = self.annotate(image=image, **kwargs)
        return image

    def annotate_from_desktop(self, **kwargs) -> cv2.typing.MatLike:
        """Add annotations to an image from the desktop."""
        # image = sv.load_image_desktop()
        # supervison does not have a load_image_desktop function - we need to write our own function
        # that can get screenshots from the desktop
        # image = self.annotate(image=image, **kwargs)
        # return image

    def annotate_from_clipboard(self, **kwargs) -> cv2.typing.MatLike:
        """Add annotations to an image from the clipboard."""
        # need to write a function that gets a screenshot from the clipboard

    @staticmethod
    def get_desktop_image() -> cv2.typing.MatLike:
        """Get image from the desktop."""
        # need to write a function that gets a screenshot from the desktop

    def annotate_from_camera(self, camera: int = 0, show_keypress: bool = False, **kwargs) -> None:
        """Add annotations to an image from a camera."""
        cap = get_cap(camera)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = self.annotate(image=frame, **kwargs)
            # need custom show_image_keypress to go slower
            cv2.imshow("Image Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

    def add_annotators(
        self, annotators: t.Iterable[t.Union[str, BaseAnnotator, sv.LabelAnnotator]]
    ) -> None:
        """Add annotators to the layer."""
        for annotator in annotators:
            self.add_annotator(annotator)

    def add_annotator(self, annotator: t.Union[str, BaseAnnotator, sv.LabelAnnotator]) -> None:
        """Add an annotator to the layer."""
        annotator = self.get_annotator(annotator)
        if not isinstance(getattr(self, "ANNOTATORS", None), list):
            self.ANNOTATORS = []
        self.ANNOTATORS.append(annotator)

    @staticmethod
    def get_annotator(
        annotator: t.Union[str, BaseAnnotator, sv.LabelAnnotator],
    ) -> t.Union[BaseAnnotator, sv.LabelAnnotator]:
        """Check if annotator is valid."""
        if isinstance(annotator, str):
            if not hasattr(sv, annotator):
                raise ValueError(
                    f"Invalid annotator: {annotator}, valids: {get_valid_annotators()}"
                )

            annotator = getattr(sv, annotator)()

        if not isinstance(annotator, (BaseAnnotator, sv.LabelAnnotator)):
            raise ValueError(f"Invalid annotator: {annotator}, valids: {get_valid_annotators()}")

        return annotator


LAYER = Layer(model_id=MODEL_ID, api_key=API_KEY)
# LAYER.annotate_from_path(IMAGE_PATH, show_keypress=True)
LAYER.annotate_from_camera()
