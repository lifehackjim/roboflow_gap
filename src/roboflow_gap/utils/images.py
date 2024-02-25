import cv2

from roboflow_gap.custom_types import PathLike


def load_image_path(path: PathLike) -> cv2.typing.MatLike:
    """Load image frame from path with opencv."""
    try:
        image = cv2.imread(str(path))
    except Exception as exc:
        raise ValueError(f"Error loading image from file at: {path}\n{exc}") from exc
    return image
