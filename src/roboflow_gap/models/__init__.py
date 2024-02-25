import dataclasses
import datetime
import typing as t

import cv2


@dataclasses.dataclass
class BaseDataClass:
    """Base dataclass with a to_dict method."""

    def to_dict(self) -> t.Dict[str, t.Any]:
        """Return a dictionary representation of the dataclass."""
        return dataclasses.asdict(self)


@dataclasses.dataclass
class ImageData(BaseDataClass):
    """Dataclass for image data."""

    original: t.Optional[cv2.typing.MatLike] = dataclasses.field(default=None, repr=False)
    # Original image frame.

    processed: t.Optional[cv2.typing.MatLike] = dataclasses.field(default=None, repr=False)
    # Processed image frame.

    context: t.Optional[t.Any] = dataclasses.field(default=None)
    # Context of the original image frame.

    analysis: t.Dict[str, t.Any] = dataclasses.field(default_factory=dict, repr=False)
    # Analysis results.

    date_started: t.Optional[datetime.datetime] = dataclasses.field(default=None)
    # Datetime the original image started loading.

    date_original_loaded: t.Optional[datetime.datetime] = dataclasses.field(default=None)
    # Datetime the original image finished loading.

    date_analyze_done: t.Optional[datetime.datetime] = dataclasses.field(default=None)
    # Datetime the original image finished analyzing.

    date_process_done: t.Optional[datetime.datetime] = dataclasses.field(default=None)
    # Datetime the processed image was finished.

    error: t.Optional[str] = dataclasses.field(default=None)
    # Error message.

    error_exc: t.Optional[Exception] = dataclasses.field(default=None)
    # Exception instance.
