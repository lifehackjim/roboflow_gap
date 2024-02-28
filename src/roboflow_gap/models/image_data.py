import datetime
import typing as t

import cv2
from pydantic import BaseModel, Field


class ImageData(BaseModel):
    """Dataclass for image data."""

    original: t.Optional[cv2.typing.MatLike] = Field(
        default=None,
        description="Original image frame.",
    )
    processed: t.Optional[cv2.typing.MatLike] = Field(
        default=None,
        description="Processed image frame.",
    )
    context: t.Optional[t.Dict[str, t.Any]] = Field(
        default_factory=dict,
        description="Context of the original image frame.",
    )
    analysis: t.Optional[t.Dict[str, t.Any]] = Field(
        default_factory=dict,
        description="Analysis results.",
    )
    date_started: t.Optional[datetime.datetime] = Field(
        default=None,
        description="Datetime the original image started loading.",
    )
    date_original_loaded: t.Optional[datetime.datetime] = Field(
        default=None,
        description="Datetime the original image finished loading.",
    )
    date_analyze_done: t.Optional[datetime.datetime] = Field(
        default=None,
        description="Datetime the original image finished analyzing.",
    )
    date_process_done: t.Optional[datetime.datetime] = Field(
        default=None,
        description="Datetime the processed image was finished.",
    )
    error: t.Optional[str] = Field(
        default=None,
        description="Error message.",
    )
    error_exc: t.Optional[Exception] = Field(
        default=None,
        description="Exception instance.",
    )
