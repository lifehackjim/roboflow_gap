import abc
import datetime
import logging
import typing as t

import pydantic

from roboflow_gap.models import ImageData
from roboflow_gap.utils.dates import get_now


# switch to pydantic.Settings
class GatherBase(abc.ABC, pydantic.BaseModel):
    """Base class for image data gathering."""

    _logger: t.Optional[logging.Logger] = pydantic.PrivateAttr(default=None)
    _date_created: datetime.datetime = pydantic.PrivateAttr(default_factory=get_now)

    @abc.abstractmethod
    async def run(self) -> t.AsyncIterator[ImageData]:
        """An asynchronous generator yielding ImageData instances."""
        yield NotImplemented

    @pydantic.validator("_logger", pre=True)
    def set_logger(cls, value: t.Optional[logging.Logger]) -> logging.Logger:
        """Set logger if not provided."""
        return value or logging.getLogger(f"{cls.__module__}.{cls.__name__}")
