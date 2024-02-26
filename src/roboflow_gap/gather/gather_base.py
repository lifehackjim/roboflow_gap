import abc
import typing as t

from roboflow_gap.models import ImageData


class GatherBase(abc.ABC):
    """Base class for image data gathering."""

    @abc.abstractmethod
    async def run(self) -> t.AsyncIterator[ImageData]:
        """An asynchronous generator yielding ImageData instances."""
        yield NotImplemented
