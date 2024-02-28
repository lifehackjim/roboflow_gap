import asyncio
import pathlib
import typing as t

import cv2
import pydantic

from roboflow_gap.models import ImageData
from roboflow_gap.utils.dates import get_now
from roboflow_gap.utils.paths import pathify
from roboflow_gap.utils.tools import listify

from .gather_base import GatherBase


class GatherFromFile(GatherBase):
    """Gather image data from a file."""

    paths: t.Union[pathlib.Path, t.List[pathlib.Path]] = pydantic.Field(
        description="The paths to gather images from.",
    )
    must_exist: bool = pydantic.Field(
        default=True,
        description="If True, each path must exist, otherwise an error will be raised.",
    )
    watch: bool = pydantic.Field(
        default=False,
        description="If True, the gatherer will watch paths for changes.",
    )
    sleep: float = pydantic.Field(
        default=0.5,
        description="The number of seconds to sleep between checking paths for changes.",
    )
    max_images: t.Optional[int] = pydantic.Field(
        default=None,
        description="The maximum number of images to gather before stopping.",
    )
    max_seconds: t.Optional[float] = pydantic.Field(
        default=None,
        description="The maximum number of seconds to gather before stopping.",
    )

    _count: int = pydantic.PrivateAttr(default=0)
    _paths_mtime: t.Dict[pathlib.Path, float] = pydantic.PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: t.Any) -> None:
        """Post init model method."""
        self.paths: t.List[pathlib.Path] = [
            pathify(path=x, as_file=self.must_exist) for x in listify(self.paths)
        ]

    async def run(self) -> t.AsyncIterator[ImageData]:
        """An asynchronous generator yielding ImageData instances."""
        for path in self.paths:
            async for image_data in self.load_path(path=path):
                yield image_data

    async def load_path(self, path: pathlib.Path) -> t.AsyncGenerator[ImageData, None]:
        """Load image from path."""
        while True:
            date_started = get_now()
            original = None
            date_loaded = None
            error = None
            error_exc = None
            path_exists: bool = path.is_file()
            path_mtime: t.Optional[float] = path.stat().st_mtime if path_exists else None

            context = {
                "path": path,
                "exists": path_exists,
                "modified": None,
                "gatherer": self.__class__.__name__,
            }
            self._logger.debug("Starting gather context=%s", context)
            if context["exists"]:
                path_stat = path.stat()

                if context["modified"]:
                    if path_stat.st_mtime == context["modified"]:
                        self._logger.debug(
                            "Not loading image from file, modification time unchanged since previous gather context=%s",
                            context,
                        )
                        continue
                    context["modified"] = path_stat.st_mtime

                self.path_last_mtime = path_stat.st_mtime
                self._logger.debug("Loading image from file context=%s", context)
                try:
                    original = cv2.imread(str(self.path))
                    date_loaded = get_now()
                    self.count += 1
                    self._logger.debug("Loaded image from file context=%s", context)
                except Exception as exc:
                    error = f"Not loading image from file, error={exc}, context={context}"
                    self._logger.exception(error)
                    error_exc = exc
            else:
                error = f"Not loading image from file, file not found context={context}"
                self._logger.error(error)

            yield ImageData(
                original=original,
                date_started=date_started,
                date_loaded=date_loaded or get_now(),
                context=context,
                error=error,
                error_exc=error_exc,
            )

            if not self.watch:
                self._logger.debug("Ending gather due to watch=False context=%s", context)
                break

            if (
                isinstance(self.max_seconds, (int, float))
                and self.max_seconds > 0
                and (get_now() - date_started).total_seconds() > self.max_seconds
            ):
                self._logger.debug(
                    "Ending gather due to max_seconds=%s exceeded context=%s",
                    self.max_seconds,
                    context,
                )
                break

            await asyncio.sleep(self.sleep)
