import asyncio
import logging
import typing as t

import cv2

from roboflow_gap.custom_types import PathLike
from roboflow_gap.models import ImageData
from roboflow_gap.utils.dates import get_now
from roboflow_gap.utils.paths import pathify

from .gather_base import GatherBase

if t.TYPE_CHECKING:
    import pathlib


class GatherFromFile(GatherBase):
    def __init__(  # noqa: PLR0913
        self,
        path: t.Optional[PathLike],
        must_exist: bool = True,
        watch: bool = False,
        sleep: float = 0.5,
        max_seconds: t.Optional[float] = None,
    ):
        super().__init__()
        self.path: pathlib.Path = pathify(path=path, as_file=must_exist)
        self.path_last_mtime: t.Optional[float] = None
        self.watch: bool = watch
        self.sleep: float = sleep
        self.max_seconds: t.Optional[float] = max_seconds
        self.count: int = 0
        self.logger: logging.Logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )
        # things to think about later
        # how to handle CLI interface for this, and how to allow python programmers
        # to use the same CLI interface
        # that will use environment variables and/or prompting as necessary

    async def run(self) -> t.AsyncIterator[ImageData]:
        """An asynchronous generator yielding ImageData instances."""
        while True:
            date_started = get_now()
            original = None
            date_loaded = None
            error = None
            error_exc = None
            context = {
                "path": self.path,
                "exists": self.path.is_file(),
                "modified": False,
                "gatherer": self.__class__.__name__,
            }
            self.logger.debug("Starting gather context=%s", context)
            if context["exists"]:
                path_stat = self.path.stat()

                if self.path_last_mtime:
                    # skip if file has already been yielded but mtime has not changed
                    if path_stat.st_mtime == self.path_last_mtime:
                        self.logger.debug(
                            "Not loading image from file, already yielded context=%s", context
                        )
                        continue
                    context["modified"] = True

                self.path_last_mtime = path_stat.st_mtime
                self.logger.debug("Loading image from file context=%s", context)
                try:
                    original = cv2.imread(str(self.path))
                    date_loaded = get_now()
                    self.count += 1
                    self.logger.debug("Loaded image from file context=%s", context)
                except Exception as exc:
                    error = f"Not loading image from file, error={exc}, context={context}"
                    self.logger.exception(error)
                    date_loaded = get_now()
                    error_exc = exc
            else:
                error = f"Not loading image from file, file not found context={context}"
                self.logger.error(error)
                date_loaded = get_now()

            yield ImageData(
                original=original,
                date_started=date_started,
                date_loaded=date_loaded,
                context=context,
                error=error,
                error_exc=error_exc,
            )

            # Exit loop if not in watch mode or max_seconds exceeded
            if not self.watch:
                self.logger.debug("Ending gather due to watch=False context=%s", context)
                break

            if (
                isinstance(self.max_seconds, (int, float))
                and self.max_seconds > 0
                and (get_now() - date_started).total_seconds() > self.max_seconds
            ):
                self.logger.debug(
                    "Ending gather due to max_seconds=%s exceeded context=%s",
                    self.max_seconds,
                    context,
                )
                break

            await asyncio.sleep(self.sleep)
