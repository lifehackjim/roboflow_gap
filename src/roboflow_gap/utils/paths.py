import pathlib
import typing as t

from roboflow_gap.models.custom_types import PathLike


def pathify(
    path: PathLike, resolve: bool = True, absolute: bool = True, as_file: t.Optional[bool] = False
) -> pathlib.Path:
    """Get path object from string or path."""
    path = pathlib.Path(path)

    if resolve is True:
        path = path.resolve()

    if absolute is True:
        path = path.absolute()

    if as_file is True and not path.is_file():
        raise FileNotFoundError(f"File not found at: {path}")

    return path
