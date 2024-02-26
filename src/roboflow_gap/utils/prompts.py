import pathlib
import typing as t

from roboflow_gap.models.custom_types import PathLike

from .paths import pathify


def prompt_for_files(*paths: PathLike) -> t.List[pathlib.Path]:
    """Prompt user to select files."""
    paths = list(paths)
    while True:
        path = input("Enter a file path or press enter to continue: ").strip()
        if path:
            path = pathify(path)
            if path in paths:
                print("File path already added: {path}")
                continue
            elif path.is_file():
                paths.append(path)
                continue
            else:
                print("File path not found at: {path}")
                continue
        elif paths:
            break
        else:
            print("At least one file path must be supplied.")
            continue
    return paths
