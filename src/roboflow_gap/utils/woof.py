import inspect
import typing as t


def get_subcls(cls: type) -> t.List[type]:
    """Recursively get all subclasses of a class."""
    subs = set(cls.__subclasses__())
    subs.update([s for c in subs for s in get_subcls(c)])
    return list(subs)


def get_method_args(cls: type, method: str) -> t.Dict[str, t.Optional[t.Union[str, t.List[str]]]]:
    actual_method = getattr(cls, method, None)
    args = []
    doc = None
    path = None
    if actual_method:
        args = inspect.getfullargspec(actual_method).args
        doc = actual_method.__doc__
        path = f"{cls.__module__}.{cls.__name__}.{method}"
    return {"cls": cls, "args": args, "doc": doc, "path": path}


def get_subcls_method_args(
    cls: type, method: str
) -> t.List[t.Dict[str, t.Optional[t.Union[str, t.List[str]]]]]:
    """Get all subclasses of a class and the arguments of a method for each subclass."""
    return [get_method_args(x, method) for x in get_subcls(cls)]


# BUG: Some of the Model subclasses do not have task_type as a class attribute,
# it only gets set in the __init__ method
"""
import inference.core.models.base
subclasses = get_subcls_method_args(cls=inference.core.models.base.Model, method="infer")
for subclass in subclasses:
    subclass['task_type'] = getattr(subclass['cls'], 'task_type', None)
    print(subclass)
"""
