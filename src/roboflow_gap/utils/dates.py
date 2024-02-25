import datetime


def get_now() -> datetime.datetime:
    """Get the current datetime."""
    return datetime.datetime.utcnow()
