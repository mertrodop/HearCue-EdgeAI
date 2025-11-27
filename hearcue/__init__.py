"""HearCue Python package: edge AI sound awareness simulation."""

from importlib.metadata import version, PackageNotFoundError

__all__ = [
    "get_version",
]


def get_version() -> str:
    """Return package version if installed as distribution."""
    try:
        return version("hearcue")
    except PackageNotFoundError:
        return "0.0.0"
