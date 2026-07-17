"""Euclidean distance transform in PyTorch."""

from . import backend  # noqa: F401
from .l1 import *  # noqa: F401, F403
from .l2 import *  # noqa: F401, F403

try:
    from ._version import __version__
except ImportError:  # pragma: no cover
    __version__ = "0+unknown"
