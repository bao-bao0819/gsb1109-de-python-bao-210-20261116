import warnings

from .loader import append_feed, delete_feed, overwrite_feed, list_feeds
from .schedule import Schedule
from .interpolation import (estimate_stop_times, interpolate_trip,
                            interpolate_schedule, TimeEstimate,
                            EXACT, PROVIDED, INTERPOLATED, UNKNOWN)

try:
    from ._version import version as __version__
except ImportError:
    warnings.warn("pygtfs should be installed for the version to work")
    __version__ = "0"
