import warnings

from .loader import append_feed, delete_feed, overwrite_feed, list_feeds
from .schedule import Schedule
from .frequencies import (DepartureInstance, FrequencyConflict,
                          FrequencyExpansion, ServiceWindow,
                          detect_conflicts, expand_frequencies,
                          frequency_instances, service_windows)

try:
    from ._version import version as __version__
except ImportError:
    warnings.warn("pygtfs should be installed for the version to work")
    __version__ = "0"
