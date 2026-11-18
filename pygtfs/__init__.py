import warnings

from .loader import append_feed, delete_feed, overwrite_feed, list_feeds
from .schedule import Schedule
from .transfer import (SCOPE_STOP, SCOPE_ROUTE, SCOPE_TRIP,
                       STOP_SOURCE_STOP, STOP_SOURCE_PARENT,
                       TRANSFER_FORBIDDEN, TransferCandidate,
                       transfer_candidates, preferred_transfer,
                       walking_time_seconds, same_named_stops, rule_scope)

try:
    from ._version import version as __version__
except ImportError:
    warnings.warn("pygtfs should be installed for the version to work")
    __version__ = "0"
