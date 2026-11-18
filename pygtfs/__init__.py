import warnings

from .loader import append_feed, delete_feed, overwrite_feed, list_feeds
from .schedule import Schedule
from .identity import (add_alias, add_aliases, add_aliases_from_file,
                       get_identity, identities_for_key, record_lineage,
                       lineage_chain, register_all_identities)

try:
    from ._version import version as __version__
except ImportError:
    warnings.warn("pygtfs should be installed for the version to work")
    __version__ = "0"
