from edfio.edf import Bdf, Edf, read_bdf, read_edf
from edfio.edf_annotations import EdfAnnotation
from edfio.edf_header import AnonymizedDateError, Patient, Recording
from edfio.edf_signal import BdfSignal, EdfSignal
from edfio.lead_matching import (
    Anchor,
    InterchangeableGroup,
    LeadAnchor,
    LeadMatcher,
    LeadMatching,
    LeadMatchingError,
    LeadMetadata,
    LeadPair,
    LeadRef,
    match_leads,
)

try:
    from importlib.metadata import version

    __version__ = version("edfio")
except Exception:  # pragma: no cover
    __version__ = "0.0.0"


__all__ = [
    "Anchor",
    "AnonymizedDateError",
    "Bdf",
    "BdfSignal",
    "Edf",
    "EdfAnnotation",
    "EdfSignal",
    "InterchangeableGroup",
    "LeadAnchor",
    "LeadMatcher",
    "LeadMatching",
    "LeadMatchingError",
    "LeadMetadata",
    "LeadPair",
    "LeadRef",
    "Patient",
    "Recording",
    "match_leads",
    "read_bdf",
    "read_edf",
]
