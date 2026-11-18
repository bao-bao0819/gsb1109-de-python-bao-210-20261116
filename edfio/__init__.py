from edfio.concatenation import (
    ChannelConflict,
    ChannelMetadata,
    ConcatenationError,
    ConcatenationResult,
    ConcatenationSegment,
    ConversionRecord,
    UnitConversion,
    concatenate,
    concatenate_edfs,
    concatenate_recordings,
    extract_reference,
    plan_concatenation,
)
from edfio.edf import Bdf, Edf, read_bdf, read_edf
from edfio.edf_annotations import EdfAnnotation
from edfio.edf_header import AnonymizedDateError, Patient, Recording
from edfio.edf_signal import BdfSignal, EdfSignal

try:
    from importlib.metadata import version

    __version__ = version("edfio")
except Exception:  # pragma: no cover
    __version__ = "0.0.0"


__all__ = [
    "AnonymizedDateError",
    "Bdf",
    "BdfSignal",
    "ChannelConflict",
    "ChannelMetadata",
    "ConcatenationError",
    "ConcatenationResult",
    "ConcatenationSegment",
    "ConversionRecord",
    "Edf",
    "EdfAnnotation",
    "EdfSignal",
    "Patient",
    "Recording",
    "UnitConversion",
    "concatenate",
    "concatenate_edfs",
    "concatenate_recordings",
    "extract_reference",
    "plan_concatenation",
    "read_bdf",
    "read_edf",
]
