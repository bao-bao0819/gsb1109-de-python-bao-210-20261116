from edfio.derived import (
    ChannelDerivationReport,
    DerivationRegistry,
    DerivationReport,
    DerivationResult,
    DerivedChannel,
    DerivedChannelSpec,
    UpdateReport,
    compute_differential_channel,
    derive_differential_channels,
    get_derived_channel,
    normalize_bad_segments,
    set_bad_segments,
)
from edfio.edf import Bdf, Edf, read_bdf, read_edf
from edfio.edf_annotations import EdfAnnotation
from edfio.edf_header import AnonymizedDateError, Patient, Recording
from edfio.edf_signal import BadSegment, BdfSignal, EdfSignal

try:
    from importlib.metadata import version

    __version__ = version("edfio")
except Exception:  # pragma: no cover
    __version__ = "0.0.0"


__all__ = [
    "AnonymizedDateError",
    "BadSegment",
    "Bdf",
    "BdfSignal",
    "ChannelDerivationReport",
    "DerivationRegistry",
    "DerivationReport",
    "DerivationResult",
    "DerivedChannel",
    "DerivedChannelSpec",
    "Edf",
    "EdfAnnotation",
    "EdfSignal",
    "Patient",
    "Recording",
    "UpdateReport",
    "compute_differential_channel",
    "derive_differential_channels",
    "get_derived_channel",
    "normalize_bad_segments",
    "read_bdf",
    "read_edf",
    "set_bad_segments",
]
