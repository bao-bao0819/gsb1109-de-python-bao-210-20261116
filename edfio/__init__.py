from edfio.bdf_to_edf import (
    BdfToEdfReport,
    ChannelConversionReport,
    ConversionResult,
    SaturatedSample,
    convert_bdf_to_edf,
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
    "BdfToEdfReport",
    "ChannelConversionReport",
    "ConversionResult",
    "Edf",
    "EdfAnnotation",
    "EdfSignal",
    "Patient",
    "Recording",
    "SaturatedSample",
    "convert_bdf_to_edf",
    "read_bdf",
    "read_edf",
]
