from edfio.edf import Bdf, Edf, read_bdf, read_edf
from edfio.edf_annotations import EdfAnnotation
from edfio.edf_header import AnonymizedDateError, Patient, Recording
from edfio.edf_quantization import (
    DigitalRecovery,
    QuantizationRange,
    QuantizedView,
    RangePreview,
    SourceDeclaration,
    preview_requantization,
    recover_digital_codes,
    requantize,
)
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
    "DigitalRecovery",
    "Edf",
    "EdfAnnotation",
    "EdfSignal",
    "Patient",
    "QuantizationRange",
    "QuantizedView",
    "RangePreview",
    "Recording",
    "SourceDeclaration",
    "preview_requantization",
    "read_bdf",
    "read_edf",
    "recover_digital_codes",
    "requantize",
]
