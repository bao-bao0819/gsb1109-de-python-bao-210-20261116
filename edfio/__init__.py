from edfio.clock_sync import (
    ClockCorrectionSegment,
    ClockCorrectionTable,
    DeviceTimeWindow,
    ExperimentWindowMapping,
    SampleRange,
    SignalSampleRanges,
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
    "ClockCorrectionSegment",
    "ClockCorrectionTable",
    "DeviceTimeWindow",
    "Edf",
    "EdfAnnotation",
    "EdfSignal",
    "ExperimentWindowMapping",
    "Patient",
    "Recording",
    "SampleRange",
    "SignalSampleRanges",
    "read_bdf",
    "read_edf",
]
