"""Concatenate EDF/BDF recordings along the time axis.

Channels are aligned by metadata rather than by label alone. Two channels with
the same label are only concatenated when their sampling frequency, reference
scheme, and (optionally) prefiltering are compatible, and when their physical
units are identical or can be converted with a known rule.

Recordings that cannot be concatenated are returned as separate outputs instead
of causing the whole operation to fail. Unit conversions are recorded in
:class:`ConcatenationResult` and are also stored as EDF+ annotations by default,
so the conversion basis remains available in the written file.
"""

from __future__ import annotations

import decimal
import math
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Union

import numpy as np
import numpy.typing as npt

from edfio.edf import Bdf, Edf, read_bdf, read_edf
from edfio.edf_annotations import EdfAnnotation
from edfio.edf_signal import BdfSignal, EdfSignal

_Recording = Union[Edf, Bdf]
_Signal = Union[EdfSignal, BdfSignal]
_ChannelPolicy = Literal["strict", "intersection"]
_OnIncompatible = Literal["split", "error"]
_ConversionInput = Union[
    "UnitConversion",
    Mapping[Any, Any],
    Iterable[Any],
    None,
]
_ConflictReason = Literal[
    "missing_channel",
    "extra_channel",
    "duplicate_channel",
    "reference",
    "unit",
    "sampling_frequency",
    "prefiltering",
    "format",
    "no_shared_channels",
]


class ConcatenationError(ValueError):
    """Raised when recordings cannot be concatenated under the chosen policy."""


@dataclass(frozen=True)
class UnitConversion:
    """A linear conversion from one physical unit to another.

    The conversion is applied as ``target = source * factor + offset``.

    Parameters
    ----------
    source : str
        The source physical dimension, e.g., ``"mV"``.
    target : str
        The target physical dimension, e.g., ``"uV"``.
    factor : float, default: 1
        The multiplicative factor.
    offset : float, default: 0
        The additive offset.
    description : str, default: ""
        Human-readable description of the basis for this conversion, e.g., the
        standard or calibration document it was taken from.
    """

    source: str
    target: str
    factor: float = 1.0
    offset: float = 0.0
    description: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.factor) or self.factor == 0:
            raise ValueError(
                f"Conversion factor must be finite and non-zero, got {self.factor}"
            )
        if not math.isfinite(self.offset):
            raise ValueError(f"Conversion offset must be finite, got {self.offset}")

    def apply(self, data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Convert physical values from `source` to `target`."""
        return data * self.factor + self.offset

    def inverted(self) -> UnitConversion:
        """Return the inverse conversion."""
        return UnitConversion(
            source=self.target,
            target=self.source,
            factor=1.0 / self.factor,
            offset=-self.offset / self.factor,
            description=self.description,
        )

    @property
    def is_identity(self) -> bool:
        """Whether this conversion leaves values unchanged."""
        return (
            _canonical_unit(self.source) == _canonical_unit(self.target)
            and self.factor == 1
            and self.offset == 0
        )


@dataclass(frozen=True)
class ChannelMetadata:
    """Metadata used to decide whether two channels can be concatenated.

    Parameters
    ----------
    label : str
        The signal label from the EDF/BDF header.
    unit : str, default: ""
        The physical dimension (unit), e.g., ``"uV"``.
    reference : str, default: ""
        The reference electrode or reference scheme. An empty string means that
        the reference is unspecified in the available metadata.
    sampling_frequency : float, default: 0
        The sampling frequency in Hz.
    prefiltering : str, default: ""
        The prefiltering description from the signal header.
    transducer_type : str, default: ""
        The transducer type from the signal header.
    """

    label: str
    unit: str = ""
    reference: str = ""
    sampling_frequency: float = 0.0
    prefiltering: str = ""
    transducer_type: str = ""

    @classmethod
    def from_signal(
        cls,
        signal: _Signal,
        *,
        unit: str | None = None,
        reference: str | None = None,
    ) -> ChannelMetadata:
        """Extract metadata from an EDF/BDF signal."""
        return cls(
            label=signal.label,
            unit=signal.physical_dimension if unit is None else unit,
            reference=extract_reference(signal) if reference is None else reference,
            sampling_frequency=signal.sampling_frequency,
            prefiltering=signal.prefiltering,
            transducer_type=signal.transducer_type,
        )


@dataclass(frozen=True)
class ConversionRecord:
    """The basis for converting one channel of one source recording."""

    output_index: int
    recording_index: int
    channel: str
    source_unit: str
    target_unit: str
    factor: float
    offset: float
    rule: str = ""

    @property
    def description(self) -> str:
        """Compact human-readable description of the conversion."""
        text = (
            f"{self.channel}: {self.source_unit or '<unspecified>'} -> "
            f"{self.target_unit or '<unspecified>'} "
            f"(factor={self.factor:g}, offset={self.offset:g})"
        )
        if self.rule:
            text += f" [{self.rule}]"
        return text


@dataclass(frozen=True)
class ChannelConflict:
    """A reason why a channel or recording was not concatenated."""

    output_index: int
    recording_index: int
    channel: str
    reason: _ConflictReason
    detail: str

    @property
    def description(self) -> str:
        """Compact human-readable description of the conflict."""
        return f"{self.channel}: {self.reason} ({self.detail})"


@dataclass(frozen=True)
class ConcatenationSegment:
    """The position of a source recording in a concatenated output."""

    output_index: int
    recording_index: int
    start_second: float
    duration: float


@dataclass(frozen=True)
class ConcatenationResult(Sequence[_Recording]):
    """The result of :func:`concatenate_recordings`.

    The object behaves like the tuple of output recordings (iteration, indexing,
    and length), while also exposing the conversion basis and the reasons for
    splitting recordings into separate outputs.
    """

    recordings: tuple[_Recording, ...]
    segments: tuple[ConcatenationSegment, ...] = ()
    conversions: tuple[ConversionRecord, ...] = ()
    conflicts: tuple[ChannelConflict, ...] = ()

    def __iter__(self) -> Iterator[_Recording]:
        return iter(self.recordings)

    def __len__(self) -> int:
        return len(self.recordings)

    def __getitem__(self, index: int) -> _Recording:  # type: ignore[override]
        return self.recordings[index]

    @property
    def outputs(self) -> tuple[_Recording, ...]:
        """Alias for :attr:`recordings`."""
        return self.recordings

    @property
    def conversion_log(self) -> tuple[ConversionRecord, ...]:
        """Alias for :attr:`conversions`."""
        return self.conversions

    @property
    def runs(self) -> tuple[tuple[int, ...], ...]:
        """Source recording indices grouped by output.

        Each tuple lists the recordings that were concatenated into the output
        at the corresponding index. This makes the concatenable ranges explicit,
        in particular when recordings had to be split into multiple outputs.
        """
        output_count = max(
            (segment.output_index for segment in self.segments),
            default=-1,
        )
        grouped: list[list[int]] = [[] for _ in range(output_count + 1)]
        for segment in self.segments:
            grouped[segment.output_index].append(segment.recording_index)
        return tuple(tuple(indices) for indices in grouped)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ConcatenationResult):
            return (
                self.recordings == other.recordings
                and self.segments == other.segments
                and self.conversions == other.conversions
                and self.conflicts == other.conflicts
            )
        if isinstance(other, Sequence) and not isinstance(other, (str, bytes)):
            return tuple(self.recordings) == tuple(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.recordings)

    def write(
        self,
        directory: Path | str,
        *,
        stem: str = "concatenated",
        suffix: str | None = None,
    ) -> tuple[Path, ...]:
        """Write all output recordings to `directory`.

        Files are named ``{stem}.{suffix}`` for a single output and
        ``{stem}-{index}.{suffix}`` when the result contains multiple outputs.
        The suffix defaults to ``edf`` or ``bdf``, depending on the format.
        """
        if not self.recordings:
            raise ValueError("Cannot write an empty concatenation plan")
        target_dir = Path(directory).expanduser()
        target_dir.mkdir(parents=True, exist_ok=True)
        if suffix is None:
            suffix = "bdf" if isinstance(self.recordings[0], Bdf) else "edf"
        suffix = suffix.lstrip(".")
        paths: list[Path] = []
        multiple = len(self.recordings) > 1
        for index, recording in enumerate(self.recordings):
            name = f"{stem}-{index}.{suffix}" if multiple else f"{stem}.{suffix}"
            path = target_dir / name
            recording.write(path)
            paths.append(path)
        return tuple(paths)

    def __repr__(self) -> str:
        return (
            f"<ConcatenationResult {len(self.recordings)} recording(s), "
            f"{len(self.conversions)} conversion(s), "
            f"{len(self.conflicts)} conflict(s)>"
        )


@dataclass
class _MergedChannel:
    label: str
    target_unit: str
    reference: str
    sampling_frequency: float
    prefiltering: str
    signals: list[_Signal] = field(default_factory=list)
    conversions: list[UnitConversion | None] = field(default_factory=list)
    use_digital: bool = True

    @property
    def first(self) -> _Signal:
        return self.signals[0]


@dataclass
class _Group:
    output_index: int
    first_recording: _Recording
    channels: dict[str, _MergedChannel] = field(default_factory=dict)
    annotations: list[EdfAnnotation] = field(default_factory=list)
    conflicts: list[ChannelConflict] = field(default_factory=list)
    segments: list[tuple[int, float, float]] = field(default_factory=list)
    offset: float = 0.0


def concatenate(
    recordings: _Recording | Iterable[_Recording | Path | str | bytes],
    *additional: _Recording | Path | str | bytes,
    unit_conversions: _ConversionInput = None,
    conversions: _ConversionInput = None,
    reference_equivalences: Any = None,
    target_units: Mapping[str, str] | None = None,
    channel_metadata: Mapping[Any, Any] | None = None,
    channel_policy: _ChannelPolicy = "strict",
    on_incompatible: _OnIncompatible = "split",
    annotate_conversions: bool = True,
    require_prefiltering_match: bool = True,
    **kwargs: Any,
) -> ConcatenationResult:
    """Concatenate recordings along the time axis.

    Channels are matched by label, but a matching label alone is not sufficient:
    sampling frequency, reference scheme, prefiltering, and units must also be
    compatible. Units that can be converted with a known rule are converted
    instead of being rejected.

    Parameters
    ----------
    recordings : Edf | Bdf | Iterable[Edf | Bdf | Path | str | bytes]
        The recordings to concatenate, in chronological order. A single
        recording or an iterable of recordings can be passed. Paths are read
        with :func:`edfio.read_edf` or :func:`edfio.read_bdf`, depending on
        their suffix.
    *additional : Edf | Bdf | Path | str | bytes
        Additional recordings, so that recordings can also be passed as
        separate positional arguments.
    unit_conversions : UnitConversion | Mapping | Iterable | None, default: None
        Explicit conversion conventions. Accepted forms are

        - an iterable of :class:`UnitConversion` objects,
        - an iterable of ``(source, target, factor[, offset])`` tuples,
        - a mapping from ``(source, target)`` to a factor or to a
          ``(factor, offset)`` tuple,
        - a nested mapping ``{source: {target: factor}}``, or
        - a scale table ``{unit: scale_to_base_unit}``.

        Built-in conversions between common metric units (for example ``uV``
        and ``mV``) are always available. Explicit rules take precedence.
    conversions : UnitConversion | Mapping | Iterable | None, default: None
        Alias for `unit_conversions`. If both are given, the rules are merged.
    reference_equivalences : Iterable | Mapping | None, default: None
        Reference schemes that should be treated as equivalent. Accepts pairs
        or groups of references, for example ``[("M1", "A1")]`` or
        ``{"linked": ("M1", "M2")}``. References are compared
        case-insensitively after common aliases have been normalized. Purely
        identical names are always equivalent; different electrode names (for
        example ``"M1"`` and ``"A1"``) are only treated as equivalent when
        declared here.
    target_units : Mapping[str, str] | None, default: None
        Optional mapping from channel label to the unit of the concatenated
        output channel. By default, the unit of the first recording containing
        the channel is used. Explicit target units are applied to every output,
        including outputs created because a recording had to be split off.
    channel_metadata : Mapping | None, default: None
        Optional metadata overrides. This is useful for files whose reference
        electrode is not encoded in a standard header field. Supported keys are
        ``"unit"``, ``"reference"``, ``"sampling_frequency"``,
        ``"prefiltering"``, and ``"transducer_type"``.

        Overrides can be keyed by channel label, or by recording index and then
        by channel label (``{0: {"EEG1": {...}}, 1: {"EEG1": {...}}}``). Label
        matching is case-insensitive.
    channel_policy : {"strict", "intersection"}, default: "strict"
        How to handle recordings with different channel sets.

        - ``"strict"``: all recordings in one output must have the same
          channels. Otherwise, a new output is started.
        - ``"intersection"``: only channels present in all recordings of an
          output are kept. Channels that are missing or incompatible are
          excluded and reported in :attr:`ConcatenationResult.conflicts`.
    on_incompatible : {"split", "error"}, default: "split"
        What to do when a recording cannot be appended to the current output.
        ``"split"`` returns separate outputs, ``"error"`` raises
        :class:`ConcatenationError`.
    annotate_conversions : bool, default: True
        Whether to add EDF+ annotations describing each unit conversion at the
        start of the affected segment. The same information is always available
        in :attr:`ConcatenationResult.conversions`.
    require_prefiltering_match : bool, default: True
        Whether differing prefiltering descriptions prevent concatenation.

    Returns
    -------
    ConcatenationResult
        The concatenated recordings, conversion records, and conflicts. The
        result can be used like a tuple of output recordings.

    Raises
    ------
    ConcatenationError
        If `on_incompatible` is ``"error"`` and recordings are incompatible,
        or if the inputs are invalid.
    ValueError
        If no recordings are provided or EDF and BDF recordings are mixed.
    """
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {unknown}")
    rules = _merge_conversion_rules(unit_conversions, conversions)
    sources = _normalize_recordings(recordings, additional)
    _validate_sources(sources)
    groups = _plan_groups(
        sources,
        rules=rules,
        target_units=target_units,
        channel_metadata=channel_metadata,
        reference_equivalences=reference_equivalences,
        channel_policy=channel_policy,
        on_incompatible=on_incompatible,
        require_prefiltering_match=require_prefiltering_match,
    )

    outputs: list[_Recording] = []
    all_conversions: list[ConversionRecord] = []
    all_conflicts: list[ChannelConflict] = []
    all_segments: list[ConcatenationSegment] = []
    for output_index, group in enumerate(groups):
        group.output_index = output_index
        output, conversions = _build_output(
            group, annotate_conversions=annotate_conversions
        )
        outputs.append(output)
        all_conversions.extend(conversions)
        all_conflicts.extend(
            ChannelConflict(
                output_index=output_index,
                recording_index=conflict.recording_index,
                channel=conflict.channel,
                reason=conflict.reason,
                detail=conflict.detail,
            )
            for conflict in group.conflicts
        )
        all_segments.extend(
            ConcatenationSegment(
                output_index=output_index,
                recording_index=recording_index,
                start_second=start,
                duration=duration,
            )
            for recording_index, start, duration in group.segments
        )

    return ConcatenationResult(
        recordings=tuple(outputs),
        segments=tuple(all_segments),
        conversions=tuple(all_conversions),
        conflicts=tuple(all_conflicts),
    )


def concatenate_recordings(
    recordings: _Recording | Iterable[_Recording | Path | str | bytes],
    *additional: _Recording | Path | str | bytes,
    **kwargs: Any,
) -> ConcatenationResult:
    """Alias for :func:`concatenate`."""
    return concatenate(recordings, *additional, **kwargs)


def concatenate_edfs(
    recordings: _Recording | Iterable[_Recording | Path | str | bytes],
    *additional: _Recording | Path | str | bytes,
    **kwargs: Any,
) -> ConcatenationResult:
    """Alias for :func:`concatenate` for EDF recordings."""
    return concatenate(recordings, *additional, **kwargs)


def plan_concatenation(
    recordings: _Recording | Iterable[_Recording | Path | str | bytes],
    *additional: _Recording | Path | str | bytes,
    unit_conversions: _ConversionInput = None,
    conversions: _ConversionInput = None,
    reference_equivalences: Any = None,
    target_units: Mapping[str, str] | None = None,
    channel_metadata: Mapping[Any, Any] | None = None,
    channel_policy: _ChannelPolicy = "strict",
    on_incompatible: _OnIncompatible = "split",
    require_prefiltering_match: bool = True,
) -> ConcatenationResult:
    """Determine the concatenable ranges without building output recordings.

    This performs the same metadata checks and unit-conversion planning as
    :func:`concatenate`, but returns a :class:`ConcatenationResult` whose
    ``recordings`` tuple is empty. Use :attr:`ConcatenationResult.runs` for the
    source recordings of each output, :attr:`ConcatenationResult.conversions`
    for the conversion basis, and :attr:`ConcatenationResult.conflicts` for the
    reasons recordings were split. No data is loaded or copied.

    Parameters are the same as for :func:`concatenate`, except for
    `annotate_conversions`, which is not applicable here.
    """
    rules = _merge_conversion_rules(unit_conversions, conversions)
    sources = _normalize_recordings(recordings, additional)
    _validate_sources(sources)
    groups = _plan_groups(
        sources,
        rules=rules,
        target_units=target_units,
        channel_metadata=channel_metadata,
        reference_equivalences=reference_equivalences,
        channel_policy=channel_policy,
        on_incompatible=on_incompatible,
        require_prefiltering_match=require_prefiltering_match,
    )

    segments: list[ConcatenationSegment] = []
    conversions_out: list[ConversionRecord] = []
    conflicts: list[ChannelConflict] = []
    for output_index, group in enumerate(groups):
        for recording_index, start, duration in group.segments:
            segments.append(
                ConcatenationSegment(
                    output_index=output_index,
                    recording_index=recording_index,
                    start_second=start,
                    duration=duration,
                )
            )
        for label, channel in group.channels.items():
            for segment_index, (recording_index, _, _) in enumerate(group.segments):
                rule = channel.conversions[segment_index]
                if rule is None:
                    continue
                conversions_out.append(
                    ConversionRecord(
                        output_index=output_index,
                        recording_index=recording_index,
                        channel=label,
                        source_unit=channel.signals[segment_index].physical_dimension,
                        target_unit=channel.target_unit,
                        factor=rule.factor,
                        offset=rule.offset,
                        rule=rule.description,
                    )
                )
        conflicts.extend(
            ChannelConflict(
                output_index=output_index,
                recording_index=conflict.recording_index,
                channel=conflict.channel,
                reason=conflict.reason,
                detail=conflict.detail,
            )
            for conflict in group.conflicts
        )

    return ConcatenationResult(
        recordings=(),
        segments=tuple(segments),
        conversions=tuple(conversions_out),
        conflicts=tuple(conflicts),
    )


def extract_reference(signal: _Signal) -> str:
    """Extract a reference electrode description from a signal's metadata.

    References are recognized in a transducer type or prefiltering field that
    contains an explicit ``ref=...``/``reference: ...`` marker, in a label with
    an explicit reference marker, and in bipolar labels such as
    ``"EEG Fpz-Cz"``. If a signal exposes a ``reference`` attribute, that value
    takes precedence.
    """
    explicit = getattr(signal, "reference", None)
    if isinstance(explicit, str):
        return explicit.strip()
    for text_field in (signal.transducer_type, signal.prefiltering):
        reference = _reference_from_text(text_field)
        if reference:
            return reference
    return _reference_from_label(signal.label)


def _reference_from_text(text: str) -> str:
    if not text:
        return ""
    pattern = re.compile(
        r"(?:^|[;,\s])(?:ref(?:erence)?(?:\s*electrode)?)\s*[:=]?\s*([^;,]+)",
        re.IGNORECASE,
    )
    match = pattern.search(text)
    if match is None:
        pattern = re.compile(r"(?:^|[;,\s])(?:against|vs\.?)\s+([^;,]+)", re.IGNORECASE)
        match = pattern.search(text)
    if match is None:
        return ""
    return match.group(1).strip()


def _reference_from_label(label: str) -> str:
    explicit = re.search(
        r"[\[(]\s*(?:ref(?:erence)?)\s*[:=]\s*([^\])]+)[\])]",
        label,
        re.IGNORECASE,
    )
    if explicit is not None:
        return explicit.group(1).strip()
    if "-" in label:
        prefix, suffix = label.rsplit("-", 1)
        if prefix.strip() and _is_electrode_reference(prefix.strip(), suffix.strip()):
            return suffix.strip()
    return ""


_ELECTRODE_MODALITIES = (
    "eeg",
    "eog",
    "emg",
    "ecg",
    "ekg",
    "meg",
    "ieeg",
    "seeg",
    "lfp",
)

_ELECTRODE_NAME = re.compile(r"[A-Za-z]{1,3}\d{0,2}z?", re.IGNORECASE)


def _is_electrode_reference(prefix: str, suffix: str) -> bool:
    if _canonical_reference(suffix) in _ELECTRODE_REFERENCE_NAMES:
        return True
    lowered_prefix = prefix.casefold()
    if not any(
        re.search(rf"(?:^|[\s_]){modality}(?:$|[\s_])", lowered_prefix)
        for modality in _ELECTRODE_MODALITIES
    ):
        return False
    return _ELECTRODE_NAME.fullmatch(suffix) is not None


def _normalize_recordings(
    recordings: _Recording | Iterable[_Recording | Path | str | bytes],
    additional: Sequence[_Recording | Path | str | bytes],
) -> list[_Recording]:
    if isinstance(recordings, (Edf, Bdf)):
        items: list[Any] = [recordings, *additional]
    else:
        if additional:
            raise ValueError(
                "Additional recordings can only be passed together with a single recording"
            )
        items = list(recordings)
    return [_read_recording(item) for item in items]


def _read_recording(item: Any) -> _Recording:
    if isinstance(item, (Edf, Bdf)):
        return item
    if isinstance(item, Path):
        return _read_path(item)
    if isinstance(item, str):
        return _read_path(Path(item))
    if isinstance(item, (bytes, bytearray, memoryview)):
        data = bytes(item)
        if data[:8] == b"\xffBIOSEMI":
            return read_bdf(data)
        return read_edf(data)
    raise TypeError(f"Unsupported recording type: {type(item).__name__}")


def _read_path(path: Path) -> _Recording:
    if path.suffix.lower() == ".bdf":
        return read_bdf(path)
    return read_edf(path)


def _recording_metadata(
    recording: _Recording,
    recording_index: int,
    channel_metadata: Mapping[Any, Any] | None,
) -> dict[str, ChannelMetadata]:
    metadata: dict[str, ChannelMetadata] = {}
    for signal in recording.signals:
        override = _metadata_override(channel_metadata, recording_index, signal.label)
        metadata[signal.label] = _signal_metadata(signal, override)
    return metadata


def _metadata_override(
    channel_metadata: Mapping[Any, Any] | None,
    recording_index: int,
    label: str,
) -> Mapping[str, str] | ChannelMetadata | None:
    if channel_metadata is None:
        return None
    if recording_index in channel_metadata:
        per_recording = channel_metadata[recording_index]
        if isinstance(per_recording, Mapping) and per_recording:
            matched = _label_override(per_recording, label)
            if matched is not None:
                return matched
    return _label_override(channel_metadata, label)


def _label_override(
    mapping: Mapping[Any, Any], label: str
) -> Mapping[str, str] | ChannelMetadata | None:
    value = mapping.get(label)
    if value is not None:
        return value
    for key, candidate in mapping.items():
        if isinstance(key, str) and key.casefold() == label.casefold():
            return candidate
    return None  # pragma: no cover - recording indices are always present


def _signal_metadata(
    signal: _Signal,
    override: Mapping[str, str] | ChannelMetadata | None,
) -> ChannelMetadata:
    if isinstance(override, ChannelMetadata):
        values: Mapping[str, Any] = {
            "unit": override.unit,
            "reference": override.reference,
            "sampling_frequency": (
                override.sampling_frequency if override.sampling_frequency else None
            ),
            "prefiltering": override.prefiltering,
            "transducer_type": override.transducer_type,
        }
    elif override:
        values = {
            "unit": override.get("unit", override.get("physical_dimension")),
            "reference": override.get("reference"),
            "sampling_frequency": override.get("sampling_frequency"),
            "prefiltering": override.get("prefiltering"),
            "transducer_type": override.get("transducer_type"),
        }
    else:
        values = {}
    return ChannelMetadata(
        label=signal.label,
        unit=(
            signal.physical_dimension
            if values.get("unit") is None
            else str(values["unit"])
        ),
        reference=(
            extract_reference(signal)
            if values.get("reference") is None
            else str(values["reference"])
        ),
        sampling_frequency=(
            signal.sampling_frequency
            if values.get("sampling_frequency") is None
            else float(values["sampling_frequency"])
        ),
        prefiltering=(
            signal.prefiltering
            if values.get("prefiltering") is None
            else str(values["prefiltering"])
        ),
        transducer_type=(
            signal.transducer_type
            if values.get("transducer_type") is None
            else str(values["transducer_type"])
        ),
    )


def _resolve_target_units(
    all_metadata: Sequence[Mapping[str, ChannelMetadata]],
    target_units: Mapping[str, str] | None,
) -> dict[str, str]:
    candidates: dict[str, list[str]] = {}
    for metadata in all_metadata:
        for label, meta in metadata.items():
            units = candidates.setdefault(label, [])
            if meta.unit not in units:
                units.append(meta.unit)
    requested = {
        str(key).casefold(): value for key, value in (target_units or {}).items()
    }
    resolved: dict[str, str] = {}
    for label, units in candidates.items():
        if label.casefold() in requested:
            resolved[label] = requested[label.casefold()]
        else:
            resolved[label] = _preferred_unit(units)
    return resolved


def _preferred_unit(units: Sequence[str]) -> str:
    """Pick the highest-resolution unit within the first unit's dimension."""
    best = units[0]
    best_entry = _unit_entry(best)
    if best_entry is None:
        return best
    dimension, best_scale, _ = best_entry
    for unit in units[1:]:
        entry = _unit_entry(unit)
        if entry is not None and entry[0] == dimension and entry[1] < best_scale:
            best = unit
            best_entry = entry
            _, best_scale, _ = entry
    return best


def _unit_entry(unit: str) -> tuple[str, float, float] | None:
    return _BUILTIN_UNITS.get(_canonical_unit(unit))


def _validate_sources(sources: Sequence[_Recording]) -> None:
    if not sources:
        raise ValueError("At least one recording is required")
    if any(type(recording) is not type(sources[0]) for recording in sources):
        raise ValueError("Cannot concatenate EDF and BDF recordings")


def _plan_groups(
    sources: Sequence[_Recording],
    *,
    rules: Mapping[tuple[str, str], UnitConversion],
    target_units: Mapping[str, str] | None,
    channel_metadata: Mapping[Any, Any] | None,
    reference_equivalences: Any,
    channel_policy: _ChannelPolicy,
    on_incompatible: _OnIncompatible,
    require_prefiltering_match: bool,
) -> list[_Group]:
    equivalence_groups = _normalize_reference_equivalences(reference_equivalences)
    all_metadata = [
        _recording_metadata(recording, index, channel_metadata)
        for index, recording in enumerate(sources)
    ]
    global_targets = _resolve_target_units(all_metadata, target_units)
    explicit_targets = frozenset(str(key).casefold() for key in (target_units or ()))
    groups: list[_Group] = []
    for recording_index, recording in enumerate(sources):
        metadata = all_metadata[recording_index]
        if not groups:
            groups.append(
                _start_group(
                    recording,
                    recording_index,
                    metadata,
                    rules,
                    global_targets,
                    explicit_targets,
                )
            )
            continue
        group = groups[-1]
        group.output_index = len(groups) - 1
        conflicts, accepted = _compare_recording(
            group,
            recording,
            recording_index,
            metadata,
            rules,
            equivalence_groups,
            channel_policy,
            require_prefiltering_match=require_prefiltering_match,
        )
        if accepted:
            _append_recording(
                group,
                recording,
                recording_index,
                metadata,
                conflicts,
                rules,
            )
            continue
        group.conflicts.extend(conflicts)
        if on_incompatible == "error":
            details = (
                "; ".join(c.description for c in conflicts) or "incompatible metadata"
            )
            raise ConcatenationError(
                f"Recording {recording_index} cannot be concatenated: {details}"
            )
        groups.append(
            _start_group(
                recording,
                recording_index,
                metadata,
                rules,
                global_targets,
                explicit_targets,
            )
        )
    return groups


def _start_group(
    recording: _Recording,
    recording_index: int,
    metadata: Mapping[str, ChannelMetadata],
    rules: Mapping[tuple[str, str], UnitConversion],
    target_units: Mapping[str, str],
    explicit_targets: frozenset[str],
) -> _Group:
    group = _Group(output_index=0, first_recording=recording)
    group.segments.append((recording_index, 0.0, recording.duration))
    group.offset = recording.duration
    for label, signal in _signals_by_label(recording).items():
        meta = metadata[label]
        target_unit = target_units.get(label, meta.unit)
        conversion = _find_conversion(meta.unit, target_unit, rules)
        if conversion is None and label.casefold() not in explicit_targets:
            target_unit = meta.unit
            conversion = _find_conversion(meta.unit, target_unit, rules)
        if conversion is None:
            raise ConcatenationError(
                f"Cannot convert channel {label!r} from unit "
                f"{meta.unit or '<unspecified>'!r} to "
                f"{target_unit or '<unspecified>'!r}"
            )
        if conversion.is_identity:
            conversion = None
        group.channels[label] = _MergedChannel(
            label=label,
            target_unit=_ascii_header_text(target_unit),
            reference=_canonical_reference(meta.reference),
            sampling_frequency=meta.sampling_frequency,
            prefiltering=_normalize_text(meta.prefiltering),
            signals=[signal],
            conversions=[conversion],
            use_digital=conversion is None,
        )
    group.annotations.extend(recording.annotations)
    return group


def _signals_by_label(recording: _Recording) -> dict[str, _Signal]:
    by_label: dict[str, list[_Signal]] = {}
    for signal in recording.signals:
        by_label.setdefault(signal.label, []).append(signal)
    duplicates = [label for label, signals in by_label.items() if len(signals) > 1]
    if duplicates:
        raise ConcatenationError(
            "Ambiguous channels with duplicate labels: " + ", ".join(duplicates)
        )
    return {label: signals[0] for label, signals in by_label.items()}


def _compare_recording(
    group: _Group,
    recording: _Recording,
    recording_index: int,
    metadata: Mapping[str, ChannelMetadata],
    rules: Mapping[tuple[str, str], UnitConversion],
    reference_equivalences: Sequence[frozenset[str]],
    channel_policy: _ChannelPolicy,
    *,
    require_prefiltering_match: bool,
) -> tuple[list[ChannelConflict], bool]:
    conflicts: list[ChannelConflict] = []
    output_index = group.output_index
    signal_labels = list(_signals_by_label(recording))
    group_labels = list(group.channels)

    if channel_policy == "strict" and set(signal_labels) != set(group_labels):
        for label in sorted(set(group_labels) - set(signal_labels)):
            conflicts.append(
                ChannelConflict(
                    output_index,
                    recording_index,
                    label,
                    "missing_channel",
                    "channel is absent from the recording",
                )
            )
        for label in sorted(set(signal_labels) - set(group_labels)):
            conflicts.append(
                ChannelConflict(
                    output_index,
                    recording_index,
                    label,
                    "extra_channel",
                    "channel is not present in the output",
                )
            )
        return conflicts, False

    shared = [label for label in group_labels if label in signal_labels]
    if not shared:
        conflicts.append(
            ChannelConflict(
                output_index,
                recording_index,
                "",
                "no_shared_channels",
                "no channel is present in both the output and the recording",
            )
        )
        return conflicts, False

    for label in group_labels:
        if label not in signal_labels:
            conflicts.append(
                ChannelConflict(
                    output_index,
                    recording_index,
                    label,
                    "missing_channel",
                    "channel is absent from the recording",
                )
            )
            continue
        channel = group.channels[label]
        meta = metadata[label]
        conflicts.extend(
            _compare_channel(
                channel,
                meta,
                rules,
                reference_equivalences,
                require_prefiltering_match=require_prefiltering_match,
                output_index=output_index,
                recording_index=recording_index,
            )
        )

    incompatible = [c for c in conflicts if c.channel in shared]
    if channel_policy == "strict" and incompatible:
        return conflicts, False
    if channel_policy == "intersection":
        for label in signal_labels:
            if label not in group_labels:  # pragma: no cover - guarded by strict check
                conflicts.append(
                    ChannelConflict(
                        output_index,
                        recording_index,
                        label,
                        "extra_channel",
                        "channel is not present in the output",
                    )
                )
        accepted_labels = [
            label for label in shared if not any(c.channel == label for c in conflicts)
        ]
        if not accepted_labels:  # pragma: no cover - guarded by the no-shared check
            return conflicts, False
    return conflicts, True


def _compare_channel(
    channel: _MergedChannel,
    meta: ChannelMetadata,
    rules: Mapping[tuple[str, str], UnitConversion],
    reference_equivalences: Sequence[frozenset[str]],
    *,
    require_prefiltering_match: bool,
    output_index: int,
    recording_index: int,
) -> list[ChannelConflict]:
    conflicts: list[ChannelConflict] = []
    if not math.isclose(
        channel.sampling_frequency,
        meta.sampling_frequency,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        conflicts.append(
            ChannelConflict(
                output_index,
                recording_index,
                channel.label,
                "sampling_frequency",
                f"{meta.sampling_frequency:g} Hz != {channel.sampling_frequency:g} Hz",
            )
        )
    if not _references_equivalent(
        channel.reference,
        _canonical_reference(meta.reference),
        reference_equivalences,
    ):
        conflicts.append(
            ChannelConflict(
                output_index,
                recording_index,
                channel.label,
                "reference",
                f"{meta.reference or '<unspecified>'} != "
                f"{channel.reference or '<unspecified>'}",
            )
        )
    if _find_conversion(meta.unit, channel.target_unit, rules) is None:
        conflicts.append(
            ChannelConflict(
                output_index,
                recording_index,
                channel.label,
                "unit",
                f"cannot convert {meta.unit or '<unspecified>'} to "
                f"{channel.target_unit or '<unspecified>'}",
            )
        )
    if (
        require_prefiltering_match
        and _normalize_text(meta.prefiltering) != channel.prefiltering
    ):
        conflicts.append(
            ChannelConflict(
                output_index,
                recording_index,
                channel.label,
                "prefiltering",
                f"{meta.prefiltering!r} != {channel.first.prefiltering!r}",
            )
        )
    return conflicts


def _append_recording(
    group: _Group,
    recording: _Recording,
    recording_index: int,
    metadata: Mapping[str, ChannelMetadata],
    conflicts: list[ChannelConflict],
    rules: Mapping[tuple[str, str], UnitConversion],
) -> None:
    start = group.offset
    group.offset += recording.duration
    group.segments.append((recording_index, start, recording.duration))
    group.conflicts.extend(conflicts)
    group.annotations.extend(
        EdfAnnotation(annotation.onset + start, annotation.duration, annotation.text)
        for annotation in recording.annotations
    )

    accepted_labels = [
        label
        for label in group.channels
        if label in metadata and not any(c.channel == label for c in conflicts)
    ]
    for label in list(group.channels):
        if label not in accepted_labels:
            del group.channels[label]
    signals = _signals_by_label(recording)
    for label in accepted_labels:
        channel = group.channels[label]
        meta = metadata[label]
        conversion = _find_conversion(meta.unit, channel.target_unit, rules)
        if conversion is not None and conversion.is_identity:
            conversion = None
        signal = signals[label]
        if (
            channel.use_digital
            and conversion is None
            and signal.physical_range == channel.first.physical_range
            and signal.digital_range == channel.first.digital_range
        ):
            channel.use_digital = True
        else:
            channel.use_digital = False
        channel.signals.append(signal)
        channel.conversions.append(conversion)


def _build_output(
    group: _Group, *, annotate_conversions: bool
) -> tuple[_Recording, list[ConversionRecord]]:
    signals: list[_Signal] = []
    conversions: list[ConversionRecord] = []
    for label, channel in group.channels.items():
        first = channel.first
        channel_conversions: list[ConversionRecord] = []
        for segment_index, (recording_index, start, _) in enumerate(group.segments):
            rule = channel.conversions[segment_index]
            if rule is None:
                continue
            record = ConversionRecord(
                output_index=group.output_index,
                recording_index=recording_index,
                channel=label,
                source_unit=channel.signals[segment_index].physical_dimension,
                target_unit=channel.target_unit,
                factor=rule.factor,
                offset=rule.offset,
                rule=rule.description,
            )
            channel_conversions.append(record)
            conversions.append(record)
            if annotate_conversions:
                group.annotations.append(
                    EdfAnnotation(
                        start,
                        None,
                        _conversion_annotation(
                            label,
                            record.source_unit,
                            channel.target_unit,
                            rule,
                        ),
                    )
                )
        if channel.use_digital and all(rule is None for rule in channel.conversions):
            digital = np.concatenate([signal.digital for signal in channel.signals])
            signal = type(first).from_digital(
                digital,
                first.sampling_frequency,
                label=label,
                transducer_type=_ascii_header_text(first.transducer_type),
                physical_dimension=channel.target_unit,
                physical_range=first.physical_range,
                digital_range=first.digital_range,
                prefiltering=first.prefiltering,
            )
        else:
            data = np.concatenate(
                [
                    (rule.apply(source.data) if rule is not None else source.data)
                    for source, rule in zip(channel.signals, channel.conversions)
                ]
            )
            signal = type(first)(
                data,
                first.sampling_frequency,
                label=label,
                transducer_type=_ascii_header_text(first.transducer_type),
                physical_dimension=channel.target_unit,
                digital_range=first.digital_range,
                prefiltering=first.prefiltering,
            )
        signal.conversion_records = tuple(channel_conversions)  # type: ignore[union-attr]
        signals.append(signal)

    first_recording = group.first_recording
    annotations = _sort_annotations(group.annotations) if group.annotations else None
    output = type(first_recording)(
        signals,  # type: ignore[arg-type]
        patient=first_recording.patient,
        recording=first_recording.recording,
        starttime=first_recording.starttime,
        annotations=annotations,
    )
    output.conversion_records = tuple(conversions)  # type: ignore[union-attr]
    return output, conversions


def _sort_annotations(
    annotations: Iterable[EdfAnnotation],
) -> tuple[EdfAnnotation, ...]:
    return tuple(sorted(annotations))


def _ascii_header_text(text: str) -> str:
    """Return `text` unchanged if it is ASCII-encodable, else a safe replacement.

    Signals read with a non-ASCII header encoding (e.g. ``latin-1``) can contain
    characters such as ``µ``. New signals are written with ASCII headers, so
    these characters have to be normalized before writing.
    """
    try:
        text.encode("ascii")
    except UnicodeEncodeError:
        return (
            text.replace("µ", "u")
            .replace("μ", "u")
            .replace("°", "deg")
            .encode("ascii", errors="replace")
            .decode("ascii")
        )
    return text


def _conversion_annotation(
    label: str,
    source_unit: str,
    target_unit: str,
    conversion: UnitConversion,
) -> str:
    text = (
        f"EDFIO: unit conversion {label}: "
        f"{source_unit or '<unspecified>'} -> {target_unit or '<unspecified>'}, "
        f"factor={conversion.factor:g}, offset={conversion.offset:g}"
    )
    if conversion.description:
        text += f", basis={conversion.description}"
    return text


def _merge_conversion_rules(
    first: _ConversionInput, second: _ConversionInput
) -> dict[tuple[str, str], UnitConversion]:
    rules: dict[tuple[str, str], UnitConversion] = {}
    for source in (first, second):
        rules.update(_normalize_conversion_rules(source))
    return rules


def _normalize_conversion_rules(
    conversions: _ConversionInput,
) -> dict[tuple[str, str], UnitConversion]:
    if conversions is None:
        return {}
    normalized: dict[tuple[str, str], UnitConversion] = {}
    items: Iterable[Any]
    if isinstance(conversions, UnitConversion):
        items = [conversions]
    elif isinstance(conversions, Mapping):
        items = list(_rules_from_any_mapping(conversions))
    else:
        items = conversions
    for item in items:
        if isinstance(item, UnitConversion):
            rule = item
        elif isinstance(item, (tuple, list)) and len(item) in (3, 4):
            source, target, factor, *rest = item
            offset = rest[0] if rest else 0.0
            rule = UnitConversion(
                str(source), str(target), float(factor), float(offset)
            )
        else:
            raise ValueError(f"Invalid unit conversion: {item!r}")
        normalized[(_canonical_unit(rule.source), _canonical_unit(rule.target))] = rule
    return normalized


def _rules_from_any_mapping(
    conversions: Mapping[Any, Any],
) -> Iterator[UnitConversion]:
    if conversions and all(
        _is_unit_scale(value) and _looks_like_unit_key(key)
        for key, value in conversions.items()
    ):
        if len(conversions) < 2:  # noqa: PLR2004
            raise ValueError(
                "Invalid conversion key(s): a unit scale table needs at "
                f"least two units, got {dict(conversions)!r}"
            )
        yield from _scales_to_rules(conversions)
        return
    for key, value in conversions.items():
        if isinstance(key, tuple) and len(key) == 2:  # noqa: PLR2004
            yield _rule_from_value(str(key[0]), str(key[1]), value)
        else:
            yield from _rules_from_mapping({key: value})


def _looks_like_unit_key(key: Any) -> bool:
    return isinstance(key, str) and not any(sep in key for sep in ("->", ":"))


def _is_unit_scale(value: Any) -> bool:
    if isinstance(value, (Mapping, UnitConversion)):
        return False
    if isinstance(value, (tuple, list)):
        return len(value) == 1
    return True


def _scales_to_rules(scales: Mapping[Any, Any]) -> Iterator[UnitConversion]:
    """Turn a ``{unit: scale to base}`` mapping into pairwise conversions."""
    normalized: dict[str, float] = {}
    for unit, raw_scale in scales.items():
        scale = raw_scale[0] if isinstance(raw_scale, (tuple, list)) else raw_scale
        normalized[_canonical_unit(str(unit))] = float(scale)
    for source, source_scale in normalized.items():
        for target, target_scale in normalized.items():
            if source == target:
                continue
            factor = source_scale / target_scale
            yield UnitConversion(source, target, factor)


def _rules_from_mapping(
    conversions: Mapping[Any, Any],
) -> Iterator[UnitConversion]:
    for key, value in conversions.items():
        if isinstance(key, str):
            if "->" in key:
                source, target = key.split("->", 1)
            elif ":" in key:
                source, target = key.split(":", 1)
            elif isinstance(value, Mapping):
                source = key
                target = ""
            else:
                raise ValueError(f"Invalid conversion key: {key!r}")  # pragma: no cover
            if isinstance(value, Mapping):
                for nested_target, nested_value in value.items():
                    yield _rule_from_value(source, str(nested_target), nested_value)
            else:
                yield _rule_from_value(source, target, value)
        else:  # pragma: no cover - unreachable via public input forms
            raise ValueError(f"Invalid conversion key: {key!r}")


def _rule_from_value(source: str, target: str, value: Any) -> UnitConversion:
    if isinstance(value, UnitConversion):  # pragma: no cover - normalized earlier
        return value
    if isinstance(value, (tuple, list)):
        if len(value) == 1:
            return UnitConversion(source, target, float(value[0]))
        if len(value) == 2:  # noqa: PLR2004
            return UnitConversion(source, target, float(value[0]), float(value[1]))
        raise ValueError(f"Invalid conversion value: {value!r}")
    return UnitConversion(source, target, float(value))


def _find_conversion(
    source: str,
    target: str,
    rules: Mapping[tuple[str, str], UnitConversion],
) -> UnitConversion | None:
    source_key = _canonical_unit(source)
    target_key = _canonical_unit(target)
    if source_key == target_key:
        return UnitConversion(source, target)
    rule = rules.get((source_key, target_key))
    if rule is not None:
        return rule
    inverse = rules.get((target_key, source_key))
    if inverse is not None:
        return inverse.inverted()
    source_unit = _BUILTIN_UNITS.get(source_key)
    target_unit = _BUILTIN_UNITS.get(target_key)
    if (
        source_unit is not None
        and target_unit is not None
        and source_unit[0] == target_unit[0]
    ):
        _, source_scale, source_offset = source_unit
        _, target_scale, target_offset = target_unit
        source_scale_d = _decimal(source_scale)
        target_scale_d = _decimal(target_scale)
        factor_d = source_scale_d / target_scale_d
        offset_d = _decimal(source_offset) * factor_d - _decimal(target_offset)
        return UnitConversion(
            source,
            target,
            float(factor_d),
            float(offset_d),
        )
    return None


def _decimal(value: float) -> decimal.Decimal:
    return decimal.Decimal(str(value))


def _canonical_unit(unit: str) -> str:
    if not unit:
        return ""
    text = unit.strip().lower().replace("µ", "u").replace("μ", "u")
    text = text.replace("°", "deg").replace("degrees", "deg")
    text = text.replace("degree", "deg")
    text = text.replace("%", "percent")
    text = re.sub(r"[^a-z0-9]+", "", text)
    return _UNIT_ALIASES.get(text, text)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


_REFERENCE_ALIASES = {
    "": "",
    "ref": "",
    "reference": "",
    "referential": "",
    "unipolar": "",
    "x": "",
    "unknown": "",
    "na": "",
    "none": "",
    "cz": "cz",
    "c z": "cz",
    "m1": "m1",
    "left mastoid": "m1",
    "lm": "m1",
    "a1": "a1",
    "left ear": "a1",
    "le": "a1",
    "m2": "m2",
    "right mastoid": "m2",
    "rm": "m2",
    "a2": "a2",
    "right ear": "a2",
    "re": "a2",
    "linked mastoid": "linked_mastoid",
    "linked mastoids": "linked_mastoid",
    "linked ear": "linked_ear",
    "m1 m2": "linked_mastoid",
    "m2 m1": "linked_mastoid",
    "m1+m2": "linked_mastoid",
    "m2+m1": "linked_mastoid",
    "a1 a2": "linked_ear",
    "a2 a1": "linked_ear",
    "a1+a2": "linked_ear",
    "a2+a1": "linked_ear",
    "linked ears": "linked_ear",
    "average": "average",
    "avg": "average",
    "common average": "average",
    "car": "average",
    "ground": "ground",
    "gnd": "ground",
}


_ELECTRODE_REFERENCE_NAMES = {
    "cz",
    "m1",
    "m2",
    "a1",
    "a2",
    "linked_mastoid",
    "linked_ear",
    "average",
    "ground",
}


def _canonical_reference(reference: str) -> str:
    if not reference:
        return ""
    text = reference.strip().lower().replace("µ", "u")
    text = re.sub(r"[^a-z0-9+]+", " ", text).strip()
    text = re.sub(r"\s+", " ", text)
    if text in _REFERENCE_ALIASES:
        return _REFERENCE_ALIASES[text]
    stripped = re.sub(r"\s*(?:reference|ref|electrode)$", "", text).strip()
    return _REFERENCE_ALIASES.get(stripped, stripped)


def _normalize_reference_equivalences(
    equivalences: Any,
) -> tuple[frozenset[str], ...]:
    if equivalences is None:
        return ()
    groups: list[frozenset[str]] = []
    items: Iterable[Any]
    if isinstance(equivalences, Mapping):
        items = [tuple(value) for value in equivalences.values()]
    else:
        items = [equivalences] if isinstance(equivalences, str) else equivalences
    for item in items:
        if isinstance(item, str):
            parts = re.split(r"\s*(?:=|<=>|~)\s*", item)
        else:
            parts = list(item)
        normalized = frozenset(
            reference
            for reference in (_canonical_reference(str(part)) for part in parts)
            if reference
        )
        if len(normalized) > 1:
            groups.append(normalized)
    return tuple(groups)


def _references_equivalent(
    first: str,
    second: str,
    equivalence_groups: Sequence[frozenset[str]],
) -> bool:
    if first == second:
        return True
    if (
        not first or not second
    ):  # pragma: no cover - unspecified refs equal only to each other
        return False
    return any(first in group and second in group for group in equivalence_groups)


_UNIT_ALIASES = {
    "microvolt": "uv",
    "microvolts": "uv",
    "microv": "uv",
    "millivolt": "mv",
    "millivolts": "mv",
    "milliv": "mv",
    "volt": "v",
    "volts": "v",
    "nanovolt": "nv",
    "nanovolts": "nv",
    "kilovolt": "kv",
    "kilovolts": "kv",
    "second": "s",
    "seconds": "s",
    "sec": "s",
    "millisecond": "ms",
    "milliseconds": "ms",
    "microsecond": "us",
    "microseconds": "us",
    "nanosecond": "ns",
    "nanoseconds": "ns",
    "minute": "min",
    "minutes": "min",
    "hour": "h",
    "hours": "h",
    "hr": "h",
    "hertz": "hz",
    "kilohertz": "khz",
    "megahertz": "mhz",
    "celsius": "degc",
    "centigrade": "degc",
    "fahrenheit": "degf",
    "kelvin": "kelvin",
    "percent": "percent",
}

# dimension, scale to the dimension's base unit, offset to the base unit
# (base = (value + offset) * scale)
_BUILTIN_UNITS: dict[str, tuple[str, float, float]] = {
    "nv": ("voltage", 1e-9, 0.0),
    "uv": ("voltage", 1e-6, 0.0),
    "mv": ("voltage", 1e-3, 0.0),
    "v": ("voltage", 1.0, 0.0),
    "kv": ("voltage", 1e3, 0.0),
    "ns": ("time", 1e-9, 0.0),
    "us": ("time", 1e-6, 0.0),
    "ms": ("time", 1e-3, 0.0),
    "s": ("time", 1.0, 0.0),
    "min": ("time", 60.0, 0.0),
    "h": ("time", 3600.0, 0.0),
    "hz": ("frequency", 1.0, 0.0),
    "khz": ("frequency", 1e3, 0.0),
    "mhz": ("frequency", 1e6, 0.0),
    "degc": ("temperature", 1.0, 273.15),
    "degf": ("temperature", 5.0 / 9.0, 459.67),
    "kelvin": ("temperature", 1.0, 0.0),
    "percent": ("ratio", 1.0, 0.0),
}
