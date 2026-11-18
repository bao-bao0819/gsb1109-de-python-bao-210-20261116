from __future__ import annotations

import datetime
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum

from edfio._header_field import (
    DATE_OR_TIME_PATTERN,
    _century_start_year,
    decode_str,
    decode_time,
)
from edfio.edf_header import _MONTH_NAMES

_ANONYMIZATION_PLACEHOLDER = (1, 1, 85)
_SHORT_YEAR_MAX_DIGITS = 2


class DateSource(str, Enum):
    """Header field that contained a recording date."""

    LEGACY_HEADER = "legacy_header"
    EDFPLUS_RECORDING = "edfplus_recording"


class DateStatus(str, Enum):
    """How much of a date can be determined from one header field."""

    EXACT = "exact"
    AMBIGUOUS = "ambiguous"
    ANONYMIZED = "anonymized"
    PLACEHOLDER = "placeholder"
    INVALID = "invalid"
    MISSING = "missing"


class TimeStatus(str, Enum):
    """Overall result of interpreting the recording start time."""

    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    CONFLICT = "conflict"
    ANONYMIZED = "anonymized"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class DateInterpretation:
    """
    Interpretation of one date-bearing header field.

    Parameters
    ----------
    source : DateSource
        The header field this interpretation was read from.
    raw : str | None
        The unparsed field value. `None` if the field does not contain a date.
    status : DateStatus
        Whether the field is exact, ambiguous, anonymized, a placeholder, invalid,
        or missing.
    value : datetime.date | None
        The exact date, if one could be determined.
    candidate_years : tuple[int, ...]
        For an ambiguous short year, every representable year consistent with the
        field. Empty for exact or unusable fields.
    day : int | None
        Parsed day of month, also retained for ambiguous dates.
    month : int | None
        Parsed month, also retained for ambiguous dates.
    year_suffix : int | None
        Parsed two-digit year, retained for short years.
    detail : str | None
        Explanation for non-exact statuses.
    """

    source: DateSource
    raw: str | None
    status: DateStatus
    value: datetime.date | None = None
    candidate_years: tuple[int, ...] = field(default=(), repr=False)
    day: int | None = None
    month: int | None = None
    year_suffix: int | None = None
    detail: str | None = None

    @property
    def earliest(self) -> datetime.date | None:
        """Earliest date compatible with this field, or `None`."""
        if self.value is not None:
            return self.value
        if self.candidate_years and self.day is not None and self.month is not None:
            return datetime.date(self.candidate_years[0], self.month, self.day)
        return None

    @property
    def latest(self) -> datetime.date | None:
        """Latest date compatible with this field, or `None`."""
        if self.value is not None:
            return self.value
        if self.candidate_years and self.day is not None and self.month is not None:
            return datetime.date(self.candidate_years[-1], self.month, self.day)
        return None

    @property
    def is_usable(self) -> bool:
        """Whether this field contributes a date or a date range."""
        return self.status in (DateStatus.EXACT, DateStatus.AMBIGUOUS)

    @property
    def candidate_dates(self) -> tuple[datetime.date, ...]:
        """All dates compatible with this field."""
        if self.value is not None:
            return (self.value,)
        if self.day is None or self.month is None:
            return ()
        return tuple(
            datetime.date(year, self.month, self.day) for year in self.candidate_years
        )


class _SourceMap(Mapping[DateSource, DateInterpretation]):
    def __init__(self, sources: tuple[DateInterpretation, ...]) -> None:
        self._sources = {source.source: source for source in sources}

    def __getitem__(self, key: DateSource) -> DateInterpretation:
        return self._sources[key]

    def __iter__(self) -> Iterator[DateSource]:
        return iter(self._sources)

    def __len__(self) -> int:
        return len(self._sources)


@dataclass(frozen=True)
class TimeInterpretation:
    """
    Recording time together with the header evidence it was derived from.

    Every date-bearing header field is kept in :attr:`sources`, including invalid
    and anonymized ones, so callers can audit why a date was accepted or rejected.
    :attr:`determinable_start_range` is the interval that follows from intersecting
    all usable source ranges.

    An exact :attr:`start` is only available if the sources determine a single date
    and the starttime field is valid. :meth:`datetime_at` advances from that actual
    start, so a recording that crosses midnight continues on the following date.

    Parameters
    ----------
    sources : tuple[DateInterpretation, ...]
        All date-bearing header fields that were considered.
    status : TimeStatus
        Overall interpretation status.
    duration : float
        Recording duration in seconds.
    start : datetime.datetime | None
        The exact recording start, if determinable.
    end : datetime.datetime | None
        The exact recording end, if determinable.
    starttime : datetime.time | None
        The effective time of day, including any subsecond annotation offset.
    raw_starttime : datetime.time | None
        The time of day as written in the header, without the subsecond offset.
    subsecond_offset : float
        Offset of the first EDF+ timekeeping annotation in seconds.
    earliest_start : datetime.datetime | None
        Earliest recording start compatible with the sources.
    latest_start : datetime.datetime | None
        Latest recording start compatible with the sources.
    detail : str | None
        Explanation for a non-resolved status.
    """

    sources: tuple[DateInterpretation, ...]
    status: TimeStatus
    duration: float
    start: datetime.datetime | None = None
    end: datetime.datetime | None = None
    starttime: datetime.time | None = None
    raw_starttime: datetime.time | None = None
    subsecond_offset: float = 0.0
    earliest_start: datetime.datetime | None = None
    latest_start: datetime.datetime | None = None
    detail: str | None = None

    @property
    def by_source(self) -> Mapping[DateSource, DateInterpretation]:
        """The per-source interpretations keyed by :class:`DateSource`."""
        return _SourceMap(self.sources)

    @property
    def has_anonymized_source(self) -> bool:
        """Whether any date-bearing field was anonymized."""
        return any(
            source.status in (DateStatus.ANONYMIZED, DateStatus.PLACEHOLDER)
            for source in self.sources
        )

    @property
    def determinable_start_range(
        self,
    ) -> tuple[datetime.datetime, datetime.datetime] | None:
        """
        The interval known to contain the recording start.

        `None` if the sources conflict, are anonymized, or are otherwise unusable.
        """
        if self.earliest_start is None or self.latest_start is None:
            return None
        return self.earliest_start, self.latest_start

    @property
    def candidate_dates(self) -> tuple[datetime.date, ...]:
        """All recording dates compatible with the usable sources."""
        return _source_dates(self.sources)

    @property
    def determinable_date_range(
        self,
    ) -> tuple[datetime.date, datetime.date] | None:
        """The interval of dates compatible with the usable sources."""
        dates = self.candidate_dates
        if not dates:
            return None
        return dates[0], dates[-1]

    @property
    def determinable_range(self) -> tuple[datetime.datetime, datetime.datetime] | None:
        """
        The interval known to contain the whole recording.

        For an exact start this is the recording interval. For an ambiguous date it
        spans the earliest possible start to the latest possible end. It is `None`
        if no absolute interval can be determined.
        """
        if self.start is not None and self.end is not None:
            return self.start, self.end
        if self.earliest_start is None or self.latest_start is None:
            return None
        latest_end = self.latest_start + datetime.timedelta(seconds=self.duration)
        return self.earliest_start, latest_end

    def datetime_at(self, seconds: float) -> datetime.datetime | None:
        """
        Absolute datetime at a recording-relative onset.

        The offset is added to the resolved actual start, so an onset past midnight
        advances the date instead of re-interpreting a fresh clock face. Returns
        `None` if no exact start is available.
        """
        if self.start is None:
            return None
        return self.start + datetime.timedelta(seconds=seconds)

    def datetime_range_at(
        self, seconds: float
    ) -> tuple[datetime.datetime, datetime.datetime] | None:
        """Possible absolute datetimes at an onset, if the start is ambiguous."""
        start_range = self.determinable_start_range
        if start_range is None:
            return None
        offset = datetime.timedelta(seconds=seconds)
        return start_range[0] + offset, start_range[1] + offset

    def get_source(self, source: DateSource | str) -> DateInterpretation:
        """Return the interpretation of a particular source."""
        return self.by_source[DateSource(source)]


def _short_year_dates(
    day: int,
    month: int,
    year_suffix: int,
    century: int | None,
) -> tuple[datetime.date, ...]:
    if century is None:
        # A two-digit EDF year is ambiguous between the 20th and 21st century.
        # Do not silently collapse this to the EDF 1985..2084 rule: an EDF+
        # recording date such as 2099 must be able to pin a legacy "99".
        years: tuple[int, ...] = (1900 + year_suffix, 2000 + year_suffix)
    else:
        years = (_century_start_year(century) + year_suffix,)
    dates = []
    for year in years:
        try:
            dates.append(datetime.date(year, month, day))
        except ValueError:
            continue
    return tuple(dates)


def _ambiguous_short_year(
    source: DateSource,
    raw: str,
    dates: tuple[datetime.date, ...],
    year_suffix: int,
    detail: str,
) -> DateInterpretation:
    return DateInterpretation(
        source,
        raw,
        DateStatus.AMBIGUOUS,
        candidate_years=tuple(date.year for date in dates),
        day=dates[0].day,
        month=dates[0].month,
        year_suffix=year_suffix,
        detail=detail,
    )


def _parse_legacy_date(raw_bytes: bytes, century: int | None) -> DateInterpretation:
    raw = decode_str(raw_bytes)
    match = DATE_OR_TIME_PATTERN.fullmatch(raw)
    if match is None:
        return DateInterpretation(
            DateSource.LEGACY_HEADER,
            raw,
            DateStatus.INVALID,
            detail=f"Invalid date for format DD.MM.YY: {raw!r}",
        )
    day, month, year_suffix = (int(group) for group in match.groups())
    dates = _short_year_dates(day, month, year_suffix, century)
    if not dates:
        return DateInterpretation(
            DateSource.LEGACY_HEADER,
            raw,
            DateStatus.INVALID,
            day=day,
            month=month,
            year_suffix=year_suffix,
            detail=f"Invalid date for format DD.MM.YY: {raw!r}",
        )
    if len(dates) == 1:
        return DateInterpretation(
            DateSource.LEGACY_HEADER,
            raw,
            DateStatus.EXACT,
            value=dates[0],
            day=day,
            month=month,
            year_suffix=year_suffix,
        )
    return _ambiguous_short_year(
        DateSource.LEGACY_HEADER,
        raw,
        dates,
        year_suffix,
        "Legacy startdate has a two-digit year and no century basis was provided.",
    )


def _parse_recording_date(  # noqa: PLR0911
    local_recording: str, century: int | None
) -> DateInterpretation:
    if not local_recording.startswith("Startdate "):
        return DateInterpretation(
            DateSource.EDFPLUS_RECORDING,
            None,
            DateStatus.MISSING,
            detail=(
                "Local recording identification does not follow the EDF+ format: "
                f"{local_recording!r}"
            ),
        )
    subfields = local_recording.split()
    raw = subfields[1] if len(subfields) > 1 else "X"
    if raw == "X":
        return DateInterpretation(
            DateSource.EDFPLUS_RECORDING,
            raw,
            DateStatus.ANONYMIZED,
            detail="Recording startdate is not available ('X').",
        )
    try:
        day_field, month_field, year_field = raw.split("-")
        month = _MONTH_NAMES.index(month_field.upper()) + 1
        day = int(day_field)
    except (ValueError, IndexError) as error:
        return DateInterpretation(
            DateSource.EDFPLUS_RECORDING,
            raw,
            DateStatus.INVALID,
            detail=str(error),
        )
    if len(year_field) > _SHORT_YEAR_MAX_DIGITS:
        try:
            value = datetime.date(int(year_field), month, day)
        except ValueError as error:
            return DateInterpretation(
                DateSource.EDFPLUS_RECORDING,
                raw,
                DateStatus.INVALID,
                day=day,
                month=month,
                detail=str(error),
            )
        return DateInterpretation(
            DateSource.EDFPLUS_RECORDING,
            raw,
            DateStatus.EXACT,
            value=value,
            day=day,
            month=month,
        )
    try:
        year_suffix = int(year_field)
    except ValueError as error:
        return DateInterpretation(
            DateSource.EDFPLUS_RECORDING,
            raw,
            DateStatus.INVALID,
            day=day,
            month=month,
            detail=str(error),
        )
    dates = _short_year_dates(day, month, year_suffix, century)
    if not dates:
        return DateInterpretation(
            DateSource.EDFPLUS_RECORDING,
            raw,
            DateStatus.INVALID,
            day=day,
            month=month,
            year_suffix=year_suffix,
            detail=f"Invalid date: {raw!r}",
        )
    if len(dates) == 1:
        return DateInterpretation(
            DateSource.EDFPLUS_RECORDING,
            raw,
            DateStatus.EXACT,
            value=dates[0],
            day=day,
            month=month,
            year_suffix=year_suffix,
        )
    return _ambiguous_short_year(
        DateSource.EDFPLUS_RECORDING,
        raw,
        dates,
        year_suffix,
        "EDF+ recording date uses a short year and no century basis was provided.",
    )


def _mark_anonymization_placeholder(
    legacy: DateInterpretation,
) -> DateInterpretation:
    if (legacy.day, legacy.month, legacy.year_suffix) != _ANONYMIZATION_PLACEHOLDER:
        return legacy
    return DateInterpretation(
        legacy.source,
        legacy.raw,
        DateStatus.PLACEHOLDER,
        day=legacy.day,
        month=legacy.month,
        year_suffix=legacy.year_suffix,
        detail=(
            "Legacy startdate contains the anonymization placeholder 01.01.85. "
            "It is not treated as an actual recording date."
        ),
    )


def _source_dates(
    sources: tuple[DateInterpretation, ...],
) -> tuple[datetime.date, ...]:
    candidates: set[datetime.date] | None = None
    for source in sources:
        if not source.is_usable:
            continue
        source_candidates = set(source.candidate_dates)
        candidates = (
            source_candidates if candidates is None else candidates & source_candidates
        )
    if not candidates:
        return ()
    return tuple(sorted(candidates))


def _conflict_detail(sources: tuple[DateInterpretation, ...]) -> str:
    values = []
    for source in sources:
        if source.status is DateStatus.EXACT and source.value is not None:
            values.append(f"{source.source.value}={source.value.isoformat()}")
        elif source.earliest is not None and source.latest is not None:
            values.append(
                f"{source.source.value}={source.earliest.isoformat()}"
                f"..{source.latest.isoformat()}"
            )
        else:
            values.append(f"{source.source.value}={source.status.value}")
    return "Startdate sources disagree: " + ", ".join(values)


def _resolve_dates(
    sources: tuple[DateInterpretation, ...],
) -> tuple[TimeStatus, tuple[datetime.date, ...], str | None]:
    dates = _source_dates(sources)
    anonymized_note = _anonymized_note(sources)
    if dates:
        if len(dates) == 1:
            return TimeStatus.RESOLVED, dates, anonymized_note
        detail = (
            "The recording startdate is only determinable as a range; pass the "
            "century basis that applies to the short date."
        )
        if anonymized_note:
            detail = f"{detail} {anonymized_note}"
        return TimeStatus.AMBIGUOUS, dates, detail
    if anonymized_note is not None:
        return (
            TimeStatus.ANONYMIZED,
            (),
            "The recording startdate is anonymized and no source provides a usable "
            "date.",
        )
    if any(source.is_usable for source in sources):
        return TimeStatus.CONFLICT, (), _conflict_detail(sources)
    details = [source.detail for source in sources if source.detail is not None]
    return TimeStatus.UNAVAILABLE, (), "; ".join(details) or None


def _anonymized_note(sources: tuple[DateInterpretation, ...]) -> str | None:
    if not any(
        source.status in (DateStatus.ANONYMIZED, DateStatus.PLACEHOLDER)
        for source in sources
    ):
        return None
    return (
        "At least one date field is anonymized; the result uses the remaining "
        "usable source."
    )


def interpret_time(
    *,
    legacy_startdate: bytes,
    local_recording_identification: str,
    header_starttime: bytes,
    subsecond_offset: float,
    duration: float,
    century: int | None = None,
) -> TimeInterpretation:
    """
    Interpret EDF/BDF date and time header fields.

    Parameters
    ----------
    legacy_startdate : bytes
        Raw legacy `startdate` header field (format `DD.MM.YY`).
    local_recording_identification : str
        Local recording identification, which may carry an EDF+ startdate.
    header_starttime : bytes
        Raw `starttime` header field (format `hh.mm.ss`).
    subsecond_offset : float
        Offset of the first EDF+ timekeeping annotation in seconds.
    duration : float
        Recording duration in seconds.
    century : int | None, default: None
        Century basis for short years, supplied by the caller. Either a two-digit
        century (`19` for 1900-1999) or its start year (`1900`). If `None`, short
        years stay ambiguous. The current year is never used.

    Returns
    -------
    TimeInterpretation
        Per-source interpretations and the determinable range.
    """
    legacy = _parse_legacy_date(legacy_startdate, century)
    recording = _parse_recording_date(
        local_recording_identification,
        century,
    )
    if recording.status is DateStatus.ANONYMIZED:
        legacy = _mark_anonymization_placeholder(legacy)
    sources = (legacy, recording)

    status, dates, detail = _resolve_dates(sources)

    try:
        raw_starttime = decode_time(header_starttime)
    except ValueError as error:
        return TimeInterpretation(
            sources=sources,
            status=TimeStatus.UNAVAILABLE,
            duration=duration,
            detail=f"Invalid starttime: {error}",
        )
    offset = datetime.timedelta(seconds=subsecond_offset)
    starttime = (
        datetime.datetime.combine(datetime.date(2000, 1, 1), raw_starttime) + offset
    ).time()

    if not dates:
        return TimeInterpretation(
            sources=sources,
            status=status,
            duration=duration,
            starttime=starttime,
            raw_starttime=raw_starttime,
            subsecond_offset=subsecond_offset,
            detail=detail,
        )

    starts = tuple(
        datetime.datetime.combine(date, raw_starttime) + offset for date in dates
    )
    if len(starts) == 1:
        return TimeInterpretation(
            sources=sources,
            status=status,
            duration=duration,
            start=starts[0],
            end=starts[0] + datetime.timedelta(seconds=duration),
            starttime=starttime,
            raw_starttime=raw_starttime,
            subsecond_offset=subsecond_offset,
            earliest_start=starts[0],
            latest_start=starts[0],
            detail=detail,
        )
    return TimeInterpretation(
        sources=sources,
        status=status,
        duration=duration,
        starttime=starttime,
        raw_starttime=raw_starttime,
        subsecond_offset=subsecond_offset,
        earliest_start=starts[0],
        latest_start=starts[-1],
        detail=detail,
    )
