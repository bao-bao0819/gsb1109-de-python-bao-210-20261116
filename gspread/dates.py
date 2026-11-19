"""Interpretation of spreadsheet date, time, and duration cells.

Google Sheets stores dates, times, and durations as numbers in the same
"serial number" format popularised by Lotus 1-2-3: whole days since a
configurable epoch, with the fractional part being the time of day or the
fraction of a duration. A cell's *number format*, not only its value, decides
which of those readings is meant, and the format and the spreadsheet timezone
can change over time.

This module converts raw serial numbers into explicit, uncertainty-aware
results. It never consults the machine's local timezone: a spreadsheet has its
own timezone, and that timezone is used only when a caller explicitly asks to
turn a wall-clock reading into an absolute instant.

The central ideas are:

* A :class:`SerialAnchor` records the confirmed epoch of a serial number. Two
  anchors are supported out of the box: the Google Sheets epoch
  (``1899-12-30``) and the legacy Apple/Excel-for-Mac epoch (``1904-01-01``).
* :func:`classify_number_format` maps a Google Sheets number-format pattern to
  a :class:`FieldKind`, correctly distinguishing ``mm`` as month from ``mm`` as
  minute.
* :func:`interpret_value` interprets one raw value under a declared field kind
  and anchor, and keeps every plausible reading. A date with no time component
  stays a :class:`DateValue` with ``time=None``; it is never promoted to
  midnight.
* :func:`interpret_column` partitions a column into confirmed cells,
  unresolved cells (two legal readings belong to different semantic
  partitions, so the cell is left for judgement), and invalid cells.
* :meth:`ColumnResult.with_timezone` recomputes only timezone-dependent
  fields. Numbers, durations, pure times, and date-only readings are returned
  as the same objects, so a measurement column cannot drift when the
  spreadsheet timezone changes.

The module is free of Google API calls, so it can be used directly on values
read with ``ValueRenderOption.unformatted`` and
``DateTimeOption.serial_number``.
"""

import re
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timedelta, timezone
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

try:  # pragma: no cover - only relevant on Python 3.8
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore[assignment]


#: The Google Sheets serial epoch.
GOOGLE_SHEETS_EPOCH = datetime(1899, 12, 30)

#: The legacy Apple/Excel-for-Mac serial epoch.
MAC_EPOCH = datetime(1904, 1, 1)

#: Seconds in one spreadsheet day. Serials are wall-clock quantities, so this
#: is a fixed 24-hour day even across DST transitions.
SECONDS_PER_DAY = 86400


class FieldKind(str, Enum):
    """The semantic meaning of a spreadsheet cell."""

    date = "date"
    time = "time"
    datetime = "datetime"
    duration = "duration"
    number = "number"


@dataclass(frozen=True)
class SerialAnchor:
    """A confirmed epoch for spreadsheet serial numbers.

    :param epoch: The wall-clock moment that serial ``0`` denotes. Any
        timezone attached to ``epoch`` is intentionally discarded: serials are
        wall-clock quantities and do not carry a timezone.
    :param name: A human-readable identifier used in diagnostics.
    """

    epoch: datetime = GOOGLE_SHEETS_EPOCH
    name: str = "google_sheets"

    def __post_init__(self) -> None:
        if self.epoch.tzinfo is not None:
            object.__setattr__(self, "epoch", self.epoch.replace(tzinfo=None))

    @classmethod
    def google_sheets(cls) -> "SerialAnchor":
        """Return the standard Google Sheets anchor (``1899-12-30``)."""
        return cls(GOOGLE_SHEETS_EPOCH, "google_sheets")

    @classmethod
    def mac(cls) -> "SerialAnchor":
        """Return the legacy Apple/Excel-for-Mac anchor (``1904-01-01``)."""
        return cls(MAC_EPOCH, "mac")

    def to_datetime(self, serial: Union[int, float]) -> datetime:
        """Convert an absolute serial to a naive wall-clock datetime."""
        if isinstance(serial, bool) or not isinstance(serial, (int, float)):
            raise TypeError("serial must be a number, not {!r}".format(serial))
        return self.epoch + timedelta(days=float(serial))

    def to_serial(self, value: datetime) -> float:
        """Convert a naive wall-clock datetime to a serial number."""
        if value.tzinfo is not None:
            raise ValueError("datetime must be naive (wall clock)")
        return (value - self.epoch).total_seconds() / SECONDS_PER_DAY


@dataclass(frozen=True)
class Interval:
    """A half-open interval that records a reading's possible range.

    ``low`` is inclusive. ``high`` is exclusive unless ``high_inclusive`` is
    true. ``None`` denotes an unbounded end.
    """

    low: Any
    high: Any
    high_inclusive: bool = False

    def contains(self, value: Any) -> bool:
        if self.low is not None and value < self.low:
            return False
        if self.high is None:
            return True
        if self.high_inclusive:
            return value <= self.high
        return value < self.high

    def __str__(self) -> str:
        close = "]" if self.high_inclusive else ")"
        return "[{}, {}{}".format(self.low, self.high, close)


@dataclass(frozen=True)
class DateValue:
    """A date reading, optionally carrying a wall-clock time.

    ``value`` is a :class:`datetime.date`. ``time`` is ``None`` when the source
    carried no time component; a date-only value must not be treated as
    midnight. ``interval`` is the possible range of the underlying wall-clock
    reading: a whole calendar day for a date-only value, or a single point for
    an exact datetime. ``timezone`` is set only when an instant was explicitly
    resolved against a spreadsheet timezone.
    """

    value: date
    time: Optional[time] = None
    interval: Optional[Interval] = None
    timezone: Optional[str] = None

    @property
    def date(self) -> date:
        """Alias for ``value``, provided for readability."""
        return self.value

    @property
    def wall_clock(self) -> Optional[datetime]:
        """The naive wall-clock datetime, or ``None`` for a date-only value."""
        if self.time is None:
            return None
        return datetime.combine(self.value, self.time)

    @property
    def datetime(self) -> Optional[datetime]:
        """The wall-clock datetime, localized when ``timezone`` is set."""
        moment = self.wall_clock
        if moment is None:
            return None
        if self.timezone is None:
            return moment
        return _localize(moment, self.timezone)


@dataclass(frozen=True)
class TimeValue:
    """A pure time-of-day reading. The interval is an offset from day start."""

    value: time
    interval: Optional[Interval] = None


@dataclass(frozen=True)
class DurationValue:
    """A duration reading. Durations are elapsed quantities, never zoned."""

    value: timedelta
    interval: Optional[Interval] = None


@dataclass(frozen=True)
class NumberValue:
    """A plain number reading. Numbers never depend on a timezone."""

    value: Union[int, float]


Reading = Union[DateValue, TimeValue, DurationValue, NumberValue]


def _reading_family(reading: Reading) -> str:
    """Return the semantic partition a reading belongs to."""
    if isinstance(reading, DateValue):
        return "datetime" if reading.time is not None else "date"
    if isinstance(reading, TimeValue):
        return "time"
    if isinstance(reading, DurationValue):
        return "duration"
    return "number"


@dataclass(frozen=True)
class Interpretation:
    """The result of interpreting one raw cell value.

    :param raw: The original value, preserved untouched.
    :param kind: The primary declared field kind.
    :param reading: The primary reading, or ``None`` when the value cannot be
        interpreted under ``kind``.
    :param alternatives: Other legal readings. When an alternative belongs to
        a different semantic family, :attr:`ambiguous` is true and consumers
        must not silently choose the primary.
    :param reason: A short explanation when ``reading`` is ``None``.
    :param anchor: The serial anchor used, when the value was interpreted from
        a serial.
    :param number_format: The number format used, if any.
    :param timezone_name: The spreadsheet timezone used, if any.
    """

    raw: Any
    kind: Optional[FieldKind]
    reading: Optional[Reading] = None
    alternatives: Tuple[Reading, ...] = ()
    reason: Optional[str] = None
    anchor: Optional[SerialAnchor] = None
    number_format: Optional[str] = None
    timezone_name: Optional[str] = None

    @property
    def all_readings(self) -> Tuple[Reading, ...]:
        """Every legal reading, primary first."""
        return tuple(r for r in (self.reading,) + self.alternatives if r is not None)

    @property
    def families(self) -> frozenset:
        """The distinct semantic families among all legal readings."""
        return frozenset(_reading_family(r) for r in self.all_readings)

    @property
    def ambiguous(self) -> bool:
        """True when legal readings fall into different semantic partitions."""
        return len(self.families) > 1

    @property
    def resolved(self) -> bool:
        """True when the value has a reading and does not need judgement."""
        return self.reading is not None and not self.ambiguous


@dataclass(frozen=True)
class ColumnResult:
    """The interpretation of a whole column.

    :param values: The original values, in order and untouched.
    :param interpretations: One :class:`Interpretation` per input value.
    :param confirmed: Values with exactly one semantic reading, keyed by index.
    :param unresolved: Values with two legal readings in different partitions,
        keyed by index and deliberately left for judgement.
    :param invalid: Values that could not be interpreted at all, keyed by
        index.
    :param timezone_name: The spreadsheet timezone used, if one was supplied.
    """

    values: Tuple[Any, ...]
    interpretations: Tuple[Interpretation, ...]
    confirmed: Mapping[int, Interpretation] = field(default_factory=dict)
    unresolved: Mapping[int, Interpretation] = field(default_factory=dict)
    invalid: Mapping[int, Interpretation] = field(default_factory=dict)
    timezone_name: Optional[str] = None
    kinds: Tuple[Optional[FieldKind], ...] = ()
    number_formats: Tuple[Optional[str], ...] = ()
    anchor: Optional[SerialAnchor] = None

    def __iter__(self):
        return iter(self.interpretations)

    def __len__(self) -> int:
        return len(self.interpretations)

    def readings(self) -> Tuple[Optional[Reading], ...]:
        """Return the primary reading per cell, or ``None`` when absent."""
        return tuple(i.reading for i in self.interpretations)

    def with_timezone(self, timezone_name: str) -> "ColumnResult":
        """Return a copy recomputed for a new spreadsheet timezone.

        Only readings with an exact wall-clock datetime are recomputed. Number,
        duration, pure-time, and date-only readings are copied by identity, so
        a quantity column is guaranteed not to change when the timezone does.
        No machine timezone is consulted.
        """
        if not timezone_name:
            raise ValueError("timezone_name is required")

        new_interpretations = tuple(
            _reinterpret_interpretation(item, timezone_name)
            for item in self.interpretations
        )

        def rebuild(indexes: Mapping[int, Interpretation]) -> Dict[int, Interpretation]:
            return {index: new_interpretations[index] for index in indexes}

        return ColumnResult(
            values=self.values,
            interpretations=new_interpretations,
            confirmed=rebuild(self.confirmed),
            unresolved=rebuild(self.unresolved),
            invalid=rebuild(self.invalid),
            timezone_name=timezone_name,
            kinds=self.kinds,
            number_formats=self.number_formats,
            anchor=self.anchor,
        )

    def with_number_formats(
        self, number_formats: Union[str, Sequence[Optional[str]]]
    ) -> "ColumnResult":
        """Return a copy re-interpreted after the cells' formats changed.

        The raw values and anchors are unchanged; only the format-derived
        semantics are recomputed. A serial that was previously an unresolved
        number-or-date can therefore become a confirmed date once a date format
        is applied. The spreadsheet timezone and serial anchor, if known, are
        retained.
        """
        return interpret_column(
            self.values,
            kind=self.kinds or None,
            anchor=self.anchor,
            number_formats=number_formats,
            timezone_name=self.timezone_name,
        )


_DURATION_TOKEN_RE = re.compile(r"\[(h+|m+|s+)\]", re.IGNORECASE)


def _clean_format(number_format: str) -> str:
    """Strip literals, escapes, colours, conditions, and secondary sections."""
    fmt = number_format
    # Keep elapsed-time brackets but drop colours and conditions such as
    # [Red] or [>=100].
    fmt = re.sub(r"\[(?!(?:h+|m+|s+)\])[^\]]*\]", "", fmt, flags=re.IGNORECASE)
    fmt = re.sub(r'"[^"]*"', "", fmt)
    fmt = re.sub(r"\\.", "", fmt)
    fmt = fmt.replace("_", "").replace("*", "")
    # Only the first section describes positive values; later sections apply to
    # negatives and zeroes and must not change the classification.
    fmt = fmt.split(";")[0]
    return fmt.lower()


def _format_profile(number_format: Optional[str]) -> Optional[Dict[str, bool]]:
    """Analyse a number-format pattern without guessing across token classes."""
    if not number_format or not number_format.strip():
        return None

    fmt = _clean_format(number_format)
    if not fmt:
        return None

    has_duration = _DURATION_TOKEN_RE.search(fmt) is not None

    # Remove time markers before looking for month/minute "m" tokens.
    no_ampm = fmt.replace("am/pm", "").replace("a/p", "")

    has_hour = "h" in no_ampm
    has_second = "s" in no_ampm
    has_ampm = "am/pm" in fmt or "a/p" in fmt
    has_day = "d" in no_ampm
    has_year = "y" in no_ampm
    has_month_marker = "m" in no_ampm

    has_time_tokens = has_hour or has_second or has_ampm
    has_date_tokens = has_year or has_day or (has_month_marker and not has_time_tokens)

    return {
        "duration": has_duration,
        "time": has_time_tokens,
        "date": has_date_tokens,
    }


def classify_number_format(number_format: Optional[str]) -> Optional[FieldKind]:
    """Classify a Google Sheets number-format pattern.

    Returns ``None`` when the pattern is empty or carries no usable
    information; callers must not interpret "unknown" as "plain number".

    ``mm`` is treated as a minute when hour or second tokens are present, and
    as a month otherwise.

    Examples::

        >>> classify_number_format("yyyy-mm-dd")
        <FieldKind.date: 'date'>

        >>> classify_number_format("h:mm:ss am/pm")
        <FieldKind.time: 'time'>

        >>> classify_number_format("[h]:mm:ss")
        <FieldKind.duration: 'duration'>

        >>> classify_number_format("yyyy-mm-dd hh:mm")
        <FieldKind.datetime: 'datetime'>

        >>> classify_number_format("0.00")
        <FieldKind.number: 'number'>
    """
    profile = _format_profile(number_format)
    if profile is None:
        return None

    if profile["duration"]:
        return FieldKind.duration
    if profile["date"] and profile["time"]:
        return FieldKind.datetime
    if profile["date"]:
        return FieldKind.date
    if profile["time"]:
        return FieldKind.time

    fmt = _clean_format(number_format)
    if fmt in ("general", "0", "0.0", "0.00") or re.search(r"[0#?]", fmt):
        return FieldKind.number
    return None


def _format_has_time_tokens(number_format: Optional[str]) -> bool:
    profile = _format_profile(number_format)
    return bool(profile and profile["time"])


def serial_anchor(
    anchor: Optional[Union[SerialAnchor, datetime, str]] = None,
) -> SerialAnchor:
    """Normalise an anchor argument to a :class:`SerialAnchor`.

    ``None`` yields the Google Sheets epoch. A :class:`SerialAnchor` is
    returned as-is; a :class:`datetime` is wrapped; the strings ``"google"``,
    ``"google_sheets"``, ``"1899"``, ``"mac"``, ``"1904"``, and
    ``"excel_mac"`` select a built-in anchor.
    """
    if anchor is None:
        return SerialAnchor.google_sheets()
    if isinstance(anchor, SerialAnchor):
        return anchor
    if isinstance(anchor, datetime):
        return SerialAnchor(anchor, "custom")
    if isinstance(anchor, date) and not isinstance(anchor, datetime):
        return SerialAnchor(datetime.combine(anchor, time()), "custom")
    if isinstance(anchor, str):
        key = anchor.strip().lower()
        if key in ("google", "google_sheets", "1899", "1899-12-30"):
            return SerialAnchor.google_sheets()
        if key in ("mac", "1904", "1904-01-01", "excel_mac"):
            return SerialAnchor.mac()
    raise ValueError("unsupported serial anchor: {!r}".format(anchor))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _split_serial(serial: float) -> Tuple[int, int]:
    """Split a serial into whole days and whole seconds within the day."""
    days = int(serial // 1)
    fraction = serial - days
    seconds = int(round(fraction * SECONDS_PER_DAY))
    if seconds >= SECONDS_PER_DAY:
        days += 1
        seconds -= SECONDS_PER_DAY
    return days, seconds


def _time_from_seconds(seconds: int) -> time:
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return time(hours, minutes, secs)


def _date_only_reading(anchor: SerialAnchor, serial: float) -> DateValue:
    """Read only the calendar date, keeping the whole day as the interval.

    The fractional part is deliberately not used: a date-only field has no
    time-of-day information, so the possible reading is the entire calendar
    day, not an instant at midnight.
    """
    days, _ = _split_serial(serial)
    day_start = anchor.epoch + timedelta(days=days)
    return DateValue(
        value=day_start.date(),
        time=None,
        interval=Interval(day_start, day_start + timedelta(days=1)),
    )


def _exact_datetime_reading(
    anchor: SerialAnchor, serial: float, timezone_name: Optional[str]
) -> DateValue:
    days, seconds = _split_serial(serial)
    moment = anchor.epoch + timedelta(days=days, seconds=seconds)
    interval = Interval(moment, moment, high_inclusive=True)
    if timezone_name:
        local = _localize(moment, timezone_name)
        interval = Interval(local, local, high_inclusive=True)
    return DateValue(
        value=moment.date(),
        time=moment.time(),
        interval=interval,
        timezone=timezone_name,
    )


def _datetime_reading(
    anchor: SerialAnchor,
    serial: float,
    timezone_name: Optional[str],
    format_has_time: bool,
) -> DateValue:
    days, seconds = _split_serial(serial)
    if seconds == 0 and not format_has_time:
        # A serial with no fractional part in a datetime field carries no
        # time-of-day information. Keep the whole-day interval rather than
        # inventing midnight.
        return _date_only_reading(anchor, serial)
    return _exact_datetime_reading(anchor, serial, timezone_name)


def _time_reading(serial: float) -> TimeValue:
    """Build a time reading from a serial whose fractional part is the time."""
    _, seconds = _split_serial(serial)
    return _time_reading_from_seconds(seconds)


def _time_reading_from_seconds(seconds: int) -> TimeValue:
    """Build a time reading from a count of seconds since the start of day."""
    start = timedelta(seconds=seconds)
    return TimeValue(
        value=_time_from_seconds(seconds),
        interval=Interval(start, start, high_inclusive=True),
    )


def _duration_reading(serial: float) -> DurationValue:
    delta = timedelta(days=float(serial))
    return DurationValue(
        value=delta, interval=Interval(delta, delta, high_inclusive=True)
    )


def _reading_for_kind(
    serial: float,
    kind: FieldKind,
    anchor: SerialAnchor,
    timezone_name: Optional[str],
    format_has_time: bool,
) -> Reading:
    if kind == FieldKind.date:
        return _date_only_reading(anchor, serial)
    if kind == FieldKind.datetime:
        return _datetime_reading(anchor, serial, timezone_name, format_has_time)
    if kind == FieldKind.time:
        return _time_reading(serial)
    if kind == FieldKind.duration:
        return _duration_reading(serial)
    return NumberValue(serial)


def _interpret_numeric(
    serial: float,
    declared_kind: Optional[FieldKind],
    format_kind: Optional[FieldKind],
    anchor: SerialAnchor,
    number_format: Optional[str],
    timezone_name: Optional[str],
) -> Interpretation:
    format_has_time = _format_has_time_tokens(number_format)
    days, seconds = _split_serial(serial)

    # With neither declared semantics nor a usable format, a numeric value can
    # legitimately be a measurement or a date serial. Those are different
    # partitions, so the cell must stay unresolved. A fractional serial also
    # has a datetime reading, which is distinct from a date-only reading.
    if declared_kind is None and format_kind is None:
        if seconds == 0:
            date_alternative: Reading = _date_only_reading(anchor, serial)
        else:
            date_alternative = _exact_datetime_reading(anchor, serial, timezone_name)
        return Interpretation(
            raw=serial,
            kind=None,
            reading=NumberValue(serial),
            alternatives=(date_alternative,),
        )

    primary_kind = declared_kind or format_kind
    assert primary_kind is not None

    primary = _reading_for_kind(
        serial, primary_kind, anchor, timezone_name, format_has_time
    )
    alternatives: List[Reading] = []

    # Declared semantics and the cell's number format are two independent
    # legal sources. When they disagree about the semantic family, keep both
    # readings instead of silently letting one override the other.
    if (
        declared_kind is not None
        and format_kind is not None
        and declared_kind != format_kind
    ):
        alternative = _reading_for_kind(
            serial, format_kind, anchor, timezone_name, format_has_time
        )
        if _reading_family(alternative) != _reading_family(primary):
            alternatives.append(alternative)

    days, seconds = _split_serial(serial)

    # A date field with a fractional serial may either discard the fraction
    # (date-only format) or mean a datetime. With no format to decide, that is
    # two legal partitions.
    if primary_kind == FieldKind.date and seconds != 0 and not format_has_time:
        alternative = _exact_datetime_reading(anchor, serial, timezone_name)
        if _reading_family(alternative) != _reading_family(primary):
            alternatives.append(alternative)

    # A time field whose serial has whole days may be an elapsed duration
    # rather than a wrapped time of day. Without a duration/time format, keep
    # both readings.
    if declared_kind == FieldKind.time and days != 0 and number_format is None:
        alternative = _duration_reading(serial)
        alternatives.append(alternative)

    return Interpretation(
        raw=serial,
        kind=primary_kind,
        reading=primary,
        alternatives=tuple(alternatives),
    )


_TIME_RE = re.compile(
    r"^\s*(?P<hour>\d{1,2}):(?P<minute>\d{2})(?::(?P<second>\d{2}(?:\.\d+)?))?"
    r"\s*(?P<ampm>am|pm)?\s*$",
    re.IGNORECASE,
)

_DURATION_RE = re.compile(
    r"^\s*(?P<sign>[+-])?(?:(?P<days>\d+)\s+days?,?\s*)?"
    r"(?:(?P<hours>\d+):(?P<minutes>\d{2})(?::(?P<seconds>\d{2}(?:\.\d+)?))?|"
    r"(?P<hours_only>\d+(?:\.\d+)?)\s*(?:h|hours?))\s*$",
    re.IGNORECASE,
)


def _parse_time_text(text: str) -> Optional[time]:
    match = _TIME_RE.match(text)
    if not match:
        return None
    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    second = float(match.group("second") or 0)
    ampm = (match.group("ampm") or "").lower()
    if ampm == "pm" and hour < 12:
        hour += 12
    if ampm == "am" and hour == 12:
        hour = 0
    if hour > 23 or minute > 59 or second >= 60:
        return None
    whole_second = int(second)
    microsecond = int(round((second - whole_second) * 1_000_000))
    return time(hour, minute, whole_second, microsecond)


def _parse_duration_text(text: str) -> Optional[timedelta]:
    match = _DURATION_RE.match(text)
    if not match:
        return None
    sign = -1 if match.group("sign") == "-" else 1
    days = int(match.group("days") or 0)
    if match.group("hours_only") is not None:
        delta = timedelta(days=days, hours=float(match.group("hours_only")))
    else:
        delta = timedelta(
            days=days,
            hours=int(match.group("hours") or 0),
            minutes=int(match.group("minutes") or 0),
            seconds=float(match.group("seconds") or 0),
        )
    return sign * delta


def _interpret_text(
    text: str,
    declared_kind: Optional[FieldKind],
    format_kind: Optional[FieldKind],
    anchor: SerialAnchor,
    number_format: Optional[str],
    timezone_name: Optional[str],
) -> Interpretation:
    target_kind = declared_kind or format_kind

    if target_kind == FieldKind.duration:
        delta = _parse_duration_text(text)
        if delta is not None:
            return Interpretation(
                raw=text,
                kind=FieldKind.duration,
                reading=DurationValue(
                    value=delta,
                    interval=Interval(delta, delta, high_inclusive=True),
                ),
            )
        return Interpretation(
            raw=text, kind=FieldKind.duration, reason="invalid duration"
        )

    if target_kind == FieldKind.time:
        parsed_time = _parse_time_text(text)
        if parsed_time is not None:
            seconds = (
                parsed_time.hour * 3600 + parsed_time.minute * 60 + parsed_time.second
            )
            return Interpretation(
                raw=text,
                kind=FieldKind.time,
                reading=_time_reading_from_seconds(seconds),
            )
        return Interpretation(raw=text, kind=FieldKind.time, reason="invalid time")

    if target_kind == FieldKind.date:
        try:
            parsed_date = date.fromisoformat(text)
        except ValueError:
            return Interpretation(raw=text, kind=FieldKind.date, reason="invalid date")
        start = datetime.combine(parsed_date, time())
        return Interpretation(
            raw=text,
            kind=FieldKind.date,
            reading=DateValue(
                value=parsed_date,
                time=None,
                interval=Interval(start, start + timedelta(days=1)),
            ),
        )

    if target_kind == FieldKind.datetime:
        parsed_dt = None
        try:
            parsed_dt = datetime.fromisoformat(text)
        except ValueError:
            try:
                parsed_date = date.fromisoformat(text)
            except ValueError:
                parsed_date = None
            if parsed_date is not None:
                start = datetime.combine(parsed_date, time())
                return Interpretation(
                    raw=text,
                    kind=FieldKind.datetime,
                    reading=DateValue(
                        value=parsed_date,
                        time=None,
                        interval=Interval(start, start + timedelta(days=1)),
                    ),
                )
        if parsed_dt is not None:
            if parsed_dt.tzinfo is not None:
                parsed_dt = parsed_dt.replace(tzinfo=None)
            return Interpretation(
                raw=text,
                kind=FieldKind.datetime,
                reading=_exact_datetime_reading(
                    anchor, anchor.to_serial(parsed_dt), timezone_name
                ),
            )
        return Interpretation(
            raw=text, kind=FieldKind.datetime, reason="invalid datetime"
        )

    if target_kind == FieldKind.number:
        try:
            number = float(text.replace(",", ""))
        except ValueError:
            return Interpretation(
                raw=text, kind=FieldKind.number, reason="invalid number"
            )
        value: Union[int, float] = int(number) if number.is_integer() else number
        return Interpretation(
            raw=text, kind=FieldKind.number, reading=NumberValue(value)
        )

    # No declared semantics and no usable format. Try the unambiguous textual
    # shapes before falling back. A plain ISO date stays date-only and is not
    # promoted to midnight. The original text remains the raw value.
    for kind, parser in (
        (FieldKind.date, lambda t: date.fromisoformat(t)),
        (FieldKind.datetime, lambda t: datetime.fromisoformat(t)),
        (FieldKind.time, _parse_time_text),
        (FieldKind.duration, _parse_duration_text),
    ):
        try:
            parsed = parser(text)
        except ValueError:
            continue
        if parsed is None:
            continue
        inferred = interpret_value(
            parsed,
            kind=kind,
            anchor=anchor,
            number_format=number_format,
            timezone_name=timezone_name,
        )
        return replace(inferred, raw=text)

    try:
        number = float(text.replace(",", ""))
    except ValueError:
        return Interpretation(
            raw=text, kind=None, reason="cannot infer a date, time, or number"
        )
    inferred = _interpret_numeric(
        number, None, None, anchor, number_format, timezone_name
    )
    return replace(inferred, raw=text)


def interpret_value(
    value: Any,
    kind: Optional[Union[FieldKind, str]] = None,
    anchor: Optional[Union[SerialAnchor, datetime, str]] = None,
    number_format: Optional[str] = None,
    timezone_name: Optional[str] = None,
) -> Interpretation:
    """Interpret one raw cell value.

    :param value: The raw value. Numbers are read as serials, strings as ISO
        dates/times/datetimes or duration text, and ``datetime``/``date``/
        ``time``/``timedelta`` objects are accepted directly.
    :param kind: The field's declared semantic kind, or ``None`` to derive it
        from ``number_format``.
    :param anchor: The confirmed serial epoch.
    :param number_format: The cell's Google Sheets number-format pattern.
    :param timezone_name: The spreadsheet timezone, used only to localize an
        exact datetime. Never defaults to the machine timezone.

    The original value is always preserved in ``raw``. An unparseable value
    returns ``reading=None`` with a ``reason`` instead of guessing.
    """
    declared_kind = FieldKind(kind) if kind is not None else None
    format_kind = classify_number_format(number_format)
    resolved_anchor = serial_anchor(anchor)

    interpretation = _interpret_value(
        value,
        declared_kind,
        format_kind,
        resolved_anchor,
        number_format,
        timezone_name,
    )
    return replace(
        interpretation,
        anchor=resolved_anchor,
        number_format=number_format,
        timezone_name=timezone_name,
    )


def _interpret_value(
    value: Any,
    declared_kind: Optional[FieldKind],
    format_kind: Optional[FieldKind],
    resolved_anchor: SerialAnchor,
    number_format: Optional[str],
    timezone_name: Optional[str],
) -> Interpretation:
    if value is None or value == "":
        return Interpretation(raw=value, kind=declared_kind, reason="empty value")

    if isinstance(value, bool):
        return Interpretation(
            raw=value, kind=declared_kind, reason="boolean is not a date value"
        )

    if _is_number(value):
        return _interpret_numeric(
            value,
            declared_kind,
            format_kind,
            resolved_anchor,
            number_format,
            timezone_name,
        )

    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.replace(tzinfo=None)
        serial = resolved_anchor.to_serial(value)
        target = declared_kind or FieldKind.datetime
        return _interpret_serial_object(
            value, serial, target, resolved_anchor, timezone_name
        )

    if isinstance(value, date):
        if declared_kind not in (None, FieldKind.date, FieldKind.datetime):
            return Interpretation(
                raw=value,
                kind=declared_kind,
                reason="date value cannot be read as {}".format(declared_kind.value),
            )
        start = datetime.combine(value, time())
        return Interpretation(
            raw=value,
            kind=declared_kind or FieldKind.date,
            reading=DateValue(
                value=value,
                time=None,
                interval=Interval(start, start + timedelta(days=1)),
            ),
        )

    if isinstance(value, time):
        seconds = value.hour * 3600 + value.minute * 60 + value.second
        if declared_kind == FieldKind.duration:
            delta = timedelta(
                hours=value.hour,
                minutes=value.minute,
                seconds=value.second,
                microseconds=value.microsecond,
            )
            return Interpretation(
                raw=value,
                kind=FieldKind.duration,
                reading=DurationValue(
                    value=delta,
                    interval=Interval(delta, delta, high_inclusive=True),
                ),
            )
        if declared_kind in (None, FieldKind.time):
            return Interpretation(
                raw=value,
                kind=FieldKind.time,
                reading=_time_reading_from_seconds(seconds),
            )
        return Interpretation(
            raw=value,
            kind=declared_kind,
            reason="time value cannot be read as {}".format(declared_kind.value),
        )

    if isinstance(value, timedelta):
        if declared_kind not in (None, FieldKind.duration):
            return Interpretation(
                raw=value,
                kind=declared_kind,
                reason="duration value cannot be read as {}".format(
                    declared_kind.value
                ),
            )
        return Interpretation(
            raw=value,
            kind=FieldKind.duration,
            reading=DurationValue(
                value=value,
                interval=Interval(value, value, high_inclusive=True),
            ),
        )

    if isinstance(value, str):
        return _interpret_text(
            value.strip(),
            declared_kind,
            format_kind,
            resolved_anchor,
            number_format,
            timezone_name,
        )

    return Interpretation(
        raw=value,
        kind=declared_kind,
        reason="unsupported value type: {}".format(type(value).__name__),
    )


def _interpret_serial_object(
    value: datetime,
    serial: float,
    kind: FieldKind,
    anchor: SerialAnchor,
    timezone_name: Optional[str],
) -> Interpretation:
    if kind == FieldKind.date:
        return Interpretation(
            raw=value,
            kind=kind,
            reading=_date_only_reading(anchor, serial),
        )
    if kind == FieldKind.datetime:
        reading = _exact_datetime_reading(anchor, serial, timezone_name)
        return Interpretation(raw=value, kind=kind, reading=reading)
    if kind == FieldKind.time:
        _, seconds = _split_serial(serial)
        return Interpretation(
            raw=value,
            kind=kind,
            reading=_time_reading_from_seconds(seconds),
        )
    if kind == FieldKind.number:
        return Interpretation(raw=value, kind=kind, reading=NumberValue(serial))
    delta = timedelta(days=serial)
    return Interpretation(
        raw=value,
        kind=kind,
        reading=DurationValue(
            value=delta, interval=Interval(delta, delta, high_inclusive=True)
        ),
    )


def _classify_interpretation(interpretation: Interpretation) -> str:
    if interpretation.reading is None:
        return "invalid"
    if interpretation.ambiguous:
        return "unresolved"
    return "confirmed"


def interpret_column(
    values: Iterable[Any],
    kind: Optional[
        Union[FieldKind, str, Sequence[Optional[Union[FieldKind, str]]]]
    ] = None,
    anchor: Optional[Union[SerialAnchor, datetime, str]] = None,
    number_formats: Optional[Union[str, Sequence[Optional[str]]]] = None,
    timezone_name: Optional[str] = None,
) -> ColumnResult:
    """Interpret a whole column and partition it by certainty.

    ``kind`` and ``number_formats`` may each be a single value applied to the
    whole column or one value per cell. A cell whose declared semantics and
    number format produce readings in different semantic families is placed in
    :attr:`ColumnResult.unresolved`, never assigned to either side of the
    partition.
    """
    value_list = list(values)
    count = len(value_list)

    if kind is None:
        kinds: List[Optional[FieldKind]] = [None] * count
    elif isinstance(kind, (FieldKind, str)):
        kinds = [FieldKind(kind)] * count
    else:
        kinds = [FieldKind(k) if k is not None else None for k in kind]
        if len(kinds) != count:
            raise ValueError("kind sequence must match the values length")

    if number_formats is None:
        formats: List[Optional[str]] = [None] * count
    elif isinstance(number_formats, str):
        formats = [number_formats] * count
    else:
        formats = list(number_formats)
        if len(formats) != count:
            raise ValueError("number_formats sequence must match the values length")

    if timezone_name is not None and ZoneInfo is None:  # pragma: no cover
        raise RuntimeError("zoneinfo is required for timezone support")

    interpretations: List[Interpretation] = []
    confirmed: Dict[int, Interpretation] = {}
    unresolved: Dict[int, Interpretation] = {}
    invalid: Dict[int, Interpretation] = {}

    for index, (value, item_kind, item_format) in enumerate(
        zip(value_list, kinds, formats)
    ):
        interpretation = interpret_value(
            value,
            kind=item_kind,
            anchor=anchor,
            number_format=item_format,
            timezone_name=timezone_name,
        )
        interpretations.append(interpretation)
        status = _classify_interpretation(interpretation)
        if status == "confirmed":
            confirmed[index] = interpretation
        elif status == "unresolved":
            unresolved[index] = interpretation
        else:
            invalid[index] = interpretation

    return ColumnResult(
        values=tuple(value_list),
        interpretations=tuple(interpretations),
        confirmed=confirmed,
        unresolved=unresolved,
        invalid=invalid,
        timezone_name=timezone_name,
        kinds=tuple(kinds),
        number_formats=tuple(formats),
        anchor=serial_anchor(anchor),
    )


def _localize(moment: datetime, zone: Union[str, Any]) -> datetime:
    """Attach a timezone to a naive wall-clock datetime.

    A nonexistent local time in a spring-forward gap is moved forward in whole
    minutes until a valid wall time is found. An ambiguous fall-back time keeps
    the first occurrence. The machine's local timezone is never consulted.
    """
    if isinstance(zone, str):
        if ZoneInfo is None:  # pragma: no cover
            raise RuntimeError("zoneinfo is required for timezone support")
        resolved_zone = ZoneInfo(zone)
    else:
        resolved_zone = zone

    candidate_moment = moment
    for _ in range(24 * 60):
        candidate = candidate_moment.replace(tzinfo=resolved_zone, fold=0)
        round_trip = (
            candidate.astimezone(timezone.utc)
            .astimezone(resolved_zone)
            .replace(tzinfo=None)
        )
        if round_trip == candidate_moment:
            return candidate
        candidate_moment += timedelta(minutes=1)
    raise ValueError(
        "could not resolve local time {!r} in timezone {!r}".format(moment, zone)
    )


def _reinterpret_interpretation(
    interpretation: Interpretation, timezone_name: str
) -> Interpretation:
    """Recompute the timezone-dependent parts of one interpretation."""
    new_reading = _rezone_reading_if_needed(interpretation.reading, timezone_name)
    new_alternatives = tuple(
        _rezone_reading_if_needed(alt, timezone_name)
        for alt in interpretation.alternatives
    )
    if (
        new_reading is interpretation.reading
        and new_alternatives == interpretation.alternatives
    ):
        return interpretation
    return replace(
        interpretation,
        reading=new_reading,
        alternatives=new_alternatives,
        timezone_name=timezone_name,
    )


def _rezone_reading_if_needed(
    reading: Optional[Reading], timezone_name: str
) -> Optional[Reading]:
    """Rezone only an exact datetime reading; leave every other reading alone."""
    if isinstance(reading, DateValue) and reading.time is not None:
        return _rezone_date_value(reading, timezone_name)
    return reading


def _rezone_date_value(reading: DateValue, timezone_name: str) -> DateValue:
    moment = reading.wall_clock
    assert moment is not None
    local = _localize(moment, timezone_name)
    return replace(
        reading,
        interval=Interval(local, local, high_inclusive=True),
        timezone=timezone_name,
    )


def reinterpret_timezone(
    interpretations: Iterable[Interpretation],
    timezone_name: str,
    anchor: Any = None,
) -> Tuple[Interpretation, ...]:
    """Recompute only timezone-dependent interpretations.

    Number, duration, pure-time, and date-only readings are returned by
    identity. Exact datetime readings keep their wall-clock value and gain the
    new timezone and a translated interval. The ``anchor`` argument is
    accepted for symmetry with :func:`interpret_column` but is not needed for
    an already interpreted value.
    """
    del anchor
    if not timezone_name:
        raise ValueError("timezone_name is required")
    return tuple(
        _reinterpret_interpretation(item, timezone_name) for item in interpretations
    )


def to_instant(
    reading: Union[DateValue, TimeValue, DurationValue, NumberValue, Interpretation],
    timezone_name: Optional[str] = None,
) -> Optional[datetime]:
    """Convert a reading to an absolute instant.

    Only a :class:`DateValue` with a time component denotes an instant. A
    date-only value, a pure time, a duration, or a number returns ``None``
    because none of them identifies a unique moment. The timezone must come
    from the spreadsheet or be passed explicitly; the machine's local timezone
    is never used.
    """
    if isinstance(reading, Interpretation):
        if reading.reading is None:
            return None
        reading = reading.reading
    if not isinstance(reading, DateValue) or reading.time is None:
        return None

    zone = timezone_name or reading.timezone
    if zone is None:
        raise ValueError(
            "an explicit timezone is required to resolve a datetime; "
            "the machine timezone is never used"
        )
    moment = reading.wall_clock
    assert moment is not None
    return _localize(moment, zone)


def timezone_offset(
    serial: Union[int, float],
    timezone_name: str,
    anchor: Optional[Union[SerialAnchor, datetime, str]] = None,
) -> timedelta:
    """Return the UTC offset that applies at ``serial`` in ``timezone_name``."""
    resolved = serial_anchor(anchor)
    moment = resolved.to_datetime(serial)
    return _localize(moment, timezone_name).utcoffset() or timedelta(0)


__all__ = [
    "ColumnResult",
    "DateValue",
    "DurationValue",
    "FieldKind",
    "GOOGLE_SHEETS_EPOCH",
    "Interval",
    "Interpretation",
    "MAC_EPOCH",
    "NumberValue",
    "SECONDS_PER_DAY",
    "SerialAnchor",
    "TimeValue",
    "classify_number_format",
    "interpret_column",
    "interpret_value",
    "reinterpret_timezone",
    "serial_anchor",
    "timezone_offset",
    "to_instant",
]
