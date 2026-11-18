from __future__ import annotations

import datetime
import math
import re

_one_or_two_digits = "([ ]?\\d{1,2})"
_separator = "[.:'\\-\\/ ]"
DATE_OR_TIME_PATTERN = re.compile(
    f"""
        {_one_or_two_digits}  # day/hour
        {_separator}
        {_one_or_two_digits}  # month/minute
        {_separator}
        {_one_or_two_digits}  # year/second
        """,
    re.VERBOSE,
)


def encode_str(value: str, length: int) -> bytes:
    if len(value) > length:
        raise ValueError(
            f"{value!r} exceeds maximum field length: {len(value)} > {length}"
        )
    if not value.isprintable():
        raise ValueError(f"{value} contains non-printable characters")
    return value.encode("ascii").ljust(length)


def decode_str(field: bytes, encoding: str = "ascii") -> str:
    return field.decode(encoding=encoding, errors="replace").rstrip()


def encode_int(value: int, length: int) -> bytes:
    return encode_str(str(value), length)


def encode_float(value: float) -> bytes:
    if float(value).is_integer():
        value = int(value)
    return encode_str(str(value), 8)


def decode_float(field: bytes) -> float:
    value = float(decode_str(field))
    if math.isinf(value):
        raise ValueError(f"Field value is outside float range: {decode_str(field)}")
    return value


def _century_start_year(century: int) -> int:
    if isinstance(century, bool) or not isinstance(century, int):
        raise TypeError("century must be an integer or None")
    if 0 <= century < 100:  # noqa: PLR2004
        return century * 100
    if 100 <= century <= 9900 and century % 100 == 0:  # noqa: PLR2004
        return century
    raise ValueError(
        "century must be a two-digit century number (0..99) or a century start "
        "year (100..9900, divisible by 100)"
    )


def decode_date(field: bytes, *, century: int | None = None) -> datetime.date:
    """
    Decode a legacy `DD.MM.YY` date field.

    Parameters
    ----------
    field : bytes
        Raw header field.
    century : int | None, default: None
        Century basis for the two-digit year. Either a two-digit century number
        (`19` for 1900-1999) or its start year (`1900`). If `None`, the historical
        EDF rule is used (85..99 map to 1985..1999, 00..84 map to 2000..2084).
        The current year is never used.
    """
    date = decode_str(field)
    match = DATE_OR_TIME_PATTERN.fullmatch(date)
    if match is None:
        raise ValueError(f"Invalid date for format DD.MM.YY: {date!r}")
    day, month, year = (int(g) for g in match.groups())
    if century is None:
        year += 1900 if year >= 85 else 2000  # noqa: PLR2004
    else:
        year = _century_start_year(century) + year
    return datetime.date(year, month, day)


def encode_date(value: datetime.date) -> bytes:
    if not 1985 <= value.year <= 2084:  # noqa: PLR2004
        raise ValueError("EDF only allows dates from 1985 to 2084")
    return encode_str(value.strftime("%d.%m.%y"), 8)


def decode_time(field: bytes) -> datetime.time:
    time = decode_str(field)
    match = DATE_OR_TIME_PATTERN.fullmatch(time)
    if match is None:
        raise ValueError(f"Invalid time for format hh.mm.ss: {time!r}")
    hours, minutes, seconds = (int(g) for g in match.groups())
    return datetime.time(hours, minutes, seconds)


def encode_time(value: datetime.time) -> bytes:
    return encode_str(value.isoformat().replace(":", "."), 8)
