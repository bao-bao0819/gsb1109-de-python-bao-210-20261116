from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pytest

from edfio import (
    DateSource,
    DateStatus,
    Edf,
    EdfAnnotation,
    EdfSignal,
    Recording,
    TimeStatus,
    interpret_time,
    read_edf,
)
from edfio._header_field import decode_date
from tests import TEST_DATA_DIR

EDF_FILE = TEST_DATA_DIR / "short_psg.edf"


def _interpret(
    *,
    legacy: bytes = b"17.11.99",
    recording: str = "Startdate 17-NOV-2099 X X X",
    starttime: bytes = b"23.59.55",
    subsecond_offset: float = 0,
    duration: float = 60,
    century: int | None = None,
):
    return interpret_time(
        legacy_startdate=legacy,
        local_recording_identification=recording,
        header_starttime=starttime,
        subsecond_offset=subsecond_offset,
        duration=duration,
        century=century,
    )


def test_short_legacy_year_is_pinned_by_explicit_edfplus_date():
    result = _interpret()
    assert result.status is TimeStatus.RESOLVED
    assert result.start == datetime.datetime(2099, 11, 17, 23, 59, 55)
    legacy = result.get_source(DateSource.LEGACY_HEADER)
    edfplus = result.get_source(DateSource.EDFPLUS_RECORDING)
    assert legacy.raw == "17.11.99"
    assert legacy.status is DateStatus.AMBIGUOUS
    assert legacy.candidate_years == (1999, 2099)
    assert edfplus.raw == "17-NOV-2099"
    assert edfplus.status is DateStatus.EXACT


def test_caller_century_resolves_short_date_without_using_today():
    result = _interpret(recording="Startdate X X X X", century=19)
    assert result.status is TimeStatus.RESOLVED
    assert result.start == datetime.datetime(1999, 11, 17, 23, 59, 55)
    assert (
        result.get_source(DateSource.EDFPLUS_RECORDING).status is DateStatus.ANONYMIZED
    )

    result = _interpret(
        legacy=b"17.11.99",
        recording="Startdate 17-NOV-1999 X X X",
        century=19,
    )
    assert result.status is TimeStatus.RESOLVED
    assert result.start == datetime.datetime(1999, 11, 17, 23, 59, 55)


def test_century_must_be_explicit_and_never_defaults_to_current_year():
    result = _interpret(recording="Startdate X X X X")
    assert result.status is TimeStatus.AMBIGUOUS
    assert result.start is None
    assert result.determinable_start_range is not None

    result = _interpret(recording="Startdate 01-JAN-2000 X X X", legacy=b"01.01.00")
    assert result.status is TimeStatus.RESOLVED
    assert result.start == datetime.datetime(2000, 1, 1, 23, 59, 55)


def test_century_values_are_validated():
    with pytest.raises(TypeError, match="century must be an integer"):
        _interpret(recording="Startdate X X X X", century=19.0)
    with pytest.raises(TypeError, match="century must be an integer"):
        _interpret(recording="Startdate X X X X", century=True)
    with pytest.raises(ValueError, match="century must be"):
        _interpret(recording="Startdate X X X X", century=-1)
    with pytest.raises(ValueError, match="century must be"):
        _interpret(recording="Startdate X X X X", century=1901)


def test_decode_date_accepts_century_basis():
    assert decode_date(b"17.11.99") == datetime.date(1999, 11, 17)
    assert decode_date(b"17.11.99", century=20) == datetime.date(2099, 11, 17)
    assert decode_date(b"17.11.99", century=2000) == datetime.date(2099, 11, 17)
    assert decode_date(b"17.11.84") == datetime.date(2084, 11, 17)
    assert decode_date(b"17.11.84", century=19) == datetime.date(1984, 11, 17)


def test_anonymized_edfplus_date_keeps_legacy_source_range():
    result = _interpret(recording="Startdate X X X X")
    assert result.status is TimeStatus.AMBIGUOUS
    assert result.start is None
    assert result.determinable_start_range == (
        datetime.datetime(1999, 11, 17, 23, 59, 55),
        datetime.datetime(2099, 11, 17, 23, 59, 55),
    )
    legacy = result.get_source(DateSource.LEGACY_HEADER)
    edfplus = result.get_source(DateSource.EDFPLUS_RECORDING)
    assert legacy.status is DateStatus.AMBIGUOUS
    assert edfplus.status is DateStatus.ANONYMIZED
    assert "anonymized" in (result.detail or "")


def test_anonymization_placeholder_is_not_reported_as_real_date():
    result = _interpret(legacy=b"01.01.85", recording="Startdate X X X X")
    legacy = result.get_source(DateSource.LEGACY_HEADER)
    assert legacy.status is DateStatus.PLACEHOLDER
    assert result.status is TimeStatus.ANONYMIZED
    assert result.start is None


def test_anonymized_edfplus_date_uses_other_usable_source():
    result = _interpret(legacy=b"17.11.99", recording="Startdate X X X X")
    legacy = result.get_source(DateSource.LEGACY_HEADER)
    assert legacy.candidate_years == (1999, 2099)
    assert result.status is TimeStatus.AMBIGUOUS
    assert result.determinable_start_range is not None


def test_missing_or_invalid_edfplus_field_uses_legacy_date():
    result = _interpret(recording="not edfplus", legacy=b"17.11.99")
    assert result.get_source(DateSource.EDFPLUS_RECORDING).status is DateStatus.MISSING
    assert result.status is TimeStatus.AMBIGUOUS

    result = _interpret(recording="Startdate 17-FOO-1999 X X X")
    assert result.get_source(DateSource.EDFPLUS_RECORDING).status is DateStatus.INVALID
    assert result.status is TimeStatus.AMBIGUOUS

    result = _interpret(
        recording="Startdate 17-FOO-1999 X X X", legacy=b"17.11.99", century=20
    )
    assert result.status is TimeStatus.RESOLVED
    assert result.start == datetime.datetime(2099, 11, 17, 23, 59, 55)


def test_differing_exact_dates_are_reported_as_conflict():
    result = _interpret(recording="Startdate 17-NOV-1998 X X X", legacy=b"17.11.99")
    assert result.status is TimeStatus.CONFLICT
    assert result.start is None
    assert result.determinable_range is None
    assert "disagree" in (result.detail or "")


def test_short_year_range_is_computed_when_no_other_source_exists():
    result = _interpret(
        legacy=b"17.11.99",
        recording="Startdate X X X X",
        century=None,
    )
    legacy = result.get_source(DateSource.LEGACY_HEADER)
    assert legacy.earliest == datetime.date(1999, 11, 17)
    assert legacy.latest == datetime.date(2099, 11, 17)
    assert legacy.candidate_dates == (
        datetime.date(1999, 11, 17),
        datetime.date(2099, 11, 17),
    )
    assert result.candidate_dates == (
        datetime.date(1999, 11, 17),
        datetime.date(2099, 11, 17),
    )
    assert result.determinable_date_range == (
        datetime.date(1999, 11, 17),
        datetime.date(2099, 11, 17),
    )
    assert result.has_anonymized_source
    assert set(result.by_source) == {
        DateSource.LEGACY_HEADER,
        DateSource.EDFPLUS_RECORDING,
    }
    assert result.datetime_range_at(60) == (
        datetime.datetime(1999, 11, 18, 0, 0, 55),
        datetime.datetime(2099, 11, 18, 0, 0, 55),
    )
    assert result.determinable_range == (
        datetime.datetime(1999, 11, 17, 23, 59, 55),
        datetime.datetime(2099, 11, 18, 0, 0, 55),
    )


def test_source_without_date_has_no_candidate_range():
    result = _interpret(recording="Startdate X X X X")
    edfplus = result.get_source(DateSource.EDFPLUS_RECORDING)
    assert edfplus.candidate_dates == ()
    assert edfplus.earliest is None
    assert edfplus.latest is None


def test_exact_date_has_single_candidate_and_conflict_has_none():
    exact = _interpret()
    assert exact.candidate_dates == (datetime.date(2099, 11, 17),)
    assert exact.determinable_date_range == (
        datetime.date(2099, 11, 17),
        datetime.date(2099, 11, 17),
    )
    assert not exact.has_anonymized_source
    conflict = _interpret(recording="Startdate 17-NOV-1998 X X X")
    assert conflict.candidate_dates == ()
    assert conflict.determinable_date_range is None
    assert conflict.datetime_range_at(0) is None


def test_raw_starttime_and_subsecond_offset_are_retained():
    result = _interpret(starttime=b"23.59.55", subsecond_offset=2.5)
    assert result.raw_starttime == datetime.time(23, 59, 55)
    assert result.subsecond_offset == 2.5
    assert result.starttime == datetime.time(23, 59, 57, 500000)


def test_subsecond_offset_rolls_into_next_day():
    result = _interpret(
        starttime=b"23.59.55",
        subsecond_offset=10,
        duration=120,
    )
    assert result.start == datetime.datetime(2099, 11, 18, 0, 0, 5)
    assert result.starttime == datetime.time(0, 0, 5)
    assert result.datetime_at(0) == result.start
    assert result.datetime_at(120) == datetime.datetime(2099, 11, 18, 0, 2, 5)


def test_datetime_at_advances_past_midnight_without_reinterpreting_clock():
    result = _interpret(duration=26 * 3600)
    assert result.datetime_at(3600) == datetime.datetime(2099, 11, 18, 0, 59, 55)
    assert result.datetime_at(24 * 3600) == datetime.datetime(2099, 11, 18, 23, 59, 55)
    assert result.datetime_at(24 * 3600 + 60) == datetime.datetime(
        2099, 11, 19, 0, 0, 55
    )


def test_edf_interpret_time_uses_raw_sources():
    edf = read_edf(EDF_FILE)
    result = edf.interpret_time()
    assert result.status is TimeStatus.RESOLVED
    assert result.start == datetime.datetime(1989, 4, 24, 16, 13)
    assert result.get_source(DateSource.LEGACY_HEADER).raw == "24.04.89"
    assert result.get_source(DateSource.EDFPLUS_RECORDING).raw == "24-APR-1989"


def test_edf_interpret_time_does_not_use_current_year_for_short_fields():
    edf = Edf([EdfSignal(np.arange(10), 1)])
    edf._startdate = b"17.11.99"
    edf._local_recording_identification = b"Startdate X X X X".ljust(80)
    result = edf.interpret_time()
    assert result.status is TimeStatus.AMBIGUOUS
    assert result.start is None
    assert edf.datetime_at(0) is None
    assert edf.interpret_time(century=20).start == datetime.datetime(2099, 11, 17, 0, 0)


def test_edf_datetime_at_and_annotation_datetimes():
    edf = Edf(
        [EdfSignal(np.arange(10), 1)],
        recording=Recording(startdate=datetime.date(2026, 7, 14)),
        starttime=datetime.time(23, 59, 55),
        annotations=(EdfAnnotation(10, None, "after midnight"),),
    )
    assert edf.datetime_at(10) == datetime.datetime(2026, 7, 15, 0, 0, 5)
    pair = edf.annotation_datetimes()[0]
    assert pair[0].text == "after midnight"
    assert pair[1] == datetime.datetime(2026, 7, 15, 0, 0, 5)


def test_edf_data_record_datetimes_advance_across_midnight():
    edf = Edf(
        [EdfSignal(np.arange(3), 1)],
        recording=Recording(startdate=datetime.date(2026, 7, 14)),
        starttime=datetime.time(23, 59, 58),
        data_record_duration=1,
    )
    datetimes = edf.data_record_datetimes()
    assert datetimes == (
        datetime.datetime(2026, 7, 14, 23, 59, 58),
        datetime.datetime(2026, 7, 14, 23, 59, 59),
        datetime.datetime(2026, 7, 15, 0, 0, 0),
    )


def test_edf_data_record_datetimes_are_none_when_date_is_unknown():
    edf = Edf([EdfSignal(np.arange(2), 1)])
    assert edf.data_record_datetimes() == (None, None)
    assert edf.datetime_at(0) is None


def test_edf_data_record_datetimes_use_timekeeping_onsets_for_short_dates():
    edf = Edf(
        [EdfSignal(np.arange(2), 1)],
        recording=Recording(startdate=datetime.date(2026, 7, 14)),
        starttime=datetime.time(12, 0, 0),
        annotations=(EdfAnnotation(1, None, "x"),),
    )
    edf._local_recording_identification = b"Startdate 14-JUL-26 X X X".ljust(80)
    assert edf.interpret_time().status is TimeStatus.AMBIGUOUS
    assert edf.data_record_datetimes() == (None, None)
    revised = edf.data_record_datetimes(century=20)
    assert revised[0] == datetime.datetime(2026, 7, 14, 12, 0, 0)
    assert revised[1] == datetime.datetime(2026, 7, 14, 12, 0, 1)


def test_interpret_time_available_to_lazy_and_written_files(tmp_file: Path):
    edf = Edf(
        [EdfSignal(np.arange(10), 1)],
        recording=Recording(startdate=datetime.date(1999, 12, 31)),
        starttime=datetime.time(23, 59, 59),
        annotations=(EdfAnnotation(2, None, "x"),),
    )
    edf.write(tmp_file)
    read = read_edf(tmp_file)
    assert read.interpret_time().start == datetime.datetime(1999, 12, 31, 23, 59, 59)
    assert read.annotation_datetimes()[0][1] == datetime.datetime(2000, 1, 1, 0, 0, 1)


def test_exact_source_earliest_and_latest_are_the_value():
    result = _interpret()
    edfplus = result.get_source(DateSource.EDFPLUS_RECORDING)
    assert edfplus.earliest == datetime.date(2099, 11, 17)
    assert edfplus.latest == datetime.date(2099, 11, 17)
    assert len(result.by_source) == 2


def test_exact_start_determinable_range_is_recording_interval():
    result = _interpret()
    assert result.determinable_range == (
        datetime.datetime(2099, 11, 17, 23, 59, 55),
        datetime.datetime(2099, 11, 18, 0, 0, 55),
    )


def test_invalid_legacy_dates_are_reported():
    no_match = _interpret(legacy=b"not-a-date")
    assert no_match.get_source(DateSource.LEGACY_HEADER).status is DateStatus.INVALID
    assert "Invalid date" in (
        no_match.get_source(DateSource.LEGACY_HEADER).detail or ""
    )

    impossible = _interpret(legacy=b"31.02.00")
    assert impossible.get_source(DateSource.LEGACY_HEADER).status is DateStatus.INVALID


def test_no_usable_source_is_unavailable():
    result = _interpret(legacy=b"not-a-date", recording="not edfplus")
    assert result.status is TimeStatus.UNAVAILABLE
    assert result.start is None
    assert result.determinable_range is None
    assert "Invalid date" in (result.detail or "")
    assert "does not follow" in (result.detail or "")


def test_short_year_invalid_in_one_century_is_kept_in_the_other():
    result = _interpret(legacy=b"29.02.00", recording="Startdate X X X X")
    legacy = result.get_source(DateSource.LEGACY_HEADER)
    assert legacy.status is DateStatus.EXACT
    assert legacy.value == datetime.date(2000, 2, 29)


def test_invalid_edfplus_dates_are_reported():
    invalid_day = _interpret(recording="Startdate 31-FEB-2000 X X X")
    assert (
        invalid_day.get_source(DateSource.EDFPLUS_RECORDING).status
        is DateStatus.INVALID
    )

    non_numeric_short_year = _interpret(recording="Startdate 01-JAN-XX X X X")
    assert (
        non_numeric_short_year.get_source(DateSource.EDFPLUS_RECORDING).status
        is DateStatus.INVALID
    )

    impossible_short = _interpret(recording="Startdate 31-FEB-00 X X X")
    assert (
        impossible_short.get_source(DateSource.EDFPLUS_RECORDING).status
        is DateStatus.INVALID
    )


def test_differing_exact_dates_with_century_conflict():
    result = _interpret(
        legacy=b"17.11.98",
        recording="Startdate 17-NOV-1999 X X X",
        century=19,
    )
    assert result.status is TimeStatus.CONFLICT
    assert result.start is None


def test_invalid_starttime_is_unavailable():
    result = _interpret(starttime=b"not-a-time")
    assert result.status is TimeStatus.UNAVAILABLE
    assert result.starttime is None
    assert "Invalid starttime" in (result.detail or "")
