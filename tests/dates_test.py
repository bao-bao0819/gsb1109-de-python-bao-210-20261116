import unittest
from datetime import date, datetime, time, timedelta

from gspread import dates


class NumberFormatTest(unittest.TestCase):
    def test_date_formats(self):
        self.assertEqual(
            dates.classify_number_format("yyyy-mm-dd"), dates.FieldKind.date
        )
        self.assertEqual(dates.classify_number_format("m/d/yyyy"), dates.FieldKind.date)
        self.assertEqual(dates.classify_number_format("mm"), dates.FieldKind.date)

    def test_time_formats(self):
        self.assertEqual(dates.classify_number_format("hh:mm"), dates.FieldKind.time)
        self.assertEqual(
            dates.classify_number_format("h:mm:ss am/pm"), dates.FieldKind.time
        )

    def test_datetime_formats(self):
        self.assertEqual(
            dates.classify_number_format("yyyy-mm-dd hh:mm"),
            dates.FieldKind.datetime,
        )

    def test_duration_formats(self):
        self.assertEqual(
            dates.classify_number_format("[h]:mm:ss"), dates.FieldKind.duration
        )
        self.assertEqual(
            dates.classify_number_format("[mm]:ss"), dates.FieldKind.duration
        )

    def test_numeric_formats(self):
        self.assertEqual(
            dates.classify_number_format("General"), dates.FieldKind.number
        )
        self.assertEqual(dates.classify_number_format("0.00"), dates.FieldKind.number)
        self.assertEqual(
            dates.classify_number_format("#,##0.00;[Red]-#,##0.00"),
            dates.FieldKind.number,
        )

    def test_unknown_formats_are_not_numbers(self):
        self.assertIsNone(dates.classify_number_format(None))
        self.assertIsNone(dates.classify_number_format(""))
        self.assertIsNone(dates.classify_number_format("   "))

    def test_literals_and_colors_are_ignored(self):
        self.assertEqual(
            dates.classify_number_format('yyyy-mm-dd"T"hh:mm'),
            dates.FieldKind.datetime,
        )


class SerialAnchorTest(unittest.TestCase):
    def test_google_anchor(self):
        anchor = dates.SerialAnchor.google_sheets()
        self.assertEqual(anchor.to_datetime(43831), datetime(2020, 1, 1))
        self.assertEqual(anchor.to_serial(datetime(2020, 1, 1)), 43831)

    def test_mac_anchor(self):
        anchor = dates.SerialAnchor.mac()
        self.assertEqual(anchor.to_datetime(1), datetime(1904, 1, 2))

    def test_string_anchors(self):
        self.assertEqual(dates.serial_anchor("1904").name, "mac")
        self.assertEqual(dates.serial_anchor("google").name, "google_sheets")
        self.assertEqual(dates.serial_anchor(None).name, "google_sheets")

    def test_timezone_on_anchor_is_discarded(self):
        anchor = dates.SerialAnchor(datetime(2020, 1, 1, tzinfo=dates.timezone.utc))
        self.assertIsNone(anchor.epoch.tzinfo)

    def test_invalid_anchor(self):
        with self.assertRaises(ValueError):
            dates.serial_anchor("nope")


class DateInterpretationTest(unittest.TestCase):
    def test_date_only_is_not_midnight(self):
        result = dates.interpret_value(43831, kind="date", number_format="yyyy-mm-dd")
        self.assertIsInstance(result.reading, dates.DateValue)
        self.assertEqual(result.reading.value, date(2020, 1, 1))
        self.assertIsNone(result.reading.time)
        self.assertIsNone(result.reading.datetime)
        self.assertEqual(
            result.reading.interval,
            dates.Interval(datetime(2020, 1, 1), datetime(2020, 1, 2)),
        )

    def test_datetime_keeps_time(self):
        result = dates.interpret_value(
            43831.5, kind="datetime", number_format="yyyy-mm-dd hh:mm"
        )
        self.assertEqual(result.reading.value, date(2020, 1, 1))
        self.assertEqual(result.reading.time, time(12, 0))
        self.assertEqual(result.reading.datetime, datetime(2020, 1, 1, 12, 0))

    def test_integer_datetime_is_date_only(self):
        result = dates.interpret_value(
            43831, kind="datetime", number_format="yyyy-mm-dd"
        )
        self.assertIsNone(result.reading.time)
        self.assertIsNone(result.reading.datetime)

    def test_fractional_date_is_ambiguous(self):
        result = dates.interpret_value(43831.5, kind="date", number_format="yyyy-mm-dd")
        self.assertTrue(result.ambiguous)
        self.assertEqual(result.families, frozenset({"date", "datetime"}))
        self.assertEqual(len(result.all_readings), 2)

    def test_integer_date_is_confirmed(self):
        result = dates.interpret_value(43831, kind="date", number_format="yyyy-mm-dd")
        self.assertFalse(result.ambiguous)
        self.assertTrue(result.resolved)

    def test_mac_anchor_changes_date(self):
        result = dates.interpret_value(
            43831, kind="date", number_format="yyyy-mm-dd", anchor="1904"
        )
        self.assertEqual(result.reading.value, date(2024, 1, 2))

    def test_invalid_date_text(self):
        result = dates.interpret_value("not a date", kind="date")
        self.assertIsNone(result.reading)
        self.assertEqual(result.reason, "invalid date")


class TimeInterpretationTest(unittest.TestCase):
    def test_serial_time(self):
        result = dates.interpret_value(0.75, kind="time", number_format="hh:mm")
        self.assertEqual(result.reading.value, time(18, 0))

    def test_serial_time_ignores_whole_days(self):
        result = dates.interpret_value(43831.75, kind="time", number_format="hh:mm")
        self.assertEqual(result.reading.value, time(18, 0))

    def test_text_time(self):
        for text, expected in (
            ("12:30", time(12, 30)),
            ("12:30 PM", time(12, 30)),
            ("12:30 AM", time(0, 30)),
            ("18:05:09", time(18, 5, 9)),
        ):
            result = dates.interpret_value(text, kind="time")
            self.assertEqual(result.reading.value, expected)

    def test_text_time_is_never_a_midnight_datetime(self):
        result = dates.interpret_value("12:30", kind="time")
        self.assertIsInstance(result.reading, dates.TimeValue)

    def test_whole_day_time_serial_is_ambiguous_without_format(self):
        result = dates.interpret_value(1.5, kind="time")
        self.assertTrue(result.ambiguous)
        self.assertEqual(result.families, frozenset({"time", "duration"}))

    def test_whole_day_time_serial_is_confirmed_with_time_format(self):
        result = dates.interpret_value(1.5, kind="time", number_format="hh:mm")
        self.assertFalse(result.ambiguous)

    def test_invalid_time_text(self):
        result = dates.interpret_value("25:99", kind="time")
        self.assertIsNone(result.reading)
        self.assertEqual(result.reason, "invalid time")


class DurationInterpretationTest(unittest.TestCase):
    def test_serial_duration(self):
        result = dates.interpret_value(1.5, kind="duration", number_format="[h]:mm")
        self.assertEqual(result.reading.value, timedelta(days=1, hours=12))

    def test_text_duration(self):
        for text, expected in (
            ("1:30", timedelta(hours=1, minutes=30)),
            ("1:30:15", timedelta(hours=1, minutes=30, seconds=15)),
            ("1 day 2:00", timedelta(days=1, hours=2)),
            ("2.5 hours", timedelta(hours=2, minutes=30)),
            ("-1:00", timedelta(hours=-1)),
        ):
            result = dates.interpret_value(text, kind="duration")
            self.assertEqual(result.reading.value, expected)

    def test_invalid_duration(self):
        result = dates.interpret_value("nope", kind="duration")
        self.assertIsNone(result.reading)
        self.assertEqual(result.reason, "invalid duration")


class NumberInterpretationTest(unittest.TestCase):
    def test_plain_number(self):
        result = dates.interpret_value(43831, kind="number", number_format="0")
        self.assertEqual(result.reading, dates.NumberValue(43831))
        self.assertEqual(result.raw, 43831)

    def test_integer_raw_is_preserved(self):
        result = dates.interpret_value(42, kind="number", number_format="0")
        self.assertIsInstance(result.raw, int)
        self.assertIsInstance(result.reading.value, int)

    def test_untyped_number_is_ambiguous(self):
        result = dates.interpret_value(43831)
        self.assertTrue(result.ambiguous)
        self.assertEqual(result.families, frozenset({"number", "date"}))

    def test_text_number(self):
        result = dates.interpret_value("2,000.50", kind="number")
        self.assertEqual(result.reading.value, 2000.5)

    def test_empty_value(self):
        result = dates.interpret_value("", kind="number")
        self.assertIsNone(result.reading)
        self.assertEqual(result.reason, "empty value")


class ColumnTest(unittest.TestCase):
    def test_partition(self):
        column = dates.interpret_column(
            [43831, 3.5, 10, "hello"],
            number_formats=[None, None, "yyyy-mm-dd", None],
        )
        self.assertEqual(set(column.confirmed), {2})
        self.assertEqual(set(column.unresolved), {0, 1})
        self.assertEqual(set(column.invalid), {3})
        self.assertEqual(column.values, (43831, 3.5, 10, "hello"))

    def test_per_cell_kinds(self):
        column = dates.interpret_column(
            [43831, 0.5, 2.5],
            kind=["date", "time", "duration"],
            number_formats=["yyyy-mm-dd", "hh:mm", "[h]:mm"],
        )
        self.assertEqual(set(column.confirmed), {0, 1, 2})
        self.assertIsNone(column.readings()[0].time)
        self.assertEqual(column.readings()[1].value, time(12, 0))
        self.assertEqual(column.readings()[2].value, timedelta(days=2, hours=12))

    def test_kind_length_mismatch(self):
        with self.assertRaises(ValueError):
            dates.interpret_column([1, 2], kind=["date"])

    def test_format_length_mismatch(self):
        with self.assertRaises(ValueError):
            dates.interpret_column([1, 2], number_formats=["0"])

    def test_format_change_reinterprets(self):
        column = dates.interpret_column([43831, 43831.5], number_formats=[None, None])
        self.assertEqual(set(column.unresolved), {0, 1})

        changed = column.with_number_formats(["yyyy-mm-dd", "yyyy-mm-dd hh:mm"])
        self.assertEqual(set(changed.confirmed), {0, 1})
        self.assertEqual(changed.readings()[0].value, date(2020, 1, 1))
        self.assertEqual(changed.readings()[1].time, time(12, 0))

    def test_format_change_keeps_original_values(self):
        column = dates.interpret_column([43831], number_formats=["0"])
        changed = column.with_number_formats(["yyyy-mm-dd"])
        self.assertEqual(changed.values, (43831,))
        self.assertEqual(changed.interpretations[0].raw, 43831)

    def test_format_change_keeps_anchor(self):
        column = dates.interpret_column([43831], anchor="1904")
        changed = column.with_number_formats(["yyyy-mm-dd"])
        self.assertEqual(changed.anchor.name, "mac")
        self.assertEqual(changed.readings()[0].value, date(2024, 1, 2))

    def test_fractional_untyped_serial_offers_datetime(self):
        column = dates.interpret_column([43831.5], number_formats=[None])
        alternative = column.interpretations[0].alternatives[0]
        self.assertIsInstance(alternative, dates.DateValue)
        self.assertEqual(alternative.time, time(12, 0))


class TimezoneTest(unittest.TestCase):
    def test_only_datetime_readings_change(self):
        column = dates.interpret_column(
            [43831, 43831.5, 1.5, 0.5],
            kind=["number", "datetime", "duration", "time"],
            number_formats=["0", "yyyy-mm-dd hh:mm", "[h]:mm", "hh:mm"],
        )
        before = column.readings()
        after = column.with_timezone("America/New_York").readings()

        self.assertIs(after[0], before[0])
        self.assertIs(after[2], before[2])
        self.assertIs(after[3], before[3])
        self.assertIsNot(after[1], before[1])
        self.assertEqual(after[1].timezone, "America/New_York")
        self.assertEqual(after[1].datetime.utcoffset(), timedelta(hours=-5))

    def test_alternative_datetime_is_rezoned(self):
        column = dates.interpret_column([43831.5], kind="date")
        interpretation = column.interpretations[0]
        self.assertTrue(interpretation.ambiguous)
        self.assertIsNone(interpretation.reading.time)

        after = column.with_timezone("America/New_York").interpretations[0]
        self.assertIs(after.reading, interpretation.reading)
        alternative = after.alternatives[0]
        self.assertEqual(alternative.timezone, "America/New_York")
        self.assertEqual(alternative.datetime.utcoffset(), timedelta(hours=-5))

    def test_number_with_datetime_alternative_stays_a_number(self):
        column = dates.interpret_column([43831.5], number_formats=[None])
        before = column.readings()[0]
        after = column.with_timezone("America/New_York").readings()[0]
        self.assertIs(before, after)
        self.assertEqual(after, dates.NumberValue(43831.5))

    def test_date_only_reading_does_not_change(self):
        column = dates.interpret_column(
            [43831], kind="date", number_formats="yyyy-mm-dd"
        )
        after = column.with_timezone("Asia/Tokyo")
        self.assertIs(after.readings()[0], column.readings()[0])
        self.assertIsNone(after.readings()[0].time)

    def test_machine_timezone_is_never_used(self):
        result = dates.interpret_value(
            43831.5, kind="datetime", number_format="yyyy-mm-dd hh:mm"
        )
        with self.assertRaises(ValueError):
            dates.to_instant(result)

    def test_to_instant_requires_explicit_zone(self):
        result = dates.interpret_value(
            43831.5,
            kind="datetime",
            number_format="yyyy-mm-dd hh:mm",
            timezone_name="UTC",
        )
        instant = dates.to_instant(result)
        self.assertEqual(instant, datetime(2020, 1, 1, 12, tzinfo=dates.timezone.utc))

    def test_date_only_has_no_instant(self):
        result = dates.interpret_value(43831, kind="date", number_format="yyyy-mm-dd")
        self.assertIsNone(dates.to_instant(result, "UTC"))

    def test_duration_has_no_instant(self):
        result = dates.interpret_value(1.5, kind="duration", number_format="[h]:mm")
        self.assertIsNone(dates.to_instant(result, "UTC"))

    def test_dst_spring_forward_gap_is_resolved(self):
        serial = dates.SerialAnchor.google_sheets().to_serial(
            datetime(2021, 3, 14, 2, 30)
        )
        result = dates.interpret_value(
            serial,
            kind="datetime",
            number_format="yyyy-mm-dd hh:mm",
            timezone_name="America/New_York",
        )
        self.assertEqual(
            result.reading.datetime,
            datetime(2021, 3, 14, 3, 0, tzinfo=dates.ZoneInfo("America/New_York")),
        )

    def test_timezone_offset(self):
        self.assertEqual(
            dates.timezone_offset(43831, "America/New_York"),
            timedelta(hours=-5),
        )

    def test_timezone_requires_name(self):
        column = dates.interpret_column([43831], kind="date")
        with self.assertRaises(ValueError):
            column.with_timezone("")
