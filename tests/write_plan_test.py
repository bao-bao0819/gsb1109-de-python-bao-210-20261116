import unittest

from gspread.exceptions import InvalidInputValue
from gspread.utils import ValueInputOption, ValueRenderOption
from gspread.write_plan import (
    Expectation,
    PlannedCell,
    apply_write_plan,
)


class FakeWorksheet:
    """Duck-typed worksheet recording writes and replaying canned reads."""

    def __init__(self, read_back):
        # read_back: dict mapping A1 label -> stored effective value
        self.read_back = read_back
        self.writes = []

    def batch_update(self, data, value_input_option=None, **kwargs):
        self.writes.append((value_input_option, data))
        return {"replies": [{} for _ in data]}

    def batch_get(self, ranges, value_render_option=None, **kwargs):
        assert value_render_option == ValueRenderOption.unformatted
        return [
            ([[self.read_back[r]]] if r in self.read_back else []) for r in ranges
        ]


class PlannedCellTest(unittest.TestCase):
    def test_raw_text_keeps_text_expectation(self):
        cell = PlannedCell(1, 1, "00123")
        self.assertEqual(cell.value_input_option, ValueInputOption.raw)
        self.assertEqual(cell.expect, Expectation.text)
        self.assertEqual(cell.expected_value, "00123")
        self.assertEqual(cell.payload, "00123")

    def test_user_entered_string_defaults_to_any(self):
        cell = PlannedCell(
            1, 1, "1,234", value_input_option=ValueInputOption.user_entered
        )
        self.assertEqual(cell.expect, Expectation.any)
        self.assertIsNone(cell.expected_value)

    def test_equal_sign_text_is_escaped(self):
        cell = PlannedCell(
            1, 1, "=SUM(A1:A2)", value_input_option=ValueInputOption.user_entered
        )
        self.assertTrue(cell.escaped)
        self.assertEqual(cell.payload, "'=SUM(A1:A2)")
        self.assertEqual(cell.expect, Expectation.text)
        self.assertEqual(cell.expected_value, "=SUM(A1:A2)")

    def test_equal_sign_formula_allowed_explicitly(self):
        cell = PlannedCell(
            1,
            1,
            "=SUM(A1:A2)",
            value_input_option=ValueInputOption.user_entered,
            allow_formula=True,
        )
        self.assertFalse(cell.escaped)
        self.assertEqual(cell.payload, "=SUM(A1:A2)")

    def test_equal_sign_text_under_raw_needs_no_escape(self):
        cell = PlannedCell(1, 1, "=1+1")
        self.assertFalse(cell.escaped)
        self.assertEqual(cell.payload, "=1+1")

    def test_rejects_none_value(self):
        with self.assertRaises(InvalidInputValue):
            PlannedCell(1, 1, None)

    def test_rejects_invalid_coordinates(self):
        with self.assertRaises(InvalidInputValue):
            PlannedCell(0, 1, "x")


class ApplyWritePlanTest(unittest.TestCase):
    def test_mixed_semantics_are_grouped_and_verified(self):
        cells = [
            PlannedCell(1, 1, "00123"),
            PlannedCell(
                1,
                2,
                "1,234",
                value_input_option=ValueInputOption.user_entered,
                expect=Expectation.number,
            ),
        ]
        worksheet = FakeWorksheet({"A1": "00123", "B1": 1234})

        report = apply_write_plan(worksheet, cells)

        # one write request per input semantics
        self.assertEqual(len(worksheet.writes), 2)
        by_option = dict(worksheet.writes)
        self.assertEqual(
            by_option[ValueInputOption.raw],
            [{"range": "A1", "values": [["00123"]]}],
        )
        self.assertEqual(
            by_option[ValueInputOption.user_entered],
            [{"range": "B1", "values": [["1,234"]]}],
        )

        self.assertTrue(report.ok)
        self.assertEqual(report.total, 2)
        self.assertEqual(len(report.mismatches), 0)
        self.assertEqual([r.actual for r in report.matched], ["00123", 1234])
        # raw write responses are kept for inspection
        self.assertEqual(len(report.responses), 2)

    def test_equal_sign_text_verified_as_text(self):
        cells = [
            PlannedCell(
                1, 1, "=1+1", value_input_option=ValueInputOption.user_entered
            )
        ]
        worksheet = FakeWorksheet({"A1": "=1+1"})

        report = apply_write_plan(worksheet, cells)

        option, data = worksheet.writes[0]
        self.assertEqual(option, ValueInputOption.user_entered)
        self.assertEqual(data[0]["values"], [["'=1+1"]])
        self.assertTrue(report.ok)

    def test_accidental_formula_is_reported(self):
        cells = [
            PlannedCell(
                1, 1, "=1+1", value_input_option=ValueInputOption.user_entered
            )
        ]
        # the sheet stored a real formula: reading back unformatted
        # returns the computed value instead of the literal text
        worksheet = FakeWorksheet({"A1": 2})

        report = apply_write_plan(worksheet, cells)

        self.assertFalse(report.ok)
        self.assertEqual(len(report.mismatches), 1)
        mismatch = report.mismatches[0]
        self.assertEqual(mismatch.a1, "A1")
        self.assertEqual(mismatch.expect, Expectation.text)
        self.assertTrue(mismatch.reason.startswith("type:"))

    def test_locale_parse_mismatch_is_reported(self):
        # convention says "00123" must stay text, but it was sent with
        # user_entered semantics and the sheet parsed it as a number
        cells = [
            PlannedCell(
                1,
                1,
                "00123",
                value_input_option=ValueInputOption.user_entered,
                expect=Expectation.text,
            )
        ]
        worksheet = FakeWorksheet({"A1": 123})

        report = apply_write_plan(worksheet, cells)

        self.assertFalse(report.ok)
        self.assertEqual(report.mismatches[0].actual, 123)
        self.assertTrue(report.mismatches[0].reason.startswith("type:"))

    def test_raw_content_mismatch_is_reported(self):
        cells = [PlannedCell(1, 1, "00123")]
        worksheet = FakeWorksheet({"A1": "0123"})

        report = apply_write_plan(worksheet, cells)

        self.assertFalse(report.ok)
        self.assertTrue(report.mismatches[0].reason.startswith("content:"))

    def test_empty_cell_after_write_is_reported(self):
        cells = [PlannedCell(1, 1, "00123")]
        worksheet = FakeWorksheet({})

        report = apply_write_plan(worksheet, cells)

        self.assertFalse(report.ok)
        self.assertTrue(report.mismatches[0].reason.startswith("empty:"))

    def test_expected_value_is_checked(self):
        cells = [
            PlannedCell(
                1,
                1,
                "1,234",
                value_input_option=ValueInputOption.user_entered,
                expect=Expectation.number,
                expected_value=1234,
            )
        ]
        worksheet = FakeWorksheet({"A1": 1234})
        self.assertTrue(apply_write_plan(worksheet, cells).ok)

        worksheet = FakeWorksheet({"A1": 12345})
        report = apply_write_plan(worksheet, cells)
        self.assertFalse(report.ok)
        self.assertTrue(report.mismatches[0].reason.startswith("content:"))

    def test_boolean_is_not_confused_with_number(self):
        cells = [PlannedCell(1, 1, True)]
        worksheet = FakeWorksheet({"A1": 1})

        report = apply_write_plan(worksheet, cells)

        self.assertFalse(report.ok)
        self.assertEqual(report.mismatches[0].expect, Expectation.boolean)

    def test_empty_plan_is_a_noop(self):
        worksheet = FakeWorksheet({})
        report = apply_write_plan(worksheet, [])
        self.assertTrue(report.ok)
        self.assertEqual(report.total, 0)
        self.assertEqual(worksheet.writes, [])
