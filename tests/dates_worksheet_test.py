import unittest
from datetime import date, timedelta
from unittest.mock import Mock

from gspread.http_client import HTTPClient
from gspread.spreadsheet import Spreadsheet
from gspread.utils import DateTimeOption, ValueRenderOption
from gspread.worksheet import Worksheet


def _make_worksheet(values, timezone_name="America/New_York"):
    client = HTTPClient(auth=None, session=Mock())
    properties = {
        "title": "Sheet1",
        "sheetId": 0,
        "index": 0,
        "gridProperties": {"rowCount": 10, "columnCount": 10},
    }
    spreadsheet = Mock(spec=Spreadsheet)
    spreadsheet.timezone = timezone_name
    worksheet = Worksheet(
        spreadsheet=spreadsheet,
        properties=properties,
        spreadsheet_id="abc123",
        client=client,
    )
    worksheet.get = Mock(return_value=values)
    return worksheet


class WorksheetDateColumnTest(unittest.TestCase):
    def test_reads_unformatted_serials(self):
        worksheet = _make_worksheet([[43831], [43831.5]])
        column = worksheet.interpret_date_column(
            "A1:A2", kind="date", number_formats="yyyy-mm-dd"
        )

        kwargs = worksheet.get.call_args.kwargs
        self.assertEqual(kwargs["value_render_option"], ValueRenderOption.unformatted)
        self.assertEqual(
            kwargs["date_time_render_option"], DateTimeOption.serial_number
        )
        self.assertEqual(column.readings()[0].value, date(2020, 1, 1))

    def test_uses_spreadsheet_timezone(self):
        worksheet = _make_worksheet([[43831.5]])
        column = worksheet.interpret_date_column(
            "A1:A1",
            kind="datetime",
            number_formats="yyyy-mm-dd hh:mm",
        )
        reading = column.readings()[0]
        self.assertEqual(reading.timezone, "America/New_York")
        self.assertEqual(reading.datetime.utcoffset(), timedelta(hours=-5))

    def test_rejects_multi_column_range(self):
        worksheet = _make_worksheet([[1, 2]])
        with self.assertRaises(ValueError):
            worksheet.interpret_date_column("A1:B1")

    def test_empty_row_becomes_blank(self):
        worksheet = _make_worksheet([[43831], []])
        column = worksheet.interpret_date_column("A1:A2", kind="date")
        self.assertIn(1, column.invalid)
