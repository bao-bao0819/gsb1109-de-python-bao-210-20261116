# -*- coding: utf-8 -*-

import contextlib
import io
import os
import tempfile
import unittest

from pygtfs import Schedule, append_feed
from pygtfs.actual_stop_times import main


class TestActualStopTimesCommand(unittest.TestCase):
    def setUp(self):
        handle, self.database = tempfile.mkstemp(suffix=".sqlite")
        os.close(handle)
        self.addCleanup(os.remove, self.database)
        schedule = Schedule(self.database)
        with contextlib.redirect_stdout(io.StringIO()):
            append_feed(schedule, os.path.join(
                os.path.dirname(__file__), "data", "sample_feed"))

    def test_persists_selected_service_date(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            code = main([self.database, "--trip", "AB1",
                         "--service-date", "20070605"])
        self.assertEqual(code, 0)
        self.assertIn("Exported 2", output.getvalue())
        rows = Schedule(self.database).exported_stop_times
        self.assertEqual([row.stop_id for row in rows],
                         ["BEATTY_AIRPORT", "BULLFROG"])

    def test_writes_csv_without_persisting(self):
        handle, path = tempfile.mkstemp(suffix=".csv")
        os.close(handle)
        self.addCleanup(os.remove, path)
        with contextlib.redirect_stdout(io.StringIO()):
            main([self.database, "--trip", "AB1", "--service-date", "20070605",
                  "--csv", path, "--no-persist"])
        with open(path, encoding="utf-8") as csv_file:
            lines = csv_file.read().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[0].startswith("feed_id,trip_id,service_id"))
        self.assertEqual(Schedule(self.database).exported_stop_times, [])


if __name__ == "__main__":
    unittest.main()
