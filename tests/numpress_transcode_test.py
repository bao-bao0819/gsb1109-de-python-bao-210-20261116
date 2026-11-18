import unittest

import numpy as np

from pymzml.numpress_transcode import (
    METHOD_ENCODING_NAMES,
    NumpressRoundtripError,
    NumpressTranscoder,
    migrate,
    transcode,
)

SMOOTH_MZ = [
    100.00066,
    100.00217,
    100.00368,
    100.00519,
    111.73335,
    111.73513,
    111.73692,
    111.73870,
    112.00694,
    112.00873,
]
INTENSITY = [10.0, 2000.0, 35.0, 100000.0, 7.0]


class TranscodeRoundTripTest(unittest.TestCase):
    def test_linear_within_relative_budget_records_identity(self):
        codec = NumpressTranscoder(original_encoding="64-bit float")
        payload, decoded, report = codec.transcode(SMOOTH_MZ, "linear", {"rtol": 1e-4})
        self.assertIsInstance(payload, bytearray)
        self.assertEqual(len(decoded), len(SMOOTH_MZ))
        self.assertTrue(report.within_budget)
        self.assertIsNone(report.reason)
        self.assertEqual(report.method, "linear")
        self.assertEqual(report.round_index, 0)
        self.assertEqual(report.source_encoding, "64-bit float")
        self.assertEqual(
            report.target_encoding,
            "MS-Numpress linear prediction compression",
        )
        self.assertEqual(report.array_length, len(SMOOTH_MZ))
        self.assertEqual(report.encoded_size, len(payload))
        self.assertGreaterEqual(report.max_rel_error, 0.0)
        self.assertEqual(codec.history, [report])
        self.assertEqual(
            report.cumulative_within_budget, report.within_budget
        )

    def test_tight_budget_is_rejected_not_silent(self):
        _payload, _decoded, report = transcode(SMOOTH_MZ, "linear", {"rtol": 1e-12})
        self.assertFalse(report.within_budget)
        self.assertIn("budget", report.reason)

    def test_pic_exact_integers_have_zero_absolute_error(self):
        _payload, decoded, report = transcode(
            [5.0, 2000.0, 7.0], "pic", {"atol": 0.0}
        )
        self.assertTrue(report.within_budget)
        self.assertEqual(report.max_abs_error, 0.0)
        np.testing.assert_array_equal(decoded, [5.0, 2000.0, 7.0])

    def test_pic_rounding_error_caught_by_absolute_budget(self):
        values = [5.4, 2000.6, 7.1]
        _payload, decoded, loose = transcode(values, "pic", {"atol": 0.6})
        self.assertTrue(loose.within_budget)
        _payload, _decoded, tight = transcode(values, "pic", {"atol": 0.1})
        self.assertFalse(tight.within_budget)
        self.assertGreater(tight.max_abs_error, 0.1)

    def test_position_and_intensity_have_independent_budgets(self):
        mz_codec = NumpressTranscoder(original_encoding="64-bit float")
        i_codec = NumpressTranscoder(original_encoding="64-bit float")
        _p, _d, mz_report = mz_codec.transcode(SMOOTH_MZ, "linear", {"rtol": 1e-4})
        _p, _d, i_report = i_codec.transcode(INTENSITY, "pic", {"atol": 0.5})
        self.assertTrue(mz_report.within_budget)
        self.assertTrue(i_report.within_budget)
        self.assertNotEqual(mz_report.budget_rtol, i_report.budget_rtol)
        self.assertNotEqual(mz_report.method, i_report.method)
        self.assertEqual(mz_report.target_encoding, METHOD_ENCODING_NAMES["linear"])
        self.assertEqual(i_report.target_encoding, METHOD_ENCODING_NAMES["pic"])

    def test_pic_zeros_at_any_position_length_mismatch_is_rejected(self):
        # pure python PIC drops every zero (leading, internal, trailing):
        # a smaller payload must never read as an in-budget round trip.
        for values in (
            [0.0, 0.0, 7.0],
            [5.0, 0.0, 7.0],
            [5.0, 7.0, 0.0],
            [5.0, 0.0, 0.0, 7.0],
        ):
            with self.subTest(values=values):
                _payload, decoded, report = transcode(
                    values, "pic", {"atol": 10.0}
                )
                self.assertFalse(report.within_budget)
                self.assertNotEqual(decoded.size, len(values))
                self.assertIn("length mismatch", report.reason)
                self.assertEqual(report.max_abs_error, float("inf"))

    def test_all_zero_linear_array_is_rejected(self):
        # pure python linear encoding drops the tail of an all zero array
        values = [0.0, 0.0, 0.0]
        _payload, decoded, report = transcode(values, "linear", {"atol": 1e-9})
        self.assertFalse(report.within_budget)
        self.assertNotEqual(decoded.size, len(values))
        self.assertIn("length mismatch", report.reason)

    def test_zero_elements_are_checked_with_absolute_tolerance(self):
        values = [0.0, 0.001, 0.002]
        _payload, decoded, report = transcode(values, "linear", {"atol": 1e-6})
        self.assertTrue(report.within_budget)
        self.assertEqual(decoded[0], 0.0)
        np.testing.assert_allclose(decoded, values, atol=1e-6)

    def test_small_peaks_slof(self):
        values = [0.0, 0.001, 1.0, 2.0]
        _payload, _decoded, loose = transcode(values, "slof", {"atol": 0.01})
        self.assertTrue(loose.within_budget)
        _payload, _decoded, tight = transcode(values, "slof", {"atol": 1e-9})
        self.assertFalse(tight.within_budget)

    def test_huge_values_linear_stay_in_range(self):
        values = [1e6, 1e6 + 1.0001, 1e6 + 2.0003]
        _payload, decoded, report = transcode(values, "linear", {"rtol": 1e-9})
        self.assertTrue(report.within_budget)
        self.assertEqual(decoded[0], 1e6)
        np.testing.assert_allclose(decoded, values, rtol=1e-9)

    def test_huge_values_slof_tight_budget_rejected(self):
        values = [1e12, 1e6, 0.0]
        _payload, _decoded, tight = transcode(values, "slof", {"atol": 1e6})
        self.assertFalse(tight.within_budget)
        _payload, decoded, loose = transcode(values, "slof", {"atol": 1e10})
        self.assertTrue(loose.within_budget)
        self.assertEqual(decoded.shape, (3,))

    def test_accepts_numpy_arrays(self):
        payload, decoded, report = NumpressTranscoder().transcode(
            np.asarray(SMOOTH_MZ, dtype=np.float64), "linear", {"rtol": 1e-4}
        )
        self.assertIsInstance(payload, bytearray)
        self.assertTrue(report.within_budget)


class MigrationTest(unittest.TestCase):
    def test_multi_round_chain_records_identity_and_cumulative_error(self):
        codec = NumpressTranscoder(original_encoding="64-bit float")
        result = codec.migrate(SMOOTH_MZ, ["linear", "linear"], {"rtol": 1e-3})
        self.assertTrue(result.accepted)
        self.assertEqual(len(result.reports), 2)
        first, second = result.reports
        self.assertEqual(first.round_index, 0)
        self.assertEqual(second.round_index, 1)
        self.assertEqual(second.source_encoding, first.target_encoding)
        self.assertEqual(first.source_encoding, "64-bit float")
        for report in result.reports:
            self.assertIsNotNone(report.cumulative_abs_error)
            self.assertIsNotNone(report.cumulative_rel_error)
            self.assertTrue(report.within_budget)
            self.assertTrue(report.cumulative_within_budget)
        expected_cum_abs = float(
            np.max(np.abs(result.values - result.original_values))
        )
        self.assertAlmostEqual(second.cumulative_abs_error, expected_cum_abs)

    def test_migration_rejected_on_cumulative_budget(self):
        result = migrate(SMOOTH_MZ, ["linear", "linear"], {"rtol": 1e-12})
        self.assertFalse(result.accepted)
        self.assertEqual(len(result.reports), 1)
        self.assertFalse(result.reports[0].within_budget)
        self.assertFalse(result.reports[0].cumulative_within_budget)

    def test_migration_alternating_methods(self):
        values = [100.0, 2000.0, 35.0, 100000.0, 7.0, 9.0]
        result = migrate(
            values,
            ["pic", "slof"],
            {"atol": 10.0},
            original_encoding="MS-Numpress positive integer compression",
        )
        self.assertTrue(result.accepted)
        self.assertEqual(
            result.reports[0].source_encoding,
            "MS-Numpress positive integer compression",
        )
        self.assertEqual(result.reports[0].method, "pic")
        self.assertEqual(result.reports[1].method, "slof")
        self.assertEqual(
            result.reports[1].source_encoding,
            METHOD_ENCODING_NAMES["pic"],
        )


class ValidationTest(unittest.TestCase):
    def test_unknown_method(self):
        with self.assertRaises(ValueError):
            transcode(SMOOTH_MZ, "nope", {"atol": 1.0})

    def test_empty_array(self):
        with self.assertRaises(NumpressRoundtripError):
            transcode([], "linear", {"atol": 1.0})

    def test_single_element_linear_not_readable(self):
        with self.assertRaises(NumpressRoundtripError):
            transcode([100.25], "linear", {"atol": 1.0})

    def test_negative_values_rejected_for_pic_and_slof(self):
        with self.assertRaises(ValueError):
            transcode([1.0, -2.0], "pic", {"atol": 1.0})
        with self.assertRaises(ValueError):
            transcode([1.0, -2.0], "slof", {"atol": 1.0})

    def test_non_finite_values_rejected(self):
        with self.assertRaises(ValueError):
            transcode([1.0, float("nan")], "linear", {"atol": 1.0})
        with self.assertRaises(ValueError):
            transcode([1.0, float("inf")], "slof", {"atol": 1.0})

    def test_budget_validation(self):
        with self.assertRaises(ValueError):
            transcode(SMOOTH_MZ, "linear", {"atol": -1.0})
        with self.assertRaises(TypeError):
            transcode(SMOOTH_MZ, "linear", "1e-3")
        codec = NumpressTranscoder()
        payload, _decoded, report = codec.transcode(
            SMOOTH_MZ, "linear", {"rtol": 1e-3}
        )
        self.assertIsInstance(payload, bytearray)
        self.assertTrue(report.within_budget)

    def test_migrate_requires_methods(self):
        with self.assertRaises(ValueError):
            migrate(SMOOTH_MZ, [], {"atol": 1.0})


if __name__ == "__main__":
    unittest.main(verbosity=3)
