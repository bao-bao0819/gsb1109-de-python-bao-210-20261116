import json
import math
import os
import tempfile
import unittest

import arff

from arff_compare import ClassMapping, ComparisonError, compare, compare_files
from arff_compare.cli import main, render_text

WEIGHTS = "yes"


def dataset(relation, rows, stratum=("s", ["A", "B"]),
            classes=("y", ["yes", "no"])):
    return {
        "relation": relation,
        "attributes": [stratum, classes, ("w", "REAL")],
        "data": rows,
    }


def mapping():
    return ClassMapping.from_obj({"yes": ["yes"], "no": ["no"]})


class TestActualShares(unittest.TestCase):
    def test_weighted_shares_not_row_counts(self):
        baseline = dataset("A", [
            ["A", "yes", 30.0], ["A", "no", 10.0],
            ["B", "yes", 5.0], ["B", "no", 55.0],
        ])
        comparison = dataset("B", [
            ["A", "yes", 2.0], ["A", "no", 2.0],
            ["B", "yes", 60.0], ["B", "no", 36.0],
        ])
        report = compare(baseline, comparison, ["s"], "y", mapping(),
                         weight_attribute="w")
        self.assertAlmostEqual(
            report["actual"]["baseline"]["completed_shares"]["yes"], 0.35)
        self.assertAlmostEqual(
            report["actual"]["comparison"]["completed_shares"]["yes"], 0.62)

    def test_unknown_label_is_residual_bounds_not_zero(self):
        baseline = dataset("A", [
            ["A", "yes", 50.0], ["A", "legacy", 50.0],
            ["B", "no", 100.0],
        ])
        report = compare(baseline, baseline, ["s"], "y", mapping(),
                         weight_attribute="w")
        actual = report["actual"]["baseline"]
        self.assertEqual(actual["residual_shares"], {"legacy": 0.25})
        self.assertEqual(actual["bounds"]["yes"], [0.25, 0.5])
        self.assertEqual(actual["known_mapped_shares"]["yes"], 0.25)
        # The proportional completion sends stratum A's residual to its only
        # mapped class, so the point is 0.5 while the bound remains [0.25, 0.75].
        self.assertAlmostEqual(actual["completed_shares"]["yes"], 0.5)
        self.assertEqual(report["missing"]["baseline_unknown_labels"],
                         {"legacy": 50.0})

    def test_missing_class_is_tracked_separately(self):
        baseline = dataset("A", [
            ["A", "yes", 60.0], ["A", None, 40.0],
        ])
        report = compare(baseline, baseline, ["s"], "y", mapping(),
                         weight_attribute="w")
        self.assertEqual(report["missing"]["baseline_missing_class_weight"], 40.0)
        self.assertIn("__MISSING_CLASS__",
                      report["actual"]["baseline"]["residual_shares"])


class TestStandardization(unittest.TestCase):
    def test_direct_standardization_uses_baseline_weights(self):
        baseline = dataset("A", [
            ["A", "yes", 75.0], ["A", "no", 25.0],
            ["B", "yes", 25.0], ["B", "no", 75.0],
        ])
        comparison = dataset("B", [
            ["A", "yes", 50.0], ["A", "no", 50.0],
            ["B", "yes", 50.0], ["B", "no", 50.0],
        ])
        report = compare(baseline, comparison, ["s"], "y", mapping(),
                         weight_attribute="w")
        # Both periods have equal weights; the comparison rate is 0.5 in both
        # strata so standardisation must leave it at 0.5.
        self.assertAlmostEqual(
            report["standardized"]["primary"]["point"]["yes"], 0.5)
        self.assertAlmostEqual(
            report["standardized"]["primary"]["bounds"]["yes"][0], 0.5)

    def test_absent_baseline_stratum_is_not_zero_rate(self):
        baseline = dataset("A", [
            ["A", "yes", 50.0], ["A", "no", 50.0],
            ["B", "yes", 50.0], ["B", "no", 50.0],
        ])
        comparison = dataset("B", [
            ["A", "yes", 100.0],
            # stratum B exists in the baseline but not in the comparison
        ])
        report = compare(baseline, comparison, ["s"], "y", mapping(),
                         weight_attribute="w")
        primary = report["standardized"]["primary"]
        self.assertEqual(primary["bounds"]["yes"], [1.0, 1.0])
        full = report["standardized"]["full_baseline"]
        # The missing B rate is unknown, not zero: its weight may all be yes.
        self.assertEqual(full["bounds"]["yes"], [0.5, 1.0])
        self.assertEqual(report["missing"]["baseline_only_strata"], ["B"])

    def test_comparison_only_stratum_has_no_baseline_weight(self):
        baseline = dataset("A", [["A", "yes", 100.0]])
        comparison = dataset("B", [
            ["A", "no", 10.0],
            ["C", "yes", 90.0],
        ], stratum=("s", ["A", "C"]))
        report = compare(baseline, comparison, ["s"], "y", mapping(),
                         weight_attribute="w")
        self.assertEqual(report["missing"]["comparison_only_strata"], ["C"])
        self.assertAlmostEqual(
            report["standardized"]["coverage"]
            ["comparison_weight_share_comparison_only"], 0.9)
        self.assertAlmostEqual(
            report["fixed_rates"]["coverage"]["comparison_only_weight_share"],
            0.9)
        # In common stratum A the baseline rate is 1; C's unobserved baseline
        # rate may be anywhere in [0, 1], so the lower endpoint is 0.1.
        self.assertEqual(report["fixed_rates"]["bounds"]["yes"], [0.1, 1.0])

    def test_composite_strata(self):
        attributes = [("region", ["n", "s"]), ("channel", ["web", "shop"]),
                      ("y", ["yes", "no"]), ("w", "REAL")]
        baseline = {"relation": "A", "attributes": attributes, "data": [
            ["n", "web", "yes", 10.0],
            ["n", "web", "no", 30.0],
            ["s", "shop", "yes", 50.0],
            ["s", "shop", "no", 10.0],
        ]}
        report = compare(baseline, baseline, ["region", "channel"], "y",
                         mapping(), weight_attribute="w")
        labels = [row["stratum"]
                  for row in report["decomposition"]["strata"]]
        self.assertEqual(labels, ["region=n | channel=web",
                                  "region=s | channel=shop"])


class TestExplanability(unittest.TestCase):
    def test_weights_only_explains_difference(self):
        # Rates are identical in both strata; only the mix changes.
        baseline = dataset("A", [
            ["A", "yes", 90.0], ["A", "no", 10.0],
            ["B", "yes", 10.0], ["B", "no", 90.0],
        ])
        # Same within-stratum rates, but stratum B now carries 80% of weight.
        comparison = dataset("B", [
            ["A", "yes", 18.0], ["A", "no", 2.0],
            ["B", "yes", 8.0], ["B", "no", 72.0],
        ])
        report = compare(baseline, comparison, ["s"], "y", mapping(),
                         weight_attribute="w")
        item = report["explanation"]["yes"]
        self.assertTrue(item["weight_only_can_reproduce_observed"])
        self.assertEqual(item["verdict"],
                         "weight_only_change_not_ruled_out")
        self.assertAlmostEqual(item["fixed_rate_point"], 0.26)
        self.assertAlmostEqual(item["actual_comparison_point"], 0.26)
        self.assertIsNotNone(item["witness"])

    def test_within_stratum_change_is_beyond_weights(self):
        baseline = dataset("A", [
            ["A", "yes", 50.0], ["A", "no", 50.0],
            ["B", "yes", 50.0], ["B", "no", 50.0],
        ])
        comparison = dataset("B", [
            ["A", "yes", 50.0], ["A", "no", 50.0],
            ["B", "yes", 90.0], ["B", "no", 10.0],
        ])
        report = compare(baseline, comparison, ["s"], "y", mapping(),
                         weight_attribute="w")
        item = report["explanation"]["yes"]
        self.assertFalse(item["weight_only_can_reproduce_observed"])
        self.assertEqual(item["verdict"],
                         "weight_only_change_ruled_out")
        self.assertAlmostEqual(item["fixed_rate_point"], 0.5)
        self.assertAlmostEqual(item["minimum_rate_component_needed"], 0.2)

    def test_unknown_mass_can_make_difference_explained(self):
        # The baseline yes share is known only as [0.2, 0.7]; a completion
        # with yes rate 0.5 exists, so the weight-only explanation is viable.
        baseline = dataset("A", [
            ["A", "yes", 20.0], ["A", "legacy", 30.0],
            ["B", "yes", 10.0], ["B", "legacy", 40.0],
        ])
        comparison = dataset("B", [
            ["A", "yes", 50.0], ["A", "no", 50.0],
            ["B", "yes", 50.0], ["B", "no", 50.0],
        ])
        report = compare(baseline, comparison, ["s"], "y", mapping(),
                         weight_attribute="w")
        item = report["explanation"]["yes"]
        self.assertTrue(item["weight_only_can_reproduce_observed"])
        witness = item["witness"]
        self.assertGreater(len(witness["baseline_rate_completion"]
                               ["allocations"]), 0)
        # The comparison period has no unknown mass, so its completion is the
        # empty allocation; the baseline rate completion carries the witness.
        self.assertEqual(witness["comparison_completion"]["allocations"], [])
        # Applying the witness must reproduce the fixed-rate target exactly.
        self.assertAlmostEqual(witness["target_share"],
                               item["actual_comparison_point"])


class TestIdentificationEdgeCases(unittest.TestCase):
    def test_no_common_strata_is_not_identified(self):
        baseline = dataset("A", [["A", "yes", 10.0]],
                           stratum=("s", ["A"]))
        comparison = dataset("B", [["B", "no", 10.0]],
                             stratum=("s", ["B"]))
        report = compare(baseline, comparison, ["s"], "y", mapping(),
                         weight_attribute="w")
        item = report["explanation"]["yes"]
        self.assertFalse(item["identified"])
        self.assertEqual(item["verdict"], "not_identifiable")
        self.assertEqual(item["fixed_rate_bounds"], [0.0, 1.0])
        self.assertEqual(report["standardized"]["strata"], [])
        self.assertEqual(report["missing"]["baseline_only_strata"], ["A"])
        self.assertEqual(report["missing"]["comparison_only_strata"], ["B"])

    def test_missing_stratum_value_is_separate_and_finite(self):
        baseline = dataset("A", [
            ["A", "yes", 10.0], [None, "no", 10.0],
        ])
        comparison = dataset("B", [
            ["A", "yes", 10.0], [None, "yes", 10.0],
        ])
        report = compare(baseline, comparison, ["s"], "y", mapping(),
                         weight_attribute="w")
        self.assertEqual(report["actual"]["baseline"]
                         ["unassigned_stratum_weight"], 10.0)
        self.assertEqual(report["actual"]["comparison"]
                         ["unassigned_stratum_weight"], 10.0)
        # The standardized primary scope excludes the unassigned rows, so it
        # is the comparison rate in A (1.0), not a pooled mix with the
        # unassigned stratum.
        self.assertEqual(report["standardized"]["primary"]["bounds"]["yes"],
                         [1.0, 1.0])
        # The full-baseline range lets the unassigned stratum's comparison
        # rate be 0 or 1, so it spans the whole identified range.
        self.assertEqual(report["standardized"]["full_baseline"]
                         ["bounds"]["yes"], [0.5, 1.0])

    def test_bounds_cannot_sum_to_one_for_two_classes(self):
        baseline = dataset("A", [
            ["A", "yes", 30.0], ["A", "legacy", 70.0],
        ])
        report = compare(baseline, baseline, ["s"], "y", mapping(),
                         weight_attribute="w")
        bounds = report["actual"]["baseline"]["bounds"]
        # Both class bounds are marginal: each can reach its upper endpoint
        # alone, but not simultaneously.
        self.assertEqual(bounds["yes"], [0.3, 1.0])
        self.assertEqual(bounds["no"], [0.0, 0.7])
        self.assertGreater(bounds["yes"][1] + bounds["no"][1], 1.0)

    def test_witness_allocation_reproduces_target(self):
        baseline = dataset("A", [
            ["A", "yes", 20.0], ["A", "legacy", 30.0],
            ["B", "yes", 10.0], ["B", "legacy", 40.0],
        ])
        comparison = dataset("B", [
            ["A", "yes", 50.0], ["A", "no", 50.0],
            ["B", "yes", 50.0], ["B", "no", 50.0],
        ])
        report = compare(baseline, comparison, ["s"], "y", mapping(),
                         weight_attribute="w")
        item = report["explanation"]["yes"]
        witness = item["witness"]
        # Rebuild the fixed-rate share from observed mass plus the witness
        # allocations; it must equal the target chosen from the interval.
        total = report["actual"]["comparison"]["total_weight"]
        target = witness["target_share"]
        self.assertTrue(0.0 <= target <= 1.0)
        self.assertAlmostEqual(
            item["actual_comparison_point"]
            if item["weight_only_interval"] is None else target,
            target)
        self.assertEqual(witness["baseline_rate_completion"]["target_share"],
                         target)
        self.assertLessEqual(
            abs(witness["baseline_rate_completion"]["allocated_component"]),
            total)


class TestStringClass(unittest.TestCase):
    def test_string_class_requires_explicit_mapping(self):
        baseline = {
            "relation": "A",
            "attributes": [("s", ["A"]), ("y", "STRING"), ("w", "REAL")],
            "data": [["A", "yes", 1.0]],
        }
        with self.assertRaises(ComparisonError):
            compare(baseline, baseline, ["s"], "y", mapping(),
                    weight_attribute="w")

    def test_string_class_with_allow(self):
        baseline = {
            "relation": "A",
            "attributes": [("s", ["A"]), ("y", "STRING"), ("w", "REAL")],
            "data": [["A", "yes", 1.0], ["A", "no", 1.0]],
        }
        report = compare(baseline, baseline, ["s"], "y", mapping(),
                         weight_attribute="w", allow_string_class=True)
        self.assertEqual(report["actual"]["baseline"]
                         ["completed_shares"]["yes"], 0.5)


class TestDecomposition(unittest.TestCase):
    def test_mix_rate_identity(self):
        baseline = dataset("A", [
            ["A", "yes", 70.0], ["A", "no", 30.0],
            ["B", "yes", 20.0], ["B", "no", 80.0],
        ])
        comparison = dataset("B", [
            ["A", "yes", 30.0], ["A", "no", 10.0],
            ["B", "yes", 30.0], ["B", "no", 30.0],
        ])
        report = compare(baseline, comparison, ["s"], "y", mapping(),
                         weight_attribute="w")
        totals = report["decomposition"]["totals"]["yes"]
        self.assertAlmostEqual(
            totals["mix_effect_point"] + totals["rate_effect_point"]
            + totals["interaction_point"],
            totals["comparable_change_point"], places=12)
        self.assertAlmostEqual(totals["identity_check"], 0.0, places=12)

    def test_per_stratum_contributions_are_reported(self):
        baseline = dataset("A", [
            ["A", "yes", 70.0], ["A", "no", 30.0],
            ["B", "yes", 20.0], ["B", "no", 80.0],
        ])
        comparison = dataset("B", [
            ["A", "yes", 30.0], ["A", "no", 10.0],
            ["B", "yes", 30.0], ["B", "no", 30.0],
        ])
        report = compare(baseline, comparison, ["s"], "y", mapping(),
                         weight_attribute="w")
        rows = {row["stratum"]: row for row in report["decomposition"]["strata"]}
        self.assertEqual(set(rows), {"A", "B"})
        for row in rows.values():
            entry = row["classes"]["yes"]
            self.assertIn("mix_effect", entry)
            self.assertIn("rate_effect", entry)
            self.assertIn("interaction", entry)


class TestInputs(unittest.TestCase):
    def test_missing_weight_policy_error(self):
        baseline = dataset("A", [["A", "yes", None], ["A", "no", 10.0]])
        with self.assertRaises(ComparisonError):
            compare(baseline, baseline, ["s"], "y", mapping(),
                    weight_attribute="w")

    def test_missing_weight_policy_one(self):
        baseline = dataset("A", [["A", "yes", None], ["A", "no", 10.0]])
        report = compare(baseline, baseline, ["s"], "y", mapping(),
                         weight_attribute="w", missing_weight="one")
        self.assertAlmostEqual(report["actual"]["baseline"]["total_weight"], 11.0)

    def test_negative_weight_rejected(self):
        baseline = dataset("A", [["A", "yes", -1.0]])
        with self.assertRaises(ComparisonError):
            compare(baseline, baseline, ["s"], "y", mapping(),
                    weight_attribute="w")

    def test_class_attribute_must_be_nominal(self):
        baseline = {
            "relation": "A",
            "attributes": [("s", ["A"]), ("y", "STRING"), ("w", "REAL")],
            "data": [["A", "yes", 1.0]],
        }
        with self.assertRaises(ComparisonError):
            compare(baseline, baseline, ["s"], "y", mapping(),
                    weight_attribute="w")

    def test_duplicate_mapping_rejected(self):
        with self.assertRaises(ComparisonError):
            ClassMapping.from_obj({
                "yes": ["x"], "no": ["x"]})


class TestStratumwiseIdentification(unittest.TestCase):
    def test_offsetting_rate_changes_are_ruled_out(self):
        # Pooled fixed-rate and observed shares both equal 0.5, but each
        # stratum's direction is opposite, so identical rates are infeasible.
        baseline = dataset("A", [
            ["A", "yes", 50.0], ["A", "no", 50.0],
            ["B", "yes", 50.0], ["B", "no", 50.0],
        ])
        comparison = dataset("B", [
            ["A", "yes", 70.0], ["A", "no", 30.0],
            ["B", "yes", 30.0], ["B", "no", 70.0],
        ])
        report = compare(baseline, comparison, ["s"], "y", mapping(),
                         weight_attribute="w")
        item = report["explanation"]["yes"]
        self.assertFalse(report["stratumwise_equality"]
                         ["all_common_strata_feasible"])
        self.assertEqual(item["verdict"], "weight_only_change_ruled_out")
        self.assertEqual(item["strata_blocking_identity"], ["A", "B"])
        self.assertAlmostEqual(item["stratum_rate_gaps"]["A"], 0.2)
        self.assertAlmostEqual(item["stratum_rate_gaps"]["B"], 0.2)
        self.assertIsNone(item["witness"])

    def test_identical_stratum_rates_are_feasible(self):
        baseline = dataset("A", [
            ["A", "yes", 50.0], ["A", "no", 50.0],
            ["B", "yes", 50.0], ["B", "no", 50.0],
        ])
        comparison = dataset("B", [
            ["A", "yes", 20.0], ["A", "no", 20.0],
            ["B", "yes", 70.0], ["B", "no", 70.0],
        ])
        report = compare(baseline, comparison, ["s"], "y", mapping(),
                         weight_attribute="w")
        self.assertTrue(report["stratumwise_equality"]
                        ["all_common_strata_feasible"])
        self.assertEqual(report["explanation"]["yes"]["verdict"],
                         "weight_only_change_not_ruled_out")


class TestFilesAndCli(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.baseline = os.path.join(self.dir, "a.arff")
        self.comparison = os.path.join(self.dir, "b.arff")
        with open(self.baseline, "w") as handle:
            handle.write(arff.dumps(dataset("A", [
                ["A", "yes", 30.0], ["A", "no", 10.0],
                ["B", "yes", 5.0], ["B", "no", 55.0],
            ])))
        with open(self.comparison, "w") as handle:
            handle.write(arff.dumps(dataset("B", [
                ["A", "yes", 2.0], ["A", "no", 2.0],
                ["B", "yes", 60.0], ["B", "no", 36.0],
            ])))

    def tearDown(self):
        for path in (self.baseline, self.comparison):
            try:
                os.remove(path)
            except OSError:
                pass
        try:
            os.rmdir(self.dir)
        except OSError:
            pass

    def test_compare_files_records_hashes(self):
        report = compare_files(self.baseline, self.comparison, ["s"], "y",
                               mapping=mapping(), weight_attribute="w")
        self.assertEqual(len(report["inputs"]["baseline"]["sha256"]), 64)
        self.assertAlmostEqual(
            report["actual"]["baseline"]["completed_shares"]["yes"], 0.35)

    def test_cli_writes_json(self):
        output = os.path.join(self.dir, "report.json")
        code = main([self.baseline, self.comparison, "--stratum", "s",
                     "--class-attribute", "y", "--weight-attribute", "w",
                     "--output", output])
        self.assertEqual(code, 0)
        with open(output) as handle:
            report = json.load(handle)
        self.assertEqual(report["schema"], "arff-stratified-comparison/1")

    def test_cli_text_mode(self):
        code = main([self.baseline, self.comparison, "--stratum", "s",
                     "--class-attribute", "y", "--weight-attribute", "w",
                     "--format", "text", "--output",
                     os.path.join(self.dir, "report.txt")])
        self.assertEqual(code, 0)
        with open(os.path.join(self.dir, "report.txt")) as handle:
            text = handle.read()
        self.assertIn("Actual weighted shares", text)
        self.assertIn("Standardized to baseline stratum weights", text)


if __name__ == "__main__":
    unittest.main()
