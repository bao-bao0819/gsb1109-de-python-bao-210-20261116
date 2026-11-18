import os
import sys
import unittest

sys.path.append(os.path.abspath("."))

import pymzml
from pymzml.fragment import (
    FragmentCandidate,
    FragmentMatcher,
    MatchResult,
    NeutralLoss,
    match_candidates,
    match_fragments,
    match_spectrum,
)


class FakeSpectrum:
    measured_precision = 5e-6

    def __init__(self, peaks, peak_type="centroided"):
        self._peaks = peaks
        self._peak_type = peak_type

    def peaks(self, peak_type):
        if peak_type != self._peak_type:
            raise KeyError(peak_type)
        return self._peaks


class FragmentMatchingTest(unittest.TestCase):
    def test_basic_support_and_unmatched_peaks(self):
        result = match_fragments(
            [(251.1504, 100.0), (260.1970, 80.0), (400.0, 5.0)],
            [{"name": "peptide", "fragments": {"b2": 251.1503, "y2": 260.1969}}],
            tolerance=20,
            tolerance_unit="ppm",
        )
        self.assertIsInstance(result, MatchResult)
        match = result["peptide"]
        self.assertEqual(match.score, 2.0)
        self.assertEqual(match.coverage, 1.0)
        self.assertEqual(match.matched_peak_indices, (0, 1))
        self.assertEqual(match.unmatched_peak_indices, (2,))
        self.assertEqual(result.unmatched_peaks[0].mz, 400.0)
        self.assertEqual(result.scores, {"peptide": 2.0})
        self.assertEqual(result.best.name, "peptide")

    def test_one_peak_cannot_score_a_candidate_twice(self):
        result = match_fragments(
            [(300.0, 10.0)],
            [{"name": "A", "fragments": {"x": 300.0, "y": 300.000001}}],
            tolerance=20,
            tolerance_unit="ppm",
        )
        match = result["A"]
        self.assertEqual(match.score, 1.0)
        self.assertEqual(match.matched_peak_count, 1)
        self.assertEqual(match.n_explanations, 2)
        ion_labels = {
            explanation[0].ion_label for explanation in match.explanations
        }
        self.assertEqual(ion_labels, {"x", "y"})

    def test_two_peaks_cannot_share_one_ion(self):
        result = match_fragments(
            [(300.0, 10.0), (300.000001, 20.0)],
            [{"name": "A", "fragments": {"x": 300.0}}],
            tolerance=20,
            tolerance_unit="ppm",
        )
        match = result["A"]
        self.assertEqual(match.score, 1.0)
        self.assertEqual(match.matched_peak_count, 1)
        self.assertEqual(match.n_explanations, 2)

    def test_null_observation_is_kept(self):
        result = match_fragments(
            [(500.0, 10.0)],
            [{"name": "A", "fragments": {"x": 300.0}}],
            tolerance=10,
            tolerance_unit="ppm",
        )
        self.assertEqual(result["A"].score, 0.0)
        self.assertEqual(result["A"].assignments, ())
        self.assertEqual(result["A"].unmatched_peak_indices, (0,))
        self.assertEqual(result.unmatched_peaks[0].mz, 500.0)

    def test_neutral_losses_are_subtracted(self):
        fragments = {"b": 251.1506}
        losses = [("H2O", 18.0106), {"label": "NH3", "mass": 17.0265}]
        result = match_fragments(
            [(233.1400, 10.0), (234.1241, 5.0)],
            [{"name": "A", "fragments": fragments}],
            neutral_losses=losses,
            tolerance=20,
            tolerance_unit="ppm",
        )
        labels = set(result["A"].matched_ion_labels)
        self.assertEqual(labels, {"b-H2O", "b-NH3"})
        self.assertEqual(result["A"].ion_count, 3)

    def test_neutral_losses_can_be_disabled(self):
        result = match_fragments(
            [(233.1400, 10.0)],
            [{"name": "A", "fragments": {"b": 251.1506}}],
            tolerance=20,
            tolerance_unit="ppm",
        )
        self.assertEqual(result["A"].score, 0.0)

    def test_candidate_specific_losses(self):
        candidates = [
            {
                "name": "with_loss",
                "fragments": {"b": 251.1506},
                "neutral_losses": [("H2O", 18.0106)],
            },
            {"name": "without_loss", "fragments": {"b": 251.1506}},
        ]
        result = match_fragments(
            [(233.1400, 10.0)],
            candidates,
            tolerance=20,
            tolerance_unit="ppm",
        )
        self.assertEqual(result["with_loss"].score, 1.0)
        self.assertEqual(result["without_loss"].score, 0.0)

    def test_weights_change_support(self):
        candidates = [
            {"name": "A", "fragments": {"x": 300.0}},
            {"name": "B", "fragments": {"x": 300.0}},
        ]
        result = match_fragments(
            [(300.0, 1.0)],
            candidates,
            tolerance=10,
            tolerance_unit="ppm",
            weights={"A": {"x": 1.0}, "B": {"x": 4.0}},
        )
        self.assertEqual(result["A"].score, 1.0)
        self.assertEqual(result["B"].score, 4.0)
        self.assertEqual(result["B"].support, 1.0)

    def test_loss_weight_multiplies_fragment_weight(self):
        result = match_fragments(
            [(233.14, 10.0)],
            [{"name": "A", "fragments": {"b": (251.1506, 2.0)}}],
            neutral_losses=[{"label": "H2O", "mass": 18.0106, "weight": 0.5}],
            tolerance=20,
            tolerance_unit="ppm",
        )
        self.assertAlmostEqual(result["A"].score, 1.0)
        self.assertAlmostEqual(result["A"].assignments[0].weight, 1.0)

    def test_tolerance_units(self):
        candidates = [{"name": "A", "fragments": {"x": 300.0}}]
        in_da = match_fragments(
            [(300.05, 1.0)],
            candidates,
            tolerance=0.1,
            tolerance_unit="Da",
        )
        out_da = match_fragments(
            [(300.05, 1.0)],
            candidates,
            tolerance=0.01,
            tolerance_unit="Da",
        )
        in_ppm = match_fragments(
            [(300.05, 1.0)],
            candidates,
            tolerance=200,
            tolerance_unit="ppm",
        )
        out_ppm = match_fragments(
            [(300.05, 1.0)],
            candidates,
            tolerance=20,
            tolerance_unit="ppm",
        )
        self.assertEqual(in_da["A"].score, 1.0)
        self.assertEqual(out_da["A"].score, 0.0)
        self.assertEqual(in_ppm["A"].score, 1.0)
        self.assertEqual(out_ppm["A"].score, 0.0)

    def test_ppm_tolerance_accepts_fraction_and_string(self):
        candidates = [{"name": "A", "fragments": {"x": 300.0}}]
        for tolerance in (20e-6, "20 ppm", (20, "ppm")):
            result = match_fragments(
                [(300.005, 1.0)],
                candidates,
                tolerance=tolerance,
            )
            self.assertEqual(result["A"].score, 1.0)

    def test_spectrum_object_precision_is_used(self):
        spectrum = FakeSpectrum([(300.0, 1.0)])
        result = match_fragments(
            spectrum, [{"name": "A", "fragments": {"x": 300.002}}]
        )
        self.assertAlmostEqual(result.tolerance, 5e-6)
        self.assertEqual(result["A"].score, 0.0)

        close = match_fragments(
            spectrum, [{"name": "A", "fragments": {"x": 300.00001}}]
        )
        self.assertEqual(close["A"].score, 1.0)

    def test_candidate_order_does_not_change_per_candidate_scores(self):
        peaks = [(300.0, 1.0)]
        first = {"name": "first", "fragments": {"x": 300.0}}
        second = {"name": "second", "fragments": {"x": 300.000001}}
        forward = match_fragments(peaks, [first, second], tolerance=20)
        backward = match_fragments(peaks, [second, first], tolerance=20)
        self.assertEqual(forward.scores, backward.scores)
        self.assertEqual(forward.best_candidates[0].score, 1.0)

    def test_exclusive_rule_reports_tie_instead_of_first_winner(self):
        peaks = [(300.0, 1.0)]
        candidates = [
            {"name": "first", "fragments": {"x": 300.0}},
            {"name": "second", "fragments": {"x": 300.000001}},
        ]
        forward = match_fragments(
            peaks, candidates, tolerance=20, peak_usage="exclusive"
        )
        backward = match_fragments(
            peaks, list(reversed(candidates)), tolerance=20, peak_usage="exclusive"
        )
        self.assertEqual(forward.scores, {"first": 1.0, "second": 1.0})
        self.assertEqual(backward.scores, forward.scores)
        self.assertEqual(forward.n_global_explanations, 2)
        self.assertEqual(backward.n_global_explanations, 2)
        winners = {
            explanation.assignments[0].candidate_name
            for explanation in forward.global_explanations
        }
        self.assertEqual(winners, {"first", "second"})
        self.assertEqual(forward["first"].n_explanations, 2)

    def test_exclusive_rule_uses_weights_not_order(self):
        peaks = [(300.0, 1.0)]
        candidates = [
            {"name": "early", "fragments": {"x": 300.0}},
            {"name": "strong", "fragments": {"x": 300.000001}},
        ]
        result = match_fragments(
            peaks,
            candidates,
            tolerance=20,
            peak_usage="exclusive",
            weights={"early": {"x": 1.0}, "strong": {"x": 3.0}},
        )
        self.assertEqual(result.scores, {"early": 0.0, "strong": 3.0})
        self.assertEqual(result.n_global_explanations, 1)

    def test_per_candidate_rule_allows_shared_peaks(self):
        peaks = [(300.0, 1.0)]
        candidates = [
            {"name": "A", "fragments": {"x": 300.0}},
            {"name": "B", "fragments": {"x": 300.0}},
        ]
        result = match_fragments(
            peaks, candidates, tolerance=20, peak_usage="per_candidate"
        )
        self.assertEqual(result.scores, {"A": 1.0, "B": 1.0})
        self.assertEqual(result.n_global_explanations, 0)

    def test_explanations_are_limited(self):
        result = match_fragments(
            [(300.0, 1.0)],
            [{"name": "A", "fragments": {"x": 300.0, "y": 300.0, "z": 300.0}}],
            tolerance=20,
            max_explanations=2,
        )
        self.assertEqual(result["A"].n_explanations, 2)

    def test_candidate_representations(self):
        by_name = match_fragments(
            [(300.0, 1.0)],
            {"A": {"fragments": {"x": 300.0}}},
            tolerance=20,
        )
        self.assertEqual(by_name.scores, {"A": 1.0})

        bare_fragments = match_fragments(
            [(300.0, 1.0)],
            {"x": 300.0},
            tolerance=20,
        )
        self.assertEqual(bare_fragments["candidate1"].score, 1.0)

        custom = FragmentCandidate(
            name="custom", fragments=(("x", 300.0, 1.0),)
        )
        from_object = match_fragments(
            [(300.0, 1.0)], [custom], tolerance=20
        )
        self.assertEqual(from_object["custom"].score, 1.0)

    def test_single_candidate_helpers(self):
        candidates = [{"name": "A", "fragments": {"x": 300.0}}]
        result_a = match_candidates(candidates, [(300.0, 1.0)], tolerance=20)
        result_b = match_spectrum((300.0, 1.0), candidates, tolerance=20)
        self.assertEqual(result_a.scores, {"A": 1.0})
        self.assertEqual(result_b.scores, {"A": 1.0})

    def test_fragment_matcher_is_reusable(self):
        matcher = FragmentMatcher(
            tolerance=20,
            tolerance_unit="ppm",
            neutral_losses=[NeutralLoss("H2O", 18.0106)],
        )
        first = matcher.match(
            [(233.14, 1.0)], [{"name": "A", "fragments": {"b": 251.1506}}]
        )
        second = matcher.match(
            [(251.1506, 1.0)], [{"name": "A", "fragments": {"b": 251.1506}}]
        )
        self.assertEqual(first["A"].score, 1.0)
        self.assertEqual(second["A"].score, 1.0)
        override = matcher.match(
            [(233.14, 1.0)],
            [{"name": "A", "fragments": {"b": 251.1506}}],
            neutral_losses=[],
        )
        self.assertEqual(override["A"].score, 0.0)

    def test_intensity_weighting(self):
        peaks = [(300.0, 10.0), (400.0, 20.0)]
        candidates = [{"name": "A", "fragments": {"x": 300.0}}]
        unweighted = match_fragments(
            peaks, candidates, tolerance=20, intensity_weighting="none"
        )
        linear = match_fragments(
            peaks, candidates, tolerance=20, intensity_weighting="linear"
        )
        self.assertEqual(unweighted["A"].score, 1.0)
        self.assertEqual(linear["A"].score, 0.5)

    def test_result_serialisation(self):
        result = match_fragments(
            [(300.0, 1.0)], [{"name": "A", "fragments": {"x": 300.0}}], tolerance=20
        )
        payload = result.as_dict()
        self.assertEqual(payload["scores"], {"A": 1.0})
        self.assertEqual(payload["candidates"][0]["assignments"][0]["ion_label"], "x")
        self.assertIn("unmatched_peaks", payload)

    def test_invalid_peak_usage_and_units(self):
        candidates = [{"name": "A", "fragments": {"x": 300.0}}]
        with self.assertRaises(ValueError):
            match_fragments(
                [(300.0, 1.0)], candidates, tolerance=10, peak_usage="bogus"
            )
        with self.assertRaises(ValueError):
            match_fragments(
                [(300.0, 1.0)],
                candidates,
                tolerance=10,
                tolerance_unit="bogus",
            )

    def test_package_exposes_module(self):
        self.assertIs(pymzml.fragment.match_fragments, match_fragments)

    def test_spectrum_method_on_real_ms2(self):
        run = pymzml.run.Reader(
            os.path.join("tests", "data", "BSA1.mzML.gz"),
            MS_precisions={2: 5e-4},
        )
        spectrum = run[3542]
        b_ions = {
            "b2": 251.1503,
            "b3": 350.2187,
            "b4": 465.2456,
            "b5": 594.2882,
            "b6": 691.3410,
        }
        y_ions = {
            "y1": 147.1128,
            "y2": 260.1969,
            "y3": 373.2809,
            "y4": 487.3239,
            "y5": 615.3824,
            "y6": 712.4352,
        }
        candidates = [
            {"name": "target", "fragments": {**b_ions, **y_ions}},
            {"name": "decoy", "fragments": {"x": 251.4, "z": 260.45}},
        ]
        result = spectrum.match_fragments(
            candidates, tolerance=500, peak_usage="per_candidate"
        )
        self.assertEqual(result["target"].score, 11.0)
        self.assertEqual(result["decoy"].score, 0.0)
        self.assertEqual(result["target"].matched_peak_count, 11)
        self.assertTrue(len(result.unmatched_peaks) > 0)

    def test_fragment_match_alias_module(self):
        from pymzml.fragment_match import match_fragments as alias

        self.assertIs(alias, match_fragments)


if __name__ == "__main__":
    unittest.main(verbosity=3)
