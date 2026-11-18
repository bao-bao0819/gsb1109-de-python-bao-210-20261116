import sys
import os
import unittest

sys.path.append(os.path.abspath("."))

import numpy as np

import pymzml.run as run
import test_file_paths
from pymzml.isotopic_envelopes import (
    NEUTRON_MASS,
    EnvelopeExplanation,
    IsotopicEnvelope,
    IsotopicEnvelopeFinder,
    IsotopeMember,
)
from pymzml.spec import PROTON


class IsotopicEnvelopeFinderTest(unittest.TestCase):
    def setUp(self):
        self.spacing_z2 = NEUTRON_MASS / 2
        self.spacing_z3 = NEUTRON_MASS / 3
        self.finder = IsotopicEnvelopeFinder(
            charges=[2, 3],
            mz_tolerance=0.02,
            max_missing_peaks=1,
            max_consecutive_missing=1,
            min_isotopes=2,
        )

    def test_isotope_spacings(self):
        self.assertEqual(
            self.finder.isotope_spacings,
            {2: NEUTRON_MASS / 2, 3: NEUTRON_MASS / 3},
        )

    def test_enumerate_envelope_and_members(self):
        m0 = 500.0
        peaks = [
            (m0, 100.0),
            (m0 + self.spacing_z2, 60.0),
            (m0 + 2 * self.spacing_z2, 30.0),
        ]
        envelopes = self.finder.find(peaks)
        full = [
            envelope
            for envelope in envelopes
            if envelope.peak_indices == (0, 1, 2)
            and envelope.missing_indices == ()
        ]
        self.assertEqual(len(full), 1)
        envelope = full[0]
        self.assertEqual(envelope.charge, 2)
        self.assertEqual(envelope.monoisotopic_mz, m0)
        self.assertEqual(
            [member.isotope_index for member in envelope.members], [0, 1, 2]
        )
        self.assertEqual(envelope.n_observed, 3)
        self.assertEqual(envelope.n_missing, 0)
        self.assertEqual(envelope.n_isotopes, 3)
        self.assertAlmostEqual(envelope.intensity, 190.0)
        self.assertIsInstance(envelope.members[0], IsotopeMember)

    def test_missing_peak_rule(self):
        m0 = 500.0
        peaks = [(m0, 100.0), (m0 + 2 * self.spacing_z2, 30.0)]
        envelopes = self.finder.find(peaks)
        with_gap = [
            envelope
            for envelope in envelopes
            if envelope.missing_indices == (1,)
            and envelope.peak_indices == (0, 1)
        ]
        self.assertEqual(len(with_gap), 1)

        strict_finder = IsotopicEnvelopeFinder(
            charges=[2],
            mz_tolerance=0.02,
            max_missing_peaks=0,
            max_consecutive_missing=0,
            min_isotopes=2,
        )
        self.assertEqual(strict_finder.find(peaks), [])

    def test_consecutive_missing_rule(self):
        m0 = 500.0
        peaks = [(m0, 100.0), (m0 + 3 * self.spacing_z2, 30.0)]
        allowing_two_gaps = IsotopicEnvelopeFinder(
            charges=[2],
            mz_tolerance=0.02,
            max_missing_peaks=2,
            max_consecutive_missing=2,
            min_isotopes=2,
        )
        one_gap_max = IsotopicEnvelopeFinder(
            charges=[2],
            mz_tolerance=0.02,
            max_missing_peaks=2,
            max_consecutive_missing=1,
            min_isotopes=2,
        )
        self.assertTrue(allowing_two_gaps.find(peaks))
        self.assertEqual(one_gap_max.find(peaks), [])

    def test_min_isotopes_threshold(self):
        m0 = 500.0
        peaks = [(m0, 100.0)]
        single_finder = IsotopicEnvelopeFinder(
            charges=[2], mz_tolerance=0.02, min_isotopes=1
        )
        pair_finder = IsotopicEnvelopeFinder(
            charges=[2], mz_tolerance=0.02, min_isotopes=2
        )
        self.assertEqual(len(single_finder.find(peaks)), 1)
        self.assertEqual(pair_finder.find(peaks), [])

    def test_charge_specific_spacing(self):
        peaks = [(600.0, 100.0), (600.0 + self.spacing_z3, 70.0)]
        envelopes = self.finder.find(peaks)
        self.assertEqual(
            set(envelope.charge for envelope in envelopes), {3}
        )

    def test_neutral_mass(self):
        peaks = [(600.0, 100.0), (600.0 + self.spacing_z3, 70.0)]
        envelope = self.finder.find(peaks)[0]
        expected_mass = (600.0 - 1.00727646677) * 3
        self.assertAlmostEqual(envelope.neutral_mass, expected_mass)

    def test_selects_non_overlapping_species(self):
        peaks = [
            (500.0, 100.0),
            (500.0 + self.spacing_z2, 60.0),
            (800.0, 100.0),
            (800.0 + self.spacing_z2, 60.0),
        ]
        explanations = self.finder.select_explanations(peaks)
        self.assertEqual(len(explanations), 1)
        explanation = explanations[0]
        self.assertEqual(explanation.n_envelopes, 2)
        self.assertEqual(
            explanation.covered_peak_indices, frozenset({0, 1, 2, 3})
        )
        self.assertEqual(explanation.n_unassigned_peaks, 0)
        claimed = [
            peak_index
            for envelope in explanation.envelopes
            for peak_index in envelope.peak_indices
        ]
        self.assertEqual(len(claimed), len(set(claimed)))

    def test_mutually_exclusive_envelopes_never_share_peak(self):
        m0 = 500.0
        peaks = [
            (m0, 100.0),
            (m0 + self.spacing_z2, 80.0),
            (m0 + 2 * self.spacing_z2, 60.0),
        ]
        explanations = self.finder.select_explanations(peaks)
        for explanation in explanations:
            claimed = [
                peak_index
                for envelope in explanation.envelopes
                for peak_index in envelope.peak_indices
            ]
            self.assertEqual(len(claimed), len(set(claimed)))

    def test_tied_charges_are_both_retained(self):
        peaks = [(500.0, 100.0), (500.5, 80.0)]
        envelope_z2 = IsotopicEnvelope(
            charge=2,
            monoisotopic_mz=500.0,
            peak_indices=(0, 1),
            missing_indices=(),
            members=(),
            intensity=180.0,
            mean_mass_error_ppm=0.5,
            score=0.0,
        )
        envelope_z3 = IsotopicEnvelope(
            charge=3,
            monoisotopic_mz=500.0,
            peak_indices=(0, 1),
            missing_indices=(),
            members=(),
            intensity=180.0,
            mean_mass_error_ppm=0.5,
            score=0.0,
        )
        envelope_z2 = envelope_z2._replace(
            score=self.finder.score_envelope(envelope_z2)
        )
        envelope_z3 = envelope_z3._replace(
            score=self.finder.score_envelope(envelope_z3)
        )
        self.assertEqual(envelope_z2.score, envelope_z3.score)

        winners = self.finder.select_explanations(
            peaks, [envelope_z2, envelope_z3]
        )
        self.assertEqual(len(winners), 2)
        self.assertEqual(
            sorted(winner.envelopes[0].charge for winner in winners),
            [2, 3],
        )
        for winner in winners:
            self.assertEqual(winner.n_envelopes, 1)

    def test_result_independent_of_input_order(self):
        peaks = [
            (500.0, 100.0),
            (500.0 + self.spacing_z2, 60.0),
            (800.0, 100.0),
            (800.0 + self.spacing_z2, 60.0),
        ]

        def signature(explanation):
            return sorted(
                (envelope.charge, round(envelope.monoisotopic_mz, 4))
                for envelope in explanation.envelopes
            )

        forward = self.finder.select_explanations(peaks)[0]
        reversed_input = self.finder.select_explanations(list(reversed(peaks)))[0]
        self.assertEqual(signature(forward), signature(reversed_input))

    def test_empty_input(self):
        self.assertEqual(self.finder.find([]), [])
        explanations = self.finder.select_explanations([])
        self.assertEqual(len(explanations), 1)
        self.assertIsInstance(explanations[0], EnvelopeExplanation)
        self.assertEqual(explanations[0].n_envelopes, 0)
        self.assertEqual(explanations[0].n_unassigned_peaks, 0)
        self.assertEqual(explanations[0].score, 0.0)

    def test_accepts_unsorted_numpy_peaks(self):
        peaks = np.array(
            [(600.0 + self.spacing_z3, 70.0), (600.0, 100.0)]
        )
        envelopes = self.finder.find(peaks)
        self.assertTrue(envelopes)
        envelopes, explanations = self.finder.assign(peaks)
        self.assertTrue(envelopes)
        self.assertTrue(explanations)

    def test_invalid_constructor_arguments(self):
        invalid_kwargs = (
            {"charges": []},
            {"charges": [1, -1]},
            {"charges": [2, 2]},
            {"mz_tolerance": 0},
            {"max_missing_peaks": -1},
            {"max_consecutive_missing": -1},
            {"min_isotopes": 0},
            {"min_isotopes": 5, "max_isotopes": 3},
        )
        for kwargs in invalid_kwargs:
            with self.assertRaises(ValueError):
                IsotopicEnvelopeFinder(**kwargs)


class SpectrumIsotopicEnvelopeTest(unittest.TestCase):
    def setUp(self):
        reader = run.Reader(test_file_paths.paths[9])
        self.spectrum = reader[2548]

    def test_spectrum_delegates_envelope_assignment(self):
        monoisotopic_mz = 430.313
        peaks = np.array(
            [
                (monoisotopic_mz, 100.0),
                (monoisotopic_mz + PROTON / 3, 49.0),
                (monoisotopic_mz + 2 * PROTON / 3, 20.0),
            ]
        )
        self.spectrum.set_peaks(peaks, "centroided")
        envelopes, explanations = self.spectrum.find_isotopic_envelopes(
            charges=[2, 3], mz_tolerance=0.02
        )
        full_envelope = [
            envelope
            for envelope in envelopes
            if envelope.charge == 3
            and envelope.peak_indices == (0, 1, 2)
        ]
        self.assertEqual(len(full_envelope), 1)
        self.assertEqual(
            explanations[0].covered_peak_indices, frozenset({0, 1, 2})
        )
        self.assertEqual(explanations[0].n_unassigned_peaks, 0)


if __name__ == "__main__":
    unittest.main(verbosity=3)
