#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for collision energy acquisition conditions.
"""
import os
import sys
import unittest
import xml.etree.ElementTree as ElementTree

sys.path.append(os.path.abspath("."))

import pymzml.run as run
import test_file_paths
from pymzml.energy import (
    AcquisitionEnergy,
    EV,
    JOULE,
    PERCENT,
    bucket_conditions,
    bucket_spectra,
    comparable_conditions,
    convert_unit,
    parse_text,
)

NS = "http://psi.hupo.org/ms/mzml"

ACTIVATION = """<activation>{params}</activation>"""

CV = (
    '<cvParam cvRef="MS" accession="{accession}" name="{name}" '
    'value="{value}"{unit}/>'
)
UNIT = (
    ' unitCvRef="UO" unitAccession="{accession}" unitName="{name}"'
)


def make_spectrum(params, spectrum_id="scan=1"):
    xml = (
        '<spectrum xmlns="{ns}" index="0" id="{spectrum_id}" defaultArrayLength="0">'
        "<precursorList count=\"1\"><precursor>"
        "{activation}"
        "</precursor></precursorList>"
        "</spectrum>"
    ).format(
        ns=NS,
        spectrum_id=spectrum_id,
        activation=ACTIVATION.format(params=params),
    )
    spectrum = ElementTree.fromstring(xml)
    from pymzml.spec import Spectrum

    return Spectrum(spectrum)


def cv(accession, name, value, unit_accession=None, unit_name=None):
    unit = ""
    if unit_accession is not None:
        unit = UNIT.format(accession=unit_accession, name=unit_name)
    return CV.format(accession=accession, name=name, value=value, unit=unit)


def collision_energy(value, unit_accession=None, unit_name=None):
    return cv(
        "MS:1000045",
        "collision energy",
        value,
        unit_accession,
        unit_name,
    )


class CollisionEnergyParsingTest(unittest.TestCase):
    def test_fixed_cv_energy(self):
        spec = make_spectrum(
            collision_energy("35", "UO:0000266", "electronvolt")
        )
        self.assertEqual(len(spec.energy_conditions), 1)
        condition = spec.energy_conditions[0]
        self.assertTrue(condition.is_exact)
        self.assertEqual(condition.low, 35.0)
        self.assertEqual(condition.high, 35.0)
        self.assertEqual(condition.unit, EV)
        self.assertEqual(condition.accessions, ("MS:1000045",))
        self.assertEqual(spec.collision_energy, 35.0)
        self.assertEqual(spec.energies, [35.0])

    def test_missing_unit_defaults_to_electronvolt(self):
        spec = make_spectrum(collision_energy("35"))
        condition = spec.energy_conditions[0]
        self.assertEqual(condition.unit, EV)
        self.assertEqual(condition.sources[0].unit, EV)

    def test_ramp_start_and_end_become_one_range(self):
        params = cv(
            "MS:1002013",
            "collision energy ramp start",
            "20",
            "UO:0000266",
            "electronvolt",
        ) + cv(
            "MS:1002014",
            "collision energy ramp end",
            "40",
            "UO:0000266",
            "electronvolt",
        )
        spec = make_spectrum(params)
        self.assertEqual(len(spec.energy_conditions), 1)
        condition = spec.energy_conditions[0]
        self.assertTrue(condition.is_range)
        self.assertFalse(condition.is_stepped)
        self.assertEqual(condition.range, (20.0, 40.0))
        self.assertEqual(
            set(condition.accessions),
            {"MS:1002013", "MS:1002014"},
        )
        self.assertEqual(spec.collision_energy, (20.0, 40.0))

    def test_lone_ramp_endpoint_is_not_a_fixed_energy(self):
        spec = make_spectrum(
            cv(
                "MS:1002013",
                "collision energy ramp start",
                "20",
                "UO:0000266",
                "electronvolt",
            )
        )
        condition = spec.energy_conditions[0]
        self.assertFalse(condition.is_exact)
        self.assertTrue(condition.continuous)
        self.assertEqual(condition.kind, "ramp")
        self.assertEqual(spec.collision_energy, (20.0, 20.0))
        self.assertTrue(
            all(bucket.possible for bucket in spec.energy_buckets(width=10))
        )

    def test_repeated_fixed_energies_become_stepped_mixture(self):
        params = (
            collision_energy("20", "UO:0000266", "electronvolt")
            + collision_energy("30", "UO:0000266", "electronvolt")
            + collision_energy("40", "UO:0000266", "electronvolt")
        )
        spec = make_spectrum(params)
        condition = spec.energy_conditions[0]
        self.assertTrue(condition.is_range)
        self.assertTrue(condition.is_stepped)
        self.assertEqual(condition.steps, (20.0, 30.0, 40.0))
        self.assertEqual(condition.range, (20.0, 40.0))
        self.assertEqual(len(condition.sources), 3)

    def test_percent_collision_energy_keeps_percent_unit(self):
        spec = make_spectrum(
            cv("MS:1000138", "percent collision energy", "35")
        )
        condition = spec.energy_conditions[0]
        self.assertEqual(condition.unit, PERCENT)
        self.assertEqual(condition.low, 35.0)
        with self.assertRaises(ValueError):
            condition.to(EV)

    def test_activation_energy_is_not_collision_energy(self):
        spec = make_spectrum(
            cv(
                "MS:1000509",
                "activation energy",
                "0",
                "UO:0000266",
                "electronvolt",
            )
        )
        self.assertEqual(spec.energy_conditions, [])
        self.assertIsNone(spec.collision_energy)

    def test_user_param_value_is_captured(self):
        spec = make_spectrum(
            '<userParam name="collision energy" type="xsd:string" value="35"/>'
        )
        condition = spec.energy_conditions[0]
        self.assertEqual(condition.low, 35.0)
        # No CV/unit declaration: the value is retained without guessing eV.
        self.assertIsNone(condition.unit)
        self.assertIn("userParam", condition.source_kinds)
        self.assertIn(
            "userParam", [source.kind for source in condition.sources]
        )

    def test_filter_string_is_kept_as_its_own_source(self):
        spec = make_spectrum(
            '<userParam name="filter string" type="xsd:string" '
            'value="ITMS + c NSI d w Full ms2 457.72@cid35.00 [115.00-930.00]"/>'
        )
        self.assertEqual(len(spec.energy_conditions), 1)
        condition = spec.energy_conditions[0]
        self.assertEqual(condition.low, 35.0)
        self.assertEqual(condition.source_kinds, ("filter string",))

    def test_cv_and_user_param_sources_are_merged(self):
        spec = make_spectrum(
            collision_energy("35", "UO:0000266", "electronvolt")
            + '<userParam name="collision energy" type="xsd:string" value="35"/>'
            + '<userParam name="filter string" value="ms2 457.72@cid35.00"/>'
        )
        # The CV term establishes the unit; identical vendor values are kept
        # as corroborating sources rather than separate conditions.
        self.assertEqual(len(spec.energy_conditions), 1)
        condition = spec.energy_conditions[0]
        self.assertEqual(condition.unit, EV)
        self.assertEqual(
            set(condition.source_kinds),
            {"cvParam", "userParam", "filter string"},
        )

    def test_undeclared_vendor_values_stay_unknown_unit(self):
        spec = make_spectrum(
            '<userParam name="collision energy" type="xsd:string" value="35"/>'
            + '<userParam name="filter string" value="ms2 457.72@cid35.00"/>'
        )
        self.assertEqual(len(spec.energy_conditions), 1)
        condition = spec.energy_conditions[0]
        self.assertIsNone(condition.unit)
        self.assertEqual(
            set(condition.source_kinds), {"userParam", "filter string"}
        )

    def test_vendor_peak_parameter_is_not_a_scan_energy(self):
        spec = make_spectrum(
            '<userParam name="parameter: spectra:select_collision_energy" '
            'type="xsd:string" value=":"/>'
        )
        self.assertEqual(spec.energy_conditions, [])

    def test_missing_energy_has_no_condition(self):
        spec = make_spectrum("")
        self.assertEqual(spec.energy_conditions, [])
        self.assertIsNone(spec.collision_energy)
        self.assertEqual(spec.energies, [])


class CollisionEnergySemanticsTest(unittest.TestCase):
    def test_single_value_inside_range_is_not_equal_to_range(self):
        fixed = AcquisitionEnergy(30.0)
        mixture = AcquisitionEnergy(20.0, 40.0)
        self.assertNotEqual(fixed, mixture)
        self.assertFalse(fixed.matches(mixture))
        self.assertFalse(mixture.matches(fixed))
        # Interval membership is a different question from identity.
        self.assertTrue(mixture.contains(30.0))
        self.assertTrue(mixture.covers(fixed))

    def test_exact_conditions_compare_by_value_and_unit(self):
        self.assertEqual(AcquisitionEnergy(35.0), AcquisitionEnergy(35.0))
        self.assertEqual(AcquisitionEnergy(35.0, unit=EV), 35.0)
        self.assertNotEqual(
            AcquisitionEnergy(35.0, unit=EV),
            AcquisitionEnergy(35.0, unit=JOULE),
        )

    def test_unit_conversion(self):
        joule = convert_unit(1.0, EV, JOULE)
        self.assertAlmostEqual(joule, 1.602176634e-19)
        self.assertAlmostEqual(convert_unit(joule, JOULE, EV), 1.0)

    def test_comparable_conditions_convert_to_common_unit(self):
        conditions = [
            AcquisitionEnergy(1.0, unit=EV),
            AcquisitionEnergy(1.602176634e-19, unit=JOULE),
        ]
        comparable = comparable_conditions(conditions, unit=EV)
        self.assertEqual(len(comparable), 1)
        self.assertEqual(comparable[0].unit, EV)
        self.assertAlmostEqual(comparable[0].low, 1.0)

    def test_percent_is_not_converted_to_energy(self):
        with self.assertRaises(ValueError):
            comparable_conditions(
                [AcquisitionEnergy(30.0, unit=PERCENT)], unit=EV
            )

    def test_stepped_and_range_are_distinct(self):
        stepped = AcquisitionEnergy(20.0, 40.0, steps=(20.0, 40.0))
        continuous = AcquisitionEnergy(20.0, 40.0)
        self.assertNotEqual(stepped, continuous)
        self.assertTrue(stepped.contains(20.0))
        self.assertFalse(stepped.contains(30.0))
        self.assertTrue(continuous.contains(30.0))


class EnergyBucketTest(unittest.TestCase):
    def test_fixed_energy_is_definite(self):
        buckets = bucket_conditions([AcquisitionEnergy(35.0)], width=10)
        self.assertEqual(len(buckets), 1)
        bucket = buckets[0]
        self.assertTrue(bucket.definite)
        self.assertFalse(bucket.possible)
        self.assertEqual(bucket.low, 30.0)
        self.assertEqual(bucket.high, 40.0)

    def test_range_only_makes_buckets_possible(self):
        buckets = bucket_conditions(
            [AcquisitionEnergy(20.0, 40.0)], width=10
        )
        self.assertEqual([bucket.low for bucket in buckets], [20.0, 30.0, 40.0])
        for bucket in buckets:
            self.assertFalse(bucket.definite)
            self.assertTrue(bucket.possible)

    def test_single_value_in_range_does_not_make_bucket_definite(self):
        buckets = bucket_conditions(
            [AcquisitionEnergy(20.0, 40.0)], width=10
        )
        covering = [bucket for bucket in buckets if bucket.low == 30.0]
        self.assertEqual(len(covering), 1)
        self.assertEqual(covering[0].membership, "possible")
        self.assertFalse(covering[0].definite)

    def test_fixed_energy_inside_range_bucket_marks_both(self):
        buckets = bucket_conditions(
            [AcquisitionEnergy(20.0, 40.0), AcquisitionEnergy(35.0)],
            width=10,
        )
        covering = [bucket for bucket in buckets if bucket.low == 30.0]
        self.assertEqual(len(covering), 1)
        self.assertTrue(covering[0].definite)
        self.assertTrue(covering[0].possible)

    def test_stepped_mixture_only_touches_its_steps(self):
        condition = AcquisitionEnergy(
            20.0, 40.0, steps=(20.0, 30.0, 40.0)
        )
        buckets = bucket_conditions([condition], width=5)
        self.assertEqual(
            [bucket.low for bucket in buckets], [20.0, 30.0, 40.0]
        )
        # A mixture is still a mixture: its steps are possible, not definite.
        self.assertTrue(all(bucket.possible for bucket in buckets))
        self.assertFalse(any(bucket.definite for bucket in buckets))
        self.assertFalse(
            any(bucket.low == 25.0 for bucket in buckets)
        )

    def test_bucket_width_is_respected(self):
        buckets = bucket_conditions(
            [AcquisitionEnergy(20.0, 40.0)], width=2.5, origin=0
        )
        self.assertEqual(
            [bucket.low for bucket in buckets],
            [20.0, 22.5, 25.0, 27.5, 30.0, 32.5, 35.0, 37.5, 40.0],
        )

    def test_bucket_spectra_splits_membership(self):
        fixed = make_spectrum(
            collision_energy("15", "UO:0000266", "electronvolt"),
            spectrum_id="scan=1",
        )
        mixture = make_spectrum(
            cv(
                "MS:1002013",
                "collision energy ramp start",
                "10",
                "UO:0000266",
                "electronvolt",
            )
            + cv(
                "MS:1002014",
                "collision energy ramp end",
                "20",
                "UO:0000266",
                "electronvolt",
            ),
            spectrum_id="scan=2",
        )
        result = bucket_spectra([fixed, mixture], width=10)
        self.assertEqual(
            result[
                next(bucket for bucket in result if bucket.low == 10.0)
            ]["definite"],
            [1],
        )
        possible = result[
            next(bucket for bucket in result if bucket.low == 10.0)
        ]["possible"]
        self.assertEqual(possible, [2])


class EnergyTextTest(unittest.TestCase):
    def test_parse_fixed(self):
        condition = parse_text("35")
        self.assertTrue(condition.is_exact)
        self.assertEqual(condition.low, 35.0)
        self.assertEqual(condition.unit, EV)

    def test_parse_unit_suffix(self):
        condition = parse_text("35 eV")
        self.assertEqual(condition.unit, EV)
        self.assertEqual(condition.low, 35.0)

    def test_parse_range(self):
        condition = parse_text("25-40 electronvolt")
        self.assertTrue(condition.is_range)
        self.assertFalse(condition.is_stepped)
        self.assertEqual(condition.range, (25.0, 40.0))

    def test_parse_stepped_list(self):
        condition = parse_text("25, 30, 35 eV")
        self.assertTrue(condition.is_stepped)
        self.assertEqual(condition.steps, (25.0, 30.0, 35.0))

    def test_parse_percent(self):
        condition = parse_text("30%")
        self.assertEqual(condition.unit, PERCENT)
        self.assertEqual(condition.low, 30.0)


class RealFileEnergyTest(unittest.TestCase):
    def setUp(self):
        self.reader = run.Reader(test_file_paths.paths[9])
        self.spectrum = self.reader[2548]

    def test_real_spectrum_has_collision_energy(self):
        self.assertTrue(self.spectrum.has_energy_conditions)
        self.assertEqual(len(self.spectrum.energy_conditions), 1)
        condition = self.spectrum.energy_conditions[0]
        self.assertTrue(condition.is_exact)
        self.assertEqual(condition.low, 35.0)
        # BSA1 only records userParam/filter-string energies; neither declares
        # a unit, so no eV conversion is assumed.
        self.assertIsNone(condition.unit)
        self.assertIn("userParam", condition.source_kinds)
        self.assertIn("filter string", condition.source_kinds)

    def test_real_spectrum_can_be_assumed_electronvolt(self):
        conditions = comparable_conditions(
            self.spectrum.energy_conditions, unit=EV, assume_unit=EV
        )
        self.assertEqual(len(conditions), 1)
        self.assertEqual(conditions[0].unit, EV)

    def test_all_sources_are_preserved(self):
        condition = self.spectrum.energy_conditions[0]
        self.assertIn("userParam", condition.source_kinds)
        self.assertIn("filter string", condition.source_kinds)

    def test_real_spectrum_bucket_is_definite(self):
        buckets = self.spectrum.energy_buckets(width=10)
        self.assertEqual(len(buckets), 1)
        self.assertTrue(buckets[0].definite)
        self.assertEqual(buckets[0].low, 30.0)


if __name__ == "__main__":
    unittest.main(verbosity=3)
