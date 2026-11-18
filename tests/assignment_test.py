import os
import sys

sys.path.append(os.path.abspath("."))

import unittest

from pymzml.assignment import (
    Adduct,
    AmbiguousGroup,
    AssignmentEngine,
    CandidateMolecule,
    CONVENTION_ELECTRON,
    CONVENTION_PROTON,
    ELECTRON_MASS,
    FeasibleInterpretation,
    Precursor,
    PROTON_MASS,
    TOLERANCE_DA,
    UniqueClaim,
    UnassignedPrecursor,
    assign,
    protonation_ladder,
    theoretical_mz,
)


class AdductTheoryTest(unittest.TestCase):
    def test_protonation_ladder_positive(self):
        adducts = protonation_ladder([2, 1])
        self.assertEqual(len(adducts), 2)
        singly, doubly = adducts
        self.assertEqual(singly.charge, 1)
        self.assertEqual(singly.name, "[M+H]+")
        self.assertAlmostEqual(singly.mass_shift, PROTON_MASS)
        self.assertEqual(doubly.charge, 2)
        self.assertEqual(doubly.name, "[M+2H]2+")
        self.assertAlmostEqual(doubly.mass_shift, 2 * PROTON_MASS)

    def test_protonation_ladder_negative(self):
        adducts = protonation_ladder([-1, -2])
        self.assertEqual([a.name for a in adducts], ["[M-H]-", "[M-2H]2-"])
        self.assertAlmostEqual(adducts[0].mass_shift, -PROTON_MASS)

    def test_theoretical_mz_proton_convention(self):
        adduct = Adduct("[M+2H]2+", 2, 2 * PROTON_MASS)
        mz = theoretical_mz(100.0, adduct)
        self.assertAlmostEqual(mz, (100.0 + 2 * PROTON_MASS) / 2)

    def test_theoretical_mz_electron_convention(self):
        adduct = Adduct("M+", 1, 0.0, convention=CONVENTION_ELECTRON)
        mz = theoretical_mz(100.0, adduct)
        self.assertAlmostEqual(mz, 100.0 - ELECTRON_MASS)

    def test_zero_charge_rejected(self):
        with self.assertRaises(ValueError):
            Adduct("bad", 0, 0.0)

    def test_unknown_convention_rejected(self):
        with self.assertRaises(ValueError):
            Adduct("bad", 1, 0.0, convention="neutrino")


class InputValidationTest(unittest.TestCase):
    def test_charge_polarity_consistency(self):
        with self.assertRaises(ValueError):
            Precursor("p1", "event-1", 100.0, (-1,), polarity=1)
        with self.assertRaises(ValueError):
            Precursor("p2", "event-1", 100.0, (1,), polarity=-1)

    def test_empty_charge_list_rejected(self):
        with self.assertRaises(ValueError):
            Precursor("p1", "event-1", 100.0, ())

    def test_non_positive_mz_rejected(self):
        with self.assertRaises(ValueError):
            Precursor("p1", "event-1", 0.0, (1,))

    def test_duplicate_molecule_id_rejected(self):
        molecules = [
            CandidateMolecule("M1", 100.0),
            CandidateMolecule("M1", 200.0),
        ]
        with self.assertRaises(ValueError):
            AssignmentEngine(molecules)

    def test_duplicate_precursor_id_rejected(self):
        engine = AssignmentEngine([CandidateMolecule("M1", 100.0)])
        with self.assertRaises(ValueError):
            engine.assign(
                [
                    Precursor("p1", "e1", 101.0, (1,)),
                    Precursor("p1", "e2", 102.0, (1,)),
                ]
            )

    def test_bad_tolerance_mode_rejected(self):
        with self.assertRaises(ValueError):
            CandidateMolecule("M1", 100.0, tolerance=5.0, tolerance_mode="xx")
        with self.assertRaises(ValueError):
            AssignmentEngine(
                [CandidateMolecule("M1", 100.0)],
                default_tolerance_mode="xx",
            )


class EnumerationTest(unittest.TestCase):
    def setUp(self):
        self.m1 = CandidateMolecule("M1", 100.0)
        self.m2 = CandidateMolecule("M2", 200.0001)
        self.engine = AssignmentEngine(
            [self.m1, self.m2], default_tolerance=5.0
        )

    def test_full_grid_enumerated_without_charge_first_pruning(self):
        # m/z 101.0073 supports M1 as [M+H]+ AND M2 as [M+2H]2+
        precursor = Precursor("p1", "event-1", 101.0073, (1, 2))
        interpretations = self.engine.enumerate_interpretations([precursor])
        pairs = {(i.molecule_id, i.charge) for i in interpretations}
        self.assertIn(("M1", 1), pairs)
        self.assertIn(("M2", 2), pairs)
        for interpretation in interpretations:
            self.assertIsInstance(interpretation, FeasibleInterpretation)
            self.assertLessEqual(
                abs(interpretation.mass_error_ppm), 5.0
            )
            self.assertEqual(
                interpretation.isolation_event_id, "event-1"
            )
            self.assertEqual(
                interpretation.convention, CONVENTION_PROTON
            )

    def test_mass_error_retained_da_and_ppm(self):
        precursor = Precursor("p1", "event-1", 101.0073, (1,))
        interpretations = self.engine.enumerate_interpretations([precursor])
        self.assertEqual(len(interpretations), 1)
        item = interpretations[0]
        expected = 100.0 + PROTON_MASS
        self.assertAlmostEqual(item.theoretical_mz, expected)
        self.assertAlmostEqual(
            item.mass_error_da, 101.0073 - expected
        )
        self.assertAlmostEqual(
            item.mass_error_ppm, (101.0073 - expected) / expected * 1e6
        )
        self.assertEqual(item.tolerance, 5.0)

    def test_per_molecule_da_tolerance(self):
        molecule = CandidateMolecule(
            "Mda", 100.0, tolerance=0.01, tolerance_mode=TOLERANCE_DA
        )
        engine = AssignmentEngine([molecule], default_tolerance=1.0)
        inside = Precursor("in", "e1", 100.0 + PROTON_MASS + 0.009, (1,))
        outside = Precursor("out", "e2", 100.0 + PROTON_MASS + 0.02, (1,))
        result = engine.assign([inside, outside])
        self.assertEqual(len(result.interpretations), 1)
        self.assertEqual(result.interpretations[0].precursor_id, "in")
        self.assertEqual(len(result.unassigned), 1)
        self.assertIsInstance(result.unassigned[0], UnassignedPrecursor)
        self.assertEqual(result.unassigned[0].precursor.precursor_id, "out")

    def test_negative_mode_adduct(self):
        molecule = CandidateMolecule("Mneg", 100.0)
        engine = AssignmentEngine([molecule], default_tolerance=5.0)
        precursor = Precursor(
            "p-", "e1", 100.0 - PROTON_MASS, (-1,), polarity=-1
        )
        interpretations = engine.enumerate_interpretations([precursor])
        self.assertEqual(len(interpretations), 1)
        self.assertEqual(interpretations[0].adduct, "[M-H]-")
        self.assertEqual(interpretations[0].charge, -1)

    def test_electron_convention_engine(self):
        molecule = CandidateMolecule("Mr", 100.0)
        engine = AssignmentEngine(
            [molecule],
            adduct_convention=CONVENTION_ELECTRON,
            default_tolerance=5.0,
        )
        precursor = Precursor(
            "pr", "e1", 100.0 - ELECTRON_MASS, (1,)
        )
        interpretations = engine.enumerate_interpretations([precursor])
        self.assertEqual(len(interpretations), 1)
        self.assertEqual(interpretations[0].adduct, "M+")

    def test_custom_adduct_list_and_charge_filtering(self):
        molecule = CandidateMolecule("Mc", 100.0)
        adducts = [
            Adduct("[M+H]+", 1, PROTON_MASS),
            Adduct("[M+Na]+", 1, 22.989218),
        ]
        engine = AssignmentEngine(
            [molecule], adducts=adducts, default_tolerance=5.0
        )
        precursor = Precursor(
            "p1", "e1", 100.0 + 22.989218, (1, 2)
        )
        interpretations = engine.enumerate_interpretations([precursor])
        self.assertEqual(len(interpretations), 1)
        self.assertEqual(interpretations[0].adduct, "[M+Na]+")
        # charge 2 has no adduct in the custom list, so nothing is invented
        self.assertTrue(
            all(item.charge == 1 for item in interpretations)
        )

    def test_callable_adducts(self):
        def pick(precursor):
            return protonation_ladder(precursor.charges)

        molecule = CandidateMolecule("M1", 100.0)
        engine = AssignmentEngine(
            [molecule],
            adducts=pick,
            default_tolerance=5.0,
        )
        precursor = Precursor("p1", "e1", 100.0 + PROTON_MASS, (1,))
        interpretations = engine.enumerate_interpretations([precursor])
        self.assertEqual(len(interpretations), 1)


class AssignmentTest(unittest.TestCase):
    def test_unique_claim_single_interpretation(self):
        molecule = CandidateMolecule("M1", 100.0)
        engine = AssignmentEngine([molecule], default_tolerance=5.0)
        precursor = Precursor("p1", "event-1", 101.0073, (1, 2))
        result = engine.assign([precursor])
        self.assertEqual(len(result.unique), 1)
        self.assertEqual(result.ambiguous_groups, [])
        self.assertEqual(result.unassigned, [])
        claim = result.unique[0]
        self.assertIsInstance(claim, UniqueClaim)
        self.assertEqual(claim.molecule.molecule_id, "M1")
        self.assertEqual(claim.precursor_ids, ["p1"])
        self.assertEqual(claim.isolation_event_ids, ["event-1"])
        self.assertFalse(claim.charge_adduct_ambiguous)
        self.assertEqual(len(claim.interpretations), 1)
        self.assertEqual(claim.interpretations[0].charge, 1)

    def test_unique_claim_with_multiple_charge_adduct_variants(self):
        molecule = CandidateMolecule("M1", 100.0)
        near_proton = Adduct("[M+H']+", 1, 1.0073)
        engine = AssignmentEngine(
            [molecule],
            adducts=[Adduct("[M+H]+", 1, PROTON_MASS), near_proton],
            default_tolerance=5.0,
        )
        precursor = Precursor("p1", "event-1", 101.0073, (1,))
        result = engine.assign([precursor])
        self.assertEqual(len(result.unique), 1)
        claim = result.unique[0]
        self.assertEqual(claim.molecule.molecule_id, "M1")
        self.assertTrue(claim.charge_adduct_ambiguous)
        self.assertEqual(len(claim.interpretations), 2)

    def test_several_precursors_sharing_one_molecule_form_one_claim(self):
        molecule = CandidateMolecule("M1", 100.0)
        engine = AssignmentEngine([molecule], default_tolerance=5.0)
        precursors = [
            Precursor("p1", "event-1", 101.0073, (1,)),
            Precursor("p2", "event-2", 101.0072, (1,)),
        ]
        result = engine.assign(precursors)
        self.assertEqual(len(result.unique), 1)
        self.assertEqual(result.unique[0].precursor_ids, ["p1", "p2"])

    def test_precursor_matching_two_molecules_is_ambiguous(self):
        m1 = CandidateMolecule("M1", 100.0)
        m2 = CandidateMolecule("M2", 100.0005)
        engine = AssignmentEngine([m1, m2], default_tolerance=10.0)
        precursor = Precursor("p1", "event-1", 101.0075, (1,))
        result = engine.assign([precursor])
        self.assertEqual(result.unique, [])
        self.assertEqual(len(result.ambiguous_groups), 1)
        group = result.ambiguous_groups[0]
        self.assertIsInstance(group, AmbiguousGroup)
        self.assertEqual(group.molecule_ids, ["M1", "M2"])
        self.assertEqual(group.precursor_ids, ["p1"])
        self.assertEqual(len(group.interpretations), 2)
        self.assertTrue(
            any("multiple candidate molecules" in r for r in group.reasons)
        )

    def test_different_charges_pointing_at_different_molecules_stay_ambiguous(self):
        # The core requirement: do not pick z=1 first and call M2 undetected.
        m1 = CandidateMolecule("M1", 100.0)
        m2 = CandidateMolecule("M2", 200.0001)
        engine = AssignmentEngine([m1, m2], default_tolerance=5.0)
        precursor = Precursor("p1", "event-1", 101.0073, (1, 2))
        result = engine.assign([precursor])
        self.assertEqual(result.unique, [])
        self.assertEqual(len(result.ambiguous_groups), 1)
        group = result.ambiguous_groups[0]
        self.assertEqual(group.molecule_ids, ["M1", "M2"])
        self.assertEqual(
            {(i.molecule_id, i.charge) for i in group.interpretations},
            {("M1", 1), ("M2", 2)},
        )

    def test_shared_molecule_chains_precursors_into_one_group(self):
        m1 = CandidateMolecule("M1", 100.0)
        m2 = CandidateMolecule("M2", 100.0005)
        engine = AssignmentEngine([m1, m2], default_tolerance=4.0)
        precursors = [
            Precursor("p1", "event-1", 101.0075, (1,)),  # M1 and M2
            Precursor("p2", "event-2", 101.0073, (1,)),  # M1 only
        ]
        result = engine.assign(precursors)
        self.assertEqual(len(result.ambiguous_groups), 1)
        group = result.ambiguous_groups[0]
        self.assertEqual(group.molecule_ids, ["M1", "M2"])
        self.assertEqual(group.precursor_ids, ["p1", "p2"])

    def test_isolation_event_blocks_two_unique_claims_by_default(self):
        m1 = CandidateMolecule("M1", 100.0)
        m2 = CandidateMolecule("M2", 200.0)
        engine = AssignmentEngine(
            [m1, m2],
            default_tolerance=5.0,
            allow_multiple_molecules_per_event=False,
        )
        precursors = [
            Precursor("p1", "event-1", 101.0073, (1,)),
            Precursor("p2", "event-1", 201.0073, (1,)),
        ]
        result = engine.assign(precursors)
        self.assertEqual(result.unique, [])
        self.assertEqual(len(result.ambiguous_groups), 1)
        group = result.ambiguous_groups[0]
        self.assertEqual(group.molecule_ids, ["M1", "M2"])
        self.assertEqual(group.isolation_event_ids, ["event-1"])
        self.assertTrue(
            any("isolation event" in r for r in group.reasons)
        )

    def test_isolation_event_allows_multiple_molecules_when_declared(self):
        m1 = CandidateMolecule("M1", 100.0)
        m2 = CandidateMolecule("M2", 200.0)
        engine = AssignmentEngine(
            [m1, m2],
            default_tolerance=5.0,
            allow_multiple_molecules_per_event=True,
        )
        precursors = [
            Precursor("p1", "event-1", 101.0073, (1,)),
            Precursor("p2", "event-1", 201.0073, (1,)),
        ]
        result = engine.assign(precursors)
        self.assertEqual(len(result.unique), 2)
        self.assertEqual(result.ambiguous_groups, [])
        claimed = {claim.molecule.molecule_id for claim in result.unique}
        self.assertEqual(claimed, {"M1", "M2"})

    def test_distinct_events_allow_independent_unique_claims(self):
        m1 = CandidateMolecule("M1", 100.0)
        m2 = CandidateMolecule("M2", 200.0)
        engine = AssignmentEngine([m1, m2], default_tolerance=5.0)
        precursors = [
            Precursor("p1", "event-1", 101.0073, (1,)),
            Precursor("p2", "event-2", 201.0073, (1,)),
        ]
        result = engine.assign(precursors)
        self.assertEqual(len(result.unique), 2)
        self.assertEqual(result.ambiguous_groups, [])

    def test_unassigned_precursor_reported(self):
        engine = AssignmentEngine(
            [CandidateMolecule("M1", 100.0)], default_tolerance=5.0
        )
        precursor = Precursor("lost", "event-1", 500.0, (1, 2))
        result = engine.assign([precursor])
        self.assertEqual(result.unique, [])
        self.assertEqual(result.ambiguous_groups, [])
        self.assertEqual(len(result.unassigned), 1)
        self.assertEqual(result.unassigned[0].precursor.precursor_id, "lost")
        self.assertIn("no feasible", result.unassigned[0].reason)

    def test_known_single_charge_still_works(self):
        engine = AssignmentEngine(
            [CandidateMolecule("M1", 100.0)], default_tolerance=5.0
        )
        result = engine.assign(
            [Precursor("p1", "event-1", 101.0073, (1,))]
        )
        self.assertEqual(len(result.unique), 1)


class SerializationTest(unittest.TestCase):
    def test_result_as_dict_roundtrip_shape(self):
        m1 = CandidateMolecule("M1", 100.0)
        m2 = CandidateMolecule("M2", 100.0005)
        result = assign(
            [Precursor("p1", "event-1", 101.0075, (1,))],
            [m1, m2],
            default_tolerance=10.0,
        )
        payload = result.as_dict()
        self.assertEqual(
            set(payload.keys()),
            {"unique", "ambiguous_groups", "unassigned", "interpretations"},
        )
        self.assertEqual(payload["unique"], [])
        group = payload["ambiguous_groups"][0]
        self.assertEqual(
            set(group.keys()),
            {
                "group_id",
                "molecule_ids",
                "precursor_ids",
                "isolation_event_ids",
                "reasons",
                "interpretations",
            },
        )
        interpretation = group["interpretations"][0]
        self.assertEqual(
            set(interpretation.keys()),
            {
                "precursor_id",
                "isolation_event_id",
                "molecule_id",
                "charge",
                "adduct",
                "convention",
                "observed_mz",
                "theoretical_mz",
                "mass_error_da",
                "mass_error_ppm",
                "tolerance",
                "tolerance_mode",
            },
        )

    def test_unassigned_as_dict(self):
        result = assign(
            [Precursor("p1", "event-1", 500.0, (1,))],
            [CandidateMolecule("M1", 100.0)],
        )
        payload = result.as_dict()
        self.assertEqual(payload["unassigned"][0]["precursor_id"], "p1")
        self.assertEqual(payload["unassigned"][0]["charges"], [1])


if __name__ == "__main__":
    unittest.main()
