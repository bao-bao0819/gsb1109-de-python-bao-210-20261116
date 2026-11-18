#!/usr/bin/env python

"""
Demonstration script for the isotopic envelope assignment utilities in
:mod:`pymzml.isotopic_envelopes`.

usage:

    ./isotopic_envelopes.py

The script builds a small synthetic centroided peak list that contains:

* a triply charged isotope envelope,
* a doubly charged envelope whose M+1 isotope is missing,
* one peak that lies inside the spacing range of both groups and therefore
  makes the two explanations mutually exclusive.

It enumerates every valid peak group and its contributing observed peaks, then
scores the competing whole-set explanations. Equal-scoring charges are kept as
separate explanations rather than resolved by input order.
"""

from pymzml.isotopic_envelopes import NEUTRON_MASS, IsotopicEnvelopeFinder


def main():
    spacing_z2 = NEUTRON_MASS / 2

    peaks = [
        (500.0000, 1000.0),
        (500.0000 + 2 * spacing_z2, 300.0),
        (700.0000, 800.0),
        (700.0000 + NEUTRON_MASS / 3, 500.0),
        (700.0000 + 2 * NEUTRON_MASS / 3, 200.0),
    ]

    finder = IsotopicEnvelopeFinder(
        charges=[2, 3],
        mz_tolerance=0.02,
        max_missing_peaks=1,
        max_consecutive_missing=1,
        min_isotopes=2,
    )

    envelopes, explanations = finder.assign(peaks)

    print("Valid isotopic peak groups:")
    for envelope in envelopes:
        contributions = ", ".join(
            "peak {0} (M+{1}, m/z {2:.4f}, intensity {3:.1f}, "
            "{4:+.2f} ppm)".format(
                member.peak_index,
                member.isotope_index,
                member.mz,
                member.intensity,
                member.mass_error_ppm,
            )
            for member in envelope.members
        )
        print(
            "  z={0} monoisotopic_m/z={1:.4f} missing={2} score={3:.3f}"
            .format(
                envelope.charge,
                envelope.monoisotopic_mz,
                envelope.missing_indices,
                envelope.score,
            )
        )
        print("      contributions: {0}".format(contributions))

    print("\nBest whole-set explanations ({0}):".format(len(explanations)))
    for explanation in explanations:
        print(
            "  {0}: score={1:.3f}, covered peaks={2}, unassigned peaks={3}"
            .format(
                explanation,
                explanation.score,
                sorted(explanation.covered_peak_indices),
                explanation.n_unassigned_peaks,
            )
        )
        for envelope in explanation.envelopes:
            print(
                "      z={0} at m/z {1:.4f} claims peaks {2}".format(
                    envelope.charge,
                    envelope.monoisotopic_mz,
                    envelope.peak_indices,
                )
            )


if __name__ == "__main__":
    main()
