#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Read collision energy acquisition conditions from an mzML file.

usage:

    ./collision_energy.py path/to/file.mzML

Vendor files such as the bundled BSA1 example only record ``userParam``
energies without a unit.  Pass ``assume_unit="electronvolt"`` explicitly when
that convention is known to apply; this script does so for demonstration.
"""

import sys

import pymzml
from pymzml.energy import EV


def main(path):
    """Print the collision energy condition and bucket of each MS2 spectrum."""
    run = pymzml.run.Reader(path)
    for spectrum in run:
        if spectrum.ms_level != 2 or not spectrum.has_energy_conditions:
            continue
        for condition in spectrum.energy_conditions:
            print(
                "spectrum {0}: {1} (sources: {2})".format(
                    spectrum.ID,
                    condition.to_string(),
                    ", ".join(condition.source_kinds),
                )
            )
        for bucket in spectrum.energy_buckets(width=10, assume_unit=EV):
            print(
                "  bucket {0:g}-{1:g} {2}: definite={3}, possible={4}".format(
                    bucket.start,
                    bucket.end,
                    bucket.unit,
                    bucket.definite,
                    bucket.possible,
                )
            )


if __name__ == "__main__":
    main(sys.argv[1])
