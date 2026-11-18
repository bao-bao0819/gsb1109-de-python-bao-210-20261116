#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demonstration of blank subtraction between spectra that do not share an
m/z grid.

The sample below is sampled every 10 m/z (with a gap), the blank every 25 m/z
(with its own gap).  The script prints the subtracted value together with the
blank measurements on both sides and lists the m/z ranges the blank does not
cover.
"""

import numpy as np

from pymzml.blank import subtract_blank


def main():
    """
    Subtract a sparsely sampled blank from a sample spectrum.

    usage:

        ./subtract_blank_spectrum.py

    Points without blank coverage are reported as unknown; they are not treated
    as having zero background.  Negative results are kept.
    """
    sample = [(60.0, 8.0), (100.0, 120.0), (110.0, 25.0), (120.0, 40.0),
              (130.0, 12.0), (150.0, 90.0), (160.0, 35.0), (170.0, 15.0),
              (200.0, 2.0), (250.0, 7.0)]
    blank = [(100.0, 20.0), (125.0, 10.0), (175.0, 30.0), (200.0, 5.0)]

    result = subtract_blank(sample, blank, scale=0.5, max_gap=60.0)

    print("mz\tsample\tscaled blank\tvalue\tstatus\tsources")
    for point in result:
        if point.sources:
            sources = ", ".join(
                "{0:.1f}@{1:.3f}".format(source.mz, source.weight)
                for source in point.sources
            )
        else:
            sources = "-"
        print(
            "{0:6.2f}\t{1:8.2f}\t{2:8.2f}\t{3:8.2f}\t{4:10s}\t{5}".format(
                point.mz,
                point.sample_intensity,
                point.blank_intensity
                if point.blank_intensity is not None
                else float("nan"),
                point.value if point.value is not None else float("nan"),
                point.status,
                sources,
            )
        )

    print()
    for interval in result.unknown_intervals:
        print(
            "unknown {reason}: {start:.2f} - {end:.2f} "
            "({count} sample points)".format(
                reason=interval.reason,
                start=interval.mz_start,
                end=interval.mz_end,
                count=interval.count,
            )
        )

    chunked = subtract_blank(sample, blank, scale=0.5, max_gap=60.0, chunk_size=2)
    print()
    print(
        "chunked result equals full pass:",
        bool(np.allclose(chunked.all_peaks, result.all_peaks, equal_nan=True)),
    )


if __name__ == "__main__":
    main()
