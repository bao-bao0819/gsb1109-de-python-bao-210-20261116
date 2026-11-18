#!/usr/bin/env python

"""
Polarity- and MS-level-aware extraction of an ion chromatogram (XIC/EIC).

If the instrument switched polarity and/or MS level in the middle of the
run, only scans compatible with the query (target polarity AND target MS
level) contribute signal. Peaks at the same m/z from the opposite polarity
are never mixed in. Every scan time point is annotated with either the
contributing peak(s) or the reason why no signal was recorded. When the
interval between consecutive compatible scans suddenly widens, a gap row
(intensity = None) is inserted so downstream plotting keeps the gap
instead of bridging it with a fake line.

usage:

    ./extract_ion_chromatogram_polarity_aware.py \
        [--file MZML_FILE] [--mz TARGET_MZ] [--polarity positive|negative] \
        [--ms-level N] [--gap-factor FACTOR]

"""

import argparse
import os
import statistics

import pymzml

POLARITY_ACCESSION = {
    "positive": "MS:1000130",  # positive scan
    "negative": "MS:1000129",  # negative scan
}


def get_polarity(spectrum):
    """
    Return 'positive', 'negative' or None for a spectrum.
    """
    for polarity, accession in POLARITY_ACCESSION.items():
        if spectrum[accession] is not None:
            return polarity
    return None


def extract_xic(mzml_file, target_mz, target_polarity, target_ms_level, gap_factor):
    """
    Extract the ion chromatogram for target_mz restricted to scans matching
    target_polarity and target_ms_level.

    Returns a list of row dicts, one per scan (plus gap sentinel rows):
        time      scan time in minutes
        scan_id   spectrum ID (None for gap rows)
        polarity  observed polarity of the scan
        ms_level  observed MS level of the scan
        status    'peak' | 'missing' | 'gap'
        mz        m/z of the contributing peak (None if missing/gap)
        intensity intensity of the contributing peak (None if missing/gap)
        note      reason for missing data or gap description
    """
    run = pymzml.run.Reader(mzml_file)
    rows = []
    compatible_times = []

    for spectrum in run:
        polarity = get_polarity(spectrum)
        ms_level = spectrum.ms_level
        time = spectrum.scan_time_in_minutes()

        base = {
            "time": time,
            "scan_id": spectrum.ID,
            "polarity": polarity,
            "ms_level": ms_level,
            "status": None,
            "mz": None,
            "intensity": None,
            "note": "",
        }

        if ms_level != target_ms_level:
            rows.append(
                dict(
                    base,
                    status="missing",
                    note="ms_level mismatch: observed MS{0}, query requires MS{1}".format(
                        ms_level, target_ms_level
                    ),
                )
            )
            continue

        if polarity != target_polarity:
            observed = polarity if polarity is not None else "unknown"
            rows.append(
                dict(
                    base,
                    status="missing",
                    note="polarity mismatch: observed '{0}', query requires '{1}' "
                    "(same-m/z peaks from this scan are excluded)".format(
                        observed, target_polarity
                    ),
                )
            )
            continue

        # compatible scan: only these may contribute signal
        compatible_times.append(time)
        matches = spectrum.has_peak(target_mz)
        if matches:
            for peak_mz, intensity in matches:
                rows.append(
                    dict(
                        base,
                        status="peak",
                        mz=peak_mz,
                        intensity=intensity,
                        note="",
                    )
                )
        else:
            rows.append(
                dict(base, status="missing", note="peak not found in compatible scan")
            )

    rows = insert_gap_rows(rows, compatible_times, gap_factor)
    return rows


def insert_gap_rows(rows, compatible_times, gap_factor):
    """
    Insert gap sentinel rows (intensity None) where the interval between
    consecutive compatible scans exceeds gap_factor * median interval.
    """
    if len(compatible_times) < 3:
        return rows

    intervals = [
        later - earlier
        for earlier, later in zip(compatible_times, compatible_times[1:])
    ]
    median_interval = statistics.median(intervals)
    if median_interval <= 0:
        return rows

    gap_before = {}
    for earlier, later in zip(compatible_times, compatible_times[1:]):
        interval = later - earlier
        if interval > gap_factor * median_interval:
            gap_before[round(later, 9)] = (earlier, interval, median_interval)

    if not gap_before:
        return rows

    rows_with_gaps = []
    inserted = set()
    for row in rows:
        key = round(row["time"], 9)
        if key in gap_before and key not in inserted:
            earlier, interval, median_interval = gap_before[key]
            inserted.add(key)
            rows_with_gaps.append(
                {
                    "time": (earlier + row["time"]) / 2.0,
                    "scan_id": None,
                    "polarity": None,
                    "ms_level": None,
                    "status": "gap",
                    "mz": None,
                    "intensity": None,
                    "note": "acquisition gap: {0:.3f} min between compatible "
                    "scans (median {1:.3f} min); gap kept, not bridged".format(
                        interval, median_interval
                    ),
                }
            )
        rows_with_gaps.append(row)
    return rows_with_gaps


def main():
    parser = argparse.ArgumentParser(
        description="Polarity/MS-level-aware ion chromatogram extraction."
    )
    parser.add_argument(
        "--file",
        default=os.path.join(
            os.path.dirname(__file__), os.pardir, "tests", "data", "example.mzML"
        ),
        help="mzML file to read",
    )
    parser.add_argument(
        "--mz", type=float, default=70.06575775, help="target m/z to follow"
    )
    parser.add_argument(
        "--polarity",
        choices=["positive", "negative"],
        default="positive",
        help="polarity of scans allowed to contribute signal",
    )
    parser.add_argument(
        "--ms-level",
        type=int,
        default=1,
        help="MS level of scans allowed to contribute signal",
    )
    parser.add_argument(
        "--gap-factor",
        type=float,
        default=3.0,
        help="intervals larger than this multiple of the median scan "
        "interval are kept as gaps",
    )
    args = parser.parse_args()

    rows = extract_xic(
        args.file, args.mz, args.polarity, args.ms_level, args.gap_factor
    )

    print(
        "Query: m/z {0} | polarity {1} | MS{2}".format(
            args.mz, args.polarity, args.ms_level
        )
    )
    print(
        "{0:>8} {1:>10} {2:>9} {3:>3} {4:>8} {5:>12} {6:>14}  {7}".format(
            "scan_id", "time(min)", "polarity", "MS", "status", "mz", "intensity", "note"
        )
    )
    for row in rows:
        print(
            "{0:>8} {1:>10.4f} {2:>9} {3:>3} {4:>8} {5:>12} {6:>14}  {7}".format(
                row["scan_id"] if row["scan_id"] is not None else "-",
                row["time"],
                row["polarity"] if row["polarity"] is not None else "-",
                row["ms_level"] if row["ms_level"] is not None else "-",
                row["status"],
                "{0:.5f}".format(row["mz"]) if row["mz"] is not None else "-",
                "{0:.4f}".format(row["intensity"])
                if row["intensity"] is not None
                else "None",
                row["note"],
            )
        )

    n_peak = sum(1 for row in rows if row["status"] == "peak")
    n_missing = sum(1 for row in rows if row["status"] == "missing")
    n_gap = sum(1 for row in rows if row["status"] == "gap")
    print(
        "\nSummary: {0} contributing peak(s), {1} excluded/missing scan(s), "
        "{2} gap(s) kept".format(n_peak, n_missing, n_gap)
    )


if __name__ == "__main__":
    main()
