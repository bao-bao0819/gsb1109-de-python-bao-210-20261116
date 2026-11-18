.. _quick-start:

Quick Start
===========

Parsing a mzML file and setting measured precision
--------------------------------------------------


.. autofunction:: simple_parser.main
	:noindex:

.. include:: code_inc/simple_parser.inc


Seeking in a mzML file
----------------------

One of the features of pymzML is the ability to (create) and read
indexed gzip which allows mzML file sizes to reach the levels of the original
RAW format. The interface to random access into a mzML file is implemented
by the magic get function in pymzMLs run class.

Alternatively, pymzML can also rapidly seek into any uncompressed mzML file,
no matter if an index was included into the file or not.

.. code-block:: python

    #!/usr/bin/env python
    import pymzml

    run = pymzml.run.Reader( 'tests/data/BSA1.mzML.gz' )
    spectrum_with_id_2540 = run[ 2540 ]


Reading mzML indices with a custom regular expression
------------------------------------------------------

When reading mzML files with indices wich is not an integer or contains "scan=1" or similar,
you can set a custom regex to parse the index when initializing the reader.

Say for example you have an index as in the example file Manuels_customs_ids.mzML:
    <offset idRef="ManuelsCustomID=1 diesdas">4026</offset>

.. code-block:: python

    #!/usr/bin/env python
    import pymzml
    import re

    index_re = re.compile(
        b'.*idRef="ManuelsCustomID=(?P<ID>.*) diesdas">(?P<offset>[0-9]*)</offset>'
    )
    run = pymzml.run.Reader(your_file_path, index_regex=index_re)
    spec_1 = run[1]

The regular expression has to contain a group called ID and a group called offset.
Also be aware that your regex need to be a byte string.


Three-dimensional search of retention time, m/z and ion mobility
----------------------------------------------------------------

Peaks with a retention time, an m/z and an ion mobility array can be searched
jointly. Build the original peaks once, attach a piecewise mobility
calibration and query all three dimensions at the same time:

.. code-block:: python

    #!/usr/bin/env python
    import pymzml
    from pymzml.search3d import (
        MobilityCalibration,
        MobilityCalibrationSegment,
        PeakIndex3D,
        SearchQuery,
        extract_reader_peaks3d,
    )

    run = pymzml.run.Reader("tests/data/your_tims_file.mzML")
    peaks = extract_reader_peaks3d(run)
    # Shorthand: run.peaks3d() and run.build_3d_index(calibration)

    calibration = MobilityCalibration(
        segments=[
            MobilityCalibrationSegment(
                0.80, 1.00, slope=1.02, error_low=0.002, error_high=0.004,
                label="low mobility",
            ),
            MobilityCalibrationSegment(
                1.00, 1.20, slope=1.05, error=0.003, label="high mobility",
            ),
        ],
        name="book 1",
        revision="2026-11",
    )
    index = PeakIndex3D(peaks, calibration)
    projection = index.projection_2d()

    result = index.search(
        SearchQuery(
            retention_time=4.20,
            mz=512.34,
            mobility=1.155,
            rt_tolerance=0.02,
            mz_tolerance=0.005,
            mobility_tolerance=0.01,
        ),
        projection=projection,
    )
    for hit in result.definite_hits:
        print(hit.peak_id, hit.status, hit.calibration.to_dict())
    for hit in result.possible_hits:
        print(hit.peak_id, hit.reasons)

A hit is always formed from a **single original peak**. If one peak matches
the retention time, another the m/z and a third the mobility, no hit is
reported. ``definite`` hits have their complete observed interval inside all
three query windows; ``possible`` hits overlap every window but at least one
interval is not fully contained, for example because of a wide calibration
error range. Every hit keeps the calibration segment, coefficients, measured
mobility and propagated error interval under ``hit.calibration``.

After a calibration table is corrected, only peaks whose calibrated values
changed are recalculated and the attached two-dimensional view is refreshed in
the same call:

.. code-block:: python

    report = index.correct_calibration_segment(
        1, slope=1.0, error=0.0005, revision="2026-12"
    )
    print(report.recomputed_peak_ids)
    print(report.unaffected_peak_ids)

    # The view always shows the current mobility values. Hit overlays from an
    # older calibration revision are cleared for the recalculated peaks.
    rows = projection.rows()
    columns = projection.to_plot_data()

    # Save the calibration evidence of every original peak.
    index.save_calibration_basis("calibration_basis.json")
