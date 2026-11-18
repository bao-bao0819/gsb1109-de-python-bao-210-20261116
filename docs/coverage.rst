.. _coverage:

Scan coverage and missed-scan review
=====================================

:mod:`laspy.coverage` builds missed-scan review lists for multi-strip
surveys.  It works at the *pulse* level: several returns of one emitted pulse
are one observation and can never be counted as several coverage
observations.

Two evidence levels are always reported separately:

raw pulse support
    Every return is used to decide whether its pulse observed a cell.  A
    pulse contributes at most one support count per cell, even if it has
    three returns there.

retained points
    Only the returns that pass the point filter and the multi-return
    selection are counted.  ``retained_pulse_count`` is de-duplicated by
    pulse again, so a retained three-return pulse is still one pulse.

Each grid cell is classified as one of:

``covered``
    retained pulse support and retained points meet the thresholds.

``gap``
    no raw pulse return reaches the cell.

``low_density``
    raw evidence exists but retained support is below a threshold.  These
    cells are listed as evidence-backed candidates, never silently promoted
    to covered.

``unconfirmed``
    the point records exist but the evidence cannot confirm or exclude
    coverage, for example because the filter removed all raw points or the
    pulse identity is missing/invalid.

``outside``
    the cell is outside the known scan coverage.

Building a review list
----------------------

.. code-block:: python

    import laspy
    from laspy.coverage import GridSpec, analyze_coverage

    grid = GridSpec.from_bounds((x_min, y_min, x_max, y_max), cell_size=1.0)
    result = analyze_coverage(
        ["strip_01.las", "strip_02.las"],
        grid=grid,
        filter_condition="classification in [2, 6] and withheld == 0",
        multi_return="last",
        min_pulses=2,
        min_points=4,
        min_density=2.0,  # retained points per m2, on top of min_points
        known_coverage="survey_area.geojson",
    )

    print(result.summary())
    for cell in result.review_list():
        print(cell.cell_id, cell.status, cell.raw_pulse_count,
              cell.retained_point_count, cell.reason)

    result.to_csv("review.csv")

When ``grid`` is omitted, pass ``cell_size`` and the grid is derived from
the point extent.  A directory may be given instead of a file list; its
``.las``/``.laz`` files are read in name order.  Points outside an explicitly
given grid are counted in ``summary()["points"]["outside_grid"]`` instead of
being folded into the border cells.

Pulse identity
--------------

Returns of one pulse must share an identity.  The default ``pulse_key="auto"``
tries, in order:

* ``gps_time`` (most LAS/LAZ files store the pulse emission time here),
  optionally combined with ``point_source_id``;
* an explicit ``pulse_id`` dimension or extra dimension;
* contiguous runs of ``return_number``/``number_of_returns``.

The last fallback is better than treating every return as a pulse, but it is
reported as ``identity_reliable: false`` in the summary.  Providing
``pulse_key="pulse_id"`` (or any field or list of fields) is preferable when
the data has a stable pulse identifier.  ``gps_time_tolerance`` controls how
closely two GPS times must match to be the same pulse.

Multi-return policy
-------------------

``multi_return`` selects the retained returns:

* ``"all"`` (default) keeps every return;
* ``"first"`` keeps the first return of each pulse;
* ``"last"`` keeps the last return;
* ``"single"`` keeps only pulses with exactly one return;
* ``"first_and_last"`` keeps both ends.

Raw pulse support is independent of this setting.

Comparing after adding a strip
------------------------------

.. code-block:: python

    from laspy.coverage import compare_coverage

    comparison = compare_coverage(
        baseline_sources,
        added_strip_sources,
        cell_size=1.0,
        filter_condition="withheld == 0",
        min_pulses=2,
    )
    print(comparison.summary()["delta"])

``newly_covered_cells`` are cells that were not covered by the baseline but
are covered after adding the strip.  ``repeat_observation_cells`` are cells
the baseline already covered and where the added strip also has *retained*
points; points removed by the filter are not counted as repeat observations.
They have separate point and pulse totals, so added observations in already
covered areas are not mistaken for new coverage.  ``remaining_candidate_cells``
lists the cells still needing review.  If baseline and added data are given
as raw sources, the combined support is recomputed exactly; if only
:class:`CoverageResult` objects are available, their stored source data is
used.  Passing ``combined=`` explicitly is also supported.

Command line
------------

With the optional CLI installed (``pip install laspy[cli]``):

.. code-block:: console

    laspy coverage review strips/*.las \
        --cell-size 1.0 \
        --filter "classification in [2, 6]" \
        --multi-return last \
        --min-pulses 2 \
        --output review.csv

    laspy coverage compare baseline/*.las \
        --added new_strip.las \
        --cell-size 1.0 \
        --output delta.json

``--grid-bounds "x_min y_min x_max y_max"`` fixes the grid,
``--min-density`` adds an area-normalised retained-point threshold, and
``--coverage-bounds`` or ``--coverage-mask`` restricts the assessed cells to
the known scan coverage.
