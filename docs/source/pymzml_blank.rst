.. pymzml_blank:

Blank subtraction
=================

.. automodule:: pymzml.blank

A blank spectrum is usually acquired on its own m/z grid, with its own gaps.
:func:`pymzml.blank.subtract_blank` aligns the blank onto every sample point
before subtracting:

* a blank m/z within ``tolerance`` is used directly;
* otherwise the blank intensity is linearly interpolated between the two
  bracketing blank points;
* outside the blank range, or across a gap wider than ``max_gap``, the value is
  reported as unknown.  It is never replaced by a blank of zero; an unknown
  region is returned as :class:`pymzml.blank.UnknownInterval`.

The scaled blank contribution is ``scale(mz) * blank_intensity`` and the result
is ``sample_intensity - blank_intensity``.  Negative results are valid
subtraction values and are kept unless ``clip=True`` is passed explicitly.

Each :class:`pymzml.blank.BlankSubtractionPoint` keeps the blank measurements
on both sides (``left`` and ``right``) together with the sources actually used
(``sources``, with interpolation weights), so every value can be traced back to
its measurements.

Chunked processing
------------------

``chunk_size`` (or :meth:`pymzml.blank.BlankSubtractor.iter_chunks`) splits the
sample into blocks while the blank grid stays global.  A sample point at a chunk
boundary is still interpolated from its real left and right neighbours, so a
chunked run is identical to a full pass: same values, same statuses, same
unknown intervals.  Results of :meth:`iter_chunks` can be merged with
:meth:`pymzml.blank.BlankSubtractionResult.combine`.

Example
-------

.. code-block:: python

    from pymzml.blank import subtract_blank

    sample = [(100.0, 10.0), (150.0, 20.0), (200.0, 4.0)]
    blank = [(100.0, 2.0), (200.0, 5.0)]

    result = subtract_blank(sample, blank, scale=0.5, max_gap=1.0)
    for point in result:
        print(point.mz, point.value, point.status)

    # Values that could be computed, as (mz, value) pairs.
    result.peaks
    # Uncovered m/z ranges the blank cannot speak for.
    result.unknown_intervals

A :class:`pymzml.spec.Spectrum` can also be used directly:

.. code-block:: python

    result = sample_spectrum.subtract_blank(blank_spectrum, scale=0.5)

API
---

.. autoclass:: pymzml.blank.BlankSubtractor
    :members:

.. autofunction:: pymzml.blank.subtract_blank

.. autofunction:: pymzml.blank.subtract_blank_arrays

.. autoclass:: pymzml.blank.BlankSubtractionResult
    :members:

.. autoclass:: pymzml.blank.BlankSubtractionPoint
    :members:

.. autoclass:: pymzml.blank.BlankSource
    :members:

.. autoclass:: pymzml.blank.UnknownInterval
    :members:

Command line
------------

The module can be used directly for two-column peak files (comma, tab or
whitespace separated)::

    python -m pymzml.blank --sample sample.tsv --blank blank.tsv \
        --scale 0.5 --tolerance 0.01 --max-gap 0.05 --format tsv
