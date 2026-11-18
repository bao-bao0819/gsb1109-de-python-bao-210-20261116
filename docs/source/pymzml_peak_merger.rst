.. pymzml_peak_merger:

.. default-domain:: py

Peak merger
===========

The peak merger groups neighboring peaks based on the combined ppm and
absolute m/z tolerance. A group is only accepted when all of its members
mutually satisfy the tolerance, otherwise an
:class:`~pymzml.peak_merger.AmbiguousPeakGroupError` is raised. The grouping is
deterministic and independent of the input order, and every merged peak keeps
the contributing original peaks so intensities remain traceable.

.. automodule:: pymzml.peak_merger
    :members:
