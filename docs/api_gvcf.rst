.. _api_gvcf:

================
gVCF Intervals
================

.. contents::

The :py:mod:`vcfpy.gvcf` module builds per-sample determinable intervals
from gVCF records, combining reference confidence blocks (``END``),
concrete variant records, and the sample genotype calls.  Each position is
classified as ``reference``, ``variant``, ``no_call`` (insufficient
evidence), or ``uncovered``.  The placeholder allele ``<NON_REF>`` is never
treated as a concrete biological allele.

State Constants
---------------

.. autodata:: vcfpy.gvcf.STATE_REFERENCE
    :noindex:

.. autodata:: vcfpy.gvcf.STATE_VARIANT
    :noindex:

.. autodata:: vcfpy.gvcf.STATE_NO_CALL
    :noindex:

.. autodata:: vcfpy.gvcf.STATE_UNCOVERED
    :noindex:

Data Types
----------

.. autoclass:: vcfpy.AlleleContribution
    :members:

.. autoclass:: vcfpy.CallClassification
    :members:

.. autoclass:: vcfpy.SampleInterval
    :members:

Functions
---------

.. autofunction:: vcfpy.is_non_ref_allele

.. autofunction:: vcfpy.is_reference_block

.. autofunction:: vcfpy.record_span

.. autofunction:: vcfpy.classify_call

.. autofunction:: vcfpy.sample_intervals

.. autofunction:: vcfpy.all_sample_intervals
