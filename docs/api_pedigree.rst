.. _api_pedigree:

==================
Pedigree Screening
==================

.. contents::

The :py:mod:`vcfpy.pedigree` module checks genotype calls against a pedigree
and an inheritance mode.  It deliberately distinguishes three outcomes:

``CONSISTENT``
    Every completion of the missing alleles is compatible.  For complete calls
    this simply means the observed genotypes are compatible.

``CONFLICT``
    No completion of the missing alleles is compatible.

``INSUFFICIENT``
    Some completions are compatible and some are not, or the available calls
    do not contain enough information.  This includes a parent that is absent
    from the pedigree and a fully missing child call.

A partially missing call such as ``0/.`` is therefore never treated as
a conflict merely because an allele is missing, and a single compatible
completion is never reported as proof.  Every result records the conditions
under which it holds, including the allele universe used for completion, in
:py:attr:`vcfpy.ScreeningResult.assumptions`.

vcfpy.Pedigree
--------------

.. autoclass:: vcfpy.Pedigree
    :members:

vcfpy.PedigreeMember
--------------------

.. autoclass:: vcfpy.PedigreeMember
    :members:

vcfpy.ScreeningResult
---------------------

.. autoclass:: vcfpy.ScreeningResult
    :members:

vcfpy.screen_calls
------------------

.. autofunction:: vcfpy.screen_calls

vcfpy.screen_record
-------------------

.. autofunction:: vcfpy.screen_record

vcfpy.screen_records
--------------------

.. autofunction:: vcfpy.screen_records

vcfpy.screen_reader
-------------------

.. autofunction:: vcfpy.screen_reader

Constants
---------

.. autodata:: vcfpy.pedigree.CONSISTENT
    :noindex:

.. autodata:: vcfpy.pedigree.CONFLICT
    :noindex:

.. autodata:: vcfpy.pedigree.INSUFFICIENT
    :noindex:

.. autodata:: vcfpy.pedigree.INHERITANCE_MODES
    :noindex:
