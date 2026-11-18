.. _api_allele:

Allele Mapping
==============

Multi-allelic VCF rows are split and merged often, and the ``ID`` column is
not a reliable identity while doing so: splitting one row produces several
rows with the same ID and merging combines the IDs of all inputs.  The
classes below therefore map *alleles* and carry the annotation version they
were built against.

.. autoclass:: vcfpy.AlleleKey
    :members:

.. autoclass:: vcfpy.SourceAllele
    :members:

.. autoclass:: vcfpy.UnmatchedValue
    :members:

.. autoclass:: vcfpy.AlleleMapping
    :members:

.. autofunction:: vcfpy.split_multi

.. autofunction:: vcfpy.merge_multi

.. autofunction:: vcfpy.infer_mapping

Record Integration
------------------

:py:meth:`vcfpy.Record.reorder_alts` re-orders ``ALT`` together with every
allele-indexed INFO/FORMAT value and every ``GT``.  It never re-attaches a
value by its old array index; values that cannot be mapped are recorded in
``record.allele_mapping.unmatched``.
