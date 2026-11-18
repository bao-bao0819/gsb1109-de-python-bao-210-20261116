.. _api_normalize:

==========================
Variant Normalization
==========================

.. contents::

The :py:mod:`vcfpy.normalize` module left-aligns sequence alleles and turns
them into stable business keys.

An indel inside a repeat has several equally valid VCF representations, for
example an inserted ``CAG`` unit can be written at any ``CAG`` boundary.  All
of these representations are canonicalized to the same :py:class:`VariantKey`.
The key includes the reference version, so identical truncated REF/ALT strings
from different reference builds never collapse into one key.

When the supplied :py:class:`ReferenceWindow` does not reach far enough to the
left, the result has no key.  Instead it carries an
:py:class:`UnresolvedRange` describing the positions that remain possible.
For batch operations the mapping from every original allele to its key (or
unresolved range) is retained.

Basic Usage
===========

.. code-block:: python

    import vcfpy

    window = vcfpy.ReferenceWindow(
        "GCAGCAGCAGCAGCAGT",
        start=0,
        reference="GRCh38",
        contig="chr1",
    )
    for pos, ref, alt in [(1, "G", "GCAG"), (2, "C", "CAGC"), (4, "G", "GCAG")]:
        result = vcfpy.normalize_variant(window, pos, ref, alt)
        print(result.key)  # GRCh38:chr1:1:G:GCAG for all three

    batch = vcfpy.normalize_alleles(window, 2, "C", ["CAGC", "T"])
    print(batch.allele_map)  # original ALT -> key

vcfpy.ReferenceWindow
---------------------

.. autoclass:: vcfpy.normalize.ReferenceWindow
    :members:

vcfpy.VariantKey
----------------

.. autoclass:: vcfpy.normalize.VariantKey
    :members:

vcfpy.UnresolvedRange
---------------------

.. autoclass:: vcfpy.normalize.UnresolvedRange
    :members:

vcfpy.NormalizationResult
-------------------------

.. autoclass:: vcfpy.normalize.NormalizationResult
    :members:

vcfpy.NormalizationBatch
------------------------

.. autoclass:: vcfpy.normalize.NormalizationBatch
    :members:

Functions
---------

.. autofunction:: vcfpy.normalize.normalize_variant

.. autofunction:: vcfpy.normalize.normalize_alleles

.. autofunction:: vcfpy.normalize.normalize_variants

.. autofunction:: vcfpy.normalize.normalize_record

.. autofunction:: vcfpy.normalize.left_align_allele

.. autofunction:: vcfpy.normalize.trim_alleles

.. autofunction:: vcfpy.normalize.make_business_key

Exceptions
----------

.. autoclass:: vcfpy.normalize.ReferenceWindowError

.. autoclass:: vcfpy.normalize.ReferenceMismatchError
