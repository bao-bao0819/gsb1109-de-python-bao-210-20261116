.. _api_haplotype:

==========
Haplotypes
==========

.. contents::

Provable haplotype fragment reconstruction joins alleles only when they are
demonstrably carried by the same chromosome copy: within a phase set
(``PS``) or through an explicit cross-block anchor.  Every emitted base keeps
a provenance entry back to its reference range and VCF allele.

Entry Point
-----------

.. autofunction:: vcfpy.reconstruct_haplotypes

Result Types
------------

.. autoclass:: vcfpy.SampleHaplotypes
    :members:

.. autoclass:: vcfpy.HaplotypeFragment
    :members:

.. autoclass:: vcfpy.Provenance
    :members:

.. autoclass:: vcfpy.UnknownRegion
    :members:

.. autoclass:: vcfpy.PhaseBlock
    :members:

.. autoclass:: vcfpy.HaplotypeConflict
    :members:

.. autoclass:: vcfpy.CrossBlockAnchor
    :members:

Provenance Operations
---------------------

.. autodata:: vcfpy.OP_MATCH
.. autodata:: vcfpy.OP_SUBST
.. autodata:: vcfpy.OP_INS
.. autodata:: vcfpy.OP_DEL

Exceptions
----------

.. autoclass:: vcfpy.HaplotypeConflictError
    :members:
