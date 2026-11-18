.. _api_phasing:

=======
Phasing
=======

.. contents::

The VCF ``FORMAT/PS`` field identifies a phase set, but the integer value is
only meaningful *within one sample and one chromosome*.  The same PS value can
be reused on both sides of a shard boundary and on another chromosome, so PS
values alone cannot be used as global block identifiers.

The :py:class:`vcfpy.PhaseGrouper` builds sample-level phase groups scoped by
``(sample, chromosome)`` from an iterable of :py:class:`vcfpy.Record` objects.
When consecutive phased calls of the same sample and chromosome *within one
sharded batch* carry different PS values, the two phase sets are joined and the
junction is recorded as an explicit :py:class:`vcfpy.PhaseBreakpoint`, so every
merged block remains traceable back to its constituent phase sets.

Only fully phased calls (a genotype such as ``0|1``) that carry a ``PS`` value
participate.  Unphased calls (``0/1``), mixed-phased calls (``0/1|2``) and
phased calls without ``PS`` never force a connection, and blocks never span
more than one chromosome.  Records fed through separate
:py:meth:`~vcfpy.PhaseGrouper.add_records` calls are not joined across the batch
boundary; add a bridging shard (for example the middle shard of a chromosome)
to make two phase sets converge into one block.

.. code-block:: python

    import vcfpy

    with vcfpy.Reader.from_path("calls.vcf") as reader:
        # a contiguous VCF is fed as one shard so within-chromosome PS
        # transitions converge; sharded callers use one add_records per shard
        grouper = vcfpy.PhaseGrouper().add_records(reader)

    for block in grouper.groups():
        print(block.id, block.phase_sets, block.start, block.end)
        for breakpoint in block.breakpoints:
            print("  joined PS", breakpoint.left_ps, "and PS", breakpoint.right_ps, "at", breakpoint.position)

vcfpy.PhaseGrouper
------------------

.. autoclass:: vcfpy.PhaseGrouper
    :members:

vcfpy.PhaseBlock
----------------

.. autoclass:: vcfpy.PhaseBlock
    :members:

vcfpy.PhaseBreakpoint
---------------------

.. autoclass:: vcfpy.PhaseBreakpoint
    :members:

vcfpy.group_phase_blocks
------------------------

.. autofunction:: vcfpy.group_phase_blocks
