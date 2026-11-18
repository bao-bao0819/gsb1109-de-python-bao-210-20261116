.. _api_utils:

=========
Utilities
=========

.. contents::

Recomputing AC/AN After Subsetting
----------------------------------

When writing out only a subset of the samples, the ``AC`` and ``AN``
entries in the ``INFO`` column still describe the full cohort.
The function :py:func:`vcfpy.utils.recompute_ac_an` recomputes them from
the genotypes of the selected samples.

.. code-block:: python

    import vcfpy

    with vcfpy.Reader.from_path("input.vcf") as reader:
        samples = ["NA00001", "NA00002"]
        header = vcfpy.Header(reader.header.lines, vcfpy.SamplesInfos(samples))
        with vcfpy.Writer.from_path("output.vcf", header) as writer:
            for record in reader:
                vcfpy.recompute_ac_an(record, reader.header, samples)
                writer.write_record(record)

vcfpy.utils
-----------

.. automodule:: vcfpy.utils
    :members:
