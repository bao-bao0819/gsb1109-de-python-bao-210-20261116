.. _examples:

========
Examples
========

This chapter contains several examples for the most important use cases of VCFPy.

Reading VCF Files
=================

The following is an example for reading VCF files and writing out a TSV file with the genotype calls of all SNVs.
You can find the example Python and VCF file in the sources below the directory ``examples/vcf_to_tsv``.

.. literalinclude:: ../examples/vcf_to_tsv/vcf_to_tsv.py
    :language: python

The program call looks as follows.

.. literalinclude::  ../examples/vcf_to_tsv/stdout.txt
    :language: shell

Writing VCF Files
=================

The following shows how to add values to the ``FILTER`` column to records of an existing VCF file.
Adding to existing records is simpler than constructing them from scratch, of course.

.. literalinclude:: ../examples/add_filter/add_filter.py
    :language: python

The program call looks as follows.

.. literalinclude::  ../examples/add_filter/stdout.txt
    :language: shell

Jumping in Tabix-indexed Files
==============================

The following shows a small program that extracts a genomic region from the input VCF file and writes it to stdout.

.. literalinclude:: ../examples/fetch_tabix/fetch_tabix.py
    :language: python

The program call looks as follows.

.. literalinclude::  ../examples/fetch_tabix/stdout.txt
    :language: shell

Fetching Multiple Overlapping Intervals
=======================================

``Reader.fetch_many`` accepts several intervals at once and merges hits by
source record occurrence.  A record reached through several overlapping
index chunks or query intervals is returned once, with all matching intervals
kept on the hit.  Two identical lines in the file remain two occurrences
because the identity is the line's byte offset, not its content.  Results are
ordered by source position, so swapping the query intervals does not change
which occurrence is retained.

.. code-block:: python

    import vcfpy

    with vcfpy.Reader.from_path("input.vcf.gz") as reader:
        for hit in reader.fetch_many(["20:1,110,000-1,111,000", "20:1,230,000-1,235,000"]):
            print(hit.POS, hit.intervals)
