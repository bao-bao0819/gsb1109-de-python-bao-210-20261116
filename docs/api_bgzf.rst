.. _api_bgzf:

====
BGZF
====

.. contents::

The BGZF module provides support for reading and writing BGZF (Blocked GNU Zip Format) files.

vcfpy.bgzf.BgzfReader
---------------------

.. autoclass:: vcfpy.bgzf.BgzfReader
    :members:

vcfpy.bgzf.BgzfWriter
---------------------

.. autoclass:: vcfpy.bgzf.BgzfWriter
    :members:

BGZF Utilities
--------------

.. autofunction:: vcfpy.bgzf.make_virtual_offset

.. autofunction:: vcfpy.bgzf.split_virtual_offset

Resuming at Record Boundaries
----------------------------

VCF readers record their position as a BGZF *virtual offset*: the raw file
offset of the compressed block in the upper 48 bits and the offset of the
byte inside that block's decompressed payload in the lower 16 bits.
A record can span two compressed blocks, so a virtual offset is not a
linear byte position and byte counts must never be added to it.

:py:meth:`vcfpy.Reader.tell` returns such an offset for the next record
boundary (or the plain stream's own ``tell()`` cookie for uncompressed
input), and :py:meth:`vcfpy.Reader.seek` resumes from an offset obtained
from a reader that parsed the same header. Resume points must be taken
between complete records; never re-read the previous line as if it were a
new record. The result does not depend on how input was divided into BGZF
blocks.

Short reads are retried by the block loader. If the file ends in the middle
of a BGZF block, :py:class:`vcfpy.BgzfTruncatedError` is raised; a clean
end of file at a block boundary ends iteration normally.
