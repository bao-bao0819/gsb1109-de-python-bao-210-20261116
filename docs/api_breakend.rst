.. _api_breakend:

=========
Breakends
=========

.. contents::

The breakend module pairs BND records into a structural variant breakpoint
table.  Two breakend records are only paired when their reciprocal
signatures match in chromosome, position, orientation, and bracket style.
Breakends without a unique reciprocal partner are reported separately as
missing or ambiguous mates rather than being forced into a pair.

vcfpy.pair_breakends
--------------------

.. autofunction:: vcfpy.pair_breakends

vcfpy.iter_breakend_endpoints
-----------------------------

.. autofunction:: vcfpy.iter_breakend_endpoints

Result classes
--------------

.. autoclass:: vcfpy.breakend.BreakendEndpoint
    :members:
    :undoc-members:

.. autoclass:: vcfpy.breakend.BreakendPair
    :members:
    :undoc-members:

.. autoclass:: vcfpy.breakend.BreakendPairing
    :members:
    :undoc-members:

Status codes
------------

.. autodata:: vcfpy.breakend.PAIRED

.. autodata:: vcfpy.breakend.MISSING_MATE

.. autodata:: vcfpy.breakend.AMBIGUOUS_MATE
