.. _api_record:

=======
Records
=======

.. contents::

Record-Related Constants
------------------------

The following constants are also available as ``vcfpy.CONSTANT``.

.. autodata:: vcfpy.record.HOM_REF
    :noindex:

.. autodata:: vcfpy.record.HOM_ALT
    :noindex:

.. autodata:: vcfpy.record.FIVE_PRIME
    :noindex:

.. autodata:: vcfpy.record.THREE_PRIME
    :noindex:

.. autodata:: vcfpy.record.FORWARD
    :noindex:

.. autodata:: vcfpy.record.REVERSE
    :noindex:


vcfpy.Record
------------

.. autoclass:: vcfpy.Record
    :members:

vcfpy.Call
----------

.. autoclass:: vcfpy.Call
    :members:

Re-ordering alleles
-------------------

.. automethod:: vcfpy.Record.reorder_alts
    :noindex:

.. automethod:: vcfpy.Record.split
    :noindex:

.. automethod:: vcfpy.Record.merge
    :noindex:

vcfpy.AltRecord
---------------

.. autoclass:: vcfpy.AltRecord
    :members:

vcfpy.Substitution
------------------

.. autoclass:: vcfpy.Substitution
    :members:

vcfpy.SV
--------

.. autoclass:: vcfpy.SV
    :members:

vcfpy.BreakEnd
--------------

.. autoclass:: vcfpy.BreakEnd
    :members:

vcfpy.SingleBreakEnd
--------------------

.. autoclass:: vcfpy.SingleBreakEnd
    :members:

vcfpy.SymbolicAllele
--------------------

.. autoclass:: vcfpy.SymbolicAllele
    :members:
