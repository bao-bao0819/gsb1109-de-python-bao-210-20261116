.. _api_filterstats:

=================
Filter Statistics
=================

.. contents::

The filter statistics module aggregates filter outcomes from multiple
sources into unified, auditable statistics.  Outcomes distinguish *not
done*, *passed*, and concrete failures, separately at record and sample
level.  Callers provide conditional mappings plus the information loss
they accept; failures without a provable correspondence are reported
under explicit ``@unmapped:<source>:<label>`` categories instead of a
generic failure bucket.

Outcomes and Status
-------------------

.. autoclass:: vcfpy.filterstats.Level
    :members:

.. autoclass:: vcfpy.filterstats.FilterStatus
    :members:

.. autoclass:: vcfpy.filterstats.FilterOutcome
    :members:

Mappings
--------

.. autoclass:: vcfpy.filterstats.Loss
    :members:

.. autoclass:: vcfpy.filterstats.MappingRule
    :members:

.. autoclass:: vcfpy.filterstats.FilterMapping
    :members:

Statistics and Reconciliation
-----------------------------

.. autoclass:: vcfpy.filterstats.UnifiedFilterStats
    :members:

.. autoclass:: vcfpy.filterstats.LabelContribution
    :members:

.. autoclass:: vcfpy.filterstats.LevelStats
    :members:

.. autoclass:: vcfpy.filterstats.SourceReconciliation
    :members:

.. autoclass:: vcfpy.filterstats.Reconciliation
    :members:

Categories and Helpers
----------------------

.. autodata:: vcfpy.filterstats.CATEGORY_NOT_DONE
    :noindex:

.. autodata:: vcfpy.filterstats.CATEGORY_PASSED
    :noindex:

.. autodata:: vcfpy.filterstats.UNMAPPED_PREFIX
    :noindex:

.. autofunction:: vcfpy.filterstats.unmapped_category

.. autofunction:: vcfpy.filterstats.record_outcome

.. autofunction:: vcfpy.filterstats.sample_outcomes
