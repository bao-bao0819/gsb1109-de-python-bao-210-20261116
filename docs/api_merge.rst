.. _api_merge:

=====
Merge
=====

.. contents::

Header compatibility is decided from each field's declared ``Number`` and
``Type``.  Observed value lengths in individual records are not used, because
they do not define the meaning of a field.

vcfpy.plan_merge
---------------

.. autofunction:: vcfpy.plan_merge

vcfpy.merge_headers
-------------------

.. autofunction:: vcfpy.merge_headers

vcfpy.MergePlan
---------------

.. autoclass:: vcfpy.MergePlan
    :members:

vcfpy.FieldRename
-----------------

.. autoclass:: vcfpy.FieldRename
    :members:

vcfpy.RenameCollection
----------------------

.. autoclass:: vcfpy.RenameCollection
    :members:

vcfpy.MergeConflict
-------------------

.. autoclass:: vcfpy.MergeConflict
    :members:

vcfpy.merge_records
-------------------

.. autofunction:: vcfpy.merge_records

vcfpy.merge_vcf_files
---------------------

.. autofunction:: vcfpy.merge_vcf_files

vcfpy.MergeError
----------------

.. autoexception:: vcfpy.MergeError

vcfpy.MergeConflictError
------------------------

.. autoexception:: vcfpy.MergeConflictError
