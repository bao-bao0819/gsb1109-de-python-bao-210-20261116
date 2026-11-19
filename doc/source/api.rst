.. _api:


.. rst-class:: html-toggle

API
===

`gffutils`
----------

.. automodule:: gffutils
    :members:

Create a database
-----------------


.. currentmodule:: gffutils.create

.. autosummary::
    :toctree: autodocs
    :nosignatures:

    create_db

Interact with a database
------------------------
First, connect to an existing database:

.. currentmodule:: gffutils.interface

.. autosummary::
    :toctree: autodocs
    :nosignatures:

    FeatureDB

Then, use the methods of :class:`FeatureDB` to interact:


.. autosummary::
    :toctree: autodocs
    :nosignatures:

    FeatureDB.children
    FeatureDB.parents
    FeatureDB.schema
    FeatureDB.features_of_type
    FeatureDB.count_features_of_type
    FeatureDB.all_features
    FeatureDB.execute
    FeatureDB.featuretypes
    FeatureDB.region
    FeatureDB.iter_by_parent_childs

Modify a :class:`FeatureDB`:

.. autosummary::
    :toctree: autodocs
    :nosignatures:

    FeatureDB.update
    FeatureDB.delete
    FeatureDB.add_relation
    FeatureDB.set_pragmas

Operate on features:

.. autosummary::
    :toctree: autodocs
    :nosignatures:

    FeatureDB.interfeatures
    FeatureDB.children_bp
    FeatureDB.merge
    FeatureDB.create_introns
    FeatureDB.explain_junctions
    FeatureDB.create_splice_sites
    FeatureDB.bed12

Resolve junction observations:

.. currentmodule:: gffutils.junctions

.. autosummary::
    :toctree: autodocs
    :nosignatures:

    read_junctions
    group_junctions
    explain_junctions
    Junction
    JunctionGroup
    JunctionEvidence
    GroupExplanation
    JunctionReport

Explaining junction observations
--------------------------------
A junction table records which junctions came from the same molecule.  Use
:func:`~gffutils.junctions.read_junctions` to parse it and
:func:`~gffutils.junctions.explain_junctions` (or
:meth:`FeatureDB.explain_junctions`) to test whether a single transcript can
explain *all* junctions in a molecule:

>>> db = gffutils.create_db(db_fn, ':memory:')  # doctest: +SKIP
>>> junctions = gffutils.read_junctions(junction_table_fn)  # doctest: +SKIP
>>> report = gffutils.explain_junctions(db, junctions)  # doctest: +SKIP
>>> for group in report:  # doctest: +SKIP
...     print(group.molecule, group.status, group.candidates)

The status is ``unique`` (one transcript explains every junction), ``shared``
(more than one does), or ``none`` (no single transcript does).  Individual
junction matches are available as ``group.evidence`` but never promote a
``none`` group to a found state.  Missing observations are not interpreted as
confirmed exon skipping; pass ``allow_exon_skipping=True`` only when an
observed junction's donor and acceptor explicitly span annotated exons.




Feature objects
---------------
Most :class:`FeatureDB` methods return :class:`Feature` objects:


.. currentmodule:: gffutils.feature

.. autosummary::
    :toctree: autodocs
    :nosignatures:

    Feature

You can extract the sequence for a feature:

.. autosummary::
    :toctree: autodocs
    :nosignatures:

    Feature.sequence

Creating a :class:`Feature` object:

.. autosummary::
    :toctree: autodocs
    :nosignatures:

    feature_from_line

Integration with other tools
----------------------------
.. currentmodule:: gffutils

.. autosummary::
    :toctree: autodocs
    :nosignatures:

    biopython_integration.to_seqfeature
    biopython_integration.from_seqfeature
    pybedtools_integration.tsses
    pybedtools_integration.to_bedtool



Utilities
---------

.. autosummary::
    :toctree: autodocs
    :nosignatures:

    helpers.asinterval
    helpers.merge_attributes
    helpers.sanitize_gff_db
    helpers.annotate_gff_db
    helpers.infer_dialect
    helpers.example_filename
    inspect.inspect
