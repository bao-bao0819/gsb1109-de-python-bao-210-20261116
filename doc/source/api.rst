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
    FeatureDB.sequence_length
    FeatureDB.set_sequence_lengths
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
    FeatureDB.bed12




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


Circular sequences
------------------
Features on circular sequences that cross the origin are stored with
``start > end``.  Providing sequence lengths (for example, a chrom.sizes or
FASTA file) lets :mod:`gffutils` resolve the logical feature and the actual
covered segments while preserving parent/child relations and strand:

.. currentmodule:: gffutils.circular

.. autosummary::
    :toctree: autodocs
    :nosignatures:

    LogicalFeature
    coverage_segments
    covered_length
    circular_extent
    merge_segments
    segments_overlap
    segments_within
    load_sequence_lengths
    reverse_complement

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
