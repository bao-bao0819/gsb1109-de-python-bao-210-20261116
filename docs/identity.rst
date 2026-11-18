Source-scoped identity, aliases, and lineage
============================================

GTFS identifiers such as ``route_id`` are only unique within a single feed.
When feeds from different agencies are loaded into one database, two agencies
may legitimately publish the same id (for example, both using
``route_id = "1"``). pygtfs keeps these apart using three complementary
concepts.

Source-scoped identity
----------------------

Every identified entity imported from a feed is registered in
:class:`pygtfs.gtfs_entities.EntityIdentity` with a natural key of
``(feed_id, entity_type, source_id)``. Registration happens automatically at
the end of :func:`pygtfs.append_feed`, so no two feeds ever merge merely
because they reuse an id. Each identity starts its own alias group, with
``identity_key == identity_id``.

Explicit aliases
----------------

Cross-feed equivalence is never inferred. To state that two source identities
denote the same real-world entity, declare an explicit alias with
:func:`pygtfs.add_alias`, a batch via :func:`pygtfs.add_aliases`, or a CSV
file via :func:`pygtfs.add_aliases_from_file`. Members of an alias group then
share one ``identity_key``, while each feed keeps its own entity and detail
rows. A shared stop, on its own, does not imply a shared route.

Feed publication lineage
------------------------

Different publications of the same upstream feed are linked, not merged. Pass
``predecessor_feed_id`` to :func:`pygtfs.append_feed` (or use the
``gtfs2db --predecessor-feed-id`` option) to record a
:class:`pygtfs.gtfs_entities.FeedLineage` row. The older publication is never
deleted, so historical detail stays queryable; use
:func:`pygtfs.lineage_chain` to walk from a publication back through its
predecessors.

.. automodule:: pygtfs.identity
   :members:
   :undoc-members:
