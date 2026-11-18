"""Source-scoped entity identity, explicit aliases, and feed lineage.

GTFS identifiers such as ``route_id`` are only unique *within one feed*.
When several feeds from different agencies are loaded into one database, the
same ``route_id`` may legitimately denote unrelated routes. The functions in
this module make that distinction explicit:

* :func:`register_feed_identities` gives every imported entity an identity row
  whose natural key is ``(feed_id, entity_type, source_id)``. Identifiers that
  collide across feeds therefore never get merged by accident.
* :func:`add_alias` (and its CSV wrapper :func:`add_aliases_from_file`) is the
  only way to state that two source identities denote the same real-world
  entity. Members of an alias group share an ``identity_key``.
* :func:`record_lineage` stores continuation relationships between different
  publications of the same feed. Lineage never merges entities and never
  deletes an older publication, so historical detail rows stay reviewable.

Nothing here is inferred from matching ids, names, or coordinates: cross-feed
equivalence and publication lineage are both supplied explicitly.
"""

import csv

from .exceptions import PygtfsException
from .gtfs_entities import (EntityAlias, EntityIdentity, Feed, FeedLineage,
                            identity_entity_map)


# Public entity type name -> (ORM class, column holding the feed-unique GTFS
# identifier). The class registry lives in gtfs_entities; every supported id
# column follows the ``<entity_type>_id`` naming convention.
_ID_FIELDS = {
    entity_type: (entity_class, entity_type + "_id")
    for entity_type, entity_class in identity_entity_map.items()
}


def _check_entity_type(entity_type):
    if entity_type not in _ID_FIELDS:
        raise PygtfsException(
            "Unknown entity type {0!r}; expected one of {1}".format(
                entity_type, sorted(_ID_FIELDS)))


def get_identity(session, feed_id, entity_type, source_id):
    """Return the :class:`EntityIdentity` for one source id, or ``None``."""
    _check_entity_type(entity_type)
    return (session.query(EntityIdentity)
            .filter(EntityIdentity.feed_id == feed_id)
            .filter(EntityIdentity.entity_type == entity_type)
            .filter(EntityIdentity.source_id == source_id)
            .one_or_none())


def register_feed_identities(session, feed_id):
    """Register a source-scoped identity for every identified entity in a feed.

    Idempotent: natural keys that already have an identity row are skipped, so
    it is safe to call after every import. Each new row starts its own alias
    group, with ``identity_key == identity_id``.

    Returns the list of newly created :class:`EntityIdentity` rows.
    """
    created = []
    for entity_type, (entity_class, id_column) in _ID_FIELDS.items():
        column = getattr(entity_class, id_column)
        source_ids = [
            row[0] for row in (session.query(column)
                               .filter(entity_class.feed_id == feed_id)
                               .distinct().all())
            if row[0] is not None
        ]
        existing = {
            row[0] for row in (session.query(EntityIdentity.source_id)
                               .filter(EntityIdentity.feed_id == feed_id)
                               .filter(EntityIdentity.entity_type ==
                                       entity_type).all())
        }
        for source_id in source_ids:
            if source_id in existing:
                continue
            # identity_key is NOT NULL; insert with a placeholder so the
            # autoincrement identity_id is allocated, then backfill the key
            # to point the new group at its own row.
            identity = EntityIdentity(feed_id=feed_id,
                                      entity_type=entity_type,
                                      source_id=source_id,
                                      identity_key=0)
            session.add(identity)
            created.append(identity)
    session.flush()
    for identity in created:
        identity.identity_key = identity.identity_id
    session.flush()
    return created


def register_all_identities(session):
    """Backfill source-scoped identities for every feed already in the DB.

    Useful for a database created before identity tracking existed: the new
    tables are created empty, so call this once to register identities for all
    existing feeds. Idempotent, mirroring
    :func:`register_feed_identities`. Returns the total number of new rows.
    """
    created_count = 0
    for (feed_id,) in session.query(Feed.feed_id).all():
        created_count += len(register_feed_identities(session, feed_id))
    return created_count


def _require_identity(session, feed_id, entity_type, source_id):
    identity = get_identity(session, feed_id, entity_type, source_id)
    if identity is None:
        raise PygtfsException(
            "No registered {0} identity for feed_id={1}, source_id={2!r}. "
            "Import the feed first so its identities are registered.".format(
                entity_type, feed_id, source_id))
    return identity


def add_alias(session, entity_type, feed_id_a, source_id_a,
              feed_id_b, source_id_b):
    """Declare that two source identities denote the same real-world entity.

    Both identities must already be registered and must share the same
    ``entity_type``. The call is idempotent: re-declaring an existing alias,
    in either direction, is a no-op. The two alias groups are unioned by
    repointing every member to the smallest existing ``identity_key``.

    Returns the shared ``identity_key`` of the merged group.
    """
    _check_entity_type(entity_type)
    identity_a = _require_identity(session, feed_id_a, entity_type,
                                   source_id_a)
    identity_b = _require_identity(session, feed_id_b, entity_type,
                                   source_id_b)
    if identity_a.identity_id == identity_b.identity_id:
        return identity_a.identity_key

    existing = (session.query(EntityAlias)
                .filter(EntityAlias.entity_type == entity_type)
                .filter(EntityAlias.identity_a.in_(
                    [identity_a.identity_id, identity_b.identity_id]))
                .filter(EntityAlias.identity_b.in_(
                    [identity_a.identity_id, identity_b.identity_id]))
                .one_or_none())
    if existing is None:
        session.add(EntityAlias(
            entity_type=entity_type,
            identity_a=identity_a.identity_id,
            identity_b=identity_b.identity_id))

    if identity_a.identity_key != identity_b.identity_key:
        target_key = min(identity_a.identity_key, identity_b.identity_key)
        stale_key = max(identity_a.identity_key, identity_b.identity_key)
        # Update through the ORM (rather than a bulk query) so objects already
        # loaded in this session reflect the new identity_key immediately.
        members = (session.query(EntityIdentity)
                   .filter(EntityIdentity.entity_type == entity_type)
                   .filter(EntityIdentity.identity_key == stale_key)
                   .all())
        for member in members:
            member.identity_key = target_key
        session.flush()
    return min(identity_a.identity_key, identity_b.identity_key)


def add_aliases(session, aliases):
    """Apply an iterable of ``(entity_type, feed_a, id_a, feed_b, id_b)``."""
    return [add_alias(session, *entry) for entry in aliases]


def add_aliases_from_file(session, alias_filename):
    """Load explicit aliases from a CSV file.

    The required header is::

        entity_type,feed_id_a,source_id_a,feed_id_b,source_id_b

    Blank lines and lines starting with ``#`` are ignored. Each row is applied
    via :func:`add_alias`, so both feeds must already be imported.
    """
    applied = []
    with open(alias_filename, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(
            line for line in handle if line.strip() and not
            line.lstrip().startswith("#"))
        required = {'entity_type', 'feed_id_a', 'source_id_a',
                    'feed_id_b', 'source_id_b'}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise PygtfsException(
                "Alias file must have columns {0}".format(
                    sorted(required)))
        for row in reader:
            applied.append(add_alias(
                session,
                row['entity_type'].strip(),
                int(row['feed_id_a']),
                row['source_id_a'].strip(),
                int(row['feed_id_b']),
                row['source_id_b'].strip()))
    return applied


def identities_for_key(session, entity_type, identity_key):
    """Return every source identity sharing one real-world ``identity_key``."""
    _check_entity_type(entity_type)
    return (session.query(EntityIdentity)
            .filter(EntityIdentity.entity_type == entity_type)
            .filter(EntityIdentity.identity_key == identity_key)
            .order_by(EntityIdentity.feed_id, EntityIdentity.source_id)
            .all())


def record_lineage(session, feed_id, predecessor_feed_id, note=None):
    """Record that one feed publication continues an earlier publication.

    Both feeds must already exist and must be distinct. A publication has at
    most one predecessor; re-recording the same link is a no-op. The older
    publication is never modified or deleted, so its detail stays reviewable.
    """
    if feed_id == predecessor_feed_id:
        raise PygtfsException(
            "A feed cannot be its own predecessor (feed_id={0})".format(
                feed_id))
    feeds = {feed.feed_id: feed for feed in session.query(Feed).all()}
    if feed_id not in feeds:
        raise PygtfsException("Unknown feed_id {0}".format(feed_id))
    if predecessor_feed_id not in feeds:
        raise PygtfsException(
            "Unknown predecessor_feed_id {0}".format(predecessor_feed_id))

    existing = (session.query(FeedLineage)
                .filter(FeedLineage.feed_id == feed_id)
                .one_or_none())
    if existing is not None:
        if existing.predecessor_feed_id == predecessor_feed_id:
            return existing
        raise PygtfsException(
            "Feed {0} already records predecessor {1}; a publication has at "
            "most one predecessor".format(feed_id,
                                          existing.predecessor_feed_id))
    link = FeedLineage(feed_id=feed_id,
                       predecessor_feed_id=predecessor_feed_id,
                       note=note)
    session.add(link)
    session.flush()
    return link


def lineage_chain(session, feed_id):
    """Walk predecessor links from ``feed_id`` back to the oldest publication.

    Returns a list of feed ids ordered newest-first, starting with
    ``feed_id``.
    """
    chain = [feed_id]
    current = feed_id
    seen = {feed_id}
    while True:
        link = (session.query(FeedLineage)
                .filter(FeedLineage.feed_id == current)
                .one_or_none())
        if link is None:
            return chain
        predecessor = link.predecessor_feed_id
        if predecessor in seen:
            raise PygtfsException(
                "Cycle detected in feed lineage at feed_id {0}".format(
                    predecessor))
        seen.add(predecessor)
        chain.append(predecessor)
        current = predecessor
