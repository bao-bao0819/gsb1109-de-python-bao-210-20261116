from collections import namedtuple

import sqlalchemy
import sqlalchemy.orm
from sqlalchemy import and_

from .gtfs_entities import (gtfs_all, gtfs_derived, Feed, Base, Translation,
                            TranslationTarget)


#: One flattened row of :meth:`Schedule.translation_view`.
TranslationViewItem = namedtuple(
    'TranslationViewItem',
    ['feed_id', 'target_table', 'target_field', 'target_record_id',
     'trans_id', 'lang', 'translation', 'match_basis', 'field_value'])


class Schedule:
    """Represents the full database.

    The schedule is the most important object in pygtfs. It represents the
    entire dataset. Most of the properties come straight from the gtfs
    reference. Two of them were renamed: calendar is called `services`, and
    calendar_dates `service_exceptions`. One addition is the `feeds` table,
    which is here to support more than one feed in a database.

    Each of the properties is a list created upon access by sqlalchemy. Then,
    each element of the list as attributes following the gtfs reference. In
    addition, if they are related to another table, this can also be accessed
    by attribute.

    :param db_conection: Either a sqlalchemy database url or a filename to be used with sqlite.

    """

    def __init__(self, db_connection):
        self.db_connection = db_connection
        self.db_filename = None
        if '://' not in db_connection:
            self.db_connection = 'sqlite:///%s' % self.db_connection
        if self.db_connection.startswith('sqlite'):
            self.db_filename = self.db_connection
        self.engine = sqlalchemy.create_engine(self.db_connection)
        Session = sqlalchemy.orm.sessionmaker(bind=self.engine)
        self.session = Session()
        Base.metadata.create_all(self.engine)

    def drop_feed(self, feed_id):
        """ Delete a feed from a database by feed id"""
        # the following does not cascade unfortunatly.
        # self.session.query(Feed).filter(Feed.feed_id == feed_id).delete()
        feed = self.session.query(Feed).get(feed_id)
        self.session.delete(feed)
        self.session.commit()

    def translation_view(self, lang=None, table_name=None, field_name=None,
                         record_id=None, feed_id=None):
        """Return resolved translations as flat view rows.

        Rows are filtered by target ``lang`` and, optionally, by target
        scope (``table_name``/``field_name``/``record_id`` and ``feed_id``).
        Each :class:`TranslationViewItem` carries the ``match_basis``
        (``record_id`` or ``field_value``) that actually produced the hit,
        together with the matched ``field_value`` when that basis applies.

        A scope without a unique translation yields one row per candidate
        instead of a single picked value: record-id scoped hits never expand
        to other records that happen to share the same field value.
        """
        target = TranslationTarget
        query = (self.session.query(
                     target.feed_id, target.target_table, target.target_field,
                     target.target_record_id, target.trans_id, target.lang,
                     Translation.translation, target.match_basis,
                     target.field_value)
                 .join(Translation, and_(
                     Translation.feed_id == target.translation_feed_id,
                     Translation.trans_id == target.trans_id,
                     Translation.lang == target.lang)))
        if lang is not None:
            query = query.filter(target.lang == lang)
        if table_name is not None:
            query = query.filter(target.target_table == table_name)
        if field_name is not None:
            query = query.filter(target.target_field == field_name)
        if record_id is not None:
            query = query.filter(target.target_record_id == record_id)
        if feed_id is not None:
            query = query.filter(target.feed_id == feed_id)
        rows = (query.order_by(target.feed_id, target.target_table,
                               target.target_field, target.target_record_id,
                               target.lang, target.match_basis).all())
        return [TranslationViewItem(*row) for row in rows]


def _meta_query_all(entity, docstring=None):
    def _query_all(instance_self):
        """ A list generated on access """
        return instance_self.session.query(entity).all()

    if docstring is not None:
        _query_all.__doc__ = docstring
    return property(_query_all)


def _meta_query_by_id(entity, docstring=None):
    def _query_by_id(self, id):
        """ A function that returns a list of entries with matching ids """
        return self.session.query(entity).filter(entity.id == id).all()
    if docstring is not None:
        _query_by_id.__doc__ = docstring
    return _query_by_id


def _meta_query_raw(entity, docstring=None):
    def _query_raw(instance_self):
        """
            A raw sqlalchemy query object that the user can then manipulate
            manually
        """
        return instance_self.session.query(entity)

    if docstring is not None:
        _query_raw.__doc__ = docstring
    return property(_query_raw)


for entity in (gtfs_all + gtfs_derived + [Feed]):
    entity_doc = "A list of :py:class:`pygtfs.gtfs_entities.{0}` objects".format(entity.__name__)
    entity_raw_doc = ("A :py:class:`sqlalchemy.orm.Query` object to fetch "
                      ":py:class:`pygtfs.gtfs_entities.{0}` objects"
                      .format(entity.__name__))
    entity_by_id_doc = "A list of :py:class:`pygtfs.gtfs_entities.{0}` objects with matching id".format(entity.__name__)
    setattr(Schedule, entity._plural_name_, _meta_query_all(entity, entity_doc))
    setattr(Schedule, entity._plural_name_ + "_query",
            _meta_query_raw(entity, entity_raw_doc))
    if hasattr(entity, 'id'):
        setattr(Schedule, entity._plural_name_ + "_by_id", _meta_query_by_id(entity, entity_by_id_doc))

