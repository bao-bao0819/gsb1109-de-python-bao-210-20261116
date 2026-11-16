"""Structured record views for wiki tables.

A table row is not always a business record.  A rowspan can tie several
physical rows to one anchored cell, and a colspan can make one cell cover
several logical columns.  The classes in this module keep those logical
relations intact while exposing each value together with:

* the display text rendered by the table,
* the sort key parsed from ``data-sort-value`` or the display text,
* the cell that supplied the value.

The view is read-only: sorting returns a new result and never rewrites the
original wikitext (in particular, a hidden ``data-sort-value`` is never
copied back into the visible value).
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Iterator, Sequence

import regex

from ._cell import Cell
from ._wikitext import WS

if TYPE_CHECKING:
    from ._table import Table


_NUMBER_RE = re.compile(
    r'^(?P<currency>[$€£¥₩₹])?'
    r'(?P<number>[+-]?(?:'
    r'(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|'
    r'\.\d+'
    r')(?:[eE][+-]?\d+)?)'
    r'(?P<suffix>.*)$'
)
_DATE_RE = re.compile(
    r'^(?P<year>\d{4})(?P<sep>[-/.])'
    r'(?P<month>\d{1,2})(?P=sep)(?P<day>\d{1,2})'
    r'(?:[T ](?P<hour>\d{1,2}):(?P<minute>\d{2})'
    r'(?::(?P<second>\d{2})(?:\.\d+)?)?'
    r'(?:Z|[+-]\d{2}:?\d{2})?)?$'
)
_INVALID_SORT_VALUES = frozenset(
    {
        '',
        '-',
        '--',
        '---',
        '—',
        '–',
        '?',
        '??',
        'n/a',
        'na',
        'nil',
        'none',
        'null',
        'unknown',
        'tbd',
    }
)
_CJK_SCALES = {
    '十': Decimal(10),
    '百': Decimal(100),
    '千': Decimal(1000),
    '万': Decimal(10000),
    '亿': Decimal(100000000),
}
_LATIN_SCALES = {'k': Decimal(1000)}
_PERCENT_UNITS = {'%', 'pct', 'percent', '百分比'}
_PERMILLE_UNITS = {'‰', 'per mille', 'permille'}
_BAD_UNIT_START = frozenset('-+/\\:.,;()[]{}<>')
_CJK_NUMBERS = frozenset('〇一二三四五六七八九十百千万亿两壹贰叁肆伍陆柒捌玖拾佰仟')
_APPROX_MARKERS = (
    '约',
    '大约',
    '大概',
    '近',
    '将近',
    '接近',
    '超过',
    '逾',
    '不足',
    '不到',
    '至少',
    '最多',
    '左右',
    '上下',
    '以上',
    '以下',
    '多',
    '余',
    '~',
    '～',
    '≈',
)
_TRAILING_PLUS_RE = re.compile(r'[+＋]\s*$')
_COMPARISON_RE = re.compile(r'[<>≤≥＜＞]\s*\d|\d\s*[<>≤≥＜＞]')
_APPROX_RE = re.compile('|'.join(map(re.escape, _APPROX_MARKERS)))
_RANGE_RE = re.compile(
    r'(?<=\d)\s*[-–—~～至到]\s*(?=\d)'
    r'|(?<=[〇一二三四五六七八九十百千万亿两])\s*'
    r'[-–—~～至到]\s*'
    r'(?=[〇一二三四五六七八九十百千万亿两])'
)
_CURRENCY_PREFIX_RE = re.compile(
    r'^(?P<code>RMB|CNY|USD|EUR|GBP|JPY|HKD|KRW|INR|US\$|HK\$)\s*',
    re.IGNORECASE,
)
_CURRENCY_CODES = {
    'rmb': 'CNY',
    'cny': 'CNY',
    'usd': 'USD',
    'us$': 'USD',
    'eur': 'EUR',
    'gbp': 'GBP',
    'jpy': 'JPY',
    'hkd': 'HKD',
    'hk$': 'HKD',
    'krw': 'KRW',
    'inr': 'INR',
}
# family -> factor from the displayed unit to the family's base unit
_CASE_SENSITIVE_UNITS = frozenset({'m', 't', 'b', 'g', 's', 'h'})
_UNIT_ALIASES: dict[str, tuple[str, Decimal]] = {
    '': ('', Decimal(1)),
    '%': ('%', Decimal(1)),
    'percent': ('%', Decimal(1)),
    '百分比': ('%', Decimal(1)),
    '$': ('USD', Decimal(1)),
    'usd': ('USD', Decimal(1)),
    '美元': ('USD', Decimal(1)),
    '€': ('EUR', Decimal(1)),
    'eur': ('EUR', Decimal(1)),
    '欧元': ('EUR', Decimal(1)),
    '£': ('GBP', Decimal(1)),
    'gbp': ('GBP', Decimal(1)),
    '英镑': ('GBP', Decimal(1)),
    '¥': ('CNY', Decimal(1)),
    'jpy': ('JPY', Decimal(1)),
    '円': ('JPY', Decimal(1)),
    '日元': ('JPY', Decimal(1)),
    'cny': ('CNY', Decimal(1)),
    'rmb': ('CNY', Decimal(1)),
    '元': ('CNY', Decimal(1)),
    '人民币': ('CNY', Decimal(1)),
    '₩': ('KRW', Decimal(1)),
    'krw': ('KRW', Decimal(1)),
    '韩元': ('KRW', Decimal(1)),
    '₹': ('INR', Decimal(1)),
    'inr': ('INR', Decimal(1)),
    'hkd': ('HKD', Decimal(1)),
    '港币': ('HKD', Decimal(1)),
    '港元': ('HKD', Decimal(1)),
    'm': ('length', Decimal(1)),
    '米': ('length', Decimal(1)),
    'km': ('length', Decimal(1000)),
    '千米': ('length', Decimal(1000)),
    '公里': ('length', Decimal(1000)),
    'cm': ('length', Decimal('0.01')),
    '厘米': ('length', Decimal('0.01')),
    'mm': ('length', Decimal('0.001')),
    '毫米': ('length', Decimal('0.001')),
    'mi': ('length', Decimal('1609.344')),
    '英里': ('length', Decimal('1609.344')),
    'ft': ('length', Decimal('0.3048')),
    '英尺': ('length', Decimal('0.3048')),
    'in': ('length', Decimal('0.0254')),
    '英寸': ('length', Decimal('0.0254')),
    'kg': ('mass', Decimal(1)),
    '千克': ('mass', Decimal(1)),
    '公斤': ('mass', Decimal(1)),
    'g': ('mass', Decimal('0.001')),
    '克': ('mass', Decimal('0.001')),
    't': ('mass', Decimal(1000)),
    '吨': ('mass', Decimal(1000)),
    's': ('time', Decimal(1)),
    '秒': ('time', Decimal(1)),
    'min': ('time', Decimal(60)),
    '分钟': ('time', Decimal(60)),
    'h': ('time', Decimal(3600)),
    '小时': ('time', Decimal(3600)),
    '天': ('time', Decimal(86400)),
    'b': ('data', Decimal(1)),
    'kb': ('data', Decimal(1024)),
    'mb': ('data', Decimal(1048576)),
    'gb': ('data', Decimal(1073741824)),
    'tb': ('data', Decimal('1099511627776')),
}
_UNIT_COLUMN_NAMES = frozenset(
    {
        'unit',
        'units',
        '单位',
        '计量单位',
        '货币',
        '币种',
        'currency',
    }
)


class RecordError(ValueError):
    """Base class for record-view errors."""


class ColumnError(RecordError, KeyError):
    """Raised when a requested column cannot be resolved."""


@dataclass
class CellSource:
    """A snapshot of the cell that supplied a record value.

    ``row`` and ``column`` are the requested logical position.  ``anchor_row``
    and ``anchor_column`` identify where the actual cell starts.  For a cell
    covered by a rowspan or colspan those are different from the requested
    position, which is exactly the provenance information a caller needs.
    """

    row: int
    column: int
    anchor_row: int
    anchor_column: int
    span: tuple[int, int] | None
    wikitext: str | None
    attrs: dict[str, str] = field(default_factory=dict)
    header: bool = False
    cell: Cell | None = field(default=None, repr=False, compare=False)

    @property
    def is_anchor(self) -> bool:
        """Whether this source is the top-left slot of the cell."""
        return (
            self.row == self.anchor_row and self.column == self.anchor_column
        )

    @property
    def rowspan(self) -> int | None:
        """The declared rowspan, or None when it was not declared."""
        value = self.attrs.get('rowspan')
        return _positive_int(value)

    @property
    def colspan(self) -> int | None:
        """The declared colspan, or None when it was not declared."""
        value = self.attrs.get('colspan')
        return _positive_int(value)

    def as_dict(self) -> dict[str, object]:
        """Return a serialisable description of this source."""
        return {
            'row': self.row,
            'column': self.column,
            'anchor_row': self.anchor_row,
            'anchor_column': self.anchor_column,
            'span': self.span,
            'wikitext': self.wikitext,
            'attrs': dict(self.attrs),
            'header': self.header,
        }


class RecordValue(str):
    """One cell value inside a logical record.

    Instances compare equal to their display text, so existing string-based
    code can consume them directly.  ``display`` is the value as written in
    the table and is never replaced by ``data-sort-value``.  ``sort`` is the
    parsed key used for ordering; it is a :class:`~decimal.Decimal` for
    numeric keys, a tuple for ISO-like dates, and :class:`str` for textual
    keys.  ``sort_issue`` is set when no usable key exists.
    """

    __slots__ = (
        'raw',
        'sort',
        'sort_kind',
        'unit',
        'unit_text',
        'sort_source',
        'sort_issue',
        'sort_detail',
        'source',
    )

    def __new__(
        cls,
        display: str,
        raw: str,
        sort: Decimal | str | tuple[int, ...] | None,
        sort_kind: str | None,
        unit: str | None,
        sort_source: str | None,
        sort_issue: str | None,
        sort_detail: str | None,
        source: CellSource | None = None,
        unit_text: str | None = None,
    ) -> RecordValue:
        value = str.__new__(cls, display)
        value.raw = raw
        value.sort = sort
        value.sort_kind = sort_kind
        value.unit = unit
        value.unit_text = unit_text
        value.sort_source = sort_source
        value.sort_issue = sort_issue
        value.sort_detail = sort_detail
        value.source = source
        return value

    @property
    def display(self) -> str:
        """The display text of this value."""
        return str(self)

    @property
    def display_value(self) -> str:
        """Alias for :attr:`display` used by record-view callers."""
        return str(self)

    @property
    def value(self) -> str:
        """The display value (alias for :attr:`display`)."""
        return str(self)

    @property
    def sort_value(self) -> Decimal | str | tuple[int, ...] | None:
        """Alias for :attr:`sort` used by record-view callers."""
        return self.sort

    @property
    def is_sortable(self) -> bool:
        """Whether this value has a usable key before column checks."""
        return self.sort is not None and self.sort_issue is None

    @property
    def has_data_sort_value(self) -> bool:
        """Whether the key came from a hidden ``data-sort-value``."""
        return self.sort_source == 'data-sort-value'

    def as_dict(self) -> dict[str, object]:
        """Return a serialisable description of this value."""
        return {
            'display': str(self),
            'sort': self.sort,
            'sort_kind': self.sort_kind,
            'unit': self.unit,
            'unit_text': self.unit_text,
            'sort_source': self.sort_source,
            'sort_issue': self.sort_issue,
            'sort_detail': self.sort_detail,
            'source': None if self.source is None else self.source.as_dict(),
        }

    def __repr__(self) -> str:
        return (
            f'RecordValue({str(self)!r}, sort={self.sort!r}, '
            f'sort_source={self.sort_source!r})'
        )


# A shorter alias is convenient next to RecordValue.
RecordCell = RecordValue


@dataclass
class Record:
    """One logical business record.

    ``rows`` contains every physical table row that belongs to the record.
    A rowspan that ties several rows together therefore produces one Record,
    not one Record per physical row.  ``cells`` maps logical column indexes
    to the distinct values covering that column.  A column may contain more
    than one value when separate physical rows contribute different cells.
    """

    rows: tuple[int, ...]
    cells: dict[int, tuple[RecordValue, ...]]
    headers: tuple[str | None, ...] = field(
        default=(), repr=False, compare=False
    )

    @property
    def row(self) -> int:
        """The first physical row of the record."""
        return self.rows[0]

    @property
    def columns(self) -> tuple[int, ...]:
        """Sorted column indexes that contain a value."""
        return tuple(sorted(self.cells))

    def values(self, column: int) -> tuple[RecordValue, ...]:
        """Return every value covering ``column``, in source order."""
        return self.cells.get(column, ())

    def distinct_values(self, column: int) -> tuple[RecordValue, ...]:
        """Return values from different anchored cells in ``column``."""
        result: list[RecordValue] = []
        seen: set[tuple[int, int] | int] = set()
        for value in self.values(column):
            source = value.source
            key: tuple[int, int] | int
            if source is None:
                key = id(value)
            else:
                key = (source.anchor_row, source.anchor_column)
            if key in seen:
                continue
            seen.add(key)
            result.append(value)
        return tuple(result)

    def value(self, column: int) -> RecordValue | None:
        """Return the primary value for ``column``.

        The first non-empty value wins; when all values are empty the first
        one is still returned so callers can inspect its source.
        """
        values = self.values(column)
        if not values:
            return None
        for value in values:
            if value.display.strip():
                return value
        return values[0]

    def display(self, column: int) -> str:
        """Return the primary display text for ``column`` (or ``''``)."""
        value = self.value(column)
        return '' if value is None else value.display

    def displays(self, column: int) -> tuple[str, ...]:
        """Return all display texts covering ``column``."""
        return tuple(value.display for value in self.values(column))

    def sort_value(
        self, column: int
    ) -> Decimal | str | tuple[int, ...] | None:
        """Return the primary sort key for ``column``."""
        value = self.value(column)
        return None if value is None else value.sort

    def all_sources(self, column: int) -> tuple[CellSource, ...]:
        """Return the sources of every value covering ``column``."""
        return tuple(
            value.source
            for value in self.values(column)
            if value.source is not None
        )

    def sources(self, column: int) -> tuple[CellSource, ...]:
        """Return one source per distinct anchored cell in ``column``."""
        return tuple(
            value.source
            for value in self.distinct_values(column)
            if value.source is not None
        )

    def source(self, column: int) -> CellSource | None:
        """Return the source of the primary value for ``column``."""
        value = self.value(column)
        return None if value is None else value.source

    def __getitem__(self, column: int | str) -> RecordValue | None:
        if isinstance(column, str):
            matches = [
                index
                for index, label in enumerate(self.headers)
                if label is not None and label.strip() == column.strip()
            ]
            if not matches:
                folded = column.strip().casefold()
                matches = [
                    index
                    for index, label in enumerate(self.headers)
                    if label is not None
                    and label.strip().casefold() == folded
                ]
            if not matches:
                raise KeyError(column)
            if len(matches) > 1:
                raise ColumnError(
                    f'column name {column!r} is ambiguous; it matches '
                    f'{matches}'
                )
            column = matches[0]
        return self.value(column)

    def __iter__(self) -> Iterator[tuple[int, tuple[RecordValue, ...]]]:
        for column in self.columns:
            yield column, self.cells[column]

    def as_dict(
        self,
        headers: Sequence[str | None] | None = None,
        *,
        multiple: bool = False,
        include_sources: bool = False,
    ) -> dict[object, object]:
        """Return the record as a mapping.

        Keys are header labels when available, else column indexes.  With
        ``multiple=True`` each value is a tuple of every contributing
        display value; otherwise the primary display value is used.  With
        ``include_sources=True`` each value becomes a serialisable mapping
        containing the display text, sort key and source cell.
        """
        result: dict[object, object] = {}
        if headers is None:
            columns = self.columns
        else:
            columns = sorted(set(range(len(headers))) | set(self.columns))
        for column in columns:
            key: object
            if headers is not None and column < len(headers):
                key = headers[column]
                if key is None:
                    key = column
            else:
                key = column
            if include_sources:
                values = self.values(column)
                if multiple:
                    result[key] = [value.as_dict() for value in values]
                else:
                    primary = self.value(column)
                    result[key] = (
                        None if primary is None else primary.as_dict()
                    )
            elif multiple:
                result[key] = self.displays(column)
            else:
                result[key] = self.display(column)
        return result

    def __repr__(self) -> str:
        values = ', '.join(
            f'{column}={self.displays(column)!r}' for column in self.columns
        )
        return f'Record(rows={self.rows!r}, {values})'


TableRecord = Record


@dataclass
class UnsortableRecord:
    """A record that cannot take part in a column's ordered output."""

    record: Record
    column: int
    reason: str
    detail: str
    value: RecordValue | None = None

    @property
    def display(self) -> str:
        """The primary display value used for the failed key."""
        return self.record.display(self.column)

    @property
    def source(self) -> CellSource | None:
        """The source of the primary value used for the failed key."""
        return self.record.source(self.column)

    def as_dict(self) -> dict[str, object]:
        """Return a serialisable description of this entry."""
        return {
            'row': self.record.row,
            'rows': self.record.rows,
            'column': self.column,
            'display': self.display,
            'reason': self.reason,
            'detail': self.detail,
        }

    def __str__(self) -> str:
        return self.display

    def __repr__(self) -> str:
        return (
            f'UnsortableRecord(rows={self.record.rows!r}, '
            f'column={self.column}, reason={self.reason!r}, '
            f'display={self.display!r})'
        )


@dataclass
class RecordSortResult:
    """The outcome of sorting a :class:`RecordView` by one column.

    ``records`` contains the ordered, comparable records.  ``unsortable``
    contains records with a missing key, an invalid key, a key of the wrong
    kind, or an incompatible unit.  The two groups are disjoint and always
    contain whole logical records.
    """

    view: RecordView
    column: int
    reverse: bool
    records: tuple[Record, ...]
    unsortable: tuple[UnsortableRecord, ...]
    kind: str | None
    unit: str | None

    def __iter__(self) -> Iterator[Record]:
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]

    @property
    def column_name(self) -> str | None:
        """The header label of the sort column, if it has one."""
        if 0 <= self.column < len(self.view.headers):
            return self.view.headers[self.column]
        return None

    @property
    def sorted(self) -> tuple[Record, ...]:
        """Alias for :attr:`records`."""
        return self.records

    @property
    def invalid(self) -> tuple[UnsortableRecord, ...]:
        """Alias for :attr:`unsortable`."""
        return self.unsortable

    @property
    def separated(self) -> tuple[UnsortableRecord, ...]:
        """Alias for :attr:`unsortable`."""
        return self.unsortable

    def all_records(self) -> tuple[Record, ...]:
        """Return ordered records followed by the unsortable records."""
        return self.records + tuple(
            entry.record for entry in self.unsortable
        )

    def as_dicts(self) -> list[dict[object, object]]:
        """Return the ordered records as mappings."""
        return [record.as_dict(self.view.headers) for record in self.records]

    def unsortable_as_dicts(self) -> list[dict[str, object]]:
        """Return the separated records with their reasons."""
        return [entry.as_dict() for entry in self.unsortable]

    def unsortable_by_reason(self) -> dict[str, tuple[UnsortableRecord, ...]]:
        """Group separated records by reason, preserving their order."""
        grouped: dict[str, list[UnsortableRecord]] = {}
        for entry in self.unsortable:
            grouped.setdefault(entry.reason, []).append(entry)
        return {reason: tuple(entries) for reason, entries in grouped.items()}


class RecordView:
    """A read-only, logical record view of a :class:`Table`.

    Leading header rows are excluded from the records.  Header labels are
    available for column lookup by name.  A rowspan group is represented by
    one :class:`Record`; a colspan makes the same source value available to
    every covered logical column.
    """

    def __init__(
        self,
        table: Table,
        *,
        headers: Sequence[str] | None = None,
        strip: bool = True,
    ) -> None:
        self.table = table
        self.strip = strip
        (
            self.records,
            self.headers,
            self.width,
            self.header_rows,
        ) = _build_records(table, headers=headers, strip=strip)

    def __iter__(self) -> Iterator[Record]:
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]

    @property
    def rows(self) -> tuple[Record, ...]:
        """Alias for :attr:`records`."""
        return self.records

    def column_index(self, column: int | str) -> int:
        """Resolve ``column`` to a zero-based logical column index.

        An integer is interpreted as a column index (negative indexes are
        accepted).  A string is matched against header labels, first exactly
        and then case-insensitively.  A repeated label may be disambiguated
        with a one-based ``'Label#2'`` suffix.  Ambiguous or unknown names
        raise :class:`ColumnError`.
        """
        if isinstance(column, int) and not isinstance(column, bool):
            index = column
            if index < 0:
                index += self.width
            if not 0 <= index < self.width:
                raise ColumnError(
                    f'column index {column} is outside 0..{self.width - 1}'
                )
            return index
        if not isinstance(column, str):
            raise ColumnError(
                f'column must be an index or header label, not '
                f'{type(column).__name__}'
            )
        target = column.strip()
        occurrence = 1
        explicit_occurrence = False
        if '#' in target:
            base, _, suffix = target.rpartition('#')
            if suffix.isdigit() and int(suffix) > 0:
                target = base
                occurrence = int(suffix)
                explicit_occurrence = True
        candidates = [
            index
            for index, label in enumerate(self.headers)
            if label is not None and label.strip() == target
        ]
        if not candidates:
            folded = target.casefold()
            candidates = [
                index
                for index, label in enumerate(self.headers)
                if label is not None and label.strip().casefold() == folded
            ]
        if not candidates:
            raise ColumnError(
                f'no column named {column!r}; headers are '
                f'{self.headers!r}'
            )
        if len(candidates) > 1 and not explicit_occurrence:
            raise ColumnError(
                f'column name {column!r} is ambiguous; it matches '
                f'{candidates}'
            )
        if occurrence > len(candidates):
            raise ColumnError(
                f'column name {column!r} only occurs '
                f'{len(candidates)} time(s)'
            )
        return candidates[occurrence - 1]

    def column(self, column: int | str) -> tuple[str, ...]:
        """Return the primary display value of every record in a column."""
        index = self.column_index(column)
        return tuple(record.display(index) for record in self.records)

    def sort_by(
        self,
        column: int | str,
        *,
        reverse: bool = False,
    ) -> RecordSortResult:
        """Sort whole records by one logical column.

        Records without a usable key, with an invalid key, with a key of the
        other kind, or with an incompatible unit are returned separately in
        :attr:`RecordSortResult.unsortable`.  They are never assigned a zero
        key and never inserted among the ordered records.
        """
        if self.width == 0 and not self.records:
            if isinstance(column, int) and not isinstance(column, bool):
                index = column
            else:
                index = -1
            return RecordSortResult(
                view=self,
                column=index,
                reverse=reverse,
                records=(),
                unsortable=(),
                kind=None,
                unit=None,
            )
        index = self.column_index(column)
        return _sort_records(self, index, reverse)

    def sorted_by(
        self,
        column: int | str,
        *,
        reverse: bool = False,
    ) -> RecordSortResult:
        """Alias for :meth:`sort_by`."""
        return self.sort_by(column, reverse=reverse)

    def sort(
        self,
        column: int | str,
        *,
        reverse: bool = False,
    ) -> RecordSortResult:
        """Alias for :meth:`sort_by`."""
        return self.sort_by(column, reverse=reverse)

    def as_dicts(
        self,
        *,
        multiple: bool = False,
        include_sources: bool = False,
    ) -> list[dict[object, object]]:
        """Return all records as mappings keyed by header label or index."""
        return [
            record.as_dict(
                self.headers,
                multiple=multiple,
                include_sources=include_sources,
            )
            for record in self.records
        ]

    def as_lists(self) -> list[list[str]]:
        """Return all records as lists of primary display values."""
        return [
            [record.display(column) for column in range(self.width)]
            for record in self.records
        ]

    def __repr__(self) -> str:
        return (
            f'RecordView(records={len(self.records)}, '
            f'headers={self.headers!r})'
        )


def record_view(
    table: Table,
    *,
    headers: Sequence[str] | None = None,
    strip: bool = True,
) -> RecordView:
    """Return a :class:`RecordView` for ``table``."""
    return RecordView(table, headers=headers, strip=strip)


class _SortEntry:
    __slots__ = ('record', 'value', 'sort', 'unit', 'unit_text')

    def __init__(
        self,
        record: Record,
        value: RecordValue,
        *,
        sort: Decimal | str | tuple[int, ...] | None = None,
        unit: str | None = None,
        unit_text: str | None = None,
    ) -> None:
        self.record = record
        self.value = value
        self.sort = value.sort if sort is None else sort
        self.unit = value.unit if unit is None else unit
        self.unit_text = (
            value.unit_text if unit_text is None else unit_text
        )


def _positive_int(value: str | None) -> int | None:
    if value is None:
        return None
    match = regex.match(r'\s*(\d+)', value)
    if match is None:
        return None
    number = int(match[1])
    return number or 1


def _display_value(cell: Cell, strip: bool) -> tuple[str, str]:
    raw = cell.value
    if strip:
        return raw.lstrip(' ').rstrip(WS), raw
    return raw, raw


def _normalise_text(value: str) -> str:
    return unicodedata.normalize('NFKC', value)


def _normalise_unit(value: str) -> str:
    return regex.sub(r'\s+', ' ', _normalise_text(value)).strip().casefold()


def _known_unit(unit_text: str) -> tuple[str, Decimal] | None:
    """Return ``(family, factor)`` for a known unit, else ``None``.

    Unit symbols such as ``m`` and ``t`` are case-sensitive; words are not.
    """
    normalised = _normalise_text(unit_text).strip()
    if not normalised:
        return '', Decimal(1)
    if normalised in _CASE_SENSITIVE_UNITS:
        return _UNIT_ALIASES.get(normalised)
    folded = _normalise_unit(normalised)
    if folded in _CASE_SENSITIVE_UNITS and normalised != folded:
        return None
    return _UNIT_ALIASES.get(folded)


def _unit_family(unit_text: str) -> tuple[str, Decimal]:
    """Return ``(family, factor)`` for a unit-like suffix.

    Unknown but well-formed units get their own family so that they are
    reported as incompatible rather than silently compared as numbers.
    """
    known = _known_unit(unit_text)
    if known is not None:
        return known
    return 'unit:' + _normalise_unit(unit_text), Decimal(1)


_canonical_unit = _unit_family


def _looks_numeric_like(text: str) -> bool:
    return any(
        character.isdigit() or character in _CJK_NUMBERS
        for character in text
        if character not in '<>≤≥＜＞'
    )


def _parse_number_suffix(
    suffix: str,
) -> tuple[Decimal, str, str] | tuple[None, None, None]:
    """Return ``(scale, family, unit_text)`` for text after a number.

    ``None, None, None`` means that the suffix cannot be a unit and the
    whole value should be treated as text.  The scale converts the parsed
    number to the family's base unit.
    """
    suffix = suffix.strip()
    if not suffix:
        return Decimal(1), '', ''
    folded = _normalise_unit(suffix)
    if folded in _PERCENT_UNITS:
        return Decimal(1), '%', '%'
    if folded in _PERMILLE_UNITS:
        return Decimal('0.1'), '%', '%'
    if folded in _LATIN_SCALES:
        return _LATIN_SCALES[folded], '', suffix
    scale = Decimal(1)
    index = 0
    while index < len(suffix) and suffix[index] in _CJK_SCALES:
        scale *= _CJK_SCALES[suffix[index]]
        index += 1
    if index:
        unit_text = suffix[index:].strip()
        if not unit_text:
            return scale, '', ''
        family, factor = _unit_family(unit_text)
        return scale * factor, family, unit_text
    first = suffix[0]
    if first.isdigit() or first in _BAD_UNIT_START:
        return None, None, None
    family, factor = _unit_family(suffix)
    return factor, family, suffix.strip()


def _parse_sort_text(
    text: str,
) -> tuple[
    Decimal | str | tuple[int, ...] | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
]:
    """Return ``sort, kind, unit, unit_text, issue, detail``."""
    normalised = _normalise_text(text).strip()
    if normalised == '':
        return None, None, None, None, 'missing', 'the key is empty'
    folded = normalised.casefold()
    if folded in _INVALID_SORT_VALUES:
        return (
            None,
            None,
            None,
            None,
            'invalid',
            f'{normalised!r} is not a usable sort key',
        )
    date_match = _DATE_RE.match(normalised)
    if date_match is not None:
        parts = [
            int(date_match['year']),
            int(date_match['month']),
            int(date_match['day']),
            int(date_match['hour'] or 0),
            int(date_match['minute'] or 0),
            int(date_match['second'] or 0),
        ]
        return tuple(parts), 'date', None, None, None, None
    if _looks_numeric_like(normalised) and _RANGE_RE.search(normalised):
        return (
            None,
            None,
            None,
            None,
            'invalid',
            f'{normalised!r} is a range value, not an exact sort key',
        )
    if _looks_numeric_like(normalised) and (
        _APPROX_RE.search(normalised)
        or _TRAILING_PLUS_RE.search(normalised)
        or _COMPARISON_RE.search(normalised)
    ):
        return (
            None,
            None,
            None,
            None,
            'approximate',
            f'{normalised!r} is an approximate value, '
            f'not an exact sort key',
        )
    prefix = _CURRENCY_PREFIX_RE.match(normalised)
    prefix_family = None
    if prefix is not None:
        code = prefix['code']
        prefix_family = _CURRENCY_CODES[code.casefold()]
        normalised = normalised[prefix.end():]
    match = _NUMBER_RE.match(normalised)
    if match is not None:
        scale, family, unit_text = _parse_number_suffix(match['suffix'])
        if scale is not None:
            number = match['number'].replace(',', '')
            try:
                value = Decimal(number) * scale
            except InvalidOperation:
                pass
            else:
                currency = match['currency']
                if currency or prefix_family is not None:
                    currency_family, prefix_factor = _canonical_unit(
                        currency or prefix_family or ''
                    )
                    value *= prefix_factor
                    if currency_family not in ('', family):
                        family = currency_family
                    if not unit_text:
                        unit_text = currency or prefix_family
                return (
                    value,
                    'number',
                    family,
                    unit_text,
                    None,
                    None,
                )
    return normalised, 'text', None, None, None, None


def _span_sort_value(
    cell: Cell, value: str
) -> tuple[str, str] | None:
    """Return ``(sort text, display text)`` for a sortable inline span.

    Sortable tables sometimes put ``data-sort-value`` on a wrapping
    ``<span>`` instead of on the cell.  The span's contents are the rendered
    value; the span itself is markup and should not leak into the display.
    """
    sort_text: str | None = None
    replacements: list[tuple[int, int, str]] = []
    offset = 0
    for tag in cell.get_tags():
        if tag.name.casefold() != 'span':
            continue
        if not tag.has_attr('data-sort-value'):
            continue
        tag_string = tag.string
        start = value.find(tag_string, offset)
        if start < 0:
            continue
        offset = start + len(tag_string)
        contents = tag.contents or ''
        replacements.append((start, offset, contents))
        if sort_text is None:
            sort_text = tag.get_attr('data-sort-value') or ''
    if sort_text is not None:
        for start, end, contents in reversed(replacements):
            value = value[:start] + contents + value[end:]
        return sort_text, value
    return _regex_span_sort_value(value)


_SPAN_RE = re.compile(
    r'<span\b(?P<attrs>[^>]*)>(?P<contents>.*?)</span\s*>',
    re.IGNORECASE | re.DOTALL,
)
_SPAN_ATTR_RE = re.compile(
    r'data-sort-value\s*=\s*(?:"(?P<double>[^"]*)"|\'(?P<single>[^\']*)\')',
    re.IGNORECASE,
)


def _regex_span_sort_value(value: str) -> tuple[str, str] | None:
    """Span fallback for uppercase or otherwise unparsed span tags."""
    sort_text: str | None = None
    output: list[str] = []
    position = 0
    for match in _SPAN_RE.finditer(value):
        attr_match = _SPAN_ATTR_RE.search(match['attrs'])
        if attr_match is None:
            continue
        content = match['contents']
        output.append(value[position:match.start()])
        output.append(content)
        position = match.end()
        if sort_text is None:
            sort_text = (
                attr_match['double']
                if attr_match['double'] is not None
                else attr_match['single']
            )
    if sort_text is None:
        return None
    output.append(value[position:])
    return sort_text, ''.join(output)


def _make_value(
    cell: Cell,
    source: CellSource,
    strip: bool,
) -> RecordValue:
    display, raw = _display_value(cell, strip)
    key_text = display
    sort_source: str | None = 'display'
    if cell.has_attr('data-sort-value'):
        attr = cell.get_attr('data-sort-value')
        key_text = '' if attr is None else attr
        sort_source = 'data-sort-value'
    else:
        span_value = _span_sort_value(cell, display)
        if span_value is not None:
            key_text, display = span_value
            sort_source = 'data-sort-value'
            if strip:
                display = display.lstrip(' ').rstrip(WS)
    sort, kind, unit, unit_text, issue, detail = _parse_sort_text(key_text)
    if sort is None and sort_source == 'display' and not display.strip():
        sort_source = None
    if (
        sort_source == 'display'
        and ',' in display
        and kind == 'text'
        and not display.strip().startswith(('[', '{'))
    ):
        parts = [part.strip() for part in display.split(',')]
        if len(parts) > 1 and all(parts):
            sort = tuple(parts)
            kind = 'list'
    return RecordValue(
        display=display,
        raw=raw,
        sort=sort,
        sort_kind=kind,
        unit=unit,
        sort_source=sort_source,
        sort_issue=issue,
        sort_detail=detail,
        source=source,
        unit_text=unit_text,
    )


def _build_records(
    table: Table,
    *,
    headers: Sequence[str] | None,
    strip: bool,
) -> tuple[
    tuple[Record, ...],
    tuple[str | None, ...],
    int,
    tuple[int, ...],
]:
    physical_rows = table.cells(span=False)
    if not physical_rows:
        labels = tuple(headers) if headers is not None else ()
        return (), labels, len(labels), ()

    grid = table.cells(span=True)
    row_count = len(physical_rows)

    ranges: dict[int, list[int]] = {}
    for row_index, grid_row in enumerate(grid):
        for column_index, cell in enumerate(grid_row):
            if cell is None:
                continue
            key = id(cell)
            old = ranges.get(key)
            if old is None:
                ranges[key] = [
                    row_index,
                    row_index,
                    column_index,
                    column_index,
                ]
            else:
                old[0] = min(old[0], row_index)
                old[1] = max(old[1], row_index)
                old[2] = min(old[2], column_index)
                old[3] = max(old[3], column_index)

    header_rows: list[int] = []
    for row_index in range(min(row_count, len(grid))):
        cells = [cell for cell in grid[row_index] if cell is not None]
        if cells and all(cell.is_header for cell in cells):
            header_rows.append(row_index)
        else:
            break

    physical_width = max(
        (
            sum(
                max(1, _positive_int(cell.attrs.get('colspan')) or 1)
                for cell in row
                if cell is not None
            )
            for row in physical_rows
        ),
        default=0,
    )
    width = max(
        (rng[3] + 1 for rng in ranges.values()),
        default=physical_width,
    )
    if headers is not None:
        width = max(width, len(headers))
    labels = _build_headers(
        headers, header_rows, grid, ranges, width, strip
    )

    first_data = len(header_rows)
    data_rows = list(range(first_data, row_count))
    groups: list[tuple[int, int]] = []
    group_end = first_data - 1
    last_row = row_count - 1
    for row_index in data_rows:
        if row_index > group_end:
            groups.append((row_index, row_index))
        for cell in grid[row_index]:
            if cell is None or cell.is_header:
                continue
            cell_range = ranges.get(id(cell))
            if cell_range is None:
                continue
            group_end = max(group_end, min(cell_range[1], last_row))
        if groups:
            groups[-1] = (groups[-1][0], group_end)

    records: list[Record] = []
    for start, end in groups:
        cells: dict[int, list[RecordValue]] = {}
        seen: dict[int, set[int]] = {}
        for row_index in range(start, min(end, last_row) + 1):
            grid_row = grid[row_index] if row_index < len(grid) else []
            for cell in grid_row:
                if cell is None or cell.is_header:
                    continue
                cell_range = ranges.get(id(cell))
                if cell_range is None or cell_range[0] < start:
                    continue
                anchor_row, _, anchor_column, max_column = cell_range
                for column_index in range(anchor_column, max_column + 1):
                    if column_index >= width:
                        continue
                    column_seen = seen.setdefault(column_index, set())
                    if id(cell) in column_seen:
                        continue
                    column_seen.add(id(cell))
                    source = CellSource(
                        row=row_index,
                        column=column_index,
                        anchor_row=anchor_row,
                        anchor_column=anchor_column,
                        span=cell.span,
                        wikitext=cell.string,
                        attrs=dict(cell.attrs),
                        header=cell.is_header,
                        cell=cell,
                    )
                    cells.setdefault(column_index, []).append(
                        _make_value(cell, source, strip)
                    )
        records.append(
            Record(
                rows=tuple(range(start, min(end, last_row) + 1)),
                cells={
                    column: tuple(values)
                    for column, values in cells.items()
                },
                headers=labels,
            )
        )
    return (
        tuple(records),
        labels,
        width,
        tuple(header_rows),
    )


def _build_headers(
    headers: Sequence[str] | None,
    header_rows: list[int],
    grid: list[list[Cell | None]],
    ranges: dict[int, list[int]],
    width: int,
    strip: bool,
) -> tuple[str | None, ...]:
    if headers is not None:
        labels: list[str | None] = list(headers)
        if len(labels) < width:
            labels.extend([None] * (width - len(labels)))
        return tuple(labels)
    if len(header_rows) == 1:
        row = grid[header_rows[0]]
        labels = [None] * width
        for cell in row:
            if cell is None:
                continue
            cell_range = ranges.get(id(cell))
            if cell_range is None:
                continue
            value, _ = _display_value(cell, strip)
            for column_index in range(cell_range[2], cell_range[3] + 1):
                labels[column_index] = value or None
        return tuple(labels)
    labels = [None] * width
    for column_index in range(width):
        parts: list[str] = []
        for row_index in header_rows:
            grid_row = grid[row_index] if row_index < len(grid) else []
            for cell in grid_row:
                if cell is None:
                    continue
                cell_range = ranges.get(id(cell))
                if cell_range is None:
                    continue
                if not cell_range[2] <= column_index <= cell_range[3]:
                    continue
                value, _ = _display_value(cell, strip)
                if value and value not in parts:
                    parts.append(value)
                break
        if parts:
            labels[column_index] = ' '.join(parts)
    return tuple(labels)


def _unit_column(view: RecordView, column: int) -> int | None:
    """Return the adjacent unit column for a numeric sort, if any."""
    candidates = []
    if column + 1 < len(view.headers):
        candidates.append(column + 1)
    if column > 0:
        candidates.append(column - 1)
    for index in candidates:
        label = view.headers[index]
        if label is None:
            continue
        if _normalise_unit(label) in _UNIT_COLUMN_NAMES:
            return index
    return None


def _unit_spec(unit_text: str) -> tuple[str, Decimal]:
    """Return ``(family, factor)`` for a unit column value."""
    unit_text = unit_text.strip()
    if not unit_text:
        return '', Decimal(1)
    folded = _normalise_unit(unit_text)
    if folded in _PERCENT_UNITS:
        return '%', Decimal(1)
    if folded in _PERMILLE_UNITS:
        return '%', Decimal('0.1')
    scale = Decimal(1)
    index = 0
    while index < len(unit_text) and unit_text[index] in _CJK_SCALES:
        scale *= _CJK_SCALES[unit_text[index]]
        index += 1
    if index:
        family, factor = _canonical_unit(unit_text[index:])
        return family, scale * factor
    return _canonical_unit(unit_text)


def _combine_units(
    value_unit: str | None,
    value: Decimal | str | None,
    unit_text: str,
) -> tuple[str | None, Decimal | str | None] | None:
    """Combine a numeric value's own unit with a separate unit column."""
    if not isinstance(value, Decimal):
        return value_unit, value
    base = value_unit or ''
    if not unit_text.strip():
        return base, value
    family, factor = _unit_spec(unit_text)
    if not base:
        return family, value * factor
    if not family:
        return base, value * factor
    if base == family:
        return base, value
    if base == '%' and family in ('', '%'):
        return '%', value
    return None


def _text_sort_key(value: object) -> tuple[str, str]:
    text = str(value)
    return text.casefold(), text


def _sort_records(
    view: RecordView,
    column: int,
    reverse: bool,
) -> RecordSortResult:
    unit_column = _unit_column(view, column)
    entries: list[_SortEntry] = []
    unsortable: list[UnsortableRecord] = []
    for record in view.records:
        candidates = [
            value
            for value in record.distinct_values(column)
            if value.display.strip() or value.sort is not None
        ]
        if not candidates:
            unsortable.append(
                UnsortableRecord(
                    record,
                    column,
                    'missing',
                    'the record has no value in this column',
                    None,
                )
            )
            continue
        value = next(
            (
                candidate
                for candidate in candidates
                if candidate.sort_issue is None
            ),
            candidates[0],
        )
        if value.sort_issue is not None:
            unsortable.append(
                UnsortableRecord(
                    record,
                    column,
                    value.sort_issue,
                    value.sort_detail or 'no usable sort key',
                    value,
                )
            )
            continue
        sort_value = value.sort
        unit = value.unit
        unit_text = value.unit_text
        if unit_column is not None and value.sort_kind == 'number':
            unit_value = record.value(unit_column)
            text = '' if unit_value is None else unit_value.display
            combined = _combine_units(value.unit, value.sort, text)
            if combined is None:
                unsortable.append(
                    UnsortableRecord(
                        record,
                        column,
                        'unit',
                        f'unit {text!r} is incompatible with '
                        f'{value.unit!r}',
                        value,
                    )
                )
                continue
            unit, sort_value = combined
            if text:
                unit_text = text
        entries.append(
            _SortEntry(
                record,
                value,
                sort=sort_value,
                unit=unit,
                unit_text=unit_text,
            )
        )

    groups: dict[str, list[_SortEntry]] = {}
    for entry in entries:
        groups.setdefault(entry.value.sort_kind or '', []).append(entry)
    kind: str | None = None
    chosen: list[_SortEntry] = []
    if groups:
        first_seen: dict[str, int] = {}
        for index, entry in enumerate(entries):
            first_seen.setdefault(entry.value.sort_kind or '', index)
        kind = max(
            groups,
            key=lambda key: (len(groups[key]), -first_seen[key]),
        )
        chosen = groups[kind]
    rejected = [
        entry for entry in entries if entry.value.sort_kind != kind
    ]

    for entry in rejected:
        other = entry.value.sort_kind or 'unknown'
        unsortable.append(
            UnsortableRecord(
                entry.record,
                column,
                'type',
                f'{other} sort key {entry.value.display!r} cannot be '
                f'compared with {kind} keys',
                entry.value,
            )
        )

    unit: str | None = None
    if kind == 'number' and chosen:
        known = [
            entry
            for entry in chosen
            if not (entry.unit or '').startswith('unit:')
        ]
        pool = known or chosen
        counts = Counter(entry.unit or '' for entry in pool)
        order: dict[str, int] = {}
        for entry in pool:
            order.setdefault(entry.unit or '', len(order))
        unit = max(order, key=lambda key: counts[key])
        compatible: list[_SortEntry] = []
        for entry in chosen:
            entry_unit = entry.unit or ''
            if entry_unit == unit:
                compatible.append(entry)
            else:
                unsortable.append(
                    UnsortableRecord(
                        entry.record,
                        column,
                        'unit',
                        f'unit {entry_unit!r} is incompatible with '
                        f'{unit!r}',
                        entry.value,
                    )
                )
        chosen = compatible
    elif kind == 'text':
        unit = None

    if kind == 'number':
        chosen.sort(key=lambda entry: entry.sort, reverse=reverse)
    elif kind == 'date':
        chosen.sort(key=lambda entry: entry.sort, reverse=reverse)
    elif kind == 'text':
        chosen.sort(
            key=lambda entry: _text_sort_key(entry.value.sort),
            reverse=reverse,
        )
    elif kind == 'list':
        chosen.sort(
            key=lambda entry: tuple(
                part.casefold() for part in entry.value.sort
            ),
            reverse=reverse,
        )

    order_index = {
        id(record): index for index, record in enumerate(view.records)
    }
    unsortable.sort(key=lambda entry: order_index[id(entry.record)])
    return RecordSortResult(
        view=view,
        column=column,
        reverse=reverse,
        records=tuple(entry.record for entry in chosen),
        unsortable=tuple(unsortable),
        kind=kind,
        unit=unit,
    )
