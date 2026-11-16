"""Incremental export of table cells with syntactic dependencies.

An offline page import needs to keep the exported values of table cells
without re-evaluating every cell on each revision.  A :class:`PageExport`
stores, for every cell:

* the exported value,
* the raw cell markup,
* the template, parser-function and parameter arguments that syntactically
  occur in the cell or in an enclosing call argument.

Dependencies are purely syntactic.  Literal text that merely looks like
template syntax is not a dependency.  When a new revision arrives,
:meth:`PageExport.apply_revision` reuses values for cells whose markup and
dependencies are unchanged and only re-extracts the remaining cells.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from ._cell import Cell
from ._parameter import Parameter
from ._parser_function import ParserFunction
from ._template import Template
from ._wikitext import WS, WikiText

__all__ = [
    'CellChange',
    'Dependency',
    'DependencyChange',
    'Field',
    'PageExport',
    'RevisionResult',
    'TableExport',
    'apply_revision',
    'default_extractor',
    'dependencies',
    'diff_exports',
    'export',
    'export_fields',
    'export_page',
    'export_tables',
    'field_dependencies',
    'incremental_export',
    'load_export',
    'page_export',
    'save_export',
]

_TEMPLATE = 'template'
_PARSER_FUNCTION = 'parser_function'
_PARAMETER = 'parameter'
_CELL_SCOPE = 'cell'
_ANCESTOR_SCOPE = 'ancestor'
_FORMAT_VERSION = 1

Address = tuple[int, int, int]
PathElement = tuple[str, str, int]
CallPath = tuple[PathElement, ...]

DEFAULT_EXTRACTOR_NAME = 'default_extractor'


def default_extractor(cell: Cell) -> str | None:
    """Return the exported value of a cell.

    This mirrors :meth:`Table.data` with ``strip=True`` and ``span=False``
    applied to the physical cell: leading spaces are removed and trailing
    whitespace is stripped.
    """
    if cell is None:
        return None
    return cell.value.lstrip(' ').rstrip(WS)


def _callable_name(function: Callable) -> str:
    return getattr(function, '__name__', type(function).__name__)


def _kind_and_name(
    node: Template | ParserFunction | Parameter,
) -> tuple[str, str]:
    if isinstance(node, Template):
        return _TEMPLATE, node.normal_name()
    if isinstance(node, Parameter):
        return _PARAMETER, node.name.strip(WS)
    return _PARSER_FUNCTION, node.name.strip(WS)


@dataclass(frozen=True)
class Dependency:
    """A syntactic template/parser-function argument used by a field.

    ``path`` identifies the call relative to the field: for ``scope='cell'``
    it is relative to the cell, and for ``scope='ancestor'`` it is the chain
    of enclosing calls.  Every element is ``(kind, name, ordinal)``.  The
    path makes a parameter that moved into a nested template
    distinguishable from an unrelated parameter.
    """

    kind: str
    name: str
    path: CallPath
    argument: str
    argument_occurrence: int
    value: str
    scope: str = _CELL_SCOPE

    @property
    def key(self) -> tuple[str, str, CallPath, str, int, str]:
        """Identity of the dependency, excluding its value."""
        return (
            self.kind,
            self.name,
            self.path,
            self.argument,
            self.argument_occurrence,
            self.scope,
        )

    @property
    def call(self) -> str:
        """Return a readable representation of the call path."""
        parts = []
        for kind, name, _ordinal in self.path:
            if kind == _PARAMETER:
                parts.append(f'{{{{{{{name}}}}}}}')
            elif kind == _PARSER_FUNCTION:
                parts.append(f'{{{{{name}:...}}}}')
            else:
                parts.append(f'{{{{{name}}}}}')
        return ' > '.join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            'kind': self.kind,
            'name': self.name,
            'path': [list(element) for element in self.path],
            'argument': self.argument,
            'argument_occurrence': self.argument_occurrence,
            'value': self.value,
            'scope': self.scope,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Dependency:
        return cls(
            kind=data['kind'],
            name=data['name'],
            path=tuple(tuple(element) for element in data['path']),
            argument=data['argument'],
            argument_occurrence=int(data.get('argument_occurrence', 0)),
            value=data.get('value', ''),
            scope=data.get('scope', _CELL_SCOPE),
        )


@dataclass(frozen=True)
class Field:
    """Exported value of one table cell and its syntactic dependencies."""

    table: int
    row: int
    column: int
    value: Any
    markup: str | None
    dependencies: tuple[Dependency, ...] = ()

    @property
    def address(self) -> Address:
        return self.table, self.row, self.column

    def dependency(
        self, argument: str, *, scope: str | None = None
    ) -> tuple[Dependency, ...]:
        """Return dependencies with the given argument name."""
        return tuple(
            dependency
            for dependency in self.dependencies
            if dependency.argument == argument
            and (scope is None or dependency.scope == scope)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            'row': self.row,
            'column': self.column,
            'value': self.value,
            'markup': self.markup,
            'dependencies': [
                dependency.to_dict() for dependency in self.dependencies
            ],
        }

    @classmethod
    def from_dict(cls, table: int, data: dict[str, Any]) -> Field:
        return cls(
            table=table,
            row=int(data['row']),
            column=int(data['column']),
            value=data.get('value'),
            markup=data.get('markup'),
            dependencies=tuple(
                Dependency.from_dict(dependency)
                for dependency in data.get('dependencies', ())
            ),
        )


@dataclass(frozen=True)
class TableExport:
    """All exported fields of one table, in row-major order."""

    index: int
    row_count: int
    column_count: int
    fields: tuple[Field, ...]
    caption: str | None = None
    key: str | None = None

    @property
    def rows(self) -> list[list[Field]]:
        rows: list[list[Field]] = []
        for row_index in range(self.row_count):
            row = sorted(
                (
                    field
                    for field in self.fields
                    if field.row == row_index
                ),
                key=lambda field: field.column,
            )
            rows.append(row)
        return rows

    @property
    def values(self) -> list[list[Any]]:
        return [[field.value for field in row] for row in self.rows]

    def field(self, row: int, column: int) -> Field | None:
        for field in self.fields:
            if field.row == row and field.column == column:
                return field
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            'index': self.index,
            'row_count': self.row_count,
            'column_count': self.column_count,
            'caption': self.caption,
            'key': self.key,
            'fields': [field.to_dict() for field in self.fields],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TableExport:
        index = int(data['index'])
        return cls(
            index=index,
            row_count=int(data['row_count']),
            column_count=int(data['column_count']),
            fields=tuple(
                Field.from_dict(index, field)
                for field in data.get('fields', ())
            ),
            caption=data.get('caption'),
            key=data.get('key'),
        )


@dataclass(frozen=True)
class _ArgumentInfo:
    name: str
    occurrence: int
    value: str
    start: int
    end: int


@dataclass(frozen=True)
class _CallInfo:
    start: int
    end: int
    kind: str
    name: str
    arguments: tuple[_ArgumentInfo, ...]
    path: CallPath = ()


def _make_call_info(
    node: Template | ParserFunction | Parameter,
) -> _CallInfo:
    span = node._span_data
    kind, name = _kind_and_name(node)
    counters: dict[str, int] = {}
    arguments: list[_ArgumentInfo] = []
    if isinstance(node, Parameter):
        arguments.append(
            _ArgumentInfo(
                name='name',
                occurrence=0,
                value=name,
                start=span.start,
                end=span.start + 3 + len(name),
            )
        )
        default = node.default
        if default is not None:
            pipe = node._shadow.find(124)
            arguments.append(
                _ArgumentInfo(
                    name='default',
                    occurrence=0,
                    value=default,
                    start=span.start + pipe,
                    end=span.end - 3,
                )
            )
    else:
        for argument in node.arguments:
            argument_name = argument.name.strip(WS)
            occurrence = counters.get(argument_name, 0)
            counters[argument_name] = occurrence + 1
            argument_span = argument._span_data
            arguments.append(
                _ArgumentInfo(
                    name=argument_name,
                    occurrence=occurrence,
                    value=argument.value,
                    start=argument_span.start,
                    end=argument_span.end,
                )
            )
    return _CallInfo(
        start=span.start,
        end=span.end,
        kind=kind,
        name=name,
        path=(),
        arguments=tuple(arguments),
    )


def _collect_calls(page: WikiText) -> list[_CallInfo]:
    nodes: list[Template | ParserFunction | Parameter] = [
        *page.templates,
        *page.parser_functions,
        *page.parameters,
    ]
    calls = [_make_call_info(node) for node in nodes]
    calls.sort(key=lambda call: (call.start, -call.end))
    return calls


def dependencies(cell: Cell) -> tuple[Dependency, ...]:
    """Return the syntactic dependencies of a cell.

    The cell may belong to a parsed page; enclosing templates, parser
    functions and parameters are taken into account.
    """
    nodes: list[Any] = [
        *cell.templates,
        *cell.parser_functions,
        *cell.parameters,
        *(
            ancestor
            for ancestor in cell.ancestors()
            if isinstance(ancestor, (Template, ParserFunction, Parameter))
        ),
    ]
    calls = [_make_call_info(node) for node in nodes]
    calls.sort(key=lambda call: (call.start, -call.end))
    span = cell._span_data
    return _cell_dependencies(span.start, span.end, calls)


def _paths_for_calls(
    calls: list[_CallInfo],
) -> dict[tuple[int, int, str], CallPath]:
    """Return cell-relative paths for calls sorted by position."""
    paths: dict[tuple[int, int, str], CallPath] = {}
    counters: dict[tuple, int] = {}
    stack: list[tuple[int, int, str, str, CallPath]] = []
    for call in calls:
        while stack and not (
            stack[-1][0] <= call.start and call.end <= stack[-1][1]
        ):
            stack.pop()
        if stack:
            parent = stack[-1]
            counter_key: tuple = (parent[0], parent[1], call.kind, call.name)
            parent_path = parent[4]
        else:
            counter_key = ('root', call.kind, call.name)
            parent_path = ()
        ordinal = counters.get(counter_key, 0)
        counters[counter_key] = ordinal + 1
        path = (*parent_path, (call.kind, call.name, ordinal))
        paths[(call.start, call.end, call.kind)] = path
        stack.append((call.start, call.end, call.kind, call.name, path))
    return paths


def _make_dependency(
    call: _CallInfo, argument: _ArgumentInfo, scope: str
) -> Dependency:    return Dependency(
        kind=call.kind,
        name=call.name,
        path=call.path,
        argument=argument.name,
        argument_occurrence=argument.occurrence,
        value=argument.value,
        scope=scope,
    )


def _cell_dependencies(
    cell_start: int,
    cell_end: int,
    calls: list[_CallInfo],
) -> tuple[Dependency, ...]:
    dependencies: list[Dependency] = []

    # Calls inside the cell.  Paths are relative to the cell, so an
    # unrelated call earlier on the page cannot shift them.
    local_calls = [
        call
        for call in calls
        if cell_start <= call.start and call.end <= cell_end
    ]
    local_paths = _paths_for_calls(local_calls)
    for call in local_calls:
        path = local_paths[(call.start, call.end, call.kind)]
        for argument in call.arguments:
            dependencies.append(
                _make_dependency(
                    replace(call, path=path), argument, _CELL_SCOPE
                )
            )

    # Calls that enclose the cell.  Their nesting chain is unique, so the
    # path is just the chain itself.  Only arguments whose value contains
    # the cell are dependencies; literal text in another argument of the
    # same call cannot affect this cell.
    enclosing = [
        call
        for call in calls
        if call.start < cell_start and cell_end < call.end
    ]
    enclosing.sort(key=lambda call: (call.start, -call.end))
    chain: list[_CallInfo] = []
    for call in enclosing:
        if not chain or (
            chain[-1].start <= call.start and call.end <= chain[-1].end
        ):
            chain.append(call)
        else:
            chain = [call]
    for depth, call in enumerate(chain, 1):
        path = tuple(
            (ancestor.kind, ancestor.name, 0)
            for ancestor in chain[:depth]
        )
        for argument in call.arguments:
            if argument.start <= cell_start and cell_end <= argument.end:
                dependencies.append(
                    _make_dependency(
                        replace(call, path=path),
                        argument,
                        _ANCESTOR_SCOPE,
                    )
                )

    dependencies.sort(
        key=lambda dependency: (
            dependency.scope,
            dependency.path,
            dependency.kind,
            dependency.name,
            dependency.argument,
            dependency.argument_occurrence,
        )
    )
    return tuple(dependencies)


class PageExport:
    """A JSON-serializable snapshot of exported table cells.

    Use :func:`export_page` to create an export, :meth:`dumps`/``loads`` (or
    :meth:`save`/``load``) to persist it, and :meth:`apply_revision` to
    update it incrementally.
    """

    __slots__ = (
        '_extractor',
        '_extractor_known',
        '_extractor_name',
        'span',
        'tables',
    )

    def __init__(
        self,
        tables: Iterable[TableExport],
        *,
        span: bool = True,
        extractor: Callable[[Cell], Any] | None = None,
        _extractor_known: bool = True,
    ) -> None:
        self.tables = tuple(tables)
        self.span = span
        self._extractor = extractor or default_extractor
        self._extractor_name = _callable_name(self._extractor)
        self._extractor_known = _extractor_known

    @property
    def extractor_name(self) -> str:
        return self._extractor_name

    @property
    def fields(self) -> tuple[Field, ...]:
        """Return every field in table, row, column order."""
        return tuple(
            field for table in self.tables for field in table.fields
        )

    @property
    def values(self) -> list[list[list[Any]]]:
        """Return exported values grouped by table.

        With the default ``extractor`` and ``span`` this equals
        ``[table.data() for table in wikitext.tables]``.
        """
        return [table.values for table in self.tables]

    def table(self, index: int) -> TableExport:
        return self.tables[index]

    def field(self, table: int, row: int, column: int) -> Field | None:
        return self.tables[table].field(row, column)

    def to_dict(self) -> dict[str, Any]:
        return {
            'version': _FORMAT_VERSION,
            'span': self.span,
            'extractor': self._extractor_name,
            'tables': [table.to_dict() for table in self.tables],
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        extractor: Callable[[Cell], Any] | None = None,
    ) -> PageExport:
        version = int(data.get('version', _FORMAT_VERSION))
        if version != _FORMAT_VERSION:
            raise ValueError(f'unsupported export format version: {version}')
        stored_name = data.get('extractor', DEFAULT_EXTRACTOR_NAME)
        if extractor is None:
            extractor = default_extractor
            known = stored_name == DEFAULT_EXTRACTOR_NAME
        else:
            known = True
        return cls(
            (TableExport.from_dict(table) for table in data.get('tables', ())),
            span=bool(data.get('span', True)),
            extractor=extractor,
            _extractor_known=known,
        )

    def dumps(self, *, indent: int | None = 2) -> str:
        """Return the export as JSON."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )

    @classmethod
    def loads(
        cls,
        string: str,
        *,
        extractor: Callable[[Cell], Any] | None = None,
    ) -> PageExport:
        """Load an export from JSON produced by :meth:`dumps`."""
        return cls.from_dict(json.loads(string), extractor=extractor)

    def save(self, path: str | Path, *, indent: int | None = 2) -> None:
        """Write the export to a UTF-8 JSON file."""
        Path(path).write_text(self.dumps(indent=indent), encoding='utf-8')

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        extractor: Callable[[Cell], Any] | None = None,
    ) -> PageExport:
        """Load an export from a JSON file."""
        return cls.loads(
            Path(path).read_text(encoding='utf-8'), extractor=extractor
        )

    def apply_revision(
        self,
        new_wikitext: str | WikiText,
        *,
        extractor: Callable[[Cell], Any] | None = None,
        verify: bool = False,
    ) -> RevisionResult:
        """Apply a new revision and reuse unaffected cell values.

        A cell is re-extracted when its raw markup changed or when any of its
        syntactic dependencies was added, removed, moved, or changed.  All
        other values are copied from this snapshot.

        Tables are aligned across revisions, so inserting or removing a
        whole table does not shift the addresses of later tables.

        :param extractor: must be the same pure function used to create the
            snapshot.  If omitted, the stored one is used.
        :param verify: if true, compare the result with a full
            :func:`export_page` and raise ``AssertionError`` on mismatch.
        """
        if extractor is None:
            if not self._extractor_known:
                raise ValueError(
                    'this snapshot was created with a custom extractor; '
                    'pass the same extractor to apply_revision'
                )
            extractor = self._extractor

        page = _as_page(new_wikitext)
        new_keys = [_table_key(table) for table in page.tables]
        table_map = _align_tables(
            [table.key for table in self.tables], new_keys
        )

        if extractor is not self._extractor:
            # Values computed by a different extractor cannot be reused.
            reuse: dict[Address, Field] = {}
        else:
            reuse = _reuse_fields(self.tables, table_map)

        new_export, recomputed, reused = _collect_page(
            page,
            extractor=extractor,
            span=self.span,
            reuse=reuse,
        )
        changes, removed = diff_exports(
            self, new_export, table_map=table_map
        )
        result = RevisionResult(
            export=new_export,
            changes=changes,
            recomputed=recomputed,
            reused=reused,
            removed=removed,
            table_map=table_map,
        )
        if verify:
            result.assert_matches_full(page)
        return result

    revision = apply_revision

    def __len__(self) -> int:
        return len(self.tables)

    def __repr__(self) -> str:
        return (
            f'PageExport(tables={len(self.tables)}, '
            f'fields={len(self.fields)}, span={self.span})'
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PageExport):
            return NotImplemented
        return self.to_dict() == other.to_dict()


@dataclass(frozen=True)
class DependencyChange:
    """A dependency added, removed, moved, or changed between revisions."""

    action: str
    old: Dependency | None = None
    new: Dependency | None = None

    def describe(self) -> str:
        if self.action == 'added':
            assert self.new is not None
            return (
                f'added {self.new.call} argument '
                f'{self.new.argument!r} = {self.new.value!r}'
            )
        if self.action == 'removed':
            assert self.old is not None
            return (
                f'removed {self.old.call} argument '
                f'{self.old.argument!r} = {self.old.value!r}'
            )
        if self.action == 'changed':
            assert self.old is not None and self.new is not None
            return (
                f'changed {self.new.call} argument '
                f'{self.new.argument!r}: {self.old.value!r} -> '
                f'{self.new.value!r}'
            )
        if self.action == 'moved':
            assert self.old is not None and self.new is not None
            return (
                f'moved {self.old.call} argument {self.old.argument!r} '
                f'to {self.new.call}'
            )
        if self.action == 'moved_and_changed':
            assert self.old is not None and self.new is not None
            return (
                f'moved {self.old.call} argument {self.old.argument!r} '
                f'to {self.new.call} and changed it: '
                f'{self.old.value!r} -> {self.new.value!r}'
            )
        return self.action

    def to_dict(self) -> dict[str, Any]:
        return {
            'action': self.action,
            'old': self.old.to_dict() if self.old is not None else None,
            'new': self.new.to_dict() if self.new is not None else None,
        }


@dataclass(frozen=True)
class CellChange:
    """A table cell whose exported value changed between revisions."""

    table: int
    row: int
    column: int
    old_value: Any = None
    new_value: Any = None
    old_markup: str | None = None
    new_markup: str | None = None
    dependency_changes: tuple[DependencyChange, ...] = ()
    source_changed: bool = False

    @property
    def address(self) -> Address:
        return self.table, self.row, self.column

    def describe(self) -> str:
        if self.old_markup is None and self.new_markup is not None:
            head = f'added with value {self.new_value!r}'
        elif self.new_markup is None and self.old_markup is not None:
            head = f'removed (was {self.old_value!r})'
        else:
            head = f'{self.old_value!r} -> {self.new_value!r}'

        reasons = [
            change.describe() for change in self.dependency_changes
        ]
        if not reasons and self.source_changed:
            reasons.append(
                'literal cell markup changed outside tracked template '
                'arguments'
            )
        elif reasons and self.source_changed:
            reasons.append('cell markup also changed')
        if not reasons:
            reasons.append('exported value changed')
        return head + '; ' + '; '.join(reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            'table': self.table,
            'row': self.row,
            'column': self.column,
            'old_value': self.old_value,
            'new_value': self.new_value,
            'old_markup': self.old_markup,
            'new_markup': self.new_markup,
            'source_changed': self.source_changed,
            'dependency_changes': [
                change.to_dict() for change in self.dependency_changes
            ],
        }


@dataclass(frozen=True)
class RevisionResult:
    """Result of :meth:`PageExport.apply_revision`."""

    export: PageExport
    changes: tuple[CellChange, ...]
    recomputed: tuple[Address, ...]
    reused: tuple[Address, ...]
    removed: tuple[Address, ...] = ()
    table_map: dict[int, int] | None = None

    def explain(self) -> str:
        """Return a short human-readable explanation of the changes."""
        if not self.changes:
            summary = 'No exported values changed.'
        else:
            summary = (
                f'{len(self.changes)} exported cell(s) changed; '
                f'{len(self.recomputed)} recomputed, '
                f'{len(self.reused)} reused.'
            )
        lines = [summary]
        for change in self.changes:
            lines.append(
                f'- table {change.table} cell '
                f'({change.row}, {change.column}): {change.describe()}'
            )
        if self.removed:
            lines.append(
                f'{len(self.removed)} cell(s) were removed from the tables.'
            )
        return '\n'.join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            'export': self.export.to_dict(),
            'changes': [change.to_dict() for change in self.changes],
            'recomputed': [list(address) for address in self.recomputed],
            'reused': [list(address) for address in self.reused],
            'removed': [list(address) for address in self.removed],
            'table_map': (
                {str(old): new for old, new in self.table_map.items()}
                if self.table_map
                else None
            ),
        }

    def matches_full(self, new_wikitext: str | WikiText) -> bool:
        """Return True if the incremental export equals a full extraction."""
        full = export_page(
            new_wikitext,
            extractor=self.export._extractor,
            span=self.export.span,
        )
        return self.export.to_dict() == full.to_dict()

    def assert_matches_full(self, new_wikitext: str | WikiText) -> None:
        """Raise AssertionError if the result differs from full extraction."""
        if not self.matches_full(new_wikitext):
            raise AssertionError(
                'incremental export does not match full extraction:\n'
                + self.explain()
            )


def _as_page(wikitext: str | WikiText) -> WikiText:
    return wikitext if isinstance(wikitext, WikiText) else WikiText(str(wikitext))


def _table_key(table) -> str:
    """Return a stable identity key for a table.

    The key is derived from the leading part of the table markup so that a
    table keeps its identity when cells later in the table change.
    """
    markup = table.string
    # Include the leading header/caption block, which usually identifies the
    # table, but not the cell bodies that revisions are likely to touch.
    head_end = markup.find('\n|-')
    if head_end == -1:
        head_end = markup.find('\n|')
    head = markup if head_end == -1 else markup[:head_end]
    return hashlib.sha1(head.encode('utf-8')).hexdigest()


def _align_tables(
    old_keys: list[str | None], new_keys: list[str]
) -> dict[int, int]:
    """Map old table indices to new table indices.

    When the number of tables is unchanged, tables are aligned by position:
    an edited table keeps its identity.  When tables were inserted or
    removed, a sequence alignment on the stable keys locates the inserted or
    deleted tables without shifting the rest.
    """
    old_count = len(old_keys)
    new_count = len(new_keys)
    if old_count == new_count:
        return {index: index for index in range(old_count)}

    mapping: dict[int, int] = {}
    matcher = SequenceMatcher(
        None,
        [key if key is not None else f'?{index}' for index, key in enumerate(old_keys)],
        new_keys,
        autojunk=False,
    )
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for offset in range(i2 - i1):
                mapping[i1 + offset] = j1 + offset
        elif tag == 'replace':
            for offset in range(min(i2 - i1, j2 - j1)):
                mapping[i1 + offset] = j1 + offset
    return mapping


def _reuse_fields(
    tables: tuple[TableExport, ...], table_map: dict[int, int]
) -> dict[Address, Field]:
    """Return old fields re-addressed for the new table indices."""
    reuse: dict[Address, Field] = {}
    for table in tables:
        new_table_index = table_map.get(table.index)
        if new_table_index is None:
            continue
        for field in table.fields:
            reuse[(new_table_index, field.row, field.column)] = field
    return reuse


def _collect_page(
    page: WikiText,
    *,
    extractor: Callable[[Cell], Any],
    span: bool,
    reuse: dict[Address, Field],
) -> tuple[PageExport, tuple[Address, ...], tuple[Address, ...]]:
    calls = _collect_calls(page)

    table_exports: list[TableExport] = []
    recomputed: list[Address] = []
    reused: list[Address] = []
    for table_index, table in enumerate(page.tables):
        grid = table.cells(span=span)
        fields: list[Field] = []
        column_count = 0
        for row_index, row in enumerate(grid):
            if len(row) > column_count:
                column_count = len(row)
            for column_index, cell in enumerate(row):
                address = (table_index, row_index, column_index)
                if cell is None:
                    fields.append(
                        Field(
                            table=table_index,
                            row=row_index,
                            column=column_index,
                            value=None,
                            markup=None,
                        )
                    )
                    continue
                cell_span = cell._span_data
                markup = cell.string
                dependencies = _cell_dependencies(
                    cell_span.start, cell_span.end, calls
                )
                previous = reuse.get(address)
                if (
                    previous is not None
                    and previous.markup == markup
                    and previous.dependencies == dependencies
                ):
                    value = previous.value
                    reused.append(address)
                else:
                    value = extractor(cell)
                    recomputed.append(address)
                fields.append(
                    Field(
                        table=table_index,
                        row=row_index,
                        column=column_index,
                        value=value,
                        markup=markup,
                        dependencies=dependencies,
                    )
                )
        table_exports.append(
            TableExport(
                index=table_index,
                row_count=len(grid),
                column_count=column_count,
                fields=tuple(fields),
                caption=table.caption,
                key=_table_key(table),
            )
        )
    return (
        PageExport(table_exports, span=span, extractor=extractor),
        tuple(recomputed),
        tuple(reused),
    )


def export_page(
    wikitext: str | WikiText,
    *,
    extractor: Callable[[Cell], Any] | None = None,
    span: bool = True,
) -> PageExport:
    """Export all table cells of ``wikitext`` with their dependencies.

    :param extractor: a pure function of a :class:`Cell` used to compute the
        exported value.  The default mirrors :meth:`Table.data`.
    :param span: passed to :meth:`Table.cells`.  With the default ``True``,
        ``export.values`` equals ``[table.data() for table in tables]``.
    """
    if extractor is None:
        extractor = default_extractor
    export_, _recomputed, _reused = _collect_page(
        _as_page(wikitext),
        extractor=extractor,
        span=span,
        reuse={},
    )
    return export_


export_tables = export_page
export = export_page
export_fields = export_page
page_export = export_page
field_dependencies = dependencies


def save_export(
    export_: PageExport,
    path: str | Path,
    *,
    indent: int | None = 2,
) -> None:
    """Save a :class:`PageExport` as UTF-8 JSON."""
    export_.save(path, indent=indent)


def load_export(
    path: str | Path,
    *,
    extractor: Callable[[Cell], Any] | None = None,
) -> PageExport:
    """Load a :class:`PageExport` from a UTF-8 JSON file."""
    return PageExport.load(path, extractor=extractor)


def _dependency_changes(
    old: Field, new: Field
) -> tuple[DependencyChange, ...]:
    old_by_key = {
        dependency.key: dependency for dependency in old.dependencies
    }
    new_by_key = {
        dependency.key: dependency for dependency in new.dependencies
    }
    changes: list[DependencyChange] = []
    for key, old_dependency in old_by_key.items():
        if key in new_by_key:
            new_dependency = new_by_key[key]
            if old_dependency.value != new_dependency.value:
                changes.append(
                    DependencyChange(
                        'changed', old=old_dependency, new=new_dependency
                    )
                )

    removed = [
        dependency
        for key, dependency in old_by_key.items()
        if key not in new_by_key
    ]
    added = [
        dependency
        for key, dependency in new_by_key.items()
        if key not in old_by_key
    ]

    # Pair a removed dependency with an added one when the same argument
    # (same name and occurrence) moved to another call.  Prefer pairs with
    # equal values so an unrelated rename does not hide a value change.
    candidates = []
    for removed_index, old_dependency in enumerate(removed):
        for added_index, new_dependency in enumerate(added):
            if (
                old_dependency.argument != new_dependency.argument
                or old_dependency.argument_occurrence
                != new_dependency.argument_occurrence
            ):
                continue
            score = 0
            if old_dependency.value == new_dependency.value:
                score += 8
            if old_dependency.name == new_dependency.name:
                score += 4
            if old_dependency.kind == new_dependency.kind:
                score += 2
            if old_dependency.scope == new_dependency.scope:
                score += 1
            candidates.append(
                (score, removed_index, added_index)
            )
    candidates.sort(key=lambda candidate: (-candidate[0], candidate[1], candidate[2]))
    paired_removed: set[int] = set()
    paired_added: set[int] = set()
    for _score, removed_index, added_index in candidates:
        if (
            removed_index in paired_removed
            or added_index in paired_added
        ):
            continue
        paired_removed.add(removed_index)
        paired_added.add(added_index)
        old_dependency = removed[removed_index]
        new_dependency = added[added_index]
        action = (
            'moved'
            if old_dependency.value == new_dependency.value
            else 'moved_and_changed'
        )
        changes.append(
            DependencyChange(action, old=old_dependency, new=new_dependency)
        )

    for index, dependency in enumerate(removed):
        if index not in paired_removed:
            changes.append(DependencyChange('removed', old=dependency))
    for index, dependency in enumerate(added):
        if index not in paired_added:
            changes.append(DependencyChange('added', new=dependency))

    changes.sort(
        key=lambda change: (
            (change.old or change.new).kind,
            (change.old or change.new).name,
            (change.old or change.new).argument,
            (change.old or change.new).argument_occurrence,
            change.action,
        )
    )
    return tuple(changes)


def diff_exports(
    old: PageExport,
    new: PageExport,
    *,
    table_map: dict[int, int] | None = None,
) -> tuple[tuple[CellChange, ...], tuple[Address, ...]]:
    """Compare two exports.

    Return ``(changes, removed_addresses)`` where ``changes`` contains only
    cells whose exported value changed.  ``table_map`` maps old table indices
    to new ones for revisions that inserted or removed whole tables.
    """
    if table_map:
        old_fields = _reuse_fields(old.tables, table_map)
    else:
        old_fields = {field.address: field for field in old.fields}
    new_fields = {field.address: field for field in new.fields}
    changes: list[CellChange] = []
    removed: list[Address] = []
    for address in sorted(set(old_fields) | set(new_fields)):
        old_field = old_fields.get(address)
        new_field = new_fields.get(address)
        if old_field is not None:
            # Keep the public address in the new revision's numbering.
            old_field = replace(old_field, table=address[0])
        if old_field is None:
            assert new_field is not None
            changes.append(
                CellChange(
                    table=address[0],
                    row=address[1],
                    column=address[2],
                    new_value=new_field.value,
                    new_markup=new_field.markup,
                    dependency_changes=tuple(
                        DependencyChange('added', new=dependency)
                        for dependency in new_field.dependencies
                    ),
                )
            )
        elif new_field is None:
            removed.append(address)
            changes.append(
                CellChange(
                    table=address[0],
                    row=address[1],
                    column=address[2],
                    old_value=old_field.value,
                    old_markup=old_field.markup,
                    dependency_changes=tuple(
                        DependencyChange('removed', old=dependency)
                        for dependency in old_field.dependencies
                    ),
                )
            )
        elif old_field.value != new_field.value:
            changes.append(
                CellChange(
                    table=address[0],
                    row=address[1],
                    column=address[2],
                    old_value=old_field.value,
                    new_value=new_field.value,
                    old_markup=old_field.markup,
                    new_markup=new_field.markup,
                    dependency_changes=_dependency_changes(
                        old_field, new_field
                    ),
                    source_changed=old_field.markup != new_field.markup,
                )
            )
    return tuple(changes), tuple(removed)


def apply_revision(
    old: PageExport | str | WikiText,
    new_wikitext: str | WikiText,
    *,
    extractor: Callable[[Cell], Any] | None = None,
    span: bool = True,
    verify: bool = False,
) -> RevisionResult:
    """Apply ``new_wikitext`` to an existing export or old wikitext."""
    if not isinstance(old, PageExport):
        old = export_page(old, extractor=extractor, span=span)
    return old.apply_revision(
        new_wikitext, extractor=extractor, verify=verify
    )


incremental_export = apply_revision
