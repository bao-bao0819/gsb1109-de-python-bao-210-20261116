"""Import template arguments while keeping every duplicate occurrence.

MediaWiki does not collapse duplicate template parameters: every occurrence is
parsed, and the last occurrence of a name supplies its value.  The same is true
for an explicit numeric parameter such as ``|1=a`` and the first positional
parameter ``|b`` -- both address parameter ``1``, but they are separate
occurrences and the later one wins.

This module exposes that model:

* :class:`ArgumentOccurrence` keeps every occurrence, its stable 1-based
  original index, and whether it was written positionally.
* :class:`EffectiveArgument` records which occurrence currently supplies a
  parameter value.
* :func:`import_arguments` applies a batch of delete/set/insert requests.  All
  request positions refer to the original occurrence list, regardless of the
  order in which the requests are supplied.
* :class:`ArgumentImportReport` reports effective value changes, newly
  revealed or hidden values, positional renumbering, and conflicts.

Whitespace rules follow MediaWiki and :class:`~wikitextparser.Argument`:
surrounding whitespace in a named argument's name is ignored, while a
positional argument's identity is its ordinal among positional arguments only.
Named values are trimmed; positional values keep their whitespace verbatim.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, overload

from ._wikitext import WS

if TYPE_CHECKING:  # pragma: no cover
    from ._argument import SubWikiTextWithArgs

__all__ = [
    'ArgumentConflict',
    'ArgumentDelete',
    'ArgumentImportReport',
    'ArgumentInsert',
    'ArgumentOccurrence',
    'ArgumentRequestConflict',
    'ArgumentRequestError',
    'ArgumentSet',
    'DeleteArgument',
    'EffectiveArgument',
    'EffectiveChange',
    'InsertArgument',
    'OccurrenceChange',
    'OccurrenceShift',
    'SetArgument',
    'apply_argument_changes',
    'import_arguments',
]

Where = Literal['before', 'after']
_WS = WS


def _require_index(index: Any, request: int) -> int:
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError(f'request {request}: occurrence index must be an int')
    if index < 0:
        raise ValueError(
            f'request {request}: occurrence index cannot be negative'
        )
    return index


def _require_str(value: Any, what: str, request: int) -> str:
    if not isinstance(value, str):
        raise TypeError(f'request {request}: {what} must be a string')
    return value


def _effective_value(raw_value: str, positional: bool) -> str:
    """Return the value MediaWiki uses for an occurrence."""
    return raw_value if positional else raw_value.strip(_WS)


@dataclass(frozen=True)
class ArgumentOccurrence:
    """One argument occurrence.

    ``index`` is the 1-based position in the list this record belongs to.
    For records in :attr:`ArgumentImportReport.occurrences_before` it is the
    stable original index used in requests.  ``origin`` is that original index
    when the occurrence survived an edit, and ``None`` for an insertion.

    ``name`` is the effective parameter name: for a named occurrence it is the
    raw name with surrounding whitespace removed; for a positional occurrence
    it is the 1-based ordinal among positional occurrences.  ``raw_name`` keeps
    the original spelling for named occurrences and is ``None`` for positional
    ones.  ``value`` is the effective value (named values trimmed, positional
    values verbatim), while ``raw_value`` keeps the exact value text.
    """

    index: int
    name: str
    value: str
    positional: bool
    source: str
    origin: int | None = None
    raw_name: str | None = None
    raw_value: str = ''
    separator: str = '|'

    @property
    def positional_index(self) -> int | None:
        """Return the ordinal for a positional occurrence, if any."""
        return int(self.name) if self.positional else None

    @property
    def identity(self) -> str:
        """Return the key MediaWiki uses to address this occurrence."""
        return self.name

    def to_dict(self) -> dict[str, Any]:
        return {
            'index': self.index,
            'origin': self.origin,
            'name': self.name,
            'value': self.value,
            'raw_value': self.raw_value,
            'positional': self.positional,
            'raw_name': self.raw_name,
            'source': self.source,
            'separator': self.separator,
        }


@dataclass(frozen=True)
class EffectiveArgument:
    """The occurrence that wins for a parameter name.

    ``occurrence`` is the 1-based index of the winning occurrence in the list
    this record belongs to.  ``origins`` lists the original indexes of all
    occurrences for this name, in order; the last entry is the winner's
    original index (or ``None`` when the winner was inserted).
    ``occurrences`` and ``overridden`` use list indexes and are provided for
    display.
    """

    name: str
    value: str
    positional: bool
    occurrence: int
    occurrences: tuple[int, ...]
    overridden: tuple[int, ...] = ()
    origins: tuple[int | None, ...] = ()
    raw_value: str = ''

    @property
    def winner_origin(self) -> int | None:
        """Return the winning occurrence's original index, if it survived."""
        return self.origins[-1] if self.origins else None

    def to_dict(self) -> dict[str, Any]:
        return {
            'name': self.name,
            'value': self.value,
            'raw_value': self.raw_value,
            'positional': self.positional,
            'occurrence': self.occurrence,
            'winner_origin': self.winner_origin,
            'occurrences': self.occurrences,
            'overridden': self.overridden,
            'origins': self.origins,
        }


@dataclass(frozen=True)
class ArgumentConflict:
    """A parameter name whose occurrences do not carry the same value."""

    name: str
    effective: int
    effective_value: str
    conflicting: tuple[int, ...]
    occurrences: tuple[int, ...]
    values: tuple[str, ...]

    @property
    def indexes(self) -> tuple[int, ...]:
        """Alias for ``occurrences``."""
        return self.occurrences

    def to_dict(self) -> dict[str, Any]:
        return {
            'name': self.name,
            'effective': self.effective,
            'effective_value': self.effective_value,
            'conflicting': self.conflicting,
            'occurrences': self.occurrences,
            'values': self.values,
        }


@dataclass(frozen=True)
class ArgumentDelete:
    """Delete the occurrence at original 1-based position ``index``."""

    index: int


@dataclass(frozen=True)
class ArgumentSet:
    """Modify the occurrence at original 1-based position ``index``.

    ``value`` and ``name`` are left unchanged when ``None``.  Supplying
    ``name`` to a positional occurrence converts it to a named occurrence.
    ``positional=True`` converts a named occurrence to positional form, but
    only when its resulting ordinal keeps the same parameter name; otherwise
    the request is rejected instead of silently changing what it addresses.
    """

    index: int
    value: str | None = None
    name: str | None = None
    positional: bool | None = None


@dataclass(frozen=True)
class ArgumentInsert:
    """Insert an occurrence relative to an original occurrence.

    ``index`` and ``where`` refer to the original list.  ``after=0`` means the
    beginning and ``after=N`` means the end, where ``N`` is the original
    occurrence count.  ``where='before'`` requires ``1 <= index <= N``.
    ``name=None`` inserts a positional occurrence; a string inserts a named
    occurrence.
    """

    index: int
    value: str = ''
    name: str | None = None
    where: Where = 'after'
    positional: bool | None = None


DeleteArgument = ArgumentDelete
SetArgument = ArgumentSet
InsertArgument = ArgumentInsert


@dataclass(frozen=True)
class ArgumentRequestError:
    """A request that could not be applied."""

    request: int
    change: Any
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            'request': self.request,
            'change': repr(self.change),
            'reason': self.reason,
        }


@dataclass(frozen=True)
class ArgumentRequestConflict:
    """Two or more requests that address the same original occurrence."""

    occurrence: int
    requests: tuple[int, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            'occurrence': self.occurrence,
            'requests': self.requests,
            'reason': self.reason,
        }


@dataclass(frozen=True)
class OccurrenceChange:
    """A change to one occurrence, or an insertion/deletion."""

    kind: str
    original: int | None
    after: int | None
    request: int | None = None
    before_name: str | None = None
    after_name: str | None = None
    before_value: str | None = None
    after_value: str | None = None
    before_source: str | None = None
    after_source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'kind': self.kind,
            'original': self.original,
            'after': self.after,
            'request': self.request,
            'before_name': self.before_name,
            'after_name': self.after_name,
            'before_value': self.before_value,
            'after_value': self.after_value,
            'before_source': self.before_source,
            'after_source': self.after_source,
        }


@dataclass(frozen=True)
class OccurrenceShift:
    """A surviving positional occurrence whose effective index changed."""

    original: int
    before_name: str
    after_name: str
    value: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            'original': self.original,
            'before_name': self.before_name,
            'after_name': self.after_name,
            'value': self.value,
            'source': self.source,
        }


@dataclass(frozen=True)
class EffectiveChange:
    """A change in the last-occurrence-wins parameter mapping.

    ``kind`` is one of ``added``, ``removed``, ``revealed``, ``updated``,
    ``winner``, ``form`` or ``occurrences``.  ``revealed`` means a later
    occurrence was removed (or otherwise stopped winning) and an earlier
    occurrence of the same name became effective.
    """

    name: str
    kind: str
    before: EffectiveArgument | None
    after: EffectiveArgument | None

    def to_dict(self) -> dict[str, Any]:
        return {
            'name': self.name,
            'kind': self.kind,
            'before': None if self.before is None else self.before.to_dict(),
            'after': None if self.after is None else self.after.to_dict(),
        }


@dataclass
class ArgumentImportReport:
    """Occurrence model plus the outcome of an edit batch."""

    source: SubWikiTextWithArgs
    string_before: str
    string_after: str
    occurrences_before: tuple[ArgumentOccurrence, ...]
    occurrences_after: tuple[ArgumentOccurrence, ...]
    effective_before: dict[str, EffectiveArgument]
    effective_after: dict[str, EffectiveArgument]
    conflicts_before: tuple[ArgumentConflict, ...]
    conflicts_after: tuple[ArgumentConflict, ...]
    effective_changes: tuple[EffectiveChange, ...]
    occurrence_changes: tuple[OccurrenceChange, ...]
    positional_shifts: tuple[OccurrenceShift, ...]
    errors: tuple[ArgumentRequestError, ...] = ()
    request_conflicts: tuple[ArgumentRequestConflict, ...] = ()
    requests: tuple[Any, ...] = ()
    _raw_records: list[_Record] = field(default_factory=list, repr=False)

    def __iter__(self) -> Iterator[ArgumentOccurrence]:
        """Iterate over the original occurrences."""
        return iter(self.occurrences_before)

    def __len__(self) -> int:
        """Return the number of original occurrences."""
        return len(self.occurrences_before)

    @overload
    def __getitem__(self, index: int) -> ArgumentOccurrence: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[ArgumentOccurrence, ...]: ...

    def __getitem__(self, index):
        """Index into the original occurrences."""
        return self.occurrences_before[index]

    @property
    def changed(self) -> bool:
        """Return True if the requests altered the wikitext."""
        return self.string_before != self.string_after

    @property
    def string(self) -> str:
        """The resulting wikitext."""
        return self.string_after

    @property
    def argument_occurrences(self) -> tuple[ArgumentOccurrence, ...]:
        """Alias for the original occurrence list."""
        return self.occurrences_before

    @property
    def effective_arguments(self) -> dict[str, EffectiveArgument]:
        """Alias for the original effective mapping."""
        return self.effective_before

    @property
    def conflicts(self) -> tuple[ArgumentConflict, ...]:
        """Value conflicts that remain after the requests."""
        return self.conflicts_after

    @property
    def new_conflicts(self) -> tuple[ArgumentConflict, ...]:
        """Conflicts introduced by the requests."""
        before_names = {conflict.name for conflict in self.conflicts_before}
        return tuple(
            conflict
            for conflict in self.conflicts_after
            if conflict.name not in before_names
        )

    @property
    def resolved_conflicts(self) -> tuple[ArgumentConflict, ...]:
        """Conflicts present in the original draft but not afterwards."""
        after_names = {conflict.name for conflict in self.conflicts_after}
        return tuple(
            conflict
            for conflict in self.conflicts_before
            if conflict.name not in after_names
        )

    @property
    def removed_occurrences(self) -> tuple[int, ...]:
        """Original indexes that no longer occur in the result."""
        return tuple(
            change.original
            for change in self.occurrence_changes
            if change.kind == 'deleted' and change.original is not None
        )

    @property
    def inserted_occurrences(self) -> tuple[int, ...]:
        """Result indexes of occurrences added by insert requests."""
        return tuple(
            change.after
            for change in self.occurrence_changes
            if change.kind == 'inserted' and change.after is not None
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the report as JSON-serializable data."""
        return {
            'changed': self.changed,
            'string_before': self.string_before,
            'string_after': self.string_after,
            'occurrences_before': [
                occurrence.to_dict() for occurrence in self.occurrences_before
            ],
            'occurrences_after': [
                occurrence.to_dict() for occurrence in self.occurrences_after
            ],
            'effective_before': {
                name: argument.to_dict()
                for name, argument in self.effective_before.items()
            },
            'effective_after': {
                name: argument.to_dict()
                for name, argument in self.effective_after.items()
            },
            'conflicts_before': [
                conflict.to_dict() for conflict in self.conflicts_before
            ],
            'conflicts_after': [
                conflict.to_dict() for conflict in self.conflicts_after
            ],
            'effective_changes': [
                change.to_dict() for change in self.effective_changes
            ],
            'occurrence_changes': [
                change.to_dict() for change in self.occurrence_changes
            ],
            'positional_shifts': [
                shift.to_dict() for shift in self.positional_shifts
            ],
            'errors': [error.to_dict() for error in self.errors],
            'request_conflicts': [
                conflict.to_dict() for conflict in self.request_conflicts
            ],
        }

    def describe(self) -> str:
        """Return a compact human-readable summary of impacts and conflicts."""
        lines: list[str] = []
        for change in self.effective_changes:
            if change.kind == 'added':
                assert change.after is not None
                lines.append(
                    f'added parameter {change.name!r}='
                    f'{change.after.value!r} '
                    f'(occurrence {change.after.occurrence})'
                )
            elif change.kind == 'removed':
                assert change.before is not None
                lines.append(
                    f'removed parameter {change.name!r} '
                    f'(was {change.before.value!r} from occurrence '
                    f'{change.before.occurrence})'
                )
            elif change.kind == 'revealed':
                assert change.before is not None and change.after is not None
                lines.append(
                    f'parameter {change.name!r} revealed '
                    f'{change.after.value!r} from occurrence '
                    f'{change.after.occurrence} '
                    f'(was {change.before.value!r} from occurrence '
                    f'{change.before.occurrence})'
                )
            elif change.kind == 'occurrences':
                assert change.before is not None and change.after is not None
                lines.append(
                    f'parameter {change.name!r} unchanged at '
                    f'{change.after.value!r}; occurrences '
                    f'{change.before.occurrences} -> '
                    f'{change.after.occurrences}'
                )
            else:
                assert change.before is not None and change.after is not None
                lines.append(
                    f'parameter {change.name!r} changed from '
                    f'{change.before.value!r} '
                    f'(occurrence {change.before.occurrence}) to '
                    f'{change.after.value!r} '
                    f'(occurrence {change.after.occurrence})'
                )
        for shift in self.positional_shifts:
            lines.append(
                f'positional occurrence {shift.original} renumbered '
                f'{shift.before_name} -> {shift.after_name} '
                f'({shift.value!r})'
            )
        for value_conflict in self.conflicts_after:
            values = ', '.join(repr(value) for value in value_conflict.values)
            lines.append(
                f'conflict for parameter {value_conflict.name!r}: {values}; '
                f'effective occurrence {value_conflict.effective}'
            )
        for request_conflict in self.request_conflicts:
            requests = ', '.join(str(r) for r in request_conflict.requests)
            lines.append(
                'request conflict at occurrence '
                f'{request_conflict.occurrence}: requests {requests}; '
                f'{request_conflict.reason}'
            )
        for error in self.errors:
            lines.append(f'request {error.request} rejected: {error.reason}')
        if not lines:
            return 'No argument changes.'
        return '\n'.join(lines)


@dataclass
class _Record:
    origin: int | None
    request: int | None
    separator: str
    raw_name: str | None
    raw_value: str
    deleted: bool = False

    @property
    def positional(self) -> bool:
        return self.raw_name is None

    @property
    def source(self) -> str:
        if self.raw_name is None:
            return self.separator + self.raw_value
        return self.separator + self.raw_name + '=' + self.raw_value


def _scan(source: SubWikiTextWithArgs) -> list[ArgumentOccurrence]:
    """Return every occurrence, preserving duplicate effective names."""
    occurrences: list[ArgumentOccurrence] = []
    positional_index = 0
    for index, argument in enumerate(source.arguments, 1):
        raw_source = argument.string
        separator = raw_source[0]
        raw_value = argument.value
        if argument.positional:
            positional_index += 1
            name = str(positional_index)
            raw_name = None
            positional = True
        else:
            raw_name = argument.name
            name = raw_name.strip(_WS)
            positional = False
        occurrences.append(
            ArgumentOccurrence(
                index=index,
                name=name,
                value=_effective_value(raw_value, positional),
                positional=positional,
                source=raw_source,
                origin=index,
                raw_name=raw_name,
                raw_value=raw_value,
                separator=separator,
            )
        )
    return occurrences


def _effective(
    occurrences: list[ArgumentOccurrence],
) -> dict[str, EffectiveArgument]:
    by_name: dict[str, list[ArgumentOccurrence]] = {}
    for occurrence in occurrences:
        by_name.setdefault(occurrence.name, []).append(occurrence)
    effective: dict[str, EffectiveArgument] = {}
    for name, group in by_name.items():
        winner = group[-1]
        effective[name] = EffectiveArgument(
            name=name,
            value=winner.value,
            positional=winner.positional,
            occurrence=winner.index,
            occurrences=tuple(item.index for item in group),
            overridden=tuple(item.index for item in group[:-1]),
            origins=tuple(item.origin for item in group),
            raw_value=winner.raw_value,
        )
    return effective


def _conflicts(
    occurrences: list[ArgumentOccurrence],
) -> tuple[ArgumentConflict, ...]:
    by_name: dict[str, list[ArgumentOccurrence]] = {}
    for occurrence in occurrences:
        by_name.setdefault(occurrence.name, []).append(occurrence)
    conflicts: list[ArgumentConflict] = []
    for name, group in by_name.items():
        winner = group[-1]
        conflicting = tuple(
            occurrence.index
            for occurrence in group
            if occurrence.value != winner.value
        )
        if conflicting:
            conflicts.append(
                ArgumentConflict(
                    name=name,
                    effective=winner.index,
                    effective_value=winner.value,
                    conflicting=conflicting,
                    occurrences=tuple(item.index for item in group),
                    values=tuple(item.value for item in group),
                )
            )
    return tuple(conflicts)


def _records(occurrences: Iterable[ArgumentOccurrence]) -> list[_Record]:
    return [
        _Record(
            origin=occurrence.origin,
            request=None,
            separator=occurrence.separator,
            raw_name=occurrence.raw_name,
            raw_value=occurrence.raw_value,
        )
        for occurrence in occurrences
    ]


def _coerce_bool(value: Any, field_name: str, request: int) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raise ValueError(f'request {request}: {field_name} must be a bool or None')


def _change_index(change: Mapping[str, Any], request: int) -> int:
    for key in ('index', 'position', 'occurrence', 'at'):
        if key in change:
            index = change[key]
            break
    else:
        raise ValueError(f'request {request}: missing occurrence index')
    return _require_index(index, request)


def _coerce_change(change: Any, request: int) -> Any:
    if isinstance(change, ArgumentDelete):
        return ArgumentDelete(_require_index(change.index, request))
    if isinstance(change, ArgumentSet):
        value = change.value
        name = change.name
        if value is not None:
            _require_str(value, 'value', request)
        if name is not None:
            _require_str(name, 'name', request)
        return ArgumentSet(
            index=_require_index(change.index, request),
            value=value,
            name=name,
            positional=_coerce_bool(change.positional, 'positional', request),
        )
    if isinstance(change, ArgumentInsert):
        _require_str(change.value, 'value', request)
        if change.name is not None:
            _require_str(change.name, 'name', request)
        if change.where not in ('before', 'after'):
            raise ValueError(
                f'request {request}: where must be "before" or "after"'
            )
        return ArgumentInsert(
            index=_require_index(change.index, request),
            value=change.value,
            name=change.name,
            where=change.where,
            positional=_coerce_bool(change.positional, 'positional', request),
        )
    if not isinstance(change, Mapping):
        raise TypeError(
            f'request {request}: expected an argument change object or mapping'
        )
    action = change.get('action', change.get('op', change.get('kind')))
    if action is None:
        raise ValueError(f'request {request}: missing action')
    action = str(action).lower()
    index = _change_index(change, request)
    if action in ('delete', 'del', 'remove'):
        return ArgumentDelete(index=index)
    if action in ('set', 'update', 'modify', 'change', 'replace'):
        value = change.get('value')
        name = change.get('name')
        if value is not None and not isinstance(value, str):
            raise TypeError(f'request {request}: value must be a string')
        if name is not None and not isinstance(name, str):
            raise TypeError(f'request {request}: name must be a string')
        return ArgumentSet(
            index=index,
            value=value,
            name=name,
            positional=_coerce_bool(
                change.get('positional'), 'positional', request
            ),
        )
    if action in ('insert', 'add', 'append'):
        where = str(change.get('where', 'after')).lower()
        if where not in ('before', 'after'):
            raise ValueError(
                f'request {request}: where must be "before" or "after"'
            )
        value = change.get('value', '')
        name = change.get('name')
        if not isinstance(value, str):
            raise TypeError(f'request {request}: value must be a string')
        if name is not None and not isinstance(name, str):
            raise TypeError(f'request {request}: name must be a string')
        return ArgumentInsert(
            index=index,
            value=value,
            name=name,
            where=where,  # type: ignore[arg-type]
            positional=_coerce_bool(
                change.get('positional'), 'positional', request
            ),
        )
    raise ValueError(f'request {request}: unknown action {action!r}')


def _insert_boundary(change: ArgumentInsert, count: int, request: int) -> int:
    if change.index < 0:
        raise ValueError(
            f'request {request}: insertion index cannot be negative'
        )
    if change.where == 'after':
        if change.index > count:
            raise ValueError(
                f'request {request}: insertion index {change.index} '
                f'is beyond the original argument count {count}'
            )
        return change.index
    if not 1 <= change.index <= count:
        raise ValueError(
            f'request {request}: insertion index {change.index} '
            f'must be in 1..{count} for where="before"'
        )
    return change.index - 1


def _probe_positional(text: str, request: int) -> None:
    """Reject text that would not stay a single positional argument."""
    from ._wikitext import WikiText

    probe = WikiText('{{t|' + text + '}}')
    templates = probe.templates
    if not templates or templates[0].string != probe.string:
        raise ValueError(
            f'request {request}: text would not stay inside one template'
        )
    arguments = templates[0].arguments
    if len(arguments) != 1:
        raise ValueError(
            f'request {request}: text would create {len(arguments)} '
            'template arguments instead of one'
        )
    argument = arguments[0]
    if (
        not argument.positional
        or argument.name != '1'
        or argument.value != text
    ):
        raise ValueError(
            f'request {request}: value would not be preserved as a single '
            'positional argument; it would change the parameter name'
        )


def _probe_named(name: str, text: str, request: int) -> None:
    """Reject text that would not stay the same single named argument."""
    from ._wikitext import WikiText

    if not name.strip(_WS):
        raise ValueError(
            f'request {request}: a parameter name cannot be empty'
        )
    probe = WikiText('{{t|' + name + '=' + text + '}}')
    templates = probe.templates
    if not templates or templates[0].string != probe.string:
        raise ValueError(
            f'request {request}: text would not stay inside one template'
        )
    arguments = templates[0].arguments
    if len(arguments) != 1:
        raise ValueError(
            f'request {request}: text would create {len(arguments)} '
            'template arguments instead of one'
        )
    argument = arguments[0]
    if argument.positional or argument.name != name:
        raise ValueError(
            f'request {request}: name/value pair would not be preserved as '
            'one named argument'
        )


def _apply_set(record: _Record, change: ArgumentSet, request: int) -> None:
    positional_after = (
        record.positional if change.positional is None else change.positional
    )
    name_after = (
        change.name
        if change.name is not None
        else (None if positional_after else record.raw_name)
    )
    if change.value is not None or change.name is not None:
        value_after = (
            record.raw_value if change.value is None else change.value
        )
        if positional_after:
            _probe_positional(value_after, request)
        else:
            assert name_after is not None
            _probe_named(
                name_after,
                record.raw_value if change.value is None else change.value,
                request,
            )
    if change.positional is True:
        if change.name is not None:
            raise ValueError(
                f'request {request}: a positional occurrence cannot have a name'
            )
        record.raw_name = None
    elif change.positional is False:
        if record.positional and change.name is None:
            raise ValueError(
                f'request {request}: converting a positional occurrence to a '
                'named one requires a name'
            )
        if change.name is not None:
            record.raw_name = change.name
    elif change.name is not None:
        record.raw_name = change.name
    if change.value is not None:
        record.raw_value = change.value


def _record_names(records: list[_Record]) -> list[str]:
    names: list[str] = []
    positional_index = 0
    for record in records:
        if record.raw_name is None:
            positional_index += 1
            names.append(str(positional_index))
        else:
            names.append(record.raw_name.strip(_WS))
    return names


def _sequence(
    records: list[_Record],
    insertions: dict[int, list[tuple[int, _Record]]],
) -> list[_Record]:
    final_records: list[_Record] = []
    for boundary in range(len(records) + 1):
        final_records.extend(
            record for _, record in insertions.get(boundary, ())
        )
        if boundary < len(records) and not records[boundary].deleted:
            final_records.append(records[boundary])
    return final_records


def _make_insert_record(change: ArgumentInsert, request: int) -> _Record:
    positional = change.positional
    if positional is None:
        positional = change.name is None
    if positional and change.name is not None:
        raise ValueError(
            f'request {request}: a positional occurrence cannot have a name'
        )
    if not positional and change.name is None:
        raise ValueError(
            f'request {request}: a named occurrence requires a name'
        )
    if positional:
        _probe_positional(change.value, request)
    else:
        assert change.name is not None
        _probe_named(change.name, change.value, request)
    return _Record(
        origin=None,
        request=request,
        separator='|',
        raw_name=None if positional else change.name,
        raw_value=change.value,
    )


def _normalize_separators(
    records: list[_Record], first_separator: str
) -> None:
    if not records:
        return
    records[0].separator = first_separator
    for record in records[1:]:
        record.separator = '|'


def _after_occurrences(records: list[_Record]) -> list[ArgumentOccurrence]:
    occurrences: list[ArgumentOccurrence] = []
    positional_index = 0
    for index, record in enumerate(records, 1):
        if record.raw_name is None:
            positional_index += 1
            name = str(positional_index)
            raw_name = None
            positional = True
        else:
            raw_name = record.raw_name
            name = raw_name.strip(_WS)
            positional = False
        occurrences.append(
            ArgumentOccurrence(
                index=index,
                name=name,
                value=_effective_value(record.raw_value, positional),
                positional=positional,
                source=record.source,
                origin=record.origin,
                raw_name=raw_name,
                raw_value=record.raw_value,
                separator=record.separator,
            )
        )
    return occurrences


def _effective_changes(
    before: dict[str, EffectiveArgument],
    after: dict[str, EffectiveArgument],
) -> tuple[EffectiveChange, ...]:
    changes: list[EffectiveChange] = []
    for name in list(before) + [n for n in after if n not in before]:
        old = before.get(name)
        new = after.get(name)
        if old is None:
            changes.append(EffectiveChange(name, 'added', None, new))
            continue
        if new is None:
            changes.append(EffectiveChange(name, 'removed', old, None))
            continue
        old_origin = old.origins[-1] if old.origins else None
        new_origin = new.origins[-1] if new.origins else None
        reveal = (
            new_origin is not None
            and new_origin != old_origin
            and new_origin in old.origins[:-1]
        )
        if old.value != new.value:
            changes.append(
                EffectiveChange(
                    name, 'revealed' if reveal else 'updated', old, new
                )
            )
        elif new_origin != old_origin:
            changes.append(
                EffectiveChange(
                    name,
                    'revealed' if reveal else 'winner',
                    old,
                    new,
                )
            )
        elif old.positional != new.positional:
            changes.append(EffectiveChange(name, 'form', old, new))
        elif old.origins != new.origins:
            changes.append(EffectiveChange(name, 'occurrences', old, new))
    return tuple(changes)


def _occurrence_changes(
    before: tuple[ArgumentOccurrence, ...],
    records: list[_Record],
    after: tuple[ArgumentOccurrence, ...],
) -> tuple[OccurrenceChange, ...]:
    changes: list[OccurrenceChange] = []
    before_by_index = {occurrence.index: occurrence for occurrence in before}
    seen_origins: set[int] = set()
    for record, occurrence in zip(records, after):
        if record.origin is None:
            changes.append(
                OccurrenceChange(
                    kind='inserted',
                    original=None,
                    after=occurrence.index,
                    request=record.request,
                    after_name=occurrence.name,
                    after_value=occurrence.value,
                    after_source=occurrence.source,
                )
            )
            continue
        seen_origins.add(record.origin)
        old = before_by_index[record.origin]

        def add_change(
            kind: str,
            before_value: str,
            after_value: str,
            *,
            origin: int | None = record.origin,
            after: int = occurrence.index,
            request: int | None = record.request,
            before_name: str = old.name,
            after_name: str = occurrence.name,
            before_source: str = old.source,
            after_source: str = occurrence.source,
        ) -> None:
            changes.append(
                OccurrenceChange(
                    kind=kind,
                    original=origin,
                    after=after,
                    request=request,
                    before_name=before_name,
                    after_name=after_name,
                    before_value=before_value,
                    after_value=after_value,
                    before_source=before_source,
                    after_source=after_source,
                )
            )

        if old.value != occurrence.value:
            add_change('value', old.value, occurrence.value)
        elif old.raw_value != occurrence.raw_value:
            add_change('raw_value', old.raw_value, occurrence.raw_value)
        if old.positional != occurrence.positional:
            add_change('form', old.value, occurrence.value)
        elif old.name != occurrence.name:
            add_change(
                'renumbered' if old.positional else 'renamed',
                old.value,
                occurrence.value,
            )
        if record.separator != old.separator:
            add_change('separator', old.value, occurrence.value)
    for occurrence in before:
        if occurrence.index not in seen_origins:
            changes.append(
                OccurrenceChange(
                    kind='deleted',
                    original=occurrence.index,
                    after=None,
                    before_name=occurrence.name,
                    before_value=occurrence.value,
                    before_source=occurrence.source,
                )
            )
    return tuple(changes)


def _positional_shifts(
    before: tuple[ArgumentOccurrence, ...],
    after: tuple[ArgumentOccurrence, ...],
    records: list[_Record],
) -> tuple[OccurrenceShift, ...]:
    before_by_index = {occurrence.index: occurrence for occurrence in before}
    shifts: list[OccurrenceShift] = []
    for record, occurrence in zip(records, after):
        if record.origin is None:
            continue
        old = before_by_index[record.origin]
        if (
            old.positional
            and occurrence.positional
            and old.name != occurrence.name
        ):
            shifts.append(
                OccurrenceShift(
                    original=record.origin,
                    before_name=old.name,
                    after_name=occurrence.name,
                    value=occurrence.value,
                    source=occurrence.source,
                )
            )
    return tuple(shifts)


def import_arguments(
    source: SubWikiTextWithArgs,
    changes: Iterable[Any] = (),
) -> ArgumentImportReport:
    """Import argument occurrences and optionally apply edit requests.

    Every request index refers to the original occurrence list (1-based),
    before any request is applied.  Deleting an occurrence therefore never
    changes what position another request addresses, and inserting a positional
    occurrence never changes which occurrence a delete/set request selects.
    Ties between insertions at the same boundary keep request order.

    Invalid requests are recorded in ``errors`` and do not abort the rest of
    the batch.  Two delete/set requests for the same original occurrence are
    both rejected and reported in ``request_conflicts``, so neither silently
    wins.
    """
    string_before = source.string
    before = tuple(_scan(source))
    effective_before = _effective(list(before))
    conflicts_before = _conflicts(list(before))
    orders = tuple(changes)

    records = _records(before)
    by_origin = {record.origin: record for record in records}
    first_separator = chr(source._first_arg_sep)

    errors: list[ArgumentRequestError] = []
    request_conflicts: list[ArgumentRequestConflict] = []

    normalized: list[Any] = []
    for request, raw_change in enumerate(orders, 1):
        try:
            normalized.append(_coerce_change(raw_change, request))
        except (TypeError, ValueError) as error:
            errors.append(
                ArgumentRequestError(request, raw_change, str(error))
            )
            normalized.append(None)

    targeted: dict[int, list[int]] = {}
    for request, change in enumerate(normalized, 1):
        if isinstance(change, (ArgumentDelete, ArgumentSet)):
            targeted.setdefault(change.index, []).append(request)
    conflicted = {
        index for index, requests in targeted.items() if len(requests) > 1
    }
    for index in sorted(conflicted):
        request_conflicts.append(
            ArgumentRequestConflict(
                occurrence=index,
                requests=tuple(targeted[index]),
                reason='more than one delete/set request targets this '
                'occurrence; none of them was applied',
            )
        )

    insertions: dict[int, list[tuple[int, _Record]]] = {}
    deferred: list[tuple[int, ArgumentSet, _Record]] = []
    for request, change in enumerate(normalized, 1):
        if change is None:
            continue
        if isinstance(change, (ArgumentDelete, ArgumentSet)):
            if change.index in conflicted:
                continue
            if change.index not in by_origin:
                errors.append(
                    ArgumentRequestError(
                        request,
                        change,
                        f'no occurrence at original index {change.index}',
                    )
                )
                continue
            record = by_origin[change.index]
            if isinstance(change, ArgumentDelete):
                record.deleted = True
                record.request = request
                continue
            assert isinstance(change, ArgumentSet)
            if change.positional is True and not record.positional:
                deferred.append((request, change, record))
                continue
            try:
                _apply_set(record, change, request)
            except ValueError as error:
                errors.append(
                    ArgumentRequestError(request, change, str(error))
                )
                continue
            record.request = request
        elif isinstance(change, ArgumentInsert):
            try:
                boundary = _insert_boundary(change, len(before), request)
                record = _make_insert_record(change, request)
            except (TypeError, ValueError) as error:
                errors.append(
                    ArgumentRequestError(request, change, str(error))
                )
                continue
            insertions.setdefault(boundary, []).append((request, record))

    for request, change, record in deferred:
        original_name = record.raw_name.strip(_WS) if record.raw_name else ''
        old_raw_name = record.raw_name
        record.raw_name = None
        sequence = _sequence(records, insertions)
        names = _record_names(sequence)
        position = next(
            index
            for index, candidate in enumerate(sequence)
            if candidate is record
        )
        if names[position] != original_name:
            record.raw_name = old_raw_name
            errors.append(
                ArgumentRequestError(
                    request,
                    change,
                    'converting to positional form would renumber this '
                    f'occurrence from {original_name!r} to '
                    f'{names[position]!r}',
                )
            )
            continue
        record.raw_name = old_raw_name
        try:
            _apply_set(record, change, request)
        except ValueError as error:
            errors.append(ArgumentRequestError(request, change, str(error)))
            continue
        record.request = request

    final_records: list[_Record] = []
    for boundary in range(len(records) + 1):
        final_records.extend(
            record for _, record in insertions.get(boundary, ())
        )
        if boundary < len(records) and not records[boundary].deleted:
            final_records.append(records[boundary])

    _normalize_separators(final_records, first_separator)
    after = tuple(_after_occurrences(final_records))

    if final_records:
        new_content = source.name + ''.join(
            record.source for record in final_records
        )
    else:
        new_content = source.name
    if source(2, -2) != new_content:
        source[2:-2] = new_content
    string_after = source.string

    effective_after = _effective(list(after))
    conflicts_after = _conflicts(list(after))

    return ArgumentImportReport(
        source=source,
        string_before=string_before,
        string_after=string_after,
        occurrences_before=before,
        occurrences_after=after,
        effective_before=effective_before,
        effective_after=effective_after,
        conflicts_before=conflicts_before,
        conflicts_after=conflicts_after,
        effective_changes=_effective_changes(
            effective_before, effective_after
        ),
        occurrence_changes=_occurrence_changes(before, final_records, after),
        positional_shifts=_positional_shifts(before, after, final_records),
        errors=tuple(errors),
        request_conflicts=tuple(request_conflicts),
        requests=orders,
        _raw_records=final_records,
    )


def apply_argument_changes(
    source: SubWikiTextWithArgs,
    changes: Iterable[Any],
) -> ArgumentImportReport:
    """Alias for :func:`import_arguments` with required ``changes``."""
    return import_arguments(source, changes)
