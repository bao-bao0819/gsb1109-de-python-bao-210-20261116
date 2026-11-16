"""Analyze ``<ref>`` tags: occurrences, definitions, and reuse.

MediaWiki allows a named reference to be defined once (or several times)
and reused through self-closing tags::

    <ref name="r">citation body</ref>
    <ref name="r" />

Named references are matched by the ``(group, name)`` pair. A reuse may
appear before the definition; matching is therefore performed against the
whole document rather than on a first-seen basis. When more than one
*distinct* body is supplied for the same key, every definition is retained
and the key is reported as a conflict instead of silently keeping only the
last body.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from ._tag import Tag
    from ._wikitext import WikiText

RefKey = tuple[str, Union[str, None]]


@dataclass
class RefOccurrence:
    """A single ``<ref>`` tag as it appears in the document.

    Every ``<ref>`` tag, whether defining a citation or reusing one, is
    represented by one occurrence ordered by its position.
    """

    __slots__ = (
        'body',
        'column',
        'definition',
        'end',
        'group',
        'in_references',
        'is_definition',
        'is_forward_reference',
        'line',
        'name',
        'ordinal',
        'start',
        'tag',
    )

    ordinal: int
    start: int
    end: int
    line: int
    column: int
    name: str | None
    group: str
    body: str | None
    in_references: bool
    is_definition: bool
    is_forward_reference: bool
    definition: RefDefinition | None
    tag: Tag

    @property
    def key(self) -> RefKey:
        """Return the ``(group, name)`` lookup key."""
        return self.group, self.name

    def as_dict(self) -> dict:
        """Return a plain dictionary representation of this occurrence."""
        return {
            'ordinal': self.ordinal,
            'start': self.start,
            'end': self.end,
            'line': self.line,
            'column': self.column,
            'name': self.name,
            'group': self.group,
            'body': self.body,
            'in_references': self.in_references,
            'is_definition': self.is_definition,
            'is_forward_reference': self.is_forward_reference,
            'definition_ordinal': (
                None if self.definition is None else self.definition.ordinal
            ),
        }


@dataclass
class RefDefinition:
    """A ``<ref>`` tag carrying citation body for a key.

    A self-closing tag never defines anything. A paired tag, even with an
    empty body, registers a definition (matching MediaWiki behavior).
    """

    __slots__ = (
        'body',
        'column',
        'conflicts',
        'end',
        'group',
        'line',
        'name',
        'occurrence',
        'ordinal',
        'reuses',
        'start',
    )

    ordinal: int
    start: int
    end: int
    line: int
    column: int
    name: str | None
    group: str
    body: str
    occurrence: RefOccurrence
    reuses: list[RefOccurrence]
    conflicts: tuple[RefDefinition, ...]

    @property
    def key(self) -> RefKey:
        """Return the ``(group, name)`` lookup key."""
        return self.group, self.name

    def as_dict(self) -> dict:
        """Return a plain dictionary representation of this definition."""
        return {
            'ordinal': self.ordinal,
            'start': self.start,
            'end': self.end,
            'line': self.line,
            'column': self.column,
            'name': self.name,
            'group': self.group,
            'body': self.body,
            'reuse_ordinals': [o.ordinal for o in self.reuses],
            'conflict_ordinals': [d.ordinal for d in self.conflicts],
        }


@dataclass
class RefReport:
    """Result of analyzing the ``<ref>`` tags of a document."""

    __slots__ = (
        '_definitions_by_key',
        'conflicts',
        'definitions',
        'occurrences',
    )

    occurrences: list[RefOccurrence]
    definitions: list[RefDefinition]
    conflicts: dict[RefKey, tuple[RefDefinition, ...]]
    _definitions_by_key: dict[RefKey, list[RefDefinition]]

    def definitions_for(
        self, name: str, group: str = ''
    ) -> list[RefDefinition]:
        """Return all definitions registered for ``(group, name)``."""
        return self._definitions_by_key.get((group, name), [])

    def occurrences_for(
        self, name: str, group: str = ''
    ) -> list[RefOccurrence]:
        """Return all occurrences (definitions and reuses) for a key."""
        return [
            occurrence
            for occurrence in self.occurrences
            if occurrence.name == name and occurrence.group == group
        ]

    def as_dict(self) -> dict:
        """Return both tables as plain dictionaries."""
        return {
            'occurrences': [o.as_dict() for o in self.occurrences],
            'definitions': [d.as_dict() for d in self.definitions],
            'conflicts': [
                {
                    'group': group,
                    'name': name,
                    'definition_ordinals': [
                        d.ordinal for d in key_definitions
                    ],
                }
                for (group, name), key_definitions in self.conflicts.items()
            ],
        }

    def __str__(self) -> str:
        lines = ['Occurrences:']
        for occurrence in self.occurrences:
            kind = 'definition' if occurrence.is_definition else 'reference'
            target = (
                ''
                if occurrence.definition is None
                else f' -> def#{occurrence.definition.ordinal}'
            )
            lines.append(
                f'  #{occurrence.ordinal} L{occurrence.line}:'
                f'{occurrence.column} group={occurrence.group!r} '
                f'name={occurrence.name!r} {kind}{target}'
            )
        lines.append('Definitions:')
        for definition in self.definitions:
            marker = ' CONFLICT' if definition.conflicts else ''
            lines.append(
                f'  #{definition.ordinal} L{definition.line}:'
                f'{definition.column} group={definition.group!r} '
                f'name={definition.name!r} body={definition.body!r}'
                f' reuses={[o.ordinal for o in definition.reuses]}'
                f'{marker}'
            )
        return '\n'.join(lines)


def _line_starts(string: str) -> list[int]:
    return [0, *(i + 1 for i, char in enumerate(string) if char == '\n')]


def _line_and_column(line_starts: list[int], start: int) -> tuple[int, int]:
    line_index = bisect_right(line_starts, start) - 1
    return line_index + 1, start - line_starts[line_index] + 1


def _enclosing_references_group(tag: Tag) -> tuple[bool, str | None]:
    """Return (inside references, group of enclosing references tag)."""
    for ancestor in tag.ancestors('ExtensionTag'):
        if ancestor.name == 'references':
            return True, ancestor.get_attr('group')
    return False, None


def analyze_refs(wikitext: WikiText) -> RefReport:
    """Build the occurrence and definition tables for ``<ref>`` tags.

    Reuses are matched to definitions by ``(group, name)`` across the whole
    document, so a self-closing reference placed before its definition is
    still linked. Distinct bodies for the same key are all kept and reported
    as conflicts; the first definition in document order is used as the
    link target for occurrences of a conflicting key.
    """
    string = wikitext._lststr[0]
    line_starts = _line_starts(string)
    occurrences: list[RefOccurrence] = []
    definitions: list[RefDefinition] = []
    definitions_by_key: dict[RefKey, list[RefDefinition]] = {}

    for ordinal, tag in enumerate(wikitext.get_tags('ref')):
        span = tag._span_data
        start, end = span.start, span.end
        line, column = _line_and_column(line_starts, start)
        # An empty name attribute cannot be used for reuse and is treated
        # as anonymous, matching MediaWiki's Cite behavior.
        name = tag.get_attr('name') or None
        group = tag.get_attr('group')
        in_references, references_group = _enclosing_references_group(tag)
        if group is None:
            group = references_group or ''
        # A self-closing tag has no "contents" capture group and can only
        # reuse an existing definition; a paired tag always defines one.
        is_definition = tag._match.span('contents') != (-1, -1)
        body = tag.contents if is_definition else None
        occurrence = RefOccurrence(
            ordinal=ordinal,
            start=start,
            end=end,
            line=line,
            column=column,
            name=name,
            group=group,
            body=body,
            in_references=in_references,
            is_definition=is_definition,
            is_forward_reference=False,
            definition=None,
            tag=tag,
        )
        occurrences.append(occurrence)
        if is_definition:
            definition = RefDefinition(
                ordinal=ordinal,
                start=start,
                end=end,
                line=line,
                column=column,
                name=name,
                group=group,
                body=body or '',
                occurrence=occurrence,
                reuses=[],
                conflicts=(),
            )
            definitions.append(definition)
            if name is not None:
                definitions_by_key.setdefault((group, name), []).append(
                    definition
                )

    # Link every named occurrence to the definitions of its key, regardless
    # of whether the reuse comes before or after the definition.
    for occurrence in occurrences:
        if occurrence.name is None:
            continue
        key_definitions = definitions_by_key.get(occurrence.key)
        if not key_definitions:
            continue
        first = key_definitions[0]
        occurrence.definition = first
        occurrence.is_forward_reference = occurrence.ordinal < first.ordinal
        if not occurrence.is_definition:
            first.reuses.append(occurrence)

    # Preserve every distinct body for a key instead of keeping only one.
    conflicts: dict[RefKey, tuple[RefDefinition, ...]] = {}
    for key, key_definitions in definitions_by_key.items():
        distinct_bodies = {d.body for d in key_definitions}
        if len(distinct_bodies) > 1:
            key_conflicts = tuple(key_definitions)
            conflicts[key] = key_conflicts
            for definition in key_definitions:
                definition.conflicts = tuple(
                    other
                    for other in key_conflicts
                    if other is not definition
                    and other.body != definition.body
                )

    return RefReport(
        occurrences=occurrences,
        definitions=definitions,
        conflicts=conflicts,
        _definitions_by_key=definitions_by_key,
    )
