"""Extract storable field values from wikitext without expanding templates.

Templates in an article are usually left unexpanded when the wikitext is
stored.  Some field values are nevertheless definite literal strings and can
be stored as they are; others can only be known after MediaWiki (or another
renderer) expands a template, parser function, or extension tag.  This module
classifies each field and records why it got that classification.

The three moving parts are:

* :class:`TemplateSchema` -- the known template/parameter convention: which
  template supplies which field and through which parameter names.
* ``bindings`` -- the transclusion frame used to evaluate ``{{{parameter}}}``
  constructs.  A missing key means *unbound* (the default applies); a key
  mapped to ``''`` means *bound to the empty string* (the default does **not**
  apply).
* :func:`extract_fields` -- returns a :class:`FieldReport` with the literal
  values, the expansion-dependent values, and the basis for each.

Example:
    >>> schema = TemplateSchema({
    ...     'Infobox person': {
    ...         'name': ('name', '1'),
    ...         'birth_date': ('birth_date', 'birthdate'),
    ...     },
    ... })
    >>> source = (
    ...     '{{Infobox person|Alice|'
    ...     'birth_date={{Birth date|1990|1|1}}}}'
    ... )
    >>> report = extract_fields(source, schema)
    >>> report.literal_values
    {'name': 'Alice'}
    >>> report.expansion_dependent[0].field
    'birth_date'
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from json import dumps as _json_dumps
from types import MappingProxyType
from typing import Any

from ._comment_bold_italic import COMMENT_PATTERN
from ._config import _tag_extensions
from ._template import Template
from ._wikitext import WS, WikiText, rc

COMMENT_SUB = rc(COMMENT_PATTERN).sub
SIGNATURE_FINDITER = rc(r'~{3,5}').finditer
CHARACTER_REFERENCE_SEARCH = rc(
    r'&(?:#\d+|#[xX][0-9a-fA-F]+|[A-Za-z]+);'
).search

#: Tags whose contents MediaWiki renders without further wiki processing.
_LITERAL_TAG = 'nowiki'


class FieldStatus(Enum):
    """How safely a field value can be stored."""

    #: The value is a definite literal string and can be stored as-is.
    LITERAL = 'literal'
    #: The value contains something that needs external expansion.
    EXPANSION_DEPENDENT = 'expansion-dependent'
    #: A ``{{{parameter}}}`` has no binding and no default.
    UNBOUND = 'unbound'
    #: The template or the parameter is not present in the source.
    ABSENT = 'absent'

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Construct:
    """A piece of wikitext that MediaWiki has to process.

    ``span`` is relative to the analyzed source.  It is ``None`` for text
    that did not come from the source (for example a value supplied through
    ``bindings``).
    """

    kind: str
    text: str
    span: tuple[int, int] | None = None
    detail: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            'kind': self.kind,
            'text': self.text,
            'span': list(self.span) if self.span is not None else None,
            'detail': self.detail,
        }


@dataclass(frozen=True)
class Resolution:
    """The result of resolving a single wikitext fragment."""

    status: FieldStatus
    value: str | None
    raw: str
    constructs: tuple[Construct, ...] = ()
    notes: tuple[str, ...] = ()
    #: Binding state when the whole fragment is one ``{{{parameter}}}``.
    binding: str | None = None
    #: Binding state of every ``{{{parameter}}}`` met while resolving.
    bindings: tuple[tuple[str, str], ...] = ()
    #: Names whose ``{{{name|default}}}`` default was substituted.
    parameter_defaults_used: tuple[str, ...] = ()

    @property
    def basis(self) -> str:
        """Human-readable reason for the status."""
        return '; '.join(self.notes)

    @property
    def literal(self) -> bool:
        return self.status is FieldStatus.LITERAL

    @property
    def expansion_dependent(self) -> bool:
        return self.status is FieldStatus.EXPANSION_DEPENDENT

    @property
    def default_used(self) -> bool:
        return bool(self.parameter_defaults_used)

    def as_dict(self) -> dict[str, Any]:
        return {
            'status': self.status.value,
            'value': self.value,
            'raw': self.raw,
            'binding': self.binding,
            'bindings': dict(self.bindings),
            'default_used': self.default_used,
            'parameter_defaults_used': list(self.parameter_defaults_used),
            'constructs': [c.as_dict() for c in self.constructs],
            'notes': list(self.notes),
            'basis': self.basis,
        }


@dataclass(frozen=True)
class FieldValue:
    """A field extracted from a template invocation."""

    field: str
    status: FieldStatus
    value: str | None
    raw: str | None
    template: str
    parameter: str | None = None
    span: tuple[int, int] | None = None
    argument_span: tuple[int, int] | None = None
    argument_present: bool = False
    #: True when the schema's declared default was used.
    schema_default_used: bool = False
    #: Binding state of every ``{{{parameter}}}`` met in the value.
    bindings: tuple[tuple[str, str], ...] = ()
    #: Names whose ``{{{name|default}}}`` default was substituted.
    parameter_defaults_used: tuple[str, ...] = ()
    constructs: tuple[Construct, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def basis(self) -> str:
        """Human-readable reason for the status."""
        return '; '.join(self.notes)

    @property
    def literal(self) -> bool:
        return self.status is FieldStatus.LITERAL

    @property
    def expansion_dependent(self) -> bool:
        return self.status is FieldStatus.EXPANSION_DEPENDENT

    @property
    def default_used(self) -> bool:
        return bool(self.parameter_defaults_used)

    def as_dict(self) -> dict[str, Any]:
        return {
            'field': self.field,
            'status': self.status.value,
            'value': self.value,
            'raw': self.raw,
            'template': self.template,
            'parameter': self.parameter,
            'span': list(self.span) if self.span is not None else None,
            'argument_span': (
                list(self.argument_span)
                if self.argument_span is not None
                else None
            ),
            'argument_present': self.argument_present,
            'schema_default_used': self.schema_default_used,
            'default_used': self.default_used,
            'bindings': dict(self.bindings),
            'parameter_defaults_used': list(self.parameter_defaults_used),
            'constructs': [c.as_dict() for c in self.constructs],
            'notes': list(self.notes),
            'basis': self.basis,
        }


@dataclass(frozen=True)
class FieldSpec:
    """Describe how one field maps onto a template's parameters.

    :param parameters: parameter names to try, in order.  They may be
        ordinary names or explicit positional indices such as ``'1'``.
    :param default: value used when none of ``parameters`` is present.  It may
        itself contain ``{{{parameter|default}}}`` constructs, which are only
        resolved according to ``bindings``.
    """

    parameters: tuple[str, ...] = ()
    default: str | None = None


@dataclass(frozen=True)
class FieldReport:
    """The fields extracted from a source, grouped by status."""

    fields: tuple[FieldValue, ...]

    def __iter__(self):
        return iter(self.fields)

    def __len__(self) -> int:
        return len(self.fields)

    def __getitem__(self, index):
        return self.fields[index]

    def with_status(self, status: FieldStatus) -> list[FieldValue]:
        return [f for f in self.fields if f.status is status]

    @property
    def literal(self) -> list[FieldValue]:
        return self.with_status(FieldStatus.LITERAL)

    @property
    def expansion_dependent(self) -> list[FieldValue]:
        return self.with_status(FieldStatus.EXPANSION_DEPENDENT)

    @property
    def unbound(self) -> list[FieldValue]:
        return self.with_status(FieldStatus.UNBOUND)

    @property
    def absent(self) -> list[FieldValue]:
        return self.with_status(FieldStatus.ABSENT)

    @property
    def literal_values(self) -> dict[str, str]:
        """The definite values that are safe to store, keyed by field."""
        return {
            f.field: f.value
            for f in self.fields
            if f.status is FieldStatus.LITERAL
        }

    def by_field(self) -> dict[str, FieldValue]:
        return {f.field: f for f in self.fields}

    def to_dict(self) -> dict[str, Any]:
        return {
            'literal_values': self.literal_values,
            'literal': [f.as_dict() for f in self.literal],
            'expansion_dependent': [
                f.as_dict() for f in self.expansion_dependent
            ],
            'unbound': [f.as_dict() for f in self.unbound],
            'absent': [f.as_dict() for f in self.absent],
            'fields': [f.as_dict() for f in self.fields],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return _json_dumps(self.to_dict(), ensure_ascii=False, indent=indent)


def _coerce_field_spec(field_name: str, spec: Any) -> FieldSpec:
    if isinstance(spec, FieldSpec):
        parameters = spec.parameters
        default = spec.default
    elif isinstance(spec, str):
        parameters, default = (spec,), None
    elif isinstance(spec, Mapping):
        unknown = set(spec) - {'parameters', 'default'}
        if unknown:
            raise ValueError(
                f'unknown FieldSpec keys for {field_name!r}: {sorted(unknown)}'
            )
        parameters = spec.get('parameters', ())
        default = spec.get('default')
    elif isinstance(spec, Sequence):
        parameters, default = spec, None
    else:
        raise TypeError(
            f'field {field_name!r} must be a FieldSpec, a string, a '
            f'sequence of strings, or a mapping; got {type(spec).__name__}'
        )

    if isinstance(parameters, str):
        parameters = (parameters,)
    else:
        parameters = tuple(parameters)
    for name in parameters:
        if not isinstance(name, str):
            raise TypeError(
                f'parameter names for field {field_name!r} must be strings'
            )
    if default is not None and not isinstance(default, str):
        raise TypeError(
            f'default for field {field_name!r} must be a string or None'
        )
    return FieldSpec(parameters, default)


class TemplateSchema:
    """A known template/parameter convention.

    :param templates: mapping of template name to a mapping of field name to
        its :class:`FieldSpec`.  A field spec may also be given as a string, a
        sequence of parameter names, or a mapping with ``parameters`` and
        ``default`` keys.

    Template names are normalized like :meth:`Template.normal_name` does
    (namespace removed, underscores to spaces, first letter capitalized), so
    ``'template:Infobox person'`` and ``'infobox_person'`` match the same
    template.
    """

    __slots__ = ('templates',)

    def __init__(self, templates: Mapping[str, Mapping[str, Any]]):
        if not isinstance(templates, Mapping):
            raise TypeError('templates must be a mapping')
        normalized: dict[str, Mapping[str, FieldSpec]] = {}
        for name, fields in templates.items():
            key = _normalize_template_name(name)
            if key in normalized:
                raise ValueError(f'duplicate template {key!r}')
            if not isinstance(fields, Mapping):
                raise TypeError(
                    f'fields of template {name!r} must be a mapping'
                )
            normalized_fields: dict[str, FieldSpec] = {}
            for field_name, spec in fields.items():
                if not isinstance(field_name, str) or not field_name:
                    raise ValueError('field names must be non-empty strings')
                normalized_fields[field_name] = _coerce_field_spec(
                    field_name, spec
                )
            normalized[key] = MappingProxyType(normalized_fields)
        self.templates = MappingProxyType(normalized)

    def __repr__(self) -> str:
        return f'TemplateSchema({dict(self.templates)!r})'


def _normalize_template_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError('template names must be strings')
    if any(char in name for char in '{}|'):
        raise ValueError(
            f'template name {name!r} must not contain braces or a pipe'
        )
    if not name.strip(WS):
        raise ValueError('template names must not be empty')
    return Template('{{' + name + '}}').normal_name(capitalize=True)


def _validate_bindings(
    bindings: Mapping[str, str] | None,
) -> Mapping[str, str]:
    if bindings is None:
        return {}
    if not isinstance(bindings, Mapping):
        raise TypeError('bindings must be a mapping of names to strings')
    for name, value in bindings.items():
        if not isinstance(name, str):
            raise TypeError('binding names must be strings')
        if not isinstance(value, str):
            raise TypeError(
                f'binding for {name!r} must be a string; use '
                f"bindings={{...: ''}} to bind an empty value"
            )
    return bindings


def _outermost_items(parsed: WikiText) -> list[tuple]:
    """Return the outermost expansion constructs, left to right."""
    items: list[tuple] = []
    for template in parsed.templates:
        items.append(
            (
                'template',
                template.span,
                template.string,
                template.name.strip(WS),
                template,
            )
        )
    for parser_function in parsed.parser_functions:
        items.append(
            (
                'parser-function',
                parser_function.span,
                parser_function.string,
                parser_function.name.strip(WS),
                parser_function,
            )
        )
    for parameter in parsed.parameters:
        items.append(
            (
                'parameter',
                parameter.span,
                parameter.string,
                COMMENT_SUB('', parameter.name).strip(WS),
                parameter,
            )
        )
    for tag in parsed.get_tags():
        name = tag.name
        if name == _LITERAL_TAG:
            items.append(
                (
                    'nowiki',
                    tag.span,
                    tag.contents or '',
                    name,
                    tag,
                )
            )
        elif name in _tag_extensions:
            items.append(('extension-tag', tag.span, tag.string, name, tag))

    # Most-enclosing first.  Nested constructs are then dropped below.
    items.sort(key=lambda item: (item[1][0], -item[1][1]))
    outermost: list[tuple] = []
    last_end = -1
    for item in items:
        start, end = item[1]
        if start >= last_end:
            outermost.append(item)
            last_end = end
    return outermost


@dataclass()
class _Fragment:
    text: str
    constructs: list[Construct]
    unresolved: list[str]
    bindings: list[tuple[str, str]]
    parameter_defaults_used: list[str]
    notes: list[str]


def _collect_signatures(
    literal: str,
    offset: int | None,
    constructs: list[Construct],
) -> None:
    for match in SIGNATURE_FINDITER(literal):
        start, end = match.span()
        constructs.append(
            Construct(
                'signature',
                match.group(),
                (offset + start, offset + end) if offset is not None else None,
            )
        )


def _absolute(
    span: tuple[int, int], offset: int | None
) -> tuple[int, int] | None:
    if offset is None:
        return None
    return offset + span[0], offset + span[1]


def _resolve_fragment(
    text: str,
    bindings: Mapping[str, str],
    offset: int | None,
) -> _Fragment:
    items = _outermost_items(WikiText(text))
    pieces: list[str] = []
    constructs: list[Construct] = []
    unresolved: list[str] = []
    binding_states: list[tuple[str, str]] = []
    parameter_defaults_used: list[str] = []
    notes: list[str] = []
    position = 0

    for kind, span, raw, detail, obj in items:
        start, end = span
        if start > position:
            literal = text[position:start]
            pieces.append(literal)
            _collect_signatures(
                literal,
                offset + position if offset is not None else None,
                constructs,
            )
        position = end

        if kind == 'nowiki':
            # The contents of <nowiki> are literal text: do not parse them.
            pieces.append(raw)
            continue

        if kind == 'parameter':
            name = detail
            if name in bindings:
                bound = bindings[name]
                if bound == '':
                    binding_states.append((name, 'bound-empty'))
                    pieces.append('')
                    note = f'{{{{{{{name}}}}}}} is bound to an empty string'
                    if obj.default is not None:
                        note += '; its default is not used'
                    notes.append(note)
                else:
                    binding_states.append((name, 'bound'))
                    notes.append(f'{{{{{{{name}}}}}}} is bound to {bound!r}')
                    nested = _resolve_fragment(bound, bindings, None)
                    pieces.append(nested.text)
                    constructs.extend(nested.constructs)
                    unresolved.extend(nested.unresolved)
                    binding_states.extend(nested.bindings)
                    parameter_defaults_used.extend(
                        nested.parameter_defaults_used
                    )
                    notes.extend(nested.notes)
                continue

            default = obj.default
            if default is None:
                unresolved.append(name)
                binding_states.append((name, 'unbound'))
                notes.append(
                    f'{{{{{{{name}}}}}}} is unbound and has no default'
                )
                pieces.append(raw)
                continue

            binding_states.append((name, 'unbound-default'))
            default_start = start + len(raw) - len(default) - 3
            nested = _resolve_fragment(
                default,
                bindings,
                offset + default_start if offset is not None else None,
            )
            pieces.append(nested.text)
            constructs.extend(nested.constructs)
            unresolved.extend(nested.unresolved)
            binding_states.extend(nested.bindings)
            parameter_defaults_used.extend(nested.parameter_defaults_used)
            parameter_defaults_used.append(name)
            notes.extend(nested.notes)
            notes.append(
                f'used the default of {{{{{{{name}}}}}}} because it is unbound'
            )
            continue

        constructs.append(
            Construct(kind, raw, _absolute(span, offset), detail)
        )
        pieces.append(raw)

    if position < len(text):
        literal = text[position:]
        pieces.append(literal)
        _collect_signatures(
            literal,
            offset + position if offset is not None else None,
            constructs,
        )

    return _Fragment(
        ''.join(pieces),
        constructs,
        unresolved,
        binding_states,
        parameter_defaults_used,
        notes,
    )


def _status_and_value(fragment: _Fragment) -> tuple[FieldStatus, str | None]:
    # An unbound parameter without a default makes the value unknowable even
    # when a template is present: expansion alone would not fill it in.
    if fragment.unresolved:
        return FieldStatus.UNBOUND, None
    if fragment.constructs:
        return FieldStatus.EXPANSION_DEPENDENT, fragment.text
    return FieldStatus.LITERAL, fragment.text


def _single_parameter(text: str, items: list[tuple]) -> tuple | None:
    if len(items) != 1 or items[0][0] != 'parameter':
        return None
    start, end = items[0][1]
    outside = COMMENT_SUB('', text[:start] + text[end:])
    if outside.strip(WS):
        return None
    return items[0]


def resolve_value(
    text: str,
    *,
    bindings: Mapping[str, str] | None = None,
) -> Resolution:
    """Resolve a wikitext fragment without expanding templates.

    ``{{{parameter|default}}}`` constructs are substituted from ``bindings``.
    A name that is absent from ``bindings`` is unbound: its default (if any)
    is used.  A name mapped to ``''`` is bound to the empty string: its
    default is **not** used.

    Templates, parser functions, magic words, signatures, and extension tags
    other than ``<nowiki>`` are reported as expansion-dependent.  Text inside
    ``<nowiki>`` is kept literally.
    """
    if not isinstance(text, str):
        raise TypeError('text must be a string')
    binding_map = _validate_bindings(bindings)
    fragment = _resolve_fragment(text, binding_map, 0)
    status, value = _status_and_value(fragment)
    if value is not None:
        value = COMMENT_SUB('', value)

    binding_state = None
    single = _single_parameter(text, _outermost_items(WikiText(text)))
    if single is not None:
        name = single[3]
        if name in binding_map:
            binding_state = (
                'bound-empty' if binding_map[name] == '' else 'bound'
            )
        else:
            binding_state = 'unbound'

    notes = list(fragment.notes)
    if fragment.constructs:
        notes.append(
            'depends on external expansion: '
            + ', '.join(c.text for c in fragment.constructs)
        )
    if status is FieldStatus.LITERAL:
        notes.append('literal text')
    if COMMENT_SUB('', text) != text:
        notes.append('comments removed')

    return Resolution(
        status=status,
        value=value,
        raw=text,
        constructs=tuple(fragment.constructs),
        notes=tuple(notes),
        binding=binding_state,
        bindings=tuple(fragment.bindings),
        parameter_defaults_used=tuple(fragment.parameter_defaults_used),
    )


def _note_character_references(value: str | None, notes: list[str]) -> None:
    if value and CHARACTER_REFERENCE_SEARCH(value):
        notes.append(
            'contains HTML character references; store the source form or '
            'decode before comparing'
        )


def _find_argument(
    template: Template, parameters: Iterable[str]
) -> tuple[Any, str] | None:
    for name in parameters:
        if not name:
            continue
        argument = template.get_arg(name)
        if argument is not None:
            return argument, name
    return None


def _value_span(argument: Any) -> tuple[int, int]:
    start = argument.span[0]
    if argument.positional:
        return start + 1, start + len(argument.string)
    value_start = start + len(argument.string) - len(argument.value)
    return value_start, start + len(argument.string)


def _append_status_notes(
    status: FieldStatus, constructs: Sequence[Construct], notes: list[str]
) -> None:
    if constructs:
        notes.append(
            'depends on external expansion: '
            + ', '.join(c.text for c in constructs)
        )
    if status is FieldStatus.LITERAL:
        notes.append('literal value')
    elif status is FieldStatus.UNBOUND:
        notes.append('value cannot be determined without a binding')


def _extract_field(
    template: Template,
    template_name: str,
    field_name: str,
    spec: FieldSpec,
    bindings: Mapping[str, str],
    occurrences: int,
) -> FieldValue:
    notes: list[str] = []
    display_name = '{{' + template.name.strip(WS) + '}}'
    if occurrences > 1:
        notes.append(
            f'{display_name} occurs {occurrences} times; '
            f'used the last invocation'
        )

    found = _find_argument(template, spec.parameters)
    if found is not None:
        argument, used_name = found
        raw = argument.value
        value_span = _value_span(argument)
        keyword = not argument.positional

        origin = f'{display_name} argument {used_name!r}'
        if not keyword:
            origin += ' (positional)'
        notes.insert(0, origin)

        cleaned = COMMENT_SUB('', raw)
        empty = cleaned.strip(WS) == '' if keyword else cleaned == ''
        fragment = _resolve_fragment(raw, bindings, value_span[0])
        if empty:
            notes.append('argument is explicitly bound to an empty string')
            value = ''
            status = FieldStatus.LITERAL
        else:
            status, value = _status_and_value(fragment)
            if value is not None:
                value = COMMENT_SUB('', value)
                if keyword:
                    value = value.strip(WS)

        notes.extend(fragment.notes)
        if cleaned != raw:
            notes.append('comments removed')
        _note_character_references(value, notes)
        _append_status_notes(status, fragment.constructs, notes)

        return FieldValue(
            field=field_name,
            status=status,
            value=value,
            raw=raw,
            template=template_name,
            parameter=used_name,
            span=value_span,
            argument_span=argument.span,
            argument_present=True,
            bindings=tuple(fragment.bindings),
            parameter_defaults_used=tuple(fragment.parameter_defaults_used),
            constructs=tuple(fragment.constructs),
            notes=tuple(notes),
        )

    if spec.default is None:
        notes.append(
            f'{display_name} has no argument in {spec.parameters!r} '
            f'and no declared default'
        )
        return FieldValue(
            field=field_name,
            status=FieldStatus.ABSENT,
            value=None,
            raw=None,
            template=template_name,
            notes=tuple(notes),
        )

    notes.append(
        f'{display_name} has no argument in {spec.parameters!r}; '
        f'used the declared default'
    )
    fragment = _resolve_fragment(spec.default, bindings, None)
    status, value = _status_and_value(fragment)
    if value is not None:
        value = COMMENT_SUB('', value).strip(WS)
    notes.extend(fragment.notes)
    _note_character_references(value, notes)
    _append_status_notes(status, fragment.constructs, notes)

    return FieldValue(
        field=field_name,
        status=status,
        value=value,
        raw=spec.default,
        template=template_name,
        schema_default_used=True,
        bindings=tuple(fragment.bindings),
        parameter_defaults_used=tuple(fragment.parameter_defaults_used),
        constructs=tuple(fragment.constructs),
        notes=tuple(notes),
    )


def _absent_field(
    template_name: str, field_name: str, spec: FieldSpec
) -> FieldValue:
    notes = [f'{{{{{template_name}}}}} not found in source']
    if spec.default is not None:
        notes.append('no declared default is used without the template')
    return FieldValue(
        field=field_name,
        status=FieldStatus.ABSENT,
        value=None,
        raw=None,
        template=template_name,
        notes=tuple(notes),
    )


def extract_fields(
    source: str,
    schema: TemplateSchema,
    *,
    bindings: Mapping[str, str] | None = None,
    template: str | None = None,
) -> FieldReport:
    """Extract field values from template invocations in ``source``.

    :param source: the wikitext to analyze (usually an article).
    :param schema: the known template/parameter convention.
    :param bindings: transclusion frame for ``{{{parameter}}}`` constructs.
        Omit a name to leave it unbound (defaults apply); map it to ``''`` to
        bind it to the empty string (defaults are suppressed).
    :param template: analyze only this template from ``schema``.

    When a template is invoked more than once, the last invocation is used
    and the field's basis records that fact.
    """
    if not isinstance(schema, TemplateSchema):
        raise TypeError('schema must be a TemplateSchema')
    if not isinstance(source, str):
        raise TypeError('source must be a string')
    if not isinstance(schema, TemplateSchema):
        raise TypeError('schema must be a TemplateSchema')
    if not isinstance(source, str):
        raise TypeError('source must be a string')
    return _extract_from_parsed(WikiText(source), schema, bindings, template)


def _extract_from_parsed(
    parsed: WikiText,
    schema: TemplateSchema,
    bindings: Mapping[str, str] | None,
    template: str | None,
) -> FieldReport:
    """Implement extraction for an already-parsed source."""
    binding_map = _validate_bindings(bindings)

    only = None
    if template is not None:
        only = _normalize_template_name(template)
        if only not in schema.templates:
            raise ValueError(f'template {template!r} is not in the schema')

    invocations: dict[str, list[Template]] = {}
    for invocation in parsed.templates:
        key = _safe_normal_name(invocation)
        if key is None or key not in schema.templates:
            continue
        invocations.setdefault(key, []).append(invocation)

    fields: list[FieldValue] = []
    for name, field_specs in schema.templates.items():
        if only is not None and name != only:
            continue
        matching = invocations.get(name, [])
        for field_name, spec in field_specs.items():
            if not matching:
                fields.append(_absent_field(name, field_name, spec))
                continue
            fields.append(
                _extract_field(
                    matching[-1],
                    name,
                    field_name,
                    spec,
                    binding_map,
                    len(matching),
                )
            )
    return FieldReport(tuple(fields))


def _safe_normal_name(template: Template) -> str | None:
    try:
        return template.normal_name(capitalize=True)
    except Exception:
        return None


__all__ = [
    'Construct',
    'FieldReport',
    'FieldSpec',
    'FieldStatus',
    'FieldValue',
    'Resolution',
    'TemplateSchema',
    'extract_fields',
    'resolve_value',
]
