"""Schema migration support for records parsed by :mod:`dissect.cstruct`.

A firmware update may change the *interpretation* of fields without changing the byte span of a record: enum
members can be renamed, a raw integer can become an enum, length expressions can be rewritten and the target
type of a pointer can gain fields. Reusing the old parsed business values for such records is wrong even when
the structure size and every field offset are identical.

The tools in this module implement a three step workflow:

1. :func:`schema_signature` / :func:`impact_report` build an impact inventory of an old and a new schema,
   covering field layout, length expressions, enum members and pointer target dependencies.
2. :func:`replay_records` replays only the records that are actually affected: layout, enum and pointer
   impacts affect every record, while length expression impacts are evaluated per record against the parsed
   controlling fields.
3. :func:`verify_replay` re-parses every record with the new schema and compares that full parse against the
   incremental replay result, proving that no impact was missed and that no old business value was reused.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum as StdEnum
from io import BytesIO
from typing import TYPE_CHECKING, cast

from dissect.cstruct.expression import Expression
from dissect.cstruct.lexer import IDENTIFIER_TYPES
from dissect.cstruct.types.base import EOF, BaseArray
from dissect.cstruct.types.enum import EnumMetaType
from dissect.cstruct.types.flag import Flag
from dissect.cstruct.types.pointer import Pointer
from dissect.cstruct.types.structure import Structure, StructureMetaType, UnionMetaType, UnionProxy

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from dissect.cstruct.cstruct import cstruct
    from dissect.cstruct.types.base import BaseType
    from dissect.cstruct.types.structure import Field

# Impact reasons. A single field can carry more than one of these.
IMPACT_LAYOUT = "layout"
"""The field offset, bit width, size, type identity or field set changed."""

IMPACT_LENGTH = "length_expression"
"""An array length expression changed; the effect depends on the parsed controlling fields of each record."""

IMPACT_ENUM = "enum"
"""The enum or flag members behind a field changed, so a raw value gets a different business meaning."""

IMPACT_POINTER = "pointer"
"""A pointer target type changed, affecting values reachable through the pointer."""

_ALL_REASONS = frozenset((IMPACT_LAYOUT, IMPACT_LENGTH, IMPACT_ENUM, IMPACT_POINTER))

_ANONYMOUS = "<anonymous>"


@dataclass(frozen=True)
class DimSignature:
    """Signature of one array dimension.

    Args:
        kind: One of ``static``, ``nullterm``, ``expr`` or ``eof``.
        expression: The source of the length expression, if any.
        tokens: The normalized token stream of the expression, used for comparison.
        identifiers: The identifiers referenced by the expression.
        constants: The constants referenced by the expression and their values at signature time.
        count: The static entry count, if any.
    """

    kind: str
    expression: str | None
    tokens: tuple[tuple[str, str], ...]
    identifiers: tuple[str, ...]
    constants: tuple[tuple[str, object], ...]
    count: int | None


@dataclass(frozen=True)
class TypeSignature:
    """Structural signature of a field type, recursively including nested fields and enum members."""

    kind: str
    name: str
    size: int | None
    align: bool = False
    fields: tuple[FieldSignature, ...] = ()
    enum_members: tuple[tuple[str, int], ...] = ()
    underlying: str | None = None


@dataclass(frozen=True)
class FieldSignature:
    """Signature of a single structure field."""

    name: str
    type: TypeSignature
    pointer_depth: int
    dimensions: tuple[DimSignature, ...]
    bits: int | None
    alignment: int
    offset: int | None


@dataclass(frozen=True)
class SchemaSignature:
    """Fingerprint of a record type within a cstruct instance."""

    type_name: str
    root: TypeSignature


@dataclass(frozen=True)
class FieldImpact:
    """A schema difference localized at a dotted field path.

    Args:
        path: Dotted field path inside the record type.
        reasons: One or more of the ``IMPACT_*`` constants.
        detail: Human readable description of the difference.
    """

    path: str
    reasons: tuple[str, ...]
    detail: str


@dataclass(frozen=True)
class ImpactReport:
    """The impact inventory of comparing an old and a new schema for one record type."""

    record_type: str
    old_size: int | None
    new_size: int | None
    impacts: tuple[FieldImpact, ...]

    @property
    def size_unchanged(self) -> bool:
        """Whether the (static) total size of the record type is identical in both schemas."""
        return self.old_size == self.new_size

    @property
    def affected_paths(self) -> frozenset[str]:
        """All field paths that carry at least one impact."""
        return frozenset(impact.path for impact in self.impacts)

    @property
    def unconditional_paths(self) -> frozenset[str]:
        """Field paths whose interpretation changed for every record, regardless of parsed values."""
        return frozenset(
            impact.path for impact in self.impacts if any(reason != IMPACT_LENGTH for reason in impact.reasons)
        )

    @property
    def length_paths(self) -> frozenset[str]:
        """Field paths whose only possible impact is a record dependent length expression change."""
        return frozenset(
            impact.path
            for impact in self.impacts
            if IMPACT_LENGTH in impact.reasons and not any(reason != IMPACT_LENGTH for reason in impact.reasons)
        )

    @property
    def affected(self) -> bool:
        """Whether the schema change affects this record type at all."""
        return bool(self.impacts)


@dataclass(frozen=True)
class ValueDifference:
    """A normalized value difference at a dotted path."""

    path: str
    old: object
    new: object


@dataclass(frozen=True)
class ReplayResult:
    """The result of replaying one affected record under the new schema."""

    index: int
    raw: bytes
    old_parsed: object
    new_parsed: object
    differences: tuple[ValueDifference, ...]

    @property
    def changed(self) -> bool:
        """Whether the new schema derives different business values from the same bytes."""
        return bool(self.differences)


@dataclass(frozen=True)
class VerifyReport:
    """Outcome of comparing the incremental replay against a full new-schema parse."""

    record_type: str
    replayed: int
    skipped: int
    mismatches: tuple[ValueDifference, ...]

    @property
    def ok(self) -> bool:
        """Whether the incremental result is equivalent to a full parse of every record."""
        return not self.mismatches


@dataclass(frozen=True)
class MigrationResult:
    """Bundle of the impact inventory, the selective replay and its verification."""

    report: ImpactReport
    replays: tuple[ReplayResult, ...]
    verification: VerifyReport


def _unwrap(type_: type[BaseType]) -> tuple[type[BaseType], int, list[int | Expression | None]]:
    dimensions: list[int | Expression | None] = []
    while issubclass(type_, BaseArray):
        dimensions.append(type_.__num_entries__)
        type_ = type_.__type__

    pointer_depth = 0
    while issubclass(type_, Pointer):
        pointer_depth += 1
        type_ = type_.__type__

    return type_, pointer_depth, dimensions


def _type_kind(type_: type[BaseType]) -> str:
    if isinstance(type_, UnionMetaType):
        return "union"
    if isinstance(type_, StructureMetaType):
        return "struct"
    if isinstance(type_, EnumMetaType):
        return "flag" if issubclass(type_, Flag) else "enum"
    return "primitive"


def _type_name(type_: type[BaseType]) -> str:
    return _ANONYMOUS if getattr(type_, "__anonymous__", False) else type_.__name__


def _expr_identifiers(expression: Expression) -> tuple[str, ...]:
    return tuple(dict.fromkeys(token.value for token in expression._tokens if token.type in IDENTIFIER_TYPES))


def _dim_signature(dim: int | Expression | None, cs: cstruct) -> DimSignature:
    if dim is None:
        return DimSignature("nullterm", None, (), (), (), None)
    if isinstance(dim, Expression):
        kind = "eof" if dim.expression == "EOF" else "expr"
        tokens = tuple((token.type.name, token.value) for token in dim._tokens)
        identifiers = _expr_identifiers(dim)
        constants = tuple((name, cs.consts[name]) for name in identifiers if name in cs.consts)
        return DimSignature(kind, dim.expression, tokens, identifiers, constants, None)
    if dim == EOF:
        return DimSignature("eof", None, (), (), (), dim)
    return DimSignature("static", None, (), (), (), dim)


def _field_signature(
    field: Field, cs: cstruct, memo: dict[type[BaseType], TypeSignature], stack: frozenset[type[BaseType]]
) -> FieldSignature:
    base, pointer_depth, dimensions = _unwrap(field.type)
    return FieldSignature(
        name=field._name,
        type=_type_signature(base, cs, memo, stack),
        pointer_depth=pointer_depth,
        dimensions=tuple(_dim_signature(dim, cs) for dim in dimensions),
        bits=field.bits,
        alignment=field.alignment,
        offset=field.offset,
    )


def _structure_field_signatures(
    type_: type[BaseType], cs: cstruct, memo: dict[type[BaseType], TypeSignature], stack: frozenset[type[BaseType]]
) -> tuple[FieldSignature, ...]:
    signatures: list[FieldSignature] = []
    for field in type_.__fields__:
        signature = _field_signature(field, cs, memo, stack)
        base, pointer_depth, _ = _unwrap(field.type)
        # Flatten anonymous struct/union fields: parsed values are exposed on the parent structure directly, so
        # the folded field paths must match ``StructureMetaType.__members__``.
        if field.name is None and pointer_depth == 0 and isinstance(base, StructureMetaType):
            signatures.extend(signature.type.fields)
        else:
            signatures.append(signature)
    return tuple(signatures)


def _type_signature(
    type_: type[BaseType],
    cs: cstruct,
    memo: dict[type[BaseType], TypeSignature],
    stack: frozenset[type[BaseType]],
) -> TypeSignature:
    if type_ in memo:
        return memo[type_]
    if type_ in stack:
        # Self or cyclic reference: the first occurrence carries the full expanded signature.
        return TypeSignature(_type_kind(type_), _type_name(type_), getattr(type_, "__size__", None))

    kind = _type_kind(type_)
    if kind in ("struct", "union"):
        signature = TypeSignature(
            kind,
            _type_name(type_),
            type_.__size__,
            align=type_.__align__,
            fields=_structure_field_signatures(type_, cs, memo, stack | {type_}),
        )
    elif kind in ("enum", "flag"):
        members = tuple(sorted((name, member.value) for name, member in type_.__members__.items()))
        signature = TypeSignature(
            kind,
            _type_name(type_),
            type_.__size__,
            enum_members=members,
            underlying=type_.__type__.__name__,
        )
    else:
        signature = TypeSignature(kind, _type_name(type_), type_.__size__)

    memo[type_] = signature
    return signature


def schema_signature(cs: cstruct, type_name: str) -> SchemaSignature:
    """Build a fingerprint of a record type and every type reachable from its fields.

    The fingerprint captures field layout (offsets, bit widths, alignment), array length expressions and their
    referenced constants, enum/flag members and pointer target types, so that two schemas with an identical
    total size can still be told apart by their field semantics.

    Args:
        cs: The cstruct instance holding the schema.
        type_name: The name of the record type.

    Returns:
        A frozen :class:`SchemaSignature` suitable for equality comparison.
    """
    root = cs.resolve(type_name)
    memo: dict[type[BaseType], TypeSignature] = {}
    return SchemaSignature(type_name, _type_signature(root, cs, memo, frozenset()))


def _enum_member_changes(old: TypeSignature, new: TypeSignature) -> str:
    old_by_value = {value: name for name, value in old.enum_members}
    new_by_value = {value: name for name, value in new.enum_members}
    changes = []
    for value in sorted(old_by_value.keys() | new_by_value.keys()):
        old_name = old_by_value.get(value)
        new_name = new_by_value.get(value)
        if old_name != new_name:
            changes.append(f"value {value:#x}: {old_name!r} -> {new_name!r}")
    return ", ".join(changes)


def _diff_dimensions(old: FieldSignature, new: FieldSignature) -> tuple[str, ...]:
    reasons = []
    for old_dim, new_dim in zip(old.dimensions, new.dimensions, strict=True):
        if old_dim == new_dim:
            continue
        if old_dim.kind == "expr" or new_dim.kind == "expr":
            reasons.append(IMPACT_LENGTH)
        else:
            reasons.append(IMPACT_LAYOUT)
    return tuple(reasons)


def _diff_type(old: TypeSignature, new: TypeSignature, path: str) -> list[tuple[str, tuple[str, ...], str]]:
    out: list[tuple[str, tuple[str, ...], str]] = []
    prefix = f"{path}." if path else ""

    if old.kind != new.kind or old.name != new.name:
        reasons = [IMPACT_LAYOUT]
        if "enum" in (old.kind, new.kind) or "flag" in (old.kind, new.kind):
            reasons.append(IMPACT_ENUM)
        out.append((path, tuple(reasons), f"type {old.kind} {old.name!r} -> {new.kind} {new.name!r}"))
        return out

    if old.kind in ("enum", "flag"):
        if old.underlying != new.underlying:
            out.append((path, (IMPACT_LAYOUT,), f"underlying type {old.underlying!r} -> {new.underlying!r}"))
        if old.enum_members != new.enum_members:
            out.append((path, (IMPACT_ENUM,), f"members changed ({_enum_member_changes(old, new)})"))
        return out

    if old.size != new.size:
        out.append((path, (IMPACT_LAYOUT,), f"size {old.size} -> {new.size}"))

    if old.kind in ("struct", "union"):
        old_fields = {field.name: field for field in old.fields}
        new_fields = {field.name: field for field in new.fields}
        out.extend(
            (f"{prefix}{name}", (IMPACT_LAYOUT,), "field removed in new schema")
            for name in sorted(old_fields.keys() - new_fields.keys())
        )
        out.extend(
            (f"{prefix}{name}", (IMPACT_LAYOUT,), "field added in new schema")
            for name in sorted(new_fields.keys() - old_fields.keys())
        )
        for name in sorted(old_fields.keys() & new_fields.keys()):
            out.extend(_diff_field(old_fields[name], new_fields[name], f"{prefix}{name}"))

    return out


def _diff_field(old: FieldSignature, new: FieldSignature, path: str) -> list[tuple[str, tuple[str, ...], str]]:
    out: list[tuple[str, tuple[str, ...], str]] = []
    reasons: list[str] = []

    if old.bits != new.bits or old.alignment != new.alignment or old.offset != new.offset:
        reasons.append(IMPACT_LAYOUT)
    if old.pointer_depth != new.pointer_depth:
        reasons.append(IMPACT_LAYOUT)
    if len(old.dimensions) != len(new.dimensions):
        reasons.append(IMPACT_LAYOUT)
    else:
        reasons.extend(_diff_dimensions(old, new))

    nested = _diff_type(old.type, new.type, path)
    if nested and old.pointer_depth:
        reasons.append(IMPACT_POINTER)

    if reasons:
        detail = f"field signature changed (offset {old.offset} -> {new.offset}, bits {old.bits} -> {new.bits})"
        out.append((path, tuple(dict.fromkeys(reasons)), detail))
    out.extend(nested)
    return out


def impact_report(old_cs: cstruct, new_cs: cstruct, type_name: str) -> ImpactReport:
    """Compare two schema versions of a record type and return its impact inventory.

    Args:
        old_cs: The cstruct instance holding the old schema.
        new_cs: The cstruct instance holding the new schema.
        type_name: The record type name, present in both instances.

    Returns:
        An :class:`ImpactReport` listing every affected field path with the reasons for the impact. The report
        distinguishes unconditional impacts (layout, enum and pointer changes, which affect every record) from
        length expression impacts, which must be evaluated against each record's controlling field values.
        The total structure size is reported separately, because an unchanged byte span never rules out an
        impact on field semantics.
    """
    old_signature = schema_signature(old_cs, type_name)
    new_signature = schema_signature(new_cs, type_name)

    merged: dict[str, tuple[list[str], list[str]]] = {}
    for path, reasons, detail in _diff_type(old_signature.root, new_signature.root, ""):
        entry = merged.setdefault(path, ([], []))
        entry[0].extend(reasons)
        entry[1].append(detail)

    impacts = tuple(
        FieldImpact(path, tuple(dict.fromkeys(reasons)), "; ".join(dict.fromkeys(details)))
        for path, (reasons, details) in merged.items()
    )
    old_cls = old_cs.resolve(type_name)
    new_cls = new_cs.resolve(type_name)
    return ImpactReport(type_name, old_cls.__size__, new_cls.__size__, impacts)


def _resolve_field(struct_cls: type[BaseType], path: str) -> Field:
    current: type[BaseType] = struct_cls
    field: Field | None = None
    for part in path.split("."):
        field = current.__members__[part]
        current, _, _ = _unwrap(field.type)
    assert field is not None
    return field


def _contexts_at(obj: Structure, path: str) -> list[dict[str, object]]:
    contexts: list[Structure] = [obj]
    for part in path.split(".")[:-1]:
        next_contexts: list[Structure] = []
        for current in contexts:
            value = getattr(current, part)
            if isinstance(value, list):
                next_contexts.extend(value)
            else:
                next_contexts.append(value)
        contexts = next_contexts
    return [dict(context.__values__) for context in contexts]


def _array_counts(type_: type[BaseType], context: dict[str, object]) -> tuple[int | None, ...]:
    counts: list[int | None] = []
    while issubclass(type_, BaseArray):
        entries = type_.__num_entries__
        if isinstance(entries, Expression):
            try:
                # Expressions resolve controlling fields through the parsed record context, which holds values
                # of arbitrary types even though only integer fields participate in arithmetic.
                counts.append(entries.evaluate(type_.__cs__, cast("dict[str, int]", context)))
            except Exception:
                if entries.expression == "EOF":
                    counts.append(EOF)
                else:
                    raise
        elif entries is None:
            counts.append(None)
        else:
            counts.append(entries)
        type_ = type_.__type__
    return tuple(counts)


def _dynamic_length_paths(
    struct_cls: type[BaseType], prefix: str = "", seen: frozenset[type[BaseType]] = frozenset()
) -> set[str]:
    paths: set[str] = set()
    if struct_cls in seen:
        return paths
    seen = seen | {struct_cls}

    for field in struct_cls.__fields__:
        base, pointer_depth, dimensions = _unwrap(field.type)
        if field.name is None:
            if pointer_depth == 0 and isinstance(base, StructureMetaType):
                paths.update(_dynamic_length_paths(base, prefix, seen))
            continue

        path = f"{prefix}{field._name}"
        if any(isinstance(dim, Expression) for dim in dimensions):
            paths.add(path)
        # Pointer targets live outside the record bytes, so only directly embedded structures are descended into.
        if pointer_depth == 0 and isinstance(base, StructureMetaType):
            paths.update(_dynamic_length_paths(base, f"{path}.", seen))
    return paths


def _record_affected(
    old_obj: Structure, report: ImpactReport, old_cls: type[BaseType], new_cls: type[BaseType]
) -> bool:
    if report.unconditional_paths:
        return True

    # Evaluate every dynamic array length under both schemas. This catches not only rewritten expressions but
    # also constant and sizeof changes that leave the expression tokens untouched. If an expression cannot be
    # evaluated for a record, fail safe and replay the record rather than risk keeping a stale business value.
    try:
        for path in _dynamic_length_paths(old_cls) | _dynamic_length_paths(new_cls):
            old_field = _resolve_field(old_cls, path)
            new_field = _resolve_field(new_cls, path)
            for context in _contexts_at(old_obj, path):
                if _array_counts(old_field.type, context) != _array_counts(new_field.type, context):
                    return True
    except Exception:
        return True
    return False


def _normalize(value: object) -> object:
    if isinstance(value, UnionProxy):
        value = value.__target__
    if isinstance(value, Structure):
        return {name: _normalize(getattr(value, name)) for name in value.__class__.__members__}
    if isinstance(value, StdEnum):
        return ("enum", value.__class__.__name__, int(value), value.name)
    if isinstance(value, Pointer):
        target = None
        if int(value) != 0:
            try:
                target = _normalize(value.dereference())
            except Exception as exc:
                # A missing stream or a truncated target must not break the audit; keep the failure visible.
                target = ("<deref error>", type(exc).__name__)
        return ("pointer", int(value), target)
    if isinstance(value, list):
        return [_normalize(entry) for entry in value]
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    return value


def _diff_values(path: str, old: object, new: object, out: list[ValueDifference]) -> None:
    old_normalized = _normalize(old)
    new_normalized = _normalize(new)
    prefix = f"{path}." if path else ""

    if isinstance(old_normalized, dict) or isinstance(new_normalized, dict):
        old_dict = old_normalized if isinstance(old_normalized, dict) else {}
        new_dict = new_normalized if isinstance(new_normalized, dict) else {}
        for name in sorted(old_dict.keys() | new_dict.keys()):
            if name in old_dict and name in new_dict:
                _diff_values(f"{prefix}{name}", old_dict[name], new_dict[name], out)
            else:
                out.append(ValueDifference(f"{prefix}{name}", old_dict.get(name), new_dict.get(name)))
    elif isinstance(old_normalized, list) or isinstance(new_normalized, list):
        old_list = old_normalized if isinstance(old_normalized, list) else []
        new_list = new_normalized if isinstance(new_normalized, list) else []
        if len(old_list) != len(new_list):
            out.append(ValueDifference(path, f"len={len(old_list)}", f"len={len(new_list)}"))
        for index, (old_entry, new_entry) in enumerate(zip(old_list, new_list, strict=False)):
            _diff_values(f"{prefix}[{index}]", old_entry, new_entry, out)
    elif old_normalized != new_normalized:
        out.append(ValueDifference(path, old_normalized, new_normalized))


def compare_values(old: object, new: object) -> tuple[ValueDifference, ...]:
    """Compare two parsed records (or field values) by their normalized business values.

    Enum values keep their member name, so a renamed enum member is reported even when the raw integer and the
    byte span are unchanged. Pointer values include the normalized dereferenced target.

    Args:
        old: The value parsed by the old schema.
        new: The value parsed by the new schema.

    Returns:
        A tuple of :class:`ValueDifference` entries; empty when the business values are equivalent.
    """
    differences: list[ValueDifference] = []
    _diff_values("", old, new, differences)
    return tuple(differences)


def replay_records(
    old_cs: cstruct,
    new_cs: cstruct,
    type_name: str,
    records: Sequence[bytes | bytearray | memoryview],
    report: ImpactReport | None = None,
) -> tuple[ReplayResult, ...]:
    """Replay only the records affected by the schema change.

    Layout, enum and pointer impacts affect every record. A length expression impact only replays records for
    which the old and new expression evaluate to different array sizes given the record's controlling fields.
    Every replayed record is parsed fresh under the new schema; old parsed values are never reused.

    Args:
        old_cs: The cstruct instance holding the old schema.
        new_cs: The cstruct instance holding the new schema.
        type_name: The record type name.
        records: The raw bytes of the stored records.
        report: An optional precomputed :class:`ImpactReport`.

    Returns:
        One :class:`ReplayResult` per affected record, in record order.
    """
    report = report or impact_report(old_cs, new_cs, type_name)
    old_cls = old_cs.resolve(type_name)
    new_cls = new_cs.resolve(type_name)

    results = []
    for index, raw in enumerate(records):
        raw_bytes = bytes(raw)
        old_obj = old_cls(BytesIO(raw_bytes))
        if not _record_affected(old_obj, report, old_cls, new_cls):
            continue
        new_obj = new_cls(BytesIO(raw_bytes))
        results.append(ReplayResult(index, raw_bytes, old_obj, new_obj, compare_values(old_obj, new_obj)))
    return tuple(results)


def verify_replay(
    old_cs: cstruct,
    new_cs: cstruct,
    type_name: str,
    records: Sequence[bytes | bytearray | memoryview],
    replays: Sequence[ReplayResult] | None = None,
) -> VerifyReport:
    """Compare the incremental replay against a full new-schema parse of every record.

    Replayed records must match a fresh parse of their bytes. Skipped records are parsed as well to prove that
    the old and new schemas derive the same business values for them; any difference is reported as a missed
    impact rather than silently reused.

    Args:
        old_cs: The cstruct instance holding the old schema.
        new_cs: The cstruct instance holding the new schema.
        type_name: The record type name.
        records: The raw bytes of the stored records.
        replays: Optional precomputed replay results.

    Returns:
        A :class:`VerifyReport`; its ``ok`` property is ``True`` only when the incremental result is
        equivalent to a full parse.
    """
    report = impact_report(old_cs, new_cs, type_name)
    replays = tuple(replays) if replays is not None else replay_records(old_cs, new_cs, type_name, records, report)
    replayed = {result.index: result for result in replays}

    old_cls = old_cs.resolve(type_name)
    new_cls = new_cs.resolve(type_name)
    mismatches: list[ValueDifference] = []

    for index, raw in enumerate(records):
        raw_bytes = bytes(raw)
        full_parse = new_cls(BytesIO(raw_bytes))
        if index in replayed:
            if _normalize(replayed[index].new_parsed) != _normalize(full_parse):
                mismatches.append(ValueDifference(str(index), "replay differs from full parse", ""))
            continue

        old_obj = old_cls(BytesIO(raw_bytes))
        for difference in compare_values(old_obj, full_parse):
            path = f"[{index}].{difference.path}".rstrip(".")
            mismatches.append(ValueDifference(path, difference.old, difference.new))

    return VerifyReport(type_name, len(replayed), len(records) - len(replayed), tuple(mismatches))


def migrate(
    old_cs: cstruct,
    new_cs: cstruct,
    type_name: str,
    records: Sequence[bytes | bytearray | memoryview],
) -> MigrationResult:
    """Run the full workflow: impact inventory, selective replay and full-parse verification.

    Args:
        old_cs: The cstruct instance holding the old schema.
        new_cs: The cstruct instance holding the new schema.
        type_name: The record type name.
        records: The raw bytes of the stored records.

    Returns:
        A :class:`MigrationResult` bundling the :class:`ImpactReport`, the replay results and the
        :class:`VerifyReport`.
    """
    report = impact_report(old_cs, new_cs, type_name)
    replays = replay_records(old_cs, new_cs, type_name, records, report)
    verification = verify_replay(old_cs, new_cs, type_name, records, replays)
    return MigrationResult(report, replays, verification)


def iter_impacts(report: ImpactReport) -> Iterator[FieldImpact]:
    """Iterate the field impacts of an :class:`ImpactReport`."""
    return iter(report.impacts)


def impacts_by_reason(report: ImpactReport, reason: str) -> tuple[FieldImpact, ...]:
    """Return the impacts carrying a given ``IMPACT_*`` reason."""
    if reason not in _ALL_REASONS:
        raise ValueError(f"Unknown impact reason: {reason!r}")
    return tuple(impact for impact in report.impacts if reason in impact.reasons)
