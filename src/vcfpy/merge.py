# -*- coding: utf-8 -*-
"""Header-aware merging of VCF files from multiple sources.

Merging records is easy; merging their *meaning* is the hard part.  The same
INFO or FORMAT ID can be declared with different ``Number`` or ``Type``
values in two source files.  A concrete record whose value happens to have
the same length under both declarations does not make them compatible:
``Number=A`` promises one value per ALT allele for every record, whereas
``Number=1`` promises exactly one value.

This module separates planning from execution:

* :func:`plan_merge` inspects headers only and returns a :class:`MergePlan`.
  The plan lists incompatibilities, contains the merged header, and records
  every field rename.
* :meth:`MergePlan.convert_record` applies the planned renames to a record
  from one source.
* :func:`merge_records` and :func:`merge_vcf_files` execute a plan.

INFO and FORMAT are distinct namespaces in VCF.  ``INFO/DP`` and
``FORMAT/DP`` can therefore differ independently, and a rename is always
scoped: an INFO conflict never renames FORMAT values, and vice versa.

Renaming a conflicting field is the only operation that changes a record.
The renamed field keeps the original source's ``Number`` and ``Type`` in the
merged header, so values remain interpretable exactly as they were parsed.
No attempt is made to coerce ``Number=A`` values into ``Number=1`` values or
to truncate one value to fit the other.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TextIO

from vcfpy import exceptions
from vcfpy.header import (
    RESERVED_FORMAT,
    RESERVED_INFO,
    FieldInfo,
    FormatHeaderLine,
    Header,
    InfoHeaderLine,
    SamplesInfos,
)
from vcfpy.record import Call, Record, UnparsedCall

__all__ = [
    "FieldRename",
    "HeaderMergePlan",
    "MergeConflict",
    "MergeConflictError",
    "MergeError",
    "MergePlan",
    "RenameCollection",
    "apply_merge",
    "dry_run_merge",
    "execute_merge",
    "merge_header",
    "merge_headers",
    "merge_record",
    "merge_records",
    "merge_vcf",
    "merge_vcf_files",
    "plan_header_merge",
    "plan_merge",
    "preview_header_merge",
    "preview_merge",
]

Scope = Literal["INFO", "FORMAT"]
_SCOPES: tuple[Scope, Scope] = ("INFO", "FORMAT")

#: Field IDs may contain more characters than this in VCF itself.  Generated
#: IDs are deliberately conservative so they also work in shell pipelines.
_INVALID_ID_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")


class MergeError(exceptions.VCFPyException):
    """Base class for errors raised while planning or executing a merge."""


class MergeConflictError(MergeError):
    """Raised when incompatible headers cannot be merged unambiguously."""


def _canonical_number(number: Any) -> Any:
    """Return a canonical value for a VCF ``Number`` declaration.

    Parsed headers already store numeric values as ``int``.  Manually built
    headers may contain ``"1"`` instead of ``1``; both mean the same thing.
    """
    if isinstance(number, bool):  # pragma: no cover - defensive only
        return int(number)
    if isinstance(number, int):
        return number
    if isinstance(number, str):
        try:
            return int(number)
        except ValueError:
            return number
    return number


def _effective_field_info(line: InfoHeaderLine | FormatHeaderLine) -> FieldInfo:
    """Return the effective definition of an INFO/FORMAT header line."""
    info = FieldInfo(
        line.type,
        _canonical_number(line.number),
        line.mapping.get("Description"),
        line.mapping.get("ID"),
    )
    return info


def _signature(info: FieldInfo) -> tuple[Any, str]:
    """Return the part of a field definition that determines value semantics."""
    return (_canonical_number(info.number), info.type)


def _signature_reason(left: FieldInfo, right: FieldInfo) -> str:
    """Return a human-readable explanation of a signature mismatch."""
    left_number = _canonical_number(left.number)
    right_number = _canonical_number(right.number)
    if left_number != right_number and left.type != right.type:
        return "Number and Type differ ({}:{} vs {}:{})".format(left.number, left.type, right.number, right.type)
    if left_number != right_number:
        return "Number differs ({!r} vs {!r})".format(left.number, right.number)
    return "Type differs ({!r} vs {!r})".format(left.type, right.type)


def _sanitize_source_name(name: Any) -> str:
    """Return a source name that can be used as part of a VCF field ID."""
    result = _INVALID_ID_CHARS.sub("_", str(name)).strip("_")
    return result or "source"


def _as_source_index(source: int | str, source_names: Sequence[str]) -> int:
    """Resolve a source index or source name to an index."""
    if isinstance(source, int):
        if 0 <= source < len(source_names):
            return source
        raise MergeError("Source index {} out of range ({} sources)".format(source, len(source_names)))
    if source in source_names:
        return list(source_names).index(source)
    raise MergeError("Unknown source name {!r} (known: {})".format(source, ", ".join(source_names)))


@dataclass(frozen=True)
class FieldRename:
    """One field rename, scoped to a source and to INFO or FORMAT."""

    #: ``"INFO"`` or ``"FORMAT"``
    scope: Scope
    #: zero-based index of the source the field comes from
    source_index: int
    #: field ID as declared by the source header
    old_id: str
    #: field ID used in the merged header
    new_id: str
    #: why the rename was necessary
    reason: str = ""
    #: human-readable source name, if one was provided
    source_name: str | None = None

    @property
    def old_name(self) -> str:
        """Alias for :py:attr:`old_id`."""
        return self.old_id

    @property
    def new_name(self) -> str:
        """Alias for :py:attr:`new_id`."""
        return self.new_id

    @property
    def source(self) -> int:
        """Alias for :py:attr:`source_index`."""
        return self.source_index

    @property
    def source_number(self) -> int:
        """One-based source number, useful for display."""
        return self.source_index + 1

    def __iter__(self) -> Iterator[str]:
        """Allow ``old, new = rename`` unpacking."""
        yield self.old_id
        yield self.new_id

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "scope": self.scope,
            "source_index": self.source_index,
            "source_number": self.source_number,
            "source_name": self.source_name,
            "old_id": self.old_id,
            "new_id": self.new_id,
            "reason": self.reason,
        }


class RenameCollection(Sequence[FieldRename]):
    """Sequence of :class:`FieldRename` objects with convenient lookups.

    Iteration yields the individual :class:`FieldRename` objects.  A rename
    can also be looked up directly with ``renames[scope, source_index, old_id]``.
    """

    def __init__(self, renames: Iterable[FieldRename] = ()):
        self._renames = tuple(renames)
        self._lookup = {(r.scope, r.source_index, r.old_id): r.new_id for r in self._renames}

    def __getitem__(self, key: int | slice | tuple[Scope, int, str] | Scope) -> Any:
        if isinstance(key, tuple):
            scope, source, old_id = key
            return self._lookup[(scope, source, old_id)]
        if isinstance(key, str):
            return self.nested()[key]  # type: ignore[index]
        return self._renames[key]

    def __len__(self) -> int:
        return len(self._renames)

    def __iter__(self) -> Iterator[FieldRename]:
        return iter(self._renames)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return "RenameCollection({})".format(list(self._renames))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, RenameCollection):
            return self._renames == other._renames
        if isinstance(other, Mapping):
            return self._lookup == dict(other)
        if isinstance(other, (list, tuple)):
            return self._renames == tuple(other)
        return NotImplemented

    __hash__ = None  # type: ignore[assignment]

    def get(self, scope: Scope, source_index: int, old_id: str, default: Any = None) -> Any:
        """Return the new ID, or ``default`` if no rename is planned."""
        return self._lookup.get((scope, source_index, old_id), default)

    def for_source(self, source_index: int, scope: Scope | None = None) -> dict[str, str]:
        """Return ``{old_id: new_id}`` for one source and optional scope."""
        result: dict[str, str] = {}
        for rename in self._renames:
            if rename.source_index == source_index and (scope is None or rename.scope == scope):
                result[rename.old_id] = rename.new_id
        return result

    def as_dict(self) -> dict[tuple[Scope, int, str], str]:
        """Return ``{(scope, source_index, old_id): new_id}``."""
        return dict(self._lookup)

    def nested(self) -> dict[Scope, dict[int, dict[str, str]]]:
        """Return ``{scope: {source_index: {old_id: new_id}}}``."""
        result: dict[Scope, dict[int, dict[str, str]]] = {scope: {} for scope in _SCOPES}
        for rename in self._renames:
            result[rename.scope].setdefault(rename.source_index, {})[rename.old_id] = rename.new_id
        return result

    def items(self):
        """Return ``((scope, source_index, old_id), new_id)`` pairs."""
        return self._lookup.items()

    def keys(self):
        """Return ``(scope, source_index, old_id)`` keys."""
        return self._lookup.keys()

    def values(self):
        """Return target IDs."""
        return self._lookup.values()

    def __contains__(self, key: object) -> bool:
        if isinstance(key, tuple):
            return key in self._lookup
        return key in self._renames


@dataclass(frozen=True)
class MergeConflict:
    """A semantic incompatibility between field definitions of two sources."""

    #: ``"INFO"`` or ``"FORMAT"``
    scope: Scope
    #: field ID that is defined differently
    field_id: str
    #: one source index per distinct definition, in first-seen order
    source_indices: tuple[int, ...]
    #: effective definitions, parallel to :py:attr:`source_indices`
    definitions: tuple[FieldInfo, ...]
    #: human-readable explanation, e.g. ``"Number differs ('A' vs 1)"``
    reason: str
    #: IDs assigned to the non-primary definitions
    renamed_to: tuple[str, ...] = ()

    @property
    def id(self) -> str:
        """Alias for :py:attr:`field_id`."""
        return self.field_id

    @property
    def resolved(self) -> bool:
        """Return whether every conflicting definition was renamed."""
        return len(self.renamed_to) == len(self.definitions) - 1

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "scope": self.scope,
            "field_id": self.field_id,
            "source_indices": list(self.source_indices),
            "definitions": [
                {"type": info.type, "number": info.number, "description": info.description}
                for info in self.definitions
            ],
            "reason": self.reason,
            "renamed_to": list(self.renamed_to),
            "resolved": self.resolved,
        }


@dataclass
class _FieldOccurrence:
    """Internal record of one explicit INFO/FORMAT line occurrence."""

    scope: Scope
    source_index: int
    line: InfoHeaderLine | FormatHeaderLine
    info: FieldInfo
    target_id: str = ""


class MergePlan:
    """Plan for merging a fixed set of VCF headers.

    Creating a plan does not modify any source header.  Use
    :meth:`convert_record` to express a source record in the merged
    vocabulary, then write it with :py:attr:`header`.
    """

    def __init__(
        self,
        headers: Sequence[Header],
        source_names: Sequence[str],
        merged_header: Header,
        renames: Iterable[FieldRename],
        conflicts: Iterable[MergeConflict],
        header_conflicts: Iterable[dict[str, Any]] = (),
        unresolved: Iterable[str] = (),
        field_metadata_conflicts: Iterable[dict[str, Any]] = (),
    ):
        #: source headers, in input order
        self.headers = tuple(headers)
        #: names used to qualify renamed fields
        self.source_names = tuple(source_names)
        #: merged header to use when writing converted records
        self.header = merged_header
        #: planned field renames
        self.renames = RenameCollection(renames)
        #: all INFO/FORMAT incompatibilities, resolved or not
        self.conflicts = tuple(conflicts)
        #: non-field header line differences resolved by keeping the first line
        self.header_conflicts = tuple(header_conflicts)
        #: differences in Description/Source/Version for otherwise compatible
        #: fields; the first definition is kept
        self.field_metadata_conflicts = tuple(field_metadata_conflicts)
        #: conflicts that could not be represented unambiguously
        self.unresolved = tuple(unresolved)
        self._rename_lookup = self.renames.as_dict()
    @property
    def merged_header(self) -> Header:
        """Alias for :py:attr:`header`."""
        return self.header

    @property
    def compatible(self) -> bool:
        """Return whether all common fields have identical ``Number``/``Type``."""
        return not self.conflicts

    @property
    def requires_renames(self) -> bool:
        """Return whether executing the plan involves a field rename."""
        return bool(self.renames)

    @property
    def mergeable(self) -> bool:
        """Return whether the plan can be executed without ambiguity."""
        return not self.unresolved

    @property
    def rename_map(self) -> dict[tuple[Scope, int, str], str]:
        """Return ``{(scope, source_index, old_id): new_id}``."""
        return dict(self._rename_lookup)

    @property
    def rename_mapping(self) -> dict[tuple[Scope, int, str], str]:
        """Alias for :py:attr:`rename_map`."""
        return self.rename_map

    @property
    def info_renames(self) -> dict[int, dict[str, str]]:
        """Return ``{source_index: {old_id: new_id}}`` for INFO."""
        return self.renames.nested()["INFO"]

    @property
    def format_renames(self) -> dict[int, dict[str, str]]:
        """Return ``{source_index: {old_id: new_id}}`` for FORMAT."""
        return self.renames.nested()["FORMAT"]

    def renames_for(self, source: int | str, scope: Scope | None = None) -> dict[str, str]:
        """Return the rename mapping for one source and optional scope."""
        return self.renames.for_source(_as_source_index(source, self.source_names), scope)

    def rename_for(self, source: int | str, scope: Scope, field_id: str) -> str:
        """Return the merged ID for ``field_id`` from ``source``."""
        source_index = _as_source_index(source, self.source_names)
        return self._rename_lookup.get((scope, source_index, field_id), field_id)

    def merged_field_info(self, source: int | str, scope: Scope, field_id: str) -> FieldInfo:
        """Return the merged header's definition for a source field."""
        new_id = self.rename_for(source, scope, field_id)
        if scope == "INFO":
            return self.header.get_info_field_info(new_id)
        return self.header.get_format_field_info(new_id)

    def convert_record(
        self,
        record: Record,
        source: int | str,
        *,
        in_place: bool = False,
        validate: bool = True,
    ) -> Record:
        """Return ``record`` expressed in this plan's merged header.

        Only field names change.  Values, including whether they are scalars
        or lists, stay exactly as parsed from the source.  The merged header
        line for a renamed field retains the source's original ``Number`` and
        ``Type`` so that those values keep their original meaning.

        :param record: record from one of the plan's sources
        :param source: source index or source name
        :param in_place: modify ``record`` instead of returning a deep copy
        :param validate: verify that every field's source definition matches
            its definition in the merged header
        :raises MergeError: if a rename would create a duplicate field or if
            the field would change meaning
        :raises MergeConflictError: if the plan is not executable
        """
        source_index = _as_source_index(source, self.source_names)
        if not self.mergeable:
            raise MergeConflictError("; ".join(self.unresolved) or "Merge plan is not executable")
        result = record if in_place else copy.deepcopy(record)
        info_renames = self.renames.for_source(source_index, "INFO")
        format_renames = self.renames.for_source(source_index, "FORMAT")
        if validate:
            self._validate_record_semantics(record, source_index)
        if info_renames:
            result.INFO = self._rename_mapping_keys(result.INFO, info_renames, "INFO", record)
        if format_renames:
            result.FORMAT = self._rename_format(result, format_renames)
            for call in result.calls:
                if isinstance(call, Call):
                    call.data = self._rename_mapping_keys(call.data, format_renames, "FORMAT", record)
        return result

    #: Descriptive alias.
    transform_record = convert_record

    def convert_records(
        self, records: Iterable[Record], source: int | str, *, in_place: bool = False
    ) -> list[Record]:
        """Convert all ``records`` from one source."""
        return [self.convert_record(record, source, in_place=in_place) for record in records]

    def pad_record(self, record: Record) -> Record:
        """Add empty calls for samples present only in other sources.

        The merged header contains the union of all sample names.  VCF requires
        every record to carry a value for each sample, so missing samples are
        represented by ``"."`` columns.
        """
        names = self.header.samples.names if self.header.samples else []
        if not names or all(name in record.call_for_sample for name in names):
            return record
        if not record.FORMAT:
            record.FORMAT = ["GT"]
            for name in names:
                if name not in record.call_for_sample:
                    record.calls.append(UnparsedCall(name, "."))
        else:
            missing = [name for name in names if name not in record.call_for_sample]
            for name in missing:
                record.calls.append(UnparsedCall(name, ":".join("." for _ in record.FORMAT)))
        record.update_calls(record.calls)
        return record

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable preview of the plan."""
        return {
            "source_names": list(self.source_names),
            "compatible": self.compatible,
            "requires_renames": self.requires_renames,
            "mergeable": self.mergeable,
            "renames": [rename.as_dict() for rename in self.renames],
            "conflicts": [conflict.as_dict() for conflict in self.conflicts],
            "header_conflicts": list(self.header_conflicts),
            "field_metadata_conflicts": list(self.field_metadata_conflicts),
            "unresolved": list(self.unresolved),
        }

    def to_json(self, **kwargs: Any) -> str:
        """Return :meth:`to_dict` serialised as JSON."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, **kwargs)

    def __iter__(self) -> Iterator[Any]:
        """Allow ``merged_header, rename_map = merge_headers(...)`` unpacking."""
        yield self.header
        yield self.rename_map

    def __getattr__(self, name: str) -> Any:
        """Delegate read-only attributes to the merged header.

        ``plan`` can therefore be used anywhere a :class:`vcfpy.Header` is
        expected for writing, e.g. ``Writer.from_path(path, plan)``.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        header = self.__dict__.get("header")
        if header is None:  # pragma: no cover - during unpickling/initialisation
            raise AttributeError(name)
        return getattr(header, name)

    def require_mergeable(self) -> None:
        """Raise :class:`MergeConflictError` if the plan is not executable."""
        if not self.mergeable:
            raise MergeConflictError("; ".join(self.unresolved) or "Merge plan is not executable")

    def _source_field_info(self, scope: Scope, field_id: str, source_index: int) -> FieldInfo | None:
        """Return an explicitly declared source definition, if available."""
        header = self.headers[source_index]
        if scope == "INFO":
            line = header._indices.get("INFO", {}).get(field_id)
        else:
            line = header._indices.get("FORMAT", {}).get(field_id)
        if isinstance(line, (InfoHeaderLine, FormatHeaderLine)):
            return _effective_field_info(line)
        return None

    def _validate_record_semantics(self, record: Record, source_index: int) -> None:
        """Check that converting ``record`` preserves each field's meaning.

        A field may only be emitted under an ID whose merged definition has
        the same ``Number`` and ``Type`` as the definition the record was
        parsed with.  This catches, for example, an unexpected record from a
        source whose header was not part of this plan.
        """
        fields: list[tuple[Scope, str]] = [("INFO", key) for key in record.INFO]
        if record.FORMAT:
            fields.extend(("FORMAT", key) for key in record.FORMAT)
        for scope, field_id in fields:
            new_id = self.rename_for(source_index, scope, field_id)
            source_info = self._source_field_info(scope, field_id, source_index)
            merged_line = self.header._indices.get(scope, {}).get(new_id)
            if merged_line is None:
                if source_info is not None:  # pragma: no cover - internal invariant
                    raise MergeError(
                        "Merged header is missing an explicit {} line for {!r}".format(scope, new_id)
                    )
                # The parser used its built-in fallback because the field had
                # no declaration, and the merged header has no declaration
                # either.  The same fallback therefore applies to the output.
                continue
            merged_info = _effective_field_info(merged_line)
            if source_info is None:
                # The parser used its built-in fallback, since the field had no
                # declaration.  That is only safe if the merged header declares
                # exactly the fallback semantics.
                reserved = RESERVED_INFO if scope == "INFO" else RESERVED_FORMAT
                fallback = reserved.get(field_id, FieldInfo("String", ".", id_=field_id))
                if _signature(fallback) != _signature(merged_info):
                    raise MergeError(
                        "Refusing to write undeclared {scope} {old} as {new}: merged header declares "
                        "{number}:{type}".format(
                            scope=scope,
                            old=field_id,
                            new=new_id,
                            number=merged_info.number,
                            type=merged_info.type,
                        )
                    )
                continue
            if _signature(source_info) != _signature(merged_info):
                raise MergeError(
                    "Refusing to write {scope} {old} as {new}: source declares {s_number}:{s_type} "
                    "but merged header declares {m_number}:{m_type}".format(
                        scope=scope,
                        old=field_id,
                        new=new_id,
                        s_number=source_info.number,
                        s_type=source_info.type,
                        m_number=merged_info.number,
                        m_type=merged_info.type,
                    )
                )

    @staticmethod
    def _rename_mapping_keys(
        mapping: Mapping[str, Any], renames: Mapping[str, str], scope: Scope, record: Record
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in mapping.items():
            new_key = renames.get(key, key)
            if new_key in result:
                raise MergeError(
                    "Cannot rename {}/{} in record {}:{}: target field {!r} already exists".format(
                        scope, key, record.CHROM, record.POS, new_key
                    )
                )
            result[new_key] = value
        return result

    @classmethod
    def _rename_format(cls, record: Record, renames: Mapping[str, str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for key in record.FORMAT:
            new_key = renames.get(key, key)
            if new_key in seen:
                raise MergeError(
                    "Cannot rename FORMAT/{} in record {}:{}: target field {!r} already exists".format(
                        key, record.CHROM, record.POS, new_key
                    )
                )
            seen.add(new_key)
            result.append(new_key)
        return result


HeaderMergePlan = MergePlan


def _normalise_headers(headers: Sequence[Header] | Header, more: Sequence[Header]) -> list[Header]:
    if more:
        if not isinstance(headers, Header):
            raise MergeError("Pass headers either as a sequence or as multiple arguments, not both")
        return [headers, *more]
    if isinstance(headers, Header):
        return [headers]
    return list(headers)


def _collect_field_occurrences(headers: Sequence[Header]) -> list[_FieldOccurrence]:
    result: list[_FieldOccurrence] = []
    for source_index, header in enumerate(headers):
        for line in header.lines:
            if line.key not in _SCOPES:
                continue
            if not isinstance(line, (InfoHeaderLine, FormatHeaderLine)):  # pragma: no cover
                raise MergeError("INFO/FORMAT header line has unexpected type: {}".format(type(line)))
            result.append(
                _FieldOccurrence(
                    scope=line.key,  # type: ignore[arg-type]
                    source_index=source_index,
                    line=line,
                    info=_effective_field_info(line),
                )
            )
    return result


def _resolve_requested_rename(
    rename: Callable[[Scope, int, str, FieldInfo], str] | Mapping[Any, str] | None,
    scope: Scope,
    source_index: int,
    field_id: str,
    info: FieldInfo,
    source_names: Sequence[str],
) -> str | None:
    """Return a user-requested target ID, if one was supplied."""
    if rename is None:
        return None
    if callable(rename):
        return rename(scope, source_index, field_id, info)
    if not isinstance(rename, Mapping):  # pragma: no cover - defensive
        raise MergeError("rename must be a callable, a mapping or None")
    candidates = (
        (scope, source_index, field_id),
        (scope, source_names[source_index], field_id),
        (scope, field_id),
        "{}/{}".format(scope, field_id),
        field_id,
    )
    for key in candidates:
        if key in rename:
            return rename[key]
    return None


def _make_unique(candidate: str, used: set[str], fallback: str) -> str:
    """Return ``candidate`` if unused, otherwise append a numeric suffix."""
    candidate = candidate or fallback
    if candidate not in used:
        return candidate
    index = 2
    while "{}_{}".format(candidate, index) in used:
        index += 1
    return "{}_{}".format(candidate, index)


def _requested_target_conflicts(
    candidate: str, scope: Scope, group: Sequence[_FieldOccurrence], groups: Mapping[tuple[Scope, str], Any]
) -> bool:
    """Return whether an explicit target ID collides with another field.

    Automatic names may be adjusted with a numeric suffix.  An explicit name
    chosen by the caller should not be silently changed; if it names a field
    that is not part of this conflicting group, the plan is ambiguous.
    """
    other = groups.get((scope, candidate))
    if other is None:
        return False
    return {id(occurrence.line) for occurrence in other} != {id(occurrence.line) for occurrence in group}


def _assign_field_targets(
    occurrences: Sequence[_FieldOccurrence],
    source_names: Sequence[str],
    rename: Callable[[Scope, int, str, FieldInfo], str] | Mapping[Any, str] | None,
    auto_rename: bool,
) -> tuple[list[FieldRename], list[MergeConflict], list[str], list[dict[str, Any]]]:
    """Assign target IDs.

    :returns: ``(renames, conflicts, unresolved, metadata_conflicts)``
    """
    groups: dict[tuple[Scope, str], list[_FieldOccurrence]] = {}
    for occurrence in occurrences:
        groups.setdefault((occurrence.scope, occurrence.info.id or ""), []).append(occurrence)

    used_ids: dict[Scope, set[str]] = {
        scope: {field_id for (group_scope, field_id) in groups if group_scope == scope} for scope in _SCOPES
    }
    renames: dict[tuple[Scope, int, str], FieldRename] = {}
    conflicts: list[MergeConflict] = []
    unresolved: list[str] = []
    metadata_conflicts: list[dict[str, Any]] = []

    for (scope, field_id), group in groups.items():
        # Partition by declared semantics, preserving first-seen order.
        semantic_groups: list[list[_FieldOccurrence]] = []
        signatures: list[tuple[Any, str]] = []
        for occurrence in group:
            signature = _signature(occurrence.info)
            if signature in signatures:
                semantic_groups[signatures.index(signature)].append(occurrence)
            else:
                signatures.append(signature)
                semantic_groups.append([occurrence])

        primary = semantic_groups[0]
        for occurrence in primary:
            occurrence.target_id = field_id

        # Description/Source/Version do not change how values are interpreted,
        # so they never force a rename.  Record the first-kept choice so the
        # preview is honest about differing metadata.
        for occurrence in primary[1:]:
            if occurrence.line.mapping != primary[0].line.mapping:
                metadata_conflicts.append(
                    {
                        "kind": "field",
                        "scope": scope,
                        "id": field_id,
                        "source_index": occurrence.source_index,
                        "action": "kept_first",
                        "reason": "Description/Source/Version differ; first definition kept",
                    }
                )

        conflict_index: int | None = None
        if len(semantic_groups) > 1:
            definitions = tuple(part[0].info for part in semantic_groups)
            reasons = sorted({_signature_reason(definitions[0], info) for info in definitions[1:]})
            conflicts.append(
                MergeConflict(
                    scope=scope,
                    field_id=field_id,
                    source_indices=tuple(part[0].source_index for part in semantic_groups),
                    definitions=definitions,
                    reason="; ".join(reasons),
                )
            )
            conflict_index = len(conflicts) - 1

        # Within one source, a field ID can only have one meaning.  A source
        # that declares the same ID twice with different semantics cannot be
        # converted unambiguously, regardless of what other sources declare.
        for source_index in sorted({occurrence.source_index for occurrence in group}):
            source_signatures = {
                _signature(occurrence.info)
                for occurrence in group
                if occurrence.source_index == source_index
            }
            if len(source_signatures) > 1:
                unresolved.append(
                    "{} {} is declared more than once in source {} with incompatible definitions".format(
                        scope, field_id, source_names[source_index]
                    )
                )

        renamed_targets: list[str] = []
        for part in semantic_groups[1:]:
            first = part[0]
            explicit = rename is not None
            candidate = _resolve_requested_rename(
                rename, scope, first.source_index, field_id, first.info, source_names
            )
            if candidate is None:
                if not auto_rename:
                    unresolved.append(
                        "{} {} is declared incompatibly and auto_rename is disabled".format(scope, field_id)
                    )
                    for occurrence in part:
                        # Keep the original ID.  Any attempt to execute this
                        # plan raises because the merged header would carry
                        # two definitions for the same ID.
                        occurrence.target_id = field_id
                    continue
                candidate = "{}_{}".format(field_id, _sanitize_source_name(source_names[first.source_index]))
                explicit = False

            if candidate == field_id:
                unresolved.append(
                    "{} {} in source {} cannot be renamed to the same ID".format(
                        scope, field_id, source_names[first.source_index]
                    )
                )
                for occurrence in part:
                    occurrence.target_id = field_id
                continue

            if explicit and _requested_target_conflicts(candidate, scope, group, groups):
                unresolved.append(
                    "{} {} in source {} cannot be renamed to {!r}: that ID belongs to another field".format(
                        scope, field_id, source_names[first.source_index], candidate
                    )
                )
                for occurrence in part:
                    occurrence.target_id = field_id
                continue

            target_id = _make_unique(candidate, used_ids[scope], field_id)
            used_ids[scope].add(target_id)
            for occurrence in part:
                occurrence.target_id = target_id
                key = (scope, occurrence.source_index, field_id)
                previous = renames.get(key)
                if previous is not None and previous.new_id != target_id:
                    unresolved.append(
                        "{} {} is declared more than once in source {} with incompatible definitions".format(
                            scope, field_id, source_names[occurrence.source_index]
                        )
                    )
                    continue
                renames[key] = FieldRename(
                    scope=scope,
                    source_index=occurrence.source_index,
                    old_id=field_id,
                    new_id=target_id,
                    reason=_signature_reason(primary[0].info, occurrence.info),
                    source_name=source_names[occurrence.source_index],
                )
            renamed_targets.append(target_id)

        if conflict_index is not None:
            conflict = conflicts[conflict_index]
            conflicts[conflict_index] = MergeConflict(
                scope=conflict.scope,
                field_id=conflict.field_id,
                source_indices=conflict.source_indices,
                definitions=conflict.definitions,
                reason=conflict.reason,
                renamed_to=tuple(renamed_targets),
            )

    return list(renames.values()), conflicts, unresolved, metadata_conflicts


def _select_non_field_lines(
    headers: Sequence[Header],
) -> tuple[set[int], list[dict[str, Any]]]:
    """Return object IDs of non-INFO/FORMAT lines to emit, plus conflicts."""
    selected: set[int] = set()
    conflicts: list[dict[str, Any]] = []
    seen_mapping: dict[tuple[str, Any], Any] = {}
    seen_plain: set[tuple[str, str]] = set()
    seen_fileformat: str | None = None
    for source_index, header in enumerate(headers):
        for line in header.lines:
            if line.key in _SCOPES:
                continue
            if hasattr(line, "mapping"):
                key = (line.key, line.mapping.get("ID"))
                if key in seen_mapping:
                    if seen_mapping[key].mapping != line.mapping:
                        conflicts.append(
                            {
                                "kind": "header",
                                "key": line.key,
                                "id": line.mapping.get("ID"),
                                "source_index": source_index,
                                "action": "kept_first",
                                "reason": "conflicting non-INFO/FORMAT header line",
                            }
                        )
                    continue
                seen_mapping[key] = line
                selected.add(id(line))
            else:
                if line.key == "fileformat":
                    if seen_fileformat is None:
                        seen_fileformat = line.value
                        selected.add(id(line))
                    elif line.value != seen_fileformat:
                        conflicts.append(
                            {
                                "kind": "header",
                                "key": line.key,
                                "id": None,
                                "source_index": source_index,
                                "action": "kept_first",
                                "reason": "conflicting fileformat lines",
                            }
                        )
                    continue
                pair = (line.key, line.value)
                if pair in seen_plain:
                    continue
                seen_plain.add(pair)
                selected.add(id(line))
    return selected, conflicts


def _build_merged_header(
    headers: Sequence[Header],
    occurrences: Sequence[_FieldOccurrence],
    non_field_selected: set[int],
) -> Header:
    """Build the merged header, preserving source order where possible."""
    occurrence_by_line = {id(occurrence.line): occurrence for occurrence in occurrences}
    emitted_fields: set[tuple[Scope, str]] = set()
    lines: list[Any] = []
    for source_index, header in enumerate(headers):
        for line in header.lines:
            if line.key in _SCOPES:
                occurrence = occurrence_by_line[id(line)]
                key = (occurrence.scope, occurrence.target_id)
                if key in emitted_fields:
                    continue
                emitted_fields.add(key)
                mapping = dict(occurrence.line.mapping)
                mapping["ID"] = occurrence.target_id
                if occurrence.scope == "INFO":
                    lines.append(InfoHeaderLine.from_mapping(mapping))
                else:
                    lines.append(FormatHeaderLine.from_mapping(mapping))
            elif id(line) in non_field_selected:
                lines.append(line.copy())

    sample_names: list[str] = []
    seen_samples: set[str] = set()
    for header in headers:
        if not header.samples:
            continue
        for name in header.samples.names:
            if name not in seen_samples:
                seen_samples.add(name)
                sample_names.append(name)

    if sample_names and not any(line.key == "FORMAT" and line.mapping.get("ID") == "GT" for line in lines):
        lines.append(
            FormatHeaderLine.from_mapping(
                {"ID": "GT", "Number": 1, "Type": "String", "Description": "Genotype"}
            )
        )

    samples = SamplesInfos(sample_names) if sample_names else None
    with warnings.catch_warnings():
        # A repeated ID in the generated header is meaningful: it is how an
        # unresolved plan stays visible, and execution refuses such plans.
        warnings.simplefilter("ignore", exceptions.DuplicateHeaderLineWarning)
        return Header(lines, samples)


def plan_merge(
    headers: Sequence[Header] | Header,
    *more_headers: Header,
    source_names: Sequence[str] | None = None,
    rename: Callable[[Scope, int, str, FieldInfo], str] | Mapping[Any, str] | None = None,
    auto_rename: bool = True,
    strict: bool = False,
) -> MergePlan:
    """Plan a merge of VCF headers without reading any records.

    Compatibility is determined exclusively from each field's declared
    ``Number`` and ``Type``.  A ``Number=A`` field and a ``Number=1`` field
    with the same ID are therefore incompatible even if a particular record
    happens to have exactly one ALT allele.  Record values are never consulted
    here.

    :param headers: source headers, as a sequence or as multiple arguments
    :param source_names: names used to qualify renamed fields; defaults to
        ``"1"``, ``"2"``, ... in source order
    :param rename: optional explicit mapping or callable.  A callable is called
        as ``rename(scope, source_index, field_id, field_info)``.  Mapping keys
        may be ``(scope, source_index, field_id)``,
        ``(scope, source_name, field_id)``, ``(scope, field_id)``,
        ``"SCOPE/field_id"`` or ``field_id``.
    :param auto_rename: automatically rename incompatible later definitions
    :param strict: reject any field-definition conflict instead of resolving it
        by renaming.  This is useful for a pre-flight compatibility check.
    :returns: :class:`MergePlan`
    :raises MergeError: for invalid arguments
    :raises MergeConflictError: if ``strict`` is true and a conflict is found
    """
    header_list = _normalise_headers(headers, more_headers)
    if not header_list:
        raise MergeError("At least one header is required")
    # Work on copies so that the plan is insulated from later header
    # mutation and so that passing the same header object twice still yields
    # two independent sources.  A source may already contain a duplicate ID;
    # that is reported through ``unresolved`` below rather than re-warned here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", exceptions.DuplicateHeaderLineWarning)
        header_list = [header.copy() for header in header_list]

    if source_names is None:
        names = [str(index + 1) for index in range(len(header_list))]
    else:
        names = [str(name) for name in source_names]
        if len(names) != len(header_list):
            raise MergeError("Expected {} source names but got {}".format(len(header_list), len(names)))
        if len(set(names)) != len(names):
            raise MergeError("Source names must be unique")

    occurrences = _collect_field_occurrences(header_list)
    renames, conflicts, unresolved, metadata_conflicts = _assign_field_targets(
        occurrences, names, rename, auto_rename
    )
    non_field_selected, header_conflicts = _select_non_field_lines(header_list)
    merged_header = _build_merged_header(header_list, occurrences, non_field_selected)

    if strict and unresolved:
        raise MergeConflictError("; ".join(unresolved))
    if strict and conflicts:
        descriptions = [
            "{} {} ({})".format(conflict.scope, conflict.field_id, conflict.reason)
            for conflict in conflicts
        ]
        raise MergeConflictError("Incompatible field definitions: {}".format("; ".join(descriptions)))

    return MergePlan(
        headers=header_list,
        source_names=names,
        merged_header=merged_header,
        renames=renames,
        conflicts=conflicts,
        header_conflicts=header_conflicts,
        unresolved=unresolved,
        field_metadata_conflicts=metadata_conflicts,
    )


#: Descriptive aliases.
plan_header_merge = plan_merge
preview_merge = plan_merge
preview_header_merge = plan_merge
dry_run_merge = plan_merge


def merge_headers(
    headers: Sequence[Header] | Header,
    *more_headers: Header,
    source_names: Sequence[str] | None = None,
    rename: Callable[[Scope, int, str, FieldInfo], str] | Mapping[Any, str] | None = None,
    auto_rename: bool = True,
    strict: bool = False,
) -> MergePlan:
    """Alias for :func:`plan_merge`.

    The result is also a two-item iterable, so both forms work::

        plan = merge_headers(header_a, header_b)
        merged_header, rename_map = merge_headers(header_a, header_b)

    ``rename_map`` maps ``(scope, source_index, old_id)`` to the new ID;
    ``plan.renames`` provides the richer per-rename objects.
    """
    return plan_merge(
        headers,
        *more_headers,
        source_names=source_names,
        rename=rename,
        auto_rename=auto_rename,
        strict=strict,
    )


merge_header = merge_headers


def merge_record(plan: MergePlan, record: Record, source: int | str, *, in_place: bool = False) -> Record:
    """Apply ``plan`` to one record from ``source``."""
    return plan.convert_record(record, source, in_place=in_place)


def merge_records(
    plan: MergePlan,
    records_by_source: Sequence[Iterable[Record]],
    *,
    sort: bool = False,
    in_place: bool = False,
    pad: bool = True,
) -> list[Record]:
    """Apply ``plan`` to records from all sources.

    :param plan: plan returned by :func:`plan_merge`
    :param records_by_source: one iterable of records per source
    :param sort: sort the result by chromosome and position
    :param in_place: modify source records instead of producing copies
    :param pad: add ``"."`` calls for samples present only in another source
        (the default, since a record must have one value per output sample)
    :raises MergeConflictError: if the plan is not executable
    :raises MergeError: if the number of record iterables is wrong
    """
    plan.require_mergeable()
    if len(records_by_source) != len(plan.headers):
        raise MergeError(
            "Expected records for {} sources but got {}".format(len(plan.headers), len(records_by_source))
        )
    result: list[Record] = []
    for source_index, records in enumerate(records_by_source):
        for record in records:
            converted = plan.convert_record(record, source_index, in_place=in_place)
            if pad:
                converted = plan.pad_record(converted)
            result.append(converted)
    if sort:
        result.sort(key=lambda record: (record.CHROM, record.POS))
    return result


#: Execution aliases.
apply_merge = merge_records
execute_merge = merge_records


def merge_vcf_files(
    paths: Sequence[str],
    output: str | None = None,
    *,
    source_names: Sequence[str] | None = None,
    rename: Callable[[Scope, int, str, FieldInfo], str] | Mapping[Any, str] | None = None,
    record_checks: Sequence[Literal["INFO", "FORMAT"]] | None = None,
    sort: bool = False,
    dry_run: bool = False,
) -> MergePlan:
    """Merge VCF files on disk, optionally writing the result.

    Headers are merged first so every record is converted before it is
    written.  With ``dry_run=True`` or ``output=None`` no file is written;
    inspect the returned plan or render it with :meth:`MergePlan.to_json`.

    :param paths: input VCF paths
    :param output: output VCF path; omitted for a dry run
    :param source_names: optional names for the sources
    :param rename: optional rename mapping/callable (see :func:`plan_merge`)
    :param record_checks: forwarded to :class:`vcfpy.Reader`
    :param sort: sort merged records by chromosome and position
    :param dry_run: only plan the merge, do not write records
    :returns: the :class:`MergePlan` that was executed
    """
    from vcfpy.reader import Reader
    from vcfpy.writer import Writer

    readers: list[Any] = []
    try:
        for path in paths:
            readers.append(Reader.from_path(path, record_checks=list(record_checks or [])))
        plan = plan_merge([reader.header for reader in readers], source_names=source_names, rename=rename)
        plan.require_mergeable()
        if dry_run or output is None:
            return plan
        if sort:
            # Sorting requires buffering all records; plain concatenation can
            # be streamed straight to the writer.
            merged_records = merge_records(plan, readers, sort=True, pad=True)
            with Writer.from_path(output, plan.header) as writer:
                for record in merged_records:
                    writer.write_record(record)
        else:
            with Writer.from_path(output, plan.header) as writer:
                for source_index, reader in enumerate(readers):
                    for record in reader:
                        converted = plan.convert_record(record, source_index)
                        writer.write_record(plan.pad_record(converted))
        return plan
    finally:
        for reader in readers:
            reader.close()


merge_vcf = merge_vcf_files


def _rename_from_argument(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("rename must have the form OLD=NEW")
    old, new = value.split("=", 1)
    if not old or not new:
        raise argparse.ArgumentTypeError("rename must have the form OLD=NEW")
    return old, new


def main(argv: Sequence[str] | None = None, stream: TextIO | None = None) -> int:
    """Command line entry point for ``python -m vcfpy.merge``."""
    parser = argparse.ArgumentParser(
        prog="python -m vcfpy.merge",
        description="Merge VCF files after previewing header compatibility.",
    )
    parser.add_argument("inputs", nargs="+", help="input VCF files")
    parser.add_argument("-o", "--output", help="output VCF; omit for a dry run")
    parser.add_argument(
        "--source-name",
        action="append",
        dest="source_names",
        help="name for a source; repeat once per input (default: 1, 2, ...)",
    )
    parser.add_argument(
        "--rename",
        action="append",
        type=_rename_from_argument,
        default=[],
        metavar="OLD=NEW",
        help="explicit rename for a conflicting definition",
    )
    parser.add_argument("--json", action="store_true", help="print the plan as JSON")
    args = parser.parse_args(argv)
    out = stream or sys.stdout

    # Scope prefixes in --rename are optional.  ``INFO/AF=AF_src`` and
    # ``FORMAT/AD=AD_src`` are accepted; a bare ``OLD=NEW`` applies wherever
    # the ID is involved.
    rename_map: dict[Any, str] = {}
    for old, new in args.rename:
        if old.startswith(("INFO/", "FORMAT/")):
            scope, field_id = old.split("/", 1)
            rename_map[(scope, field_id)] = new
        else:
            rename_map[old] = new

    plan = merge_vcf_files(
        args.inputs,
        args.output,
        source_names=args.source_names,
        rename=rename_map or None,
        dry_run=args.output is None,
    )

    if args.json:
        print(plan.to_json(), file=out)
        return 0

    print("sources: {}".format(", ".join(plan.source_names)), file=out)
    print("compatible: {}".format("yes" if plan.compatible else "no"), file=out)
    if plan.renames:
        print("renames:", file=out)
        for rename in plan.renames:
            print(
                "  {scope} {old} (source {source}) -> {new}: {reason}".format(
                    scope=rename.scope,
                    old=rename.old_id,
                    source=rename.source_name,
                    new=rename.new_id,
                    reason=rename.reason,
                ),
                file=out,
            )
    else:
        print("renames: none", file=out)
    if not plan.mergeable:
        print("unresolved:", file=out)
        for message in plan.unresolved:
            print("  {}".format(message), file=out)
    if args.output:
        print("wrote: {}".format(args.output), file=out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
