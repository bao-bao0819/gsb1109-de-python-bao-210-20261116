# -*- coding: utf-8 -*-
"""Allele-level provenance and allele-index aware transformations.

VCF files describe multi-allelic records positionally: ``Number=A``,
``Number=R`` and ``Number=G`` annotations, as well as ``GT`` values, refer to
ALT alleles by their index.  Once records are split, merged or re-ordered,
those indices no longer identify the original alleles.  Record IDs are not a
reliable identity either, because splitting one record yields several rows
with the same ID and merging rows combines their IDs.

This module provides two related pieces:

``AlleleMapping``
    An explicit, ID-independent mapping from every output allele back to the
    original allele(s) it was derived from.  Each correspondence carries the
    annotation version it was built from and values that cannot be mapped are
    collected in :py:attr:`AlleleMapping.unmatched` instead of being guessed
    from their old array position.

:py:func:`infer_mapping`
    Build such a mapping for already split/merged records by matching allele
    values at the same locus.  IDs are deliberately ignored.

The functions in this module only operate on *existing* allele-indexed
annotations.  Re-ordering an ALT column is implemented by
:py:meth:`vcfpy.record.Record.reorder_alts`, which uses the same low-level
helpers so that INFO/FORMAT values and sample genotypes follow the mapping.
"""

from __future__ import annotations

import itertools
import math
import re
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import vcfpy.header as header_module

if TYPE_CHECKING:  # pragma: no cover
    from vcfpy.record import Call, Record, UnparsedCall

#: Allele index of the reference allele in VCF genotype/Number=G arrays.
REF_ALLELE = 0
#: ``Number`` values that describe one value per ALT allele.
NUMBER_ALLELES = "A"
#: ``Number`` values that describe one value per allele, including REF.
NUMBER_REF = "R"
#: ``Number`` values that describe one value per genotype.
NUMBER_GENOTYPES = "G"
#: The allele-indexed ``Number`` values supported by the remapping helpers.
ALLELE_NUMBERS = (NUMBER_ALLELES, NUMBER_REF, NUMBER_GENOTYPES)

# GT is split on the separators while keeping them so that phase information
# survives a remapping.
GT_SPLIT = re.compile(r"([|/])")


def _normalize_key(record_key: Hashable) -> Hashable:
    """Return a hashable key, rejecting mutable/ambiguous inputs early."""
    if not isinstance(record_key, Hashable):
        raise TypeError("record_key must be hashable, got {!r}".format(record_key))
    return record_key


@dataclass(frozen=True, order=True)
class AlleleKey:
    """Identity of one allele in one record.

    ``record_key`` must distinguish records independently of the VCF ``ID``
    column, for example ``(path, line_number)`` or an ordinal assigned while
    reading.  ``allele_index`` is ``0`` for REF and ``1``, ``2``, ... for ALT
    alleles in their VCF order.
    """

    record_key: Hashable
    allele_index: int

    def __post_init__(self):
        _normalize_key(self.record_key)
        if self.allele_index < 0:
            raise ValueError("allele_index must be >= 0, got {}".format(self.allele_index))

    @property
    def is_ref(self) -> bool:
        """Return ``True`` for the reference allele."""
        return self.allele_index == REF_ALLELE

    @property
    def alt_index(self) -> int | None:
        """Return the 1-based ALT index or ``None`` for REF."""
        return None if self.is_ref else self.allele_index


@dataclass(frozen=True)
class RecordMeta:
    """Metadata about a source record, kept for traceability.

    The VCF ``ID`` values are stored here but are never used as identity.
    """

    record_key: Hashable
    record_ids: tuple[str, ...] = ()
    chrom: str | None = None
    pos: int | None = None
    ref: str | None = None
    alt_values: tuple[str, ...] = ()
    annotation_version: Hashable | None = None


@dataclass
class SourceAllele:
    """One original allele and the annotation version it belongs to."""

    key: AlleleKey
    annotation: Any = None
    annotation_version: Hashable | None = None
    record_ids: tuple[str, ...] = ()

    @property
    def record_key(self) -> Hashable:
        """Return the key of the record this allele belongs to."""
        return self.key.record_key

    @property
    def allele_index(self) -> int:
        """Return the allele index within the source record."""
        return self.key.allele_index


@dataclass
class UnmatchedValue:
    """A value that could not be placed without guessing.

    Values are reported here instead of being attached to an output allele by
    their old array position.  ``field`` uses ``INFO/<ID>`` or
    ``FORMAT/<ID>`` notation; ``sample`` is set for call-level values.
    """

    field: str
    value: Any
    reason: str
    output_key: AlleleKey | None = None
    source_key: AlleleKey | None = None
    sample: str | None = None
    annotation_version: Hashable | None = None


class AlleleMapping:
    """Mapping from output alleles to every original allele they derive from.

    The mapping is directional for lookups (:py:meth:`sources_for` and
    :py:meth:`outputs_for` are both available) but is normally built from the
    output side.  One output allele may have several source alleles, e.g. when
    identical ALTs from two records are merged.  One source ALT may also be
    used by several output alleles, e.g. when its record was split into
    several rows sharing the same REF.

    ``annotation_version`` records which version of the annotations the
    mapping was built against.  Individual :class:`SourceAllele` entries can
    carry their own version when source records differ.  Values whose version
    or allele identity cannot be established are appended to
    :py:attr:`unmatched`.
    """

    def __init__(self, annotation_version: Hashable | None = None):
        #: annotation version this mapping was built against
        self.annotation_version = annotation_version
        # mapping output allele -> list of source alleles
        self._output_to_sources: dict[AlleleKey, list[SourceAllele]] = {}
        # reverse mapping source allele -> list of output alleles
        self._source_to_outputs: dict[AlleleKey, list[AlleleKey]] = {}
        #: source/output allele values that could not be mapped
        self.unmatched: list[UnmatchedValue] = []
        #: metadata per record key (IDs, locus, original ALT values, version)
        self.records: dict[Hashable, RecordMeta] = {}

    # -- construction -----------------------------------------------------

    def register_record(
        self,
        record_key: Hashable,
        *,
        record_ids: Iterable[str] = (),
        chrom: str | None = None,
        pos: int | None = None,
        ref: str | None = None,
        alt_values: Iterable[str] = (),
        annotation_version: Hashable | None = None,
    ) -> RecordMeta:
        """Register metadata for a record key, overwriting an earlier entry."""
        record_key = _normalize_key(record_key)
        meta = RecordMeta(
            record_key=record_key,
            record_ids=tuple(record_ids),
            chrom=chrom,
            pos=pos,
            ref=ref,
            alt_values=tuple(alt_values),
            annotation_version=annotation_version,
        )
        self.records[record_key] = meta
        return meta

    def register_record_object(
        self,
        record: "Record",
        record_key: Hashable | None = None,
        *,
        annotation_version: Hashable | None = None,
    ) -> RecordMeta:
        """Register a :class:`vcfpy.Record` and return its metadata.

        The key defaults to ``record.record_key`` and falls back to the same
        deterministic locus/allele key used by the split/merge helpers.
        """
        key = record_key_for(record) if record_key is None else record_key
        return self.register_record(
            key,
            record_ids=record.ID,
            chrom=record.CHROM,
            pos=record.POS,
            ref=record.REF,
            alt_values=(alt.serialize() for alt in record.ALT),
            annotation_version=annotation_version,
        )

    def add(
        self,
        output_key: AlleleKey,
        source_key: AlleleKey,
        *,
        annotation: Any = None,
        annotation_version: Hashable | None = None,
        record_ids: Iterable[str] = (),
    ) -> SourceAllele:
        """Add one output allele -> original allele correspondence.

        Adding the same pair twice is a no-op apart from refreshing the
        stored annotation/version.
        """
        if not isinstance(output_key, AlleleKey) or not isinstance(source_key, AlleleKey):
            raise TypeError("output_key and source_key must be AlleleKey instances")
        sources = self._output_to_sources.setdefault(output_key, [])
        for existing in sources:
            if existing.key == source_key:
                if annotation is not None:
                    existing.annotation = annotation
                if annotation_version is not None:
                    existing.annotation_version = annotation_version
                if record_ids:
                    existing.record_ids = tuple(record_ids)
                return existing
        source = SourceAllele(
            key=source_key,
            annotation=annotation,
            annotation_version=annotation_version,
            record_ids=tuple(record_ids),
        )
        sources.append(source)
        outputs = self._source_to_outputs.setdefault(source_key, [])
        if output_key not in outputs:
            outputs.append(output_key)
        return source

    def report_unmatched(
        self,
        field: str,
        value: Any,
        reason: str,
        *,
        output_key: AlleleKey | None = None,
        source_key: AlleleKey | None = None,
        sample: str | None = None,
        annotation_version: Hashable | None = None,
    ) -> UnmatchedValue:
        """Record a value that cannot be mapped and return the entry."""
        entry = UnmatchedValue(
            field=field,
            value=value,
            reason=reason,
            output_key=output_key,
            source_key=source_key,
            sample=sample,
            annotation_version=annotation_version,
        )
        self.unmatched.append(entry)
        return entry

    #: Backwards-compatible alias for :py:meth:`report_unmatched`.
    add_unmatched = report_unmatched

    # -- lookups ----------------------------------------------------------

    def sources_for(self, output_key: AlleleKey) -> tuple[SourceAllele, ...]:
        """Return all original alleles mapped to ``output_key``."""
        return tuple(self._output_to_sources.get(output_key, ()))

    def outputs_for(self, source_key: AlleleKey) -> tuple[AlleleKey, ...]:
        """Return all output alleles that use ``source_key``."""
        return tuple(self._source_to_outputs.get(source_key, ()))

    def annotation_for(self, output_key: AlleleKey, default: Any = None) -> Any:
        """Return the annotation of the first source allele for an output allele."""
        sources = self._output_to_sources.get(output_key)
        if not sources:
            return default
        return sources[0].annotation

    @property
    def output_keys(self) -> tuple[AlleleKey, ...]:
        """Return output allele keys in insertion order."""
        return tuple(self._output_to_sources)

    @property
    def output_to_sources(self) -> dict[AlleleKey, tuple[SourceAllele, ...]]:
        """Return a read-only view of output allele -> source alleles."""
        return {key: tuple(sources) for key, sources in self._output_to_sources.items()}

    @property
    def source_to_outputs(self) -> dict[AlleleKey, tuple[AlleleKey, ...]]:
        """Return a read-only view of source allele -> output alleles."""
        return {key: tuple(outputs) for key, outputs in self._source_to_outputs.items()}

    @property
    def unmatched_values(self) -> tuple[UnmatchedValue, ...]:
        """Alias for :py:attr:`unmatched`."""
        return tuple(self.unmatched)

    @property
    def source_keys(self) -> tuple[AlleleKey, ...]:
        """Return source allele keys in insertion order."""
        return tuple(self._source_to_outputs)

    @property
    def versions(self) -> set[Hashable | None]:
        """Return the set of annotation versions used by source alleles."""
        return {source.annotation_version for sources in self._output_to_sources.values() for source in sources}

    @property
    def is_version_consistent(self) -> bool:
        """Return whether all mapped alleles share the mapping annotation version."""
        return not self.versions or self.versions <= {self.annotation_version}

    def record_ids_for(self, record_key: Hashable) -> tuple[str, ...]:
        """Return the original VCF IDs for a source record key."""
        meta = self.records.get(record_key)
        return () if meta is None else meta.record_ids

    # -- transformation ---------------------------------------------------

    def copy(self) -> "AlleleMapping":
        """Return a deep-enough copy of this mapping."""
        result = AlleleMapping(self.annotation_version)
        result.records = dict(self.records)
        for output_key, sources in self._output_to_sources.items():
            for source in sources:
                result.add(
                    output_key,
                    source.key,
                    annotation=source.annotation,
                    annotation_version=source.annotation_version,
                    record_ids=source.record_ids,
                )
        result.unmatched = [
            UnmatchedValue(
                field=entry.field,
                value=entry.value,
                reason=entry.reason,
                output_key=entry.output_key,
                source_key=entry.source_key,
                sample=entry.sample,
                annotation_version=entry.annotation_version,
            )
            for entry in self.unmatched
        ]
        return result

    def permute_outputs(self, record_key: Hashable, order: Sequence[int]) -> "AlleleMapping":
        """Return mapping with one record's output ALT indices permuted.

        ``order`` uses the same convention as
        :py:meth:`vcfpy.record.Record.reorder_alts`: ``order[new_alt - 1]`` is
        the old ALT index (0-based into the old ``ALT`` list) that becomes the
        new ALT at that position.  REF always remains index ``0``.
        """
        record_key = _normalize_key(record_key)
        order = list(order)
        n_alts = len(order)
        if sorted(order) != list(range(n_alts)):
            raise ValueError("order must be a permutation of 0..{}, got {!r}".format(n_alts - 1, order))
        old_to_new = {old + 1: new + 1 for new, old in enumerate(order)}

        def moved(key: AlleleKey) -> AlleleKey:
            if key.record_key != record_key or key.allele_index == REF_ALLELE:
                return key
            try:
                return AlleleKey(record_key, old_to_new[key.allele_index])
            except KeyError:
                raise ValueError(
                    "mapping contains output allele {} outside the {} ALTs being re-ordered".format(key, n_alts)
                ) from None

        if not any(key.record_key == record_key for key in self._output_to_sources):
            raise ValueError("mapping has no output alleles for record key {!r}".format(record_key))
        result = AlleleMapping(self.annotation_version)
        result.records = dict(self.records)
        for output_key, sources in self._output_to_sources.items():
            for source in sources:
                result.add(
                    moved(output_key),
                    source.key,
                    annotation=source.annotation,
                    annotation_version=source.annotation_version,
                    record_ids=source.record_ids,
                )
        result.unmatched = [
            UnmatchedValue(
                field=entry.field,
                value=entry.value,
                reason=entry.reason,
                output_key=None if entry.output_key is None else moved(entry.output_key),
                source_key=entry.source_key,
                sample=entry.sample,
                annotation_version=entry.annotation_version,
            )
            for entry in self.unmatched
        ]
        return result

    def __len__(self) -> int:
        return len(self._output_to_sources)

    def __iter__(self):
        return iter(self._output_to_sources)

    def __getitem__(self, output_key: AlleleKey) -> tuple[SourceAllele, ...]:
        return self.sources_for(output_key)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the mapping."""
        def key_repr(key: AlleleKey) -> str:
            return "{}#{}".format(key.record_key, key.allele_index)

        return {
            "annotation_version": self.annotation_version,
            "records": {
                str(record_key): {
                    "record_ids": list(meta.record_ids),
                    "chrom": meta.chrom,
                    "pos": meta.pos,
                    "ref": meta.ref,
                    "alt_values": list(meta.alt_values),
                    "annotation_version": meta.annotation_version,
                }
                for record_key, meta in self.records.items()
            },
            "outputs": {
                key_repr(output_key): [
                    {
                        "source": key_repr(source.key),
                        "annotation": source.annotation,
                        "annotation_version": source.annotation_version,
                        "record_ids": list(source.record_ids),
                    }
                    for source in sources
                ]
                for output_key, sources in self._output_to_sources.items()
            },
            "unmatched": [
                {
                    "field": entry.field,
                    "value": entry.value,
                    "reason": entry.reason,
                    "output": None if entry.output_key is None else key_repr(entry.output_key),
                    "source": None if entry.source_key is None else key_repr(entry.source_key),
                    "sample": entry.sample,
                    "annotation_version": entry.annotation_version,
                }
                for entry in self.unmatched
            ],
        }


# ---------------------------------------------------------------------------
# Field-number helpers and low-level value remapping


def get_field_number(header: Any, section: str, key: str) -> str | None:
    """Return the declared ``Number`` for an INFO/FORMAT key.

    ``header`` may be ``None``, in which case only the reserved VCF fields are
    known.  Unknown fields return ``None`` instead of emitting a warning.
    """
    if header is not None:
        indices = getattr(header, "_indices", {})
        line = indices.get(section, {}).get(key)
        if line is not None and hasattr(line, "mapping"):
            number = line.mapping.get("Number")
            return None if number is None else str(number)
    reserved = header_module.RESERVED_INFO if section == "INFO" else header_module.RESERVED_FORMAT
    info = reserved.get(key)
    return None if info is None else str(info.number)


def infer_field_number(values: Any, num_alleles: int, ploidy: int | None = None) -> str | None:
    """Infer ``A``/``R``/``G`` from an array length.

    This is only used for fields that are absent from the header.  The
    declared ``Number`` always wins.  ``num_alleles`` includes REF.

    A length can be ambiguous (with one ALT and one allele, an array of length
    one could be ``A`` or ``R``); in that case no inference is made so that
    callers do not silently guess.  ``G`` is only considered when ``ploidy``
    is explicitly given.
    """
    if not isinstance(values, list) or num_alleles < 1:
        return None
    candidates = set()
    n_alts = num_alleles - 1
    if n_alts and len(values) == n_alts:
        candidates.add(NUMBER_ALLELES)
    if len(values) == num_alleles:
        candidates.add(NUMBER_REF)
    if ploidy is not None and len(values) == genotype_count(num_alleles, ploidy):
        candidates.add(NUMBER_GENOTYPES)
    return candidates.pop() if len(candidates) == 1 else None


def resolve_field_number(
    header: Any,
    section: str,
    key: str,
    values: Any,
    num_alleles: int,
    *,
    ploidy: int | None = None,
    infer: bool = True,
    numbers: dict[str, str] | None = None,
) -> str | None:
    """Resolve the allele-number semantics for one field.

    ``numbers`` optionally overrides header lookup, which is useful for
    non-standard fields whose ``Number`` is not declared as ``A``/``R``/``G``.
    """
    if numbers and key in numbers:
        return numbers[key]
    number = get_field_number(header, section, key)
    if number in ALLELE_NUMBERS:
        return number
    if number is None and infer:
        return infer_field_number(values, num_alleles, ploidy)
    return None


def genotype_count(num_alleles: int, ploidy: int) -> int:
    """Return the number of genotype combinations for ``Number=G``."""
    if ploidy < 0:
        raise ValueError("ploidy must be >= 0")
    return math.comb(num_alleles + ploidy - 1, ploidy)


def genotype_order(num_alleles: int, ploidy: int) -> list[tuple[int, ...]]:
    """Return VCF ``Number=G`` genotype order.

    VCF orders diploid genotypes as ``0/0, 0/1, 1/1, 0/2, 1/2, 2/2``: by the
    largest allele index first and then lexicographically.  The same ordering
    rule is used for other ploidies.
    """
    combos = list(itertools.combinations_with_replacement(range(num_alleles), ploidy))
    combos.sort(key=lambda combo: (max(combo) if combo else -1, combo))
    return combos


def _copy_value(value: Any) -> Any:
    return list(value) if isinstance(value, list) else value


def remap_index_vector(
    values: Any,
    number: str,
    order: Sequence[int],
    *,
    ploidy: int = 2,
) -> tuple[list[Any] | None, str | None]:
    """Remap one ``A``/``R``/``G`` value array for a re-ordered ALT column.

    Returns ``(values, None)`` on success and ``(None, reason)`` when the
    array length does not match the declared ``Number``.  The caller reports
    the failure as an :class:`UnmatchedValue`; it must not fall back to the
    old positions.
    """
    if not isinstance(values, list):
        return values, None
    if not values:
        return values, None
    order = list(order)
    num_alleles = len(order) + 1
    if number == NUMBER_ALLELES:
        if len(values) != len(order):
            return None, "expected {} Number=A values, found {}".format(len(order), len(values))
        return [_copy_value(values[idx]) for idx in order], None
    if number == NUMBER_REF:
        if len(values) != num_alleles:
            return None, "expected {} Number=R values, found {}".format(num_alleles, len(values))
        return [_copy_value(values[REF_ALLELE])] + [_copy_value(values[idx + 1]) for idx in order], None
    if number == NUMBER_GENOTYPES:
        old_order = genotype_order(num_alleles, ploidy)
        if len(values) != len(old_order):
            return None, "expected {} Number=G values for ploidy {}, found {}".format(
                len(old_order), ploidy, len(values)
            )
        allele_map = {REF_ALLELE: REF_ALLELE}
        allele_map.update({old + 1: new + 1 for new, old in enumerate(order)})
        new_order = genotype_order(num_alleles, ploidy)
        new_index = {genotype: idx for idx, genotype in enumerate(new_order)}
        result: list[Any] = [None] * len(values)
        for idx, genotype in enumerate(old_order):
            # Genotypes are unordered allele combinations; normalize before
            # looking them up in the canonical Number=G order.
            mapped = tuple(sorted(allele_map[allele] for allele in genotype))
            result[new_index[mapped]] = _copy_value(values[idx])
        return result, None
    return values, None


def slice_index_vector(
    values: Any,
    number: str,
    alt_index: int,
    num_alleles: int,
    *,
    ploidy: int = 2,
) -> tuple[list[Any] | None, str | None]:
    """Project a multi-allelic vector onto one ALT allele.

    Used by :py:func:`split_multi`.  ``alt_index`` is 1-based.
    """
    if not isinstance(values, list):
        return values, None
    if not values:
        return values, None
    if number == NUMBER_ALLELES:
        if len(values) != num_alleles - 1:
            return None, "expected {} Number=A values, found {}".format(num_alleles - 1, len(values))
        return [_copy_value(values[alt_index - 1])], None
    if number == NUMBER_REF:
        if len(values) != num_alleles:
            return None, "expected {} Number=R values, found {}".format(num_alleles, len(values))
        return [_copy_value(values[REF_ALLELE]), _copy_value(values[alt_index])], None
    if number == NUMBER_GENOTYPES:
        old_order = genotype_order(num_alleles, ploidy)
        if len(values) != len(old_order):
            return None, "expected {} Number=G values for ploidy {}, found {}".format(
                len(old_order), ploidy, len(values)
            )
        old_index = {genotype: idx for idx, genotype in enumerate(old_order)}
        new_order = genotype_order(2, ploidy)
        result: list[Any] = []
        for genotype in new_order:
            source_genotype = tuple(0 if allele == 0 else alt_index for allele in genotype)
            result.append(_copy_value(values[old_index[source_genotype]]))
        return result, None
    return values, None


def remap_gt(gt: Any, order: Sequence[int]) -> tuple[str | None, str | None]:
    """Remap allele numbers in a ``GT`` string for a re-ordered ALT column."""
    if gt is None:
        return None, None
    order = list(order)
    parts = GT_SPLIT.split(str(gt))
    result: list[str] = []
    for part in parts:
        if part in ("|", "/"):
            result.append(part)
        elif part == ".":
            result.append(part)
        else:
            try:
                allele = int(part)
            except ValueError:
                return None, "invalid GT allele {!r}".format(part)
            if allele == REF_ALLELE:
                result.append("0")
            elif 1 <= allele <= len(order):
                result.append(str(order.index(allele - 1) + 1))
            else:
                return None, "GT allele {} is outside 0..{}".format(allele, len(order))
    return "".join(result), None


def split_gt(gt: Any, alt_index: int, num_alts: int | None = None) -> tuple[str | None, str | None]:
    """Project a ``GT`` string onto one ALT allele of a split record.

    The allele matching ``alt_index`` becomes ``1`` and all other ALT alleles
    become ``0``; this is the standard projection used when one multi-allelic
    record is written as one row per ALT.  A genotype allele outside the
    source record's ALT range cannot be projected and yields a reason.
    """
    if gt is None:
        return None, None
    if alt_index < 1:
        raise ValueError("alt_index must be >= 1")
    parts = GT_SPLIT.split(str(gt))
    result: list[str] = []
    other_alts: list[int] = []
    for part in parts:
        if part in ("|", "/"):
            result.append(part)
        elif part == ".":
            result.append(part)
        else:
            try:
                allele = int(part)
            except ValueError:
                return None, "invalid GT allele {!r}".format(part)
            if num_alts is not None and allele > num_alts:
                return None, "GT allele {} is outside 0..{}".format(allele, num_alts)
            if allele == alt_index:
                result.append("1")
            else:
                result.append("0")
                if allele != REF_ALLELE and allele not in other_alts:
                    other_alts.append(allele)
    if other_alts:
        return (
            "".join(result),
            "genotype also uses ALT allele(s) {} which are not represented in this split row".format(
                ", ".join(map(str, other_alts))
            ),
        )
    return "".join(result), None


# ---------------------------------------------------------------------------
# Splitting and mapping inference


def _has_cross_alt_genotype(num_alts: int, ploidy: int) -> bool:
    """Return whether any genotype combines two distinct ALT alleles."""
    return any(
        len({allele for allele in genotype if allele != REF_ALLELE}) > 1
        for genotype in genotype_order(num_alts + 1, ploidy)
    )


def _default_record_key(record: "Record") -> Hashable:
    """Return a deterministic key that does not rely on the VCF ``ID`` column.

    Split rows share IDs and merged rows combine them, so IDs cannot identify
    a row.  The content tuple is stable across runs; callers can pass an
    explicit ``record_key`` (for example ``(path, line_number)``) when even
    byte-identical duplicate rows must stay distinct.
    """
    return (
        "record",
        record.CHROM,
        record.POS,
        record.REF,
        tuple(alt.serialize() for alt in record.ALT),
    )


def record_key_for(record: "Record") -> Hashable:
    """Return the record's explicit key or a deterministic content-based one."""
    return _default_record_key(record) if record.record_key is None else record.record_key


_record_key_or_default = record_key_for


def split_multi(
    record: "Record",
    *,
    record_key: Hashable | None = None,
    header: Any = None,
    annotation_version: Hashable | None = None,
    infer: bool = True,
    numbers: dict[str, str] | None = None,
) -> tuple[list["Record"], AlleleMapping]:
    """Split one multi-allelic record into one record per ALT allele.

    All allele-indexed INFO and FORMAT fields (plus ``GT``) are projected onto
    the ALT allele of each output record.  The returned mapping records, for
    every output allele, the original REF/ALT allele and annotation version.
    ``record_key`` defaults to ``record.record_key``; output record keys are
    ``(source_key, alt_index)`` so that duplicate VCF IDs cannot collide.

    Values that cannot be projected (for example a genotype that uses two
    different ALTs) are kept in ``mapping.unmatched``; they are not silently
    attached to the split row by their old array index.
    """
    source_key = _record_key_or_default(record) if record_key is None else record_key
    n_alts = len(record.ALT)
    mapping = AlleleMapping(annotation_version)
    mapping.register_record_object(record, source_key, annotation_version=annotation_version)
    if n_alts == 0:
        return [], mapping

    outputs: list[Record] = []
    reported_lossy: set[tuple[Any, ...]] = set()
    from vcfpy.record import Call, Record, UnparsedCall

    def report_lossy_g(field: str, value: Any, sample: str | None, alt_index: int):
        key = (field, sample)
        if key in reported_lossy:
            return
        reported_lossy.add(key)
        mapping.report_unmatched(
            field,
            _copy_value(value),
            "Number=G value contains genotypes combining several ALT alleles; a split row can only carry genotypes involving its own ALT",
            output_key=AlleleKey(out_key, 0),
            source_key=AlleleKey(source_key, alt_index),
            sample=sample,
            annotation_version=annotation_version,
        )

    for alt_index in range(1, n_alts + 1):
        out_key = (source_key, alt_index) if not isinstance(source_key, tuple) else source_key + (alt_index,)
        info: dict[str, Any] = {}
        for key, value in record.INFO.items():
            number = resolve_field_number(
                header,
                "INFO",
                key,
                value,
                n_alts + 1,
                infer=infer,
                numbers=numbers,
            )
            if number is None:
                info[key] = _copy_value(value)
                continue
            if number == NUMBER_GENOTYPES and isinstance(value, list) and _has_cross_alt_genotype(n_alts, 2):
                report_lossy_g("INFO/{}".format(key), value, None, alt_index)
            sliced, reason = slice_index_vector(value, number, alt_index, n_alts + 1)
            if reason is not None:
                mapping.report_unmatched(
                    "INFO/{}".format(key),
                    _copy_value(value),
                    reason,
                    output_key=AlleleKey(out_key, 1),
                    source_key=AlleleKey(source_key, alt_index),
                    annotation_version=annotation_version,
                )
                continue
            info[key] = sliced

        calls: list[Call | UnparsedCall] = []
        for call in record.calls:
            if isinstance(call, UnparsedCall):
                mapping.report_unmatched(
                    "FORMAT",
                    call.unparsed_data,
                    "unparsed call cannot be split by allele",
                    output_key=AlleleKey(out_key, 1),
                    source_key=AlleleKey(source_key, alt_index),
                    sample=call.sample,
                    annotation_version=annotation_version,
                )
                calls.append(UnparsedCall(call.sample, call.unparsed_data))
                continue
            ploidy = len(call.gt_alleles) if call.gt_alleles is not None else 2
            data: dict[str, Any] = {}
            for key, value in call.data.items():
                if key == "GT":
                    new_gt, unmatched_gt = split_gt(value, alt_index, n_alts)
                    if unmatched_gt is not None:
                        mapping.report_unmatched(
                            "FORMAT/GT",
                            value,
                            unmatched_gt,
                            output_key=AlleleKey(out_key, 1),
                            source_key=AlleleKey(source_key, alt_index),
                            sample=call.sample,
                            annotation_version=annotation_version,
                        )
                    data[key] = new_gt
                    continue
                number = resolve_field_number(
                    header,
                    "FORMAT",
                    key,
                    value,
                    n_alts + 1,
                    ploidy=ploidy,
                    infer=infer,
                    numbers=numbers,
                )
                if number is None:
                    data[key] = _copy_value(value)
                    continue
                if number == NUMBER_GENOTYPES and isinstance(value, list) and _has_cross_alt_genotype(n_alts, ploidy):
                    report_lossy_g("FORMAT/{}".format(key), value, call.sample, alt_index)
                sliced, reason = slice_index_vector(value, number, alt_index, n_alts + 1, ploidy=ploidy)
                if reason is not None:
                    mapping.report_unmatched(
                        "FORMAT/{}".format(key),
                        _copy_value(value),
                        reason,
                        output_key=AlleleKey(out_key, 1),
                        source_key=AlleleKey(source_key, alt_index),
                        sample=call.sample,
                        annotation_version=annotation_version,
                    )
                    continue
                data[key] = sliced
            calls.append(Call(call.sample, data))

        out_record = Record(
            record.CHROM,
            record.POS,
            list(record.ID),
            record.REF,
            [record.ALT[alt_index - 1]],
            record.QUAL,
            list(record.FILTER),
            info,
            list(record.FORMAT) if record.FORMAT else None,
            calls if record.calls else None,
        )
        out_record.record_key = out_key
        out_record.allele_mapping = mapping
        outputs.append(out_record)
        mapping.register_record(
            out_key,
            record_ids=record.ID,
            chrom=record.CHROM,
            pos=record.POS,
            ref=record.REF,
            alt_values=(record.ALT[alt_index - 1].serialize(),),
            annotation_version=annotation_version,
        )
        # Every split row keeps the original REF, and its single ALT stems
        # from the corresponding original ALT allele.
        mapping.add(
            AlleleKey(out_key, REF_ALLELE),
            AlleleKey(source_key, REF_ALLELE),
            annotation_version=annotation_version,
            record_ids=record.ID,
        )
        mapping.add(
            AlleleKey(out_key, 1),
            AlleleKey(source_key, alt_index),
            annotation_version=annotation_version,
            record_ids=record.ID,
        )
    return outputs, mapping


def _allele_value(record: "Record", allele_index: int) -> str:
    if allele_index == REF_ALLELE:
        return record.REF
    return record.ALT[allele_index - 1].serialize()


def _unique_keys(keys: list[Hashable], explicit: bool, what: str) -> list[Hashable]:
    """Disambiguate repeated default keys; reject repeated explicit keys."""
    if len(set(keys)) == len(keys):
        return keys
    if explicit:
        raise ValueError("{} record keys must be unique, got {!r}".format(what, keys))
    seen: dict[Hashable, int] = {}
    result: list[Hashable] = []
    for key in keys:
        count = seen.get(key, 0)
        seen[key] = count + 1
        result.append(key if count == 0 else (key, count))
    return result


def infer_mapping(
    source_records: Sequence["Record"],
    output_records: Sequence["Record"],
    *,
    source_keys: Sequence[Hashable] | None = None,
    output_keys: Sequence[Hashable] | None = None,
    source_versions: Sequence[Hashable | None] | dict[Hashable, Hashable | None] | None = None,
    annotation_version: Hashable | None = None,
    match_chrom: bool = True,
    match_pos: bool = True,
    match_ref: bool = True,
) -> AlleleMapping:
    """Infer an output -> source allele mapping for split/merged records.

    Alleles are matched by value at the same locus, never by record ID or by
    array position.  This makes the result stable when IDs are duplicated
    (split rows) or combined (merged rows), and lets identical ALTs from
    several source records map to one output allele.

    Records whose allele value has no counterpart are reported in
    :py:attr:`AlleleMapping.unmatched` in both directions.
    """
    sources = list(source_records)
    outputs = list(output_records)
    explicit_source_keys = source_keys is not None
    explicit_output_keys = output_keys is not None
    if source_keys is None:
        source_keys = [_record_key_or_default(record) for record in sources]
    if output_keys is None:
        output_keys = [_record_key_or_default(record) for record in outputs]
    if len(source_keys) != len(sources):
        raise ValueError("source_keys must have one entry per source record")
    if len(output_keys) != len(outputs):
        raise ValueError("output_keys must have one entry per output record")
    source_keys = _unique_keys(source_keys, explicit_source_keys, "source")
    output_keys = _unique_keys(output_keys, explicit_output_keys, "output")

    def version_for(key: Hashable, idx: int) -> Hashable | None:
        if isinstance(source_versions, dict):
            return source_versions.get(key)
        if source_versions is None:
            return annotation_version
        return source_versions[idx]

    mapping = AlleleMapping(annotation_version)
    source_index: dict[tuple[str, Any, Any, str], list[AlleleKey]] = {}
    for idx, (record, key) in enumerate(zip(sources, source_keys, strict=True)):
        version = version_for(key, idx)
        mapping.register_record_object(record, key, annotation_version=version)
        for allele_index in range(len(record.ALT) + 1):
            lookup = (
                record.CHROM if match_chrom else None,
                record.POS if match_pos else None,
                record.REF if match_ref else None,
                _allele_value(record, allele_index),
            )
            source_index.setdefault(lookup, []).append(AlleleKey(key, allele_index))

    matched_sources: set[AlleleKey] = set()
    for idx, (record, key) in enumerate(zip(outputs, output_keys, strict=True)):
        mapping.register_record_object(record, key, annotation_version=annotation_version)
        for allele_index in range(len(record.ALT) + 1):
            lookup = (
                record.CHROM if match_chrom else None,
                record.POS if match_pos else None,
                record.REF if match_ref else None,
                _allele_value(record, allele_index),
            )
            matches = source_index.get(lookup, [])
            if not matches:
                mapping.report_unmatched(
                    "ALT" if allele_index else "REF",
                    _allele_value(record, allele_index),
                    "no source allele with the same value at this locus",
                    output_key=AlleleKey(key, allele_index),
                    annotation_version=annotation_version,
                )
                continue
            for source_key in matches:
                meta = mapping.records.get(source_key.record_key)
                mapping.add(
                    AlleleKey(key, allele_index),
                    source_key,
                    annotation_version=None if meta is None else meta.annotation_version,
                    record_ids=() if meta is None else meta.record_ids,
                )
                matched_sources.add(source_key)

    for lookup, keys in source_index.items():
        for source_key in keys:
            if source_key not in matched_sources:
                mapping.report_unmatched(
                    "ALT" if source_key.allele_index else "REF",
                    lookup[-1],
                    "source allele has no output allele with the same value at this locus",
                    source_key=source_key,
                    annotation_version=annotation_version,
                )
    return mapping


# ---------------------------------------------------------------------------
# Merging records at the same locus

#: Sentinel for "no source record provided this position".
MISSING = object()


def _same_value(left: Any, right: Any) -> bool:
    return left == right


def _merge_non_allele(
    mapping: AlleleMapping,
    field: str,
    values: list[tuple[Hashable, Any]],
    *,
    output_key: AlleleKey | None = None,
    sample: str | None = None,
    annotation_version: Hashable | None = None,
    on_conflict: str = "first",
) -> Any:
    """Combine a non-allele-indexed field from several source records."""
    if not values:
        return None
    _, chosen = values[0]
    for _, value in values[1:]:
        if not _same_value(chosen, value):
            mapping.report_unmatched(
                field,
                {"kept": _copy_value(chosen), "discarded": _copy_value(value)},
                "conflicting values for a non-allele field while merging",
                output_key=output_key,
                sample=sample,
                annotation_version=annotation_version,
            )
            if on_conflict == "error":
                raise ValueError("cannot merge {}: conflicting values".format(field))
            if on_conflict == "last":
                chosen = value
    return _copy_value(chosen)


def _place(
    mapping: AlleleMapping,
    result: list[Any],
    index: int,
    value: Any,
    *,
    field: str,
    source_key: Hashable,
    output_key: AlleleKey,
    sample: str | None,
    annotation_version: Hashable | None,
    on_conflict: str,
) -> None:
    """Place one value into a merged allele-indexed vector."""
    current = result[index]
    if current is MISSING:
        result[index] = _copy_value(value)
        return
    if _same_value(current, value):
        return
    mapping.report_unmatched(
        field,
        {"kept": _copy_value(current), "discarded": _copy_value(value)},
        "conflicting values for the same output allele while merging",
        output_key=output_key,
        source_key=AlleleKey(source_key, 0),
        sample=sample,
        annotation_version=annotation_version,
    )
    if on_conflict == "error":
        raise ValueError("cannot merge {} at output allele {}: conflicting values".format(field, output_key))
    if on_conflict == "last":
        result[index] = _copy_value(value)


def _combined_ids(records: Sequence["Record"]) -> list[str]:
    result: list[str] = []
    for record in records:
        for id_ in record.ID:
            if id_ not in result:
                result.append(id_)
    return result


def _merge_qual(records: Sequence["Record"], mapping: AlleleMapping, record_key: Hashable) -> float | None:
    values = [record.QUAL for record in records if record.QUAL is not None]
    if not values:
        return None
    if len(set(values)) == 1:
        return values[0]
    mapping.report_unmatched(
        "QUAL",
        [record.QUAL for record in records],
        "records at the same locus have different QUAL values; highest value kept",
        output_key=AlleleKey(record_key, 0),
    )
    return max(values)


def _merge_filters(records: Sequence["Record"]) -> list[str]:
    result: list[str] = []
    for record in records:
        for filter_ in record.FILTER:
            if filter_ not in result:
                result.append(filter_)
    if "PASS" in result and len(result) > 1:
        result = [filter_ for filter_ in result if filter_ != "PASS"]
    return result


def _merge_gt(
    mapping: AlleleMapping,
    genotypes: list[tuple[Hashable, str, list[int | None], str]],
    *,
    output_key: AlleleKey,
    sample: str,
    annotation_version: Hashable | None,
) -> str | None:
    """Combine genotypes of one sample from several records at one locus.

    Each input tuple is ``(source_key, original_gt, remapped_alleles, phase)``
    where ``remapped_alleles`` already use output allele numbers.
    """
    if not genotypes:
        return None
    witnesses = {(tuple(alleles), phase) for _, _, alleles, phase in genotypes}
    if len(witnesses) == 1:
        alleles, phase = next(iter(witnesses))
        return _format_gt(alleles, phase)
    if len({phase for _, _, _, phase in genotypes}) > 1:
        # Combining phased and unphased evidence would invent phasing that no
        # source record asserted.  Keep the first genotype and report it.
        mapping.report_unmatched(
            "FORMAT/GT",
            [gt for _, gt, _, _ in genotypes],
            "phased and unphased genotypes cannot be combined without inventing phase",
            output_key=output_key,
            sample=sample,
            annotation_version=annotation_version,
        )
        return genotypes[0][1]

    max_ploidy = max(len(alleles) for _, _, alleles, _ in genotypes)
    phase = genotypes[0][3]
    non_ref: list[int] = []
    for _, _, alleles, _ in genotypes:
        for allele in alleles:
            if allele not in (None, 0) and allele not in non_ref:
                non_ref.append(allele)

    if max_ploidy == 1:
        if len(non_ref) > 1:
            mapping.report_unmatched(
                "FORMAT/GT",
                [gt for _, gt, _, _ in genotypes],
                "cannot combine haploid genotypes with different ALT alleles",
                output_key=output_key,
                sample=sample,
                annotation_version=annotation_version,
            )
            return genotypes[0][1]
        return _format_gt([0 if not non_ref else non_ref[0]], phase)
    if max_ploidy == 2:
        if len(non_ref) > 2:
            mapping.report_unmatched(
                "FORMAT/GT",
                [gt for _, gt, _, _ in genotypes],
                "cannot combine genotypes with more than two distinct ALT alleles at diploid ploidy",
                output_key=output_key,
                sample=sample,
                annotation_version=annotation_version,
            )
            return genotypes[0][1]
        if len(non_ref) == 2:
            return _format_gt([non_ref[0], non_ref[1]], phase)
        if len(non_ref) == 1:
            allele = non_ref[0]
            homozygous = any(
                len(alleles) == 2 and all(value == allele for value in alleles) for _, _, alleles, _ in genotypes
            )
            return _format_gt([allele, allele] if homozygous else [0, allele], phase)
        return _format_gt([0, 0], phase)

    mapping.report_unmatched(
        "FORMAT/GT",
        [gt for _, gt, _, _ in genotypes],
        "automatic genotype merging is only implemented for ploidy 1 and 2",
        output_key=output_key,
        sample=sample,
        annotation_version=annotation_version,
    )
    return genotypes[0][1]


def _format_gt(alleles: Sequence[int | None], phase: str) -> str:
    return phase.join("." if allele is None else str(allele) for allele in alleles)


def _gt_ploidy(gt: str | None) -> int:
    if gt is None:
        return 2
    return len([part for part in GT_SPLIT.split(gt) if part not in ("|", "/")])


def _gt_alleles(gt: Any) -> list[int | None]:
    if gt is None:
        return []
    return [None if part == "." else int(part) for part in GT_SPLIT.split(str(gt)) if part not in ("|", "/")]


def _merge_multi_impl(
    records: Sequence["Record"],
    *,
    record_key: Hashable | None,
    header: Any,
    annotation_version: Hashable | None,
    source_versions: Sequence[Hashable | None] | None,
    numbers: dict[str, str] | None,
    on_conflict: str,
    on_unparsed: str,
    require_same_ref: bool,
) -> tuple["Record", AlleleMapping]:
    from vcfpy.record import Call, Record, UnparsedCall

    sources = list(records)
    if not sources:
        raise ValueError("merge_multi requires at least one record")
    first = sources[0]
    for other in sources[1:]:
        if other.CHROM != first.CHROM or other.POS != first.POS:
            raise ValueError(
                "merge_multi requires all records at the same locus, got {}:{} and {}:{}".format(
                    first.CHROM, first.POS, other.CHROM, other.POS
                )
            )
        if require_same_ref and other.REF != first.REF:
            raise ValueError(
                "merge_multi requires identical REF alleles, got {!r} and {!r}".format(first.REF, other.REF)
            )

    explicit_keys = [record.record_key is not None for record in sources]
    source_keys = [_record_key_or_default(record) for record in sources]
    if len(set(source_keys)) != len(source_keys):
        duplicate_explicit = any(
            explicit_keys[idx] and source_keys.count(key) > 1
            for idx, key in enumerate(source_keys)
        )
        if duplicate_explicit:
            raise ValueError("explicit source record keys must be unique, got {!r}".format(source_keys))
        # Content-identical rows (same locus/alleles) still need distinct
        # source identities for the mapping; only the rows are ambiguous, not
        # their alleles, so suffix them in input order.
        seen: dict[Hashable, int] = {}
        unique_keys: list[Hashable] = []
        for key in source_keys:
            count = seen.get(key, 0)
            seen[key] = count + 1
            unique_keys.append(key if count == 0 else (key, count))
        source_keys = unique_keys
    if record_key is None:
        record_key = ("merged",) + tuple(source_keys)

    # Output ALT order follows first appearance; identical ALTs merge.
    alt_values: list[str] = []
    alt_lookup: dict[str, Any] = {}
    maps: list[dict[int, int]] = []
    for record in sources:
        record_map = {REF_ALLELE: REF_ALLELE}
        for old_index, alt in enumerate(record.ALT, start=1):
            value = alt.serialize()
            if value not in alt_values:
                alt_values.append(value)
                alt_lookup[value] = alt
            record_map[old_index] = alt_values.index(value) + 1
        maps.append(record_map)
    n_alleles = len(alt_values) + 1

    mapping = AlleleMapping(annotation_version)
    mapping.register_record(
        record_key,
        record_ids=_combined_ids(sources),
        chrom=first.CHROM,
        pos=first.POS,
        ref=first.REF,
        alt_values=alt_values,
        annotation_version=annotation_version,
    )
    for idx, (record, source_key) in enumerate(zip(sources, source_keys, strict=True)):
        if source_versions is not None:
            version = source_versions[idx]
        else:
            # Preserve the annotation version a source row already carries
            # (e.g. from an earlier split) instead of pretending it was
            # produced by this merge.
            existing = getattr(record, "allele_mapping", None)
            version = getattr(existing, "annotation_version", None)
            if version is None:
                version = annotation_version
        mapping.register_record_object(record, source_key, annotation_version=version)
        for old_index in range(len(record.ALT) + 1):
            mapping.add(
                AlleleKey(record_key, maps[idx][old_index]),
                AlleleKey(source_key, old_index),
                annotation_version=version,
                record_ids=record.ID,
            )

    def resolve_number(
        section: str,
        key: str,
        declared: str | None,
        samples: Sequence[tuple[int, Any, int | None]],
    ) -> str | None:
        """Resolve A/R/G, allowing per-record inference when undeclared."""
        if declared in ALLELE_NUMBERS:
            return declared
        inferred = set()
        for source_idx, value, ploidy in samples:
            inferred.add(
                resolve_field_number(
                    header,
                    section,
                    key,
                    value,
                    len(sources[source_idx].ALT) + 1,
                    ploidy=ploidy,
                    infer=True,
                    numbers=numbers,
                )
            )
        return inferred.pop() if len(inferred) == 1 else None

    def merge_vector(
        field: str,
        number: str,
        entries: Sequence[tuple[int, Hashable, Any, int | None]],
        *,
        sample: str | None,
    ) -> list[Any]:
        """Expand per-record allele-indexed arrays into the merged set."""
        if number == NUMBER_GENOTYPES:
            ploidy = entries[0][3] if entries and entries[0][3] else 2
            result: list[Any] = [MISSING] * genotype_count(n_alleles, ploidy)
            for source_idx, source_key, value, _ in entries:
                old_order = genotype_order(len(sources[source_idx].ALT) + 1, ploidy)
                if not isinstance(value, list) or len(value) != len(old_order):
                    mapping.report_unmatched(
                        field,
                        value,
                        "expected {} Number=G values, found {}".format(
                            len(old_order), len(value) if isinstance(value, list) else 1
                        ),
                        output_key=AlleleKey(record_key, 0),
                        source_key=AlleleKey(source_key, 0),
                        sample=sample,
                        annotation_version=annotation_version,
                    )
                    continue
                new_order = genotype_order(n_alleles, ploidy)
                new_index = {genotype: pos for pos, genotype in enumerate(new_order)}
                for old_pos, genotype in enumerate(old_order):
                    mapped_genotype = tuple(sorted(maps[source_idx][allele] for allele in genotype))
                    _place(
                        mapping,
                        result,
                        new_index[mapped_genotype],
                        value[old_pos],
                        field=field,
                        source_key=source_key,
                        output_key=AlleleKey(record_key, 0),
                        sample=sample,
                        annotation_version=annotation_version,
                        on_conflict=on_conflict,
                    )
            return [None if value is MISSING else value for value in result]

        expected = len(alt_values) if number == NUMBER_ALLELES else n_alleles
        result = [MISSING] * expected
        for source_idx, source_key, value, _ in entries:
            source_alts = len(sources[source_idx].ALT)
            source_expected = source_alts if number == NUMBER_ALLELES else source_alts + 1
            if not isinstance(value, list) or len(value) != source_expected:
                mapping.report_unmatched(
                    field,
                    value,
                    "expected {} Number={} values, found {}".format(
                        source_expected, number, len(value) if isinstance(value, list) else 1
                    ),
                    output_key=AlleleKey(record_key, 0),
                    source_key=AlleleKey(source_key, 0),
                    sample=sample,
                    annotation_version=annotation_version,
                )
                continue
            for old_index in range(source_expected):
                if number == NUMBER_REF:
                    # Number=R has REF at index 0, ALT k at index k.
                    new_index = 0 if old_index == 0 else maps[source_idx][old_index]
                else:
                    # Number=A only contains ALTs, at index k - 1.
                    new_index = maps[source_idx][old_index + 1] - 1
                _place(
                    mapping,
                    result,
                    new_index,
                    value[old_index],
                    field=field,
                    source_key=source_key,
                    output_key=AlleleKey(record_key, 0),
                    sample=sample,
                    annotation_version=annotation_version,
                    on_conflict=on_conflict,
                )
        return [None if value is MISSING else value for value in result]

    # --- INFO ---------------------------------------------------------
    info: dict[str, Any] = {}
    info_keys: list[str] = []
    for record in sources:
        for key in record.INFO:
            if key not in info_keys:
                info_keys.append(key)
    for key in info_keys:
        entries = [(idx, record.INFO[key]) for idx, record in enumerate(sources) if key in record.INFO]
        declared = get_field_number(header, "INFO", key)
        number = resolve_number("INFO", key, declared, [(idx, value, None) for idx, value in entries])
        if number is None:
            info[key] = _merge_non_allele(
                mapping,
                "INFO/{}".format(key),
                [(source_keys[idx], value) for idx, value in entries],
                output_key=AlleleKey(record_key, 0),
                annotation_version=annotation_version,
                on_conflict=on_conflict,
            )
        else:
            info[key] = merge_vector(
                "INFO/{}".format(key),
                number,
                [(idx, source_keys[idx], value, None) for idx, value in entries],
                sample=None,
            )

    # --- FORMAT / calls ----------------------------------------------
    format_keys: list[str] = []
    for record in sources:
        for key in record.FORMAT:
            if key not in format_keys:
                format_keys.append(key)
    sample_names: list[str] = []
    for record in sources:
        for call in record.calls:
            if call.sample not in sample_names:
                sample_names.append(call.sample)

    calls: list[Call | UnparsedCall] = []
    for sample in sample_names:
        sample_calls: list[tuple[int, Hashable, Any]] = []
        for idx, record in enumerate(sources):
            call = record.call_for_sample.get(sample)
            if call is not None:
                sample_calls.append((idx, source_keys[idx], call))
        unparsed = [(idx, key, call) for idx, key, call in sample_calls if isinstance(call, UnparsedCall)]
        if unparsed:
            if on_unparsed == "error":
                raise ValueError(
                    "cannot merge sample {!r}: UnparsedCall data cannot be remapped by allele".format(sample)
                )
            mapping.report_unmatched(
                "FORMAT",
                [call.unparsed_data for _, _, call in unparsed],
                "unparsed call cannot be merged by allele",
                output_key=AlleleKey(record_key, 0),
                sample=sample,
                annotation_version=annotation_version,
            )
            calls.append(UnparsedCall(sample, unparsed[0][2].unparsed_data))
            continue
        parsed: list[tuple[int, Hashable, Call]] = [(idx, key, call) for idx, key, call in sample_calls]

        # GT: remap every source genotype to output allele numbers first.
        gt_entries: list[tuple[Hashable, str, list[int | None], str]] = []
        for source_idx, source_key, call in parsed:
            if "GT" not in call.data or call.data["GT"] is None:
                continue
            original = str(call.data["GT"])
            alleles: list[int | None] = []
            failed = None
            for allele in _gt_alleles(original):
                if allele is None:
                    alleles.append(None)
                elif allele == REF_ALLELE:
                    alleles.append(REF_ALLELE)
                elif 1 <= allele <= len(sources[source_idx].ALT):
                    alleles.append(maps[source_idx][allele])
                else:
                    failed = "GT allele {} is outside 0..{}".format(allele, len(sources[source_idx].ALT))
                    break
            if failed is not None:
                mapping.report_unmatched(
                    "FORMAT/GT",
                    original,
                    failed,
                    output_key=AlleleKey(record_key, 0),
                    source_key=AlleleKey(source_key, 0),
                    sample=sample,
                    annotation_version=annotation_version,
                )
                continue
            phase = "|" if "|" in original else "/"
            gt_entries.append((source_key, original, alleles, phase))
        gt = _merge_gt(
            mapping, gt_entries, output_key=AlleleKey(record_key, 0), sample=sample, annotation_version=annotation_version
        )
        data: dict[str, Any] = {}
        if gt is not None:
            data["GT"] = gt
        ploidy = _gt_ploidy(gt)

        for key in format_keys:
            if key == "GT":
                continue
            entries = [
                (source_idx, source_key, call.data[key], _gt_ploidy(call.data.get("GT")))
                for source_idx, source_key, call in parsed
                if key in call.data
            ]
            if not entries:
                continue
            declared = get_field_number(header, "FORMAT", key)
            number = resolve_number(
                "FORMAT", key, declared, [(idx, value, call_ploidy) for idx, _, value, call_ploidy in entries]
            )
            if number is None:
                data[key] = _merge_non_allele(
                    mapping,
                    "FORMAT/{}".format(key),
                    [(source_key, value) for _, source_key, value, _ in entries],
                    output_key=AlleleKey(record_key, 0),
                    sample=sample,
                    annotation_version=annotation_version,
                    on_conflict=on_conflict,
                )
            else:
                if number == NUMBER_GENOTYPES:
                    entries = [(idx, key_, value, ploidy) for idx, key_, value, _ in entries]
                data[key] = merge_vector("FORMAT/{}".format(key), number, entries, sample=sample)
        calls.append(Call(sample, data))

    merged = Record(
        first.CHROM,
        first.POS,
        _combined_ids(sources),
        first.REF,
        [alt_lookup[value] for value in alt_values],
        _merge_qual(sources, mapping, record_key),
        _merge_filters(sources),
        info,
        format_keys or None,
        calls if sample_names else None,
    )
    merged.record_key = record_key
    merged.allele_mapping = mapping
    return merged, mapping


def merge_multi(
    records: Sequence["Record"],
    *,
    record_key: Hashable | None = None,
    header: Any = None,
    annotation_version: Hashable | None = None,
    source_versions: Sequence[Hashable | None] | None = None,
    numbers: dict[str, str] | None = None,
    on_conflict: str = "first",
    on_unparsed: str = "error",
    require_same_ref: bool = True,
) -> tuple["Record", AlleleMapping]:
    """Merge records at the same locus into one multi-allelic record.

    All records must share ``CHROM``, ``POS`` and (by default) ``REF``.  The
    output ALT order follows first appearance across the input records and
    identical ALT values become one output allele.

    ``Number=A``/``R``/``G`` annotations are expanded to the merged allele
    set by mapping every source allele to its output allele.  Positions not
    covered by any source record stay ``None``.  Conflicting values for the
    same output allele are reported in :py:attr:`AlleleMapping.unmatched`
    (and raise when ``on_conflict="error"``).

    Record IDs are combined (de-duplicated, first-seen order) but never used
    to identify input rows.  The returned mapping maps every output allele to
    all original alleles it was built from, together with each source's
    annotation version.
    """
    if on_conflict not in ("first", "last", "error"):
        raise ValueError('on_conflict must be "first", "last" or "error"')
    if on_unparsed not in ("error", "keep"):
        raise ValueError('on_unparsed must be "error" or "keep"')
    return _merge_multi_impl(
        records,
        record_key=record_key,
        header=header,
        annotation_version=annotation_version,
        source_versions=source_versions,
        numbers=numbers,
        on_conflict=on_conflict,
        on_unparsed=on_unparsed,
        require_same_ref=require_same_ref,
    )
