# -*- coding: utf-8 -*-
"""Normalization of sequence alleles and stable business keys.

The functions in this module implement the usual VCF rule that an indel in a
repeat can be written in several equivalent ways.  All equivalent forms are
left-aligned to a single canonical representation.  A canonical representation
is turned into a :class:`VariantKey` (the "business key") that also contains the
reference version.  Consequently, identical truncated REF/ALT strings from
different reference versions never collapse into the same key.

When the supplied reference window does not contain enough left context to
prove that the left-most representation has been reached, no business key is
returned.  Instead, :class:`UnresolvedRange` describes the interval in which
the canonical position may lie.  The range is also scoped to the reference
version for the same reason.

The module deliberately works on plain strings and does not require an indexed
reference.  Callers provide a :class:`ReferenceWindow`, i.e. a reference
fragment together with its 0-based start coordinate and a reference version
identifier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping, NamedTuple

__all__ = [
    "Allele",
    "AlleleMapping",
    "Alignment",
    "GroupedNormalization",
    "NormalizationBatch",
    "NormalizationResult",
    "NormalizedAllele",
    "NormalizedVariant",
    "ReferenceContext",
    "ReferenceMismatchError",
    "ReferenceSequence",
    "ReferenceWindow",
    "ReferenceWindowError",
    "UnresolvedRange",
    "VariantKey",
    "VariantNormalization",
    "business_key",
    "canonical_key",
    "canonicalize",
    "left_align",
    "left_align_allele",
    "left_align_variant",
    "make_business_key",
    "normalize",
    "normalize_allele",
    "normalize_alleles",
    "normalize_and_map",
    "normalize_record",
    "normalize_variant",
    "normalize_variants",
    "trim_alleles",
]

#: Characters that can be handled as literal sequence alleles.
_SEQUENCE_RE = re.compile(r"^[A-Za-z*]+$")
#: Special ALT values that are kept verbatim and never shifted.
_NON_LITERAL_ALT = {"*", ".", ""}


class ReferenceWindowError(ValueError):
    """Raised when a variant does not fit into the supplied reference window."""


class ReferenceMismatchError(ReferenceWindowError):
    """Raised when the REF allele does not match the supplied reference."""


@dataclass(frozen=True, init=False)
class ReferenceWindow:
    """A reference sequence fragment together with its coordinate bounds.

    :param sequence: reference bases of the window
    :param start: 0-based coordinate of the first base in ``sequence``
    :param reference: reference version/name, e.g. ``"GRCh38"``.  This value is
        part of every business key and is what prevents accidental merging of
        identical strings from different reference versions.
    :param contig: optional contig name; defaults to ``"."``
    :param complete_left: whether the window reaches the beginning of the
        contig.  By default this is inferred from ``start == 0``.  Set it to
        ``False`` explicitly for a window that starts at coordinate zero of a
        sub-sequence which is not the true chromosome start (for example a
        transcript or an amplicon reference).

    ``start`` is 0-based and ``end`` is exclusive, matching Python slices and
    ``Record.begin``.  Variant positions passed to this module are 1-based VCF
    ``POS`` values.
    """

    sequence: str
    start: int
    reference: str
    contig: str
    complete_left: bool | None

    def __init__(
        self,
        sequence: str,
        start: int = 0,
        reference: str = ".",
        contig: str = ".",
        complete_left: bool | None = None,
    ):
        if not isinstance(sequence, str) or not sequence:
            raise ValueError("ReferenceWindow.sequence must be a non-empty string")
        if not isinstance(start, int) or start < 0:
            raise ValueError("ReferenceWindow.start must be a non-negative integer")
        if not isinstance(reference, str) or not reference:
            raise ValueError("ReferenceWindow.reference must be a non-empty string")
        if not isinstance(contig, str) or not contig:
            raise ValueError("ReferenceWindow.contig must be a non-empty string")
        object.__setattr__(self, "sequence", sequence.upper())
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "contig", contig)
        object.__setattr__(self, "complete_left", complete_left)

    @classmethod
    def from_bounds(
        cls,
        sequence: str,
        start: int,
        end: int,
        reference: str = ".",
        contig: str = ".",
        complete_left: bool | None = None,
    ) -> "ReferenceWindow":
        """Create a window and check its exclusive ``end`` coordinate."""
        window = cls(
            sequence=sequence,
            start=start,
            reference=reference,
            contig=contig,
            complete_left=complete_left,
        )
        if end != window.end:
            raise ValueError(f"Window end {end} does not match start + length ({window.end})")
        return window

    @property
    def end(self) -> int:
        """0-based exclusive end coordinate of the window."""
        return self.start + len(self.sequence)

    @property
    def start_1based(self) -> int:
        """1-based inclusive start coordinate of the window."""
        return self.start + 1

    @property
    def end_1based(self) -> int:
        """1-based inclusive end coordinate of the window."""
        return self.end

    @property
    def version(self) -> str:
        """Alias for :attr:`reference`."""
        return self.reference

    @property
    def chrom(self) -> str:
        """Alias for :attr:`contig`."""
        return self.contig

    @property
    def has_left_context(self) -> bool:
        """Whether the window is known to reach the start of the contig."""
        if self.complete_left is None:
            return self.start == 0
        return self.complete_left

    def contains(self, pos: int, length: int = 1) -> bool:
        """Return whether a 1-based VCF interval fits into the window."""
        pos0 = pos - 1
        return self.start <= pos0 and pos0 + length <= self.end

    def base_at(self, pos: int) -> str:
        """Return the base at 0-based coordinate ``pos``.

        Raises :class:`ReferenceWindowError` if ``pos`` is outside the window.
        """
        if pos < self.start or pos >= self.end:
            raise ReferenceWindowError(
                f"Position {pos} (0-based) is outside reference window [{self.start}, {self.end})"
            )
        return self.sequence[pos - self.start]

    def slice(self, start: int, end: int) -> str:
        """Return reference bases for 0-based half-open interval ``[start, end)``."""
        if start < self.start or end > self.end or start > end:
            raise ReferenceWindowError(
                f"Interval [{start}, {end}) is outside reference window [{self.start}, {self.end})"
            )
        return self.sequence[start - self.start : end - self.start]

    def __str__(self) -> str:
        return (
            f"ReferenceWindow(reference={self.reference!r}, contig={self.contig!r}, "
            f"start={self.start}, end={self.end})"
        )


#: Compatibility alias for callers that prefer "context" terminology.
ReferenceContext = ReferenceWindow
#: Compatibility alias for callers that prefer "sequence" terminology.
ReferenceSequence = ReferenceWindow


class VariantKey(NamedTuple):
    """Stable business key of a fully determined variant.

    The tuple form is ``(reference, contig, pos, ref, alt)``.  Being a
    :class:`typing.NamedTuple`, it compares equal to a plain tuple with the
    same values as well as to another :class:`VariantKey`.
    """

    reference: str
    contig: str
    pos: int
    ref: str
    alt: str

    @property
    def chrom(self) -> str:
        """Alias for :attr:`contig`."""
        return self.contig

    @property
    def reference_version(self) -> str:
        """Alias for :attr:`reference`."""
        return self.reference

    @property
    def value(self) -> str:
        """String representation of the key."""
        return str(self)

    def as_tuple(self) -> tuple[str, str, int, str, str]:
        """Return the key as a plain tuple."""
        return tuple(self)  # type: ignore[return-value]

    def __str__(self) -> str:
        return f"{self.reference}:{self.contig}:{self.pos}:{self.ref}:{self.alt}"


#: Compatibility alias.
AlleleKey = VariantKey


@dataclass(frozen=True)
class Allele:
    """An original allele as it occurred in the input.

    ``pos`` is a 1-based VCF ``POS`` value.
    """

    pos: int
    ref: str
    alt: str

    @property
    def label(self) -> str:
        """Stable label used in mapping dictionaries."""
        return f"{self.pos}:{self.ref}>{self.alt}"

    def __str__(self) -> str:
        return self.label


@dataclass(frozen=True)
class NormalizedVariant:
    """A canonical, left-aligned variant."""

    reference: str
    contig: str
    pos: int
    ref: str
    alt: str

    @property
    def key(self) -> VariantKey:
        """Business key for this canonical variant."""
        return VariantKey(self.reference, self.contig, self.pos, self.ref, self.alt)

    @property
    def chrom(self) -> str:
        """Alias for :attr:`contig`."""
        return self.contig

    @property
    def reference_version(self) -> str:
        """Alias for :attr:`reference`."""
        return self.reference

    @property
    def allele(self) -> Allele:
        """Return the canonical allele as an :class:`Allele`."""
        return Allele(self.pos, self.ref, self.alt)

    def __str__(self) -> str:
        return f"{self.contig}:{self.pos}:{self.ref}>{self.alt}"


#: Compatibility alias.
NormalizedAllele = NormalizedVariant


@dataclass(frozen=True)
class UnresolvedRange:
    """Positions that cannot be distinguished with the available left context.

    ``start`` and ``end`` are 1-based inclusive VCF positions.  When the window
    is truncated on the left, ``open_left`` is true: the canonical position may
    lie anywhere from the first valid VCF position (``start == 1``) up to the
    original position.  ``verified_start`` is the earliest position whose left
    flank was actually present in the window, which is useful for diagnostics
    but must not be treated as the lower bound of the true range.

    The range is intentionally scoped to a reference version: an unresolved
    variant from ``GRCh37`` must never be merged with one from ``GRCh38`` even
    when the visible, truncated allele strings are identical.
    """

    reference: str
    contig: str
    start: int
    end: int
    ref: str
    alt: str
    reason: str = "insufficient_left_context"
    original_pos: int | None = None
    window_start: int | None = None
    window_end: int | None = None
    open_left: bool = False
    verified_start: int | None = None

    @property
    def pos_min(self) -> int:
        """Smallest possible 1-based position."""
        return self.start

    @property
    def left_truncated(self) -> bool:
        """Alias for :attr:`open_left`."""
        return self.open_left

    @property
    def pos_max(self) -> int:
        """Largest possible 1-based position."""
        return self.end

    @property
    def range(self) -> tuple[int, int]:
        """The inclusive ``(start, end)`` interval."""
        return (self.start, self.end)

    @property
    def positions(self) -> range:
        """All positions in the unresolved interval."""
        return range(self.start, self.end + 1)

    @property
    def reference_version(self) -> str:
        """Alias for :attr:`reference`."""
        return self.reference

    @property
    def chrom(self) -> str:
        """Alias for :attr:`contig`."""
        return self.contig

    @property
    def key(self) -> str:
        """Version-scoped provisional key for the unresolved range."""
        return f"{self.reference}:{self.contig}:{self.start}-{self.end}:{self.ref}:{self.alt}"

    @property
    def value(self) -> str:
        """Alias for :attr:`key`."""
        return self.key

    def as_tuple(self) -> tuple[int, int]:
        """Return ``(start, end)``."""
        return self.range

    def contains(self, pos: int) -> bool:
        """Return whether a 1-based position lies in the range."""
        return self.start <= pos <= self.end

    def __contains__(self, pos: object) -> bool:
        return isinstance(pos, int) and self.contains(pos)

    def __iter__(self) -> Iterator[int]:
        yield self.start
        yield self.end

    def __str__(self) -> str:
        return f"{self.contig}:{self.start}-{self.end}:{self.ref}>{self.alt}"


@dataclass(frozen=True)
class Alignment:
    """Result of the low-level left-alignment step."""

    pos: int
    ref: str
    alt: str
    unresolved_range: UnresolvedRange | None = None

    @property
    def resolved(self) -> bool:
        """Whether the alignment is fully determined."""
        return self.unresolved_range is None

    @property
    def unresolved(self) -> bool:
        """Whether left context was insufficient."""
        return self.unresolved_range is not None


@dataclass(frozen=True)
class NormalizationResult:
    """Normalization result for one original allele."""

    original: Allele
    normalized: NormalizedVariant | None
    unresolved_range: UnresolvedRange | None = None
    status: str = "normalized"

    @property
    def key(self) -> VariantKey | None:
        """Business key, or ``None`` for an unresolved variant."""
        return None if self.normalized is None else self.normalized.key

    @property
    def resolved(self) -> bool:
        """Whether a business key is available."""
        return self.normalized is not None

    @property
    def is_resolved(self) -> bool:
        """Alias for :attr:`resolved`."""
        return self.resolved

    @property
    def unresolved(self) -> bool:
        """Whether the variant could not be fully determined."""
        return self.unresolved_range is not None

    @property
    def is_unresolved(self) -> bool:
        """Alias for :attr:`unresolved`."""
        return self.unresolved

    @property
    def pos(self) -> int:
        """Canonical 1-based position, if resolved."""
        return self.original.pos if self.normalized is None else self.normalized.pos

    @property
    def ref(self) -> str:
        """Canonical REF, if resolved."""
        return self.original.ref if self.normalized is None else self.normalized.ref

    @property
    def alt(self) -> str:
        """Canonical ALT, if resolved."""
        return self.original.alt if self.normalized is None else self.normalized.alt

    @property
    def allele_map(self) -> dict[str, VariantKey | UnresolvedRange]:
        """Mapping from the original ALT string to its key/range."""
        target: VariantKey | UnresolvedRange | None = self.key or self.unresolved_range
        return {self.original.alt: target} if target is not None else {}

    @property
    def mapping(self) -> dict[str, VariantKey | UnresolvedRange]:
        """Alias for :attr:`allele_map`."""
        return self.allele_map

    @property
    def labeled_map(self) -> dict[str, VariantKey | UnresolvedRange]:
        """Mapping from the original ``POS:REF>ALT`` label to its key/range."""
        target: VariantKey | UnresolvedRange | None = self.key or self.unresolved_range
        return {self.original.label: target} if target is not None else {}

    @property
    def value(self) -> VariantKey | UnresolvedRange | None:
        """The key or unresolved range."""
        return self.key or self.unresolved_range

    def __bool__(self) -> bool:
        return self.resolved


#: Compatibility alias.
VariantNormalization = NormalizationResult


@dataclass
class NormalizationBatch:
    """A group of normalized alleles with the original-to-canonical mapping."""

    results: list[NormalizationResult]
    reference: str = "."
    contig: str = "."
    pos: int | None = None
    ref: str | None = None

    @property
    def resolved(self) -> list[NormalizationResult]:
        """All results that produced a business key."""
        return [r for r in self.results if r.resolved]

    @property
    def unresolved(self) -> list[NormalizationResult]:
        """All results without enough left context."""
        return [r for r in self.results if r.unresolved]

    @property
    def allele_map(self) -> dict[str, VariantKey | UnresolvedRange]:
        """Mapping from each original ALT string to its key/range."""
        return self.alt_map

    @property
    def mapping(self) -> dict[str, VariantKey | UnresolvedRange]:
        """Alias for :attr:`allele_map`."""
        return self.allele_map

    @property
    def original_to_key(self) -> dict[str, VariantKey | UnresolvedRange]:
        """Alias for :attr:`allele_map`."""
        return self.allele_map

    @property
    def alt_map(self) -> dict[str, VariantKey | UnresolvedRange]:
        """Mapping from original ALT string to key or unresolved range."""
        result: dict[str, VariantKey | UnresolvedRange] = {}
        for item in self.results:
            value = item.key or item.unresolved_range
            if value is not None:
                result[item.original.alt] = value
        return result

    @property
    def labeled_map(self) -> dict[str, VariantKey | UnresolvedRange]:
        """Mapping from original ``POS:REF>ALT`` label to key/range."""
        result: dict[str, VariantKey | UnresolvedRange] = {}
        for item in self.results:
            value = item.key or item.unresolved_range
            if value is not None:
                result[item.original.label] = value
        return result

    @property
    def by_key(self) -> dict[VariantKey, list[Allele]]:
        """Original alleles grouped by business key."""
        result: dict[VariantKey, list[Allele]] = {}
        for item in self.results:
            if item.key is not None:
                result.setdefault(item.key, []).append(item.original)
        return result

    @property
    def keys(self) -> list[VariantKey]:
        """Business keys in input order."""
        return [item.key for item in self.results if item.key is not None]  # type: ignore[misc]

    @property
    def groups(self) -> dict[VariantKey, list[Allele]]:
        """Original alleles grouped by business key."""
        return self.by_key

    @property
    def grouped(self) -> dict[VariantKey, list[Allele]]:
        """Alias for :attr:`groups`."""
        return self.groups

    @property
    def unresolved_groups(self) -> dict[str, list[str]]:
        """Group original allele labels by unresolved range key."""
        result: dict[str, list[str]] = {}
        for item in self.results:
            if item.unresolved_range is not None:
                result.setdefault(item.unresolved_range.key, []).append(item.original.label)
        return result

    def result_for(self, alt: str) -> NormalizationResult | None:
        """Return the result for an original ALT string."""
        for item in self.results:
            if item.original.alt == alt:
                return item
        return None

    def key_for(self, alt: str) -> VariantKey | None:
        """Return the business key for an original ALT string, if resolved."""
        item = self.result_for(alt)
        return None if item is None else item.key

    def __iter__(self) -> Iterator[NormalizationResult]:
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)


#: Compatibility alias.
GroupedNormalization = NormalizationBatch
#: Compatibility alias for the mapping returned by some callers.
AlleleMapping = NormalizationBatch


def _validate_allele(ref: str, alt: str) -> None:
    if not isinstance(ref, str) or not ref:
        raise ValueError("REF must be a non-empty string")
    if not isinstance(alt, str) or not alt:
        raise ValueError("ALT must be a non-empty string")


def _is_literal(allele: str) -> bool:
    return allele not in _NON_LITERAL_ALT and bool(_SEQUENCE_RE.match(allele))


def _check_ref(window: ReferenceWindow, pos0: int, ref: str) -> str:
    if pos0 < 0:
        raise ReferenceWindowError(f"POS must be >= 1, got {pos0 + 1}")
    if pos0 < window.start or pos0 + len(ref) > window.end:
        raise ReferenceWindowError(
            f"REF {ref!r} at 0-based position {pos0} does not fit reference window "
            f"[{window.start}, {window.end})"
        )
    observed = window.slice(pos0, pos0 + len(ref))
    if observed != ref:
        raise ReferenceMismatchError(
            f"REF {ref!r} at 0-based position {pos0} does not match reference {observed!r}"
        )
    return observed


def trim_alleles(pos: int, ref: str, alt: str) -> tuple[int, str, str]:
    """Trim common bases while keeping both alleles non-empty.

    ``pos`` is 1-based.  The function first removes a common suffix and then a
    common prefix, the same order used by common VCF normalization tools.
    """
    if not isinstance(pos, int) or pos < 1:
        raise ValueError("POS must be a positive integer")
    _validate_allele(ref, alt)
    if ref == alt:
        return pos, ref, alt
    while len(ref) > 1 and len(alt) > 1 and ref[-1] == alt[-1]:
        ref = ref[:-1]
        alt = alt[:-1]
    while len(ref) > 1 and len(alt) > 1 and ref[0] == alt[0]:
        pos += 1
        ref = ref[1:]
        alt = alt[1:]
    return pos, ref, alt


def _unresolved(
    window: ReferenceWindow,
    pos0: int,
    ref: str,
    alt: str,
    original_pos: int,
) -> UnresolvedRange:
    """Build the range reported when the left edge of the window is reached."""
    return UnresolvedRange(
        reference=window.reference,
        contig=window.contig,
        start=1,
        end=max(original_pos, pos0 + 1),
        ref=ref,
        alt=alt,
        reason="insufficient_left_context",
        original_pos=original_pos,
        window_start=window.start + 1,
        window_end=window.end,
        open_left=True,
        verified_start=window.start + 1,
    )


def _left_align(
    window: ReferenceWindow,
    pos0: int,
    ref: str,
    alt: str,
    original_pos: int,
) -> tuple[int, str, str, UnresolvedRange | None]:
    """Left-align alleles using the standard trim/shift loop.

    Returns ``(pos0, ref, alt, unresolved_range)``.  A shift is only possible
    while the alleles share their last base; when one allele has length one the
    shared base is removed and the preceding reference base is prepended.  If
    that preceding base is outside a truncated window, the variant is reported
    as unresolved rather than assigned a possibly wrong key.
    """
    while ref[-1] == alt[-1]:
        if len(ref) > 1 and len(alt) > 1:
            ref = ref[:-1]
            alt = alt[:-1]
            continue
        # One allele is already a single base.  Shifting left rotates the
        # shared base out and the preceding reference base in.
        if pos0 == window.start:
            if not window.has_left_context:
                return pos0, ref, alt, _unresolved(window, pos0, ref, alt, original_pos)
            break
        pos0 -= 1
        base = window.sequence[pos0 - window.start]
        ref = base + ref[:-1]
        alt = base + alt[:-1]
    return pos0, ref, alt, None


def left_align_allele(
    window: ReferenceWindow | str | None = None,
    pos: int | None = None,
    ref: str | None = None,
    alt: str | None = None,
    *,
    sequence: str | None = None,
    start: int = 0,
    reference: str | ReferenceWindow = ".",
    contig: str = ".",
    complete_left: bool | None = None,
) -> Alignment:
    """Left-align one allele and report unresolved ranges.

    The first positional argument may be a :class:`ReferenceWindow`, a
    reference sequence string, or ``None`` (in which case ``sequence`` and
    ``start`` must be given).
    """
    ctx, pos, ref, alt = _resolve_inputs(
        window,
        pos,
        ref,
        alt,
        sequence=sequence,
        start=start,
        reference=reference,
        contig=contig,
        complete_left=complete_left,
    )
    if pos is None or ref is None or alt is None:
        raise ValueError("pos, ref and alt are required")
    _validate_allele(ref, alt)
    pos0 = pos - 1
    ref_u = ref.upper()
    alt_u = alt.upper()
    _check_ref(ctx, pos0, ref_u)
    if not _is_literal(ref_u) or not _is_literal(alt_u):
        return Alignment(pos, ref, alt)
    # The left-alignment loop itself removes common suffixes while walking
    # through the repeat.  Only the final common prefix is trimmed afterwards.
    new_pos0, new_ref, new_alt, unresolved = _left_align(
        ctx,
        pos0,
        ref_u,
        alt_u,
        original_pos=pos,
    )
    if unresolved is not None:
        return Alignment(new_pos0 + 1, new_ref, new_alt, unresolved)
    new_pos, new_ref, new_alt = trim_alleles(new_pos0 + 1, new_ref, new_alt)
    new_ref, new_alt = new_ref.upper(), new_alt.upper()
    return Alignment(new_pos, new_ref, new_alt, unresolved)


#: Short alias.
left_align = left_align_allele
#: Compatibility alias.
left_align_variant = left_align_allele


def _resolve_inputs(
    window: ReferenceWindow | str | None,
    pos: int | None,
    ref: str | None,
    alt: str | None,
    *,
    sequence: str | None,
    start: int,
    reference: str | ReferenceWindow,
    contig: str,
    complete_left: bool | None,
) -> tuple[ReferenceWindow, int, str, str]:
    if isinstance(window, ReferenceWindow):
        ctx = window
    elif isinstance(reference, ReferenceWindow):
        ctx = reference
    else:
        seq = sequence
        if seq is None and isinstance(window, str):
            seq = window
        if seq is None:
            raise ValueError("Provide a ReferenceWindow, sequence= or a sequence as first argument")
        ctx = ReferenceWindow(
            sequence=seq,
            start=start,
            reference=reference,
            contig=contig,
            complete_left=complete_left,
        )
    return ctx, pos, ref, alt


def make_business_key(
    reference: str,
    contig: str,
    pos: int,
    ref: str,
    alt: str,
) -> VariantKey:
    """Build a version-scoped business key from canonical values."""
    if not reference or not contig:
        raise ValueError("reference and contig must be non-empty")
    if not isinstance(pos, int) or pos < 1:
        raise ValueError("pos must be a positive integer")
    _validate_allele(ref, alt)
    return VariantKey(reference, contig, pos, ref, alt)


#: Short aliases.
business_key = make_business_key
canonical_key = make_business_key


def normalize_variant(
    window: ReferenceWindow | str | None = None,
    pos: int | None = None,
    ref: str | None = None,
    alt: str | None = None,
    *,
    sequence: str | None = None,
    start: int = 0,
    reference: str | ReferenceWindow = ".",
    contig: str = ".",
    complete_left: bool | None = None,
) -> NormalizationResult:
    """Normalize one allele and return its business key or unresolved range.

    Equivalent inputs (for example the different placements of the same
    insertion in a short tandem repeat) produce the same :class:`VariantKey`.
    If the window does not contain enough left context, ``key`` is ``None`` and
    :attr:`NormalizationResult.unresolved_range` is populated.
    """
    ctx, pos, ref, alt = _resolve_inputs(
        window,
        pos,
        ref,
        alt,
        sequence=sequence,
        start=start,
        reference=reference,
        contig=contig,
        complete_left=complete_left,
    )
    if pos is None or ref is None or alt is None:
        raise ValueError("pos, ref and alt are required")
    _validate_allele(ref, alt)
    original = Allele(pos, ref, alt)
    pos0 = pos - 1
    ref_u = ref.upper()
    alt_u = alt.upper()
    _check_ref(ctx, pos0, ref_u)

    if alt_u == "." or ref_u == alt_u:
        normalized = NormalizedVariant(ctx.reference, ctx.contig, pos, ref_u, alt)
        return NormalizationResult(original, normalized, status="reference")
    if not _is_literal(ref_u) or not _is_literal(alt_u) or alt_u in {"*"}:
        # Symbolic alleles, breakends, spanning deletions and other special
        # values are not shifted.  They still receive a version-scoped key.
        normalized = NormalizedVariant(ctx.reference, ctx.contig, pos, ref_u, alt)
        return NormalizationResult(original, normalized, status="not_normalizable")

    alignment = left_align_allele(ctx, pos, ref, alt)
    if alignment.unresolved_range is not None:
        return NormalizationResult(
            original,
            None,
            unresolved_range=alignment.unresolved_range,
            status="unresolved",
        )

    normalized = NormalizedVariant(
        ctx.reference,
        ctx.contig,
        alignment.pos,
        alignment.ref,
        alignment.alt,
    )
    status = "unchanged" if (alignment.pos, alignment.ref, alignment.alt) == (pos, ref_u, alt_u) else "normalized"
    if alignment.ref == alignment.alt:
        status = "reference"
    return NormalizationResult(original, normalized, status=status)


#: Short aliases.
normalize = normalize_variant
normalize_allele = normalize_variant
canonicalize = normalize_variant


def normalize_alleles(
    window: ReferenceWindow | str | None = None,
    pos: int | None = None,
    ref: str | None = None,
    alts: Iterable[str] | None = None,
    *,
    sequence: str | None = None,
    start: int = 0,
    reference: str | ReferenceWindow = ".",
    contig: str = ".",
    complete_left: bool | None = None,
) -> NormalizationBatch:
    """Normalize all ALT alleles of one site and retain the original mapping.

    ``alts`` may be any iterable of ALT strings.  The returned batch exposes
    ``allele_map`` (original ``POS:REF>ALT`` label to key/range), ``alt_map``
    (original ALT string to key/range) and ``groups`` (key to original labels).
    """
    if alts is None:
        raise ValueError("alts must be provided")
    if isinstance(alts, str):
        alts = [alts]
    ctx, pos, ref, _ = _resolve_inputs(
        window,
        pos,
        ref,
        "",
        sequence=sequence,
        start=start,
        reference=reference,
        contig=contig,
        complete_left=complete_left,
    )
    if pos is None or ref is None:
        raise ValueError("pos and ref are required")
    results = [
        normalize_variant(
            ctx,
            pos,
            ref,
            alt,
            complete_left=complete_left,
        )
        for alt in alts
    ]
    return NormalizationBatch(list(results), reference=ctx.reference, contig=ctx.contig, pos=pos, ref=ref)


def _coerce_variants(variants: Any) -> list[Allele]:
    """Turn common variant containers into a list of :class:`Allele`."""
    if isinstance(variants, str):
        raise ValueError("A bare allele string has no position; pass (pos, ref, alt) tuples")
    if isinstance(variants, Mapping):
        pos = variants.get("pos", variants.get("POS"))
        ref = variants.get("ref", variants.get("REF"))
        alt = variants.get("alt", variants.get("ALT"))
        if pos is None or ref is None or alt is None:
            raise ValueError("Variant mapping must contain pos/ref/alt")
        if isinstance(alt, (list, tuple)):
            return [Allele(int(pos), str(ref), str(a)) for a in alt]
        return [Allele(int(pos), str(ref), str(alt))]
    if hasattr(variants, "REF") and hasattr(variants, "ALT") and hasattr(variants, "POS"):
        items = []
        for alt_record in variants.ALT:
            if isinstance(alt_record, str):
                alt = alt_record
            else:
                alt = getattr(alt_record, "value", None) or alt_record.serialize()
            items.append(Allele(int(variants.POS), str(variants.REF), str(alt)))
        return items
    if isinstance(variants, Allele):
        return [variants]
    if isinstance(variants, (tuple, list)) and len(variants) == 3 and isinstance(variants[0], int):
        pos, ref, alt = variants
        return [Allele(int(pos), str(ref), str(alt))]
    result: list[Allele] = []
    for item in variants:
        result.extend(_coerce_variants(item))
    return result


def normalize_variants(
    window: ReferenceWindow | str | None = None,
    variants: Any | None = None,
    *,
    sequence: str | None = None,
    start: int = 0,
    reference: str | ReferenceWindow = ".",
    contig: str = ".",
    complete_left: bool | None = None,
) -> NormalizationBatch:
    """Normalize an iterable of variants and keep all original mappings.

    Each item may be an :class:`Allele`, a ``(pos, ref, alt)`` tuple, a mapping
    with ``pos``/``ref``/``alt`` keys, or a vcfpy ``Record`` (all ALT alleles of
    the record are expanded).
    """
    if variants is None:
        raise ValueError("variants must be provided")
    ctx, _, _, _ = _resolve_inputs(
        window,
        None,
        None,
        None,
        sequence=sequence,
        start=start,
        reference=reference,
        contig=contig,
        complete_left=complete_left,
    )
    alleles = _coerce_variants(variants)
    results = [
        normalize_variant(ctx, allele.pos, allele.ref, allele.alt, complete_left=complete_left)
        for allele in alleles
    ]
    return NormalizationBatch(results, reference=ctx.reference, contig=ctx.contig)


def normalize_and_map(
    window: ReferenceWindow | str | None = None,
    pos: int | None = None,
    ref: str | None = None,
    alts: Iterable[str] | None = None,
    **kwargs: Any,
) -> dict[str, VariantKey | UnresolvedRange]:
    """Convenience wrapper returning only the original-to-key mapping."""
    return normalize_alleles(window, pos, ref, alts, **kwargs).allele_map


def normalize_record(record: Any, window: ReferenceWindow | str | None = None, **kwargs: Any) -> NormalizationBatch:
    """Normalize all ALT alleles of a vcfpy ``Record``.

    The record itself is not modified.  The returned batch retains the mapping
    from each original record ALT to its business key (or unresolved range).
    """
    if window is None:
        raise ValueError("window must be provided")
    return normalize_variants(window, record, **kwargs)
