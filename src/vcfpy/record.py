# -*- coding: utf-8 -*-
"""Code for representing a VCF record

The VCF record structure is modeled after the one of PyVCF
"""

import dataclasses
import enum
import re
import warnings
from typing import Any, Iterator, Literal, Sequence

from vcfpy.exceptions import CannotModifyUnparsedCallWarning

#: Code for single nucleotide variant allele
SNV = "SNV"
#: Code for a multi nucleotide variant allele
MNV = "MNV"
#: Code for "clean" deletion allele
DEL = "DEL"
#: Code for "clean" insertion allele
INS = "INS"
#: Code for indel allele, includes substitutions of unequal length
INDEL = "INDEL"
#: Code for structural variant allele
SV = "SV"
#: Code for break-end allele
BND = "BND"
#: Code for symbolic allele that is neither SV nor BND
SYMBOLIC = "SYMBOLIC"

#: Code for mixed variant type
MIXED = "MIXED"

#: Code for homozygous reference
HOM_REF = 0
#: Code for heterozygous
HET = 1
#: Code for homozygous alternative
HOM_ALT = 2

#: Characters reserved in VCF, have to be escaped in INFO fields
RESERVED_CHARS: dict[Literal["INFO", "FORMAT"], str] = {"INFO": ";=%,\r\n\t", "FORMAT": ":=%,\r\n\t"}
#: Mapping for escaping reserved characters
ESCAPE_MAPPING = [
    ("%", "%25"),
    (":", "%3A"),
    (";", "%3B"),
    ("=", "%3D"),
    (",", "%2C"),
    ("\r", "%0D"),
    ("\n", "%0A"),
    ("\t", "%09"),
]
#: Mapping from escaped characters to reserved one
UNESCAPE_MAPPING = [(v, k) for k, v in ESCAPE_MAPPING]


class IntervalClass(enum.Enum):
    """How a query interval relates to a record.

    A query that lies completely within the record's definite interval is a
    definite hit.  A query that only reaches into the confidence region
    (``CIPOS``/``CIEND``) is a possible hit.  Everything else is no hit.
    """

    #: query overlaps the definite (point-estimate) interval of the record
    DEFINITE = "definite"
    #: query only overlaps the confidence interval, not the definite interval
    POSSIBLE = "possible"
    #: query does not overlap the record's confidence interval at all
    NO_HIT = "no_hit"


@dataclasses.dataclass(frozen=True)
class RecordInterval:
    """Half-open, 0-based interval describing where a record may be located.

    All coordinates are 0-based and the interval end is exclusive, matching
    the convention of :py:attr:`Record.affected_start` and
    :py:attr:`Record.affected_end`.  ``definite`` gives the location derived
    from the point estimates (``POS``, ``END``/REF length, or the insertion
    point), while ``possible`` expands it by the ``CIPOS`` and ``CIEND``
    confidence intervals.  A bound of ``None`` marks an unknown extent, as is
    the right side of a symbolic structural variant whose length is not given
    by ``END``; breakends instead keep the finite anchor base interval.
    """

    #: definite interval ``[start, end)`` from point estimates; ``end`` may be
    #: ``None`` when the extent of the record is unknown
    start: int
    end: int | None
    #: confidence-expanded interval ``[start, end)``; either bound may be
    #: ``None`` when that side of the confidence interval is unknown
    ci_start: int | None = None
    ci_end: int | None = None

    @property
    def definite(self) -> tuple[int, int | None]:
        """Return the definite interval as a ``(start, end)`` tuple."""
        return (self.start, self.end)

    @property
    def possible(self) -> tuple[int | None, int | None]:
        """Return the confidence-expanded interval as a ``(start, end)`` tuple."""
        return (self.ci_start, self.ci_end)

    @property
    def is_imprecise(self) -> bool:
        """Return whether a confidence interval actually widens the definite interval."""
        widened_left = self.ci_start is not None and self.ci_start < self.start
        widened_right = self.ci_end is not None and (self.end is None or self.ci_end > self.end)
        return widened_left or widened_right

    @staticmethod
    def _overlaps(start: int | None, end: int | None, query_start: int, query_end: int) -> bool:
        """Check overlap of ``[start, end)`` with a 0-based half-open query.

        ``None`` marks an unbounded side.  A zero-length interval (an insertion
        point) is treated as the point at ``start`` and matches any query that
        contains that position, instead of never overlapping as strict
        half-open arithmetic would imply.
        """
        if start is not None and end is not None and start == end:
            return query_start <= start < query_end
        if start is not None and query_end <= start:
            return False
        if end is not None and end <= query_start:
            return False
        return True

    def classify(self, query_start: int, query_end: int) -> IntervalClass:
        """Classify a 0-based, half-open query against this interval.

        A point query at 0-based position ``p`` must be given as ``[p, p + 1)``.

        :param query_start: 0-based inclusive query start
        :param query_end: 0-based exclusive query end
        :return: :py:class:`IntervalClass` member
        """
        if query_end < query_start:
            raise ValueError("query_end must not be smaller than query_start")
        if self._overlaps(self.start, self.end, query_start, query_end):
            return IntervalClass.DEFINITE
        if self._overlaps(self.ci_start, self.ci_end, query_start, query_end):
            return IntervalClass.POSSIBLE
        return IntervalClass.NO_HIT

    def overlaps(self, query_start: int, query_end: int) -> bool:
        """Return whether the query overlaps the confidence interval at all."""
        return self.classify(query_start, query_end) is not IntervalClass.NO_HIT


class Record:
    """Represent one record from the VCF file

    Record objects are iterators of their calls
    """

    def __init__(
        self,
        CHROM: str,
        POS: int,
        ID: list[str],
        REF: str,
        ALT: list["AltRecord"],
        QUAL: float | None,
        FILTER: list[str],
        INFO: dict[str, Any],
        FORMAT: list[str] | None = None,
        calls: Sequence["Call | UnparsedCall"] | None = None,
    ):
        if bool(FORMAT) != bool(calls):
            raise ValueError("Either provide both FORMAT and calls or none.")
        #: A ``str`` with the chromosome name
        self.CHROM = CHROM
        #: An ``int`` with a 1-based begin position
        self.POS = POS
        #: An ``int`` with a 0-based begin position
        self.begin = POS - 1
        #: A list of the semicolon-separated values of the ID column
        self.ID = list(ID)
        #: A ``str`` with the REF value
        self.REF = REF
        #: A list of alternative allele records of type :py:class:`AltRecord`
        self.ALT: list["AltRecord"] = list(ALT)
        #: The quality value, can be ``None``
        self.QUAL = QUAL
        #: A list of strings for the FILTER column
        self.FILTER = FILTER
        #: An OrderedDict giving the values of the INFO column, flags are
        #: mapped to ``True``
        self.INFO = INFO
        #: A list of strings for the FORMAT column.  Optional, must be given if
        #: and only if ``calls`` is also given.
        self.FORMAT = FORMAT or []
        #: A list of genotype :py:class:`Call` objects.  Optional, must be
        #: given if and only if ``FORMAT`` is also given.
        self.calls = calls or []
        #: A mapping from sample name to entry in self.calls.
        self.call_for_sample = {}
        self.update_calls(self.calls)

    @property
    def interval(self) -> "RecordInterval":
        """The definite and confidence-expanded intervals of this record.

        Recomputed from the current ``POS``, ``ALT`` and ``INFO`` values on
        every access, so it stays consistent after a record is modified.
        """
        return self._build_interval()

    def update_calls(self, calls: Sequence["Call | UnparsedCall"]):
        """Update ``self.calls`` and other fields as necessary."""
        for call in calls:
            call.site = self
        self.call_for_sample = {call.sample: call for call in calls}

    def is_snv(self):
        """Return ``True`` if it is a SNV"""
        return len(self.REF) == 1 and all(a.type == "SNV" for a in self.ALT)

    @property
    def affected_start(self):
        """Return affected start position in 0-based coordinates

        For SNVs, MNVs, and deletions, the behaviour is the start position.
        In the case of insertions, the position behind the insert position is
        returned, yielding a 0-length interval together with
        :py:meth:`~Record.affected_end`
        """
        types = {alt.type for alt in self.ALT}  # set!
        BAD_MIX = {INS, SV, BND, SYMBOLIC}  # don't mix well with others
        if (BAD_MIX & types) and len(types) == 1 and list(types)[0] == INS:
            # Only insertions, return 0-based position right of first base
            return self.POS  # right of first base
        else:  # Return 0-based start position of first REF base
            return self.POS - 1  # left of first base

    @property
    def affected_end(self):
        """Return affected start position in 0-based coordinates

        For SNVs, MNVs, and deletions, the behaviour is based on the start
        position and the length of the REF.  In the case of insertions, the
        position behind the insert position is returned, yielding a 0-length
        interval together with :py:meth:`~Record.affected_start`
        """
        types = {alt.type for alt in self.ALT}  # set!
        BAD_MIX = {INS, SV, BND, SYMBOLIC}  # don't mix well with others
        if (BAD_MIX & types) and len(types) == 1 and list(types)[0] == INS:
            # Only insertions, return 0-based position right of first base
            return self.POS  # right of first base
        else:  # Return 0-based end position, behind last REF base
            return (self.POS - 1) + len(self.REF)

    @staticmethod
    def _is_int_pair(value: Any) -> bool:
        """Return whether ``value`` is a two-element integer pair (CIPOS/CIEND)."""
        return (
            isinstance(value, (list, tuple))
            and len(value) == 2
            and all(isinstance(v, int) and not isinstance(v, bool) for v in value)
        )

    def _ci_offsets(self, key: str) -> tuple[int | None, int | None]:
        """Return the ``(left, right)`` offsets of a CIPOS/CIEND INFO field."""
        value = self.INFO.get(key)
        if self._is_int_pair(value):
            return int(value[0]), int(value[1])
        return None, None

    def _build_interval(self) -> RecordInterval:
        """Build the definite and confidence intervals for this record.

        Coordinates are 0-based and half-open.  The definite interval uses
        the point estimates only; the confidence interval applies the
        ``CIPOS`` and ``CIEND`` offsets.  Insertions are represented by a
        zero-length interval at the insertion point, deletions and other
        spanning variants use ``END`` (or the REF span), and breakends as
        well as structural variants without a usable extent leave the end
        side unbounded instead of silently using the REF length.
        """
        types = {alt.type for alt in self.ALT}
        only_insertion_substitutions = bool(types) and types == {INS}
        symbolic_insertions = [
            alt
            for alt in self.ALT
            if isinstance(alt, SymbolicAllele) and (alt.value == "INS" or alt.value.startswith("INS:"))
        ]
        is_breakend = any(isinstance(alt, BreakEnd) for alt in self.ALT)
        is_insertion = only_insertion_substitutions or (
            bool(symbolic_insertions) and len(symbolic_insertions) == len(self.ALT)
        )

        pos_left, pos_right = self._ci_offsets("CIPOS")
        _end_left, end_right = self._ci_offsets("CIEND")
        info_end = self.INFO.get("END")
        if not isinstance(info_end, int) or isinstance(info_end, bool):
            info_end = None

        if is_breakend:
            # Only the local anchor breakpoint is pinned down; the mate lives
            # elsewhere (possibly on another contig), so there is no span.
            start = self.POS - 1
            end = start + 1
            ci_start = start + pos_left if pos_left is not None else start
            ci_end = start + 1 + pos_right if pos_right is not None else end
            return RecordInterval(start, end, ci_start, ci_end)

        if is_insertion:
            # Insertion point: between POS-1 and POS, i.e. the 0-based POS,
            # represented as a zero-length interval even when END == POS.
            start = self.POS
            end = self.POS
            ci_start = start + pos_left if pos_left is not None else start
            ci_end = end + pos_right if pos_right is not None else end
            return RecordInterval(start, end, ci_start, ci_end)

        # Definite start is always the 0-based position of the first REF base.
        start = self.POS - 1
        ci_start = start + pos_left if pos_left is not None else start

        if types & {SV, SYMBOLIC}:
            # Symbolic alleles other than insertions: use END when the record
            # spans a region (deletions, duplications, inversions, ...).  Never
            # fall back to the REF length, which is just the anchor base.
            if info_end is not None:
                end = info_end  # END is 1-based inclusive; == 0-based exclusive
                ci_end = end + end_right if end_right is not None else end
            else:
                # Unknown length: the anchor base is definite, but the end side
                # of the confidence interval stays open rather than pretending
                # the single REF base limits the extent.
                end = start + 1
                ci_end = None
            return RecordInterval(start, end, ci_start, ci_end)

        # Ordinary substitutions (SNV, MNV, explicit indel): the REF allele is
        # the authoritative span; a contradicting INFO/END (as emitted by some
        # callers for explicit deletions) must not invert or shrink it.
        end = start + len(self.REF)
        ci_end = end + end_right if end_right is not None else end
        return RecordInterval(start, end, ci_start, ci_end)

    def classify_interval(self, query_start: int, query_end: int) -> IntervalClass:
        """Classify a 0-based, half-open query against this record's intervals.

        A query overlapping the definite interval is reported as
        :py:attr:`IntervalClass.DEFINITE`; one that only reaches a CIPOS/CIEND
        confidence edge is :py:attr:`IntervalClass.POSSIBLE`; otherwise the
        result is :py:attr:`IntervalClass.NO_HIT`.

        :param query_start: 0-based inclusive start of the query
        :param query_end: 0-based exclusive end of the query
        """
        if query_end < query_start:
            raise ValueError("query_end must not be smaller than query_start")
        return self.interval.classify(query_start, query_end)

    def add_filter(self, label: str):
        """Add label to FILTER if not set yet, removing ``PASS`` entry if
        present
        """
        if label not in self.FILTER:
            if "PASS" in self.FILTER:
                self.FILTER = [f for f in self.FILTER if f != "PASS"]
            self.FILTER.append(label)

    def add_format(self, key: str, value: Any | None = None):
        """Add an entry to format

        The record's calls ``data[key]`` will be set to ``value`` if not yet
        set and value is not ``None``.  If key is already in FORMAT then
        nothing is done.
        """
        if key in self.FORMAT:
            return
        self.FORMAT.append(key)
        if value is not None:
            for call in self:
                if isinstance(call, UnparsedCall):
                    warnings.warn("UnparsedCall encountered, skipping", CannotModifyUnparsedCallWarning)
                else:
                    call.data.setdefault(key, value)

    def __iter__(self) -> Iterator["Call | UnparsedCall"]:
        """Return generator yielding from ``self.calls``"""
        yield from self.calls

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        raise NotImplementedError  # pragma: no cover

    def __ne__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        raise NotImplementedError  # pragma: no cover

    def __hash__(self):
        return hash(tuple(sorted(self.__dict__.items())))

    def __str__(self):
        tpl = "Record({})"
        lst: list[Any] = [
            self.CHROM,
            self.POS,
            self.ID,
            self.REF,
            self.ALT,
            self.QUAL,
            self.FILTER,
            self.INFO,
            self.FORMAT,
            self.calls,
        ]
        return tpl.format(", ".join(map(repr, lst)))

    def __repr__(self):
        return str(self)


class UnparsedCall:
    """Placeholder for :py:class:`Call` when parsing only a subset of fields"""

    def __init__(self, sample: str, unparsed_data: Any, site: Record | None = None):
        #: the name of the sample for which the call was made
        self.sample = sample
        #: ``str`` with the unparsed data
        self.unparsed_data = unparsed_data
        #: the :py:class:`Record` of this :py:class:`Call`
        self.site = site


#: Regular expression for splitting alleles
ALLELE_DELIM = re.compile(r"[|/]")


class Call:
    """The information for a genotype callable

    By VCF, this should always include the genotype information and
    can contain an arbitrary number of further annotation, e.g., the
    coverage at the variant position.
    """

    def __init__(self, sample: str, data: dict[str, Any], site: Record | None = None):
        #: the name of the sample for which the call was made
        self.sample = sample
        #: an OrderedDict with the key/value pair information from the
        #: call's data
        self.data: dict[str, Any] = data
        #: the :py:class:`Record` of this :py:class:`Call`
        self.site = site
        #: the allele numbers (0, 1, ...) in this calls or None for no-call
        self.gt_alleles: list[int | None] | None = None
        #: whether or not the variant is fully called
        self.called = None
        #: the number of alleles in this sample's call
        self.ploidy = None
        self._genotype_updated()

    def set_genotype(self, genotype: str | None):
        """Set ``self.data["GT"]`` to ``genotype`` and properly update related
        properties.
        """
        self.data["GT"] = genotype
        self._genotype_updated()

    def _genotype_updated(self):
        """Update fields related to ``self.data["GT"]``."""
        if self.data.get("GT", None) is None:
            self.gt_alleles = None
            self.called = None
            self.ploidy = None
        else:
            self.gt_alleles = []
            for allele in ALLELE_DELIM.split(str(self.data["GT"])):
                if allele == ".":
                    self.gt_alleles.append(None)
                else:
                    self.gt_alleles.append(int(allele))
            self.called = all(al is not None for al in self.gt_alleles)
            self.ploidy = len(self.gt_alleles)

    @property
    def is_phased(self):
        """Return boolean indicating whether this call is phased"""
        return "|" in self.data.get("GT", "")

    @property
    def gt_phase_char(self):
        """Return character to use for phasing"""
        return "/" if not self.is_phased else "|"

    @property
    def gt_bases(self) -> tuple[str | None, ...]:
        """Return the actual genotype bases, e.g. if VCF genotype is 0/1,
        could return ('A', 'T')
        """
        result: list[str | None] = []
        for a in self.gt_alleles or []:
            if a is None:
                result.append(None)
            elif a == 0:
                if self.site is None:
                    raise ValueError("Cannot determine bases without site being set")
                result.append(self.site.REF)
            else:
                if self.site is None:  # pragma: no cover
                    raise ValueError("Cannot determine bases without site being set")
                result.append(getattr(self.site.ALT[a - 1], "value", None))
        return tuple(result)

    @property
    def gt_type(self) -> Literal[0, 1, 2] | None:
        """The type of genotype, returns one of ``HOM_REF``, ``HOM_ALT``, and
        ``HET``.
        """
        if not self.called or not self.gt_alleles:
            return None  # not called
        elif all(a == 0 for a in self.gt_alleles):
            return HOM_REF
        elif len(set(self.gt_alleles)) == 1:
            return HOM_ALT
        else:
            return HET

    def is_filtered(self, require: list[str] | None = None, ignore: list[str] | None = None):
        """Return ``True`` for filtered calls

        :param iterable ignore: if set, the filters to ignore, make sure to
            include 'PASS', when setting, default is ``['PASS']``
        :param iterable require: if set, the filters to require for returning
            ``True``
        """
        ignore = ignore or ["PASS"]
        if "FT" not in self.data or not self.data["FT"]:
            return False
        for ft in self.data["FT"]:
            if ft in ignore:
                continue  # skip
            if not require:
                return True
            elif ft in require:
                return True
        return False

    @property
    def is_het(self) -> bool:
        """Return ``True`` for heterozygous calls"""
        return self.gt_type == HET

    @property
    def is_variant(self) -> bool:
        """Return ``True`` for non-hom-ref calls"""
        return bool(self.gt_type)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        raise NotImplementedError  # pragma: no cover

    def __ne__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        raise NotImplementedError  # pragma: no cover

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.__dict__.items())))

    def __str__(self):
        tpl = "Call({})"
        lst: list[str | dict[str, Any]] = [self.sample, self.data]
        return tpl.format(", ".join(map(repr, lst)))

    def __repr__(self):
        return str(self)


class AltRecord:
    """An alternative allele Record

    Currently, can be a substitution, an SV placeholder, or breakend
    """

    def __init__(
        self, type_: Literal["SNV", "MNV", "DEL", "INS", "INDEL", "SV", "BND", "SYMBOLIC", "MIXED"] | None = None
    ):
        #: String describing the type of the variant, could be one of
        #: SNV, MNV, could be any of teh types described in the ALT
        #: header lines, such as DUP, DEL, INS, ...
        self.type = type_

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        raise NotImplementedError  # pragma: no cover

    def __ne__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        raise NotImplementedError  # pragma: no cover

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.__dict__.items())))

    def serialize(self) -> str:
        """Return ``str`` with representation for VCF file"""
        raise NotImplementedError("Abstract class, implemented in sub class")


class Substitution(AltRecord):
    """A basic alternative allele record describing a REF->AltRecord
    substitution

    Note that this subsumes MNVs, insertions, and deletions.
    """

    def __init__(
        self, type_: Literal["SNV", "MNV", "DEL", "INS", "INDEL", "SV", "BND", "SYMBOLIC", "MIXED"], value: str
    ):
        super().__init__(type_)
        #: The alternative base sequence to use in the substitution
        self.value = value

    def serialize(self) -> str:
        return self.value

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        raise NotImplementedError  # pragma: no cover

    def __ne__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        raise NotImplementedError  # pragma: no cover

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.__dict__.items())))

    def __str__(self) -> str:
        tpl = "Substitution(type_={}, value={})"
        return tpl.format(*map(repr, [self.type, self.value]))

    def __repr__(self):
        return str(self)


#: code for five prime orientation :py:class:`BreakEnd`
FIVE_PRIME = "5"
#: code for three prime orientation :py:class:`BreakEnd`
THREE_PRIME = "3"

#: code for forward orientation
FORWARD = "+"
#: code for reverse orientation
REVERSE = "-"


class BreakEnd(AltRecord):
    """A placeholder for a breakend"""

    def __init__(
        self,
        mate_chrom: str | None,
        mate_pos: int | None,
        orientation: str | None,
        mate_orientation: Literal["+", "-"] | None,
        sequence: str,
        within_main_assembly: bool | None,
    ):
        super().__init__("BND")
        #: chromosome of the mate breakend
        self.mate_chrom = mate_chrom
        #: position of the mate breakend
        self.mate_pos = mate_pos
        #: orientation of this breakend
        self.orientation = orientation
        #: orientation breakend's mate
        self.mate_orientation = mate_orientation
        #: breakpoint's connecting sequence
        self.sequence = sequence
        #: ``bool`` specifying if the breakend mate is within the assembly
        #: (``True``) or in an ancillary assembly (``False``)
        self.within_main_assembly = within_main_assembly

    def serialize(self):
        """Return string representation for VCF"""
        if self.mate_chrom is None:
            remote_tag = "."
        else:
            if self.within_main_assembly:
                mate_chrom = self.mate_chrom
            else:
                mate_chrom = "<{}>".format(self.mate_chrom)
            if self.mate_orientation is None:  # pragma: no cover
                raise ValueError("mate_orientation must be set if mate_chrom is set")
            tpl = {FORWARD: "[{}:{}[", REVERSE: "]{}:{}]"}[self.mate_orientation]
            remote_tag = tpl.format(mate_chrom, self.mate_pos)
        if self.orientation == FORWARD:
            return remote_tag + self.sequence
        else:
            return self.sequence + remote_tag

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        raise NotImplementedError  # pragma: no cover

    def __ne__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        raise NotImplementedError  # pragma: no cover

    def __hash__(self):
        return hash(tuple(sorted(self.__dict__.items())))

    def __str__(self):
        tpl = "BreakEnd({})"
        vals: list[Any] = [
            self.mate_chrom,
            self.mate_pos,
            self.orientation,
            self.mate_orientation,
            self.sequence,
            self.within_main_assembly,
        ]
        return tpl.format(", ".join(map(repr, vals)))

    def __repr__(self):
        return str(self)


class SingleBreakEnd(BreakEnd):
    """A placeholder for a single breakend"""

    def __init__(self, orientation: str, sequence: str):
        super().__init__(None, None, orientation, None, sequence, None)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        raise NotImplementedError  # pragma: no cover

    def __ne__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        raise NotImplementedError  # pragma: no cover

    def __hash__(self):
        return hash(tuple(sorted(self.__dict__.items())))

    def __str__(self):
        tpl = "SingleBreakEnd({})"
        vals: list[Any] = [self.orientation, self.sequence]
        return tpl.format(", ".join(map(repr, vals)))


class SymbolicAllele(AltRecord):
    """A placeholder for a symbolic allele

    The allele symbol must be defined in the header using an ``ALT`` header
    before being parsed.  Usually, this is used for succinct descriptions of
    structural variants or IUPAC parameters.
    """

    def __init__(self, value: str):
        super().__init__(SYMBOLIC)
        #: The symbolic value, e.g. 'DUP'
        self.value = value

    def serialize(self):
        return "<{}>".format(self.value)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        raise NotImplementedError  # pragma: no cover

    def __ne__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        raise NotImplementedError  # pragma: no cover

    def __hash__(self):
        return hash(tuple(sorted(self.__dict__.items())))

    def __str__(self):
        return "SymbolicAllele({})".format(repr(self.value))

    def __repr__(self):
        return str(self)
