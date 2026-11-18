# -*- coding: utf-8 -*-
"""Per-sample determinable interval analysis for gVCF files

In a gVCF, the absence of ordinary variant records in a region does **not**
imply that the sample was called homozygous reference there.  This module
combines reference confidence blocks (records carrying only ``<NON_REF>``
alleles and an ``END`` INFO value), concrete variant records, and the
per-sample genotype calls to build, for each sample, a set of
non-overlapping intervals classified as one of:

- ``REFERENCE`` -- the sample was called homozygous reference
- ``VARIANT`` -- the sample carries at least one concrete alternative allele
- ``NO_CALL`` -- evidence is insufficient (missing or partially missing GT,
  or a genotype that only references the symbolic ``<NON_REF>`` allele)
- ``UNCOVERED`` -- no record covers the position (only emitted when an
  explicit ``region`` is given)

``<NON_REF>`` (and the spanning deletion ``*``) are placeholders, not
concrete biological alleles, and are never reported as variants.

When a reference block and a variant record intersect, the block only
supports the part of its range outside the variant record's span; the
variant record takes precedence.  Polyploid genotypes and partially missing
genotypes keep their full allele contribution in the resulting intervals.
Identical reference blocks reported twice at shard boundaries are counted
only once, and gaps between records are never silently filled in as
reference by neighbouring blocks.
"""

from typing import Any, Iterable, NamedTuple

from vcfpy.record import Call, Record, Substitution, SymbolicAllele

#: State for intervals called homozygous reference
STATE_REFERENCE = "reference"
#: State for intervals carrying at least one concrete alternative allele
STATE_VARIANT = "variant"
#: State for intervals with insufficient evidence (missing/partial GT)
STATE_NO_CALL = "no_call"
#: State for intervals not covered by any record
STATE_UNCOVERED = "uncovered"

#: All interval states
STATES = (STATE_REFERENCE, STATE_VARIANT, STATE_NO_CALL, STATE_UNCOVERED)

#: Symbolic allele values that are placeholders rather than concrete alleles
NON_REF_VALUES = frozenset(["NON_REF", "*"])

#: Severity order used to resolve overlaps of equally prioritised records
_STATE_SEVERITY = {
    STATE_VARIANT: 3,
    STATE_NO_CALL: 2,
    STATE_REFERENCE: 1,
    STATE_UNCOVERED: 0,
}


class AlleleContribution(NamedTuple):
    """Per-sample allele contribution within an interval

    Polyploid genotypes contribute all of their alleles and partially
    missing genotypes keep their missing alleles instead of being
    collapsed into a diploid, fully-called view.
    """

    #: number of alleles in the genotype (``None`` if no GT present)
    ploidy: int | None
    #: number of alleles called as reference (allele index 0)
    ref_alleles: int
    #: number of alleles called as a concrete (non-``<NON_REF>``) alternative
    alt_alleles: int
    #: number of missing (``.``) alleles
    missing_alleles: int


class CallClassification(NamedTuple):
    """Classification of one sample's call at one record"""

    #: one of ``STATE_REFERENCE``, ``STATE_VARIANT``, ``STATE_NO_CALL``
    state: str
    #: the allele contribution behind the classification
    contribution: AlleleContribution


class SampleInterval(NamedTuple):
    """One classified, per-sample interval (0-based, half-open)"""

    #: chromosome name
    chrom: str
    #: 0-based begin position
    start: int
    #: 0-based end position (exclusive)
    end: int
    #: one of the ``STATE_*`` constants
    state: str
    #: allele contribution of the underlying call (``ploidy`` is ``None``
    #: and all counts are zero for ``STATE_UNCOVERED`` intervals)
    contribution: AlleleContribution


#: Empty contribution used for uncovered intervals
_NO_CONTRIBUTION = AlleleContribution(None, 0, 0, 0)


def is_non_ref_allele(alt: Any) -> bool:
    """Return ``True`` if ``alt`` is a placeholder allele

    Placeholder alleles (``<NON_REF>`` and the spanning deletion ``*``)
    do not describe a concrete biological allele.
    """
    if isinstance(alt, SymbolicAllele):
        return alt.value in NON_REF_VALUES
    if isinstance(alt, Substitution):
        return alt.value == "*"
    return False


def is_reference_block(record: Record) -> bool:
    """Return ``True`` if ``record`` is a pure reference confidence block

    A reference confidence block only carries placeholder alleles
    (``<NON_REF>``) in its ALT column.
    """
    return bool(record.ALT) and all(is_non_ref_allele(alt) for alt in record.ALT)


def record_span(record: Record) -> tuple[int, int]:
    """Return the 0-based, half-open genomic span covered by ``record``

    The span covers at least the REF bases and is extended to the ``END``
    INFO value (1-based, inclusive) when present, as used by reference
    confidence blocks and symbolic alleles.
    """
    start = record.POS - 1
    end = start + max(1, len(record.REF))
    info_end = record.INFO.get("END")
    if info_end is not None:
        end = max(end, int(info_end))
    return (start, end)


def classify_call(record: Record, call: Call) -> CallClassification:
    """Classify one sample's ``call`` at ``record``

    ``<NON_REF>`` alleles are never treated as concrete biological alleles:
    a genotype containing them (but no concrete alternative allele) yields
    ``STATE_NO_CALL`` rather than ``STATE_VARIANT``.  Missing alleles yield
    ``STATE_NO_CALL`` unless a concrete alternative allele is also present,
    in which case the call is ``STATE_VARIANT`` and the missing alleles are
    kept in the contribution.
    """
    gt_alleles = getattr(call, "gt_alleles", None)
    if not gt_alleles:
        contribution = AlleleContribution(getattr(call, "ploidy", None), 0, 0, 0)
        return CallClassification(STATE_NO_CALL, contribution)
    ploidy = len(gt_alleles)
    ref_alleles = 0
    alt_alleles = 0
    missing_alleles = 0
    non_ref_alleles = 0
    for allele in gt_alleles:
        if allele is None:
            missing_alleles += 1
        elif allele == 0:
            ref_alleles += 1
        elif 1 <= allele <= len(record.ALT):
            if is_non_ref_allele(record.ALT[allele - 1]):
                non_ref_alleles += 1
            else:
                alt_alleles += 1
        else:  # allele index out of range, cannot be interpreted
            missing_alleles += 1
    contribution = AlleleContribution(ploidy, ref_alleles, alt_alleles, missing_alleles)
    if alt_alleles > 0:
        state = STATE_VARIANT
    elif missing_alleles > 0 or non_ref_alleles > 0:
        state = STATE_NO_CALL
    else:
        state = STATE_REFERENCE
    return CallClassification(state, contribution)


class _Entry(NamedTuple):
    """Internal: one record's classified span for one sample"""

    start: int
    end: int
    priority: int
    classification: CallClassification


def _sweep_chrom(
    chrom: str,
    entries: list[_Entry],
    region: tuple[int, int] | None,
) -> list[SampleInterval]:
    """Resolve overlaps on one chromosome and emit elementary intervals"""
    points: set[int] = set()
    for entry in entries:
        points.add(entry.start)
        points.add(entry.end)
    if region is not None:
        points.add(region[0])
        points.add(region[1])
        points = {p for p in points if region[0] <= p <= region[1]}
    ordered = sorted(points)
    result: list[SampleInterval] = []
    prev_interval: SampleInterval | None = None
    for seg_start, seg_end in zip(ordered, ordered[1:]):
        if seg_start >= seg_end:
            continue
        best: _Entry | None = None
        for entry in entries:
            if entry.start <= seg_start and entry.end >= seg_end:
                if best is None or (entry.priority, _STATE_SEVERITY[entry.classification.state]) > (
                    best.priority,
                    _STATE_SEVERITY[best.classification.state],
                ):
                    best = entry
        if best is None:
            if region is None:
                # Gap between records: leave implicit and do not let the
                # neighbouring blocks merge across it into one call.
                prev_interval = None
                continue  # gap between records, leave implicit
            state, contribution = STATE_UNCOVERED, _NO_CONTRIBUTION
        else:
            state, contribution = best.classification
        interval = SampleInterval(chrom, seg_start, seg_end, state, contribution)
        if (
            prev_interval is not None
            and prev_interval.state == state
            and prev_interval.contribution == contribution
        ):
            prev_interval = SampleInterval(chrom, prev_interval.start, seg_end, state, contribution)
            result[-1] = prev_interval
        else:
            prev_interval = interval
            result.append(interval)
    return result


def sample_intervals(
    records: Iterable[Record],
    sample: str,
    region: tuple[str, int, int] | None = None,
) -> list[SampleInterval]:
    """Build the classified intervals for ``sample`` from ``records``

    :param records: iterable of :py:class:`vcfpy.Record` (e.g. a
        :py:class:`vcfpy.Reader` over a gVCF file)
    :param sample: name of the sample to classify
    :param region: optional ``(chrom, start, end)`` triple with 0-based,
        half-open coordinates; when given, output is clipped to the region
        and positions not covered by any record are reported as
        ``STATE_UNCOVERED`` instead of being left out
    :returns: list of non-overlapping :py:class:`SampleInterval`, sorted by
        chromosome (order of first appearance) and start position
    """
    by_chrom: dict[str, list[_Entry]] = {}
    for record in records:
        call = record.call_for_sample.get(sample)
        if call is None:
            continue
        start, end = record_span(record)
        if region is not None and record.CHROM != region[0]:
            continue
        priority = 0 if is_reference_block(record) else 1
        entry = _Entry(start, end, priority, classify_call(record, call))
        by_chrom.setdefault(record.CHROM, []).append(entry)
    if region is not None:
        by_chrom.setdefault(region[0], [])
    result: list[SampleInterval] = []
    for chrom, entries in by_chrom.items():
        span = (region[1], region[2]) if region is not None and chrom == region[0] else None
        result.extend(_sweep_chrom(chrom, entries, span))
    return result


def all_sample_intervals(
    records: Iterable[Record],
    samples: Iterable[str] | None = None,
    region: tuple[str, int, int] | None = None,
) -> dict[str, list[SampleInterval]]:
    """Build classified intervals for every sample in ``records``

    :param records: iterable of :py:class:`vcfpy.Record`
    :param samples: sample names to classify; when ``None``, all samples
        seen in the records are used (in order of first appearance)
    :param region: optional ``(chrom, start, end)`` region, see
        :py:func:`sample_intervals`
    :returns: mapping from sample name to its list of intervals
    """
    materialized = list(records)
    if samples is None:
        seen: dict[str, None] = {}
        for record in materialized:
            for call in record.calls:
                seen.setdefault(call.sample)
        samples = list(seen)
    return {sample: sample_intervals(materialized, sample, region) for sample in samples}
