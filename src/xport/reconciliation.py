"""
Reconcile case numbers that collide after display formatting.

Historical extracts often retain a case number only as the text a SAS
format produced, for example ``1.2346E+12``.  A display like that covers a
whole interval of possible numbers, so several distinct cases can share
one display string.  Other extracts retain the numeric value, sometimes
with the format that was used for display.

:func:`reconcile` works with both kinds of evidence and with a linkage
table of confirmed identities.  It deliberately does not merge records
just because their display text matches: records from different sources
are joined only when a confirmed link says they are the same case.

The result separates:

* ``unique`` -- correspondences where the two clusters are each other's
  only overlapping candidate and at least one side retains an exact
  value, so the connection can be recovered without guessing digits;
* ``possible`` -- correspondences whose precision intervals overlap but
  that remain ambiguous.

Every correspondence carries a :class:`~xport.formatting.PrecisionInterval`
describing the precision that the evidence actually preserved.
"""

# Standard Library
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

from .formatting import (
    FormatSpec,
    PrecisionInterval,
    coerce_format,
    format_value,
    precision_interval,
    value_interval,
)

__all__ = [
    'CaseRecord',
    'Cluster',
    'Conflict',
    'Correspondence',
    'DisplayGroup',
    'IdentityCluster',
    'Link',
    'Reconciliation',
    'reconcile',
    'reconcile_case_numbers',
    'resolve_case_numbers',
]


def _normalize_display(value) -> Optional[str]:
    """
    Normalize a display string without changing its numeric precision.

    Whitespace, thousands separators, a leading currency sign, and the
    exponent spelling are normalized.  Digits, including trailing zeros,
    are preserved because they describe how precise the display is.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.startswith('(') and text.endswith(')'):
        text = '-' + text[1:-1].strip()
    text = text.replace(',', '').replace('$', '').strip()
    text = text.replace('D', 'E').replace('d', 'e')
    if 'e' in text:
        mantissa, _, exponent = text.partition('e')
        exponent = exponent.strip()
        sign = ''
        if exponent.startswith(('+', '-')):
            sign, exponent = exponent[0], exponent[1:]
        exponent = exponent.lstrip('0') or '0'
        if sign == '+':
            sign = ''
        text = f'{mantissa}E{sign}{exponent}'
    return text


def _as_hashable(value):
    """Convert a record identifier to a hashable value."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return repr(value)


def _iter_mappings(data) -> Iterable[Mapping]:
    """Accept a DataFrame, a single mapping, or an iterable of mappings."""
    if data is None:
        return ()
    if isinstance(data, Mapping):
        return (data,)
    if hasattr(data, 'to_dict'):
        try:
            return data.to_dict('records')
        except TypeError:
            pass
    return data


@dataclass(frozen=True)
class CaseRecord:
    """
    One case-number observation from a source.

    ``display`` is the text that was retained, if any.  ``value`` is the
    original numeric value, if retained.  ``format`` is the SAS format
    used for display.  A record with only a display string has a wide
    interval; a record with a retained numeric value is exact.
    """

    source: str
    record_id: Any
    display: Optional[str] = None
    value: Optional[float] = None
    format: Optional[FormatSpec] = None
    batch: Optional[str] = None

    @property
    def key(self):
        """Hashable identity within the input."""
        return (self.source, self.record_id)

    @property
    def effective_display(self) -> Optional[str]:
        """Retained display, or the display implied by value and format."""
        if self.display is not None:
            return _normalize_display(self.display)
        if self.value is not None and self.format is not None:
            if self.format.name or self.format.width or self.format.decimals:
                if self.format.is_numeric:
                    return _normalize_display(format_value(self.value, self.format))
        return None

    @property
    def interval(self) -> Optional[PrecisionInterval]:
        """
        Numeric interval supported by this record's evidence.

        A retained value is exact.  A display-only record gets the
        interval implied by the digits that were displayed.  When both
        are present, the interval is their intersection: a contradictory
        row narrows to nothing and is reported as a conflict.
        """
        if self.value is not None:
            exact = value_interval(self.value)
            if self.display is not None:
                shown = precision_interval(self.display, self.format)
                if shown is not None:
                    return exact & shown
            return exact
        if self.display is not None:
            return precision_interval(self.display, self.format)
        return None

    @property
    def reproduces_display(self) -> bool:
        """
        Whether the retained value renders to its own retained display.

        ``None`` means the question does not apply (no value, no display,
        or no usable format).  ``False`` means the row's value and display
        disagree, which is reported rather than silently accepted.
        """
        if self.value is None or self.display is None:
            return None
        if self.format is None or not (self.format.name or self.format.width):
            return None
        return _normalize_display(format_value(self.value, self.format)) == self.effective_display

    @property
    def value_display_conflict(self) -> bool:
        """
        Whether a retained value falls outside its own display precision.

        This check does not require a format.  A value of ``1235`` beside
        the display ``1.2E+03`` is contradictory even if the batch's
        format is unknown.
        """
        if self.value is None or self.display is None:
            return False
        interval = precision_interval(self.display, self.format)
        if interval is None:
            return False
        return not interval.contains(self.value)

    @property
    def has_value(self):
        """Whether the original numeric value was retained."""
        return self.value is not None

    @property
    def has_display(self):
        """Whether a display string was retained."""
        return self.display is not None

    def with_value(self, value, format=None):
        """Copy this record with a newly supplied original value."""
        return CaseRecord(
            source=self.source,
            record_id=self.record_id,
            display=self.display,
            value=value,
            format=coerce_format(format) if format is not None else self.format,
            batch=self.batch,
        )

    def as_dict(self):
        """JSON-friendly representation."""
        interval = self.interval
        return {
            'source': self.source,
            'record_id': self.record_id,
            'batch': self.batch,
            'display': self.display,
            'effective_display': self.effective_display,
            'value': self.value,
            'format': str(self.format) if self.format is not None else None,
            'has_value': self.has_value,
            'interval': interval.as_dict() if interval is not None else None,
        }


@dataclass(frozen=True)
class Link:
    """
    An identity link from the linkage table.

    ``left`` and ``right`` are record keys ``(source, record_id)`` or,
    when unique in the input, bare record identifiers.  Only ``confirmed``
    links merge evidence; unconfirmed links are kept for auditing but do
    not create an identity.  ``evidence`` is copied into recovered
    correspondences so a caller can trace why a match was accepted.
    """

    left: Any
    right: Any
    confirmed: bool = True
    evidence: Optional[str] = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self):
        """JSON-friendly representation."""
        return {
            'left': self.left,
            'right': self.right,
            'confirmed': self.confirmed,
            'evidence': self.evidence,
            'attributes': dict(self.attributes),
        }


@dataclass(frozen=True)
class Cluster:
    """
    Records that a confirmed link says belong to the same case.
    """

    cluster_id: int
    records: Tuple[CaseRecord, ...]
    interval: Optional[PrecisionInterval]
    displays: Tuple[str, ...]
    warnings: Tuple[str, ...] = ()

    @property
    def sources(self):
        """Sources represented in this cluster."""
        return tuple(sorted({record.source for record in self.records}))

    @property
    def record_keys(self):
        """Keys of the records in this cluster."""
        return tuple(record.key for record in self.records)

    @property
    def provenance(self):
        """
        ``(source, batch)`` pairs represented in this cluster.

        Two records from the same source *and* batch are rows of one
        extract; their matching display is a collision, not a
        correspondence.  Different batches of the same source may be
        compared, because they are separate snapshots.
        """
        return frozenset((record.source, record.batch) for record in self.records)

    @property
    def display(self):
        """The single display string, or ``None`` when ambiguous."""
        return self.displays[0] if len(self.displays) == 1 else None

    @property
    def exact_values(self):
        """Distinct retained numeric values in this cluster."""
        return tuple(sorted(
            {record.value for record in self.records if record.value is not None},
            key=str,
        ))

    @property
    def exact_value(self):
        """The cluster's single exact value, or ``None``."""
        values = self.exact_values
        if len(values) == 1:
            return values[0]
        return None

    @property
    def has_exact_value(self):
        """Whether the cluster contains any retained numeric value."""
        return any(record.value is not None for record in self.records)

    @property
    def has_conflicting_values(self):
        """Whether the cluster retains more than one distinct value."""
        return len(self.exact_values) > 1

    @property
    def label(self):
        """Short human-readable label."""
        first = self.records[0]
        return f'{first.source}:{first.record_id}'

    def as_dict(self):
        """JSON-friendly representation."""
        return {
            'cluster_id': self.cluster_id,
            'label': self.label,
            'sources': list(self.sources),
            'displays': list(self.displays),
            'interval': self.interval.as_dict() if self.interval is not None else None,
            'records': [record.as_dict() for record in self.records],
            'warnings': list(self.warnings),
        }


# ``IdentityCluster`` is the longer name used in prose and documentation.
IdentityCluster = Cluster


@dataclass(frozen=True)
class DisplayGroup:
    """
    Clusters that share one normalized display string.
    """

    display: str
    clusters: Tuple[Cluster, ...]

    @property
    def sources(self):
        """Sources represented in this group."""
        return tuple(sorted({source for cluster in self.clusters for source in cluster.sources}))

    @property
    def is_ambiguous(self):
        """Whether more than one cluster shares this display."""
        return len(self.clusters) > 1

    def as_dict(self):
        """JSON-friendly representation."""
        return {
            'display': self.display,
            'ambiguous': self.is_ambiguous,
            'sources': list(self.sources),
            'clusters': [cluster.as_dict() for cluster in self.clusters],
        }


@dataclass(frozen=True)
class Correspondence:
    """
    A possible or uniquely recovered match between two identity clusters.

    ``left_key`` and ``right_key`` identify the exact record pair when a
    correspondence comes from a confirmed link.  For derived matches they
    are ``None`` and the clusters themselves are the endpoints.
    """

    left: Cluster
    right: Cluster
    interval: Optional[PrecisionInterval]
    unique: bool
    display: Optional[str] = None
    reason: str = ''
    left_key: Optional[Tuple[str, Any]] = None
    right_key: Optional[Tuple[str, Any]] = None

    @property
    def left_label(self):
        """Label for the left endpoint."""
        return _label(self.left_key, self.left)

    @property
    def right_label(self):
        """Label for the right endpoint."""
        return _label(self.right_key, self.right)

    @property
    def source_pair(self):
        """The two source sets involved, excluding shared sources."""
        left_only = tuple(sorted(set(self.left.sources) - set(self.right.sources)))
        right_only = tuple(sorted(set(self.right.sources) - set(self.left.sources)))
        return (left_only, right_only)

    def as_dict(self):
        """JSON-friendly representation."""
        return {
            'left': self.left_label,
            'right': self.right_label,
            'left_cluster': self.left.cluster_id,
            'right_cluster': self.right.cluster_id,
            'left_sources': list(self.left.sources),
            'right_sources': list(self.right.sources),
            'display': self.display,
            'interval': self.interval.as_dict() if self.interval is not None else None,
            'unique': self.unique,
            'reason': self.reason,
        }


def _label(key, cluster):
    """Record label when a key is known, otherwise the cluster label."""
    if key is None:
        return cluster.label
    return f'{key[0]}:{key[1]}'
    """Record label when a key is known, otherwise the cluster label."""
    if key is None:
        return cluster.label
    return f'{key[0]}:{key[1]}'


@dataclass(frozen=True)
class Conflict:
    """
    Evidence that cannot all be true at once.
    """

    cluster: Cluster
    reason: str

    def as_dict(self):
        """JSON-friendly representation."""
        return {
            'cluster_id': self.cluster.cluster_id,
            'label': self.cluster.label,
            'reason': self.reason,
            'records': [record.as_dict() for record in self.cluster.records],
        }


class Reconciliation:
    """
    Result of :func:`reconcile`.

    ``records``
        Normalized input records.
    ``clusters``
        Confirmed identity groups, each with a precision interval.
    ``groups``
        Display strings shared by more than one cluster.
    ``unique``
        Cross-source matches that can be recovered uniquely.
    ``possible``
        Cross-source matches that remain possible but ambiguous.
    ``conflicts``
        Contradictory evidence.
    ``warnings``
        Input problems that could not be resolved.
    """

    def __init__(
        self,
        records: Sequence[CaseRecord],
        links: Sequence[Link],
        clusters: Sequence[Cluster],
        groups: Sequence[DisplayGroup],
        correspondences: Sequence[Correspondence],
        conflicts: Sequence[Conflict],
        warnings: Sequence[str],
    ):
        self.records = tuple(records)
        self.links = tuple(links)
        self.clusters = tuple(clusters)
        self.groups = tuple(groups)
        self.correspondences = tuple(correspondences)
        self.conflicts = tuple(conflicts)
        self.warnings = tuple(warnings)

    @property
    def confirmed_links(self):
        """Input links that were marked confirmed and resolved."""
        known = {record.key for record in self.records}
        return tuple(
            link for link in self.links
            if link.confirmed and link.left in known and link.right in known
        )

    @property
    def unique(self):
        """Cross-source matches recovered uniquely."""
        return tuple(c for c in self.correspondences if c.unique)

    @property
    def possible(self):
        """Cross-source matches that remain possible but not unique."""
        return tuple(c for c in self.correspondences if not c.unique)

    def cluster_for(self, record_or_key) -> Optional[Cluster]:
        """
        Find the cluster containing a record, record key, or unique id.
        """
        if isinstance(record_or_key, CaseRecord):
            key = record_or_key.key
        elif isinstance(record_or_key, tuple) and len(record_or_key) == 2:
            key = record_or_key
        else:
            matches = [r for r in self.records if r.record_id == record_or_key]
            if len(matches) != 1:
                return None
            key = matches[0].key
        for cluster in self.clusters:
            if key in cluster.record_keys:
                return cluster
        return None

    def candidates_for(self, record_or_key) -> Tuple[Correspondence, ...]:
        """
        Correspondences involving the cluster that contains a record.
        """
        cluster = self.cluster_for(record_or_key)
        if cluster is None:
            return ()
        return tuple(
            c for c in self.correspondences
            if c.left.cluster_id == cluster.cluster_id or c.right.cluster_id == cluster.cluster_id
        )

    def with_original(self, source, record_id, value, format=None):
        """
        Recompute after supplying one record's original numeric value.

        The new value narrows only clusters whose precision intervals
        contain it.  Other clusters keep their original evidence, so an
        unrelated number cannot create new correspondences and no
        trailing digits are invented.
        """
        target = None
        for record in self.records:
            if record.source == source and record.record_id == record_id:
                target = record
                break
        if target is None:
            raise KeyError(f'No record {source!r}:{record_id!r}')

        if target.display is not None:
            shown = precision_interval(target.display, target.format)
            if shown is not None and not shown.contains(value):
                raise ValueError(
                    f'Value {value!r} lies outside the retained precision '
                    f'{shown.as_dict()!r} for {source!r}:{record_id!r}'
                )

        updated = []
        for record in self.records:
            if record is target:
                updated.append(record.with_value(value, format))
            else:
                updated.append(record)
        return reconcile(updated, self.links)

    def summary(self) -> str:
        """A short human-readable report."""
        lines = [
            f'records: {len(self.records)}',
            f'identity clusters: {len(self.clusters)}',
            f'display collisions: {len(self.groups)}',
            f'unique correspondences: {len(self.unique)}',
            f'possible correspondences: {len(self.possible)}',
            f'conflicts: {len(self.conflicts)}',
        ]
        for correspondence in self.unique:
            bounds = correspondence.interval.as_dict() if correspondence.interval else {}
            lines.append(
                f'unique: {correspondence.left_label} <-> {correspondence.right_label} '
                f'[{bounds.get("low")}, {bounds.get("high")})'
            )
        for group in self.groups:
            labels = ', '.join(cluster.label for cluster in group.clusters)
            lines.append(f'collision: {group.display!r} -> {labels}')
        for conflict in self.conflicts:
            lines.append(f'conflict: {conflict.cluster.label}: {conflict.reason}')
        return '\n'.join(lines)

    def as_dict(self):
        """JSON-friendly representation."""
        return {
            'records': [record.as_dict() for record in self.records],
            'clusters': [cluster.as_dict() for cluster in self.clusters],
            'groups': [group.as_dict() for group in self.groups],
            'unique': [correspondence.as_dict() for correspondence in self.unique],
            'possible': [correspondence.as_dict() for correspondence in self.possible],
            'confirmed_links': [link.as_dict() for link in self.confirmed_links],
            'conflicts': [conflict.as_dict() for conflict in self.conflicts],
            'warnings': list(self.warnings),
        }

    def __repr__(self):
        """REPL-format."""
        return (
            f'<{type(self).__name__} records={len(self.records)} '
            f'clusters={len(self.clusters)} unique={len(self.unique)} '
            f'possible={len(self.possible)} conflicts={len(self.conflicts)}>'
        )


def _first(get, *names):
    """First non-``None`` value among several candidate column names."""
    for name in names:
        value = get(name)
        if value is not None:
            return value
    return None


def _coerce_record(record, index: int, formats: Mapping[str, Any]) -> CaseRecord:
    """Convert one input record to a :class:`CaseRecord`."""
    if isinstance(record, CaseRecord):
        return record

    if isinstance(record, Mapping):
        get = record.get
        source = _first(
            get, 'source', 'source_name', 'extract', 'batch', 'dataset'
        ) or ''
        record_id = _first(get, 'record_id', 'record', 'row_id', 'id', 'case_id')
        if record_id is None:
            record_id = index
        display = _first(
            get, 'display', 'display_string', 'case_display', 'printed', 'text',
        )
        value = _first(get, 'value', 'number', 'case_number', 'original_value')
        batch = _first(get, 'batch', 'batch_id', 'period', 'file')
        fmt = _first(get, 'format', 'field_format', 'sas_format', 'fmt')
        if fmt is None and batch is not None:
            fmt = formats.get(batch)
        if fmt is None:
            fmt = formats.get(source)
        return CaseRecord(
            source=str(source),
            record_id=_as_hashable(record_id),
            display=_normalize_display(display),
            value=value,
            format=coerce_format(fmt),
            batch=batch,
        )

    if isinstance(record, (tuple, list)):
        if len(record) < 3:
            raise TypeError(f'Record {record!r} must have at least source, id, and display')
        source, record_id, display = record[:3]
        value = record[3] if len(record) > 3 else None
        fmt = record[4] if len(record) > 4 else None
        batch = record[5] if len(record) > 5 else None
        if fmt is None and batch is not None:
            fmt = formats.get(batch)
        if fmt is None:
            fmt = formats.get(str(source or ''))
        return CaseRecord(
            source=str(source or ''),
            record_id=_as_hashable(record_id),
            display=_normalize_display(display),
            value=value,
            format=coerce_format(fmt),
            batch=batch,
        )

    raise TypeError(f'Unsupported record type {type(record).__name__}')


def _resolve_reference(reference, records_by_id):
    """Resolve a bare record identifier when it is unambiguous."""
    if isinstance(reference, (tuple, list)) and len(reference) == 2:
        return (reference[0], _as_hashable(reference[1]))
    try:
        hash(reference)
    except TypeError:
        raise TypeError(f'Record reference {reference!r} is not a source/id pair')
    matches = records_by_id.get(reference, ())
    if len(matches) == 1:
        return matches[0]
    return reference


def _coerce_link(link, records_by_id) -> Link:
    """Convert one input link to a :class:`Link`."""
    if isinstance(link, Link):
        return link

    if isinstance(link, Mapping):
        get = link.get
        left = _first(get, 'left', 'left_record', 'a', 'source_record')
        right = _first(get, 'right', 'right_record', 'b', 'target_record')
        if left is None and right is None:
            left_source = _first(get, 'left_source', 'a_source', 'source_a')
            left_id = _first(get, 'left_record_id', 'left_id', 'a_id', 'record_a')
            right_source = _first(get, 'right_source', 'b_source', 'source_b')
            right_id = _first(get, 'right_record_id', 'right_id', 'b_id', 'record_b')
            if None not in (left_source, left_id, right_source, right_id):
                left = (left_source, left_id)
                right = (right_source, right_id)
        confirmed = _first(get, 'confirmed', 'verified', 'is_confirmed')
        if confirmed is None:
            confirmed = True
        evidence = _first(get, 'evidence', 'basis', 'note', 'source')
        attributes = _first(get, 'attributes', 'attrs') or {}
    elif isinstance(link, (tuple, list)) and len(link) >= 2:
        left, right = link[0], link[1]
        confirmed = link[2] if len(link) > 2 else True
        evidence = link[3] if len(link) > 3 else None
        attributes = {}
    else:
        raise TypeError(f'Unsupported link type {type(link).__name__}')

    return Link(
        left=_resolve_reference(left, records_by_id),
        right=_resolve_reference(right, records_by_id),
        confirmed=bool(confirmed),
        evidence=evidence,
        attributes=attributes,
    )


class _DisjointSet:
    """Small union-find implementation."""

    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, item):
        """Find the representative for ``item``."""
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            item, self.parent[item] = self.parent[item], root
        return root

    def union(self, left, right):
        """Merge two groups."""
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _build_clusters(records, links, warnings):
    """Group records by confirmed links and compute cluster intervals."""
    disjoint = _DisjointSet(record.key for record in records)
    for link in links:
        if link.confirmed and link.left in disjoint.parent and link.right in disjoint.parent:
            disjoint.union(link.left, link.right)

    members = defaultdict(list)
    for record in records:
        members[disjoint.find(record.key)].append(record)

    ordered = sorted(
        members.values(),
        key=lambda group: min((r.source, str(r.record_id)) for r in group),
    )
    clusters = []
    for cluster_id, group in enumerate(ordered):
        group = sorted(group, key=lambda r: (r.source, str(r.record_id)))
        cluster_warnings = []
        interval = None
        for record in group:
            record_interval = record.interval
            if record_interval is None:
                cluster_warnings.append(f'Record {record.key!r} has no usable numeric evidence')
                interval = None
                break
            interval = record_interval if interval is None else interval & record_interval
            if interval is None:
                cluster_warnings.append('Precision intervals do not overlap')
                break

        displays = tuple(sorted({
            display for display in (r.effective_display for r in group) if display is not None
        }))
        if len(displays) > 1:
            cluster_warnings.append(
                f'Confirmed identity has different display strings: {list(displays)!r}'
            )

        provenance = [(record.source, record.batch) for record in group]
        if len(provenance) != len(set(provenance)):
            cluster_warnings.append(
                'Confirmed identity contains two rows from the same source and batch'
            )

        clusters.append(
            Cluster(
                cluster_id=cluster_id,
                records=tuple(group),
                interval=interval,
                displays=displays,
                warnings=tuple(cluster_warnings),
            )
        )
        warnings.extend(w for w in cluster_warnings if w not in warnings)
    return clusters


def _exact_matches_display(value, display, format) -> bool:
    """
    Whether ``value`` is consistent with ``display`` under ``format``.

    When the format reproduces the display exactly, that is the strongest
    evidence.  A batch format is metadata and may not be the one actually
    used for a given row, so a mismatch is not automatically fatal: the
    value may still fall inside the display's precision interval.  A
    value outside that interval is inconsistent regardless of format.
    """
    interval = precision_interval(display, format)
    if interval is not None and not interval.contains(value):
        return False

    spec = coerce_format(format)
    if spec.name or spec.width or spec.decimals:
        if spec.is_numeric:
            rendered = _normalize_display(format_value(value, spec))
            if rendered == _normalize_display(display):
                return True
    return interval is not None


def _display_records(cluster: Cluster):
    """Records in a cluster that retained display text."""
    return tuple(record for record in cluster.records if record.display is not None)


def _records_compatible(left: Cluster, right: Cluster) -> Tuple[bool, str]:
    """
    Whether two clusters can be the same case, and why.

    The precision intervals must overlap.  Two retained values must also
    agree.  A retained value on one side and a display on the other must
    be consistent: the value must render back to that display under the
    display side's format, or fall inside its precision interval when the
    format is unknown.
    """
    if left.interval is None or right.interval is None:
        return False, 'missing precision interval'
    if (left.interval & right.interval) is None:
        return False, 'precision intervals do not overlap'
    if left.has_conflicting_values or right.has_conflicting_values:
        return False, 'identity retains conflicting values'

    if left.exact_value is not None and right.exact_value is not None:
        if value_interval(left.exact_value) == value_interval(right.exact_value):
            return True, 'same retained value'
        return False, 'different retained values'

    if left.exact_value is not None:
        displays = _display_records(right)
        if displays and not all(
                _exact_matches_display(left.exact_value, record.display, record.format)
                for record in displays):
            return False, 'retained value does not reproduce the displayed text'
        return True, 'retained value is consistent with the displayed precision'

    if right.exact_value is not None:
        displays = _display_records(left)
        if displays and not all(
                _exact_matches_display(right.exact_value, record.display, record.format)
                for record in displays):
            return False, 'retained value does not reproduce the displayed text'
        return True, 'retained value is consistent with the displayed precision'

    shared = set(left.displays) & set(right.displays)
    if shared:
        return True, 'same display string; identity not confirmed'
    return False, 'different display strings without a retained value'


def _value_reproduces_other(exact_cluster, display_cluster) -> bool:
    """
    Whether an exact value reproduces one of the other side's displays.

    This is evidence that two records were associated even when their
    stored display strings differ (for example, one batch printed fewer
    digits than another).  A value that simply falls inside a wide
    interval is not evidence, because virtually every large case number
    would then match every display.
    """
    if exact_cluster.exact_value is None:
        return False
    return any(
        _exact_matches_display(exact_cluster.exact_value, record.display, record.format)
        for record in _display_records(display_cluster)
    )


def _correspondences(clusters, links):
    """
    Build correspondences between clusters.

    An edge exists when two clusters are compatible.  An edge is unique
    only when both endpoints have degree one and at least one endpoint
    retains an exact value.  Display-only matches, even with identical
    text, are reported as possible rather than merged.

    Confirmed links are emitted as unique correspondences even when the
    two records already belong to one cluster, so the linkage table's
    accepted identities are visible in the result.
    """
    edges = {}
    neighbors = defaultdict(set)
    for i, left in enumerate(clusters):
        for right in clusters[i + 1:]:
            if left.provenance & right.provenance:
                continue

            # Candidates need evidence of an association: a shared
            # display string, two retained values, or a retained value
            # that reproduces the other side's display under a known
            # batch format.  Overlapping intervals alone are not enough:
            # otherwise an added number would pull in unrelated cases.
            shared_display = bool(set(left.displays) & set(right.displays))
            both_exact = left.has_exact_value and right.has_exact_value
            format_evidenced = _value_reproduces_other(left, right) or _value_reproduces_other(
                right, left
            )
            if not (shared_display or both_exact or format_evidenced):
                continue

            compatible, reason = _records_compatible(left, right)
            if not compatible:
                continue
            overlap = left.interval & right.interval
            if overlap is None:
                continue
            edges[(left.cluster_id, right.cluster_id)] = (left, right, overlap, reason)
            neighbors[left.cluster_id].add(right.cluster_id)
            neighbors[right.cluster_id].add(left.cluster_id)

    correspondences = []
    for (left_id, right_id), (left, right, overlap, reason) in edges.items():
        anchored = left.has_exact_value or right.has_exact_value
        unique = anchored and len(neighbors[left_id]) == 1 and len(neighbors[right_id]) == 1
        if unique:
            reason = f'{reason}; mutually exclusive candidate'
        elif not anchored:
            reason = f'{reason}; no retained value to anchor the identity'
        correspondences.append(
            Correspondence(
                left=left,
                right=right,
                interval=overlap,
                unique=unique,
                display=next(
                    (display for display in left.displays if display in right.displays),
                    None,
                ),
                reason=reason,
            )
        )

    known = {
        record.key: record
        for cluster in clusters
        for record in cluster.records
    }
    cluster_by_key = {
        record.key: cluster for cluster in clusters for record in cluster.records
    }
    seen_links = {
        frozenset((correspondence.left.cluster_id, correspondence.right.cluster_id))
        for correspondence in correspondences
        if correspondence.unique
    }
    for link in links:
        if not link.confirmed:
            continue
        if link.left not in known or link.right not in known:
            continue
        left = cluster_by_key[link.left]
        right = cluster_by_key[link.right]
        key = frozenset((left.cluster_id, right.cluster_id))
        if key in seen_links:
            continue
        overlap = left.interval & right.interval if (
            left.interval is not None and right.interval is not None
        ) else None
        correspondences.append(
            Correspondence(
                left=left,
                right=right,
                interval=overlap,
                unique=left.interval is not None and right.interval is not None,
                display=next(
                    (display for display in left.displays if display in right.displays),
                    None,
                ),
                reason=link.evidence or 'confirmed identity link',
                left_key=link.left,
                right_key=link.right,
            )
        )
        seen_links.add(key)

    correspondences.sort(key=lambda c: (c.left.cluster_id, c.right.cluster_id))
    return correspondences


def _normalize_formats(formats) -> Mapping[str, Any]:
    """
    Normalize a batch-format table.

    Accepts a mapping of batch/source to format, an iterable of rows with
    ``batch`` and ``format`` keys, or a DataFrame with those columns.
    """
    if formats is None:
        return {}
    if isinstance(formats, Mapping):
        return dict(formats)
    result = {}
    for row in _iter_mappings(formats):
        if not isinstance(row, Mapping):
            raise TypeError(f'Unsupported format row {row!r}')
        batch = _first(row.get, 'batch', 'batch_id', 'period', 'source', 'name')
        fmt = _first(row.get, 'format', 'sas_format', 'fmt', 'field_format')
        if batch is not None and fmt is not None:
            result[batch] = fmt
    return result


def reconcile(records: Iterable, links: Iterable = (), formats: Mapping[str, Any] = None):
    """
    Reconcile case records using display text, retained values, and links.

    ``records`` is an iterable of :class:`CaseRecord` objects, mappings,
    or a pandas ``DataFrame``.  Mappings and tuples accept ``source``,
    ``record_id``, ``display``, ``value``, ``format``, and ``batch``.
    ``formats`` maps a batch or source name to a default SAS format.

    ``links`` is an iterable of :class:`Link` objects, two-item pairs, or
    mappings with ``left``, ``right``, ``confirmed``, and ``evidence``.
    Only confirmed links merge records into one identity.  Links whose
    endpoints are ambiguous are reported as warnings rather than guessed.

    The returned :class:`Reconciliation` separates uniquely recovered
    matches from matches that remain possible, and gives each one a
    precision interval.  Display text is used to find candidates, never
    to merge identities: two sources that show the same text remain two
    clusters until a confirmed link says otherwise.
    """
    formats = _normalize_formats(formats)
    mappings = _iter_mappings(records)
    records = tuple(_coerce_record(record, i, formats) for i, record in enumerate(mappings))

    warnings: List[str] = []
    records_by_id = defaultdict(list)
    seen_keys = set()
    unique_records = []
    for record in records:
        if record.key in seen_keys:
            warnings.append(f'Duplicate record key {record.key!r}; keeping the first')
            continue
        seen_keys.add(record.key)
        unique_records.append(record)
        records_by_id[record.record_id].append(record.key)

    links = tuple(_coerce_link(link, records_by_id) for link in _iter_mappings(links))

    known = {record.key for record in unique_records}
    for link in links:
        if link.left not in known or link.right not in known:
            warnings.append(
                f'Link {link.left!r} -- {link.right!r} references an unknown or ambiguous record'
            )

    clusters = _build_clusters(unique_records, links, warnings)

    grouped = defaultdict(list)
    for cluster in clusters:
        for display in cluster.displays:
            grouped[display].append(cluster)

    groups = []
    for display in sorted(grouped):
        group_clusters = sorted(grouped[display], key=lambda c: c.cluster_id)
        if len(group_clusters) > 1:
            groups.append(DisplayGroup(display, tuple(group_clusters)))

    correspondences = _correspondences(clusters, links)

    conflicts = []
    for cluster in clusters:
        if cluster.has_conflicting_values:
            conflicts.append(Conflict(cluster, 'Identity retains conflicting values'))
        if any(
                'two rows from the same source and batch' in warning
                for warning in cluster.warnings):
            conflicts.append(
                Conflict(cluster, 'Confirmed identity contains two rows from the same batch')
            )
        if 'Precision intervals do not overlap' in cluster.warnings:
            conflicts.append(Conflict(cluster, 'Precision intervals do not overlap'))
        elif any(record.interval is None for record in cluster.records):
            conflicts.append(Conflict(cluster, 'No usable numeric evidence'))
        for record in cluster.records:
            if record.value_display_conflict:
                conflicts.append(
                    Conflict(
                        cluster,
                        f'Record {record.key!r} value lies outside its retained display precision',
                    )
                )
            elif record.reproduces_display is False:
                warnings.append(
                    f'Record {record.key!r} format cannot produce its retained display; '
                    'the value is still used as exact evidence'
                )

    return Reconciliation(
        records=tuple(unique_records),
        links=links,
        clusters=clusters,
        groups=groups,
        correspondences=correspondences,
        conflicts=conflicts,
        warnings=warnings,
    )


reconcile_case_numbers = reconcile
resolve_case_numbers = reconcile


def _read_table(path):
    """Read a JSON, CSV, or XPT table into a list of mappings."""
    import csv
    import json
    import os

    _, extension = os.path.splitext(path)
    extension = extension.lower()
    if extension in ('.json', '.jsonl'):
        with open(path, encoding='utf-8') as file:
            if extension == '.jsonl':
                return [json.loads(line) for line in file if line.strip()]
            payload = json.load(file)
        if isinstance(payload, Mapping):
            for key in ('records', 'rows', 'data'):
                if key in payload:
                    return list(payload[key])
            return [payload]
        return list(payload)
    if extension in ('.csv', '.tsv'):
        delimiter = '\t' if extension == '.tsv' else ','
        with open(path, newline='', encoding='utf-8') as file:
            return list(csv.DictReader(file, delimiter=delimiter))
    if extension in ('.xpt', '.xpt5', '.xpt8'):
        from . import v56, v89
        with open(path, 'rb') as file:
            bytestring = file.read()
        loader = v89.loads if bytestring.startswith(
            b'HEADER RECORD*******LIBV8') else v56.loads
        library = loader(bytestring)
        dataset = next(iter(library.values()))
        return dataset.to_dict('records')
    raise ValueError(f'Unsupported table format {extension!r}')


def _main(argv=None):
    """
    Reconcile tables from the command line.

        python -m xport.reconciliation records.json --links links.json \\
            --formats formats.json --output report.json
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog='xport.reconciliation',
        description='Reconcile case numbers across sources and formats.',
    )
    parser.add_argument('records', help='JSON, JSONL, CSV, TSV, or XPT records')
    parser.add_argument('--links', help='confirmed identity links (JSON or CSV)')
    parser.add_argument('--formats', help='per-batch format table (JSON or CSV)')
    parser.add_argument(
        '--original',
        action='append',
        default=[],
        metavar='SOURCE:ID=VALUE',
        help='supply an original number, for example SAE:s2=1234567890125',
    )
    parser.add_argument('--output', help='write the JSON report here instead of stdout')
    args = parser.parse_args(argv)

    records = _read_table(args.records)
    links = _read_table(args.links) if args.links else ()
    formats = _read_table(args.formats) if args.formats else {}
    result = reconcile(records, links, formats)

    for assignment in args.original:
        try:
            target, raw = assignment.split('=', 1)
            source, record_id = target.split(':', 1)
        except ValueError:
            parser.error(f'Expected SOURCE:ID=VALUE, got {assignment!r}')
        try:
            value = float(raw)
        except ValueError:
            parser.error(f'Expected a number, got {raw!r}')
        try:
            result = result.with_original(source, _coerce_original_id(record_id), value)
        except (KeyError, ValueError) as error:
            parser.error(str(error))

    payload = json.dumps(result.as_dict(), ensure_ascii=False, indent=2, default=str)
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as file:
            file.write(payload)
    else:
        print(payload)
    return result


def _coerce_original_id(text):
    """Convert a command-line identifier to the record's stored type."""
    try:
        return int(text)
    except ValueError:
        return text


if __name__ == '__main__':
    _main()
