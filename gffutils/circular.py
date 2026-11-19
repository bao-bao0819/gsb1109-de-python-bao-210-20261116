"""
Support for features on circular sequences (for example, bacterial
chromosomes or plasmids) whose coordinates cross the origin.

GFF and GTF coordinates are normally linear and satisfy ``start <= end``.  A
feature that crosses the origin of a circular sequence has no such linear
representation; a common convention is to store it with ``start > end``.  In
that case the feature covers two segments::

    [start, sequence_length] and [1, end]

Resolving those segments requires the length of the sequence.  Without it the
coordinates are *undecided*: :class:`~gffutils.exceptions.MissingSequenceLengthError`
is raised instead of guessing by swapping the endpoints (which would select
almost the entire chromosome).
"""

import json
import os
from collections.abc import Mapping

from gffutils.exceptions import MissingSequenceLengthError

__all__ = [
    "MissingSequenceLengthError",
    "LogicalFeature",
    "load_sequence_lengths",
    "dump_sequence_lengths",
    "is_wrapped",
    "crosses_origin",
    "coverage_segments",
    "covered_length",
    "logical_span",
    "merge_segments",
    "circular_extent",
    "segments_overlap",
    "segments_within",
    "reverse_complement",
]


_COMPLEMENT = str.maketrans(
    "ACGTRYMKBDHVNacgtrymkbdhvn", "TGCAYRKMVHDBNtgcayrkmvhdbn"
)


def reverse_complement(sequence):
    """Return the reverse complement of a DNA sequence."""
    return sequence[::-1].translate(_COMPLEMENT)


def load_sequence_lengths(sequence_lengths):
    """
    Normalize a sequence-length specification into a ``{seqid: length}`` dict.

    Accepted inputs are:

    - ``None`` (returns an empty dict)
    - a mapping of ``{seqid: length}``
    - an iterable of ``(seqid, length)`` pairs
    - a single integer, used as a fallback length for any sequence (stored
      under the ``None`` key)
    - a path to a two-column "chrom.sizes" file (whitespace-delimited)
    - a path to a FASTA file, in which case sequence lengths are read from it
    - a ``pyfaidx.Fasta`` object

    Values are converted to integers.
    """
    if sequence_lengths is None:
        return {}

    if isinstance(sequence_lengths, bool):
        raise TypeError("sequence_lengths must not be a boolean")

    if isinstance(sequence_lengths, int):
        return {None: int(sequence_lengths)}

    if isinstance(sequence_lengths, Mapping):
        return {
            (None if k is None else str(k)): int(v)
            for k, v in sequence_lengths.items()
        }

    if isinstance(sequence_lengths, (list, tuple)):
        return {
            (None if k is None else str(k)): int(v)
            for k, v in sequence_lengths
        }

    if isinstance(sequence_lengths, (str, os.PathLike)):
        path = os.fspath(sequence_lengths)
        with open(path) as fin:
            first_line = fin.readline()

        if first_line.startswith(">"):
            from pyfaidx import Fasta

            fasta = Fasta(path)
            return {str(name): len(fasta[name]) for name in fasta.keys()}

        lengths = {}
        if first_line.strip() and not first_line.lstrip().startswith("#"):
            fields = first_line.split()
            if len(fields) < 2:
                raise ValueError(
                    "Expected whitespace-delimited 'seqid length' lines in %s" % path
                )
            lengths[fields[0]] = int(fields[1])
        with open(path) as fin:
            for line in fin:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                fields = line.split()
                if len(fields) < 2:
                    raise ValueError(
                        "Expected whitespace-delimited 'seqid length' lines in %s"
                        % path
                    )
                lengths[fields[0]] = int(fields[1])
        return lengths

    # A pyfaidx.Fasta object (or anything that behaves like one).
    keys = getattr(sequence_lengths, "keys", None)
    if keys is not None:
        return {str(name): len(sequence_lengths[name]) for name in keys()}

    raise TypeError(
        "Unsupported sequence_lengths type: %s" % type(sequence_lengths).__name__
    )


def dump_sequence_lengths(sequence_lengths):
    """
    Serialize a ``{seqid: length}`` mapping for storage in the database.

    A fallback length (stored under the ``None`` key) is written as the
    ``"null"`` key, since JSON object keys must be strings; it is restored by
    :func:`load_sequence_lengths` / :meth:`FeatureDB.sequence_length`.
    """
    payload = {}
    for seqid, length in sequence_lengths.items():
        key = "null" if seqid is None else str(seqid)
        payload[key] = int(length)
    return json.dumps(payload, separators=(",", ":"))


def is_wrapped(start, end):
    """Return True if coordinates are stored with ``start > end``."""
    return start is not None and end is not None and int(start) > int(end)


def crosses_origin(start, end, sequence_length=None):
    """
    Report whether a feature crosses the origin.

    Returns ``False`` for ordinary coordinates.  For ``start > end``, returns
    ``None`` if the sequence length is unknown (the coordinates are
    undecided), and ``True`` once a sequence length is available.
    """
    if not is_wrapped(start, end):
        return False
    if sequence_length is None:
        return None
    return True


def coverage_segments(start, end, sequence_length=None, order="genomic"):
    """
    Return the genomic segments actually covered by a feature.

    Ordinary coordinates yield a single ``(start, end)`` segment.  Coordinates
    that cross the origin yield ``(1, end)`` and ``(start, sequence_length)``.

    With ``order="genomic"`` (default) the segments are in ascending genomic
    order.  With ``order="annotation"`` they follow the GFF start-to-end
    traversal -- that is, ``(start, sequence_length)`` followed by ``(1,
    end)`` -- which is the order in which the feature's bases are read before
    applying strand.

    Raises
    ------
    MissingSequenceLengthError
        If ``start > end`` and no sequence length is available.
    ValueError
        If ``start > end`` and the coordinates do not fit the supplied
        sequence length.
    """
    if start is None or end is None:
        return ()

    start = int(start)
    end = int(end)

    if start <= end:
        return ((start, end),)

    if sequence_length is None:
        raise MissingSequenceLengthError(
            "Cannot resolve coordinates %s-%s without a sequence length"
            % (start, end)
        )

    sequence_length = int(sequence_length)
    if not (1 <= end < start <= sequence_length):
        raise ValueError(
            "Coordinates %s-%s cannot wrap a sequence of length %s"
            % (start, end, sequence_length)
        )

    if order in ("annotation", "strand", "transcription"):
        return ((start, sequence_length), (1, end))
    return ((1, end), (start, sequence_length))


def covered_length(start, end, sequence_length=None):
    """Total number of bases covered by a (possibly wrapped) feature."""
    return sum(e - s + 1 for s, e in coverage_segments(start, end, sequence_length))


def logical_span(start, end, sequence_length=None):
    """
    Describe the logical object as ``(start, end, length)``.

    The original ``start`` and ``end`` are preserved so callers can round-trip
    the circular annotation; ``length`` is the number of bases actually
    covered.  If the coordinates are wrapped, a sequence length is required.
    """
    return (start, end, covered_length(start, end, sequence_length))


def merge_segments(segments):
    """
    Merge overlapping or adjacent 1-based closed segments.

    Segments are returned in ascending genomic order.  A segment ending at the
    last position and one starting at the first position are *not* joined
    here, since doing so would lose the fact that they meet across the origin.
    """
    segments = sorted(
        (int(s), int(e)) for s, e in segments if s is not None and e is not None
    )
    if not segments:
        return []

    merged = [list(segments[0])]
    for start, end in segments[1:]:
        if start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return [tuple(i) for i in merged]


def circular_extent(segments, sequence_length):
    """
    Return the smallest circular interval ``(start, end)`` containing all
    segments.

    The result uses ``start > end`` to indicate that the interval crosses the
    origin.  When the intervals cover the whole sequence, ``(1,
    sequence_length)`` is returned.  Among equally sized gaps, a non-wrapping
    extent is preferred.
    """
    if sequence_length is None:
        raise MissingSequenceLengthError(
            "Cannot compute a circular extent without a sequence length"
        )

    sequence_length = int(sequence_length)
    merged = merge_segments(segments)
    if not merged:
        return None
    if len(merged) == 1:
        return merged[0]

    total = sum(e - s + 1 for s, e in merged)
    if total >= sequence_length:
        return (1, sequence_length)

    # Gap between the last and first segment, measured across the origin.
    best_gap = sequence_length - merged[-1][1] + merged[0][0] - 1
    best_index = -1  # -1 means the wrap gap

    for index in range(len(merged) - 1):
        gap = merged[index + 1][0] - merged[index][1] - 1
        if gap > best_gap:
            best_gap = gap
            best_index = index

    if best_index == -1:
        return (merged[0][0], merged[-1][1])

    start = merged[best_index + 1][0]
    end = merged[best_index][1]
    return (start, end)


def segments_overlap(segments_a, segments_b):
    """Return True if any pair of 1-based closed segments overlaps."""
    for a_start, a_end in segments_a:
        for b_start, b_end in segments_b:
            if a_start <= b_end and b_start <= a_end:
                return True
    return False


def segments_within(segments, container):
    """
    Return True if every segment in ``segments`` is fully contained in one of
    the segments in ``container``.
    """
    for start, end in segments:
        if not any(c_start <= start and end <= c_end for c_start, c_end in container):
            return False
    return True


def definitely_disjoint(start, end, segments):
    """
    Decide non-overlap for an origin-crossing feature of unknown length.

    For ``start > end`` the feature covers ``start..sequence_length`` plus
    ``1..end``.  Even without knowing ``sequence_length``, it is disjoint from
    all query ``segments`` when ``start`` is beyond every query end and ``end``
    is before every query start.  Returns ``True`` only in that provable case;
    otherwise the result is undecided and ``False`` is returned.
    """
    if start is None or end is None or start <= end:
        return False
    if not segments:
        return True
    for seg_start, seg_end in segments:
        if start <= seg_end or end >= seg_start:
            return False
    return True


class LogicalFeature:
    """
    A logical view of a feature on a (possibly circular) sequence.

    The original :class:`~gffutils.feature.Feature` is available as
    ``feature``.  ``segments`` contains the actual covered 1-based closed
    genomic segments, and ``length`` is their total size.  Parent/child
    relationships and strand are not altered.
    """

    def __init__(self, feature, sequence_length=None):
        if sequence_length is None:
            sequence_length = getattr(feature, "sequence_length", None)
        self.feature = feature
        self.sequence_length = sequence_length
        self.id = feature.id
        self.seqid = feature.seqid
        self.source = feature.source
        self.featuretype = feature.featuretype
        self.start = feature.start
        self.end = feature.end
        self.score = feature.score
        self.strand = feature.strand
        self.frame = feature.frame
        self.attributes = feature.attributes
        self.extra = feature.extra
        self.bin = feature.bin
        self.file_order = feature.file_order
        self.crosses_origin = crosses_origin(
            feature.start, feature.end, sequence_length
        )
        self.segments = coverage_segments(
            feature.start,
            feature.end,
            sequence_length,
            order="annotation",
        )
        self.length = sum(e - s + 1 for s, e in self.segments)

    def __getattr__(self, name):
        return getattr(self.feature, name)

    def __str__(self):
        return str(self.feature)

    def __repr__(self):
        return "<LogicalFeature %s (%s:%s-%s[%s], %s segments)>" % (
            self.featuretype,
            self.seqid,
            self.start,
            self.end,
            self.strand,
            len(self.segments),
        )
