"""
Compare two annotation versions (GFF3/GTF) and map features across them.

The goal is to recover *provable* relationships between the features of an
old and a new annotation of the same genome:

- continuation: a feature that survives unchanged (identical coordinates)
- boundary shift: 1:1 overlap with changed bounds
- split: one old feature corresponds to several new features
- merge: several old features correspond to one new feature
- novel / retired: features only present in one version

Matching is based on coordinates, parent-child relationships and explicit
identity (feature IDs, version-stripped IDs, ``previous_ID`` attributes) --
never on ordinal attributes such as ``exon_number``, which are only carried
along as labels.  This matters when e.g. a transcript revision inserts one
exon and every downstream ``exon_number`` shifts by one: coordinate-based
matching still maps each old exon to its true counterpart.

When repeated (identical) segments make a unique assignment impossible, all
candidates are listed instead of forcing a 1:1 pairing.  Every relation
record keeps the old feature's original ID and coordinates, so old-version
hits always remain traceable to their original positions rather than being
silently renumbered to the new ordinals.

Can be used as a library::

    import gffutils.compare
    result = gffutils.compare.compare_annotations('v1.gff3', 'v2.gff3')
    print(result.summary())
    result.to_tsv('mapping.tsv')

or as a command line tool::

    python -m gffutils.compare old.gff3 new.gff3 -o mapping.tsv
"""

import argparse
import re
import sys
from dataclasses import dataclass, field

import gffutils

# Relationship labels
CONTINUATION = "continuation"      # exact coordinate match, 1:1
BOUNDARY_SHIFT = "boundary_shift"  # 1:1 overlap, different bounds
SPLIT = "split"                    # one old -> several new
MERGED = "merged"                  # several old -> one new
COMPLEX = "complex"                # several old -> several new
AMBIGUOUS = "ambiguous"            # duplicates prevent a unique pairing
NOVEL = "novel"                    # only in the new annotation
RETIRED = "retired"                # only in the old annotation

# featuretypes treated as transcripts, in addition to anything ending
# in "_transcript" or "RNA" found in the files.
TRANSCRIPT_TYPES = (
    "mRNA",
    "transcript",
    "ncRNA",
    "lncRNA",
    "lnc_RNA",
    "rRNA",
    "tRNA",
    "snRNA",
    "snoRNA",
    "miRNA",
    "pseudogenic_transcript",
)

# attributes on a *new* transcript that explicitly point back at an old ID
PREVIOUS_ID_ATTRIBUTES = ("previous_ID", "old_ID", "replaces")

# attributes consulted (read-only) for exon ordinals
EXON_NUMBER_ATTRIBUTES = ("exon_number", "exonnumber", "rank")

_VERSION_SUFFIX = re.compile(r"\.\d+$")


def _base_id(feature_id):
    """Strip a trailing version suffix (``.N``) from an ID."""
    return _VERSION_SUFFIX.sub("", feature_id)


def _first_attr(feature, names):
    for name in names:
        if name in feature.attributes:
            values = feature.attributes[name]
            if values:
                return values[0]
    return None


@dataclass
class Segment:
    """
    A comparable child feature (e.g. an exon) of one annotation version.

    ``number`` is the ordinal label from the source file (e.g. exon_number);
    it is reported for traceability but never used for matching.
    """

    feature_id: str
    seqid: str
    start: int
    end: int
    strand: str
    transcript_id: str
    number: str = None

    @property
    def key(self):
        """Coordinate identity: the only thing treated as provable sameness."""
        return (self.seqid, self.start, self.end, self.strand)

    @property
    def label(self):
        return "%s:%s-%s(%s)" % (self.seqid, self.start, self.end, self.strand)


@dataclass
class Transcript:
    feature_id: str
    gene_id: str
    segments: list = field(default_factory=list)
    previous_ids: tuple = ()


@dataclass
class Candidate:
    """One possible new-annotation counterpart of an old segment."""

    transcript_id: str
    feature_id: str
    seqid: str
    start: int
    end: int
    strand: str
    number: str = None

    @classmethod
    def from_segment(cls, segment):
        return cls(
            transcript_id=segment.transcript_id,
            feature_id=segment.feature_id,
            seqid=segment.seqid,
            start=segment.start,
            end=segment.end,
            strand=segment.strand,
            number=segment.number,
        )


@dataclass
class Relation:
    """
    How one old-annotation segment (or, for ``novel``, one new segment)
    relates to the other annotation version.

    ``old`` is None for novel segments; ``candidates`` holds every plausible
    new-annotation counterpart (more than one only when the assignment is
    not provably unique).
    """

    relation: str
    old: Segment = None
    candidates: tuple = ()
    note: str = ""

    def as_row(self):
        old = self.old
        rows = []
        cands = self.candidates or (None,)
        for cand in cands:
            rows.append(
                [
                    self.relation,
                    old.transcript_id if old else "",
                    old.feature_id if old else "",
                    old.seqid if old else "",
                    str(old.start) if old else "",
                    str(old.end) if old else "",
                    old.number or "" if old else "",
                    cand.transcript_id if cand else "",
                    cand.feature_id if cand else "",
                    cand.seqid if cand else "",
                    str(cand.start) if cand else "",
                    str(cand.end) if cand else "",
                    cand.number or "" if cand else "",
                    self.note,
                ]
            )
        return rows


TSV_HEADER = [
    "relation",
    "old_transcript",
    "old_feature_id",
    "old_seqid",
    "old_start",
    "old_end",
    "old_exon_number",
    "new_transcript",
    "new_feature_id",
    "new_seqid",
    "new_start",
    "new_end",
    "new_exon_number",
    "note",
]


def _load_db(source, create_kwargs=None):
    if hasattr(source, "features_of_type"):
        return source
    kwargs = dict(
        force=True,
        keep_order=True,
        # keep duplicated features as separate entries so repeated
        # segments stay visible instead of being merged away
        merge_strategy="create_unique",
    )
    if create_kwargs:
        kwargs.update(create_kwargs)
    return gffutils.create_db(source, ":memory:", **kwargs)


def _transcript_types(db, extra=()):
    wanted = set(TRANSCRIPT_TYPES) | set(extra)
    found = set()
    for ftype in db.featuretypes():
        if ftype in wanted or ftype.endswith("_transcript"):
            found.add(ftype)
    return sorted(found)


def _gene_id_of(db, transcript_feature):
    for parent in db.parents(transcript_feature, featuretype="gene"):
        return parent.id
    return _first_attr(transcript_feature, ("gene_id", "gene", "gene_name"))


def load_transcripts(source, featuretype="exon", transcript_types=None,
                     create_kwargs=None):
    """
    Load transcripts and their child segments from a file or FeatureDB.

    Returns a list of Transcript objects whose ``segments`` are sorted by
    coordinate (never by exon_number).
    """
    db = _load_db(source, create_kwargs)
    types = transcript_types or _transcript_types(db)
    transcripts = []
    for tx_feature in db.features_of_type(types):
        previous_ids = []
        for name in PREVIOUS_ID_ATTRIBUTES:
            previous_ids.extend(tx_feature.attributes.get(name, []))
        tx = Transcript(
            feature_id=tx_feature.id,
            gene_id=_gene_id_of(db, tx_feature),
            previous_ids=tuple(previous_ids),
        )
        for child in db.children(tx_feature, featuretype=featuretype):
            tx.segments.append(
                Segment(
                    feature_id=child.id,
                    seqid=child.seqid,
                    start=child.start,
                    end=child.end,
                    strand=child.strand,
                    transcript_id=tx.feature_id,
                    number=_first_attr(child, EXON_NUMBER_ATTRIBUTES),
                )
            )
        tx.segments.sort(key=lambda s: (s.seqid, s.start, s.end))
        transcripts.append(tx)
    transcripts.sort(key=lambda t: t.feature_id)
    return transcripts


def _overlap(a, b):
    return (
        a.seqid == b.seqid
        and a.strand == b.strand
        and a.start <= b.end
        and b.start <= a.end
    )


def _exon_overlap(a, b):
    return any(_overlap(sa, sb) for sa in a.segments for sb in b.segments)


def match_transcripts(old_transcripts, new_transcripts):
    """
    Pair transcripts across versions using explicit identity first
    (exact ID, version-stripped ID, ``previous_ID`` attributes), then
    gene membership plus coordinate overlap, then bare overlap.

    Returns (pairs, ambiguous, unmatched_old, unmatched_new) where
    ``pairs`` is a list of (old, new, method) tuples and ``ambiguous``
    is a list of (old, [new candidates], method) tuples.
    """
    by_id = {}
    by_base = {}
    by_previous = {}
    for tx in new_transcripts:
        by_id.setdefault(tx.feature_id, []).append(tx)
        by_base.setdefault(_base_id(tx.feature_id), []).append(tx)
        for prev in tx.previous_ids:
            by_previous.setdefault(prev, []).append(tx)

    pairs = []
    ambiguous = []
    claimed = set()

    def candidates_by_gene(old):
        return [
            tx
            for tx in new_transcripts
            if old.gene_id and tx.gene_id == old.gene_id and _exon_overlap(old, tx)
        ]

    def candidates_by_overlap(old):
        return [tx for tx in new_transcripts if _exon_overlap(old, tx)]

    # explicit identity always wins over coordinate overlap, so run the
    # methods in priority passes rather than transcript by transcript
    passes = [
        ("id", lambda old: by_id.get(old.feature_id, [])),
        ("id_base", lambda old: by_base.get(_base_id(old.feature_id), [])),
        ("previous_ID", lambda old: by_previous.get(old.feature_id, [])),
        ("gene_overlap", candidates_by_gene),
        ("overlap", candidates_by_overlap),
    ]

    remaining = list(old_transcripts)
    for method, get_candidates in passes:
        still_unmatched = []
        for old in remaining:
            candidates = [c for c in get_candidates(old) if id(c) not in claimed]
            if len(candidates) == 1:
                pairs.append((old, candidates[0], method))
                claimed.add(id(candidates[0]))
            elif len(candidates) > 1:
                ambiguous.append((old, candidates, method))
                for c in candidates:
                    claimed.add(id(c))
            else:
                still_unmatched.append(old)
        remaining = still_unmatched

    unmatched_old = remaining

    paired_new = {id(new) for _, new, _ in pairs}
    for _, cands, _ in ambiguous:
        paired_new.update(id(c) for c in cands)
    unmatched_new = [tx for tx in new_transcripts if id(tx) not in paired_new]
    return pairs, ambiguous, unmatched_old, unmatched_new


def _connected_components(old_segments, new_segments):
    """Connected components of the bipartite coordinate-overlap graph."""
    n_old, n_new = len(old_segments), len(new_segments)
    adj_old = {i: set() for i in range(n_old)}
    adj_new = {j: set() for j in range(n_new)}
    for i, o in enumerate(old_segments):
        for j, n in enumerate(new_segments):
            if _overlap(o, n):
                adj_old[i].add(j)
                adj_new[j].add(i)
    seen_old, seen_new = set(), set()
    components = []
    for i in range(n_old):
        if i in seen_old or not adj_old[i]:
            continue
        comp_old, comp_new = set(), set()
        stack = [("old", i)]
        while stack:
            side, idx = stack.pop()
            if side == "old":
                if idx in comp_old:
                    continue
                comp_old.add(idx)
                stack.extend(("new", j) for j in adj_old[idx])
            else:
                if idx in comp_new:
                    continue
                comp_new.add(idx)
                stack.extend(("old", j) for j in adj_new[idx])
        seen_old.update(comp_old)
        seen_new.update(comp_new)
        components.append(
            ([old_segments[i] for i in sorted(comp_old)],
             [new_segments[j] for j in sorted(comp_new)])
        )
    matched_old = seen_old
    matched_new = seen_new
    components.append(
        ([old_segments[i] for i in range(n_old) if i not in matched_old], [])
    )
    components.append(
        ([], [new_segments[j] for j in range(n_new) if j not in matched_new])
    )
    return components


def compare_segment_sets(old_segments, new_segments):
    """
    Compare the segments (e.g. exons) of one old transcript against those
    of one or more candidate new transcripts.

    Exact-coordinate duplicates on either side are never forced into a 1:1
    pairing; they yield ``ambiguous`` relations listing every candidate.
    """
    relations = []

    old_by_key = {}
    for seg in old_segments:
        old_by_key.setdefault(seg.key, []).append(seg)
    new_by_key = {}
    for seg in new_segments:
        new_by_key.setdefault(seg.key, []).append(seg)

    consumed_old = set()
    consumed_new = set()

    # pass 1: exact coordinate identity
    for key in sorted(old_by_key):
        olds = old_by_key[key]
        news = new_by_key.get(key, [])
        if not news:
            continue
        for seg in olds:
            consumed_old.add(id(seg))
        for seg in news:
            consumed_new.add(id(seg))
        if len(olds) == 1 and len(news) == 1:
            relations.append(
                Relation(
                    CONTINUATION,
                    old=olds[0],
                    candidates=(Candidate.from_segment(news[0]),),
                )
            )
        else:
            note = "identical segment occurs %d time(s) in old and %d in new" % (
                len(olds),
                len(news),
            )
            for seg in olds:
                relations.append(
                    Relation(
                        AMBIGUOUS,
                        old=seg,
                        candidates=tuple(Candidate.from_segment(n) for n in news),
                        note=note,
                    )
                )

    remaining_old = [s for s in old_segments if id(s) not in consumed_old]
    remaining_new = [s for s in new_segments if id(s) not in consumed_new]

    # pass 2: overlap components for split / merge / shift
    for olds, news in _connected_components(remaining_old, remaining_new):
        if not olds and not news:
            continue
        if olds and not news:
            for seg in olds:
                relations.append(Relation(RETIRED, old=seg))
            continue
        if news and not olds:
            for seg in news:
                relations.append(
                    Relation(NOVEL, candidates=(Candidate.from_segment(seg),))
                )
            continue
        if len(olds) == 1 and len(news) == 1:
            relations.append(
                Relation(
                    BOUNDARY_SHIFT,
                    old=olds[0],
                    candidates=(Candidate.from_segment(news[0]),),
                )
            )
        elif len(olds) == 1:
            relations.append(
                Relation(
                    SPLIT,
                    old=olds[0],
                    candidates=tuple(Candidate.from_segment(n) for n in news),
                )
            )
        elif len(news) == 1:
            for seg in olds:
                relations.append(
                    Relation(
                        MERGED,
                        old=seg,
                        candidates=(Candidate.from_segment(news[0]),),
                    )
                )
        else:
            cands = tuple(Candidate.from_segment(n) for n in news)
            for seg in olds:
                relations.append(
                    Relation(
                        COMPLEX,
                        old=seg,
                        candidates=cands,
                        note="%d old x %d new overlapping segments"
                        % (len(olds), len(news)),
                    )
                )
    return relations


class ComparisonResult:
    """Outcome of :func:`compare_annotations`."""

    def __init__(self):
        self.relations = []
        self.transcript_pairs = []       # (old_id, new_id, method)
        self.ambiguous_transcripts = []  # (old_id, [new_ids], method)
        self.unmatched_old = []          # transcript IDs
        self.unmatched_new = []          # transcript IDs

    def summary(self):
        counts = {}
        for rel in self.relations:
            counts[rel.relation] = counts.get(rel.relation, 0) + 1
        return {
            "transcript_pairs": len(self.transcript_pairs),
            "ambiguous_transcripts": len(self.ambiguous_transcripts),
            "unmatched_old_transcripts": len(self.unmatched_old),
            "unmatched_new_transcripts": len(self.unmatched_new),
            "relations": counts,
        }

    def to_tsv(self, outfile):
        close = False
        if isinstance(outfile, str):
            outfile = open(outfile, "w")
            close = True
        try:
            outfile.write("\t".join(TSV_HEADER) + "\n")
            for rel in self.relations:
                for row in rel.as_row():
                    outfile.write("\t".join(row) + "\n")
        finally:
            if close:
                outfile.close()


def compare_annotations(old, new, featuretype="exon", transcript_types=None,
                        create_kwargs=None):
    """
    Compare two annotation versions.

    ``old`` and ``new`` may be filenames or gffutils FeatureDB objects.
    Returns a ComparisonResult.  Matching uses coordinates, parent
    relationships and explicit identity only -- never exon_number.
    """
    old_txs = load_transcripts(
        old, featuretype, transcript_types, create_kwargs
    )
    new_txs = load_transcripts(
        new, featuretype, transcript_types, create_kwargs
    )

    result = ComparisonResult()
    pairs, ambiguous, unmatched_old, unmatched_new = match_transcripts(
        old_txs, new_txs
    )
    result.transcript_pairs = [
        (o.feature_id, n.feature_id, method) for o, n, method in pairs
    ]
    result.ambiguous_transcripts = [
        (o.feature_id, [c.feature_id for c in cands], method)
        for o, cands, method in ambiguous
    ]
    result.unmatched_old = [tx.feature_id for tx in unmatched_old]
    result.unmatched_new = [tx.feature_id for tx in unmatched_new]

    # global index of new segments by coordinate key, so that segments of
    # retired transcripts can still be traced to identical segments that
    # survived elsewhere (listed as candidates, never renumbered)
    new_key_index = {}
    for tx in new_txs:
        for seg in tx.segments:
            new_key_index.setdefault(seg.key, []).append(seg)

    matched_pairs = [(o, [n]) for o, n, _ in pairs]
    matched_pairs += [(o, cands) for o, cands, _ in ambiguous]

    for old_tx, new_txs_for_old in matched_pairs:
        new_segments = [
            seg for tx in new_txs_for_old for seg in tx.segments
        ]
        result.relations.extend(
            compare_segment_sets(old_tx.segments, new_segments)
        )

    for old_tx in unmatched_old:
        for seg in old_tx.segments:
            survivors = new_key_index.get(seg.key, [])
            if survivors:
                result.relations.append(
                    Relation(
                        RETIRED,
                        old=seg,
                        candidates=tuple(
                            Candidate.from_segment(s) for s in survivors
                        ),
                        note="transcript retired; identical segment survives "
                        "in %d new feature(s)" % len(survivors),
                    )
                )
            else:
                result.relations.append(Relation(RETIRED, old=seg))

    for new_tx in unmatched_new:
        for seg in new_tx.segments:
            result.relations.append(
                Relation(NOVEL, candidates=(Candidate.from_segment(seg),))
            )

    result.relations.sort(
        key=lambda r: (
            r.old.seqid if r.old else "",
            r.old.start if r.old else 0,
            r.old.end if r.old else 0,
            r.relation,
        )
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare two annotation versions and map features "
        "across them by coordinates, parentage and explicit identity "
        "(never by exon_number)."
    )
    parser.add_argument("old", help="old annotation (GFF3/GTF)")
    parser.add_argument("new", help="new annotation (GFF3/GTF)")
    parser.add_argument(
        "-f",
        "--featuretype",
        default="exon",
        help="child featuretype to compare (default: exon)",
    )
    parser.add_argument(
        "-t",
        "--transcript-type",
        action="append",
        default=None,
        help="featuretype(s) treated as transcripts; may be repeated",
    )
    parser.add_argument("-o", "--out", help="write TSV mapping here")
    args = parser.parse_args(argv)

    result = compare_annotations(
        args.old,
        args.new,
        featuretype=args.featuretype,
        transcript_types=args.transcript_type,
    )
    summary = result.summary()
    for key, value in summary.items():
        print("%s: %s" % (key, value), file=sys.stderr)
    if args.out:
        result.to_tsv(args.out)
    else:
        result.to_tsv(sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
