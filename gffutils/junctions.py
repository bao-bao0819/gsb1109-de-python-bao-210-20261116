"""
Resolve splice-junction observations against annotated transcripts.

The functions here are separate from the database layer:
:func:`read_junctions` parses an offline junction table, and
:func:`explain_junctions` takes a :class:`~gffutils.interface.FeatureDB`
(already loaded with the annotation) and explains each molecule as a whole.

The central rule is that all junctions from one molecule must be explained by
one transcript.  Per-junction matches are still reported as supporting
evidence, but they never promote a molecule to a "found" state unless a single
transcript is compatible with every junction in that molecule.
"""

import collections
import csv
import io
import os


# Column names accepted by read_junctions().  The first name in each group is
# the canonical spelling; the rest are common aliases.
_JUNCTION_COLUMNS = {
    "molecule": (
        "molecule", "molecule_id", "read", "read_id", "fragment",
        "fragment_id", "query", "query_id",
    ),
    "start": (
        "start", "intron_start", "junction_start", "donor", "donor_end",
        "left",
    ),
    "end": (
        "end", "intron_end", "junction_end", "acceptor", "acceptor_start",
        "right",
    ),
    "seqid": ("seqid", "chrom", "chromosome", "contig"),
    "strand": ("strand",),
    "order": ("order", "junction_order", "rank", "position"),
    "id": ("id", "junction_id", "name"),
    "count": ("count", "support", "n", "reads", "depth"),
}

# Status values for a molecule group.
UNIQUE = "unique"
SHARED = "shared"
NONE = "none"

# Coordinate conventions.
INTRON = "intron"
DONOR_ACCEPTOR = "donor_acceptor"
_COORDINATE_ALIASES = {
    INTRON: INTRON,
    "intron_interval": INTRON,
    "interval": INTRON,
    DONOR_ACCEPTOR: DONOR_ACCEPTOR,
    "donor-acceptor": DONOR_ACCEPTOR,
    "splice_site": DONOR_ACCEPTOR,
    "splice_sites": DONOR_ACCEPTOR,
    "boundary": DONOR_ACCEPTOR,
    "boundaries": DONOR_ACCEPTOR,
}


def _norm_column(name):
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _norm_coordinate(value):
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return _COORDINATE_ALIASES[key]
    except KeyError:
        raise ValueError(
            "coordinate must be %r or %r, got %r"
            % (INTRON, DONOR_ACCEPTOR, value)
        )


def _resolve_columns(fieldnames, required, path):
    lookup = {}
    for canonical, aliases in _JUNCTION_COLUMNS.items():
        for alias in aliases:
            if alias in fieldnames:
                lookup[canonical] = alias
                break

    missing = [name for name in required if name not in lookup]
    if missing:
        raise ValueError(
            "%s is missing required column(s): %s (found: %s)"
            % (path, ", ".join(missing), ", ".join(fieldnames))
        )
    return lookup


def _open_source(source):
    """Return (file-like object, should_close, label)."""
    if hasattr(source, "read"):
        return source, False, getattr(source, "name", "<stream>")
    if isinstance(source, os.PathLike):
        source = os.fspath(source)
    if isinstance(source, str):
        if os.path.exists(source):
            return open(source, "r", newline=""), True, source
        # Be forgiving about tables pasted in as strings; only do this when
        # the text cannot plausibly be a missing path.
        if "\n" in source or "\t" in source:
            return io.StringIO(source), True, "<string>"
        return open(source, "r", newline=""), True, source
    raise TypeError(
        "source must be a path, a file-like object, or a table string"
    )


def _is_blank(value):
    return value is None or str(value).strip() == ""


def _parse_int(value, column, label, line_number):
    if _is_blank(value):
        raise ValueError(
            "%s line %s: empty %s value" % (label, line_number, column)
        )
    try:
        return int(str(value).strip())
    except ValueError:
        raise ValueError(
            "%s line %s: %s must be an integer, got %r"
            % (label, line_number, column, value)
        )


class Junction(object):
    """
    One splice-junction observation.

    Coordinates are 1-based and closed, and ``start``/``end`` always refer to
    the lower/higher genomic coordinate, regardless of strand.  The
    ``coordinate`` argument selects the convention:

    - ``"intron"`` (default): ``start`` and ``end`` delimit the intron itself.
      For exons ending at 200 and starting at 301, the intron is 201-300.
    - ``"donor_acceptor"``: ``start`` is the genomic exon boundary immediately
      left of the intron (the last base of the left exon) and ``end`` is the
      genomic exon boundary immediately right of the intron (the first base of
      the right exon).  For the same exons this is 200-301.

    On a minus-strand molecule the biological donor is at the higher genomic
    coordinate, but the numeric fields are unchanged; use :attr:`intron` for
    matching.  In both conventions :attr:`intron` returns the canonical
    1-based inclusive intron interval.
    """

    __slots__ = (
        "molecule", "seqid", "start", "end", "strand", "order", "id",
        "count", "coordinate", "index", "_intron",
    )

    def __init__(self, molecule, start, end, seqid=None, strand=".",
                 order=None, id=None, count=1, coordinate=INTRON,
                 index=None):
        self.coordinate = _norm_coordinate(coordinate)
        start = int(start)
        end = int(end)
        if self.coordinate == INTRON:
            intron_start, intron_end = start, end
        else:
            intron_start, intron_end = start + 1, end - 1

        if start > end:
            raise ValueError(
                "junction start (%s) must be <= end (%s)" % (start, end)
            )
        if intron_start > intron_end:
            raise ValueError(
                "junction at %s-%s does not contain an intron" % (start, end)
            )

        if strand is None:
            strand = "."
        strand = str(strand).strip()
        if strand not in ("+", "-", "."):
            raise ValueError("strand must be '+', '-', or '.', got %r" % strand)

        self.molecule = str(molecule)
        self.seqid = None if seqid is None else str(seqid)
        self.start = start
        self.end = end
        self.strand = strand
        self.order = None if order is None else int(order)
        self.id = None if id is None else str(id)
        self.count = 1 if count is None else int(count)
        self.index = None if index is None else int(index)
        self._intron = (intron_start, intron_end)

    @property
    def intron(self):
        """Canonical 1-based inclusive intron interval ``(start, end)``."""
        return self._intron

    @property
    def donor(self):
        """Genomic exon boundary immediately left of the intron (1-based)."""
        return self._intron[0] - 1

    @property
    def acceptor(self):
        """Genomic exon boundary immediately right of the intron (1-based)."""
        return self._intron[1] + 1

    def __repr__(self):
        return (
            "<Junction {molecule} {seqid}:{start}-{end}[{strand}] "
            "intron={intron} order={order}>".format(
                molecule=self.molecule,
                seqid=self.seqid,
                start=self.start,
                end=self.end,
                strand=self.strand,
                intron=self._intron,
                order=self.order,
            )
        )

    def __eq__(self, other):
        if not isinstance(other, Junction):
            return NotImplemented
        return (
            self.molecule, self.seqid, self.start, self.end, self.strand,
            self.order, self.id, self.count, self.coordinate,
        ) == (
            other.molecule, other.seqid, other.start, other.end, other.strand,
            other.order, other.id, other.count, other.coordinate,
        )

    def __hash__(self):
        return hash(
            (self.molecule, self.seqid, self.start, self.end, self.strand,
             self.order, self.id, self.coordinate)
        )


def read_junctions(source, delimiter=None, comment="#", coordinate=None,
                   **kwargs):
    """
    Read an offline junction table into :class:`Junction` objects.

    The table must be delimited (tab by default) and have a header row.
    Required columns are ``molecule``, ``start``, and ``end``.  A ``seqid``
    column is optional.  Column names are matched case-insensitively and a
    handful of common aliases are accepted (for example ``read_id``,
    ``intron_start``, ``junction_order``).

    Parameters
    ----------
    source : path, file-like object, or str
        Path to a delimited table, an open file-like object, or the table
        itself as a string.
    delimiter : str or None
        Field delimiter.  Default is a tab.
    comment : str
        Lines beginning with this string are ignored.
    coordinate : "intron", "donor_acceptor", or None
        Coordinate convention for ``start``/``end``.  If None (default),
        ``donor_acceptor`` is inferred when the header uses donor/acceptor
        names; otherwise ``intron`` is used.  Passing this explicitly avoids
        any ambiguity.

    Returns
    -------
    list of :class:`Junction`
        Junctions in input order.  Use :func:`group_junctions` or
        :func:`explain_junctions` to group them by molecule.
    """
    f, should_close, label = _open_source(source)
    try:
        lines = (
            line for line in f if not line.lstrip().startswith(comment)
        )
        reader = csv.reader(lines, delimiter=delimiter or "\t")
        try:
            header = next(reader)
        except StopIteration:
            return []
        header = [_norm_column(i) for i in header]
        if len(set(header)) != len(header):
            raise ValueError(
                "%s has duplicate column names: %s" % (label, header)
            )

        columns = _resolve_columns(header, ["molecule", "start", "end"],
                                   label)
        index = {name: header.index(col) for name, col in columns.items()}

        if coordinate is None:
            donor_names = {"donor", "donor_end"}
            acceptor_names = {"acceptor", "acceptor_start"}
            if (
                columns["start"] in donor_names
                and columns["end"] in acceptor_names
            ):
                coordinate = DONOR_ACCEPTOR
            else:
                coordinate = INTRON
        coordinate = _norm_coordinate(coordinate)

        def value(row, name, default=None):
            if name not in index:
                return default
            v = row[index[name]]
            return default if _is_blank(v) else v.strip()

        junctions = []
        for line_number, row in enumerate(reader, start=2):
            if not row or all(_is_blank(v) for v in row):
                continue
            width = max(index.values()) + 1
            if len(row) < width:
                raise ValueError(
                    "%s line %s: expected at least %s columns, got %s"
                    % (label, line_number, width, len(row))
                )

            start = _parse_int(row[index["start"]], "start", label,
                               line_number)
            end = _parse_int(row[index["end"]], "end", label, line_number)
            order = value(row, "order")
            if order is not None:
                order = _parse_int(order, "order", label, line_number)
            count = value(row, "count")
            if count is not None:
                count = _parse_int(count, "count", label, line_number)

            junctions.append(
                Junction(
                    molecule=value(row, "molecule"),
                    seqid=value(row, "seqid"),
                    start=start,
                    end=end,
                    strand=value(row, "strand", "."),
                    order=order,
                    id=value(row, "id"),
                    count=count,
                    coordinate=coordinate,
                    index=len(junctions),
                )
            )
        return junctions
    finally:
        if should_close:
            f.close()


def group_junctions(junctions, require_order=False):
    """
    Group junction observations by molecule.

    Within a molecule, junctions are ordered by their explicit ``order``
    column when every junction has one.  Otherwise the input row order is
    preserved -- the junction table already states which observations belong
    together, and re-sorting by genomic coordinate would silently change the
    molecule's stated order (and would be wrong for minus-strand molecules).

    Parameters
    ----------
    junctions : iterable of :class:`Junction`
    require_order : bool
        If True, raise ValueError when a molecule has more than one junction
        and at least one has no explicit order.

    Returns
    -------
    OrderedDict mapping molecule name to a :class:`JunctionGroup`.
    """
    grouped = collections.OrderedDict()
    for j in junctions:
        grouped.setdefault(j.molecule, []).append(j)

    result = collections.OrderedDict()
    for molecule, members in grouped.items():
        if require_order and len(members) > 1:
            missing = [j for j in members if j.order is None]
            if missing:
                raise ValueError(
                    "molecule %s has multiple junctions but no explicit "
                    "order" % molecule
                )
        if len(members) > 1 and all(j.order is not None for j in members):
            members = sorted(members, key=lambda j: (j.order, j.index or 0))
        result[molecule] = JunctionGroup(molecule, members)
    return result


class JunctionGroup(object):
    """All junction observations belonging to one molecule."""

    def __init__(self, molecule, junctions, seqid=None, strand=None):
        self.molecule = molecule
        self.junctions = list(junctions)
        if seqid is None:
            seqids = {j.seqid for j in self.junctions if j.seqid is not None}
            if len(seqids) == 1:
                seqid = next(iter(seqids))
        if strand is None:
            strands = {j.strand for j in self.junctions if j.strand != "."}
            if len(strands) == 1:
                strand = next(iter(strands))
            elif len(strands) > 1:
                strand = "."
        self.seqid = seqid
        self.strand = strand

    def __len__(self):
        return len(self.junctions)

    def __iter__(self):
        return iter(self.junctions)

    def __getitem__(self, item):
        return self.junctions[item]

    def __repr__(self):
        return "<JunctionGroup %s: %s junctions>" % (
            self.molecule, len(self.junctions)
        )


class JunctionEvidence(object):
    """
    Why a single junction did (or did not) match transcripts.

    ``transcripts`` is the set of transcript IDs with a compatible intron for
    this junction alone.  This is intentionally *not* sufficient to explain
    the molecule; see :class:`GroupExplanation`.

    ``matches`` maps transcript ID to the annotated intron that matched.
    ``spanning`` is the subset of transcripts whose junction explicitly spans
    annotated internal exons; this is possible only when
    ``allow_exon_skipping=True`` and is never inferred from a missing
    observation.
    """

    def __init__(self, junction, transcripts, seqid=None, matches=None,
                 spanning=()):
        self.junction = junction
        self.transcripts = frozenset(transcripts)
        self.seqid = seqid
        self.matches = dict(matches or {})
        self.spanning = frozenset(spanning)

    @property
    def supported(self):
        return bool(self.transcripts)

    @property
    def spanning_transcripts(self):
        """Transcripts matched by an explicitly spanning junction."""
        return self.spanning

    def __repr__(self):
        return "<JunctionEvidence %r: %s transcripts>" % (
            self.junction, len(self.transcripts)
        )


class GroupExplanation(object):
    """
    The joint explanation of one molecule.

    Attributes
    ----------
    status : "unique", "shared", or "none"
        - ``unique``: exactly one transcript explains every junction.
        - ``shared``: more than one transcript explains every junction.
        - ``none``: no single transcript explains every junction.  Per-junction
          evidence may still exist; ``candidates`` is then empty and
          ``partial`` records the closest explanations.
    candidates : tuple of str
        Transcript IDs jointly compatible with the whole group.
    partial : tuple of (str, tuple of Junction)
        For ``none`` results, transcripts matching the most junctions and the
        junctions they match.  This makes it explicit that a molecule could
        not be explained as one unit.
    evidence : tuple of :class:`JunctionEvidence`
        Per-junction support.  Never used on its own to declare success.
    """

    def __init__(self, molecule, junctions, candidates, evidence,
                 partial=(), status=None, reason=None):
        self.molecule = molecule
        self.junctions = tuple(junctions)
        self.candidates = tuple(candidates)
        if status is None:
            if not self.candidates:
                status = NONE
            elif len(self.candidates) == 1:
                status = UNIQUE
            else:
                status = SHARED
        self.status = status
        self.evidence = tuple(evidence)
        self.partial = tuple(partial)
        self.reason = reason

    @property
    def transcript(self):
        """The single explaining transcript, or None."""
        if self.status == UNIQUE:
            return self.candidates[0]
        return None

    @property
    def supported(self):
        return self.status != NONE

    @property
    def unique_support(self):
        return self.status == UNIQUE

    @property
    def shared_support(self):
        return self.status == SHARED

    @property
    def no_compatible_path(self):
        return self.status == NONE

    def __repr__(self):
        return "<GroupExplanation %s: %s %s>" % (
            self.molecule, self.status, list(self.candidates)
        )


class JunctionReport(object):
    """Container for all molecule explanations from a call."""

    def __init__(self, groups, db=None):
        self.groups = tuple(groups)
        self.db = db

    def __iter__(self):
        return iter(self.groups)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, item):
        if isinstance(item, str):
            for group in self.groups:
                if group.molecule == item:
                    return group
            raise KeyError(item)
        return self.groups[item]

    def by_status(self, status):
        return tuple(g for g in self.groups if g.status == status)

    @property
    def unique(self):
        return self.by_status(UNIQUE)

    @property
    def shared(self):
        return self.by_status(SHARED)

    @property
    def unresolved(self):
        return self.by_status(NONE)

    @property
    def counts(self):
        c = collections.Counter(g.status for g in self.groups)
        return {
            UNIQUE: c.get(UNIQUE, 0),
            SHARED: c.get(SHARED, 0),
            NONE: c.get(NONE, 0),
        }

    def as_dicts(self):
        """Return a JSON-friendly list of group explanations."""
        out = []
        for g in self.groups:
            out.append(
                {
                    "molecule": g.molecule,
                    "status": g.status,
                    "transcript": g.transcript,
                    "candidates": list(g.candidates),
                    "junctions": [
                        {
                            "seqid": e.junction.seqid,
                            "start": e.junction.start,
                            "end": e.junction.end,
                            "strand": e.junction.strand,
                            "order": e.junction.order,
                            "id": e.junction.id,
                            "intron": list(e.junction.intron),
                            "transcripts": sorted(e.transcripts),
                            "matches": {
                                transcript: list(intron)
                                for transcript, intron in sorted(
                                    e.matches.items()
                                )
                            },
                            "spanning_transcripts": sorted(
                                e.spanning_transcripts
                            ),
                        }
                        for e in g.evidence
                    ],
                    "partial": [
                        {
                            "transcript": transcript,
                            "matched_junctions": [
                                {
                                    "seqid": j.seqid,
                                    "start": j.start,
                                    "end": j.end,
                                    "strand": j.strand,
                                    "order": j.order,
                                    "intron": list(j.intron),
                                }
                                for j in matched
                            ],
                        }
                        for transcript, matched in g.partial
                    ],
                    "reason": g.reason,
                }
            )
        return out


class _Transcript(object):
    """Internal, normalized transcript model."""

    __slots__ = (
        "id", "feature", "seqid", "strand", "exons", "introns", "intron_set",
        "exon_ends", "exon_starts",
    )

    def __init__(self, feature, exons):
        self.id = feature.id
        self.feature = feature
        self.seqid = feature.seqid
        self.strand = feature.strand
        self.exons = sorted((int(e.start), int(e.end)) for e in exons)
        self.introns = []
        self.exon_ends = set()
        self.exon_starts = set()
        for left, right in zip(self.exons, self.exons[1:]):
            self.exon_ends.add(left[1])
            self.exon_starts.add(right[0])
            intron = (left[1] + 1, right[0] - 1)
            if intron[0] <= intron[1]:
                self.introns.append(intron)
        if self.exons:
            self.exon_starts.add(self.exons[0][0])
            self.exon_ends.add(self.exons[-1][1])
        self.intron_set = frozenset(self.introns)


def _transcript_models(db, transcripts=None, transcript_featuretype=None,
                       exon_featuretype="exon"):
    """
    Build transcript models from exons.

    When no explicit transcript set is given, transcript-like features are
    discovered as level-1 parents of exons rather than assuming a featuretype.
    """
    if transcript_featuretype is not None:
        transcripts = db.features_of_type(transcript_featuretype)

    if transcripts is not None:
        models = []
        seen = set()
        for t in transcripts:
            if isinstance(t, (str, bytes)):
                t = db[t]
            elif not hasattr(t, "id"):
                t = db[t]
            if t.id in seen:
                continue
            seen.add(t.id)
            exons = list(
                db.children(t, level=1, featuretype=exon_featuretype)
            )
            if exons:
                models.append(_Transcript(t, exons))
        return models

    exon_parents = collections.OrderedDict()
    for exon in db.features_of_type(exon_featuretype):
        for parent in db.parents(exon, level=1):
            exon_parents.setdefault(parent.id, (parent, []))[1].append(exon)
    return [
        _Transcript(feature, exons)
        for feature, exons in exon_parents.values()
    ]


def _strand_compatible(junction_strand, transcript_strand):
    """
    An unknown observation strand imposes no constraint.  A known observation
    strand must agree with the annotation; an unstranded annotation cannot
    support a strand-specific observation.
    """
    if junction_strand == ".":
        return True
    return transcript_strand == junction_strand


def _find_match(model, junction, allow_exon_skipping):
    """
    Return ``(intron, spanning)`` for a match, or None.

    Exact matching requires the observed intron to equal an annotated intron
    (``spanning`` is False).  With ``allow_exon_skipping=True``, an observed
    intron whose donor and acceptor are exon boundaries in the transcript is
    also accepted (``spanning`` is True); this is an explicit statement by the
    observation that the intervening annotated exon(s) were skipped.  A mere
    absence of observations is never used to invoke this rule.
    """
    observed = junction.intron
    if observed in model.intron_set:
        return observed, False

    if allow_exon_skipping:
        donor, acceptor = observed[0] - 1, observed[1] + 1
        if donor in model.exon_ends and acceptor in model.exon_starts:
            # Require at least one annotated intron between these boundaries,
            # so this is genuinely a spanning (exon-skipping) observation.
            between = [
                intron for intron in model.introns
                if intron[0] > donor and intron[1] < acceptor
            ]
            if between:
                return observed, True
    return None


def _match_order(model, matched_junctions):
    """
    Check that matched introns can appear in the stated biological order.

    ``matched_junctions`` is a list of ``(junction, intron, spanning)`` in
    observation order.  For a plus-strand transcript, intron starts must not
    decrease; for a minus-strand transcript, intron ends must not increase.
    This detects junctions that are individually present but cannot belong to
    one molecule/transcript in the observed order.
    """
    introns = [intron for _, intron, _ in matched_junctions]
    if len(introns) < 2:
        return True
    if model.strand == "+":
        positions = [i[0] for i in introns]
        return all(a <= b for a, b in zip(positions, positions[1:]))
    if model.strand == "-":
        positions = [i[1] for i in introns]
        return all(a >= b for a, b in zip(positions, positions[1:]))
    # Unknown transcript strand: allow either consistent genomic direction.
    starts = [i[0] for i in introns]
    ends = [i[1] for i in introns]
    return (
        all(a <= b for a, b in zip(starts, starts[1:]))
        or all(a >= b for a, b in zip(ends, ends[1:]))
    )


def explain_junctions(db, junctions, transcripts=None,
                      transcript_featuretype=None, exon_featuretype="exon",
                      allow_exon_skipping=False, require_seqid=True,
                      require_order=False):
    """
    Find transcript(s) that jointly explain each molecule's junctions.

    This is the main entry point.  Junctions are grouped by ``molecule`` and
    each group is evaluated as a unit: a transcript must be compatible with
    every junction in the group.  Matching individual junctions to different
    transcripts is recorded only as evidence and never reported as a group
    explanation.

    A junction is compatible with a transcript when:

    - the junction strand is unknown or equals the transcript strand;
    - the junction lies on the same ``seqid`` (if supplied and
      ``require_seqid`` is True);
    - the observed intron exactly equals an annotated intron, or, with
      ``allow_exon_skipping=True``, its donor and acceptor are exon boundaries
      spanning at least one annotated intron;
    - all matched introns occur in a consistent biological order (see
      ``order``).

    Missing observations are never interpreted as confirmed exon skipping.
    If a junction between two observed junctions is absent from the table, the
    transcript can still explain the molecule (the molecule may simply not
    cover it), but no claim is made that an exon was skipped.

    Parameters
    ----------
    db : FeatureDB
    junctions : iterable of :class:`Junction` or :class:`JunctionGroup`
        If groups are provided, their order and membership are preserved.
    transcripts : iterable of transcript IDs/Features or None
        Transcripts to test.  If None, all level-1 exon parents are used.
        Providing a smaller set narrows candidates without changing the joint
        rule.
    transcript_featuretype : str or None
        If provided, only features of this type are tested (e.g. ``"mRNA"``).
    exon_featuretype : str
        Feature type used as exons (default ``"exon"``).
    allow_exon_skipping : bool
        If True, an observed junction may span annotated internal exons when
        its donor and acceptor match exon boundaries.  This is an explicit
        rule applied to an observed junction, not an inference from missing
        data.
    require_seqid : bool
        If True (default), reject matches across different ``seqid`` values
        when the junction table supplies a seqid.
    require_order : bool
        Forwarded to :func:`group_junctions`; require an explicit order for
        multi-junction molecules.

    Returns
    -------
    :class:`JunctionReport`
    """
    if isinstance(junctions, JunctionReport):
        groups = list(junctions.groups)
    elif isinstance(junctions, JunctionGroup):
        groups = [junctions]
    else:
        groups = list(junctions)
        if groups and isinstance(groups[0], JunctionGroup):
            pass
        else:
            groups = list(
                group_junctions(groups, require_order=require_order).values()
            )

    if not groups:
        return JunctionReport((), db=db)

    models = _transcript_models(
        db,
        transcripts=transcripts,
        transcript_featuretype=transcript_featuretype,
        exon_featuretype=exon_featuretype,
    )

    explanations = [
        _explain_group(
            group,
            models,
            allow_exon_skipping=allow_exon_skipping,
            require_seqid=require_seqid,
        )
        for group in groups
    ]
    return JunctionReport(explanations, db=db)


def _explain_group(group, models, allow_exon_skipping=False,
                   require_seqid=True):
    junctions = list(group.junctions)
    evidence = []
    supporting = None
    matched_by_transcript = collections.OrderedDict()

    # The group itself must describe one molecule on one strand and one
    # sequence.  Conflicting fields cannot be rescued by per-junction matches.
    observed_strands = {j.strand for j in junctions if j.strand != "."}
    if len(observed_strands) > 1:
        for junction in junctions:
            evidence.append(
                JunctionEvidence(junction, (), seqid=junction.seqid)
            )
        return GroupExplanation(
            group.molecule,
            junctions,
            candidates=(),
            evidence=evidence,
            status=NONE,
            reason="conflicting strands within molecule",
        )
    observed_seqids = {
        j.seqid for j in junctions if j.seqid is not None
    }
    if require_seqid and len(observed_seqids) > 1:
        for junction in junctions:
            evidence.append(
                JunctionEvidence(junction, (), seqid=junction.seqid)
            )
        return GroupExplanation(
            group.molecule,
            junctions,
            candidates=(),
            evidence=evidence,
            status=NONE,
            reason="conflicting seqids within molecule",
        )

    for junction in junctions:
        matched = []
        matches = {}
        spanning = set()
        for model in models:
            if not _strand_compatible(junction.strand, model.strand):
                continue
            if (
                require_seqid
                and junction.seqid is not None
                and model.seqid is not None
                and junction.seqid != model.seqid
            ):
                continue
            match = _find_match(model, junction, allow_exon_skipping)
            if match is None:
                continue
            intron, is_spanning = match
            matched.append(model.id)
            matches[model.id] = intron
            if is_spanning:
                spanning.add(model.id)
            matched_by_transcript.setdefault(model.id, []).append(
                (junction, intron, is_spanning)
            )
        evidence.append(
            JunctionEvidence(
                junction, matched, seqid=junction.seqid, matches=matches,
                spanning=spanning,
            )
        )

        ids = frozenset(matched)
        supporting = ids if supporting is None else supporting & ids

    if supporting is None or not supporting:
        partial = []
        if matched_by_transcript:
            ranked = sorted(
                matched_by_transcript.items(),
                key=lambda item: (-len(item[1]), item[0]),
            )
            best = len(ranked[0][1])
            for transcript, matched in ranked:
                if len(matched) != best:
                    break
                partial.append(
                    (transcript, tuple(item[0] for item in matched))
                )
        reason = "no single transcript matches all junctions"
        if all(not e.transcripts for e in evidence):
            strand_blocked = False
            for j in junctions:
                eligible = [
                    model for model in models
                    if not (
                        require_seqid
                        and j.seqid is not None
                        and model.seqid is not None
                        and j.seqid != model.seqid
                    )
                ]
                if eligible and all(
                    j.strand != "." and model.strand != j.strand
                    for model in eligible
                ):
                    strand_blocked = True
                    break
            if strand_blocked:
                reason = "strand mismatch between junction and transcript"
            else:
                reason = "no transcript matches any junction"
        return GroupExplanation(
            group.molecule,
            junctions,
            candidates=(),
            evidence=evidence,
            partial=partial,
            status=NONE,
            reason=reason,
        )

    # Apply the joint order rule per transcript using its full set of matched
    # introns.  Per-junction matches must never be recombined across
    # transcripts.
    candidate_ids = []
    order_conflicts = []
    by_id = {model.id: model for model in models}
    for transcript_id in supporting:
        model = by_id[transcript_id]
        matched = matched_by_transcript[transcript_id]
        if len(matched) != len(junctions):
            continue
        if not _match_order(model, matched):
            order_conflicts.append(transcript_id)
            continue
        candidate_ids.append(transcript_id)

    if not candidate_ids:
        reason = "matching junctions are not in a consistent biological order"
        return GroupExplanation(
            group.molecule,
            junctions,
            candidates=(),
            evidence=evidence,
            status=NONE,
            reason=reason,
        )

    candidates = tuple(sorted(set(candidate_ids)))
    return GroupExplanation(
        group.molecule,
        junctions,
        candidates=candidates,
        evidence=evidence,
        status=UNIQUE if len(candidates) == 1 else SHARED,
    )


def resolve_junction_groups(db, junctions, **kwargs):
    """Alias for :func:`explain_junctions`."""
    return explain_junctions(db, junctions, **kwargs)


def format_report(report, show_evidence=False):
    """
    Format a :class:`JunctionReport` as tab-delimited text.

    Columns: molecule, status, transcript, candidate_count, detail.  With
    ``show_evidence=True``, one extra row per junction has status
    ``"evidence"`` and its per-junction supporting transcripts in the
    ``detail`` column.
    """
    header = ["molecule", "status", "transcript", "candidate_count", "detail"]
    rows = ["\t".join(header)]
    for group in report:
        if group.status == UNIQUE:
            detail = ""
        elif group.status == SHARED:
            detail = ",".join(group.candidates)
        else:
            detail = group.reason or ""
        rows.append(
            "\t".join(
                [
                    group.molecule,
                    group.status,
                    group.transcript or "",
                    str(len(group.candidates)),
                    detail,
                ]
            )
        )
        if show_evidence:
            for ev in group.evidence:
                rows.append(
                    "\t".join(
                        [
                            group.molecule,
                            "evidence",
                            "",
                            str(len(ev.transcripts)),
                            "%s:%s-%s:%s;%s"
                            % (
                                ev.junction.seqid or ".",
                                ev.junction.start,
                                ev.junction.end,
                                ev.junction.strand,
                                ",".join(sorted(ev.transcripts)),
                            ),
                        ]
                    )
                )
    return "\n".join(rows)
