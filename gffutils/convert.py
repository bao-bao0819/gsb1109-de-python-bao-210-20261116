"""
Conversion functions that operate on :class:`FeatureDB` classes.

The GFF family uses 1-based, closed intervals.  BED12 uses 0-based,
half-open intervals plus a block list.  This module keeps that conversion
explicit in both directions so that a round trip restores the genomic
positions of every block, not merely the total number of bases.
"""

import copy
from dataclasses import dataclass, field
from gffutils.feature import Feature


class Bed12Error(ValueError):
    """Base class for BED12 conversion errors."""


class MalformedBed12Error(Bed12Error):
    """Raised when a BED12 line cannot be parsed or is internally inconsistent."""


class UnrepresentableFeatureError(Bed12Error):
    """Raised when a GFF feature has no lossless BED12 representation."""

    def __init__(self, feature, reason):
        self.feature = feature
        self.reason = reason
        Bed12Error.__init__(self, "%s: %s" % (reason, _feature_label(feature)))


def _feature_label(feature):
    if feature is None:
        return "<unknown>"
    if isinstance(feature, str):
        return feature
    return "%s (%s:%s-%s[%s])" % (
        getattr(feature, "id", None) or getattr(feature, "featuretype", "?"),
        getattr(feature, "seqid", "?"),
        getattr(feature, "start", "?"),
        getattr(feature, "stop", "?"),
        getattr(feature, "strand", "?"),
    )


def _attr_first(feature, field, default="."):
    value = feature.attributes.get(field)
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        return value[0]
    return value


def _score_field(score):
    if score is None:
        return "0"
    score = str(score).strip()
    if score in ("", ".", "None"):
        return "0"
    try:
        float(score)
    except ValueError:
        return "0"
    return score


def _color_field(color):
    if color is None:
        return "0,0,0"
    color = str(color).replace(" ", "").strip()
    parts = color.split(",")
    if len(parts) != 3:
        raise ValueError("color must be a comma-separated R,G,B triplet")
    try:
        values = [int(i) for i in parts]
    except ValueError:
        raise ValueError("color must be a comma-separated R,G,B triplet")
    if any(i < 0 or i > 255 for i in values):
        raise ValueError("color values must be between 0 and 255")
    return ",".join(map(str, values))


def _int_coord(value, label, feature):
    if value is None or value == "." or value == "":
        raise UnrepresentableFeatureError(feature, "missing %s coordinate" % label)
    try:
        return int(value)
    except (TypeError, ValueError):
        raise UnrepresentableFeatureError(
            feature, "non-integer %s coordinate %r" % (label, value)
        )


def _validate_span_values(start, end, feature, label):
    if start < 1:
        raise UnrepresentableFeatureError(
            feature,
            "%s starts at %s, which is not a valid 1-based GFF "
            "coordinate" % (label, start),
        )
    if start > end:
        if start == end + 1:
            raise UnrepresentableFeatureError(
                feature,
                "zero-length point event (%s start %s == end %s + 1) has no "
                "BED12 block representation; refusing to swap endpoints"
                % (label, start, end),
            )
        raise UnrepresentableFeatureError(
            feature,
            "reversed coordinates (%s start %s > end %s); refusing to swap "
            "endpoints" % (label, start, end),
        )


def _validate_feature_span(feature):
    start = _int_coord(feature.start, "start", feature)
    end = _int_coord(feature.stop, "end", feature)
    _validate_span_values(start, end, feature, "feature")
    if not feature.seqid or feature.seqid == ".":
        raise UnrepresentableFeatureError(feature, "missing seqid/chrom")
    return start, end


def _validate_child(feature, child):
    start = _int_coord(child.start, "start", child)
    end = _int_coord(child.stop, "end", child)
    _validate_span_values(start, end, child, "block")
    if child.seqid != feature.seqid:
        raise UnrepresentableFeatureError(
            child,
            "child seqid %r does not match parent seqid %r"
            % (
                child.seqid,
                feature.seqid,
            ),
        )
    if (
        feature.strand in ("+", "-")
        and child.strand in ("+", "-")
        and child.strand != feature.strand
    ):
        raise UnrepresentableFeatureError(
            child,
            "child strand %r does not match parent strand %r"
            % (
                child.strand,
                feature.strand,
            ),
        )
    if start < feature.start or end > feature.stop:
        raise UnrepresentableFeatureError(
            child,
            "block lies outside parent bounds %s-%s"
            % (
                feature.start,
                feature.stop,
            ),
        )
    return start, end


def _ordered_blocks(feature, db, block_featuretype):
    if db is None:
        children = []
    else:
        children = list(
            db.children(feature, featuretype=block_featuretype, order_by="start")
        )

    if not children:
        children = [feature]

    blocks = []
    for child in children:
        start, end = _validate_child(feature, child)
        blocks.append((start, end, child))

    blocks.sort(key=lambda item: (item[0], item[1]))

    previous_end = None
    for start, end, child in blocks:
        if previous_end is not None and start <= previous_end:
            raise UnrepresentableFeatureError(
                child,
                "overlapping or duplicate blocks (block starts at %s, previous "
                "block ends at %s)" % (start, previous_end),
            )
        previous_end = end

    first_start = blocks[0][0]
    last_end = blocks[-1][1]
    if first_start != feature.start:
        raise UnrepresentableFeatureError(
            feature,
            "start of first block (%s) does not match start of feature (%s)"
            % (first_start, feature.start),
        )
    if last_end != feature.stop:
        raise UnrepresentableFeatureError(
            feature,
            "end of last block (%s) does not match end of feature (%s)"
            % (last_end, feature.stop),
        )
    return blocks


def _thick_bounds(feature, blocks, db, thick_featuretype, thin_featuretype):
    chrom_start = feature.start - 1
    chrom_end = feature.stop

    if thick_featuretype and thin_featuretype:
        raise ValueError(
            "Can only specify one of `thick_featuretype` or `thin_featuretype`"
        )

    if thick_featuretype:
        thick = []
        if db is not None:
            thick = list(
                db.children(feature, featuretype=thick_featuretype, order_by="start")
            )
        if not thick:
            return chrom_start, chrom_end
        starts = []
        ends = []
        for child in thick:
            start, end = _validate_child(feature, child)
            starts.append(start)
            ends.append(end)
        return min(starts) - 1, max(ends)

    if thin_featuretype:
        thin = []
        if db is not None:
            thin = list(
                db.children(feature, featuretype=thin_featuretype, order_by="start")
            )
        if not thin:
            return chrom_start, chrom_end
        intervals = []
        for child in thin:
            start, end = _validate_child(feature, child)
            intervals.append((start, end))
        intervals.sort()
        return _coding_bounds_from_thin(feature, intervals)

    return chrom_start, chrom_end


def _coding_bounds_from_thin(feature, intervals):
    """
    Infer the coding span from UTR (thin) intervals.

    The original BED12 conversion used the end of the first UTR and the start
    of the last UTR (in genomic order); with the usual two UTR groups this is
    the coding gap between them, and it is strand-independent.  A single UTR
    does not define both coding boundaries, so it is reported as
    unrepresentable rather than guessed.
    """
    if len(intervals) < 2:
        raise UnrepresentableFeatureError(
            feature,
            "a single UTR interval does not define both thickStart and "
            "thickEnd; provide thick_featuretype or both UTR groups",
        )
    return intervals[0][1], intervals[-1][0] - 1


@dataclass(frozen=True)
class Bed12Record:
    """A parsed BED12 record.

    ``chrom_start``, ``chrom_end``, block starts and block ends are stored in
    BED's 0-based, half-open coordinate system.  Use :attr:`gff_blocks` or
    :meth:`to_features` to obtain 1-based, closed GFF coordinates.

    ``source_blocks`` holds the GFF child features in the same order as the
    BED12 blocks.  It is populated by :func:`to_bed12_record` and is empty for
    records parsed from text, where the source objects are not available.
    """

    chrom: str
    chrom_start: int
    chrom_end: int
    name: str
    score: str
    strand: str
    thick_start: int
    thick_end: int
    item_rgb: str
    block_sizes: tuple
    block_starts: tuple
    extra: tuple = field(default_factory=tuple)
    source_blocks: tuple = field(default_factory=tuple, compare=False, repr=False)
    source_parent: object = field(default=None, compare=False, repr=False)

    @classmethod
    def parse(cls, line):
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        line = str(line).rstrip("\r\n")
        fields = line.split("\t")
        if len(fields) < 12:
            fields = line.split()
        if len(fields) < 12:
            raise MalformedBed12Error(
                "BED12 requires at least 12 tab-delimited fields; got %s" % len(fields)
            )

        chrom = fields[0]
        if not chrom or chrom == ".":
            raise MalformedBed12Error("BED12 chrom must not be empty")

        try:
            chrom_start = int(fields[1])
            chrom_end = int(fields[2])
        except ValueError:
            raise MalformedBed12Error("BED12 chromStart/chromEnd must be integers")
        if chrom_start < 0:
            raise MalformedBed12Error("BED12 chromStart must be >= 0")
        if chrom_end <= chrom_start:
            raise MalformedBed12Error(
                "BED12 chromEnd (%s) must be greater than chromStart (%s)"
                % (chrom_end, chrom_start)
            )

        name = fields[3]
        score = fields[4]
        strand = fields[5]
        if strand not in ("+", "-", "."):
            raise MalformedBed12Error("BED12 strand must be '+', '-' or '.'")

        try:
            thick_start = int(fields[6])
            thick_end = int(fields[7])
        except ValueError:
            raise MalformedBed12Error("BED12 thickStart/thickEnd must be integers")

        item_rgb = fields[8]

        try:
            block_count = int(fields[9])
        except ValueError:
            raise MalformedBed12Error("BED12 blockCount must be an integer")
        if block_count < 1:
            raise MalformedBed12Error("BED12 blockCount must be at least 1")

        block_sizes = _parse_block_list(fields[10], "blockSizes")
        block_starts = _parse_block_list(fields[11], "blockStarts")
        if len(block_sizes) != block_count:
            raise MalformedBed12Error(
                "blockCount %s does not match %s blockSizes"
                % (block_count, len(block_sizes))
            )
        if len(block_starts) != block_count:
            raise MalformedBed12Error(
                "blockCount %s does not match %s blockStarts"
                % (block_count, len(block_starts))
            )

        span = chrom_end - chrom_start
        previous_end = None
        for i, (size, start) in enumerate(zip(block_sizes, block_starts)):
            if size < 1:
                raise MalformedBed12Error(
                    "block %s has non-positive size %s" % (i, size)
                )
            if start < 0:
                raise MalformedBed12Error("block %s has negative start %s" % (i, start))
            if start + size > span:
                raise MalformedBed12Error(
                    "block %s (start %s, size %s) extends past chromEnd"
                    % (i, start, size)
                )
            if previous_end is not None and start < previous_end:
                raise MalformedBed12Error(
                    "block %s starts at %s, before previous block ends at %s"
                    % (i, start, previous_end)
                )
            previous_end = start + size

        if block_starts[0] != 0:
            raise MalformedBed12Error(
                "first block must start at chromStart (blockStart is %s)"
                % block_starts[0]
            )
        if block_starts[-1] + block_sizes[-1] != span:
            raise MalformedBed12Error(
                "last block must end at chromEnd (ends at %s, span is %s)"
                % (block_starts[-1] + block_sizes[-1], span)
            )

        if not (chrom_start <= thick_start <= thick_end <= chrom_end):
            raise MalformedBed12Error(
                "thickStart/thickEnd must lie within chromStart/chromEnd"
            )

        return cls(
            chrom=chrom,
            chrom_start=chrom_start,
            chrom_end=chrom_end,
            name=name,
            score=score,
            strand=strand,
            thick_start=thick_start,
            thick_end=thick_end,
            item_rgb=item_rgb,
            block_sizes=tuple(block_sizes),
            block_starts=tuple(block_starts),
            extra=tuple(fields[12:]),
        )

    @property
    def blocks(self):
        return self.gff_blocks

    @property
    def block_count(self):
        return len(self.block_sizes)

    @property
    def span(self):
        return self.chrom_end - self.chrom_start

    def block_at(self, index):
        """Return the 1-based closed ``(start, end)`` of block ``index``."""
        start, end = self.gff_blocks[index]
        return start, end

    def index_of(self, start, end):
        """Return the block index matching a 1-based closed interval, or None."""
        for i, block in enumerate(self.gff_blocks):
            if block == (start, end):
                return i
        return None

    def block_records(self):
        """Yield ``(index, start, end, size, offset)`` for each block.

        ``offset`` is the 0-based BED offset from chromStart.
        """
        for i, (size, offset) in enumerate(zip(self.block_sizes, self.block_starts)):
            yield (
                i,
                self.chrom_start + offset + 1,
                self.chrom_start + offset + size,
                size,
                offset,
            )

    @property
    def bed_blocks(self):
        return tuple(
            (self.chrom_start + start, self.chrom_start + start + size)
            for size, start in zip(self.block_sizes, self.block_starts)
        )

    @property
    def gff_blocks(self):
        return tuple((start + 1, end) for start, end in self.bed_blocks)

    def coverage(self):
        """Return the ordered, 1-based closed blocks represented by the record."""
        return self.gff_blocks

    def to_bed12(self):
        return "\t".join(
            map(
                str,
                [
                    self.chrom,
                    self.chrom_start,
                    self.chrom_end,
                    self.name,
                    self.score,
                    self.strand,
                    self.thick_start,
                    self.thick_end,
                    self.item_rgb,
                    self.block_count,
                    ",".join(map(str, self.block_sizes)),
                    ",".join(map(str, self.block_starts)),
                ]
                + list(self.extra),
            )
        )

    def to_features(
        self,
        source=".",
        parent_featuretype="mRNA",
        block_featuretype="exon",
        name_field="ID",
        parent_attributes=None,
        block_attributes=None,
    ):
        """Return ``(parent, blocks)`` as 1-based, closed GFF Features.

        Block order is the order stored in the BED12 record; blocks are never
        sorted or merged here.
        """
        attributes = {}
        if self.name not in ("", "."):
            attributes[name_field] = [self.name]
        if parent_attributes:
            attributes.update(parent_attributes)

        parent = Feature(
            seqid=self.chrom,
            source=source,
            featuretype=parent_featuretype,
            start=self.chrom_start + 1,
            end=self.chrom_end,
            score=self.score,
            strand=self.strand,
            frame=".",
            attributes=attributes,
        )

        parent_id = _attr_first(parent, name_field, default=".")
        children = []
        for i, (start, end) in enumerate(self.gff_blocks):
            source_block = (
                self.source_blocks[i] if i < len(self.source_blocks) else None
            )
            child_attrs = {"Parent": [parent_id]} if parent_id != "." else {}
            if block_attributes:
                child_attrs.update(block_attributes)
            child = Feature(
                seqid=self.chrom,
                source=source_block.source if source_block is not None else source,
                featuretype=(
                    source_block.featuretype
                    if source_block is not None
                    else block_featuretype
                ),
                start=start,
                end=end,
                score=self.score,
                strand=self.strand,
                frame=".",
                attributes=child_attrs,
            )
            if source_block is not None and not block_attributes:
                child.attributes = copy.deepcopy(source_block.attributes)
                if parent_id != ".":
                    child.attributes.setdefault("Parent", [parent_id])
            children.append(child)
        return parent, children


def _parse_block_list(value, label):
    value = str(value).strip()
    if value.endswith(","):
        value = value[:-1]
    if not value:
        raise MalformedBed12Error("%s must not be empty" % label)
    parts = value.split(",")
    if any(part.strip() == "" for part in parts):
        raise MalformedBed12Error("%s contains an empty value" % label)
    try:
        return [int(part) for part in parts]
    except ValueError:
        raise MalformedBed12Error("%s must contain integers" % label)


def to_bed12_record(
    f,
    db=None,
    child_type="exon",
    name_field="ID",
    color=None,
    thick_featuretype=None,
    thin_featuretype=None,
    block_featuretype=None,
):
    """Build a :class:`Bed12Record` from a GFF feature and its block children.

    ``child_type`` is retained for compatibility with the original
    :func:`to_bed12` API.  ``block_featuretype`` is an alias for it.

    GFF3 represents zero-length point events as ``start == end + 1``; those
    are rejected.  ``start == end`` is a genuine one-base segment and remains
    a one-base block.
    """
    if block_featuretype is not None:
        child_type = block_featuretype
    if isinstance(f, str):
        if db is None:
            raise ValueError("a FeatureDB is required when f is a string")
        f = db[f]

    start, end = _validate_feature_span(f)
    blocks = _ordered_blocks(f, db, child_type)

    chrom_start = start - 1
    chrom_end = end
    block_sizes = [block_end - block_start + 1 for block_start, block_end, _ in blocks]
    block_starts = [block_start - start for block_start, _, _ in blocks]

    for offset, size in zip(block_starts, block_sizes):
        if offset + size > chrom_end - chrom_start:
            raise UnrepresentableFeatureError(
                f, "block extends beyond the transcript span"
            )

    thick_start, thick_end = _thick_bounds(
        f, blocks, db, thick_featuretype, thin_featuretype
    )

    if thick_start > thick_end:
        raise UnrepresentableFeatureError(
            f, "thickStart %s is greater than thickEnd %s" % (thick_start, thick_end)
        )
    if thick_start < chrom_start or thick_end > chrom_end:
        raise UnrepresentableFeatureError(
            f, "thick bounds fall outside the BED12 record"
        )

    return Bed12Record(
        chrom=f.seqid,
        chrom_start=chrom_start,
        chrom_end=chrom_end,
        name=_attr_first(f, name_field, default="."),
        score=_score_field(f.score),
        strand=f.strand if f.strand in ("+", "-", ".") else ".",
        thick_start=thick_start,
        thick_end=thick_end,
        item_rgb=_color_field(color),
        block_sizes=tuple(block_sizes),
        block_starts=tuple(block_starts),
        source_blocks=tuple(child for _, _, child in blocks),
        source_parent=f,
    )


def to_bed12(
    f,
    db=None,
    child_type="exon",
    name_field="ID",
    color=None,
    thick_featuretype=None,
    thin_featuretype=None,
    block_featuretype=None,
    include_newline=True,
):
    """
    Given a top-level feature (e.g., transcript), construct a BED12 entry.

    Parameters
    ----------
    f : Feature object or string
        This is the top-level feature represented by one BED12 line.  For
        a canonical GFF or GTF, this will generally be a transcript.
    db : a FeatureDB object
        This is needed to get the children for the feature.  If no children
        of the block type are found, the feature itself is used as one block.
    child_type : str or list
        Featuretypes that will be represented by the BED12 "blocks".  Typically
        "exon".
    name_field : str
        Attribute to be used in the "name" field of the BED12 entry.  Usually
        "ID" for GFF; "transcript_id" for GTF.
    color : None or str
        RGB triplet for the itemRgb field; default is "0,0,0".
    thick_featuretype, thin_featuretype : str, list or None
        Optional child featuretypes used to set thickStart/thickEnd.
    block_featuretype : str or list or None
        Alias for `child_type`.
    include_newline : bool
        If True (default), append a trailing newline, matching the original
        convert.to_bed12 behavior.

    Raises
    ------
    UnrepresentableFeatureError
        If the feature cannot be represented without changing its coordinates
        or logical block coverage.  Reversed coordinates are never normalized
        by swapping endpoints.
    """
    record = to_bed12_record(
        f,
        db=db,
        child_type=child_type,
        name_field=name_field,
        color=color,
        thick_featuretype=thick_featuretype,
        thin_featuretype=thin_featuretype,
        block_featuretype=block_featuretype,
    )
    line = record.to_bed12()
    if include_newline:
        line += "\n"
    return line


def parse_bed12(line):
    """Parse one BED12 line into a :class:`Bed12Record`."""
    return Bed12Record.parse(line)


def bed12_to_features(
    line,
    source=".",
    parent_featuretype="mRNA",
    block_featuretype="exon",
    name_field="ID",
    parent_attributes=None,
    block_attributes=None,
):
    """Convert a BED12 line to ``(parent, blocks)`` GFF Features."""
    record = Bed12Record.parse(line)
    return record.to_features(
        source=source,
        parent_featuretype=parent_featuretype,
        block_featuretype=block_featuretype,
        name_field=name_field,
        parent_attributes=parent_attributes,
        block_attributes=block_attributes,
    )


def from_bed12(line, **kwargs):
    """Convert a BED12 line to GFF Features (parent first).

    This is the inverse of :func:`to_bed12`; use :func:`parse_bed12` when the
    structured record is wanted instead.
    """
    return bed12_to_gff(line, **kwargs)


def bed12_to_gff(line, **kwargs):
    """Convert a BED12 line to a list of GFF Features (parent first)."""
    parent, children = bed12_to_features(line, **kwargs)
    return [parent] + children


@dataclass(frozen=True)
class Unrepresentable:
    """A top-level object that could not be represented as BED12.

    ``feature`` is the object supplied to :func:`gff_to_bed12`.  ``cause`` is
    the specific child feature responsible for the failure when the problem is
    inside a block, which makes per-block diagnostics possible.
    """

    feature: object
    reason: str
    index: int = None
    cause: object = None

    def __iter__(self):
        yield self.feature
        yield self.reason

    def __len__(self):
        return 2

    def __getitem__(self, index):
        if index == 0:
            return self.feature
        if index == 1:
            return self.reason
        raise IndexError(index)

    def __str__(self):
        if self.cause is not None and self.cause is not self.feature:
            return "%s: %s [cause: %s]" % (
                _feature_label(self.feature),
                self.reason,
                _feature_label(self.cause),
            )
        return "%s: %s" % (_feature_label(self.feature), self.reason)


@dataclass
class Bed12Conversion:
    """Result of converting many GFF features to BED12.

    ``records`` and ``lines`` contain only representable features.  Features
    that BED12 cannot express losslessly are kept in ``unrepresentable`` so
    they are never silently dropped or rewritten.
    """

    records: list = field(default_factory=list)
    unrepresentable: list = field(default_factory=list)
    newline: bool = True

    @property
    def lines(self):
        suffix = "\n" if self.newline else ""
        return [record.to_bed12() + suffix for record in self.records]

    @property
    def skipped(self):
        """Alias for :attr:`unrepresentable`."""
        return self.unrepresentable

    def __iter__(self):
        return iter((self.lines, self.unrepresentable))

    def __len__(self):
        return len(self.records)


def gff_to_bed12(
    features,
    db=None,
    child_type="exon",
    name_field="ID",
    color=None,
    thick_featuretype=None,
    thin_featuretype=None,
    block_featuretype=None,
    include_newline=True,
):
    """Convert an iterable of GFF features, separating unrepresentable ones.

    Returns
    -------
    Bed12Conversion
        Unpacks as ``(lines, unrepresentable)``.  ``lines`` and ``records``
        contain successful conversions; ``unrepresentable`` contains
        :class:`Unrepresentable` entries with the original feature and reason.
    """
    if isinstance(features, (str, Feature)):
        features = [features]

    result = Bed12Conversion(newline=include_newline)
    for i, feature in enumerate(features):
        original = feature
        if isinstance(feature, str) and db is not None:
            original = db[feature]
        try:
            record = to_bed12_record(
                feature,
                db=db,
                child_type=child_type,
                name_field=name_field,
                color=color,
                thick_featuretype=thick_featuretype,
                thin_featuretype=thin_featuretype,
                block_featuretype=block_featuretype,
            )
        except UnrepresentableFeatureError as err:
            result.unrepresentable.append(
                Unrepresentable(
                    feature=original,
                    reason=err.reason,
                    index=i,
                    cause=err.feature,
                )
            )
        else:
            result.records.append(record)
    return result
