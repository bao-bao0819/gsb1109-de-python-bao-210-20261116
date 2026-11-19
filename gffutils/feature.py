from pyfaidx import Fasta
import os
import simplejson as json
from gffutils import constants
from gffutils import helpers
from gffutils import parser
from gffutils import bins
from gffutils import circular
from gffutils.attributes import dict_class
from gffutils.exceptions import MissingSequenceLengthError


_position_lookup = dict(
    enumerate(
        [
            "seqid",
            "source",
            "featuretype",
            "start",
            "end",
            "score",
            "strand",
            "frame",
            "attributes",
        ]
    )
)


class Feature(object):
    def __init__(
        self,
        seqid=".",
        source=".",
        featuretype=".",
        start=".",
        end=".",
        score=".",
        strand=".",
        frame=".",
        attributes=None,
        extra=None,
        bin=None,
        id=None,
        dialect=None,
        file_order=None,
        keep_order=False,
        sort_attribute_values=False,
        sequence_length=None,
    ):
        """
        Represents a feature from the database.

        Usually you won't want to use this directly, since it has various
        implementation details needed for operating in the context of FeatureDB
        objects.  Instead, try the :func:`gffutils.feature.feature_from_line`
        function.

        When printed, reproduces the original line from the file as faithfully
        as possible using `dialect`.


        Parameters
        ----------

        seqid : string
            Name of the sequence (often chromosome)

        source : string
            Source of the feature; typically the originating database or
            program that predicted the feature

        featuretype : string
            Type of feature.  For example "gene", "exon", "TSS", etc

        start, end : int or "."
            1-based coordinates; start must be <= end.  If "." (the default
            placeholder for GFF files), then the corresponding attribute will
            be None.

        score : string
            Stored as a string.

        strand : "+" | "-" | "."
            Strand of the feature; "." when strand is not relevant.

        frame : "0" | "1" | "2"
            Coding frame.  0 means in-frame; 1 means there is one extra base at
            the beginning, so the first codon starts at the second base;
            2 means two extra bases at the beginning.  Interpretation is strand
            specific; "beginning" for a minus-strand feature is at the end
            coordinate.

        attributes : string or dict
            If a string, first assume it is serialized JSON; if this fails then
            assume it's the original key/vals string.  If it's a dictionary
            already, then use as-is.

            The end result is that this instance's `attributes` attribute will
            always be a dictionary.

            Upon printing, the attributes will be reconstructed based on this
            dictionary and the dialect -- except if the original attributes
            string was provided, in which case that will be used directly.

            Notes on encoding/decoding: the only time unquoting
            (e.g., "%2C" becomes ",") happens is if `attributes` is a string
            and if `settings.ignore_url_escape_characters = False`. If dict or
            JSON, the contents are used as-is.

            Similarly, the only time characters are quoted ("," becomes "%2C")
            is when the feature is printed (`__str__` method).

        extra : string or list
            Additional fields after the canonical 9 fields for GFF/GTF.

            If a string, then first assume it's serialized JSON; if this fails
            then assume it's a tab-delimited string of additional fields.  If
            it's a list already, then use as-is.

        bin : int
            UCSC genomic bin. If None, will be created based on provided
            start/end; if start or end is "." then bin will be None.

        id : None or string
            Database-specific primary key for this feature.  The only time this
            should not be None is if this feature is coming from a database, in
            which case it will be filled in automatically.

        dialect : dict or None

            The dialect to use when reconstructing attribute strings; defaults
            to the GFF3 spec.  :class:`FeatureDB` objects will automatically
            attach the dialect from the original file.

        file_order : int
            This is the `rowid` special field used in a sqlite3 database; this
            is provided by FeatureDB.

        keep_order : bool
            If True, then the attributes in the printed string will be in the
            order specified in the dialect.  Disabled by default, since this
            sorting step is time-consuming over many features.

        sort_attribute_values : bool
            If True, then the values of each attribute will be sorted when the
            feature is printed.  Mostly useful for testing, where the order is
            important for checking against expected values. Disabled by
            default, since it can be time-consuming over many features.

        sequence_length : int or None
            Length of the sequence (chromosome, contig) this feature is on.

            This is only needed for features on circular sequences that cross
            the origin and are therefore stored with ``start > end``.  With
            a sequence length, such features report two covered segments
            (``start..sequence_length`` and ``1..end``) and a positive
            ``len()``.  Without one, the coordinates remain undecided: methods
            that need the actual coverage raise
            :class:`gffutils.exceptions.MissingSequenceLengthError` rather than
            guessing by swapping the endpoints.

        """
        # start/end can be provided as int-like, ".", or None, but will be
        # converted to int or None
        if start == "." or start == "":
            start = None
        elif start is not None:
            start = int(start)
        if end == "." or end == "":
            end = None
        elif end is not None:
            end = int(end)

        # Flexible handling of attributes:
        # If dict, then use that; otherwise assume JSON and convert to a dict;
        # otherwise assume original string and convert to a dict.
        #
        # dict_class is set at the module level above...this is so you can swap
        # in and out different dict implementations (ordered, defaultdict, etc)
        # for testing.
        attributes = attributes or dict_class()

        if isinstance(attributes, str):
            try:
                attributes = helpers._unjsonify(attributes, isattributes=True)

            # it's a string but not JSON: assume original attributes string.
            except json.JSONDecodeError:

                # But Feature.attributes is still a dict
                attributes, _dialect = parser._split_keyvals(attributes)

                # Use this dialect if none provided.
                dialect = dialect or _dialect

        # If string, then try un-JSONifying it into a list; if that doesn't
        # work then assume it's tab-delimited and convert to a list.
        extra = extra or []
        if isinstance(extra, str):
            try:
                extra = helpers._unjsonify(extra)
            except json.JSONDecodeError:
                extra = extra.split("\t")

        self.seqid = seqid
        self.source = source
        self.featuretype = featuretype
        self.start = start
        self.end = end
        self.score = score
        self.strand = strand
        self.frame = frame
        self.attributes = attributes
        self.extra = extra
        self.bin = self.calc_bin(bin)
        self.id = id
        self.dialect = dialect or constants.dialect
        self.file_order = file_order
        self.keep_order = keep_order
        self.sort_attribute_values = sort_attribute_values
        if sequence_length is not None:
            sequence_length = int(sequence_length)
        self.sequence_length = sequence_length

    def calc_bin(self, _bin=None):
        """
        Calculate the smallest UCSC genomic bin that will contain this feature.
        """
        if _bin is None:
            try:
                # A feature crossing the origin does not fit in any single
                # small bin, so fall back to the whole-chromosome bin.
                if self.start is not None and self.end is not None:
                    if self.start > self.end:
                        return 1
                _bin = bins.bins(self.start, self.end, one=True)
            except TypeError:
                _bin = None
        return _bin

    @property
    def crosses_origin(self):
        """
        True if this feature wraps around the origin of its sequence.

        This is inferred from ``start > end``.  ``None`` means the
        coordinates imply a wrap but the sequence length is unavailable, so
        the feature remains undecided.
        """
        return circular.crosses_origin(self.start, self.end, self.sequence_length)

    @property
    def segments(self):
        """
        The actual 1-based closed genomic segments covered by this feature.

        Ordinary features cover a single ``(start, end)`` segment.  Features
        that cross the origin cover ``(start, sequence_length)`` followed by
        ``(1, end)``, i.e. the order implied by the annotation's start and end.
        Raises :class:`~gffutils.exceptions.MissingSequenceLengthError` for
        wrapped features whose sequence length is unknown.
        """
        return circular.coverage_segments(
            self.start,
            self.end,
            self.sequence_length,
            order="annotation",
        )

    def logical(self):
        """
        Return a :class:`gffutils.circular.LogicalFeature` view of this
        feature, including its original coordinates, actual covered segments,
        and covered length.
        """
        return circular.LogicalFeature(self, sequence_length=self.sequence_length)

    @property
    def covered_segments(self):
        """Alias for :attr:`segments`."""
        return self.segments

    def logical_span(self):
        """
        Describe the logical object as ``(start, end, covered_length)``.

        The original ``start`` and ``end`` are preserved so the circular
        annotation can be round-tripped; the third item is the number of bases
        actually covered.  A sequence length is required for wrapped
        coordinates.
        """
        if self.start is None or self.end is None:
            return (self.start, self.end, 0)
        return circular.logical_span(
            self.start, self.end, self.sequence_length
        )

    def __repr__(self):
        memory_loc = hex(id(self))

        # Reconstruct start/end as "."
        if self.start is None:
            start = "."
        else:
            start = self.start
        if self.end is None:
            end = "."
        else:
            end = self.end

        return (
            "<Feature {x.featuretype} ({x.seqid}:{start}-{end}"
            "[{x.strand}]) at {loc}>".format(
                x=self, start=start, end=end, loc=memory_loc
            )
        )

    def __getitem__(self, key):
        if isinstance(key, int):
            # TODO: allow access to "extra" fields
            attr = _position_lookup[key]
            return getattr(self, attr)
        else:
            return self.attributes[key]

    def __setitem__(self, key, value):
        if isinstance(key, int):
            # TODO: allow setting of "extra" fields
            attr = _position_lookup[key]
            setattr(self, attr, value)

        else:
            self.attributes[key] = value

    def __str__(self):
        return self.__unicode__()

    def __unicode__(self):

        # All fields but attributes (and extra).
        items = [getattr(self, k) for k in constants._gffkeys[:-1]]

        # Handle start/stop, which are either None or int
        if items[3] is None:
            items[3] = "."
        else:
            items[3] = str(items[3])

        if items[4] is None:
            items[4] = "."
        else:
            items[4] = str(items[4])

        # Reconstruct from dict and dialect
        reconstructed_attributes = parser._reconstruct(
            self.attributes,
            self.dialect,
            keep_order=self.keep_order,
            sort_attribute_values=self.sort_attribute_values,
        )

        # Final line includes reconstructed as well as any previously-added
        # "extra" fields
        items.append(reconstructed_attributes)
        if self.extra:
            items.append("\t".join(self.extra))

        return "\t".join(items)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)

    def __ne__(self, other):
        return str(self) != str(other)

    def __len__(self):
        if self.start is None or self.end is None:
            raise TypeError("Cannot take the length of a feature without coordinates")
        if self.start > self.end:
            if self.sequence_length is None:
                raise MissingSequenceLengthError(
                    "Cannot take the length of %s:%s-%s without a sequence "
                    "length; the feature crosses the origin"
                    % (self.seqid, self.start, self.end)
                )
            return circular.covered_length(
                self.start, self.end, self.sequence_length
            )
        return self.stop - self.start + 1

    # aliases for official GFF field names; this way x.chrom == x.seqid; and
    # x.stop == x.end.
    @property
    def chrom(self):
        return self.seqid

    @chrom.setter
    def chrom(self, value):
        self.seqid = value

    @property
    def stop(self):
        return self.end

    @stop.setter
    def stop(self, value):
        self.end = value

    def astuple(self, encoding=None):
        """
        Return a tuple suitable for import into a database.

        Attributes field and extra field jsonified into strings. The order of
        fields is such that they can be supplied as arguments for the query
        defined in :attr:`gffutils.constants._INSERT`.

        If `encoding` is not None, then convert string fields to unicode using
        the provided encoding.

        Returns
        -------
        Tuple
        """
        if not encoding:
            return (
                self.id,
                self.seqid,
                self.source,
                self.featuretype,
                self.start,
                self.end,
                self.score,
                self.strand,
                self.frame,
                helpers._jsonify(self.attributes),
                helpers._jsonify(self.extra),
                self.calc_bin(),
            )
        return (
            self.id.decode(encoding),
            self.seqid.decode(encoding),
            self.source.decode(encoding),
            self.featuretype.decode(encoding),
            self.start,
            self.end,
            self.score.decode(encoding),
            self.strand.decode(encoding),
            self.frame.decode(encoding),
            helpers._jsonify(self.attributes).decode(encoding),
            helpers._jsonify(self.extra).decode(encoding),
            self.calc_bin(),
        )

    def sequence(self, fasta, use_strand=True):
        """
        Retrieves the sequence of this feature as a string.

        Uses the pyfaidx package.

        Parameters
        ----------

        fasta : str
            If str, then it's a FASTA-format filename; otherwise assume it's
            a pyfaidx.Fasta object.

        use_strand : bool
            If True (default), the sequence returned will be
            reverse-complemented for minus-strand features.

        Returns
        -------
        string

        Notes
        -----
        Features that cross the origin (``start > end``) are resolved against
        the length of their sequence, so the returned string is the
        concatenation of the segments in transcription order.  A sequence
        length must be known either from the feature itself or from ``fasta``.
        """
        if isinstance(fasta, os.PathLike):
            fasta = os.fspath(fasta)
        if isinstance(fasta, str):
            fasta = Fasta(fasta, as_raw=False)

        if self.start is None or self.end is None:
            return ""

        sequence_length = self.sequence_length
        if sequence_length is None and self.start > self.end:
            try:
                sequence_length = len(fasta[self.chrom])
            except (KeyError, IndexError):
                raise MissingSequenceLengthError(
                    "Cannot extract the sequence of %s:%s-%s without a "
                    "sequence length; the feature crosses the origin"
                    % (self.seqid, self.start, self.end)
                )

        # recall GTF/GFF is 1-based closed;  pyfaidx uses Python slice notation
        # and is therefore 0-based half-open.
        if self.start > self.end:
            segments = circular.coverage_segments(
                self.start,
                self.end,
                sequence_length,
                order="annotation",
            )
            seq = "".join(fasta[self.chrom][s - 1 : e].seq for s, e in segments)
            if use_strand and self.strand == "-":
                seq = circular.reverse_complement(seq)
            return seq

        seq = fasta[self.chrom][self.start - 1 : self.stop]
        if use_strand and self.strand == "-":
            seq = seq.reverse.complement
        return seq.seq


def feature_from_line(line, dialect=None, strict=True, keep_order=False):
    """
    Given a line from a GFF file, return a Feature object

    Parameters
    ----------
    line : string

    strict : bool
        If True (default), assume `line` is a single, tab-delimited string that
        has at least 9 fields.

        If False, then the input can have a more flexible format, useful for
        creating single ad hoc features or for writing tests.  In this case,
        `line` can be a multi-line string (as long as it has a single non-empty
        line), and, as long as there are only 9 fields (standard GFF/GTF), then
        it's OK to use spaces instead of tabs to separate fields in `line`.
        But if >9 fields are to be used, then tabs must be used.

    keep_order, dialect
        Passed directly to :class:`Feature`; see docstring for that class for
        description

    Returns
    -------
    A new :class:`Feature` object.
    """
    if not strict:
        lines = line.splitlines(False)
        _lines = []
        for i in lines:
            i = i.strip()
            if len(i) > 0:
                _lines.append(i)

        assert len(_lines) == 1, _lines
        line = _lines[0]

        if "\t" in line:
            fields = line.rstrip("\n\r").split("\t")
        else:
            fields = line.rstrip("\n\r").split(None, 8)
    else:
        fields = line.rstrip("\n\r").split("\t")
    try:
        attr_string = fields[8]
    except IndexError:
        attr_string = ""
    attrs, _dialect = parser._split_keyvals(attr_string, dialect=dialect)
    d = dict(list(zip(constants._gffkeys, fields)))
    d["attributes"] = attrs
    d["extra"] = fields[9:]
    d["keep_order"] = keep_order
    if dialect is None:
        dialect = _dialect
    return Feature(dialect=dialect, **d)
