"""Support for using reading tabix files and using them in bgzip files."""

import dataclasses
import enum
import gzip
import pathlib
import struct

from vcfpy.bgzf import BgzfReader


class FileFormat(enum.Enum):
    """Enum for file formats supported by tabix."""

    #: Generic tabix file.
    GENERIC = 0
    #: SAM file.
    SAM = 1
    #: VCF file.
    VCF = 2


@dataclasses.dataclass
class Chunk:
    """Chunk."""

    #: Begin virtual offset.
    beg: int
    #: End virtual offset.
    end: int


@dataclasses.dataclass
class Bin:
    """Bin with chunks."""

    #: Bin number.
    number: int
    #: Chunks in this bin.
    chunks: list[Chunk]


@dataclasses.dataclass
class SequenceIndex:
    """Per-sequence index."""

    #: Bins containing chunks.
    bins: list[Bin]
    #: Linear index intervals.
    offsets: list[int]


@dataclasses.dataclass
class TabixIndex:
    """Index as read from Tabix files and relevant after reading the index."""

    #: Format of underlying file.
    format: FileFormat
    #: Column for sequence name.
    col_seq: int
    #: Column for begin position.
    col_beg: int
    #: Column for end position.
    col_end: int
    #: Meta character.
    meta: bytes
    #: Lines to skip at the beginning.
    skip: int
    #: Per-sequence indices.
    indices: dict[str, SequenceIndex]
    #: Optional number of unmapped reads.
    num_no_coord: int | None = None


def reg2bins(beg: int, end: int) -> list[int]:
    """Get list of bins that may overlap a region [beg, end).

    Based on the UCSC binning scheme used by tabix.

    :param beg: 0-based start position (inclusive)
    :param end: 0-based end position (exclusive)
    :return: list of bin numbers that may overlap the region (in reverse order)
    """
    bins = []
    end -= 1  # Convert to exclusive end to inclusive end

    # UCSC binning scheme with 5 levels and min_shift=14
    t = 0
    s = 14 + (5 << 1) + 5  # min_shift + (n_levels << 1) + n_levels = 14 + 10 + 5 = 29

    for level in range(5 + 1):  # n_levels + 1 = 6 levels (0-5)
        b = t + (beg >> s)
        e = t + (end >> s)
        for k in range(b, e + 1):
            bins.append(k)
        t += 1 << ((level << 1) + level)  # Update t for next level
        s -= 3  # Decrease shift by 3 for next level

    # Return in reverse order as per legacy implementation
    return list(reversed(bins))


def _parse_info_field(info: str, key: str) -> str | None:
    """Return the raw string value of an INFO ``key`` entry, or ``None``."""
    for entry in info.split(";"):
        if "=" not in entry:
            continue
        entry_key, value = entry.split("=", 1)
        if entry_key == key:
            return value
    return None


def _parse_ci_offsets(info: str, key: str) -> tuple[int, int] | None:
    """Parse a CIPOS/CIEND-style ``"left,right"`` INFO value."""
    raw = _parse_info_field(info, key)
    if raw is None:
        return None
    parts = raw.split(",")
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def parse_record_span(line: str) -> tuple[int, int, int | None, int | None, bool]:
    """Parse the positional span of a VCF record line.

    :return: tuple ``(begin, end, ci_begin, ci_end, is_breakend)`` with 1-based
        inclusive coordinates; ``ci_begin``/``ci_end`` are ``None`` when the
        corresponding confidence interval is absent.  Structural variants use
        the ``END`` INFO field instead of the REF length, insertions collapse
        to their insertion point, and breakends have no span on this contig.
        When a symbolic variant lacks ``END`` (unknown length) only the anchor
        base is definite and, if present, ``CIPOS`` is applied to both sides;
        an actually unbounded extent cannot be represented by the finite tabix
        intervals and is modeled by :py:class:`vcfpy.RecordInterval` instead.
    """
    fields = line.split("\t")
    pos = int(fields[1])
    ref = fields[3]
    alt = fields[4]
    info = fields[7] if len(fields) > 7 else "."

    alt_alleles = [] if alt in (".", "") else alt.split(",")
    is_breakend = any(("[" in a or "]" in a or a.startswith(".") or a.endswith(".")) for a in alt_alleles)
    symbolic_ins = any(a == "<INS>" or a.startswith("<INS:") for a in alt_alleles)
    substitution_insertions = [
        a for a in alt_alleles if not a.startswith("<") and not ("[" in a or "]" in a) and len(a) > len(ref)
    ]
    only_insertions = bool(alt_alleles) and len(substitution_insertions) == len(alt_alleles)
    is_insertion = symbolic_ins or only_insertions

    end_raw = _parse_info_field(info, "END")
    end = int(end_raw) if end_raw is not None else None

    if is_breakend:
        # Only the local breakpoint is anchored; there is no spanning interval.
        begin, definite_end = pos, pos
    elif is_insertion:
        # Insertion point between POS-1 and POS; both bounds meet at POS.
        begin, definite_end = pos, pos
    elif any(a.startswith("<") for a in alt_alleles) and end is not None:
        # Spanning symbolic variant (DEL, DUP, INV, ...): trust INFO/END.
        # Explicit alleles always use the REF span below, even if END is set.
        begin, definite_end = pos, end
    else:
        # Ordinary small variant: span the REF bases (1-based inclusive).
        begin, definite_end = pos, pos + len(ref) - 1

    cipos = _parse_ci_offsets(info, "CIPOS")
    ciend = _parse_ci_offsets(info, "CIEND")
    ci_begin = pos + cipos[0] if cipos is not None else None
    if ciend is not None:
        ci_end = definite_end + ciend[1]
    elif cipos is not None:
        ci_end = definite_end + cipos[1]
    else:
        ci_end = None

    return begin, definite_end, ci_begin, ci_end, is_breakend


def classify_record_overlap(
    begin: int,
    end: int,
    ci_begin: int | None,
    ci_end: int | None,
    query_start: int,
    query_end: int,
) -> str:
    """Classify overlap of a record with a 1-based inclusive query.

    :return: ``"definite"`` when the query overlaps the point-estimate span,
        ``"possible"`` when it only touches the CIPOS/CIEND confidence region,
        and ``"no_hit"`` otherwise.
    """
    if not (end < query_start or begin > query_end):
        return "definite"
    if ci_begin is not None and ci_end is not None and not (ci_end < query_start or ci_begin > query_end):
        return "possible"
    return "no_hit"


def read_index(path_tbi: pathlib.Path | str) -> TabixIndex:
    """Read tabix index from given path.

    :param path_tbi: path to the tabix index file
    :return: the read index
    """
    with gzip.open(path_tbi, "rb") as f:
        # Read header
        magic = f.read(4)
        if magic != b"TBI\x01":
            raise ValueError(f"Invalid tabix magic: {magic}")

        # Read configuration
        struct.unpack("<i", f.read(4))[0]
        format_val = struct.unpack("<i", f.read(4))[0]
        col_seq = struct.unpack("<i", f.read(4))[0]
        col_beg = struct.unpack("<i", f.read(4))[0]
        col_end = struct.unpack("<i", f.read(4))[0]
        meta = struct.unpack("<i", f.read(4))[0]
        skip = struct.unpack("<i", f.read(4))[0]
        l_nm = struct.unpack("<i", f.read(4))[0]

        # Read sequence names
        names_data = f.read(l_nm)
        names = names_data.rstrip(b"\x00").split(b"\x00")
        sequence_names = [name.decode("utf-8") for name in names if name]

        # Create indices dictionary
        indices = {}

        # Read per-sequence indices
        for seq_name in sequence_names:
            # Read number of bins
            n_bin = struct.unpack("<i", f.read(4))[0]

            bins = []
            # Read bins
            for _ in range(n_bin):
                bin_number = struct.unpack("<I", f.read(4))[0]
                n_chunk = struct.unpack("<i", f.read(4))[0]

                # Read chunks for this bin
                chunks = []
                for _ in range(n_chunk):
                    cnk_beg = struct.unpack("<Q", f.read(8))[0]
                    cnk_end = struct.unpack("<Q", f.read(8))[0]
                    chunks.append(Chunk(beg=cnk_beg, end=cnk_end))

                bins.append(Bin(number=bin_number, chunks=chunks))

            # Read linear index
            n_intv = struct.unpack("<i", f.read(4))[0]
            offsets = []
            for _ in range(n_intv):
                ioff = struct.unpack("<Q", f.read(8))[0]
                offsets.append(ioff)

            indices[seq_name] = SequenceIndex(bins=bins, offsets=offsets)

        # Try to read optional unmapped reads count
        num_no_coord = None
        try:
            remaining = f.read(8)
            if len(remaining) == 8:
                num_no_coord = struct.unpack("<Q", remaining)[0]
        except (IOError, struct.error):  # pragma: no cover
            pass  # Optional field, ignore if not present

        # Convert format value to enum
        format_enum = FileFormat(format_val & 0xFFFF)  # Mask off coordinate type flag

        return TabixIndex(
            format=format_enum,
            col_seq=col_seq,
            col_beg=col_beg,
            col_end=col_end,
            meta=chr(meta).encode("utf-8"),
            skip=skip,
            indices=indices,
            num_no_coord=num_no_coord,
        )


class TabixFile:
    """Provides easy access for reading tabix files."""

    def __init__(self, *, filename: pathlib.Path | str, index: pathlib.Path | str | None = None):
        """Create new ``TabixFile``.

        :param filename: path to the bgzip-ed file
        :param index: path to the tabix index file
        """

        self.filename: str = str(filename)
        self.index_filename: str = f"{filename}.tbi" if index is None else str(index)
        self.closed: bool = False
        self.file: BgzfReader = BgzfReader(self.filename)
        self.index_file: TabixIndex = read_index(self.index_filename)

    def fetch(
        self,
        *,
        reference: str | None = None,
        start: int | None = None,
        end: int | None = None,
        region: str | None = None,
    ) -> "TabixFileIter":
        """Fetch iterator for given region."""
        if region and reference is not None:
            raise ValueError("Cannot specify both region and reference/start/end")
        if (start is not None or end is not None) and reference is None:
            raise ValueError("If start or end is given, reference must be given too")

        # Parse out the region, can be one of "sequence" or "sequence:start-end",
        # commas are stripped from start and end.
        if region:
            if ":" in region:
                ref, pos = region.split(":", 1)
                if "-" in pos:
                    start_str, end_str = pos.split("-", 1)
                    start = int(start_str.replace(",", ""))
                    end = int(end_str.replace(",", ""))
                else:
                    start = int(pos.replace(",", ""))
                    end = None
                reference = ref
            else:
                reference = region
                start = None
                end = None
        assert reference is not None

        # Validate reference.
        if reference not in self.index_file.indices:
            raise ValueError(f"Reference {reference} not found in index")

        # Construct and return the iterator
        return TabixFileIter(
            index=self.index_file,
            reference=reference,
            start=start,
            end=end,
            bgzf_file=self.file,
        )

    def close(self) -> None:
        """Close the TabixFile."""
        if self.closed:  # pragma: no cover
            raise ValueError("TabixFile already closed - double close?")
        self.file.close()
        self.closed = True

    def __enter__(self) -> "TabixFile":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class TabixFileIter:
    """Allows for easy iteration over a tabix file."""

    def __init__(
        self,
        *,
        index: TabixIndex,
        reference: str,
        start: int | None = None,
        end: int | None = None,
        bgzf_file: BgzfReader | None = None,
    ):
        """Initialize TabixFileIter.

        :param index: The tabix index to use for lookups
        :param reference: Reference sequence name
        :param start: Start position (1-based, inclusive) or None for start of sequence
        :param end: End position (1-based, inclusive) or None for end of sequence
        :param bgzf_file: BGZF file reader (if None, iteration will not work)
        """
        self.index = index
        self.reference = reference
        self.start = start
        self.end = end
        self.bgzf_file = bgzf_file

        # Validate that the reference exists in the index
        if reference not in index.indices:  # pragma: no cover
            raise ValueError(f"Reference '{reference}' not found in tabix index")

        # Store the sequence index for this reference
        self.sequence_index = index.indices[reference]

        # Initialize iteration state
        self._chunks: list[Chunk] = []
        self._current_chunk_idx = 0
        self._current_lines: list[str] = []
        self._current_line_idx = 0
        self._finished = False
        #: Class (``"definite"``/``"possible"``) of the last line returned by
        #: the iterator; ``None`` before the first match
        self.last_match_class: str | None = None

        # Find relevant chunks for the region
        self._find_chunks()

    def _find_chunks(self) -> None:
        """Find chunks that may contain data for the requested region."""
        if self.start is None and self.end is None:
            # No region specified - get all chunks for this reference, but filter out special chunks
            # Use the lowest-level bins (bin 0 and highest numbered bins) to avoid duplicates
            chunks = []

            # Create a mapping of bin numbers to bins for efficient lookup
            bin_map = {bin_obj.number: bin_obj for bin_obj in self.sequence_index.bins}

            # For whole chromosome, prefer the most specific bins to avoid reading overlapping data
            # Sort bins by number and take chunks from highest numbered bins first
            sorted_bins = sorted(self.sequence_index.bins, key=lambda b: b.number, reverse=True)

            covered_ranges = []  # Track (start, end) ranges we've already covered

            for bin_obj in sorted_bins:
                for chunk in bin_obj.chunks:
                    # Skip special chunks with end=0
                    if chunk.end == 0:
                        continue

                    # Check if this chunk overlaps with already covered ranges
                    overlaps = False
                    for covered_start, covered_end in covered_ranges:
                        if not (chunk.end <= covered_start or chunk.beg >= covered_end):
                            overlaps = True
                            break

                    if not overlaps:
                        chunks.append(chunk)
                        covered_ranges.append((chunk.beg, chunk.end))

            self._chunks = sorted(chunks, key=lambda c: c.beg)
        else:
            # Region specified - convert 1-based coordinates to 0-based for internal use
            start_0based = (self.start - 1) if self.start is not None else 0
            end_0based = self.end if self.end is not None else 2**29  # Large value for unbounded end

            # Find overlapping bins using UCSC binning scheme
            bin_numbers = reg2bins(start_0based, end_0based)

            # Create a mapping of bin numbers to bins for efficient lookup
            bin_map = {bin_obj.number: bin_obj for bin_obj in self.sequence_index.bins}

            # Collect chunks from overlapping bins in the order returned by reg2bins (reverse order)
            chunks = []
            for bin_number in bin_numbers:
                if bin_number in bin_map:
                    bin_obj = bin_map[bin_number]
                    for chunk in bin_obj.chunks:
                        # Skip special chunks with end=0
                        if chunk.end != 0:
                            chunks.append(chunk)

            # Use linear index to find minimum virtual offset
            linear_min_offset = self._lookup_linear_offset(start_0based)

            # Filter chunks that end before our linear minimum
            if linear_min_offset is not None:
                chunks = [c for c in chunks if c.end >= linear_min_offset]

            # Merge overlapping chunks to create a single virtual range
            if chunks:
                # Sort chunks by start position
                chunks.sort(key=lambda c: c.beg)

                # Merge overlapping chunks
                merged_chunks = [chunks[0]]
                for chunk in chunks[1:]:
                    last_chunk = merged_chunks[-1]
                    if chunk.beg <= last_chunk.end:
                        # Merge overlapping chunks
                        merged_chunks[-1] = Chunk(
                            beg=min(last_chunk.beg, chunk.beg), end=max(last_chunk.end, chunk.end)
                        )
                    else:  # pragma: no cover
                        merged_chunks.append(chunk)

                self._chunks = merged_chunks
            else:
                self._chunks = []

    def _lookup_linear_offset(self, pos: int) -> int | None:
        """Use linear index to find minimum virtual offset for position."""
        # Linear index uses 16kb intervals
        interval_idx = pos >> 14  # pos // 16384

        if interval_idx < len(self.sequence_index.offsets):
            return self.sequence_index.offsets[interval_idx]
        return None

    def _load_next_chunk(self) -> bool:
        """Load lines from the next chunk. Returns True if successful."""
        if self._current_chunk_idx >= len(self._chunks) or self.bgzf_file is None:
            return False

        chunk = self._chunks[self._current_chunk_idx]
        self._current_chunk_idx += 1

        # Seek to chunk start and read until chunk end
        self.bgzf_file.seek(chunk.beg)

        lines = []
        current_offset = chunk.beg

        while current_offset < chunk.end:
            line = self.bgzf_file.readline()
            if not line:  # pragma: no cover
                break

            lines.append(line.rstrip("\n\r"))
            current_offset = self.bgzf_file.tell()

        self._current_lines = lines
        self._current_line_idx = 0
        return len(lines) > 0

    def _line_matches_region(self, line: str) -> bool:
        """Check if a line matches our region criteria and record its class.

        Records overlapping the definite span are definite hits; records whose
        CIPOS/CIEND confidence interval alone reaches the query are possible
        hits and kept separate in :py:attr:`last_match_class`.
        """
        try:
            fields = line.split("\t")
            seq_name = fields[self.index.col_seq - 1] if self.index.col_seq > 0 else ""

            # Check sequence name
            if seq_name != self.reference:  # pragma: no cover
                return False

            if self.start is None and self.end is None:
                # Whole-reference query: every record is a definite hit.
                self.last_match_class = "definite"
                return True

            begin_pos, end_pos, ci_begin, ci_end, _is_bnd = parse_record_span(line)
            query_start = self.start if self.start is not None else 1
            query_end = self.end if self.end is not None else 2**29
            match_class = classify_record_overlap(
                begin_pos, end_pos, ci_begin, ci_end, query_start, query_end
            )
            if match_class == "no_hit":
                return False
            self.last_match_class = match_class
            return True
        except (ValueError, IndexError):  # pragma: no cover
            # Skip malformed lines
            return False

    def __iter__(self) -> "TabixFileIter":
        """Return self as iterator."""
        return self

    def __next__(self) -> str:
        """Return the next record."""
        if self._finished:  # pragma: no cover
            raise StopIteration

        # Load chunks until we find a matching line or run out
        while True:
            # Check if we have lines in current batch
            if self._current_line_idx >= len(self._current_lines):
                # Need to load next chunk
                if not self._load_next_chunk():
                    self._finished = True
                    raise StopIteration

            # Check remaining lines in current batch
            while self._current_line_idx < len(self._current_lines):
                line = self._current_lines[self._current_line_idx]
                self._current_line_idx += 1

                # Skip header/comment lines
                if line.startswith("#"):  # pragma: no cover
                    continue

                # Check if line matches our region
                if self._line_matches_region(line):
                    return line

                # Check if we've gone past our region
                try:
                    fields = line.split("\t")
                    seq_name = fields[self.index.col_seq - 1] if self.index.col_seq > 0 else ""
                    begin_pos, _end_pos, ci_begin, _ci_end, _is_bnd = parse_record_span(line)

                    # If we're past our reference sequence, we're done
                    # Use the index order, not lexical order
                    sequence_names = list(self.index.indices.keys())
                    if seq_name in sequence_names and self.reference in sequence_names:
                        seq_idx = sequence_names.index(seq_name)
                        ref_idx = sequence_names.index(self.reference)
                        if seq_idx > ref_idx:  # pragma: no cover
                            self._finished = True
                            raise StopIteration

                    # If even the left confidence edge is past our end position,
                    # no later record (sorted by POS) can match anymore.
                    earliest_begin = ci_begin if ci_begin is not None else begin_pos
                    if seq_name == self.reference and self.end is not None and earliest_begin > self.end:
                        self._finished = True
                        raise StopIteration

                except (ValueError, IndexError):  # pragma: no cover
                    # Skip malformed lines
                    continue

            # Processed all lines in current batch, loop to load next chunk
