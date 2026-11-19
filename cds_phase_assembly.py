"""Assemble a multi-part CDS from a reference fragment, respecting strand and phase.

Why this exists
---------------
For a minus-strand CDS split into several segments, concatenating the
segments in ascending genomic coordinate order (and/or translating the
plus strand directly) yields the wrong protein.  The correct assembly
must:

1. order segments in *transcription* order (5'->3' of the mRNA), which
   is ascending coordinates for '+' and *descending* for '-';
2. reverse-complement minus-strand segments;
3. honour the GFF3 ``phase`` of every segment: phase is the number of
   bases at the 5' end (in transcription orientation) of a segment that
   belong to the codon started in the previous segment.  It is a frame
   marker, **not** an instruction to delete bases;
4. verify phase consistency between consecutive segments and name the
   segment that cannot be joined when they contradict;
5. never drop trailing bases just to force a multiple of three -- an
   incomplete 3' codon is reported, and both ends are flagged as
   complete/incomplete explicitly;
6. allow verification of a known N-terminal protein prefix even when
   the 3' end is incomplete.

Coordinates are 1-based, inclusive, relative to the start of the given
reference fragment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

STOP_CODONS = {"TAA", "TAG", "TGA"}

_COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")


def reverse_complement(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


@dataclass(frozen=True)
class CDSSegment:
    """One CDS part as it would appear in a GFF3 file."""

    name: str
    start: int  # 1-based inclusive, on the reference fragment
    end: int
    strand: str  # '+' or '-'
    phase: int  # GFF3 phase: 0, 1 or 2

    def __post_init__(self):
        if self.strand not in {"+", "-"}:
            raise ValueError(f"{self.name}: strand must be '+' or '-'")
        if self.phase not in {0, 1, 2}:
            raise ValueError(f"{self.name}: phase must be 0, 1 or 2")
        if self.end < self.start:
            raise ValueError(f"{self.name}: end < start")

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass
class SegmentContribution:
    """Provenance of one segment inside the assembled CDS."""

    segment: CDSSegment
    order: int  # 0-based position in transcription order
    cds_offset: int  # 0-based offset of this segment in the assembled CDS
    length: int
    expected_phase: int | None  # None for the first (frame-defining) segment
    phase_ok: bool
    sequence: str  # segment sequence in transcription orientation


@dataclass
class AssemblyResult:
    cds: str  # assembled coding sequence, transcription orientation, no bases removed
    contributions: list[SegmentContribution]
    contradictions: list[str]
    five_prime_complete: bool
    three_prime_complete: bool
    leftover_3prime: str  # trailing bases not forming a full codon (never discarded silently)
    translation: str  # full-codon translation; '*' marks stop codons
    notes: list[str] = field(default_factory=list)

    def check_prefix(self, known_prefix: str) -> bool:
        """Verify a known complete N-terminal protein prefix."""
        return self.translation.startswith(known_prefix)


def _transcription_order(segments: list[CDSSegment]) -> list[CDSSegment]:
    strands = {s.strand for s in segments}
    if len(strands) != 1:
        raise ValueError("all CDS segments must share one strand")
    strand = strands.pop()
    # '+' : transcribed low->high coordinates; '-' : high->low coordinates.
    return sorted(segments, key=lambda s: s.start, reverse=(strand == "-"))


def assemble_cds(reference: str, segments: list[CDSSegment]) -> AssemblyResult:
    """Assemble the CDS from ``segments`` cut out of ``reference``.

    Segments may be given in any order.  Every base of every segment is
    kept; phase is used only to locate codon boundaries and to check
    that consecutive segments can be joined.
    """
    if not segments:
        raise ValueError("no CDS segments given")
    ordered = _transcription_order(segments)

    contributions: list[SegmentContribution] = []
    contradictions: list[str] = []
    notes: list[str] = []
    parts: list[str] = []
    cds_offset = 0
    phase0 = ordered[0].phase

    for i, seg in enumerate(ordered):
        genomic = reference[seg.start - 1 : seg.end]
        if len(genomic) != seg.length:
            raise ValueError(f"{seg.name}: segment extends beyond reference fragment")
        seq = genomic if seg.strand == "+" else reverse_complement(genomic)

        if i == 0:
            expected = None
            phase_ok = True
            if seg.phase != 0:
                notes.append(
                    f"5' end is partial: first segment {seg.name} has phase "
                    f"{seg.phase}, so the first {seg.phase} base(s) belong to a "
                    f"codon that started upstream of the annotated CDS"
                )
        else:
            # Bases since the first codon start must leave exactly
            # `phase` bases of an unfinished codon at the junction.
            expected = (phase0 - cds_offset) % 3
            phase_ok = seg.phase == expected
            if not phase_ok:
                prev = ordered[i - 1]
                contradictions.append(
                    f"segment {seg.name} cannot be joined after {prev.name}: "
                    f"declared phase {seg.phase}, expected {expected} "
                    f"({cds_offset - phase0} coding bases precede it)"
                )

        contributions.append(
            SegmentContribution(
                segment=seg,
                order=i,
                cds_offset=cds_offset,
                length=seg.length,
                expected_phase=expected,
                phase_ok=phase_ok,
                sequence=seq,
            )
        )
        parts.append(seq)
        cds_offset += seg.length

    cds = "".join(parts)

    # Translate full codons only, starting at the frame set by the first
    # segment's phase.  Trailing bases are reported, never dropped.
    body = cds[phase0:]
    leftover = body[len(body) - (len(body) % 3) :] if len(body) % 3 else ""
    codons = [body[j : j + 3] for j in range(0, len(body) - len(body) % 3, 3)]
    translation = "".join(CODON_TABLE.get(c, "X") for c in codons)

    if len(body) % 3:
        notes.append(
            f"3' end is partial: {len(body) % 3} trailing base(s) "
            f"({leftover}) do not form a complete codon; kept in the CDS, "
            f"excluded from translation"
        )

    five_prime_complete = phase0 == 0 and codons[:1] == ["ATG"]
    three_prime_complete = not leftover and codons[-1:] and codons[-1] in STOP_CODONS
    if not five_prime_complete:
        notes.append("5' end incomplete: CDS does not start with ATG in frame 0")
    if not three_prime_complete:
        notes.append("3' end incomplete: CDS does not end with an in-frame stop codon")
    internal_stops = translation[:-1].count("*") if translation else 0
    if internal_stops:
        notes.append(f"warning: {internal_stops} internal stop codon(s) in translation")

    return AssemblyResult(
        cds=cds,
        contributions=contributions,
        contradictions=contradictions,
        five_prime_complete=five_prime_complete,
        three_prime_complete=three_prime_complete,
        leftover_3prime=leftover,
        translation=translation,
        notes=notes,
    )


def assemble_naive_ascending(reference: str, segments: list[CDSSegment]) -> str:
    """The buggy approach: ascending coordinates, plus strand, frame 0.

    Kept only to demonstrate *why* coordinate-sorted assembly of a
    minus-strand CDS produces the wrong protein.
    """
    seq = "".join(
        reference[s.start - 1 : s.end]
        for s in sorted(segments, key=lambda s: s.start)
    )
    return "".join(
        CODON_TABLE.get(seq[j : j + 3], "X") for j in range(0, len(seq) - 2, 3)
    )


def format_report(result: AssemblyResult, known_prefix: str | None = None) -> str:
    lines = []
    lines.append(f"assembled CDS ({len(result.cds)} nt): {result.cds}")
    lines.append("per-segment provenance (transcription order):")
    for c in result.contributions:
        s = c.segment
        expected = "-" if c.expected_phase is None else str(c.expected_phase)
        ok = "ok" if c.phase_ok else "MISMATCH"
        lines.append(
            f"  [{c.order}] {s.name}: {s.start}-{s.end} ({s.strand}) "
            f"phase={s.phase} expected={expected} [{ok}] -> "
            f"CDS[{c.cds_offset}:{c.cds_offset + c.length}] {c.sequence}"
        )
    lines.append(
        f"ends: 5' {'complete' if result.five_prime_complete else 'INCOMPLETE'}, "
        f"3' {'complete' if result.three_prime_complete else 'INCOMPLETE'}"
        + (f" (leftover: {result.leftover_3prime})" if result.leftover_3prime else "")
    )
    for note in result.notes:
        lines.append(f"  note: {note}")
    for contra in result.contradictions:
        lines.append(f"  PHASE CONTRADICTION: {contra}")
    lines.append(f"translation: {result.translation}")
    if known_prefix is not None:
        verdict = "MATCHES" if result.check_prefix(known_prefix) else "does NOT match"
        lines.append(f"known prefix {known_prefix!r}: {verdict}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demo: a minus-strand CDS in three segments on a 400 bp reference fragment.
# ---------------------------------------------------------------------------

# Coding sequence in transcription orientation (27 sense codons + stop).
# Built codon-by-codon so the length cannot be miscounted.
DEMO_CDS_TRANSCRIPTION = "".join([
    "ATG", "AGC", "ACC", "AAT", "CCG", "AAG", "CCG", "CAG", "CGT", "AAG",
    "ACC", "AAG", "CGC", "AAC", "ACC", "AAC", "CGT", "CGC", "CCG", "CAG",
    "GAC", "GTG", "AAG", "TTC", "CCG", "GGC", "GGT", "TAA",
])
DEMO_PROTEIN = "MSTNPKPQRKTKRNTNRRPQDVKFPGG"
KNOWN_PREFIX = "MSTNPKPQRK"

# Split points chosen so internal phases are non-zero: 40, 31, 13 nt.
_SPLIT1, _SPLIT2 = 40, 71


def build_demo_reference() -> tuple[str, list[CDSSegment]]:
    """Embed the demo CDS on the minus strand of a 400 bp fragment.

    Returns (reference, segments) with segments listed in ascending
    coordinate order, as a GFF3 parser would typically yield them.
    """
    part1 = DEMO_CDS_TRANSCRIPTION[:_SPLIT1]          # 40 nt, phase 0
    part2 = DEMO_CDS_TRANSCRIPTION[_SPLIT1:_SPLIT2]   # 31 nt, phase 2
    part3 = DEMO_CDS_TRANSCRIPTION[_SPLIT2:]          # 13 nt, phase 1

    ref = list(("GATTACA" * 60)[:400])

    def place(part: str, start: int) -> None:
        rc = reverse_complement(part)
        ref[start - 1 : start - 1 + len(rc)] = rc

    # Minus strand: the transcription-first part sits at the highest
    # coordinates.
    place(part3, 100)  # 100-112
    place(part2, 200)  # 200-230
    place(part1, 300)  # 300-339

    segments = [
        CDSSegment("cds3", 100, 112, "-", 1),
        CDSSegment("cds2", 200, 230, "-", 2),
        CDSSegment("cds1", 300, 339, "-", 0),
    ]
    return "".join(ref), segments


def main() -> None:
    reference, segments = build_demo_reference()

    print("=== naive assembly (ascending coordinates, plus strand) ===")
    print("protein:", assemble_naive_ascending(reference, segments))
    print()

    print("=== correct assembly (strand-aware, phase-checked) ===")
    result = assemble_cds(reference, segments)
    print(format_report(result, KNOWN_PREFIX))
    assert result.cds == DEMO_CDS_TRANSCRIPTION
    assert result.translation == DEMO_PROTEIN + "*"
    print()

    print("=== phase contradiction: cds2 mis-annotated with phase 0 ===")
    bad = [segments[0], CDSSegment("cds2", 200, 230, "-", 0), segments[2]]
    print(format_report(assemble_cds(reference, bad), KNOWN_PREFIX))
    print()

    print("=== 3'-incomplete CDS: last segment truncated by 2 nt ===")
    # Minus strand: the 3' (transcription) end is the LOW-coordinate end.
    truncated = [CDSSegment("cds3", 102, 112, "-", 1), segments[1], segments[2]]
    print(format_report(assemble_cds(reference, truncated), KNOWN_PREFIX))


if __name__ == "__main__":
    main()
