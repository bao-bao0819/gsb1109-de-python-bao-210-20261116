#!/usr/bin/env python3
"""Compare splicing patterns between two annotations.

Builds, for every transcript, a strand-aware exon-boundary / junction model
in transcription direction and compares the two annotations gene by gene.
Every verdict keeps per-transcript support.

Comparison semantics
--------------------
* Each transcript makes *positive claims*: exonic intervals (observed exons)
  and intronic intervals (observed junctions).  A junction is stored as
  (donor, acceptor) in transcription direction, so the same genomic exons on
  opposite strands never compare equal.
* A transcript whose start/end is incomplete (GFF3 ``start_range`` /
  ``end_range`` attributes, or IDs supplied via ``--incomplete``) only
  constrains the observed part.  Unknown endpoints are NEVER extended to the
  endpoints of the longest transcript.
* Sharing the same exon *set* is not enough: the junction path must agree.

Pairwise verdicts
-----------------
IDENTICAL     (necessarily identical)
              Same exon chain, same junction path, and both transcripts have
              known start and known end.
DIFFERENT     (clearly different)
              An exonic claim of one transcript overlaps an intronic claim of
              the other, or one transcript provably ends while the other
              extends further (known endpoint vs. extra exons), or strand
              mismatch within a matched gene.
INSUFFICIENT  (not enough evidence)
              No conflict, but one transcript only constrains part of the
              other (partial observation / unknown endpoints), so the pair
              can be neither proven identical nor proven different.

Usage
-----
    python compare_splicing.py ANN_A.gff3 ANN_B.gff3 [-o report.tsv]
    python compare_splicing.py A.gtf B.gtf --incomplete-b partial.tsv
"""

import argparse
import sys
from dataclasses import dataclass, field

import gffutils

TX_TYPES = ("mRNA", "transcript")

IDENTICAL = "IDENTICAL"
DIFFERENT = "DIFFERENT"
INSUFFICIENT = "INSUFFICIENT"


@dataclass(frozen=True)
class Junction:
    """A splice junction in transcription direction.

    donor/acceptor are the exonic edge positions (1-based inclusive):
    the last transcribed base of the upstream exon and the first
    transcribed base of the downstream exon.
    """

    donor: int
    acceptor: int


@dataclass
class TranscriptModel:
    source: str          # label of the annotation, e.g. "A" or "B"
    tx_id: str
    gene_key: str
    chrom: str
    strand: str
    exons: tuple         # ((start, end), ...), ascending genomic, 1-based incl.
    lo_known: bool       # genomic low-coordinate end is complete
    hi_known: bool       # genomic high-coordinate end is complete

    @property
    def lo(self):
        return self.exons[0][0]

    @property
    def hi(self):
        return self.exons[-1][1]

    @property
    def label(self):
        return "%s:%s" % (self.source, self.tx_id)

    @property
    def junctions(self):
        """Junctions in transcription direction (donor -> acceptor)."""
        out = []
        for (s1, e1), (s2, e2) in zip(self.exons, self.exons[1:]):
            if self.strand == "-":
                out.append(Junction(donor=s2, acceptor=e1))
            else:
                out.append(Junction(donor=e1, acceptor=s2))
        return tuple(out)

    @property
    def introns(self):
        """Intronic intervals as inclusive genomic (lo, hi); empty ones dropped."""
        out = []
        for (s1, e1), (s2, e2) in zip(self.exons, self.exons[1:]):
            if e1 + 1 <= s2 - 1:
                out.append((e1 + 1, s2 - 1))
        return tuple(out)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def sniff_gtf(path):
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) == 9:
                return "gene_id" in fields[8] and "=" not in fields[8]
    return False


def load_db(path):
    gtf = sniff_gtf(path)
    kwargs = dict(dbfn=":memory:", force=True, keep_order=True,
                  merge_strategy="create_unique")
    if gtf:
        # GTF has no transcript/gene lines; let gffutils infer them.
        kwargs.update(disable_infer_genes=False, disable_infer_transcripts=False)
    else:
        kwargs.update(disable_infer_genes=True, disable_infer_transcripts=True)
    return gffutils.create_db(path, **kwargs)


def _gene_key(db, tx):
    for attr in ("gene_id", "gene", "gene_name"):
        if attr in tx.attributes and tx.attributes[attr]:
            return tx.attributes[attr][0]
    for parent in db.parents(tx, featuretype=("gene", "pseudogene")):
        return parent.id
    if "Parent" in tx.attributes and tx.attributes["Parent"]:
        return tx.attributes["Parent"][0]
    return tx.id


def build_transcripts(db, source, incomplete):
    """Build TranscriptModels from a FeatureDB.

    ``incomplete`` maps tx_id -> set of unknown ends among {"5", "3"}
    (transcription orientation).
    """
    models = {}
    for tx in db.features_of_type(TX_TYPES, order_by=("seqid", "start")):
        exons = sorted(
            {(c.start, c.end) for c in db.children(tx, featuretype="exon")},
            key=lambda se: (se[0], se[1]),
        )
        if not exons:
            sys.stderr.write("warning: %s has no exons, skipped\n" % tx.id)
            continue
        lo_known = "start_range" not in tx.attributes
        hi_known = "end_range" not in tx.attributes
        unknown = incomplete.get(tx.id, set())
        if tx.strand == "-":
            if "3" in unknown:
                lo_known = False
            if "5" in unknown:
                hi_known = False
        else:
            if "5" in unknown:
                lo_known = False
            if "3" in unknown:
                hi_known = False
        models[tx.id] = TranscriptModel(
            source=source,
            tx_id=tx.id,
            gene_key=_gene_key(db, tx),
            chrom=tx.seqid,
            strand=tx.strand,
            exons=tuple(exons),
            lo_known=lo_known,
            hi_known=hi_known,
        )
    return models


def read_incomplete(path):
    """Read a TSV of incomplete transcripts.

    Format: ``tx_id[<TAB>ends]`` where ``ends`` contains "5" and/or "3"
    (transcription orientation).  No second column means both ends unknown.
    """
    incomplete = {}
    if path is None:
        return incomplete
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            ends = set(parts[1]) & {"5", "3"} if len(parts) > 1 else {"5", "3"}
            incomplete[parts[0]] = ends
    return incomplete


# ---------------------------------------------------------------------------
# pairwise comparison
# ---------------------------------------------------------------------------

def _overlap(a_lo, a_hi, b_lo, b_hi):
    return a_lo <= b_hi and b_lo <= a_hi


def compare_pair(a, b):
    """Return (verdict, reason) for two transcripts of the same gene."""
    if a.chrom != b.chrom or a.strand != b.strand:
        return DIFFERENT, "different chrom/strand (%s%s vs %s%s)" % (
            a.chrom, a.strand, b.chrom, b.strand)

    # Positive-claim conflict: exonic claim vs intronic claim.
    for (es, ee) in a.exons:
        for (il, ih) in b.introns:
            if _overlap(es, ee, il, ih):
                return DIFFERENT, (
                    "%s exon %s:%d-%d overlaps %s intron %d-%d"
                    % (a.label, a.chrom, es, ee, b.label, il, ih))
    for (es, ee) in b.exons:
        for (il, ih) in a.introns:
            if _overlap(es, ee, il, ih):
                return DIFFERENT, (
                    "%s exon %s:%d-%d overlaps %s intron %d-%d"
                    % (b.label, b.chrom, es, ee, a.label, il, ih))

    # Known-endpoint constraint: a complete end forbids the other
    # transcript from extending beyond it.  Unknown ends constrain nothing
    # and are never filled in from the longer transcript.
    if a.lo_known and b.lo < a.lo:
        return DIFFERENT, ("%s starts at %d (known) but %s extends to %d"
                           % (a.label, a.lo, b.label, b.lo))
    if b.lo_known and a.lo < b.lo:
        return DIFFERENT, ("%s starts at %d (known) but %s extends to %d"
                           % (b.label, b.lo, a.label, a.lo))
    if a.hi_known and b.hi > a.hi:
        return DIFFERENT, ("%s ends at %d (known) but %s extends to %d"
                           % (a.label, a.hi, b.label, b.hi))
    if b.hi_known and a.hi > b.hi:
        return DIFFERENT, ("%s ends at %d (known) but %s extends to %d"
                           % (b.label, b.hi, a.label, a.hi))

    if (a.exons == b.exons
            and a.lo_known and a.hi_known and b.lo_known and b.hi_known):
        return IDENTICAL, "same exon chain, same junction path, all ends known"

    # No conflict, but cannot prove equality.
    if a.exons == b.exons:
        reason = ("same observed exon chain but endpoint(s) unknown "
                  "(unknown ends are not imputed from the longer transcript)")
    else:
        shorter, longer = (a, b) if len(a.exons) <= len(b.exons) else (b, a)
        reason = ("%s constrains only the observed part of %s; "
                  "unobserved region may or may not agree"
                  % (shorter.label, longer.label))
    return INSUFFICIENT, reason


# ---------------------------------------------------------------------------
# gene matching and grouping
# ---------------------------------------------------------------------------

def match_genes(a_models, b_models):
    """Match gene keys across annotations.

    Primary: shared gene key.  Fallback: same-strand span overlap for keys
    present in only one annotation.  Returns (matched_pairs, only_a, only_b).
    """
    def gene_spans(models):
        spans = {}
        for m in models.values():
            cur = spans.get(m.gene_key)
            if cur is None:
                spans[m.gene_key] = [m.chrom, m.strand, m.lo, m.hi]
            else:
                cur[2] = min(cur[2], m.lo)
                cur[3] = max(cur[3], m.hi)
        return spans

    a_spans, b_spans = gene_spans(a_models), gene_spans(b_models)
    matched, only_a = [], []
    used_b = set()
    for key in sorted(a_spans):
        if key in b_spans:
            matched.append((key, key, "by_id"))
            used_b.add(key)
            continue
        ac, astrand, alo, ahi = a_spans[key]
        hit = None
        for bkey, (bc, bstrand, blo, bhi) in sorted(b_spans.items()):
            if bkey in used_b or bkey in a_spans:
                continue
            if ac == bc and astrand == bstrand and _overlap(alo, ahi, blo, bhi):
                hit = bkey
                break
        if hit is None:
            only_a.append(key)
        else:
            matched.append((key, hit, "by_overlap"))
            used_b.add(hit)
    only_b = [k for k in sorted(b_spans) if k not in used_b and k not in a_spans]
    return matched, only_a, only_b


class _UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        self.parent[self.find(x)] = self.find(y)


def compare_annotations(a_models, b_models):
    """Compare two annotations; returns a list of per-gene report dicts."""
    matched, only_a, only_b = match_genes(a_models, b_models)
    reports = []

    for a_key, b_key, how in matched:
        a_txs = [m for m in a_models.values() if m.gene_key == a_key]
        b_txs = [m for m in b_models.values() if m.gene_key == b_key]
        pairs, uf = [], _UnionFind()
        for m in a_txs + b_txs:
            uf.find(m.label)
        for a in a_txs:
            for b in b_txs:
                verdict, reason = compare_pair(a, b)
                pairs.append((a.label, b.label, verdict, reason))
                if verdict == IDENTICAL:
                    uf.union(a.label, b.label)
        # Identical pattern groups (per-transcript support preserved).
        groups = {}
        for m in a_txs + b_txs:
            groups.setdefault(uf.find(m.label), []).append(m.label)
        identical_groups = [sorted(v) for v in groups.values()
                            if any(l.startswith("A:") for l in v)
                            and any(l.startswith("B:") for l in v)]
        reports.append({
            "gene": a_key if a_key == b_key else "%s=%s" % (a_key, b_key),
            "matched_by": how,
            "chrom": a_txs[0].chrom,
            "strand": a_txs[0].strand,
            "identical_groups": sorted(identical_groups),
            "different": [(x, y, r) for x, y, v, r in pairs if v == DIFFERENT],
            "insufficient": [(x, y, r) for x, y, v, r in pairs
                             if v == INSUFFICIENT],
        })

    for key in only_a:
        reports.append({"gene": key, "matched_by": "none",
                        "chrom": None, "strand": None,
                        "identical_groups": [], "different": [],
                        "insufficient": [],
                        "note": "gene only in annotation A"})
    for key in only_b:
        reports.append({"gene": key, "matched_by": "none",
                        "chrom": None, "strand": None,
                        "identical_groups": [], "different": [],
                        "insufficient": [],
                        "note": "gene only in annotation B"})
    return reports


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def write_report(reports, out):
    n_id = n_diff = n_ins = 0
    for rep in reports:
        header = "## gene %s" % rep["gene"]
        if rep["chrom"]:
            header += " (%s, strand %s, matched %s)" % (
                rep["chrom"], rep["strand"], rep["matched_by"])
        out.write(header + "\n")
        if rep.get("note"):
            out.write("  %s\n" % rep["note"])
            continue
        for grp in rep["identical_groups"]:
            n_id += 1
            out.write("  IDENTICAL group: %s\n" % ", ".join(grp))
        for x, y, reason in rep["different"]:
            n_diff += 1
            out.write("  DIFFERENT: %s vs %s -- %s\n" % (x, y, reason))
        for x, y, reason in rep["insufficient"]:
            n_ins += 1
            out.write("  INSUFFICIENT: %s vs %s -- %s\n" % (x, y, reason))
        if not (rep["identical_groups"] or rep["different"]
                or rep["insufficient"]):
            out.write("  (no cross-annotation transcript pairs)\n")
    out.write("\nsummary: %d identical group(s), %d clearly different pair(s), "
              "%d insufficient-evidence pair(s)\n" % (n_id, n_diff, n_ins))


def write_tsv(reports, path):
    with open(path, "w") as fh:
        fh.write("gene\tchrom\tstrand\trelation\ttranscript_A\ttranscript_B"
                 "\tdetail\n")
        for rep in reports:
            for grp in rep["identical_groups"]:
                a = [l for l in grp if l.startswith("A:")]
                b = [l for l in grp if l.startswith("B:")]
                fh.write("%s\t%s\t%s\tIDENTICAL\t%s\t%s\t%s\n" % (
                    rep["gene"], rep["chrom"], rep["strand"],
                    ",".join(a), ",".join(b),
                    "same exon chain and junction path, all ends known"))
            for relation, key in (("DIFFERENT", "different"),
                                  ("INSUFFICIENT", "insufficient")):
                for x, y, reason in rep[key]:
                    fh.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (
                        rep["gene"], rep["chrom"], rep["strand"],
                        relation, x, y, reason))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("annotation_a", help="annotation A (GFF3 or GTF)")
    p.add_argument("annotation_b", help="annotation B (GFF3 or GTF)")
    p.add_argument("--incomplete-a", metavar="TSV",
                   help="transcripts of A with unknown 5'/3' ends")
    p.add_argument("--incomplete-b", metavar="TSV",
                   help="transcripts of B with unknown 5'/3' ends")
    p.add_argument("-o", "--tsv", metavar="OUT.tsv",
                   help="also write machine-readable TSV report")
    args = p.parse_args(argv)

    db_a = load_db(args.annotation_a)
    db_b = load_db(args.annotation_b)
    a_models = build_transcripts(db_a, "A", read_incomplete(args.incomplete_a))
    b_models = build_transcripts(db_b, "B", read_incomplete(args.incomplete_b))
    sys.stderr.write("loaded %d transcripts from A, %d from B\n"
                     % (len(a_models), len(b_models)))

    reports = compare_annotations(a_models, b_models)
    write_report(reports, sys.stdout)
    if args.tsv:
        write_tsv(reports, args.tsv)
        sys.stderr.write("TSV report written to %s\n" % args.tsv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
