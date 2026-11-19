#!/usr/bin/env python
"""Select representative transcript(s) for each gene in a GFF3/GTF file.

Selection policy (per gene):

1. Candidates are first filtered by structural integrity. Coding
   transcripts with missing CDS phase (``缺相位``) or incomplete
   boundaries (``边界不完整``: no start_codon and/or stop_codon, when the
   file carries such features at all) are *not* eligible. They are
   reported separately in the excluded-candidates table.
2. Eligible candidates are ordered by the configured source priority
   (``指定来源等级``) and coding completeness (complete coding before
   non-coding); transcript length is only the final tie-breaker.
   Because length ranks last, an abnormally long transcript (e.g. a
   readthrough) can never outrank a complete, slightly shorter formal
   record from a better source.
3. Ties are kept: every candidate equal on (source rank, completeness,
   length) is reported as a representative (``并列保留``).

Coverage accounting: exons shared between sibling transcripts (or
duplicated within one transcript) are merged before counting, so shared
sub-items contribute to the actual coverage exactly once
(``共享子项只计实际覆盖一次``).

Outputs (per --out-prefix):
  <prefix>.representatives.tsv  one row per representative transcript
  <prefix>.excluded.tsv         one row per structurally incomplete candidate

Example:
    python representative_transcript.py annotation.gff3 \
        --source-priority ensembl,refseq,augustus \
        --out-prefix output/rep
"""

import argparse
import sys

import gffutils

TRANSCRIPT_TYPES = {
    "mRNA", "transcript", "ncRNA", "lncRNA", "lnc_RNA", "miRNA", "rRNA",
    "tRNA", "snRNA", "snoRNA", "scRNA", "pseudogenic_transcript",
    "primary_transcript", "V_gene_segment", "C_gene_segment",
}
GENE_TYPES = ("gene", "pseudogene")
VALID_PHASES = {"0", "1", "2"}

COMPLETENESS_RANK = {"complete_coding": 0, "non_coding": 1}


def merged_bp(intervals):
    """Total bp covered by intervals, counting overlaps only once."""
    total = 0
    last_start = last_end = None
    for start, end in sorted(intervals):
        if last_start is None:
            last_start, last_end = start, end
        elif start > last_end:
            total += last_end - last_start + 1
            last_start, last_end = start, end
        else:
            last_end = max(last_end, end)
    if last_start is not None:
        total += last_end - last_start + 1
    return total


def load_db(path):
    return gffutils.create_db(
        path, dbfn=":memory:", force=True, keep_order=True,
        merge_strategy="merge", sort_attribute_values=True,
    )


def children_of_type(db, feature, types):
    return [c for c in db.children(feature) if c.featuretype in types]


class Candidate:
    def __init__(self, db, tx, boundary_assessable, extra_children=None):
        self.id = tx.id
        self.source = tx.source or "."
        self.featuretype = tx.featuretype
        extra_children = extra_children or {}

        def kids(types):
            found = {}
            for c in children_of_type(db, tx, types):
                found[c.id] = c
            for t in types:
                for c in extra_children.get(t, []):
                    found.setdefault(c.id, c)
            return list(found.values())

        exons = kids({"exon"})
        cds = kids({"CDS"})
        starts = kids({"start_codon"})
        stops = kids({"stop_codon"})
        self.n_exons = len(exons)
        self.n_cds = len(cds)
        self.exon_intervals = [(e.start, e.end) for e in exons]
        self.tx_len = merged_bp(self.exon_intervals)
        self.cds_len = merged_bp([(c.start, c.end) for c in cds])
        if self.tx_len == 0:
            # No exon records: fall back to CDS union, then feature span.
            self.tx_len = self.cds_len or (tx.end - tx.start + 1)
        self.has_cds = bool(cds)
        self.missing_phase = self.has_cds and any(
            str(c.frame) not in VALID_PHASES for c in cds
        )
        self.missing_start = self.has_cds and boundary_assessable and not starts
        self.missing_stop = self.has_cds and boundary_assessable and not stops
        self.reasons = []
        if self.missing_phase:
            self.reasons.append("missing_phase")
        if self.missing_start:
            self.reasons.append("incomplete_boundary_5p")
        if self.missing_stop:
            self.reasons.append("incomplete_boundary_3p")
        if not (exons or cds):
            self.reasons.append("no_exon_no_cds")
        self.eligible = not self.reasons
        self.boundary_assessable = boundary_assessable
        self.completeness = (
            "complete_coding" if self.has_cds else "non_coding"
        )

    def evidence(self, source_rank):
        if not (self.has_cds and self.boundary_assessable):
            boundary = "not_assessable"
        elif self.missing_start or self.missing_stop:
            boundary = "incomplete"
        else:
            boundary = "5p+3p"
        return (
            "source={source}(rank={rank});class={cls};"
            "tx_len={tx}bp;cds_len={cds}bp;exons={nex};"
            "phase={phase};boundary={bd}"
        ).format(
            source=self.source, rank=source_rank, cls=self.completeness,
            tx=self.tx_len, cds=self.cds_len, nex=self.n_exons,
            phase="ok" if not self.missing_phase else "missing", bd=boundary,
        )


def gene_transcripts(db, gene):
    """Level-1 children that look like transcripts; fall back to a
    pseudo-transcript built from the gene's own exon/CDS children."""
    txs = []
    for child in db.children(gene, level=1):
        if child.featuretype in TRANSCRIPT_TYPES:
            txs.append(child)
        elif child.featuretype not in {"exon", "CDS"} and list(
            db.children(child, featuretype=("exon", "CDS"))
        ):
            txs.append(child)
    if not txs and list(children_of_type(db, gene, {"exon", "CDS"})):
        txs.append(gene)  # exon/CDS hang directly off the gene
    return txs


def source_rank(source, priority):
    try:
        return priority.index(source)
    except ValueError:
        return len(priority)


def sort_key(cand, rank_order, priority):
    parts = {
        "source": source_rank(cand.source, priority),
        "completeness": COMPLETENESS_RANK[cand.completeness],
        "length": -cand.tx_len,
    }
    return tuple(parts[k] for k in rank_order)


def build_extra_children(db):
    """Fallback parent->children map for exon/CDS/start_codon/stop_codon.

    Real-world files are not always sanitized (e.g. ``ID=x; Parent=y``
    with a space before ``Parent``), which makes strict relationship
    queries miss children. Index those features by every whitespace-
    tolerant Parent value so they still attach to their transcript.
    """
    extra = {}
    for ftype in ("exon", "CDS", "start_codon", "stop_codon"):
        try:
            feats = db.features_of_type(ftype)
        except gffutils.exceptions.FeatureNotFoundError:
            continue
        for f in feats:
            for key in f.attributes:
                if key.strip() == "Parent":
                    for parent in f.attributes[key]:
                        bucket = extra.setdefault(parent.strip(), {})
                        bucket.setdefault(ftype, []).append(f)
    return extra


def analyze(db, priority, rank_order):
    """Returns (gene_rows, excluded_rows). gene_rows has one entry per
    representative transcript (ties included)."""
    boundary_assessable = bool(
        {"start_codon", "stop_codon"} & set(db.featuretypes())
    )
    extra_children = build_extra_children(db)
    gene_rows, excluded_rows = [], []
    genes = list(db.features_of_type(GENE_TYPES))
    for gene in sorted(genes, key=lambda g: (g.seqid, g.start, g.id)):
        candidates = [
            Candidate(db, tx, boundary_assessable,
                      extra_children.get(tx.id))
            for tx in gene_transcripts(db, gene)
        ]
        eligible = [c for c in candidates if c.eligible]
        excluded = [c for c in candidates if not c.eligible]
        union_bp = merged_bp(
            iv for c in candidates for iv in c.exon_intervals
        )
        sum_bp = sum(merged_bp(c.exon_intervals) for c in candidates)
        for c in excluded:
            excluded_rows.append({
                "gene_id": gene.id,
                "transcript_id": c.id,
                "source": c.source,
                "source_rank": source_rank(c.source, priority),
                "tx_len_bp": c.tx_len,
                "cds_len_bp": c.cds_len,
                "n_exons": c.n_exons,
                "reasons": ";".join(c.reasons),
                "detail": c.evidence(source_rank(c.source, priority)),
            })
        if eligible:
            best = min(sort_key(c, rank_order, priority) for c in eligible)
            reps = [
                c for c in eligible
                if sort_key(c, rank_order, priority) == best
            ]
        else:
            reps = []
        if not reps:
            gene_rows.append({
                "gene_id": gene.id, "seqid": gene.seqid,
                "strand": gene.strand, "representative_id": "NONE",
                "tied_representatives": 0, "source": ".", "source_rank": ".",
                "completeness_class": ".", "tx_len_bp": 0, "cds_len_bp": 0,
                "n_exons": 0, "n_cds": 0,
                "gene_exon_union_bp": union_bp, "all_tx_exon_sum_bp": sum_bp,
                "n_candidates": len(candidates), "n_excluded": len(excluded),
                "evidence": "no eligible candidate; see excluded table",
            })
        for rep in reps:
            gene_rows.append({
                "gene_id": gene.id, "seqid": gene.seqid,
                "strand": gene.strand, "representative_id": rep.id,
                "tied_representatives": len(reps), "source": rep.source,
                "source_rank": source_rank(rep.source, priority),
                "completeness_class": rep.completeness,
                "tx_len_bp": rep.tx_len, "cds_len_bp": rep.cds_len,
                "n_exons": rep.n_exons, "n_cds": rep.n_cds,
                "gene_exon_union_bp": union_bp, "all_tx_exon_sum_bp": sum_bp,
                "n_candidates": len(candidates), "n_excluded": len(excluded),
                "evidence": rep.evidence(source_rank(rep.source, priority)),
            })
    return gene_rows, excluded_rows


def write_tsv(path, rows, columns):
    with open(path, "w") as fh:
        fh.write("\t".join(columns) + "\n")
        for row in rows:
            fh.write("\t".join(str(row[c]) for c in columns) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", help="GFF3/GTF annotation file")
    parser.add_argument(
        "--source-priority", default="",
        help="comma-separated sources, best first; unlisted sources tie last",
    )
    parser.add_argument(
        "--rank-order", default="source,completeness,length",
        help="permutation of source,completeness,length (default: %(default)s)",
    )
    parser.add_argument("--out-prefix", default="representative")
    args = parser.parse_args(argv)

    priority = [s for s in args.source_priority.split(",") if s]
    rank_order = [s.strip() for s in args.rank_order.split(",")]
    if sorted(rank_order) != ["completeness", "length", "source"]:
        parser.error("--rank-order must be a permutation of "
                     "source,completeness,length")

    db = load_db(args.input)
    gene_rows, excluded_rows = analyze(db, priority, rank_order)

    rep_cols = [
        "gene_id", "seqid", "strand", "representative_id",
        "tied_representatives", "source", "source_rank",
        "completeness_class", "tx_len_bp", "cds_len_bp", "n_exons", "n_cds",
        "gene_exon_union_bp", "all_tx_exon_sum_bp",
        "n_candidates", "n_excluded", "evidence",
    ]
    exc_cols = [
        "gene_id", "transcript_id", "source", "source_rank",
        "tx_len_bp", "cds_len_bp", "n_exons", "reasons", "detail",
    ]
    rep_path = args.out_prefix + ".representatives.tsv"
    exc_path = args.out_prefix + ".excluded.tsv"
    write_tsv(rep_path, gene_rows, rep_cols)
    write_tsv(exc_path, excluded_rows, exc_cols)

    by_gene = {}
    for row in gene_rows:
        by_gene.setdefault(row["gene_id"], []).append(row)
    for gene_id, rows in by_gene.items():
        first = rows[0]
        reps = ";".join(r["representative_id"] for r in rows)
        print("{gene}\treps={reps}\t{ev}\tunion={u}bp,sum={s}bp"
              "\tcandidates={n},excluded={x}".format(
                  gene=gene_id, reps=reps, ev=first["evidence"],
                  u=first["gene_exon_union_bp"],
                  s=first["all_tx_exon_sum_bp"],
                  n=first["n_candidates"], x=first["n_excluded"]))
    print("\nwrote {} and {}".format(rep_path, exc_path), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
