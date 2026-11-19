# Representative transcript selection

`representative_transcript.py` picks representative transcript(s) per gene
from a GFF3/GTF file, using gffutils for parsing.

## Rules

1. **Integrity filter first** — coding transcripts with missing CDS phase
   (`missing_phase`) or incomplete boundaries (`incomplete_boundary_5p` /
   `incomplete_boundary_3p`, i.e. no start/stop codon when the file carries
   such features) are ineligible and listed separately in
   `<prefix>.excluded.tsv`.
2. **Ranking** — eligible candidates are ordered by
   `--rank-order` (default `source,completeness,length`):
   source priority (`--source-priority`, unlisted sources tie last),
   then coding completeness (`complete_coding` > `non_coding`),
   then merged exon length. Because length is only the final tie-breaker,
   an abnormally long transcript (e.g. a readthrough) can never outrank a
   complete, slightly shorter formal record from a better source.
3. **Ties kept** — all candidates equal on the full sort key are reported
   (`tied_representatives` > 1).
4. **Shared sub-items counted once** — exon intervals are merged before
   counting, both per transcript and at gene level
   (`gene_exon_union_bp` vs `all_tx_exon_sum_bp`).

## Usage

```sh
python representative_transcript.py annotation.gff3 \
    --source-priority ensembl,refseq,augustus \
    --out-prefix output/rep
```

Outputs `output/rep.representatives.tsv` (one row per representative,
with evidence: source rank, completeness class, tx/CDS length, exon
count, phase/boundary status) and `output/rep.excluded.tsv`
(incomplete candidates with reasons).

## Files

- `rep_tx_fixture.gff3` — synthetic genes covering: length tie-break (gA),
  ties kept (gB), source rank beating abnormal length (gC), incomplete
  3' boundary (gD), missing phase (gE, gF), non-coding gene (gG).
- `test_representative_transcript.py` — pytest suite for the fixture and
  the real `gffutils/test/data/FBgn0031208.gff` example.
- `output/` — reports generated from the fixture and the FlyBase example.

Note: real files are not always sanitized (e.g. `ID=x; Parent=y` with a
space before `Parent`), so children are additionally attached via a
whitespace-tolerant Parent index (`build_extra_children`).
