# -*- coding: utf-8 -*-
"""Tests for allele-level provenance when record IDs are not unique.

Splitting one row produces many rows with the same ID; merging rows combines
their IDs.  The tests therefore use record keys/positions and allele values,
never IDs, to decide which annotation belongs to which output allele.
"""

import pytest

import vcfpy
from vcfpy import allele, record


def make_record(
    alts=("C", "G"),
    info=None,
    calls=None,
    format_=None,
    id_=("rs1",),
    pos=100,
):
    calls = calls or []
    return vcfpy.Record(
        "1",
        pos,
        list(id_),
        "A",
        [vcfpy.Substitution(vcfpy.SNV, alt) for alt in alts],
        50,
        ["PASS"],
        {} if info is None else info,
        format_ if format_ is not None else (["GT"] if calls else None),
        calls or None,
    )


# ---------------------------------------------------------------------------
# Re-ordering the ALT column


def test_reorder_alts_remaps_every_allele_indexed_value():
    rec = make_record(
        info={"AF": [0.1, 0.2], "AC": [3, 4], "AD": [10, 3, 4], "DP": 100},
        format_=["GT", "AD", "PL", "GL"],
        calls=[
            vcfpy.Call("s1", {"GT": "1/2", "AD": [10, 3, 4], "PL": [0, 10, 20, 30, 40, 50], "GL": [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0]}),
            vcfpy.Call("s2", {"GT": "0|1", "AD": [10, 3, 4], "PL": [0, 10, 20, 30, 40, 50], "GL": [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0]}),
        ],
    )
    rec.reorder_alts([1, 0], header=None)

    assert [alt.serialize() for alt in rec.ALT] == ["G", "C"]
    # Number=A follows the new ALT order.
    assert rec.INFO["AF"] == [0.2, 0.1]
    assert rec.INFO["AC"] == [4, 3]
    # Number=R keeps REF first and reorders the ALT part.
    assert rec.INFO["AD"] == [10, 4, 3]
    # Number=1 is untouched.
    assert rec.INFO["DP"] == 100

    s1 = rec.call_for_sample["s1"]
    assert s1.data["GT"] == "2/1"
    assert s1.data["AD"] == [10, 4, 3]
    # Number=G order is 0/0, 0/1, 1/1, 0/2, 1/2, 2/2.
    assert s1.data["PL"] == [0, 30, 50, 10, 40, 20]
    assert s1.data["GL"] == [-5.0, -2.0, 0.0, -4.0, -1.0, -3.0]
    # Genotype parsing is refreshed after the genotype string changed.
    assert s1.gt_alleles == [2, 1]
    assert s1.gt_bases == ("C", "G")
    assert rec.call_for_sample["s2"].data["GT"] == "0|2"
    assert rec.call_for_sample["s2"].is_phased


def test_reorder_alts_rejects_non_permutations():
    rec = make_record()
    with pytest.raises(ValueError, match="permutation"):
        rec.reorder_alts([0])
    with pytest.raises(ValueError, match="permutation"):
        rec.reorder_alts([0, 0])
    with pytest.raises(ValueError, match="permutation"):
        rec.reorder_alts([0, 2])


def test_reorder_alts_does_not_fall_back_to_old_positions():
    # A Number=A field with the wrong length cannot be mapped positionally.
    rec = make_record(info={"AF": [0.1, 0.2, 0.3]}, format_=None, calls=None)
    rec.record_key = ("file", 7)
    rec.allele_mapping = vcfpy.AlleleMapping("v1")

    rec.reorder_alts([1, 0], header=None)

    assert [alt.serialize() for alt in rec.ALT] == ["G", "C"]
    # The value is left untouched and reported as unmatched ...
    assert rec.INFO["AF"] == [0.1, 0.2, 0.3]
    assert len(rec.allele_mapping.unmatched) == 1
    entry = rec.allele_mapping.unmatched[0]
    assert entry.field == "INFO/AF"
    assert "expected 2 Number=A values" in entry.reason
    # ... and is not attached to a new allele by its old index.
    assert entry.output_key.allele_index == 0


def test_reorder_alts_leaves_non_allele_fields_alone():
    rec = make_record(info={"DP": 12, "AF": [0.1, 0.2]}, format_=None, calls=None)
    rec.reorder_alts([1, 0], header=None)
    assert rec.INFO == {"DP": 12, "AF": [0.2, 0.1]}


def test_reorder_alts_with_unparsed_calls_is_refused():
    rec = vcfpy.Record(
        "1",
        100,
        ["rs1"],
        "A",
        [vcfpy.Substitution(vcfpy.SNV, "C"), vcfpy.Substitution(vcfpy.SNV, "G")],
        50,
        ["PASS"],
        {},
        ["GT"],
        [vcfpy.UnparsedCall("s1", "1/2")],
    )
    with pytest.raises(ValueError, match="UnparsedCall"):
        rec.reorder_alts([1, 0])


def test_reorder_alts_permutes_the_allele_mapping():
    rec = make_record(format_=None, calls=None)
    rec.record_key = "row-1"
    mapping = vcfpy.AlleleMapping("v1")
    mapping.add(vcfpy.AlleleKey("row-1", 0), vcfpy.AlleleKey("orig", 0))
    mapping.add(vcfpy.AlleleKey("row-1", 1), vcfpy.AlleleKey("orig", 1))
    mapping.add(vcfpy.AlleleKey("row-1", 2), vcfpy.AlleleKey("orig", 2))
    rec.allele_mapping = mapping

    rec.reorder_alts([1, 0], header=None)

    assert rec.allele_mapping.sources_for(vcfpy.AlleleKey("row-1", 1))[0].key == vcfpy.AlleleKey("orig", 2)
    assert rec.allele_mapping.sources_for(vcfpy.AlleleKey("row-1", 2))[0].key == vcfpy.AlleleKey("orig", 1)
    assert rec.allele_mapping.sources_for(vcfpy.AlleleKey("row-1", 0))[0].key == vcfpy.AlleleKey("orig", 0)


# ---------------------------------------------------------------------------
# Splitting


def test_split_multi_keeps_ids_but_maps_by_allele():
    rec = make_record(
        info={"AF": [0.1, 0.2], "AD": [10, 3, 4]},
        format_=["GT", "AD"],
        calls=[vcfpy.Call("s1", {"GT": "1/2", "AD": [10, 3, 4]})],
    )
    rec.record_key = ("sample.vcf", 42)

    split, mapping = rec.split(header=None)

    # Both rows keep the same VCF ID: the ID cannot identify them.
    assert [row.ID for row in split] == [["rs1"], ["rs1"]]
    assert [row.POS for row in split] == [100, 100]
    assert [[alt.serialize() for alt in row.ALT] for row in split] == [["C"], ["G"]]
    assert [row.record_key for row in split] == [("sample.vcf", 42, 1), ("sample.vcf", 42, 2)]

    assert split[0].INFO == {"AF": [0.1], "AD": [10, 3]}
    assert split[1].INFO == {"AF": [0.2], "AD": [10, 4]}
    assert split[0].calls[0].data["GT"] == "1/0"
    assert split[1].calls[0].data["GT"] == "0/1"

    # REF stays REF in every split row; each ALT maps to its original ALT.
    assert mapping.sources_for(vcfpy.AlleleKey(split[0].record_key, 0))[0].key == vcfpy.AlleleKey(("sample.vcf", 42), 0)
    assert mapping.sources_for(vcfpy.AlleleKey(split[0].record_key, 1))[0].key == vcfpy.AlleleKey(("sample.vcf", 42), 1)
    assert mapping.sources_for(vcfpy.AlleleKey(split[1].record_key, 0))[0].key == vcfpy.AlleleKey(("sample.vcf", 42), 0)
    assert mapping.sources_for(vcfpy.AlleleKey(split[1].record_key, 1))[0].key == vcfpy.AlleleKey(("sample.vcf", 42), 2)
    # The original multi-allelic ALT 1 is used by both output rows.
    assert len(mapping.outputs_for(vcfpy.AlleleKey(("sample.vcf", 42), 0))) == 2


def test_split_reports_alleles_it_cannot_represent_in_a_row():
    rec = make_record(
        format_=["GT"],
        calls=[vcfpy.Call("s1", {"GT": "1/2"})],
    )
    split, mapping = rec.split(header=None)

    # The projected genotype is written, but the dropped allele is reported
    # rather than silently lost.
    assert split[0].calls[0].data["GT"] == "1/0"
    reasons = [entry.reason for entry in mapping.unmatched if entry.field == "FORMAT/GT"]
    assert any("ALT allele(s) 2" in reason for reason in reasons)
    assert any("ALT allele(s) 1" in reason for reason in reasons)


def test_split_multi_without_alts_returns_no_records():
    rec = make_record(alts=(), format_=None, calls=None)
    split, mapping = rec.split(header=None)
    assert split == []
    assert len(mapping) == 0


# ---------------------------------------------------------------------------
# Merging


def make_split_row(alt, ad, af, gt, sample="s1", id_=("rs1",), pos=100):
    return make_record(
        alts=(alt,),
        info={"AF": [af], "AD": [10, ad]},
        format_=["GT", "AD", "PL"],
        calls=[
            vcfpy.Call(
                sample,
                {"GT": gt, "AD": [10, ad], "PL": [0, 10, 20]},
            )
        ],
        id_=id_,
        pos=pos,
    )


def test_merge_multi_maps_each_output_allele_to_all_sources():
    first = make_split_row("C", 3, 0.1, "0/1")
    first.record_key = ("rows", 1)
    second = make_split_row("G", 4, 0.2, "0/1")
    second.record_key = ("rows", 2)

    merged, mapping = vcfpy.merge_multi([first, second], header=None, annotation_version="v2")

    assert merged.ID == ["rs1"]
    assert [alt.serialize() for alt in merged.ALT] == ["C", "G"]
    assert merged.INFO["AD"] == [10, 3, 4]
    assert merged.INFO["AF"] == [0.1, 0.2]
    assert merged.record_key == ("merged", ("rows", 1), ("rows", 2))

    ref_sources = mapping.sources_for(vcfpy.AlleleKey(merged.record_key, 0))
    assert {source.key for source in ref_sources} == {
        vcfpy.AlleleKey(("rows", 1), 0),
        vcfpy.AlleleKey(("rows", 2), 0),
    }
    assert mapping.sources_for(vcfpy.AlleleKey(merged.record_key, 1))[0].key == vcfpy.AlleleKey(("rows", 1), 1)
    assert mapping.sources_for(vcfpy.AlleleKey(merged.record_key, 2))[0].key == vcfpy.AlleleKey(("rows", 2), 1)
    assert mapping.annotation_version == "v2"
    assert mapping.records[("rows", 1)].annotation_version == "v2"


def test_merge_multi_combines_ids_and_genotypes():
    first = make_split_row("C", 3, 0.1, "0/1")
    first.ID = ["rs1", "shared"]
    second = make_split_row("G", 4, 0.2, "1/1")
    second.ID = ["shared", "rs2"]
    first.record_key = ("rows", 1)
    second.record_key = ("rows", 2)

    merged, mapping = vcfpy.merge_multi([first, second], header=None)

    # IDs are unioned, not used for identity.
    assert merged.ID == ["rs1", "shared", "rs2"]
    assert merged.calls[0].data["GT"] == "1/2"
    assert mapping.unmatched == []


def test_merge_multi_reuses_identical_alts_instead_of_duplicating():
    first = make_split_row("C", 3, 0.1, "0/1")
    second = make_split_row("C", 3, 0.1, "0/1")
    first.record_key = ("rows", 1)
    second.record_key = ("rows", 2)

    merged, mapping = vcfpy.merge_multi([first, second], header=None)

    assert [alt.serialize() for alt in merged.ALT] == ["C"]
    sources = mapping.sources_for(vcfpy.AlleleKey(merged.record_key, 1))
    assert {source.key for source in sources} == {
        vcfpy.AlleleKey(("rows", 1), 1),
        vcfpy.AlleleKey(("rows", 2), 1),
    }


def test_merge_multi_requires_same_locus_and_ref():
    first = make_split_row("C", 3, 0.1, "0/1")
    second = make_split_row("G", 4, 0.2, "0/1", pos=101)
    with pytest.raises(ValueError, match="same locus"):
        vcfpy.merge_multi([first, second], header=None)

    same_pos = make_split_row("G", 4, 0.2, "0/1")
    same_pos.REF = "T"
    with pytest.raises(ValueError, match="identical REF"):
        vcfpy.merge_multi([first, same_pos], header=None)


def test_merge_multi_reports_conflicting_values_instead_of_picking_by_position():
    first = make_split_row("C", 3, 0.1, "0/1")
    second = make_split_row("C", 99, 0.1, "0/1")
    first.record_key = ("rows", 1)
    second.record_key = ("rows", 2)

    merged, mapping = vcfpy.merge_multi([first, second], header=None)

    # First value wins, the disagreement is listed separately.
    assert merged.INFO["AD"] == [10, 3]
    assert any(entry.field == "INFO/AD" for entry in mapping.unmatched)
    conflict = next(entry for entry in mapping.unmatched if entry.field == "INFO/AD")
    assert conflict.value == {"kept": 3, "discarded": 99}

    with pytest.raises(ValueError, match="INFO/AD"):
        vcfpy.merge_multi([first, second], header=None, on_conflict="error")


# ---------------------------------------------------------------------------
# Inference from already transformed records


def test_infer_mapping_uses_values_not_ids_or_positions():
    source = make_record(
        alts=("C", "G"),
        info={"AF": [0.1, 0.2]},
        format_=["GT"],
        calls=[vcfpy.Call("s1", {"GT": "1/2"})],
    )
    source.record_key = "source"
    output = make_record(
        alts=("G", "C"),  # re-ordered ALT column
        info={"AF": [0.2, 0.1]},
        format_=["GT"],
        calls=[vcfpy.Call("s1", {"GT": "2/1"})],
    )
    output.record_key = "output"

    mapping = vcfpy.infer_mapping([source], [output])

    assert mapping.sources_for(vcfpy.AlleleKey("output", 1))[0].key == vcfpy.AlleleKey("source", 2)
    assert mapping.sources_for(vcfpy.AlleleKey("output", 2))[0].key == vcfpy.AlleleKey("source", 1)
    assert mapping.unmatched == []
    # VCF IDs were identical and therefore useless, but they are preserved.
    assert mapping.record_ids_for("source") == ("rs1",)


def test_infer_mapping_reports_both_directions_of_unmatched_alleles():
    source = make_record(alts=("C", "G"), format_=None, calls=None)
    source.record_key = "source"
    output = make_record(alts=("C", "T"), format_=None, calls=None)
    output.record_key = "output"

    mapping = vcfpy.infer_mapping([source], [output])

    output_unmatched = [entry for entry in mapping.unmatched if entry.output_key is not None]
    source_unmatched = [entry for entry in mapping.unmatched if entry.source_key is not None]
    assert [entry.value for entry in output_unmatched] == ["T"]
    assert [entry.value for entry in source_unmatched] == ["G"]


def test_infer_mapping_keeps_annotation_versions_per_source():
    first = make_record(alts=("C",), format_=None, calls=None)
    first.record_key = "first"
    second = make_record(alts=("C",), format_=None, calls=None)
    second.record_key = "second"
    merged = make_record(alts=("C",), format_=None, calls=None)
    merged.record_key = "merged"

    mapping = vcfpy.infer_mapping(
        [first, second],
        [merged],
        source_versions={"first": "v1", "second": "v2"},
        annotation_version="v3",
    )

    by_source = {source.key: source.annotation_version for source in mapping.sources_for(vcfpy.AlleleKey("merged", 1))}
    assert by_source == {
        vcfpy.AlleleKey("first", 1): "v1",
        vcfpy.AlleleKey("second", 1): "v2",
    }
    assert mapping.annotation_version == "v3"
    assert not mapping.is_version_consistent

    mapping.add_unmatched("ALT", "T", "no source", annotation_version="v3")
    exported = mapping.to_dict()
    assert exported["unmatched"][0]["value"] == "T"
    assert "merged#1" in exported["outputs"]


def test_split_then_merge_restores_alleles_and_annotations():
    original = make_record(
        info={"AF": [0.1, 0.2], "AD": [10, 3, 4]},
        format_=["GT", "AD"],
        calls=[vcfpy.Call("s1", {"GT": "0/2", "AD": [10, 3, 4]})],
    )
    original.record_key = "original"

    split, _ = original.split(header=None)
    merged, mapping = vcfpy.merge_multi(split, header=None)

    assert [alt.serialize() for alt in merged.ALT] == ["C", "G"]
    assert merged.INFO == {"AF": [0.1, 0.2], "AD": [10, 3, 4]}
    assert merged.calls[0].data["AD"] == [10, 3, 4]
    assert merged.calls[0].gt_bases == ("A", "G")
    assert mapping.unmatched == []


def test_allele_mapping_copy_and_permute_are_independent():
    mapping = vcfpy.AlleleMapping("v1")
    mapping.add(vcfpy.AlleleKey("row", 0), vcfpy.AlleleKey("src", 0), annotation="ref")
    mapping.add(vcfpy.AlleleKey("row", 1), vcfpy.AlleleKey("src", 1), annotation="alt1")
    mapping.add_unmatched("INFO/X", 1, "reason", output_key=vcfpy.AlleleKey("row", 1))

    copied = mapping.copy()
    permuted = copied.permute_outputs("row", [0])

    assert len(permuted) == 2
    assert permuted.sources_for(vcfpy.AlleleKey("row", 1))[0].annotation == "alt1"
    assert permuted.unmatched[0].output_key == vcfpy.AlleleKey("row", 1)
    # The original and the copy are not modified.
    assert len(mapping) == 2
    assert len(copied) == 2


def test_gt_remap_preserves_phase_and_missing_values():
    assert allele.remap_gt("0|2", [1, 0]) == ("0|1", None)
    assert allele.remap_gt("./.", [1, 0]) == ("./.", None)
    assert allele.remap_gt("3", [1, 0])[0] is None
    assert "outside" in allele.remap_gt("3", [1, 0])[1]


def test_record_merge_classmethod_delegates():
    first = make_split_row("C", 3, 0.1, "0/1")
    second = make_split_row("G", 4, 0.2, "0/1")
    first.record_key = ("rows", 1)
    second.record_key = ("rows", 2)
    merged, mapping = record.Record.merge([first, second], header=None)
    assert [alt.serialize() for alt in merged.ALT] == ["C", "G"]
    assert isinstance(mapping, vcfpy.AlleleMapping)


def make_header():
    return vcfpy.Header(
        lines=[
            vcfpy.HeaderLine("fileformat", "VCFv4.3"),
            vcfpy.FormatHeaderLine.from_mapping(
                {"ID": "GT", "Number": "1", "Type": "String", "Description": "Genotype"}
            ),
            vcfpy.FormatHeaderLine.from_mapping(
                {"ID": "AD", "Number": "R", "Type": "Integer", "Description": "Allele depth"}
            ),
            vcfpy.FormatHeaderLine.from_mapping(
                {"ID": "PL", "Number": "G", "Type": "Integer", "Description": "Likelihoods"}
            ),
            vcfpy.InfoHeaderLine.from_mapping(
                {"ID": "AF", "Number": "A", "Type": "Float", "Description": "Allele frequency"}
            ),
            vcfpy.InfoHeaderLine.from_mapping(
                {"ID": "XX", "Number": "R", "Type": "Integer", "Description": "Per allele"}
            ),
        ],
        samples=vcfpy.SamplesInfos(["s1", "s2"]),
    )


def test_header_declared_numbers_drive_split_and_reorder():
    rec = vcfpy.Record(
        "1",
        100,
        ["rs1"],
        "A",
        [vcfpy.Substitution(vcfpy.SNV, "C"), vcfpy.Substitution(vcfpy.SNV, "G")],
        50,
        ["PASS"],
        {"AF": [0.25, 0.75], "XX": [9, 3, 4]},
        ["GT", "AD", "PL"],
        [
            vcfpy.Call("s1", {"GT": "1/2", "AD": [9, 3, 4], "PL": [0, 10, 20, 30, 40, 50]}),
            vcfpy.Call("s2", {"GT": "0/1", "AD": [9, 3, 4], "PL": [0, 10, 20, 30, 40, 50]}),
        ],
    )
    rec.record_key = "row"
    header = make_header()

    split, _ = rec.split(header=header)
    assert split[0].INFO == {"AF": [0.25], "XX": [9, 3]}
    assert split[1].INFO == {"AF": [0.75], "XX": [9, 4]}

    # Re-order the two-row-free record: single ALT is a valid no-op.
    split[1].reorder_alts([0], header=header)
    assert split[1].INFO == {"AF": [0.75], "XX": [9, 4]}
    assert split[1].calls[0].data["AD"] == [9, 4]


def test_split_then_reorder_then_merge_tracks_versions_and_real_alleles():
    rec = make_record(
        info={"AF": [0.1, 0.2], "AD": [10, 3, 4]},
        format_=["GT", "AD"],
        calls=[vcfpy.Call("s1", {"GT": "1/2", "AD": [10, 3, 4]})],
    )
    rec.record_key = ("orig", 1)
    header = None

    split, split_mapping = rec.split(header=header, annotation_version="2026-11-01")
    assert len(split) == 2
    # Re-order the first split row (no-op) and rebuild a mapping from it.
    split[0].reorder_alts([0], header=header)
    assert split[0].allele_mapping.annotation_version == "2026-11-01"

    merged, merged_mapping = vcfpy.merge_multi(
        split,
        header=header,
        annotation_version="2026-11-02",
    )
    assert [alt.serialize() for alt in merged.ALT] == ["C", "G"]
    assert merged.calls[0].data["GT"] == "1/2"
    assert merged.calls[0].data["AD"] == [10, 3, 4]
    # Output allele 1 came from source row 1's ALT, allele 2 from row 2's ALT.
    assert merged_mapping.sources_for(vcfpy.AlleleKey(merged.record_key, 1))[0].key == vcfpy.AlleleKey(
        split[0].record_key, 1
    )
    assert merged_mapping.sources_for(vcfpy.AlleleKey(merged.record_key, 2))[0].key == vcfpy.AlleleKey(
        split[1].record_key, 1
    )
    assert merged_mapping.annotation_version == "2026-11-02"
    # The rows were annotated at the earlier split version; that is preserved.
    assert {source.annotation_version for source in merged_mapping.sources_for(vcfpy.AlleleKey(merged.record_key, 1))} == {
        "2026-11-01"
    }
    assert not merged_mapping.is_version_consistent


def test_merge_refuses_unparsed_calls_unless_asked_to_keep_them():
    first = make_split_row("C", 3, 0.1, "0/1")
    second = make_split_row("G", 4, 0.2, "0/1")
    first.record_key = ("rows", 1)
    second.record_key = ("rows", 2)
    first.calls = [vcfpy.UnparsedCall("s1", "0/1:3")]
    first.update_calls(first.calls)

    with pytest.raises(ValueError, match="UnparsedCall"):
        vcfpy.merge_multi([first, second], header=None)

    merged, mapping = vcfpy.merge_multi([first, second], header=None, on_unparsed="keep")
    assert any(entry.field == "FORMAT" for entry in mapping.unmatched)


def test_merge_rejects_different_chrom_and_duplicate_source_keys():
    first = make_split_row("C", 3, 0.1, "0/1")
    second = make_split_row("G", 4, 0.2, "0/1")
    first.record_key = "same"
    second.record_key = "same"
    with pytest.raises(ValueError, match="unique"):
        vcfpy.merge_multi([first, second], header=None)

    first.record_key = "a"
    second.record_key = "b"
    second.CHROM = "2"
    with pytest.raises(ValueError, match="same locus"):
        vcfpy.merge_multi([first, second], header=None)


def test_merge_default_source_keys_do_not_collide_for_content_identical_rows():
    first = make_split_row("C", 3, 0.1, "0/1")
    second = make_split_row("C", 3, 0.1, "0/1")
    first.record_key = None
    second.record_key = None
    merged, mapping = vcfpy.merge_multi([first, second], header=None)
    assert len(mapping.sources_for(vcfpy.AlleleKey(merged.record_key, 1))) == 2


def test_unmatched_values_are_listed_separately_and_json_serializable():
    mapping = vcfpy.AlleleMapping("v1")
    mapping.report_unmatched(
        "INFO/AF",
        [0.1, 0.2, 0.3],
        "count mismatch",
        output_key=vcfpy.AlleleKey("out", 0),
        annotation_version="v1",
    )
    exported = mapping.to_dict()
    assert exported["unmatched"] == [
        {
            "field": "INFO/AF",
            "value": [0.1, 0.2, 0.3],
            "reason": "count mismatch",
            "output": "out#0",
            "source": None,
            "sample": None,
            "annotation_version": "v1",
        }
    ]
    assert mapping.unmatched_values == tuple(mapping.unmatched)


def test_reorder_alts_creates_a_trace_without_an_existing_mapping():
    rec = make_record(info={"AF": [0.1, 0.2]}, format_=None, calls=None)
    rec.reorder_alts([1, 0], header=None)
    assert rec.record_key is not None
    assert rec.allele_mapping is not None
    assert rec.allele_mapping.sources_for(vcfpy.AlleleKey(rec.record_key, 1))[0].key == vcfpy.AlleleKey(
        rec.record_key, 2
    )


def test_reorder_alts_preserves_unmapped_inputs_in_place():
    rec = make_record(
        info={"AF": [0.1, 0.2]},
        format_=["GT", "AD"],
        calls=[vcfpy.Call("s1", {"GT": "0/2", "AD": [10, 3, 4]})],
    )
    rec.reorder_alts([1, 0], header=None)
    # GT 0/2 -> 0/1 (old ALT 2 becomes new ALT 1), and AD follows the ALTs.
    assert rec.calls[0].data["GT"] == "0/1"
    assert rec.calls[0].data["AD"] == [10, 4, 3]
    assert rec.calls[0].gt_bases == ("A", "G")


def test_merge_reports_mixed_phasing_instead_of_inventing_phase():
    first = make_split_row("C", 3, 0.1, "0|1")
    second = make_split_row("G", 4, 0.2, "0/1")
    first.record_key = ("rows", 1)
    second.record_key = ("rows", 2)

    merged, mapping = vcfpy.merge_multi([first, second], header=None)
    assert merged.calls[0].data["GT"] == "0|1"
    assert any(
        entry.field == "FORMAT/GT" and "phase" in entry.reason for entry in mapping.unmatched
    )


def test_infer_mapping_disambiguates_repeated_default_keys():
    left = make_record(alts=("C",), format_=None, calls=None)
    right = make_record(alts=("C",), format_=None, calls=None)
    merged = make_record(alts=("C",), format_=None, calls=None)
    mapping = vcfpy.infer_mapping([left, right], [merged])

    output_key = mapping.output_keys[0]
    sources = mapping.sources_for(output_key)
    # Both content-identical sources map onto the single output ALT, and the
    # repeated content-based keys were disambiguated rather than merged.
    assert len(sources) == 2
    assert len({source.key.record_key for source in sources}) == 2
    assert not mapping.unmatched


def test_infer_mapping_rejects_repeated_explicit_keys():
    left = make_record(alts=("C",), format_=None, calls=None)
    right = make_record(alts=("C",), format_=None, calls=None)
    merged = make_record(alts=("C",), format_=None, calls=None)
    with pytest.raises(ValueError, match="source record keys must be unique"):
        vcfpy.infer_mapping([left, right], [merged], source_keys=["same", "same"])


def test_split_reports_number_g_values_that_cannot_be_represented():
    rec = make_record(
        format_=["GT", "PL"],
        calls=[vcfpy.Call("s1", {"GT": "1/2", "PL": [0, 10, 20, 30, 40, 50]})],
    )
    split, mapping = rec.split(header=None)

    # The per-row likelihoods are projected, but genotypes combining both ALTs
    # cannot be kept in either single-ALT row.
    assert split[0].calls[0].data["PL"] == [0, 10, 20]
    assert split[1].calls[0].data["PL"] == [0, 30, 50]
    lossy = [entry for entry in mapping.unmatched if entry.field == "FORMAT/PL"]
    assert len(lossy) == 1
    assert lossy[0].value == [0, 10, 20, 30, 40, 50]
    assert "combining several ALT alleles" in lossy[0].reason


def test_reorder_alts_remaps_number_g_using_actual_ploidy():
    rec = make_record(
        format_=["GT", "PL"],
        calls=[vcfpy.Call("s1", {"GT": "1", "PL": [0, 10, 20]})],
    )
    rec.reorder_alts([1, 0], header=None)
    # Haploid G order is 0, 1, 2 and must not be treated as diploid.
    assert rec.calls[0].data["PL"] == [0, 20, 10]
