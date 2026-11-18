# -*- coding: utf-8 -*-
"""Tests for header-aware merging in :py:mod:`vcfpy.merge`"""

import io
import json
import warnings

import pytest

import vcfpy
from vcfpy import merge
from vcfpy.merge import (
    FieldRename,
    MergeConflictError,
    MergeError,
    RenameCollection,
    main as merge_main,
    merge_records,
    plan_merge,
)


HEADER_A = """\
##fileformat=VCFv4.3
##INFO=<ID=AF,Number=A,Type=Float,Description="Allele frequency per ALT">
##INFO=<ID=DP,Number=1,Type=Integer,Description="Total depth">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Sample depth">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1
20\t1\t.\tG\tA\t.\t.\tAF=0.5;DP=10\tGT:AD:DP\t0/1:5,5:10
"""

HEADER_B = """\
##fileformat=VCFv4.3
##INFO=<ID=AF,Number=1,Type=Float,Description="Allele frequency scalar">
##INFO=<ID=DP,Number=1,Type=Integer,Description="Total depth">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=AD,Number=A,Type=Integer,Description="Depth per ALT">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Sample depth">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS2
20\t2\t.\tG\tA\t.\t.\tAF=0.7;DP=20\tGT:AD:DP\t0/1:8:20
"""


def _read_vcf(text):
    return vcfpy.Reader.from_stream(io.StringIO(text))


def _headers():
    return _read_vcf(HEADER_A).header, _read_vcf(HEADER_B).header


def _records():
    reader_a = _read_vcf(HEADER_A)
    reader_b = _read_vcf(HEADER_B)
    return reader_a, reader_b, list(reader_a), list(reader_b)


def test_same_length_values_do_not_make_number_a_and_number_1_compatible():
    header_a, header_b = _headers()
    plan = plan_merge([header_a, header_b], source_names=["A", "B"])

    assert not plan.compatible
    assert plan.mergeable
    conflicts = {(conflict.scope, conflict.field_id): conflict for conflict in plan.conflicts}
    assert ("INFO", "AF") in conflicts
    assert conflicts[("INFO", "AF")].reason == "Number differs ('A' vs 1)"
    # Both source records have a single ALT and therefore one observed value;
    # the plan still treats the declarations as different.
    assert plan.renames[("INFO", 1, "AF")] == "AF_B"


def test_info_and_format_are_separate_scopes():
    header_a, header_b = _headers()
    plan = plan_merge([header_a, header_b], source_names=["A", "B"])

    # INFO/AF conflicts, so only INFO/AF from source B is renamed.
    assert plan.renames[("INFO", 1, "AF")] == "AF_B"
    # INFO/DP and FORMAT/DP are identical on both sides and must not be
    # renamed just because they share a name with each other.
    assert plan.renames.get("INFO", 1, "DP") is None
    assert plan.renames.get("FORMAT", 1, "DP") is None
    assert plan.renames_for(1, "FORMAT") == {"AD": "AD_B"}

    assert set(plan.info_renames) == {1}
    assert plan.info_renames[1] == {"AF": "AF_B"}
    assert plan.format_renames[1] == {"AD": "AD_B"}


def test_record_conversion_preserves_number_semantics():
    _, _, records_a, records_b = _records()
    header_a, header_b = _headers()
    plan = plan_merge([header_a, header_b], source_names=["A", "B"])

    record_a = plan.convert_record(records_a[0], "A")
    record_b = plan.convert_record(records_b[0], "B")

    # Number=A values are lists; Number=1 values are scalars.  Renaming must
    # not change either the value or its type.
    assert record_a.INFO == {"AF": [0.5], "DP": 10}
    assert record_b.INFO == {"AF_B": 0.7, "DP": 20}
    assert isinstance(record_a.INFO["AF"], list)
    assert not isinstance(record_b.INFO["AF_B"], list)

    assert record_a.FORMAT == ["GT", "AD", "DP"]
    assert record_b.FORMAT == ["GT", "AD_B", "DP"]
    assert records_b[0].calls[0].data == {"GT": "0/1", "AD": [8], "DP": 20}
    assert record_b.calls[0].data == {"GT": "0/1", "AD_B": [8], "DP": 20}


def test_merged_header_keeps_original_number_and_type():
    header_a, header_b = _headers()
    plan = plan_merge([header_a, header_b], source_names=["A", "B"])

    assert plan.header.get_info_field_info("AF").number == "A"
    assert plan.header.get_info_field_info("AF").type == "Float"
    assert plan.header.get_info_field_info("AF_B").number == 1
    assert plan.header.get_info_field_info("AF_B").type == "Float"

    assert plan.header.get_format_field_info("AD").number == "R"
    assert plan.header.get_format_field_info("AD_B").number == "A"

    # Every (scope, target ID) is defined exactly once.  INFO/DP and FORMAT/DP
    # may legitimately coexist.
    ids = [(line.key, line.mapping["ID"]) for line in plan.header.lines if line.key in ("INFO", "FORMAT")]
    assert len(ids) == len(set(ids))


def test_converted_records_round_trip_and_reparse_cleanly():
    header_a, header_b = _headers()
    _, _, records_a, records_b = _records()
    plan = plan_merge([header_a, header_b], source_names=["A", "B"])

    records = merge_records(plan, [records_a, records_b], pad=True)
    out = io.StringIO()
    writer = vcfpy.Writer.from_stream(out, plan.header, use_bgzf=False)
    for record in records:
        writer.write_record(record)

    out.seek(0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reparsed = list(vcfpy.Reader.from_stream(out, record_checks=["INFO"]))

    # No IncorrectListLength warnings: each INFO annotation still matches the
    # Number declared for the ID it is emitted under.
    assert [str(w.message) for w in caught] == []
    assert reparsed[0].INFO == {"AF": [0.5], "DP": 10}
    assert reparsed[0].call_for_sample["S1"].data == {"GT": "0/1", "AD": [5, 5], "DP": 10}
    assert reparsed[1].INFO == {"AF_B": 0.7, "DP": 20}
    assert reparsed[1].call_for_sample["S2"].data == {"GT": "0/1", "AD_B": [8], "DP": 20}
    # FORMAT values still satisfy the Number of the ID they are emitted under.
    assert len(reparsed[0].call_for_sample["S1"].data["AD"]) == 2  # Number=R, 1 ALT
    assert len(reparsed[1].call_for_sample["S2"].data["AD_B"]) == 1  # Number=A, 1 ALT
    # Samples are unioned and missing ones are padded.
    assert plan.header.samples.names == ["S1", "S2"]


def test_compatible_field_is_not_renamed_even_if_description_differs():
    header_a = _read_vcf(HEADER_A).header
    header_b = _read_vcf(
        HEADER_A.replace("Allele frequency per ALT", "Different description")
    ).header
    plan = plan_merge([header_a, header_b])
    assert plan.compatible
    assert not plan.requires_renames
    assert plan.renames_for(1) == {}


def test_same_declaration_with_different_values_stays_compatible():
    header_a = _read_vcf(
        '##fileformat=VCFv4.3\n'
        '##INFO=<ID=AF,Number=A,Type=Float,Description="af">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "20\t1\t.\tG\tA,T\t.\t.\tAF=0.1,0.2\n"
    ).header
    header_b = _read_vcf(
        '##fileformat=VCFv4.3\n'
        '##INFO=<ID=AF,Number=A,Type=Float,Description="af">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "20\t2\t.\tG\tA\t.\t.\tAF=0.9\n"
    ).header
    plan = plan_merge([header_a, header_b])
    assert plan.compatible
    assert not plan.requires_renames


def test_each_source_field_keeps_its_own_definition_after_merge():
    header_a, header_b = _headers()
    plan = plan_merge([header_a, header_b], source_names=["A", "B"])
    for scope, field_id, source in [("INFO", "AF", 0), ("INFO", "AF", 1),
                                    ("FORMAT", "AD", 0), ("FORMAT", "AD", 1)]:
        original = (header_a, header_b)[source]
        before = (
            original.get_info_field_info(field_id)
            if scope == "INFO"
            else original.get_format_field_info(field_id)
        )
        after = plan.merged_field_info(source, scope, field_id)
        assert (after.number, after.type) == (before.number, before.type)
        assert plan.rename_for(source, scope, field_id) == (
            field_id if source == 0 else field_id + "_B"
        )


def test_equal_observed_lengths_do_not_make_declarations_compatible():
    # Number=R and Number=2 both yield two values for a biallelic site, so the
    # observed lengths are identical while the declarations are not.
    header_a = _read_vcf(
        '##fileformat=VCFv4.3\n'
        '##FORMAT=<ID=AD,Number=R,Type=Integer,Description="ref+alt">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\n"
        "20\t1\t.\tG\tA\t.\t.\t.\tAD\t5,5\n"
    ).header
    header_b = _read_vcf(
        '##fileformat=VCFv4.3\n'
        '##FORMAT=<ID=AD,Number=2,Type=Integer,Description="two values">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS2\n"
        "20\t1\t.\tG\tA\t.\t.\t.\tAD\t5,5\n"
    ).header
    plan = plan_merge([header_a, header_b], source_names=["A", "B"])
    assert not plan.compatible
    assert [conflict.reason for conflict in plan.conflicts] == ["Number differs ('R' vs 2)"]
    assert plan.renames_for(1) == {"AD": "AD_B"}


def test_auto_rename_disabled_makes_conflict_unmergeable():
    header_a, header_b = _headers()
    _, _, records_a, _ = _records()
    plan = plan_merge([header_a, header_b], source_names=["A", "B"], auto_rename=False)
    assert not plan.compatible
    assert not plan.mergeable
    # The same field ID cannot carry two definitions in the merged header.
    with pytest.raises(MergeConflictError):
        merge_records(plan, [[], []])
    with pytest.raises(MergeConflictError):
        plan.convert_record(records_a[0], "A")


def test_strict_rejects_conflicts_instead_of_renaming():
    header_a, header_b = _headers()
    with pytest.raises(MergeConflictError, match="INFO AF"):
        plan_merge([header_a, header_b], source_names=["A", "B"], strict=True)


def test_explicit_rename_mapping_and_callable():
    header_a, header_b = _headers()
    mapping_plan = plan_merge(
        [header_a, header_b],
        source_names=["A", "B"],
        rename={("INFO", "AF"): "AF_SCALAR", ("FORMAT", "AD"): "AD_ALT"},
    )
    assert {rename.old_id: rename.new_id for rename in mapping_plan.renames} == {
        "AF": "AF_SCALAR",
        "AD": "AD_ALT",
    }

    def naming(scope, source_index, field_id, info):
        if source_index == 1:
            return "{}_{}".format(field_id, scope.lower())
        return None

    callable_plan = plan_merge([header_a, header_b], source_names=["A", "B"], rename=naming)
    assert callable_plan.renames_for(1) == {"AF": "AF_info", "AD": "AD_format"}


def test_rename_collision_gets_unique_suffix():
    header_a = _read_vcf(
        '##fileformat=VCFv4.3\n'
        '##INFO=<ID=X,Number=A,Type=Float,Description="a">\n'
        '##INFO=<ID=X_B,Number=1,Type=Float,Description="already there">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "20\t1\t.\tG\tA\t.\t.\t.\n"
    ).header
    header_b = _read_vcf(
        '##fileformat=VCFv4.3\n'
        '##INFO=<ID=X,Number=1,Type=Float,Description="b">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "20\t1\t.\tG\tA\t.\t.\t.\n"
    ).header
    plan = plan_merge([header_a, header_b], source_names=["A", "B"])
    assert plan.renames_for(1) == {"X": "X_B_2"}


def test_format_conflict_does_not_touch_info_fields_and_vice_versa():
    header_a = _read_vcf(
        '##fileformat=VCFv4.3\n'
        '##INFO=<ID=DP,Number=A,Type=Integer,Description="info a">\n'
        '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="format 1">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\n"
        "20\t1\t.\tG\tA\t.\t.\tDP=1\tDP\t5\n"
    ).header
    header_b = _read_vcf(
        '##fileformat=VCFv4.3\n'
        '##INFO=<ID=DP,Number=1,Type=Integer,Description="info b">\n'
        '##FORMAT=<ID=DP,Number=A,Type=Integer,Description="format a">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS2\n"
        "20\t2\t.\tG\tA\t.\t.\tDP=5\tDP\t6\n"
    ).header
    plan = plan_merge([header_a, header_b], source_names=["A", "B"])
    assert {(c.scope, c.field_id) for c in plan.conflicts} == {("INFO", "DP"), ("FORMAT", "DP")}
    # Both scopes conflict independently and both get scoped renames.
    assert plan.renames_for(1, "INFO") == {"DP": "DP_B"}
    assert plan.renames_for(1, "FORMAT") == {"DP": "DP_B"}

    target_plan = plan_merge(
        [header_a, header_b],
        source_names=["A", "B"],
        rename={("FORMAT", "DP"): "DP_FORMAT"},
    )
    assert target_plan.renames_for(1, "INFO") == {"DP": "DP_B"}
    assert target_plan.renames_for(1, "FORMAT") == {"DP": "DP_FORMAT"}


def test_rename_collection_queries_and_comparison():
    header_a, header_b = _headers()
    plan = plan_merge([header_a, header_b], source_names=["A", "B"])
    renames = RenameCollection([FieldRename("INFO", 1, "AF", "AF_B", "Number differs")])
    assert renames[("INFO", 1, "AF")] == "AF_B"
    assert renames.get("INFO", 0, "AF") is None
    assert renames.as_dict() == {("INFO", 1, "AF"): "AF_B"}
    assert renames == renames.as_dict()
    assert renames == [FieldRename("INFO", 1, "AF", "AF_B", "Number differs")]
    assert renames.nested()["INFO"] == {1: {"AF": "AF_B"}}
    assert plan.rename_map == {("INFO", 1, "AF"): "AF_B", ("FORMAT", 1, "AD"): "AD_B"}
    assert plan.rename_mapping == plan.rename_map


def test_preview_to_dict_and_json():
    header_a, header_b = _headers()
    plan = plan_merge([header_a, header_b], source_names=["A", "B"])
    preview = plan.to_dict()
    assert preview["compatible"] is False
    assert preview["mergeable"] is True
    assert preview["requires_renames"] is True
    assert preview["renames"][0]["scope"] == "INFO"
    assert preview["renames"][0]["old_id"] == "AF"
    assert preview["renames"][0]["new_id"] == "AF_B"
    assert preview["conflicts"][0]["field_id"] == "AF"
    assert preview["conflicts"][0]["resolved"] is True
    assert json.loads(plan.to_json()) == preview


def test_plan_unpacks_as_header_and_rename_map():
    header_a, header_b = _headers()
    merged_header, rename_map = merge.merge_headers([header_a, header_b], source_names=["A", "B"])
    assert merged_header.get_info_field_info("AF_B").number == 1
    assert rename_map == {("INFO", 1, "AF"): "AF_B", ("FORMAT", 1, "AD"): "AD_B"}


def test_plan_can_be_used_as_a_header():
    header_a, header_b = _headers()
    plan = plan_merge([header_a, header_b], source_names=["A", "B"])
    assert plan.info_ids() == ["AF", "DP", "AF_B"]
    assert plan.header.get_info_field_info("AF_B").number == 1
    assert plan.renames["INFO"] == {1: {"AF": "AF_B"}}

    _, _, records_a, _ = _records()
    out = io.StringIO()
    vcfpy.Writer.from_stream(out, plan, use_bgzf=False).write_record(
        plan.pad_record(plan.convert_record(records_a[0], "A"))
    )
    assert "##INFO=<ID=AF_B,Number=1" in out.getvalue()


def test_merge_records_rejects_wrong_source_count():
    header_a, header_b = _headers()
    plan = plan_merge([header_a, header_b])
    with pytest.raises(MergeError):
        merge_records(plan, [[]])


def test_duplicate_conflicting_definitions_in_one_source_are_not_mergeable():
    source = """\
##fileformat=VCFv4.3
##INFO=<ID=X,Number=A,Type=Float,Description="first">
##INFO=<ID=X,Number=1,Type=Float,Description="second">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
20\t1\t.\tG\tA\t.\t.\t.
"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        header = _read_vcf(source).header
    plan = plan_merge([header])
    assert not plan.mergeable
    with pytest.raises(MergeConflictError):
        merge_records(plan, [[]])


def test_merge_files_end_to_end_with_different_sample_sets(tmp_path):
    path_a = tmp_path / "a.vcf"
    path_b = tmp_path / "b.vcf"
    path_a.write_text(HEADER_A)
    path_b.write_text(HEADER_B)
    output = tmp_path / "merged.vcf"

    plan = merge.merge_vcf_files([str(path_a), str(path_b)], str(output), source_names=["A", "B"])

    assert plan.header.samples.names == ["S1", "S2"]
    text = output.read_text()
    assert "##INFO=<ID=AF,Number=A,Type=Float" in text
    assert "##INFO=<ID=AF_B,Number=1,Type=Float" in text
    assert "##FORMAT=<ID=AD_B,Number=A,Type=Integer" in text
    with vcfpy.Reader.from_path(str(output)) as reader:
        records = list(reader)
    assert [record.INFO for record in records] == [{"AF": [0.5], "DP": 10}, {"AF_B": 0.7, "DP": 20}]
    assert records[0].call_for_sample["S2"].data == {"GT": None, "AD": [], "DP": None}


def test_merge_files_dry_run_writes_nothing(tmp_path):
    path_a = tmp_path / "a.vcf"
    path_b = tmp_path / "b.vcf"
    path_a.write_text(HEADER_A)
    path_b.write_text(HEADER_B)
    output = tmp_path / "merged.vcf"

    plan = merge.merge_vcf_files([str(path_a), str(path_b)], str(output), dry_run=True)

    assert not output.exists()
    assert not plan.compatible
    assert plan.renames_for(1) == {"AF": "AF_2", "AD": "AD_2"}


def test_cli_dry_run_reports_renames(tmp_path):
    path_a = tmp_path / "a.vcf"
    path_b = tmp_path / "b.vcf"
    path_a.write_text(HEADER_A)
    path_b.write_text(HEADER_B)
    output = io.StringIO()

    assert merge_main([str(path_a), str(path_b), "--source-name", "A", "--source-name", "B"], stream=output) == 0
    text = output.getvalue()
    assert "compatible: no" in text
    assert "INFO AF (source B) -> AF_B" in text
    assert "FORMAT AD (source B) -> AD_B" in text

    json_output = io.StringIO()
    assert merge_main([str(path_a), str(path_b), "--json"], stream=json_output) == 0
    assert json.loads(json_output.getvalue())["renames"][0]["old_id"] == "AF"


def test_cli_json_and_written_output(tmp_path):
    path_a = tmp_path / "a.vcf"
    path_b = tmp_path / "b.vcf"
    path_a.write_text(HEADER_A)
    path_b.write_text(HEADER_B)
    output = tmp_path / "merged.vcf"
    report = io.StringIO()

    assert merge_main([str(path_a), str(path_b), "-o", str(output)], stream=report) == 0
    assert output.exists()
    assert "wrote:" in report.getvalue()
