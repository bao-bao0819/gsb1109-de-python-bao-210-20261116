# -*- coding: utf-8 -*-
"""Tests for pedigree consistency screening"""

import io

import pytest

import vcfpy
from vcfpy.pedigree import CONFLICT, CONSISTENT, INSUFFICIENT


def call(sample: str, gt: str, **data) -> vcfpy.Call:
    return vcfpy.Call(sample, {"GT": gt, **data})


def trio(child: str = "child", father: str = "father", mother: str = "mother", sex=None) -> vcfpy.Pedigree:
    return vcfpy.Pedigree.from_trio(child, father=father, mother=mother, sex=sex)


def screen(calls, inheritance="autosomal_dominant", **kwargs):
    return vcfpy.screen_calls(calls, trio(), inheritance, **kwargs)[0]


# ---------------------------------------------------------------------------
# Complete calls
# ---------------------------------------------------------------------------


def test_complete_trio_consistent():
    result = screen(
        {
            "child": call("child", "0/1"),
            "father": call("father", "0/0"),
            "mother": call("mother", "1/1"),
        }
    )
    assert result.status == CONSISTENT
    assert result.consistent
    assert result.proven
    assert result.compatible_completions == result.total_completions == 1
    assert result.assumptions == ()


def test_complete_trio_conflict():
    result = screen(
        {
            "child": call("child", "1/1"),
            "father": call("father", "0/0"),
            "mother": call("mother", "0/0"),
        }
    )
    assert result.status == CONFLICT
    assert result.conflict
    assert "absent from every parent" in result.explanation
    assert result.compatible_completions == 0


def test_conflict_is_not_masked_by_one_missing_parent_allele():
    """A child hom-alt cannot be explained when the other parent is hom-ref"""
    result = screen(
        {
            "child": call("child", "1/1"),
            "father": call("father", "0/."),
            "mother": call("mother", "0/0"),
        }
    )
    assert result.status == CONFLICT
    assert result.total_completions == 2


def test_missing_parent_allele_is_not_automatic_conflict():
    """One compatible completion among two is not proof: report insufficient"""
    result = screen(
        {
            "child": call("child", "0/1"),
            "father": call("father", "0/."),
            "mother": call("mother", "0/0"),
        }
    )
    assert result.status == INSUFFICIENT
    assert result.compatible_completions == 1
    assert result.total_completions == 2
    assert result.witnesses
    assert result.counter_witnesses
    assert any("0/." in assumption for assumption in result.assumptions)


def test_missing_parent_allele_all_completions_compatible_is_proven():
    result = screen(
        {
            "child": call("child", "0/1"),
            "father": call("father", "0/."),
            "mother": call("mother", "1/1"),
        }
    )
    # Whether the father is 0/0 or 0/1, the child 0/1 is possible.
    assert result.status == CONSISTENT
    assert result.compatible_completions == result.total_completions == 2
    assert result.witnesses
    assert "all 2 completions" in result.explanation


def test_child_partial_call_all_completions_compatible():
    result = screen(
        {
            "child": call("child", "0/."),
            "father": call("father", "0/0"),
            "mother": call("mother", "0/0"),
        }
    )
    assert result.status == CONSISTENT
    assert result.compatible_completions == result.total_completions
    assert any("partially called" in assumption for assumption in result.assumptions)


def test_child_partial_call_material():
    result = vcfpy.screen_calls(
        {
            "child": call("child", "0/."),
            "father": call("father", "0/0"),
            "mother": call("mother", "0/0"),
        },
        trio(),
        "autosomal_dominant",
        known_alleles=(0, 1),
    )[0]
    assert result.status == INSUFFICIENT
    assert result.compatible_completions == 1
    assert result.total_completions == 2


def test_fully_missing_child_is_never_consistent():
    result = screen(
        {
            "child": call("child", "./."),
            "father": call("father", "0/0"),
            "mother": call("mother", "0/0"),
        }
    )
    assert result.status == INSUFFICIENT
    assert not result.consistent
    assert not result.informative


def test_missing_required_parent_is_not_a_conflict():
    result = screen({"child": call("child", "0/1"), "mother": call("mother", "0/0")})
    assert result.status == INSUFFICIENT
    assert not result.conflict
    assert any("no genotype call" in assumption for assumption in result.assumptions)


def test_missing_required_parent_can_still_conflict():
    """A missing father cannot explain a child allele that the mother lacks"""
    result = screen({"child": call("child", "1/1"), "mother": call("mother", "0/0")})
    assert result.status == CONFLICT


def test_missing_father_with_known_alleles():
    """With a known allele universe, the missing father is enumerated"""
    calls = {
        "child": call("child", "1/1"),
        "father": None,
        "mother": call("mother", "0/1"),
    }
    result = vcfpy.screen_calls(calls, trio(), "autosomal_recessive", known_alleles=(0, 1))[0]
    # The father may be 0/0, 0/1, or 1/1.  The child is explained when the
    # father carries allele 1 (0/1 or 1/1) and the mother also carries it.
    assert result.status == INSUFFICIENT
    assert result.total_completions == 3
    assert result.compatible_completions == 2


def test_missing_father_cannot_be_hom_alt_without_evidence():
    calls = {
        "child": call("child", "1/1"),
        "father": None,
        "mother": call("mother", "0/1"),
    }
    result = vcfpy.screen_calls(calls, trio(), "autosomal_recessive", known_alleles=(0, 1))[0]
    # The father could be 1/1, so this is not a conflict, but only two of
    # three paternal genotypes work, so the evidence is insufficient.
    assert result.status == INSUFFICIENT
    assert result.compatible_completions == 2
    assert result.total_completions == 3


def test_missing_father_still_conflicts_when_mother_lacks_allele():
    calls = {
        "child": call("child", "1/1"),
        "father": None,
        "mother": call("mother", "0/0"),
    }
    result = vcfpy.screen_calls(calls, trio(), "autosomal_recessive", known_alleles=(0, 1))[0]
    # A diploid child needs one allele from each parent.  Regardless of the
    # missing father, the mother 0/0 cannot transmit allele 1.
    assert result.status == CONFLICT


# ---------------------------------------------------------------------------
# Polyploid children
# ---------------------------------------------------------------------------


def test_triploid_child_consistent():
    result = screen(
        {
            "child": call("child", "0/0/1"),
            "father": call("father", "0/1"),
            "mother": call("mother", "0/0"),
        }
    )
    assert result.status == CONSISTENT
    assert any("polyploid (ploidy 3)" in assumption for assumption in result.assumptions)


def test_triploid_child_conflict():
    result = screen(
        {
            "child": call("child", "1/1/1"),
            "father": call("father", "0/0"),
            "mother": call("mother", "0/0"),
        }
    )
    assert result.status == CONFLICT


def test_triploid_child_supports_both_parental_splits():
    result = screen(
        {
            "child": call("child", "0/1/2"),
            "father": call("father", "0/1"),
            "mother": call("mother", "1/2"),
        }
    )
    assert result.status == CONSISTENT


def test_tetraploid_child_consistent():
    result = screen(
        {
            "child": call("child", "0/1/1/2"),
            "father": call("father", "0/1"),
            "mother": call("mother", "1/2"),
        }
    )
    assert result.status == CONSISTENT


def test_tetraploid_child_conflict_with_only_two_parental_alleles():
    result = screen(
        {
            "child": call("child", "0/1/2/3"),
            "father": call("father", "0/1"),
            "mother": call("mother", "0/1"),
        }
    )
    assert result.status == CONFLICT


def test_polyploid_parent_is_reported():
    result = screen(
        {
            "child": call("child", "0/1"),
            "father": call("father", "0/1/2"),
            "mother": call("mother", "0/0"),
        }
    )
    assert any("polyploid" in assumption for assumption in result.assumptions)


# ---------------------------------------------------------------------------
# Inheritance modes
# ---------------------------------------------------------------------------


def test_autosomal_modes_share_genotype_transmission():
    """Phenotype-based dominance is not part of the genotype check"""
    calls = {
        "child": call("child", "1/1"),
        "father": call("father", "0/1"),
        "mother": call("mother", "0/1"),
    }
    dominant = vcfpy.screen_calls(calls, trio(), "autosomal_dominant")[0]
    recessive = vcfpy.screen_calls(calls, trio(), "autosomal_recessive")[0]
    # Genotype-level transmission is the same for both modes; the difference
    # is phenotype interpretation, which is not part of the input here.
    assert dominant.status == recessive.status == CONSISTENT
    assert dominant.compatible_completions == recessive.compatible_completions == 1


def test_unknown_inheritance_mode_rejected():
    with pytest.raises(ValueError):
        vcfpy.screen_calls({}, trio(), "mitochondrial")


def test_x_linked_male_inherits_from_mother_only():
    calls = {
        "child": call("child", "1"),
        "father": call("father", "0"),
        "mother": call("mother", "1/1"),
    }
    result = vcfpy.screen_calls(calls, trio(sex="male"), "x_linked_recessive")[0]
    assert result.status == CONSISTENT


def test_x_linked_male_conflict_when_mother_lacks_allele():
    calls = {
        "child": call("child", "1"),
        "father": call("father", "1"),
        "mother": call("mother", "0/0"),
    }
    result = vcfpy.screen_calls(calls, trio(sex="male"), "x_linked_recessive")[0]
    assert result.status == CONFLICT


def test_x_linked_unknown_sex_records_assumption():
    calls = {
        "child": call("child", "1"),
        "father": call("father", "1"),
        "mother": call("mother", "1/1"),
    }
    result = vcfpy.screen_calls(calls, trio(sex=None), "x_linked_recessive")[0]
    assert result.status == CONSISTENT
    assert any("sex is unknown" in assumption for assumption in result.assumptions)


def test_de_novo_requires_novel_allele():
    novel = {
        "child": call("child", "0/1"),
        "father": call("father", "0/0"),
        "mother": call("mother", "0/0"),
    }
    inherited = {
        "child": call("child", "0/0"),
        "father": call("father", "0/0"),
        "mother": call("mother", "0/0"),
    }
    assert vcfpy.screen_calls(novel, trio(), "de_novo")[0].status == CONSISTENT
    assert vcfpy.screen_calls(inherited, trio(), "de_novo")[0].status == CONFLICT


def test_de_novo_with_missing_parent_is_insufficient():
    calls = {
        "child": call("child", "0/1"),
        "father": call("father", "0/."),
        "mother": call("mother", "0/0"),
    }
    result = vcfpy.screen_calls(calls, trio(), "de_novo")[0]
    assert result.status == INSUFFICIENT
    assert not result.consistent


# ---------------------------------------------------------------------------
# Pedigree model
# ---------------------------------------------------------------------------


def test_pedigree_rejects_duplicate_member():
    with pytest.raises(ValueError):
        vcfpy.Pedigree([vcfpy.PedigreeMember("a"), vcfpy.PedigreeMember("a")])


def test_pedigree_rejects_unknown_parent():
    with pytest.raises(ValueError):
        vcfpy.Pedigree([vcfpy.PedigreeMember("child", father="ghost")])


def test_pedigree_rejects_self_parent():
    with pytest.raises(ValueError):
        vcfpy.PedigreeMember("child", father="child")


def test_pedigree_from_trio_requires_parent():
    with pytest.raises(ValueError):
        vcfpy.Pedigree.from_trio("child")


def test_screen_multiple_children():
    family = vcfpy.Pedigree(
        [
            vcfpy.PedigreeMember("father"),
            vcfpy.PedigreeMember("mother"),
            vcfpy.PedigreeMember("c1", father="father", mother="mother"),
            vcfpy.PedigreeMember("c2", father="father", mother="mother"),
        ]
    )
    calls = {
        "father": call("father", "0/1"),
        "mother": call("mother", "0/0"),
        "c1": call("c1", "0/0"),
        "c2": call("c2", "1/1"),
    }
    results = vcfpy.screen_calls(calls, family, "autosomal_dominant")
    assert [r.sample for r in results] == ["c1", "c2"]
    assert [r.status for r in results] == [CONSISTENT, CONFLICT]


def test_screen_unknown_sample_raises():
    with pytest.raises(KeyError):
        vcfpy.screen_calls({}, trio(), "autosomal_dominant", samples=["nobody"])


def test_result_to_dict_is_serializable():
    import json

    result = screen(
        {
            "child": call("child", "0/."),
            "father": call("father", "0/0"),
            "mother": call("mother", "0/0"),
        }
    )
    encoded = json.dumps(result.to_dict())
    assert '"status"' in encoded


# ---------------------------------------------------------------------------
# Record and header integration
# ---------------------------------------------------------------------------


def build_record(genotypes: dict[str, str], alt: str = "T") -> vcfpy.Record:
    calls = [call(sample, gt) for sample, gt in genotypes.items()]
    return vcfpy.Record(
        CHROM="1",
        POS=100,
        ID=[],
        REF="C",
        ALT=[vcfpy.Substitution(vcfpy.SNV, alt)],
        QUAL=None,
        FILTER=[],
        INFO={},
        FORMAT=["GT"],
        calls=calls,
    )


def test_screen_record():
    record = build_record({"child": "0/1", "father": "0/.", "mother": "0/0"})
    results = vcfpy.screen_record(record, trio(), "autosomal_dominant")
    assert len(results) == 1
    assert results[0].status == INSUFFICIENT


def test_screen_record_conflict():
    record = build_record({"child": "1/1", "father": "0/0", "mother": "0/0"})
    results = vcfpy.screen_record(record, trio(), "autosomal_dominant")
    assert results[0].status == CONFLICT


def test_screen_records_preserves_order():
    records = [
        build_record({"child": "0/1", "father": "0/0", "mother": "1/1"}),
        build_record({"child": "1/1", "father": "0/0", "mother": "0/0"}),
    ]
    results = vcfpy.screen_records(records, trio(), "autosomal_dominant")
    assert [r[0].POS for r in results] == [100, 100]
    assert [r[1][0].status for r in results] == [CONSISTENT, CONFLICT]


def test_pedigree_from_header():
    vcf = """##fileformat=VCFv4.3
##PEDIGREE=<ID=child,Father=father,Mother=mother,Sex=1>
##PEDIGREE=<ID=father,Sex=1>
##PEDIGREE=<ID=mother,Sex=2>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tchild\tfather\tmother
1\t100\t.\tC\tT\t.\t.\t.\tGT\t0/1\t0/0\t1/1
"""
    reader = vcfpy.Reader.from_stream(io.StringIO(vcf))
    family = vcfpy.Pedigree.from_header(reader.header)
    assert family.get("child").father == "father"
    assert family.get("child").sex == "male"
    assert family.get("father").sex == "male"
    assert family.get("mother").sex == "female"
    record = next(reader)
    result = vcfpy.screen_record(record, family, "autosomal_dominant")[0]
    assert result.status == CONSISTENT


def test_screen_reader_uses_header_pedigree():
    vcf = """##fileformat=VCFv4.3
##PEDIGREE=<ID=child,Father=father,Mother=mother,Sex=1>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tchild\tfather\tmother
1\t100\t.\tC\tT\t.\t.\t.\tGT\t1/1\t0/0\t0/0
"""
    reader = vcfpy.Reader.from_stream(io.StringIO(vcf))
    results = list(vcfpy.screen_reader(reader, "autosomal_dominant"))
    assert len(results) == 1
    assert results[0][1][0].status == CONFLICT


# ---------------------------------------------------------------------------
# Safety and configuration
# ---------------------------------------------------------------------------


def test_completion_limit_yields_insufficient():
    calls = {
        "child": call("child", "0/1"),
        "father": call("father", "./."),
        "mother": call("mother", "./."),
    }
    result = vcfpy.screen_calls(
        calls,
        trio(),
        "autosomal_dominant",
        known_alleles=(0, 1, 2),
        max_completions=2,
    )[0]
    assert result.status == INSUFFICIENT
    assert "limit" in result.explanation


def test_optional_parent_without_call_remains_insufficient():
    calls = {"child": call("child", "1/1"), "mother": call("mother", "1/1")}
    result = vcfpy.screen_calls(
        calls,
        trio(),
        "autosomal_dominant",
        father_alleles="optional",
    )[0]
    # The missing father may be 1/1, 0/1, or 0/0.  Only some of these explain
    # the child, so "optional" does not turn an existence witness into proof.
    assert result.status == INSUFFICIENT
    assert any("optional" in assumption for assumption in result.assumptions)


def test_ignored_parent_is_insufficient():
    calls = {
        "child": call("child", "0/1"),
        "father": call("father", "0/0"),
        "mother": call("mother", "1/1"),
    }
    result = vcfpy.screen_calls(calls, trio(), "autosomal_dominant", father_alleles="ignore")[0]
    assert result.status == INSUFFICIENT


def test_ignored_child_is_insufficient():
    calls = {
        "child": call("child", "0/1"),
        "father": call("father", "0/0"),
        "mother": call("mother", "1/1"),
    }
    result = vcfpy.screen_calls(calls, trio(), "autosomal_dominant", child_alleles="ignore")[0]
    assert result.status == INSUFFICIENT


def test_invalid_allele_policy_rejected():
    with pytest.raises(ValueError):
        vcfpy.screen_calls({}, trio(), "autosomal_dominant", father_alleles="sometimes")


def test_known_alleles_extend_completion_universe():
    calls = {
        "child": call("child", "0/1"),
        "father": call("father", "0/."),
        "mother": call("mother", "0/0"),
    }
    result = vcfpy.screen_calls(calls, trio(), "autosomal_dominant", known_alleles=(0, 1, 2))[0]
    # The father may be 0/0, 0/1, or 0/2.  The child 0/1 is possible when the
    # father is 0/1 or when the father is 0/0 (then the mother would have to
    # be 1/0, which she is not here).  Only the 0/1 completion works.
    assert result.total_completions == 3
    assert result.compatible_completions == 1
    assert result.status == INSUFFICIENT


def test_single_parent_pedigree_is_not_overclaimed():
    family = vcfpy.Pedigree(
        [
            vcfpy.PedigreeMember("mother", sex="female"),
            vcfpy.PedigreeMember("child", mother="mother"),
        ]
    )
    calls = {"child": call("child", "0/1"), "mother": call("mother", "1/1")}
    result = vcfpy.screen_calls(calls, family, "autosomal_dominant")[0]
    # The child's paternal allele is unobserved, so a single-parent check
    # cannot prove full pedigree consistency.
    assert result.status == INSUFFICIENT
    assert any("absent from the pedigree" in assumption for assumption in result.assumptions)


def test_single_parent_pedigree_can_still_conflict():
    family = vcfpy.Pedigree(
        [
            vcfpy.PedigreeMember("mother", sex="female"),
            vcfpy.PedigreeMember("child", mother="mother"),
        ]
    )
    calls = {"child": call("child", "1/1"), "mother": call("mother", "0/0")}
    result = vcfpy.screen_calls(calls, family, "autosomal_dominant")[0]
    # The mother cannot transmit allele 1, and a diploid child needs one
    # maternal allele.
    assert result.status == CONFLICT


def test_haploid_child_is_supported():
    result = screen(
        {
            "child": call("child", "1"),
            "father": call("father", "0/1"),
            "mother": call("mother", "0/0"),
        }
    )
    assert result.status == CONSISTENT


def test_haploid_child_conflict():
    result = screen(
        {
            "child": call("child", "1"),
            "father": call("father", "0/0"),
            "mother": call("mother", "0/0"),
        }
    )
    assert result.status == CONFLICT


def test_missing_gt_field():
    child = vcfpy.Call("child", {"DP": 10})
    calls = {"child": child, "father": call("father", "0/0"), "mother": call("mother", "1/1")}
    result = screen(calls)
    assert result.status == INSUFFICIENT
    assert not result.consistent


def test_no_parents_in_pedigree_yields_insufficient():
    family = vcfpy.Pedigree([vcfpy.PedigreeMember("child")])
    result = vcfpy.screen_calls(
        {"child": call("child", "0/1")}, family, "autosomal_dominant", samples=["child"]
    )[0]
    assert result.status == INSUFFICIENT


def test_result_bool_only_true_for_proven_consistency():
    assert screen({"child": call("child", "0/1"), "father": call("father", "0/0"), "mother": call("mother", "1/1")})
    assert not screen({"child": call("child", "1/."), "father": call("father", "0/0"), "mother": call("mother", "0/0")})
    assert not screen({"child": call("child", "1/1"), "father": call("father", "0/0"), "mother": call("mother", "0/0")})
