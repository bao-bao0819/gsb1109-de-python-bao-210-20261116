import pytest

from gffutils import feature as feature_module
from gffutils.seqidmap import (
    Candidate,
    ReferenceCatalog,
    SeqidResolver,
    SourceRecord,
    records_from_features,
)


def resolver(**catalogs):
    return SeqidResolver(catalogs)


class TestUnaryConstraints:
    def test_length_disambiguates_identical_names(self):
        # Same reference name in both assemblies; only the feature range
        # can decide.
        res = resolver(
            asmA=ReferenceCatalog("asmA", {"chr1": 1000}),
            asmB=ReferenceCatalog("asmB", {"chr1": 100000}),
        )
        result = res.resolve([SourceRecord("chr1", 1, 5000, "f1")])
        assert result.conflicts == []
        assert result.unique == {"chr1": Candidate("asmB", "chr1")}

    def test_name_similarity_is_not_decisive(self):
        # "chr2" fits both "chr2" and the similarly-named "chr2_random";
        # without further evidence the mapping must stay ambiguous.
        res = resolver(
            asm=ReferenceCatalog("asm", {"chr2": 1000, "chr2_random": 1000})
        )
        result = res.resolve([SourceRecord("chr2", 1, 500, "f1")])
        assert result.conflicts == []
        assert result.unique == {}
        assert result.possible["chr2"] == frozenset(
            [Candidate("asm", "chr2"), Candidate("asm", "chr2_random")]
        )

    def test_alias_restricts_domain(self):
        res = resolver(
            asm=ReferenceCatalog(
                "asm",
                {"chr3": 10000, "chr4": 10000},
                aliases={"seg1": "chr3"},
            )
        )
        result = res.resolve([SourceRecord("seg1", 1, 500, "f1")])
        assert result.unique == {"seg1": Candidate("asm", "chr3")}

    def test_alias_target_missing_from_catalog(self):
        res = resolver(
            asm=ReferenceCatalog("asm", {"chr3": 10000},
                                 aliases={"seg1": "nope"})
        )
        result = res.resolve([SourceRecord("seg1", 1, 500, "f1")])
        assert result.possible == {}
        assert len(result.conflicts) == 1
        conflict = result.conflicts[0]
        assert conflict.seqids == ["seg1"]
        reasons = " ".join(r for _, r in conflict.eliminations["seg1"])
        assert "alias" in reasons

    def test_summary_region_restricts_domain(self):
        res = resolver(
            asm=ReferenceCatalog(
                "asm",
                {"chr1": 100000, "chr2": 100000},
                summaries={"chr1": (1, 2000)},
            )
        )
        result = res.resolve([SourceRecord("chr1", 10, 5000, "f1")])
        # chr1's declared region is too small despite its length.
        assert result.unique == {"chr1": Candidate("asm", "chr2")}


class TestParentChildConstraints:
    def catalogs(self, **kwargs):
        return resolver(
            asm1=ReferenceCatalog("asm1", {"contig1": 100}),
            asm2=ReferenceCatalog(
                "asm2",
                {"contig1": 100, "refP": 1000},
                aliases={"plink": "refP", "contig1": "contig1"},
                **kwargs
            ),
        )

    def test_related_records_share_one_mapping(self):
        # The child seqid "contig1" exists in both assemblies, but its
        # parent is pinned to asm2 by an alias, so the child must follow
        # the parent's catalog instead of picking a per-line favorite.
        res = self.catalogs(parent_edges={("refP", "contig1")})
        records = [
            SourceRecord("plink", 1, 900, "gene1"),
            SourceRecord("contig1", 1, 50, "mrna1", parents=["gene1"]),
        ]
        result = res.resolve(records)
        assert result.conflicts == []
        assert result.unique == {
            "plink": Candidate("asm2", "refP"),
            "contig1": Candidate("asm2", "contig1"),
        }

    def test_cross_reference_parent_edge_required(self):
        # Without a declared edge, parent and child on different
        # references cannot be reconciled.
        res = self.catalogs()
        records = [
            SourceRecord("plink", 1, 900, "gene1"),
            SourceRecord("contig1", 1, 50, "mrna1", parents=["gene1"]),
        ]
        result = res.resolve(records)
        assert result.possible == {}
        assert len(result.conflicts) == 1
        conflict = result.conflicts[0]
        assert conflict.seqids == ["contig1", "plink"]
        assert len(conflict.inconsistent_edges) == 1
        edge = conflict.inconsistent_edges[0]
        assert edge.parent_feature_id == "gene1"
        assert edge.child_feature_id == "mrna1"

    def test_same_reference_parent_child(self):
        res = resolver(
            asm=ReferenceCatalog("asm", {"chrA": 1000, "chrB": 100})
        )
        records = [
            SourceRecord("up", 1, 900, "gene1"),
            SourceRecord("down", 1, 90, "mrna1", parents=["gene1"]),
        ]
        result = res.resolve(records)
        # "up" only fits chrA; the child must land on the same reference
        # even though chrB would accommodate its own range.
        assert result.unique == {
            "up": Candidate("asm", "chrA"),
            "down": Candidate("asm", "chrA"),
        }

    def test_cross_catalog_parent_not_allowed(self):
        # Aliases pin the parent to asm1 and the child to asm2 (the
        # cross-aliases point at references absent from the other
        # catalog), so no single catalog can hold both.
        res = resolver(
            asm1=ReferenceCatalog("asm1", {"chrA": 1000},
                                  aliases={"down": "missing"}),
            asm2=ReferenceCatalog("asm2", {"chrB": 1000},
                                  aliases={"up": "missing"}),
        )
        records = [
            SourceRecord("up", 1, 900, "gene1"),
            SourceRecord("down", 1, 90, "mrna1", parents=["gene1"]),
        ]
        result = res.resolve(records)
        assert result.possible == {}
        assert len(result.conflicts) == 1
        conflict = result.conflicts[0]
        assert conflict.seqids == ["down", "up"]
        assert len(conflict.inconsistent_edges) == 1

    def test_unresolved_parent_imposes_no_constraint(self):
        res = resolver(asm=ReferenceCatalog("asm", {"chrA": 1000}))
        records = [SourceRecord("chrA", 1, 90, "mrna1", parents=["ghost"])]
        result = res.resolve(records)
        assert result.unique == {"chrA": Candidate("asm", "chrA")}
        assert result.unresolved_parents == [("mrna1", "ghost")]


class TestConflicts:
    def test_range_conflict_reports_offending_record(self):
        res = resolver(asm=ReferenceCatalog("asm", {"chr1": 100}))
        records = [
            SourceRecord("chr1", 1, 50, "ok_feature"),
            SourceRecord("chr1", 60, 500, "too_long"),
        ]
        result = res.resolve(records)
        assert result.possible == {}
        assert len(result.conflicts) == 1
        conflict = result.conflicts[0]
        assert conflict.seqids == ["chr1"]
        text = str(conflict)
        assert "too_long" in text
        assert "500" in text
        assert "chr1" in text

    def test_conflict_does_not_block_other_components(self):
        res = resolver(
            asm=ReferenceCatalog(
                "asm",
                {"chr1": 100, "chr2": 300},
                aliases={"chr2": "chr2"},
            )
        )
        records = [
            SourceRecord("chr1", 1, 500, "too_long"),
            SourceRecord("chr2", 1, 50, "fine"),
        ]
        result = res.resolve(records)
        assert len(result.conflicts) == 1
        assert result.unique == {"chr2": Candidate("asm", "chr2")}


class TestUnifiedIndex:
    def test_only_unique_mappings_enter_index(self):
        res = resolver(
            asm=ReferenceCatalog(
                "asm", {"chr1": 1000, "chr1_alt": 1000, "chr2": 5000}
            )
        )
        records = [
            SourceRecord("chr2", 1, 4000, "resolved"),
            SourceRecord("chr1", 1, 500, "ambiguous"),
        ]
        result = res.resolve(records)
        index = result.unified_index()
        assert set(index) == {Candidate("asm", "chr2"), "chr1"}
        assert index[Candidate("asm", "chr2")] == [records[0]]
        # Unresolved records keep their original seqid and range.
        kept = index["chr1"][0]
        assert (kept.seqid, kept.start, kept.end) == ("chr1", 1, 500)

    def test_conflicted_seqids_stay_out_of_index(self):
        res = resolver(asm=ReferenceCatalog("asm", {"chr1": 100}))
        records = [SourceRecord("chr1", 1, 500, "too_long")]
        result = res.resolve(records)
        index = result.unified_index()
        assert set(index) == {"chr1"}
        assert index["chr1"][0].end == 500


class TestFeasibleSets:
    def test_possible_unions_all_solutions(self):
        # Two solutions exist for the component; each seqid's possible
        # set is the union over solutions, so nothing is unique.
        res = resolver(
            asm=ReferenceCatalog(
                "asm",
                {"r1": 100, "r2": 100},
                parent_edges={("r1", "r2"), ("r2", "r1")},
            )
        )
        records = [
            SourceRecord("a", 1, 50, "p"),
            SourceRecord("b", 1, 50, "c", parents=["p"]),
        ]
        result = res.resolve(records)
        assert result.conflicts == []
        assert result.unique == {}
        assert result.possible["a"] == frozenset(
            [Candidate("asm", "r1"), Candidate("asm", "r2")]
        )
        assert result.possible["b"] == result.possible["a"]

    def test_empty_records(self):
        res = resolver(asm=ReferenceCatalog("asm", {"chr1": 100}))
        result = res.resolve([])
        assert result.possible == {}
        assert result.conflicts == []
        assert result.unified_index() == {}


class TestFeatureIntegration:
    def test_records_from_features(self):
        features = [
            feature_module.feature_from_line(
                "chr1\tsrc\tgene\t1\t900\t.\t+\t.\tID=gene1"
            ),
            feature_module.feature_from_line(
                "chr1\tsrc\tmRNA\t1\t900\t.\t+\t.\tID=mrna1;Parent=gene1"
            ),
            feature_module.feature_from_line(
                "ctg\tsrc\tgene\t1\t80\t.\t+\t.\tID=gene2"
            ),
        ]
        records = records_from_features(features)
        assert records[0] == SourceRecord("chr1", 1, 900, "gene1")
        assert records[1].parents == ("gene1",)

        res = resolver(
            asm=ReferenceCatalog(
                "asm",
                {"chr1": 1000, "scaffold7": 500},
                aliases={"ctg": "scaffold7"},
            )
        )
        result = res.resolve(records)
        assert result.conflicts == []
        assert result.unique == {
            "chr1": Candidate("asm", "chr1"),
            "ctg": Candidate("asm", "scaffold7"),
        }

    def test_requires_at_least_one_catalog(self):
        with pytest.raises(ValueError):
            SeqidResolver([])
