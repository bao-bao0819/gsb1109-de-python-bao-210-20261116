import tempfile
from gffutils import parser, create, feature, iterators, constants, helpers, exceptions
from gffutils import example_filename, create_db
from . import attr_test_cases
from textwrap import dedent

import pytest

TEST_FILENAMES = [
    example_filename(i)
    for i in [
        "c_elegans_WS199_ann_gff.txt",
        "ensembl_gtf.txt",
        "hybrid1.gff3",
        "ncbi_gff3.txt",
        "c_elegans_WS199_dna_shortened.fa",
        "F3-unique-3.v2.gff",
        "jgi_gff2.txt",
        "wormbase_gff2_alt.txt",
        "c_elegans_WS199_shortened_gff.txt",
        "glimmer_nokeyval.gff3",
        "mouse_extra_comma.gff3",
        "wormbase_gff2.txt",
    ]
]


def test_directives():
    data = dedent(
        """
    ##directive1 example
    .	.	.	.	.	.	.	.
    .	.	.	.	.	.	.	.
    .	.	.	.	.	.	.	.
    .	.	.	.	.	.	.	.
    """
    )

    it = iterators.DataIterator(data, from_string=True)
    assert it.directives == ["directive1 example"]

    db = create_db(data, dbfn=":memory:", from_string=True, verbose=False)
    assert db.directives == ["directive1 example"], db.directives


@pytest.mark.parametrize("item", attr_test_cases.attrs)
def test_attrs_OK(item):
    """
    Given an attribute string and a dictionary of what you expect, test the
    attribute splitting and reconstruction (invariant roundtrip).

    There are some corner cases for the roundtrip invariance that don't work
    (see attr_test_cases.py for details); `acceptable_reconstruction` handles
    those.
    """
    attr_str = item["str"]
    attr_dict = item["attrs"]
    acceptable_reconstruction = item["ok"]
    dialect_mods = item["dialect_mods"]

    result, inferred_dialect = parser._split_keyvals(attr_str)
    result = dict(result)
    assert result == attr_dict, result

    reconstructed = parser._reconstruct(result, inferred_dialect, keep_order=True)
    if acceptable_reconstruction:
        assert reconstructed == acceptable_reconstruction, reconstructed
    else:
        assert reconstructed == attr_str, reconstructed

    # Get the default dialect for comparison, and update it with any diffs
    # indicated by the test case
    default_dialect = constants.dialect.copy()
    default_dialect.update(dialect_mods)

    print(inferred_dialect)
    print(dialect_mods)
    print(attr_str)
    assert default_dialect == inferred_dialect



def parser_smoke_test():
    """
    Just confirm we can iterate completely through the test files....
    """
    # Don't show the warnings for tests
    import logging

    parser.logger.setLevel(logging.CRITICAL)
    for filename in TEST_FILENAMES:
        p = iterators._FileIterator(filename)
        for _ in p:
            continue


def test_empty_recontruct():
    """
    reconstructing attributes with incomplete information returns empty string
    """
    assert parser._reconstruct(None, constants.dialect) == ""
    with pytest.raises(exceptions.AttributeStringError):
        parser._reconstruct(dict(ID="asdf"), None)
    with pytest.raises(exceptions.AttributeStringError):
        parser._reconstruct(None, None)


def test_empty_split_keyvals():
    attrs, dialect = parser._split_keyvals(keyval_str=None)
    assert attrs == feature.dict_class()
    # assert dialect == constants.dialect
    assert dialect is None


def test_repeated_keys_conflict():
    """
    if dialect says repeated keys, but len(vals) > 1, then the keys are not
    actually repeated....
    """
    #
    # This is now only checked when infer_dialect is True -- and is disabled
    # when a dialect is provided
    #
    # dialect = constants.dialect.copy()
    # dialect['repeated keys'] = True
    # assert_raises(exceptions.AttributeStringError, parser._split_keyvals, "Parent=1,2,3", dialect)


def test_parser_from_string():
    # DEPRECATED
    #
    # _StringIterator has been removed and is instead handled by DataIterator
    # creating a temp file and returning a _FileIterator.
    pass


def test_valid_line_count():
    p = iterators._FileIterator(example_filename("ncbi_gff3.txt"))
    assert len(list(p)) == 17

    p = iterators._FileIterator(example_filename("hybrid1.gff3"))
    assert len(list(p)) == 6

    p = iterators._FileIterator(example_filename("FBgn0031208.gff"))
    assert len(list(p)) == 27


def test_inconsistent_dialect():
    """
    The second feature does not have a trailing semicolon (wormbase_gff2_alt is
    like this).  But since the first feature does, that's what the db's dialect
    is set to, which can cause errors when parsing attributes.
    """
    db = create.create_db(
        """
    chr1	.	gene	1	100	.	+	.	gene_id "gene1";
    chr1	.	mRNA	1	100	.	+	.	transcript_id "mRNA1"
    """,
        ":memory:",
        from_string=True,
    )
    items = list(db.all_features())
    print(items[0])
    # before, was ['"mRNA1'] -- note extra "
    assert items[1].attributes["transcript_id"] == ["mRNA1"], items[1].attributes[
        "transcript_id"
    ]


def test_mixed_quoting_roundtrip():
    """
    A file that mixes quoted and unquoted values must round-trip exactly. In
    particular, the first feature's quoting convention cannot be applied to
    features later in the file, and per-key markers keep each attribute's own
    convention.
    """
    # Every line uses the "; " field separator and has multiple attributes so
    # the field-separator and ordering heuristics are unambiguous; this leaves
    # only the quoting convention under test.
    lines = [
        'chr1\t.\tgene\t1\t100\t.\t+\t.\tgene_id "gene1"; gene_name "g";',
        'chr1\t.\tmRNA\t1\t100\t.\t+\t.\tgene_id "gene1"; transcript_id t1;',
        'chr1\t.\texon\t1\t50\t.\t+\t.\tgene_id "gene1"; name "fgenesh"; transcriptId 873;',
    ]
    data = "\n".join(lines) + "\n"
    db = create.create_db(data, ":memory:", from_string=True, keep_order=True)
    observed = [str(f).split("\t")[8] for f in db.all_features()]
    assert observed == [line.split("\t")[8] for line in lines]

    # The file-level dialect remembers that transcript_id/transcriptId were
    # unquoted even though gene_id/gene_name/name were quoted.
    assert db.dialect["quoted GFF2 values"] is True
    assert db.dialect["quoted value markers"]["transcript_id"] is None
    assert db.dialect["quoted value markers"]["transcriptId"] is None
    assert "gene_id" not in db.dialect["quoted value markers"]


def test_encoded_comma_is_not_a_separator():
    """
    Percent-encoded separators are decoded only after all structural splitting,
    so an encoded comma (%2C) remains part of its value and round-trips
    unchanged, both alone and alongside real list commas.
    """
    single = "Note=a%2Cb;ID=x"
    attrs, dialect = parser._split_keyvals(single)
    assert attrs == {"Note": ["a,b"], "ID": ["x"]}
    assert parser._reconstruct(attrs, dialect, keep_order=True) == single

    mixed = "Note=one,two%2Cthree,four"
    attrs, dialect = parser._split_keyvals(mixed)
    assert attrs == {"Note": ["one", "two,three", "four"]}
    assert parser._reconstruct(attrs, dialect, keep_order=True) == mixed


def test_inconsistent_repeated_key_quoting_keeps_clean_values():
    """
    When the same repeated key is quoted in opposite orders across features, no
    single per-occurrence rule can describe the whole file. The parser must
    still extract clean values (literal quote characters must never leak into
    the parsed dictionary); only the re-quoting falls back to the file-wide
    convention.
    """
    data = "\n".join(
        [
            'chr1\t.\ta\t1\t2\t.\t+\t.\tx "1"; x 2',
            'chr1\t.\ta\t1\t2\t.\t+\t.\tx 3; x "4"',
        ]
    ) + "\n"
    db = create.create_db(data, ":memory:", from_string=True)
    # No reliable per-occurrence marker can be shared across the two features.
    assert db.dialect["quoted value markers"] is None
    values = [list(f.attributes["x"]) for f in db.all_features()]
    assert values == [["1", "2"], ["3", "4"]]
    for feature in db.all_features():
        for val in feature.attributes["x"]:
            assert not val.startswith('"') and not val.endswith('"')


def test_attributes():
    s = "chr2L	FlyBase	mRNA	7529	9484	.	+	.	ID=FBtr0300690;Name=CG11023-RC;Parent=FBgn0031208;"
    f = feature.feature_from_line(s)
    f.keep_order = True
    assert str(f) == s, str(f)
