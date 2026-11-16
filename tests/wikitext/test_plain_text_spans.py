from wikitextparser import PlainText, PlainTextChar, parse, remove_markup


def test_plain_text_still_compares_as_str():
    result = parse("'''a'''<!--x-->[[b|c]]").plain_text()
    assert isinstance(result, PlainText)
    assert isinstance(result, str)
    assert result == 'ac'
    assert result.upper() == 'AC'
    assert result[0] == 'a'
    assert remove_markup("'''a'''<!--x-->[[b|c]]") == 'ac'


def test_result_can_be_copied_and_pickled():
    import copy
    import pickle

    result = parse('a&amp;b').plain_text()
    for other in (
        copy.copy(result),
        copy.deepcopy(result),
        pickle.loads(pickle.dumps(result)),
    ):
        assert other == result
        assert other.chars == result.chars


def test_each_visible_character_maps_to_its_source_region():
    result = parse('ab[[c|d]]e').plain_text()
    assert result == 'abde'
    assert result.chars == (
        PlainTextChar('a', ((0, 1),)),
        PlainTextChar('b', ((1, 2),)),
        PlainTextChar('d', ((6, 7),)),
        PlainTextChar('e', ((9, 10),)),
    )
    assert result.char_spans == ((0, 1), (1, 2), (6, 7), (9, 10))


def test_one_visible_character_may_come_from_a_multi_character_entity():
    result = parse('&Sigma;x&#931;y&#x3a3;').plain_text()
    assert result == 'ΣxΣyΣ'
    assert result.chars[0] == PlainTextChar('Σ', ((0, 7),))
    assert result.source_spans(0, 1) == [(0, 7)]
    assert result.source_spans(2, 3) == [(8, 14)]
    assert result.source_spans(4, 5) == [(15, 22)]


def test_source_region_can_produce_no_text():
    result = parse('a<!--hidden-->b').plain_text()
    assert result == 'ab'
    assert result.chars[1] == PlainTextChar('b', ((14, 15),))
    assert result.source_spans(1, 2) == [(14, 15)]


def test_hit_across_a_hidden_block_returns_separate_ranges():
    result = parse('a<!--hidden-->b').plain_text()
    assert result.source_spans() == [(0, 1), (14, 15)]
    assert result.source_spans(0, 2) == [(0, 1), (14, 15)]
    assert result.source_spans(0, 1) == [(0, 1)]


def test_hit_across_a_hidden_template_returns_separate_ranges():
    result = parse('ab{{hidden}}cd').plain_text()
    assert result == 'abcd'
    assert result.source_spans(1, 3) == [(1, 2), (12, 13)]


def test_adjacent_source_regions_are_merged():
    result = parse('<b>ab</b>').plain_text()
    assert result == 'ab'
    assert result.source_spans() == [(3, 5)]


def test_selection_at_the_end_returns_a_zero_width_point():
    result = parse('a<!--hidden-->').plain_text()
    assert result == 'a'
    assert result.source_spans(1, 1) == [(14, 14)]


def test_empty_selection_returns_an_insertion_point():
    result = parse('a[[b|c]]').plain_text()
    assert result == 'ac'
    assert result.source_spans(0, 0) == [(0, 0)]
    assert result.source_spans(1, 1) == [(5, 5)]
    assert result.source_spans(2, 2) == [(8, 8)]

    empty = parse('{{t}}').plain_text()
    assert empty == ''
    assert empty.source_spans() == [(5, 5)]


def test_negative_indexes_follow_str_slicing():
    result = parse('abc').plain_text()
    assert result.source_spans(-2, -1) == [(1, 2)]
    assert result.source_spans(-1) == [(2, 3)]


def test_out_of_range_and_reversed_selections_raise_index_error():
    result = parse('abc').plain_text()
    for args in ((-4, 1), (0, 4), (2, 1)):
        try:
            result.source_spans(*args)
        except IndexError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f'{args} did not raise IndexError')


def test_offsets_are_relative_to_the_root_wikitext():
    parsed = parse('xx [[a|bc]] yy')
    link = parsed.wikilinks[0]
    result = link.plain_text()
    assert result == 'bc'
    assert result.chars == (
        PlainTextChar('b', ((7, 8),)),
        PlainTextChar('c', ((8, 9),)),
    )
    assert result.source_start == link.span[0]
    assert result.source_end == link.span[1]


def test_replacements_map_to_the_replaced_source():
    result = parse('a{{t}}b').plain_text(replace_templates=lambda t: 'T')
    assert result == 'aTb'
    assert result.chars[1] == PlainTextChar('T', ((1, 6),))

    result = parse('a{{t}}b').plain_text(replace_templates=lambda t: None)
    assert result == 'ab'
    assert result.chars == (
        PlainTextChar('a', ((0, 1),)),
        PlainTextChar('b', ((6, 7),)),
    )


def test_entity_inside_replacement_maps_to_the_replacement():
    result = parse('x{{t}}y').plain_text(
        replace_templates=lambda t: '&nbsp;'
    )
    assert result == 'x\xa0y'
    assert result.chars[1] == PlainTextChar('\xa0', ((1, 6),))


def test_ambiguous_named_reference_matches_html_unescape():
    result = parse('a&notit;b').plain_text()
    assert result == 'a¬it;b'
    assert result.chars[1] == PlainTextChar('¬', ((1, 5),))
    assert result.chars[2] == PlainTextChar('i', ((5, 6),))


def test_disabling_entity_unescaping_keeps_individual_offsets():
    result = parse('&Sigma;').plain_text(unescape_html_entities=False)
    assert result == '&Sigma;'
    assert result.chars[0] == PlainTextChar('&', ((0, 1),))
    assert result.chars[-1] == PlainTextChar(';', ((6, 7),))


def test_table_output_maps_to_the_table_source():
    result = parse('a\n{|\n|A\n|B\n|}\nb').plain_text()
    assert result == 'a\n\nA\tB\n\nb'
    table = parse('a\n{|\n|A\n|B\n|}\nb').tables[0]
    assert result.source_spans(2, 5) == [table.span]
