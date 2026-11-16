from wikitextparser import (
    Ambiguity,
    AmbiguityCandidate,
    StyleRange,
    ViewSegment,
    WikiText,
    entity_view,
)

BOLD = frozenset({'bold'})
ITALIC = frozenset({'italic'})
BOLD_ITALIC = frozenset({'bold', 'italic'})


def test_apostrophe_name_is_literal():
    view = entity_view("O'Brien's")
    assert view.text == "O'Brien's"
    assert view.segments == (ViewSegment(0, 9, 0, 9),)
    assert view.styles == ()
    assert view.ambiguities == ()
    assert view.source_span(0, 7) == (0, 7)


def test_trailing_possessive_is_literal():
    view = entity_view("the Joneses' company")
    assert view.text == "the Joneses' company"
    assert view.ambiguities == ()


def test_bold_name():
    view = entity_view("'''ACME''' acquired")
    assert view.text == 'ACME acquired'
    assert view.segments == (
        ViewSegment(0, 4, 3, 7),
        ViewSegment(4, 13, 10, 19),
    )
    assert view.styles == (StyleRange(0, 4, 3, 7, BOLD),)
    assert view.ambiguities == ()
    assert view.source_span(0, 4) == (3, 7)


def test_italic_name_with_apostrophe():
    view = entity_view("''Lloyd's''")
    assert view.text == "Lloyd's"
    assert view.styles == (StyleRange(0, 7, 2, 9, ITALIC),)
    assert view.ambiguities == ()


def test_bold_italic_name_with_apostrophe():
    view = entity_view("'''''O'Connell'''''")
    assert view.text == "O'Connell"
    assert view.styles == (StyleRange(0, 9, 5, 14, BOLD_ITALIC),)
    assert view.ambiguities == ()
    assert view.source_span(0, 9) == (5, 14)


def test_adjacent_styles():
    view = entity_view("'''A''' ''B''")
    assert view.text == 'A B'
    assert view.styles == (
        StyleRange(0, 1, 3, 4, BOLD),
        StyleRange(2, 3, 10, 11, ITALIC),
    )


def test_four_quotes_are_ambiguous_not_dropped():
    view = entity_view("ACME''''Inc")
    # primary interpretation: literal apostrophe + bold toggle
    assert view.text == "ACME'Inc"
    assert view.styles == (StyleRange(5, 8, 8, 11, BOLD),)
    (ambiguity,) = view.ambiguities
    assert ambiguity == Ambiguity(
        4,
        5,
        4,
        8,
        (
            AmbiguityCandidate("'", ('bold',)),
            AmbiguityCandidate("''''", ()),
            AmbiguityCandidate('', ('italic', 'italic')),
            AmbiguityCandidate("''", ('italic',)),
        ),
    )
    # the affected source range is the quote run itself
    assert view.source_span(ambiguity.view_start, ambiguity.view_end) == (4, 5)


def test_six_quotes_are_ambiguous():
    view = entity_view("''''''")
    assert view.text == "'"
    (ambiguity,) = view.ambiguities
    assert (ambiguity.source_start, ambiguity.source_end) == (0, 6)
    assert ambiguity.candidates[0] == AmbiguityCandidate(
        "'", ('bold', 'italic')
    )
    assert AmbiguityCandidate("''''''", ()) in ambiguity.candidates


def test_unambiguous_names_indexed_despite_ambiguity():
    view = entity_view("O'Brien '''' ACME")
    assert view.text == "O'Brien ' ACME"
    assert len(view.ambiguities) == 1
    # the unambiguous names are still in the view, mapped to the source
    assert view.source_span(0, 7) == (0, 7)
    acme_view_start = view.text.index('ACME')
    assert view.source_span(acme_view_start, acme_view_start + 4) == (13, 17)
    assert view.styles == (StyleRange(9, 14, 12, 17, BOLD),)


def test_comments_are_removed():
    view = entity_view('A<!-- hidden -->B')
    assert view.text == 'AB'
    assert view.segments == (
        ViewSegment(0, 1, 0, 1),
        ViewSegment(1, 2, 16, 17),
    )
    assert view.source_span(0, 2) == (0, 17)


def test_comments_between_quotes_are_transparent():
    view = entity_view("''<!--x-->''i'''")
    # '' + comment + '' is a 4-quote run, i.e. ambiguous
    assert view.text == "'i"
    assert len(view.ambiguities) == 1
    assert view.styles == (StyleRange(1, 2, 12, 13, BOLD),)


def test_wikitext_input_and_method():
    wt = WikiText("'''ACME''' and O'Brien")
    view = wt.entity_view()
    assert view.text == "ACME and O'Brien"
    assert view.styles == (StyleRange(0, 4, 3, 7, BOLD),)
    assert view.source_span(9, 16) == (15, 22)


def test_empty_string():
    view = entity_view('')
    assert view.text == ''
    assert view.segments == ()
    assert view.styles == ()
    assert view.ambiguities == ()
    assert view.source_span(0) is None


def test_source_span_clamps_markup_gaps():
    view = entity_view("'''ACME'''")
    # positions inside the removed quote tokens clamp to the nearest
    # mapped source position
    assert view.source_span(0, 4) == (3, 7)
