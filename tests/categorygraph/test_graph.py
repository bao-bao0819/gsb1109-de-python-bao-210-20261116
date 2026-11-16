import pytest

from categorygraph import (
    AncestorConclusion,
    EvidenceHop,
    NamespaceDeclaration,
)
from categorygraph.graph import build_graph

NS = NamespaceDeclaration('Category', aliases=('Kategorie',))


def graph(pages):
    return build_graph(pages, NS)


def test_member_and_parent_edges_are_generated():
    result = graph(
        {
            'Dog': '[[Category:Mammals]] some text',
            'Category:Mammals': '[[Category:Animals]]',
            'Category:Animals': '[[Category:Life]]',
        }
    )
    assert ('Dog', 'Mammals') in {
        (edge.page, edge.category) for edge in result.member_edges
    }
    assert result.parents_of('Mammals')[0].parent == 'Animals'
    assert result.parents_of('Mammals')[0].source_page == 'Category:Mammals'
    assert result.members_of('Mammals') == frozenset({'Dog'})


def test_missing_category_means_unknown_parents_not_empty_graph():
    result = graph({'Dog': '[[Category:Mammals]]'})
    assert result.missing_categories == frozenset({'Mammals'})
    assert result.parents_of('Mammals') == ()
    assert result.members_of('Mammals') == frozenset({'Dog'})


def test_ancestor_conclusions_keep_source_page_evidence():
    result = graph(
        {
            'Dog': '[[Category:Mammals]]',
            'Category:Mammals': '[[Category:Animals]]',
            'Category:Animals': '[[Category:Life]]',
        }
    )
    conclusions = {c.ancestor: c for c in result.ancestors_of('Dog')}
    assert set(conclusions) == {'Mammals', 'Animals', 'Life'}
    assert conclusions['Life'].chains == (
        (
            EvidenceHop('Dog', 'Mammals'),
            EvidenceHop('Category:Mammals', 'Animals'),
            EvidenceHop('Category:Animals', 'Life'),
        ),
    )
    assert all(c.page == 'Dog' for c in conclusions.values())


def test_diamond_reports_each_ancestor_once_with_both_chains():
    result = graph(
        {
            'Dog': '[[Category:Mammals]]',
            'Category:Mammals': (
                '[[Category:LandAnimals]] [[Category:WaterAnimals]]'
            ),
            'Category:LandAnimals': '[[Category:Animals]]',
            'Category:WaterAnimals': '[[Category:Animals]]',
            'Category:Animals': '',
        }
    )
    conclusions = result.ancestors_of('Dog')
    animal = next(c for c in conclusions if c.ancestor == 'Animals')
    assert isinstance(animal, AncestorConclusion)
    assert len(animal.chains) == 2
    chain_pages = {
        tuple(hop.source_page for hop in chain) for chain in animal.chains
    }
    assert chain_pages == {
        ('Dog', 'Category:Mammals', 'Category:LandAnimals'),
        ('Dog', 'Category:Mammals', 'Category:WaterAnimals'),
    }
    assert sum(c.ancestor == 'Animals' for c in conclusions) == 1


def test_cycle_is_reported_not_broken_into_a_tree():
    result = graph(
        {
            'Dog': '[[Category:A]]',
            'Category:A': '[[Category:B]]',
            'Category:B': '[[Category:A]] [[Category:Root]]',
            'Category:Root': '',
        }
    )
    assert result.is_cyclic()
    assert frozenset({'A', 'B'}) in result.cyclic_groups
    assert {(edge.category, edge.parent) for edge in result.parent_edges} == {
        ('A', 'B'),
        ('B', 'A'),
        ('B', 'Root'),
    }
    ancestors = {c.ancestor for c in result.ancestors_of('Dog')}
    assert ancestors == {'A', 'B', 'Root'}


def test_self_loop_is_a_cyclic_group():
    result = graph(
        {
            'Category:Loop': '[[Category:Loop]]',
        }
    )
    assert result.cyclic_groups == (frozenset({'Loop'}),)
    assert result.parents_of('Loop')[0].parent == 'Loop'


def test_cycle_does_not_infinite_loop_and_chain_stops_at_repeat():
    result = graph(
        {
            'Dog': '[[Category:A]]',
            'Category:A': '[[Category:B]]',
            'Category:B': '[[Category:A]]',
        }
    )
    conclusions = {c.ancestor: c for c in result.ancestors_of('Dog')}
    chain_lengths = {
        ancestor: len(conclusions[ancestor].chains) for ancestor in conclusions
    }
    assert chain_lengths == {'A': 1, 'B': 1}


def test_escaped_and_protected_links_create_no_edges():
    result = graph(
        {
            'Cat': (
                '[[:Category:JustALink]]'
                '<nowiki>[[Category:Protected]]</nowiki>'
                '[[Category:Real]]'
            ),
            'Category:Real': '',
        }
    )
    assert result.members_of('JustALink') == frozenset()
    assert result.members_of('Protected') == frozenset()
    assert result.members_of('Real') == frozenset({'Cat'})
    assert result.missing_categories == frozenset()


def test_category_page_is_also_member_of_its_parents():
    result = graph({'Category:A': '[[Category:B]]', 'Category:B': ''})
    assert result.members_of('B') == frozenset({'Category:A'})
    assert ('A', 'B') in {
        (edge.category, edge.parent) for edge in result.parent_edges
    }


def test_pages_without_category_links_have_no_conclusions():
    result = graph({'Plain': 'no links here', 'Category:B': ''})
    assert result.ancestors_of('Plain') == ()


def test_unknown_page_lookup_returns_empty():
    result = graph({'Dog': '[[Category:A]]', 'Category:A': ''})
    assert result.ancestors_of('Ghost') == ()
    assert result.members_of('Ghost') == frozenset()


def test_title_normalization_uses_site_namespace():
    german = NamespaceDeclaration('Kategorie', aliases=('Category',))
    result = build_graph(
        {
            'Hund': '[[Kategorie:Saeugetiere]]',
            'Kategorie:Saeugetiere': '[[Kategorie:Lebewesen]]',
            'Kategorie:Lebewesen': '',
        },
        german,
    )
    assert result.members_of('Saeugetiere') == frozenset({'Hund'})
    assert {c.ancestor for c in result.ancestors_of('Hund')} == {
        'Saeugetiere',
        'Lebewesen',
    }


def test_underscore_titles_are_normalized():
    result = graph(
        {
            'Red_Fox': '[[Category:Wild_Animals]]',
            'Category:Wild Animals': '',
        }
    )
    assert result.members_of('Wild Animals') == frozenset({'Red Fox'})
    assert {c.ancestor for c in result.ancestors_of('Red_Fox')} == {
        'Wild Animals'
    }


if __name__ == '__main__':
    pytest.main([__file__])
