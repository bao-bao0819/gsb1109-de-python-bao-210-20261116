# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from unittest import TestCase

from hwp5.treeop import STARTEVENT, ENDEVENT
from hwp5.treeop import MissingAncestor
from hwp5.treeop import build_subtree
from hwp5.treeop import is_missing_ancestor
from hwp5.treeop import iter_subevents
from hwp5.treeop import prefix_ancestors_from_level
from hwp5.treeop import prefix_event
from hwp5.treeop import tree_events


class Test_ancestors_from_level(TestCase):

    def test_ancestors_from_level(self):

        level_prefixed = [
            (0, 'a0'),
            (0, 'b0'),   # sibling
            (1, 'b0-a1'),  # child
            (2, 'b0-a1-a2'),  # child
            (1, 'b0-b1'),  # jump to parent level
            (2, 'b0-b1-b2'),  # child
            (0, 'c0'),  # jump to grand-parent level
        ]

        ancestors_prefixed = prefix_ancestors_from_level(level_prefixed)
        result = list((list(ancestors), item)
                      for ancestors, item in ancestors_prefixed)

        self.assertEqual(result.pop(0), ([None], 'a0'))
        self.assertEqual(result.pop(0), ([None], 'b0'))
        self.assertEqual(result.pop(0), ([None, 'b0'], 'b0-a1'))
        self.assertEqual(result.pop(0), ([None, 'b0', 'b0-a1'], 'b0-a1-a2'))
        self.assertEqual(result.pop(0), ([None, 'b0'], 'b0-b1'))
        self.assertEqual(result.pop(0), ([None, 'b0', 'b0-b1'], 'b0-b1-b2'))
        self.assertEqual(result.pop(0), ([None], 'c0'))

    def test_ancestors_from_level_from_nonzero_baselevel(self):
        level_prefixed = [
            (7, 'a0'),  # baselevel 7
            (8, 'a0-a1'),
            (9, 'a0-a1-a2'),
            (7, 'b0'),
        ]
        ancestors_prefixed = prefix_ancestors_from_level(level_prefixed)
        result = list((list(ancestors), item)
                      for ancestors, item in ancestors_prefixed)
        self.assertEqual(result.pop(0), ([None], 'a0'))
        self.assertEqual(result.pop(0), ([None, 'a0'], 'a0-a1'))
        self.assertEqual(result.pop(0), ([None, 'a0', 'a0-a1'], 'a0-a1-a2'))
        self.assertEqual(result.pop(0), ([None], 'b0'))

    def test_ancestors_from_level_fails_at_level_below_baselevel(self):
        level_prefixed = [
            (7, 'a7'),  # baselevel 7
            (8, 'a7-a8'),
            (9, 'a7-a8-a9'),
            (6, 'b7'),  # level below the base level
        ]
        try:
            list(prefix_ancestors_from_level(level_prefixed))
            # TODO: 현재로서는 스택에 기본으로 root_item을 넣어놓는 구현 방식 때문에
            # base level 바로 아래 level은 에러가 나지 않음
            # self.fail('exception expected')
        except:
            pass

        level_prefixed = [
            (7, 'a7'),  # baselevel 7
            (8, 'a7-a8'),
            (9, 'a7-a8-a9'),
            (5, 'b7'),  # level below the base level
        ]
        try:
            list(prefix_ancestors_from_level(level_prefixed))
            self.fail('exception expected')
        except:
            pass

    def test_ancestors_from_level_marks_missing_ancestor_on_jump(self):
        level_prefixed = [
            (1, 'a1'),
            (3, 'a1-c3'),  # the level-2 ancestor is missing
            (1, 'b1'),  # later sibling must not be captured by the gap
        ]
        result = list((list(ancestors), item)
                      for ancestors, item
                      in prefix_ancestors_from_level(level_prefixed))

        ancestors_a1, item = result[0]
        self.assertEqual('a1', item)
        self.assertEqual([None], ancestors_a1)

        ancestors_c3, item = result[1]
        self.assertEqual('a1-c3', item)
        self.assertEqual('a1', ancestors_c3[1])
        self.assertTrue(is_missing_ancestor(ancestors_c3[2]))
        self.assertEqual(2, ancestors_c3[2]['level'])

        # the later level-1 record keeps the same ancestors as a1:
        # it is not adopted by the placeholder
        ancestors_b1, item = result[2]
        self.assertEqual('b1', item)
        self.assertEqual([None], ancestors_b1)

    def test_prefix_event_inserts_missing_ancestor_for_each_gap(self):
        level_prefixed = [
            (0, 'a0'),
            (3, 'a0-d3'),  # two missing levels: 1 and 2
        ]
        events = list(prefix_event(level_prefixed))

        placeholders = [item for event, item in events
                        if event is STARTEVENT and is_missing_ancestor(item)]
        self.assertEqual([1, 2], [p['level'] for p in placeholders])
        self.assertTrue(all(isinstance(p, MissingAncestor)
                            for p in placeholders))

        # the emitted events stay balanced
        depth = 0
        for event, item in events:
            if event is STARTEVENT:
                depth += 1
            elif event is ENDEVENT:
                depth -= 1
            self.assertGreaterEqual(depth, 0)
        self.assertEqual(0, depth)

    def test_prefix_event_keeps_later_sibling_out_of_placeholder(self):
        level_prefixed = [
            (1, 'a1'),
            (3, 'a1-c3'),
            (1, 'b1'),
            (2, 'b1-d2'),
        ]
        events = list(prefix_event(level_prefixed))
        sequence = [('S' if event is STARTEVENT else 'E',
                     ('MISS@%d' % item['level'])
                     if is_missing_ancestor(item) else item)
                    for event, item in events]

        self.assertEqual(sequence, [
            ('S', 'a1'),
            ('S', 'MISS@2'),
            ('S', 'a1-c3'),
            ('E', 'a1-c3'),
            ('E', 'MISS@2'),
            ('E', 'a1'),
            ('S', 'b1'),
            ('S', 'b1-d2'),
            ('E', 'b1-d2'),
            ('E', 'b1'),
        ])

    def test_prefix_event_emits_no_placeholder_for_wellformed_stream(self):
        level_prefixed = [
            (0, 'a0'),
            (0, 'b0'),
            (1, 'b0-a1'),
            (2, 'b0-a1-a2'),
        ]
        events = list(prefix_event(level_prefixed))
        self.assertFalse(any(is_missing_ancestor(item)
                             for event, item in events))


class TestTreeEvents(TestCase):
    def test_tree_events(self):
        event_prefixed_items = [(STARTEVENT, 'a'), (ENDEVENT, 'a')]
        rootitem, childs = build_subtree(iter(event_prefixed_items[1:]))
        self.assertEqual('a', rootitem)
        self.assertEqual(0, len(childs))

        event_prefixed_items = [
            (STARTEVENT, 'a'),
            (STARTEVENT, 'b'),
            (ENDEVENT, 'b'),
            (ENDEVENT, 'a')
        ]
        self.assertEqual(
            ('a', [('b', [])]),
            build_subtree(iter(event_prefixed_items[1:]))
        )

        event_prefixed_items = [
            (STARTEVENT, 'a'),
            (STARTEVENT, 'b'),
            (STARTEVENT, 'c'), (ENDEVENT, 'c'),
            (STARTEVENT, 'd'), (ENDEVENT, 'd'),
            (ENDEVENT, 'b'),
            (ENDEVENT, 'a')
        ]

        result = build_subtree(iter(event_prefixed_items[1:]))
        self.assertEqual(
            ('a', [('b', [('c', []), ('d', [])])]),
            result
        )

        back = list(tree_events(*result))
        self.assertEqual(event_prefixed_items, back)


class TestSubevents(TestCase):

    def test_iter_subevents(self):

        events = iter([(STARTEVENT, 'a'), (ENDEVENT, 'a')])
        next(events)
        subevents = iter_subevents(events)
        self.assertEqual([(ENDEVENT, 'a')], list(subevents))

        events = iter([(STARTEVENT, 'a'),
                       (STARTEVENT, 'b'),
                       (None, 'c'),
                       (ENDEVENT, 'b'),
                       (ENDEVENT, 'a')])
        next(events)
        subevents = iter_subevents(events)
        self.assertEqual([(STARTEVENT, 'b'),
                          (None, 'c'),
                          (ENDEVENT, 'b'),
                          (ENDEVENT, 'a')], list(subevents))

        events = iter([(STARTEVENT, 'a'),
                       (None, 'c'),
                       (ENDEVENT, 'a'),
                       (STARTEVENT, 'b'),
                       (None, 'd'),
                       (ENDEVENT, 'b')])
        next(events)
        subevents = iter_subevents(events)
        self.assertEqual([(None, 'c'),
                          (ENDEVENT, 'a')], list(subevents))
        self.assertEqual([(STARTEVENT, 'b'),
                          (None, 'd'),
                          (ENDEVENT, 'b')], list(events))
