# -*- coding: utf-8 -*-
import random
import unittest

from coo_aggregation import (
    ACCUMULATE,
    FORBID_DUPLICATE,
    ABSENT,
    MISSING,
    ZERO,
    CONFLICT,
    LineState,
    Contribution,
    COOAggregator,
    COOAggregationError,
    merge_contributions,
    to_dense,
    from_dense,
)


def c(coord, source, value, policy, delivery, fragment=None):
    return Contribution(coord, source, value, policy, delivery, fragment)


class TestContribution(unittest.TestCase):

    def test_validation(self):
        with self.assertRaises(COOAggregationError):
            Contribution((0,), 's', 1, 'unknown', 'd1')
        with self.assertRaises(COOAggregationError):
            Contribution((), 's', 1, ACCUMULATE, 'd1')
        with self.assertRaises(COOAggregationError):
            Contribution((0,), 's', 'x', ACCUMULATE, 'd1')

    def test_dedup_key_ignores_fragment(self):
        first = c((0, 0), 's1', 1, ACCUMULATE, 'd1', fragment='f1')
        retry = c((0, 0), 's1', 1, ACCUMULATE, 'd1', fragment='f2')
        self.assertEqual(first.dedup_key, retry.dedup_key)


class TestAccumulate(unittest.TestCase):

    def test_multiple_sources_accumulate(self):
        lines = merge_contributions([
            c((0, 0), 's1', 2, ACCUMULATE, 'd1'),
            c((0, 0), 's2', 3, ACCUMULATE, 'd2'),
            c((0, 1), 's1', 5, ACCUMULATE, 'd3'),
        ])
        self.assertEqual([line.coord for line in lines], [(0, 0), (0, 1)])
        first, second = lines
        self.assertEqual(first.state, LineState.VALUE)
        self.assertEqual(first.value, 5)
        self.assertEqual(len(first.contributions), 2)
        self.assertEqual(second.value, 5)

    def test_sum_exactly_zero_keeps_evidence(self):
        line = merge_contributions([
            c((1, 1), 's1', 3, ACCUMULATE, 'd1'),
            c((1, 1), 's2', -3, ACCUMULATE, 'd2'),
        ])[0]
        self.assertIs(line.state, ZERO)
        self.assertEqual(line.value, 0)
        self.assertEqual(len(line.contributions), 2)

    def test_float_zero_sum(self):
        line = merge_contributions([
            c((0,), 's1', 0.1, ACCUMULATE, 'd1'),
            c((0,), 's2', -0.1, ACCUMULATE, 'd2'),
        ])[0]
        self.assertIs(line.state, ZERO)
        self.assertEqual(line.value, 0.0)

    def test_missing_value_is_distinct_state(self):
        line = merge_contributions([
            c((0,), 's1', 1, ACCUMULATE, 'd1'),
            c((0,), 's2', None, ACCUMULATE, 'd2'),
        ])[0]
        self.assertIs(line.state, MISSING)
        self.assertIsNone(line.value)
        self.assertEqual(len(line.contributions), 2)


class TestForbidDuplicate(unittest.TestCase):

    def test_single_forbid_delivery_is_authoritative(self):
        line = merge_contributions([
            c((0,), 's1', 7, FORBID_DUPLICATE, 'd1'),
        ])[0]
        self.assertEqual(line.state, LineState.VALUE)
        self.assertEqual(line.value, 7)
        self.assertEqual(line.conflicts, ())

    def test_two_sources_forbid_is_conflict(self):
        line = merge_contributions([
            c((0,), 's1', 7, FORBID_DUPLICATE, 'd1'),
            c((0,), 's2', 7, FORBID_DUPLICATE, 'd2'),
        ])[0]
        self.assertIs(line.state, CONFLICT)
        self.assertIsNone(line.value)
        self.assertTrue(line.conflicts)
        self.assertEqual(len(line.contributions), 2)

    def test_forbid_against_accumulate_is_conflict(self):
        line = merge_contributions([
            c((0,), 's1', 7, FORBID_DUPLICATE, 'd1'),
            c((0,), 's2', 1, ACCUMULATE, 'd2'),
        ])[0]
        self.assertIs(line.state, CONFLICT)

    def test_same_source_two_deliveries_under_forbid_is_conflict(self):
        line = merge_contributions([
            c((0,), 's1', 7, FORBID_DUPLICATE, 'd1'),
            c((0,), 's1', 8, FORBID_DUPLICATE, 'd2'),
        ])[0]
        self.assertIs(line.state, CONFLICT)
        self.assertEqual(len(line.contributions), 2)


class TestRedelivery(unittest.TestCase):

    def test_retry_same_fragment_counted_once(self):
        aggregator = COOAggregator()
        aggregator.add(c((0,), 's1', 2, ACCUMULATE, 'd1', 'f1'))
        aggregator.add(c((0,), 's1', 2, ACCUMULATE, 'd1', 'f1'))
        line = aggregator.lines()[0]
        self.assertEqual(line.value, 2)
        self.assertEqual(len(line.contributions), 1)
        self.assertEqual(line.redeliveries, 1)

    def test_retry_other_fragment_counted_once(self):
        # Multi-fragment delivery of the same contribution.
        aggregator = COOAggregator()
        aggregator.add_many([
            c((0,), 's1', 2, ACCUMULATE, 'd1', 'fragment-a'),
            c((0,), 's1', 2, ACCUMULATE, 'd1', 'fragment-b'),
        ])
        line = aggregator.lines()[0]
        self.assertEqual(line.value, 2)
        self.assertEqual(len(line.contributions), 1)
        self.assertEqual(line.redeliveries, 1)

    def test_retry_does_not_double_count_in_sum(self):
        line = merge_contributions([
            c((0,), 's1', 10, ACCUMULATE, 'd1'),
            c((0,), 's1', 10, ACCUMULATE, 'd1'),
            c((0,), 's2', 5, ACCUMULATE, 'd2'),
            c((0,), 's2', 5, ACCUMULATE, 'd2'),
        ])[0]
        self.assertEqual(line.value, 15)
        self.assertEqual(len(line.contributions), 2)
        self.assertEqual(line.redeliveries, 2)

    def test_redelivery_with_changed_payload_is_conflict_with_evidence(self):
        aggregator = COOAggregator()
        aggregator.add(c((0,), 's1', 2, ACCUMULATE, 'd1'))
        aggregator.add(c((0,), 's1', 3, ACCUMULATE, 'd1'))
        line = aggregator.lines()[0]
        self.assertIs(line.state, CONFLICT)
        values = sorted(item.value for item in line.contributions)
        self.assertEqual(values, [2, 3])
        self.assertEqual(line.redeliveries, 1)


class TestOrderIndependence(unittest.TestCase):

    def setUp(self):
        self.contributions = [
            c((0, 0), 's1', 2, ACCUMULATE, 'd1'),
            c((0, 0), 's2', -2, ACCUMULATE, 'd2'),
            c((0, 0), 's3', 4, ACCUMULATE, 'd3'),
            c((0, 1), 's1', 1.5, ACCUMULATE, 'd4'),
            c((0, 1), 's2', 2.25, ACCUMULATE, 'd5'),
            c((1, 0), 's1', 9, FORBID_DUPLICATE, 'd6'),
            c((2, 2), 's1', None, ACCUMULATE, 'd7'),
        ]

    def test_permutations_produce_identical_lines(self):
        baseline = merge_contributions(self.contributions)
        rng = random.Random(1234)
        for _ in range(20):
            shuffled = self.contributions[:]
            rng.shuffle(shuffled)
            self.assertEqual(merge_contributions(shuffled), baseline)

    def test_floating_point_sum_is_deterministic(self):
        values = [0.1, 0.2, 0.3, -0.3, -0.2, -0.1, 1.0]
        rng = random.Random(99)
        results = set()
        for _ in range(10):
            rng.shuffle(values)
            line = merge_contributions(
                [c((0,), 's%d' % i, v, ACCUMULATE, 'd%d' % i)
                 for i, v in enumerate(values)])[0]
            results.add(line.value)
        self.assertEqual(len(results), 1)
        self.assertEqual(results.pop(), 1.0)

    def test_fragment_split_and_merge_matches_single_stream(self):
        halves = [self.contributions[:3], self.contributions[3:]]
        left, right = COOAggregator(), COOAggregator()
        left.add_many(halves[0])
        right.add_many(halves[1])
        merged = left.absorb(right)
        self.assertEqual(merged.lines(),
                         merge_contributions(self.contributions))

    def test_absorb_dedups_shared_retries(self):
        contribution = c((0,), 's1', 4, ACCUMULATE, 'd1')
        left = COOAggregator().add(contribution)
        right = COOAggregator().add(contribution)
        line = left.absorb(right).lines()[0]
        self.assertEqual(line.value, 4)
        self.assertEqual(len(line.contributions), 1)
        self.assertEqual(line.redeliveries, 1)


class TestDenseRoundTrip(unittest.TestCase):

    def setUp(self):
        self.lines = merge_contributions([
            c((0, 0), 's1', 3, ACCUMULATE, 'd1'),
            c((0, 0), 's2', -3, ACCUMULATE, 'd2'),       # genuine zero
            c((0, 2), 's1', None, ACCUMULATE, 'd3'),     # missing
            c((1, 1), 's1', 8, FORBID_DUPLICATE, 'd4'),
            c((2, 1), 's1', 1, FORBID_DUPLICATE, 'd5'),
            c((2, 1), 's2', 2, FORBID_DUPLICATE, 'd6'),  # conflict
        ])
        # (0,1), (1,0), (1,2), (2,0), (2,2) are absent.

    def test_dense_cells_keep_four_states_distinct(self):
        grid = to_dense(self.lines, (3, 3))
        self.assertEqual(grid[0][0], 0)
        self.assertIs(grid[0][1], ABSENT)
        self.assertIs(grid[0][2], MISSING)
        self.assertIs(grid[1][0], ABSENT)
        self.assertEqual(grid[1][1], 8)
        self.assertIs(grid[2][1], CONFLICT)
        self.assertIs(grid[2][2], ABSENT)

    def test_round_trip_preserves_states(self):
        grid = to_dense(self.lines, (3, 3))
        rebuilt = from_dense(grid)
        by_coord = {line.coord: line for line in rebuilt}
        self.assertNotIn((0, 1), by_coord)
        self.assertIs(by_coord[(0, 0)].state, ZERO)
        self.assertEqual(by_coord[(0, 0)].value, 0)
        self.assertIs(by_coord[(0, 2)].state, MISSING)
        self.assertIs(by_coord[(1, 1)].state, LineState.VALUE)
        self.assertEqual(by_coord[(1, 1)].value, 8)
        self.assertIs(by_coord[(2, 1)].state, CONFLICT)

    def test_round_trip_is_stable(self):
        grid = to_dense(self.lines, (3, 3))
        once = from_dense(grid)
        twice = from_dense(to_dense(once, (3, 3)))
        self.assertEqual(once, twice)

    def test_one_dimensional_grid(self):
        lines = merge_contributions([
            c((0,), 's1', 1, ACCUMULATE, 'd1'),
            c((2,), 's1', -1, ACCUMULATE, 'd2'),
            c((2,), 's2', 1, ACCUMULATE, 'd3'),
        ])
        grid = to_dense(lines, (4,))
        self.assertEqual(grid, [1, ABSENT, 0, ABSENT])
        rebuilt = {line.coord: line for line in from_dense(grid)}
        self.assertIs(rebuilt[(2,)].state, ZERO)
        self.assertNotIn((1,), rebuilt)

    def test_out_of_range_and_bad_shape(self):
        with self.assertRaises(COOAggregationError):
            to_dense(self.lines, (2, 3))
        with self.assertRaises(COOAggregationError):
            to_dense(self.lines, (3,))
        with self.assertRaises(COOAggregationError):
            to_dense([], (0,))

    def test_from_dense_rejects_garbage(self):
        with self.assertRaises(COOAggregationError):
            from_dense(['nope'])


class TestEvidenceRetention(unittest.TestCase):

    def test_conflict_line_retains_all_contributions(self):
        line = merge_contributions([
            c((0,), 's1', 1, ACCUMULATE, 'd1'),
            c((0,), 's2', 2, FORBID_DUPLICATE, 'd2'),
            c((0,), 's3', 3, ACCUMULATE, 'd3'),
        ])[0]
        self.assertIs(line.state, CONFLICT)
        self.assertEqual(len(line.contributions), 3)
        self.assertEqual(
            sorted(item.source for item in line.contributions),
            ['s1', 's2', 's3'])

    def test_get_returns_none_for_absent_coordinate(self):
        aggregator = COOAggregator()
        aggregator.add(c((0,), 's1', 1, ACCUMULATE, 'd1'))
        self.assertIsNone(aggregator.get((9,)))
        self.assertEqual(aggregator.get((0,)).value, 1)


if __name__ == '__main__':
    unittest.main()
