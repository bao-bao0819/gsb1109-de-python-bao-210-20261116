import math
import unittest

import arff
import arff_stats
from arff_stats import (
    ArffBaseline,
    ColumnStatistics,
    ConflictingRevisionError,
    SchemaMismatchError,
)


SPARSE = '''@RELATION sparse
@ATTRIBUTE a REAL
@ATTRIBUTE b REAL
@ATTRIBUTE c REAL
@DATA
{0 2, 1 4, 2 6}
{1 8}
{0 10, 2 ?}
{}
'''


class TestColumnStatistics(unittest.TestCase):
    def test_empty(self):
        column = ColumnStatistics('a')
        snap = column.snapshot()
        self.assertEqual(snap.count, 0)
        self.assertEqual(snap.missing, 0)
        self.assertEqual(snap.sum, 0.0)
        self.assertIsNone(snap.mean)
        self.assertIsNone(snap.variance)

    def test_add_remove_round_trip(self):
        column = ColumnStatistics('a')
        values = [1e16, 1.0, -1e16, 5.5, -2.25]
        for value in values:
            column.add(value)
        for value in values:
            column.remove(value)
        snap = column.snapshot()
        self.assertEqual(snap.count, 0)
        self.assertEqual(snap.sum, 0.0)
        self.assertIsNone(snap.mean)

    def test_remove_absent_value(self):
        column = ColumnStatistics('a')
        self.assertRaises(ValueError, column.remove, 1.0)

    def test_missing_is_not_counted(self):
        column = ColumnStatistics('a')
        column.add(None)
        column.add(float('nan'))
        column.add(4.0)
        snap = column.snapshot()
        self.assertEqual(snap.count, 1)
        self.assertEqual(snap.missing, 2)
        self.assertEqual(snap.mean, 4.0)

    def test_merge_is_additive(self):
        left = ColumnStatistics('a')
        right = ColumnStatistics('a')
        for value in [1e16, 3.0, -1e16]:
            left.add(value)
        for value in [2.0, -4.0]:
            right.add(value)
        left.merge(right)
        self.assertEqual(left.count, 5)
        self.assertAlmostEqual(left.snapshot().sum, 1.0)

    def test_nominal_frequencies(self):
        column = ColumnStatistics('c', numeric=False)
        for value in ['x', 'y', 'x']:
            column.add(value)
        snap = column.snapshot()
        self.assertEqual(snap.count, 3)
        self.assertEqual(dict(snap.frequencies), {'x': 2, 'y': 1})
        self.assertIsNone(snap.mean)
        column.remove('x')
        self.assertRaises(ValueError, column.remove, 'z')


class TestBaselineIngest(unittest.TestCase):
    def test_sparse_omitted_zero_and_explicit_missing(self):
        baseline = ArffBaseline.from_arff(SPARSE, partition='p')
        snap = baseline.snapshot()

        self.assertEqual(snap.sample_count, 4)
        self.assertEqual(snap['a'].count, 4)
        self.assertEqual(snap['a'].sum, 12.0)
        self.assertEqual(snap['b'].count, 4)
        self.assertEqual(snap['b'].sum, 12.0)
        self.assertEqual(snap['c'].count, 3)
        self.assertEqual(snap['c'].missing, 1)
        self.assertEqual(snap['c'].sum, 6.0)

    def test_retransmission_is_counted_once(self):
        baseline = ArffBaseline.from_arff(SPARSE, partition='p')
        baseline.ingest(SPARSE, partition='p')
        self.assertLen(baseline, 4)
        self.assertEqual(baseline.snapshot()['a'].sum, 12.0)

    def assertLen(self, baseline, expected):
        self.assertEqual(len(baseline), expected)

    def test_retract(self):
        baseline = ArffBaseline.from_arff(SPARSE, partition='p')
        self.assertTrue(baseline.retract(('p', 0)))
        self.assertFalse(baseline.retract(('p', 0)))
        snap = baseline.snapshot()
        self.assertEqual(snap['a'].count, 3)
        self.assertEqual(snap['a'].sum, 10.0)

    def test_retract_only_valid_value(self):
        baseline = ArffBaseline(
            [('a', 'REAL'), ('b', 'REAL')]
        )
        baseline.ingest_rows([[7.0, None]], [('k',)])
        snap = baseline.snapshot()
        self.assertEqual(snap['a'].mean, 7.0)

        self.assertTrue(baseline.retract(('k',)))
        snap = baseline.snapshot()
        self.assertEqual(snap['a'].count, 0)
        self.assertEqual(snap['a'].sum, 0.0)
        self.assertIsNone(snap['a'].mean)
        self.assertIsNone(snap['a'].variance)
        self.assertIsNone(snap['a'].minimum)
        self.assertEqual(snap['b'].missing, 0)

    def test_correction_replaces_values(self):
        baseline = ArffBaseline([('a', 'REAL')])
        baseline.ingest_rows([[10.0]], [('k',)])
        baseline.ingest_rows([[4.0]], [('k',)])
        snap = baseline.snapshot()
        self.assertEqual(len(baseline), 1)
        self.assertEqual(snap['a'].sum, 4.0)

    def test_revision_ordering(self):
        baseline = ArffBaseline([('a', 'REAL')])
        baseline.ingest_rows([[10.0]], [('k',)], revision=5)
        self.assertFalse(baseline.upsert(('k',), [99.0], revision=3))
        self.assertEqual(baseline.snapshot()['a'].sum, 10.0)

        self.assertTrue(baseline.upsert(('k',), [40.0], revision=6))
        self.assertEqual(baseline.snapshot()['a'].sum, 40.0)

    def test_same_revision_conflict(self):
        baseline = ArffBaseline([('a', 'REAL')])
        baseline.ingest_rows([[1.0]], [('k',)], revision=2)
        self.assertRaises(
            ConflictingRevisionError,
            baseline.upsert, ('k',), [2.0], 2,
        )

    def test_same_revision_same_values_is_idempotent(self):
        baseline = ArffBaseline([('a', 'REAL')])
        self.assertTrue(baseline.upsert(('k',), [1.0], revision=2))
        self.assertFalse(baseline.upsert(('k',), [1.0], revision=2))
        self.assertEqual(len(baseline), 1)

    def test_lod_rows(self):
        baseline = ArffBaseline([('a', 'REAL'), ('c', ['x', 'y'])])
        baseline.ingest_rows([{0: 3.0}, {1: 'y'}], [('k1',), ('k2',)])
        snap = baseline.snapshot()
        self.assertEqual(snap['a'].count, 2)
        self.assertEqual(snap['a'].sum, 3.0)
        self.assertEqual(dict(snap['c'].frequencies), {'x': 1, 'y': 1})

    def test_wrong_row_length(self):
        baseline = ArffBaseline([('a', 'REAL')])
        self.assertRaises(
            ValueError, baseline.ingest_rows, [[1.0, 2.0]], [('k',)],
        )


class TestSnapshotAndMerge(unittest.TestCase):
    def test_published_snapshot_does_not_change(self):
        baseline = ArffBaseline.from_arff(SPARSE, partition='p')
        published = baseline.snapshot()
        published_sum = published['a'].sum

        baseline.ingest('''@RELATION sparse
@ATTRIBUTE a REAL
@ATTRIBUTE b REAL
@ATTRIBUTE c REAL
@DATA
{0 1000}
''', partition='q')
        baseline.retract(('p', 1))

        self.assertEqual(published['a'].sum, published_sum)
        self.assertEqual(published.sample_count, 4)
        self.assertNotEqual(baseline.snapshot()['a'].sum, published_sum)

    def test_merge_shards(self):
        left = ArffBaseline.from_arff(SPARSE, partition='left')
        right = ArffBaseline.from_arff('''@RELATION sparse
@ATTRIBUTE a REAL
@ATTRIBUTE b REAL
@ATTRIBUTE c REAL
@DATA
{0 1}
{0 3}
''', partition='right')

        merged = left.copy().merge(right)
        snap = merged.snapshot()
        self.assertEqual(snap.sample_count, 6)
        self.assertEqual(snap['a'].sum, 16.0)

    def test_merge_is_order_independent(self):
        left = ArffBaseline.from_arff(SPARSE, partition='left')
        right = ArffBaseline.from_arff('''@RELATION sparse
@ATTRIBUTE a REAL
@ATTRIBUTE b REAL
@ATTRIBUTE c REAL
@DATA
{0 1}
{0 3}
''', partition='right')

        one = left.copy().merge(right).snapshot()
        two = right.copy().merge(left).snapshot()
        self.assertEqual(one.as_dict(), two.as_dict())

    def test_cross_shard_correction(self):
        old = ArffBaseline([('a', 'REAL')])
        old.ingest_rows([[1.0]], [('row',)], revision=1)

        new = ArffBaseline([('a', 'REAL')])
        new.ingest_rows([[9.0]], [('row',)], revision=2)

        merged = old.copy().merge(new)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged.snapshot()['a'].sum, 9.0)

        merged = new.copy().merge(old)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged.snapshot()['a'].sum, 9.0)

    def test_merge_conflicting_revision(self):
        left = ArffBaseline([('a', 'REAL')])
        right = ArffBaseline([('a', 'REAL')])
        left.ingest_rows([[1.0]], [('row',)], revision=1)
        right.ingest_rows([[2.0]], [('row',)], revision=1)
        self.assertRaises(ConflictingRevisionError, left.merge, right)

    def test_merge_rejects_different_schema(self):
        left = ArffBaseline([('a', 'REAL')])
        right = ArffBaseline([('b', 'REAL')])
        self.assertRaises(SchemaMismatchError, left.merge, right)

    def test_retransmitted_withdrawn_row_stays_withdrawn(self):
        baseline = ArffBaseline([('a', 'REAL')])
        baseline.upsert(('k',), [1.0], revision=1)
        baseline.retract(('k',), revision=2)

        baseline.merge(ArffBaseline([('a', 'REAL')]))
        stale = ArffBaseline([('a', 'REAL')])
        stale.upsert(('k',), [1.0], revision=1)
        baseline.merge(stale)

        self.assertEqual(len(baseline), 0)
        self.assertEqual(baseline.snapshot()['a'].count, 0)

        baseline.upsert(('k',), [5.0], revision=3)
        self.assertEqual(baseline.snapshot()['a'].sum, 5.0)

    def test_stale_retraction_does_not_remove_newer_value(self):
        baseline = ArffBaseline([('a', 'REAL')])
        baseline.retract(('k',), revision=2)

        newer = ArffBaseline([('a', 'REAL')])
        newer.upsert(('k',), [9.0], revision=4)
        merged = newer.merge(baseline)
        self.assertEqual(merged.snapshot()['a'].sum, 9.0)

    def test_retraction_and_upsert_conflict_at_same_revision(self):
        retracted = ArffBaseline([('a', 'REAL')])
        updated = ArffBaseline([('a', 'REAL')])
        updated.upsert(('k',), [1.0], revision=1)
        retracted.retract(('k',), revision=2)
        updated.upsert(('k',), [4.0], revision=2)
        self.assertRaises(ConflictingRevisionError, retracted.merge, updated)
        self.assertRaises(ConflictingRevisionError, updated.merge, retracted)

    def test_merge_order_with_tombstone(self):
        retracted = ArffBaseline([('a', 'REAL')])
        retracted.upsert(('k',), [1.0], revision=1)
        retracted.retract(('k',), revision=2)

        original = ArffBaseline([('a', 'REAL')])
        original.upsert(('k',), [1.0], revision=1)

        one = retracted.copy().merge(original).snapshot()
        two = original.copy().merge(retracted).snapshot()
        self.assertEqual(one.as_dict(), two.as_dict())
        self.assertEqual(one['a'].count, 0)

    def test_matches_recomputation_after_cancellation(self):
        baseline = ArffBaseline([('a', 'REAL')])
        key = 0

        def row(value):
            nonlocal key
            key += 1
            baseline.ingest_rows([[value]], [('k', key)])

        for value in [1e16, 1.0, -1e16, 7.5, -2.5, 1e16, -1e16]:
            row(value)

        # Retract the only small positive value, then correct a huge one.
        baseline.retract(('k', 2))
        baseline.ingest_rows([[-1e16 + 3]], [('k', 7)])

        retained = [1e16, -1e16, 7.5, -2.5, 1e16, -1e16 + 3]
        expected_mean = math.fsum(retained) / len(retained)
        snap = baseline.snapshot()
        self.assertEqual(snap['a'].count, len(retained))
        self.assertAlmostEqual(snap['a'].mean, expected_mean, places=9)

        expected_var = math.fsum(
            (value - expected_mean) ** 2 for value in retained
        ) / len(retained)
        self.assertAlmostEqual(snap['a'].variance, expected_var, places=6)

    def test_exposed_from_arff_module(self):
        self.assertIs(arff.ArffBaseline, ArffBaseline)
        self.assertIs(arff.ColumnStatistics, ColumnStatistics)
        self.assertIs(arff.merge_baselines, arff_stats.merge_baselines)


class TestPublicContract(unittest.TestCase):
    def test_snapshot_is_immutable(self):
        baseline = ArffBaseline.from_arff(SPARSE, partition='p')
        snap = baseline.snapshot()
        with self.assertRaises(AttributeError):
            snap.sample_count = 99
        with self.assertRaises(AttributeError):
            snap['a'].mean = 99

    def test_content_identity_is_idempotent(self):
        baseline = ArffBaseline.from_arff(SPARSE)
        baseline.ingest(SPARSE)
        self.assertEqual(len(baseline), 4)
        self.assertEqual(baseline.snapshot()['a'].sum, 12.0)

    def test_growing_partition_uses_explicit_identity(self):
        first = '''@RELATION g
@ATTRIBUTE a REAL
@DATA
1
'''
        grown = first + '2\n'
        baseline = ArffBaseline.from_arff(first, partition='g')
        baseline.ingest(first, partition='g')
        baseline.ingest(grown, partition='g')
        self.assertEqual(len(baseline), 2)
        self.assertEqual(baseline.snapshot()['a'].sum, 3.0)

    def test_duplicate_keys_rejected(self):
        baseline = ArffBaseline([('a', 'REAL')])
        self.assertRaises(
            ValueError, baseline.ingest_rows, [[1.0], [2.0]], [('k',), ('k',)],
        )

    def test_negative_revision_rejected(self):
        baseline = ArffBaseline([('a', 'REAL')])
        self.assertRaises(
            ValueError, baseline.upsert, ('k',), [1.0], -1,
        )
        self.assertRaises(
            TypeError, baseline.upsert, ('k',), [1.0], 1.5,
        )

    def test_freezer_and_column_helpers(self):
        baseline = ArffBaseline.from_arff(SPARSE, partition='p')
        self.assertEqual(baseline.freeze().as_dict(),
                         baseline.snapshot().as_dict())
        self.assertEqual(baseline.column('a').count, 4)

    def test_merge_baselines_does_not_mutate_inputs(self):
        left = ArffBaseline.from_arff(SPARSE, partition='p')
        right = ArffBaseline.from_arff('''@RELATION sparse
@ATTRIBUTE a REAL
@ATTRIBUTE b REAL
@ATTRIBUTE c REAL
@DATA
{0 1}
''', partition='q')
        merged = arff_stats.merge_baselines([left, right])
        self.assertEqual(len(merged), 5)
        self.assertEqual(len(left), 4)
        self.assertEqual(len(right), 1)

    def test_number_aliases(self):
        baseline = ArffBaseline.from_arff(SPARSE, partition='p')
        snap = baseline.snapshot()['a']
        self.assertEqual(snap.n, snap.count)
        self.assertEqual(snap.total, snap.sum)
        self.assertEqual(snap.std, snap.stdev)
        self.assertEqual(baseline.column('a').total, 12)

    def test_normalization_uses_snapshot_basis(self):
        baseline = ArffBaseline.from_arff('''@RELATION n
@ATTRIBUTE a REAL
@ATTRIBUTE c {x,y}
@DATA
1,x
2,y
3,x
''', partition='p')
        published = baseline.snapshot()
        self.assertEqual(published['a'].mean, 2.0)
        self.assertAlmostEqual(published.normalize([2, 'y'])[0], 0.0)
        self.assertAlmostEqual(published.normalize([1, 'x'])[0],
                               -2.0 / math.sqrt(8.0 / 3.0))
        self.assertEqual(published.normalize([2, 'y'])[1], 'y')
        self.assertIsNone(published['a'].normalize(None))

        baseline.ingest('''@RELATION n
@ATTRIBUTE a REAL
@ATTRIBUTE c {x,y}
@DATA
100,x
''', partition='q')
        self.assertEqual(published['a'].mean, 2.0)
        self.assertEqual(published.normalize([2, 'y'])[0], 0.0)

    def test_constant_column_normalization(self):
        baseline = ArffBaseline([('a', 'REAL')])
        baseline.ingest_rows([[4.0], [4.0]], [('k',), ('j',)])
        column = baseline.snapshot()['a']
        self.assertEqual(column.stdev, 0.0)
        self.assertEqual(column.normalize(4.0), 0.0)
        self.assertRaises(ValueError, column.normalize, 5.0)

    def test_sparse_row_rejects_out_of_range_index(self):
        baseline = ArffBaseline([('a', 'REAL'), ('b', 'REAL')])
        self.assertRaises(
            ValueError, baseline.ingest_rows, [{2: 1.0}], [('k',)],
        )
        self.assertRaises(
            ValueError, baseline.ingest_rows, [{'x': 1.0}], [('k',)],
        )

    def test_shared_retransmitted_rows_merge_once(self):
        shared = [
            (('shared', index), [index * 1.0, None], 1)
            for index in range(10)
        ]

        def build(prefix, count):
            baseline = ArffBaseline([('x', 'REAL'), ('y', 'REAL')])
            for key, values, revision in shared:
                baseline.upsert(key, values, revision=revision)
            for index in range(count):
                baseline.upsert((prefix, index), [index * 2.0, 3.0],
                                revision=1)
            return baseline

        one = arff_stats.merge_baselines([
            build('a', 5), build('b', 7), build('c', 3),
        ]).snapshot()
        two = arff_stats.merge_baselines([
            build('c', 3), build('a', 5), build('b', 7),
        ]).snapshot()
        self.assertEqual(one.as_dict(), two.as_dict())
        self.assertEqual(one.sample_count, 10 + 5 + 7 + 3)

    def test_source_key_deduplicates_across_documents(self):
        first = '''@RELATION t
@ATTRIBUTE id INTEGER
@ATTRIBUTE v REAL
@DATA
1,10
2,20
'''
        second = '''@RELATION t
@ATTRIBUTE id INTEGER
@ATTRIBUTE v REAL
@DATA
1,15
3,30
'''
        baseline = ArffBaseline.from_arff(first, source_key='id')
        baseline.ingest(second, source_key='id')
        snap = baseline.snapshot()
        self.assertEqual(len(baseline), 3)
        self.assertEqual(snap['v'].count, 3)
        self.assertEqual(snap['v'].sum, 65.0)

    def test_source_key_by_index_and_bad_name(self):
        first = '''@RELATION t
@ATTRIBUTE id INTEGER
@ATTRIBUTE v REAL
@DATA
1,10
'''
        baseline = ArffBaseline.from_arff(first, source_key=0)
        baseline.ingest(first, source_key=0)
        self.assertEqual(len(baseline), 1)
        self.assertRaises(ValueError, baseline.ingest, first,
                          None, None, None, 'missing')

    def test_recompute_after_mixed_workload(self):
        baseline = ArffBaseline([('x', 'REAL'), ('y', 'REAL')])
        revoked = set()
        values = {}

        for index in range(200):
            key = ('k', index)
            value = float((index * 37) % 101) - 50.0
            baseline.upsert(key, [value, value / 2], revision=1)
            values[key] = (value, value / 2)
            if index % 5 == 0:
                baseline.retract(key, revision=2)
                revoked.add(key)
            elif index % 3 == 0:
                baseline.upsert(key, [-value, value], revision=2)
                values[key] = (-value, value)

        retained = [v for k, v in values.items() if k not in revoked]
        snap = baseline.snapshot()
        self.assertEqual(snap.sample_count, len(retained))
        for offset, name in enumerate(('x', 'y')):
            column = [pair[offset] for pair in retained]
            mean = math.fsum(column) / len(column)
            variance = math.fsum((value - mean) ** 2
                                 for value in column) / len(column)
            self.assertEqual(snap[name].count, len(column))
            self.assertAlmostEqual(snap[name].mean, mean, places=10)
            self.assertAlmostEqual(snap[name].variance, variance, places=8)


if __name__ == '__main__':
    unittest.main()
