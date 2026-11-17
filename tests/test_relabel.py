import unittest
from io import StringIO

import arff
import relabel
from relabel import RelabelConflict, solve, load_arff_inputs, \
    result_to_arff, write_conflict_report


def arff_fp(relation, attributes, rows):
    if rows:
        text = arff.dumps({
            'relation': relation,
            'attributes': attributes,
            'data': rows,
        })
    else:
        # arff.dumps cannot encode an empty data section, but the decoder
        # accepts a header-only document.
        lines = ['@RELATION %s' % relation, '']
        lines += ['@ATTRIBUTE %s %s'
                  % (name, 'STRING' if isinstance(atype, str)
                     else '{%s}' % ', '.join(atype))
                  for name, atype in attributes]
        lines += ['', '@DATA', '']
        text = '\n'.join(lines)
    return StringIO(text)


def history_fp(rows):
    return arff_fp('history', [
        ('sample_id', 'STRING'),
        ('old_label', 'STRING'),
        ('partition', 'STRING'),
    ], rows)


def candidates_fp(rows):
    return arff_fp('candidates', [
        ('old_label', 'STRING'),
        ('new_label', 'STRING'),
    ], rows)


def confirmed_fp(rows):
    return arff_fp('confirmed', [
        ('sample_id', 'STRING'),
        ('new_label', 'STRING'),
    ], rows)


def controls_fp(rows):
    return arff_fp('controls', [
        ('partition', 'STRING'),
        ('new_label', 'STRING'),
        ('max_count', 'INTEGER'),
    ], rows)


def load_solve(history, candidates, confirmed, controls):
    inputs = load_arff_inputs(
        history_fp(history), candidates_fp(candidates),
        confirmed_fp(confirmed), controls_fp(controls))
    return solve(*inputs)


class TestSolve(unittest.TestCase):
    def test_singleton_candidate_set_migrates_automatically(self):
        result = load_solve(
            history=[['s1', 'X', 'p1']],
            candidates=[['X', 'D']],
            confirmed=[],
            controls=[])
        self.assertEqual(result['auto'], {'s1': 'D'})
        self.assertEqual(result['status']['s1'], 'auto')
        self.assertEqual(set(result['domains']['s1']), {'D'})

    def test_ambiguous_samples_keep_full_candidate_set(self):
        result = load_solve(
            history=[['s1', 'Other', 'p1'], ['s2', 'Other', 'p1']],
            candidates=[['Other', 'A'], ['Other', 'B']],
            confirmed=[],
            controls=[])
        self.assertEqual(result['auto'], {})
        self.assertEqual(result['status']['s1'], 'ambiguous')
        self.assertEqual(set(result['domains']['s1']), {'A', 'B'})
        self.assertEqual(set(result['domains']['s2']), {'A', 'B'})

    def test_quota_prunes_to_unique_proof(self):
        # s2 is confirmed as A and partition p1 allows at most one A, so the
        # only surviving explanation for s1 is B.
        result = load_solve(
            history=[['s1', 'Other', 'p1'], ['s2', 'Other', 'p1']],
            candidates=[['Other', 'A'], ['Other', 'B']],
            confirmed=[['s2', 'A']],
            controls=[['p1', 'A', 1]])
        self.assertEqual(result['auto']['s1'], 'B')
        self.assertEqual(result['status']['s1'], 'auto')
        self.assertEqual(result['status']['s2'], 'confirmed')
        self.assertEqual(set(result['domains']['s2']), {'A'})

    def test_shared_sample_keeps_one_domain_across_partitions(self):
        # s1 occurs in both partitions.  In p1 the A quota is consumed by
        # confirmed s2, so s1's shared domain collapses to B; p2 stays
        # compatible with that single explanation instead of assembling a
        # different tag set per partition.
        result = load_solve(
            history=[
                ['s1', 'Other', 'p1'], ['s2', 'Other', 'p1'],
                ['s1', 'Other', 'p2'], ['s3', 'Other', 'p2'],
            ],
            candidates=[['Other', 'A'], ['Other', 'B']],
            confirmed=[['s2', 'A'], ['s3', 'A']],
            controls=[['p1', 'A', 1], ['p2', 'A', 1]])
        self.assertEqual(result['auto']['s1'], 'B')
        self.assertEqual(result['partitions_of']['s1'], ['p1', 'p2'])
        self.assertEqual(set(result['domains']['s1']), {'B'})
        self.assertTrue(result['feasibility_certified'])

    def test_confirmed_label_outside_candidates_conflicts(self):
        with self.assertRaises(RelabelConflict) as ctx:
            load_solve(
                history=[['s1', 'Other', 'p1']],
                candidates=[['Other', 'A'], ['Other', 'B']],
                confirmed=[['s1', 'Z']],
                controls=[])
        self.assertEqual(
            ctx.exception.conflicts[0]['kind'],
            'confirmed_not_candidate')

    def test_partition_capacity_below_demand_conflicts(self):
        with self.assertRaises(RelabelConflict) as ctx:
            load_solve(
                history=[
                    ['s1', 'Other', 'p1'], ['s2', 'Other', 'p1'],
                    ['s3', 'Other', 'p1'], ['s4', 'Other', 'p1'],
                ],
                candidates=[['Other', 'A'], ['Other', 'B'],
                            ['Other', 'C']],
                confirmed=[],
                controls=[['p1', 'A', 1], ['p1', 'B', 1],
                          ['p1', 'C', 1]])
        kinds = {c['kind'] for c in ctx.exception.conflicts}
        self.assertIn('hall_violation', kinds)
        hall = next(c for c in ctx.exception.conflicts
                    if c['kind'] == 'hall_violation')
        self.assertEqual(hall['demand'], 4)
        self.assertEqual(hall['capacity'], 3)

    def test_cross_partition_controls_clash_on_shared_sample(self):
        # p1 forces s1 to B; p2 already reserves the single B slot for s3.
        with self.assertRaises(RelabelConflict) as ctx:
            load_solve(
                history=[
                    ['s1', 'Other', 'p1'], ['s2', 'Other', 'p1'],
                    ['s1', 'Other', 'p2'], ['s3', 'Other', 'p2'],
                ],
                candidates=[['Other', 'A'], ['Other', 'B']],
                confirmed=[['s2', 'A'], ['s3', 'B']],
                controls=[['p1', 'A', 1], ['p2', 'B', 1]])
        kinds = {c['kind'] for c in ctx.exception.conflicts}
        self.assertTrue(
            kinds & {'quota_violation', 'hall_violation', 'empty_domain'})

    def test_triangle_is_locally_consistent_but_globally_infeasible(self):
        history = [
            ['s1', 'Other', 'p1'], ['s2', 'Other', 'p1'],
            ['s1', 'Other', 'p2'], ['s3', 'Other', 'p2'],
            ['s2', 'Other', 'p3'], ['s3', 'Other', 'p3'],
        ]
        candidates = [['Other', 'A'], ['Other', 'B']]
        controls = [
            ['p1', 'A', 1], ['p1', 'B', 1],
            ['p2', 'A', 1], ['p2', 'B', 1],
            ['p3', 'A', 1], ['p3', 'B', 1],
        ]
        with self.assertRaises(RelabelConflict) as ctx:
            load_solve(history, candidates, [], controls)
        # Exact global support disproves every value of at least one shared
        # sample; the conflict names that sample and its partitions instead
        # of silently producing an impossible domain.
        conflict = ctx.exception.conflicts[0]
        self.assertEqual(conflict['kind'], 'empty_domain')
        self.assertIn(conflict['sample'], ('s1', 's2', 's3'))
        self.assertTrue(conflict['partitions'])

    def test_feasible_pair_of_partitions_is_certified(self):
        history = [
            ['s1', 'Other', 'p1'], ['s2', 'Other', 'p1'],
            ['s1', 'Other', 'p2'], ['s3', 'Other', 'p2'],
        ]
        result = load_solve(
            history=history,
            candidates=[['Other', 'A'], ['Other', 'B']],
            confirmed=[],
            controls=[['p1', 'A', 1], ['p1', 'B', 1],
                      ['p2', 'A', 1], ['p2', 'B', 1]])
        self.assertTrue(result['feasibility_certified'])
        self.assertEqual(result['auto'], {})
        for sid in ('s1', 's2', 's3'):
            self.assertEqual(set(result['domains'][sid]), {'A', 'B'})


class TestArffIO(unittest.TestCase):
    def build_result(self):
        return load_solve(
            history=[
                ['s1', 'Other', 'p1'],
                ['s1', 'Other', 'p2'],
                ['s2', 'X', 'p1'],
            ],
            candidates=[
                ['Other', 'A'], ['Other', 'B'],
                ['X', 'D'],
            ],
            confirmed=[],
            controls=[])

    def test_one_row_per_sample_with_shared_partitions(self):
        result = self.build_result()
        obj = result_to_arff(result)
        reloaded = arff.loads(arff.dumps(obj))
        names = [a[0] for a in reloaded['attributes']]
        self.assertEqual(names, [
            'sample_id', 'original_label', 'partitions',
            'candidate_labels', 'auto_label', 'status'])
        rows = {row[0]: row for row in reloaded['data']}
        self.assertEqual(len(rows), 2)
        # s1 spans two partitions but gets a single shared explanation.
        s1 = rows['s1']
        self.assertEqual(s1[1], 'Other')
        self.assertEqual(s1[2], 'p1 p2')
        self.assertEqual(set(s1[3].split()), {'A', 'B'})
        self.assertIsNone(s1[4])  # ambiguous -> no auto migration
        self.assertEqual(s1[5], 'ambiguous')
        # original and inferred labels live in separate columns.
        s2 = rows['s2']
        self.assertEqual(s2[1], 'X')
        self.assertEqual(s2[4], 'D')
        self.assertEqual(s2[5], 'auto')

    def test_round_trip_through_file_objects(self):
        result = self.build_result()
        fp = StringIO()
        arff.dump(result_to_arff(result), fp)
        fp.seek(0)
        reloaded = arff.load(fp)
        self.assertEqual(len(reloaded['data']), 2)

    def test_conflict_report_names_offending_controls(self):
        try:
            load_solve(
                history=[
                    ['s1', 'Other', 'p1'], ['s2', 'Other', 'p1'],
                    ['s3', 'Other', 'p1'], ['s4', 'Other', 'p1'],
                ],
                candidates=[['Other', 'A'], ['Other', 'B'],
                            ['Other', 'C']],
                confirmed=[],
                controls=[['p1', 'A', 1], ['p1', 'B', 1],
                          ['p1', 'C', 1]])
        except RelabelConflict as exc:
            fp = StringIO()
            write_conflict_report(exc.conflicts, fp)
            report = fp.getvalue()
        else:
            self.fail('expected RelabelConflict')
        self.assertIn('p1', report)
        self.assertIn('capacity', report)


if __name__ == '__main__':
    unittest.main()
