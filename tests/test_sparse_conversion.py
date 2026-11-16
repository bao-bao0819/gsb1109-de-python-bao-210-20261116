import unittest

import arff


ATTRIBUTES = [
    ('num', 'REAL'),
    ('color', ['red', 'green', 'blue']),
    ('label', ['yes', 'no']),
]


class TestDenseToSparse(unittest.TestCase):
    def test_omits_implicit_values(self):
        obj = {
            'relation': 't',
            'attributes': ATTRIBUTES,
            'data': [
                [0.0, 'red', 'yes'],   # all implicit
                [1.5, 'green', 'no'],  # all explicit
                [0.0, 'blue', 'yes'],  # mixed
            ],
        }
        sparse = arff.dense_to_sparse(obj)
        self.assertEqual(sparse['data'], [
            {},
            {0: 1.5, 1: 'green', 2: 'no'},
            {1: 'blue'},
        ])

    def test_missing_is_always_explicit(self):
        obj = {
            'relation': 't',
            'attributes': ATTRIBUTES,
            'data': [[None, 'red', None]],
        }
        sparse = arff.dense_to_sparse(obj)
        self.assertEqual(sparse['data'], [{0: None, 2: None}])

    def test_explicit_zero_is_valid_and_omittable(self):
        # an explicit numeric zero is a genuine value, but sparse decoding
        # restores it, so it may be omitted; it must not become missing
        obj = {
            'relation': 't',
            'attributes': [('n', 'INTEGER')],
            'data': [[0], [3]],
        }
        sparse = arff.dense_to_sparse(obj)
        self.assertEqual(sparse['data'], [{}, {0: 3}])
        back = arff.loads(arff.dumps(sparse), return_type=arff.DENSE)
        self.assertEqual(back['data'], [[0], [3]])

    def test_round_trip_through_arff_text(self):
        obj = {
            'relation': 't',
            'attributes': ATTRIBUTES,
            'data': [
                [0.0, 'red', 'yes'],
                [2.5, 'green', 'no'],
                [None, 'blue', None],
            ],
        }
        sparse = arff.dense_to_sparse(obj)
        text = arff.dumps(sparse)
        self.assertIn('{ 0 ?,1 blue,2 ? }', text)
        back = arff.loads(text, return_type=arff.DENSE)
        self.assertEqual(back['data'], obj['data'])


class TestResparsifyRows(unittest.TestCase):
    def test_reorder_reports_added_and_removed_per_row(self):
        rows = [
            {},                    # label implicitly 'yes'
            {2: 'no'},             # label explicitly 'no'
            {0: 1.0, 2: 'yes'},    # label explicitly 'yes'
        ]
        new_attributes = [
            ('num', 'REAL'),
            ('color', ['red', 'green', 'blue']),
            ('label', ['no', 'yes']),  # 'no' becomes first
        ]
        new_rows, report = arff.resparsify_rows(ATTRIBUTES, rows,
                                                new_attributes)
        self.assertEqual(new_rows, [
            {2: 'yes'},   # old first category must now be written
            {},             # 'no' is now implicit
            {0: 1.0, 2: 'yes'},
        ])
        self.assertEqual(report, [
            {'added': [2], 'removed': []},
            {'added': [], 'removed': [2]},
            {'added': [], 'removed': []},
        ])

    def test_missing_stays_explicit_when_reordering(self):
        rows = [{2: None}]
        new_attributes = [
            ('num', 'REAL'),
            ('color', ['red', 'green', 'blue']),
            ('label', ['no', 'yes']),
        ]
        new_rows, report = arff.resparsify_rows(ATTRIBUTES, rows,
                                                new_attributes)
        self.assertEqual(new_rows, [{2: None}])
        self.assertEqual(report, [{'added': [], 'removed': []}])

    def test_rejects_category_changes(self):
        bad = [('num', 'REAL'), ('color', ['red', 'green']),
               ('label', ['yes', 'no'])]
        with self.assertRaises(arff.BadObject):
            arff.resparsify_rows(ATTRIBUTES, [{}], bad)

    def test_round_trip_after_reorder(self):
        obj = {
            'relation': 't',
            'attributes': ATTRIBUTES,
            'data': [[0.0, 'red', 'yes'], [1.0, 'green', 'no']],
        }
        sparse = arff.dense_to_sparse(obj)
        new_attributes = [
            ('num', 'REAL'),
            ('color', ['blue', 'red', 'green']),
            ('label', ['no', 'yes']),
        ]
        new_rows, _ = arff.resparsify_rows(ATTRIBUTES, sparse['data'],
                                           new_attributes)
        new_obj = dict(obj, attributes=new_attributes, data=new_rows)
        back = arff.loads(arff.dumps(new_obj), return_type=arff.DENSE)
        self.assertEqual(back['data'], obj['data'])


class TestPlanSparseBudget(unittest.TestCase):
    def test_minimizes_entries_and_reorders(self):
        attributes = [
            ('a', ['x', 'y']),
            ('b', 'REAL'),
        ]
        data = [
            ['x', 0],
            ['x', 1],
            ['y', 0],
            [None, 0],
        ]
        plan = arff.plan_sparse_budget(attributes, data)
        # 'x' first: 2 omissions; costs: a -> 1 missing + 1 'y', b -> 1
        self.assertEqual(plan['required_entries'], 3)
        self.assertEqual(plan['first_categories'], {'a': 'x'})
        self.assertEqual(plan['attributes'][0], ('a', ['x', 'y']))
        self.assertTrue(plan['feasible'])

    def test_budget_feasible_and_infeasible(self):
        attributes = [('a', ['x', 'y'])]
        data = [['x'], ['x'], ['y']]
        ok = arff.plan_sparse_budget(attributes, data, budget=1)
        self.assertTrue(ok['feasible'])
        self.assertEqual(ok['required_entries'], 1)
        no = arff.plan_sparse_budget(attributes, data, budget=0)
        self.assertFalse(no['feasible'])
        # reports the minimum needed instead of faking a fit
        self.assertEqual(no['required_entries'], 1)

    def test_missing_never_treated_as_zero(self):
        attributes = [('a', ['x', 'y']), ('n', 'REAL')]
        data = [[None, 0], [None, 0]]
        plan = arff.plan_sparse_budget(attributes, data, budget=0)
        # both missing values must be written; cannot be saved as zeros
        self.assertEqual(plan['required_entries'], 2)
        self.assertFalse(plan['feasible'])

    def test_plan_matches_actual_sparse_size(self):
        attributes = [
            ('a', ['x', 'y', 'z']),
            ('b', ['p', 'q']),
            ('n', 'REAL'),
        ]
        data = [
            ['x', 'p', 0],
            ['y', 'p', 1],
            ['x', None, 0],
            ['z', 'q', 0],
            ['x', 'p', 0],
        ]
        plan = arff.plan_sparse_budget(attributes, data)
        obj = {'relation': 't', 'attributes': plan['attributes'],
               'data': data}
        sparse = arff.dense_to_sparse(obj)
        self.assertEqual(sum(len(r) for r in sparse['data']),
                         plan['required_entries'])


if __name__ == '__main__':
    unittest.main()
