import json
import os
import tempfile
import textwrap
import unittest

import arff
import arff_filter as af


def make_obj(data_lines, sparse=False, attributes=None):
    if attributes is None:
        attributes = [
            ('a', 'REAL'),
            ('b', 'REAL'),
            ('cat', ['x', 'y', '?']),
            ('label', 'STRING'),
        ]
    txt = textwrap.dedent('''
    @RELATION testing

    {attributes}

    @DATA
    {data}
    ''').format(
        attributes='\n'.join(
            '@ATTRIBUTE %s %s' % (
                name,
                '{%s}' % ', '.join('"%s"' % v if v == '?' else v
                                   for v in values)
                if isinstance(values, list) else values)
            for name, values in attributes),
        data='\n'.join(data_lines))
    return arff.loads(txt, return_type=arff.LOD if sparse else arff.DENSE)


class TestThreeValuedLogic(unittest.TestCase):
    def test_and_truth_table(self):
        table = [
            (True, True, True),
            (True, False, False),
            (False, True, False),
            (False, False, False),
            (True, None, None),
            (None, True, None),
            (False, None, False),
            (None, False, False),
            (None, None, None),
        ]
        for left, right, expected in table:
            self.assertIs(af._and(left, right), expected)

    def test_or_truth_table(self):
        table = [
            (True, True, True),
            (True, False, True),
            (False, True, True),
            (False, False, False),
            (False, None, None),
            (None, False, None),
            (True, None, True),
            (None, True, True),
            (None, None, None),
        ]
        for left, right, expected in table:
            self.assertIs(af._or(left, right), expected)

    def test_not_truth_table(self):
        self.assertIs(af._not(True), False)
        self.assertIs(af._not(False), True)
        self.assertIs(af._not(None), None)


class TestParsing(unittest.TestCase):
    ATTRS = [('a', 'REAL'), ('b', 'INTEGER'),
             ('cat', ['x', 'y', '?']), ('label', 'STRING')]

    def test_operators_and_precedence(self):
        expr = af.parse_expression(
            'a >= 1 and b = 2 or cat = x and not label = z', self.ATTRS)
        self.assertEqual(
            str(expr),
            "((a >= 1.0 and b = 2) or (cat = x and not label = z))")

    def test_parentheses(self):
        expr = af.parse_expression('(a > 0 or b > 0) and cat = x',
                                   self.ATTRS)
        self.assertEqual(str(expr), '((a > 0.0 or b > 0) and cat = x)')

    def test_case_insensitive_keywords(self):
        expr = af.parse_expression('a > 1 AND (b < 2 OR NOT cat = x)',
                                   self.ATTRS)
        self.assertIsInstance(expr.root, af.AndNode)

    def test_membership_and_missing(self):
        expr = af.parse_expression(
            'cat in (x, y) and a not in (1, 2) and b is not missing',
            self.ATTRS)
        self.assertEqual(expr.fields, {'a', 'b', 'cat'})

    def test_quoted_question_mark_value(self):
        expr = af.parse_expression('cat = "?"', self.ATTRS)
        self.assertEqual(expr.evaluate([0.0, 0, '?', 'row']), True)
        self.assertEqual(expr.evaluate([0.0, 0, 'x', 'row']), False)

    def test_bare_question_mark_is_rejected(self):
        with self.assertRaises(af.FilterError):
            af.parse_expression('cat = ?', self.ATTRS)

    def test_unknown_field(self):
        with self.assertRaises(af.BadExpression):
            af.parse_expression('a >', self.ATTRS)
        with self.assertRaises(af.FilterError):
            af.parse_expression('missing_field > 1', self.ATTRS)

    def test_bad_nominal_and_numeric_values(self):
        with self.assertRaises(af.FilterError):
            af.parse_expression('cat = nope', self.ATTRS)
        with self.assertRaises(af.FilterError):
            af.parse_expression('a = nope', self.ATTRS)

    def test_empty_expression(self):
        with self.assertRaises(af.BadExpression):
            af.parse_expression('', self.ATTRS)

    def test_programmatic_builder(self):
        root = af.and_(
            af.or_(af.condition(self.ATTRS, 'a', '>', 0),
                   af.condition(self.ATTRS, 'b', '>', 0)),
            af.not_(af.condition(self.ATTRS, 'cat', af.IS_MISSING)))
        expr = af.Expression(root, self.ATTRS)
        self.assertEqual(expr.evaluate([1.0, None, 'x', 'r']), True)
        self.assertEqual(expr.evaluate([None, None, 'x', 'r']), None)

    def test_quoted_field_name_and_double_not(self):
        attrs = [('a field', 'REAL')]
        expr = af.parse_expression('not not "a field" > 0', attrs)
        self.assertEqual(expr.evaluate([1.0]), True)
        self.assertEqual(expr.evaluate([-1.0]), False)
        self.assertEqual(expr.evaluate([None]), None)

    def test_membership_without_parentheses(self):
        expr = af.parse_expression('cat in x, y', self.ATTRS)
        self.assertEqual(expr.evaluate([0.0, 0, 'y', 'r']), True)
        self.assertEqual(expr.evaluate([0.0, 0, '?', 'r']), False)

    def test_unbalanced_and_dangling_tokens(self):
        with self.assertRaises(af.BadExpression):
            af.parse_expression('(a > 0', self.ATTRS)
        with self.assertRaises(af.BadExpression):
            af.parse_expression('a > 0 and', self.ATTRS)
        with self.assertRaises(af.BadExpression):
            af.parse_expression('a 0 1', self.ATTRS)


class TestDenseEvaluation(unittest.TestCase):
    def setUp(self):
        self.obj = make_obj([
            '1.0,2.0,x,keep',          # 0 keep
            '?,2.0,x,unknown-a',       # 1 unknown on a
            '-1.0,?,x,exclude',        # 2 definite exclude (a fails)
            '?,?,?,both-missing',      # 3 unknown on a and b; cat missing
            '5.0,-1.0,x,exclude-b',    # 4 definite exclude (b fails)
            '1.0,?,x,unknown-b',       # 5 unknown on b
        ])
        self.expr = af.parse_expression(
            'a > 0 and b > 0', self.obj['attributes'])

    def test_partition_labels(self):
        result = af.partition(self.expr, self.obj['data'])
        self.assertEqual(result.keep_indices, [0])
        self.assertEqual(result.exclude_indices, [2, 4])
        self.assertEqual(result.unknown_indices, [1, 3, 5])
        # Unknown rows are never silently dropped: every row is accounted for.
        self.assertEqual(
            len(result.keep) + len(result.exclude) + len(result.unknown),
            len(self.obj['data']))

    def test_dependencies(self):
        result = af.partition(self.expr, self.obj['data'])
        self.assertEqual(result.dependencies[1], ['a'])
        self.assertEqual(result.dependencies[3], ['a', 'b'])
        self.assertEqual(result.dependencies[5], ['b'])
        self.assertEqual(result.labels[2], af.EXCLUDE)

    def test_missing_is_not_false_in_or(self):
        obj = make_obj([
            '1.0,?,x,r',   # keep: a satisfies the or-branch
            '-1.0,?,x,r',  # unknown: only b could save it
            '?,?,x,r',     # unknown on both
            '-1.0,-2.0,x,r',  # definite exclude
        ])
        expr = af.parse_expression('a > 0 or b > 0', obj['attributes'])
        result = af.partition(expr, obj['data'])
        self.assertEqual(result.keep_indices, [0])
        self.assertEqual(result.exclude_indices, [3])
        self.assertEqual(result.unknown_indices, [1, 2])
        self.assertEqual(result.dependencies[1], ['b'])
        self.assertEqual(result.dependencies[2], ['a', 'b'])

    def test_not_preserves_unknown_and_deps(self):
        obj = make_obj(['?,1.0,x,r'])
        expr = af.parse_expression('not a > 0', obj['attributes'])
        label, deps = expr.classify(obj['data'][0])
        self.assertEqual(label, af.UNKNOWN)
        self.assertEqual(deps, ('a',))

    def test_is_missing_is_always_definite(self):
        obj = make_obj(['?,1.0,x,r', '1.0,?,x,r'])
        expr = af.parse_expression('a is missing', obj['attributes'])
        self.assertEqual(expr.classify(obj['data'][0]), (af.KEEP, ()))
        self.assertEqual(expr.classify(obj['data'][1]), (af.EXCLUDE, ()))

        combined = af.parse_expression(
            'a is missing or a > 5', obj['attributes'])
        # Missing a satisfies the first branch: no dependency remains.
        self.assertEqual(combined.classify(obj['data'][0]), (af.KEEP, ()))
        label, deps = combined.classify(obj['data'][1])
        self.assertEqual(label, af.EXCLUDE)
        self.assertEqual(deps, ())

    def test_explicit_missing_marker_is_none(self):
        obj = make_obj(['?,1.0,x,r'])
        self.assertIsNone(obj['data'][0][0])
        expr = af.parse_expression('a > 0', obj['attributes'])
        self.assertIsNone(expr.evaluate(obj['data'][0]))


class TestSparseSemantics(unittest.TestCase):
    SPARSE_ROWS = [
        '{}',              # 0 all defaults
        '{0 ?}',           # 1 a explicitly missing
        '{0 1.5, 1 2}',    # 2 positive values
        '{2 "?"}',         # 3 quoted question mark category
        '{3 "?"}',         # 4 quoted question mark string value
        '{2 y, 3 z}',      # 5
    ]

    def _load(self, sparse):
        return make_obj(self.SPARSE_ROWS, sparse=sparse)

    def test_omitted_indices_take_declared_defaults_dense(self):
        obj = self._load(sparse=False)
        row = obj['data'][0]
        self.assertEqual(row, [0.0, 0, 'x', '0'])

    def test_omitted_indices_take_declared_defaults_lod(self):
        obj = self._load(sparse=True)
        self.assertEqual(obj['data'][0], {})
        expr = af.parse_expression(
            'a = 0 and b = 0 and cat = x and label = 0',
            obj['attributes'])
        self.assertEqual(expr.evaluate(obj['data'][0]), True)

    def test_explicit_question_mark_is_missing_in_sparse(self):
        for sparse in (False, True):
            obj = self._load(sparse=sparse)
            expr = af.parse_expression('a > 0', obj['attributes'])
            label, deps = expr.classify(obj['data'][1])
            self.assertEqual(label, af.UNKNOWN)
            self.assertEqual(deps, ('a',))

    def test_quoted_question_mark_is_a_value_not_missing(self):
        for sparse in (False, True):
            obj = self._load(sparse=sparse)
            cat_expr = af.parse_expression('cat = "?"', obj['attributes'])
            self.assertEqual(cat_expr.evaluate(obj['data'][3]), True)
            missing_expr = af.parse_expression(
                'cat is missing', obj['attributes'])
            self.assertEqual(missing_expr.evaluate(obj['data'][3]), False)

            label_expr = af.parse_expression(
                'label = "?"', obj['attributes'])
            self.assertEqual(label_expr.evaluate(obj['data'][4]), True)
            label_missing = af.parse_expression(
                'label is missing', obj['attributes'])
            self.assertEqual(label_missing.evaluate(obj['data'][4]), False)

    def test_sparse_and_dense_partitions_match(self):
        dense = self._load(sparse=False)
        lod = self._load(sparse=True)
        text = 'a >= 0 and (cat = x or cat = "?") and not b is missing'
        r1 = af.partition(af.parse_expression(text, dense['attributes']),
                          dense['data'])
        r2 = af.partition(af.parse_expression(text, lod['attributes']),
                          lod['data'])
        self.assertEqual(r1.labels, r2.labels)
        self.assertEqual(r1.dependencies, r2.dependencies)

    def test_coo_rows_are_supported(self):
        txt = textwrap.dedent('''
        @RELATION testing
        @ATTRIBUTE a REAL
        @ATTRIBUTE cat {x, y}
        @DATA
        {0 1.0, 1 y}
        {}
        ''')
        obj = arff.loads(txt, return_type=arff.COO)
        records = af.coo_to_lod(obj['data'], n_rows=2)
        expr = af.parse_expression('a > 0 and cat = y', obj['attributes'])
        result = af.partition(expr, records)
        self.assertEqual(result.keep_indices, [0])
        # The empty sparse row expands to the declared defaults (a = 0.0,
        # cat = x), so it is a definite exclusion rather than unknown.
        self.assertEqual(result.exclude_indices, [1])

    def test_coo_preserves_explicit_missing(self):
        # An explicit '?' survives in every representation (DENSE, LOD and
        # COO); only omitted sparse indices take declared defaults.
        txt = textwrap.dedent('''
        @RELATION testing
        @ATTRIBUTE a REAL
        @DATA
        {0 ?}
        ''')
        loaded = arff.loads(txt)
        coo_records = af.coo_to_lod(
            arff.loads(txt, return_type=arff.COO)['data'], n_rows=1)
        lod_records = arff.loads(txt, return_type=arff.LOD)['data']
        expr = af.parse_expression('a > 0', loaded['attributes'])
        self.assertEqual(expr.classify(coo_records[0])[0], af.UNKNOWN)
        self.assertEqual(expr.classify(lod_records[0])[0], af.UNKNOWN)
        self.assertEqual(expr.classify(loaded['data'][0])[0], af.UNKNOWN)


class TestIncrementalReclassification(unittest.TestCase):
    def setUp(self):
        self.obj = make_obj([
            '?,?,x,r0',   # 0 unknown on a and b
            '?,1.0,x,r1',  # 1 unknown on a only
            '-1.0,?,x,r2',  # 2 definite exclude despite missing b
            '1.0,2.0,x,r3',  # 3 definite keep
            '?,-1.0,x,r4',  # 4 definite exclude despite missing a
        ])
        self.expr = af.parse_expression(
            'a > 0 and b > 0', self.obj['attributes'])
        self.state = af.FilterState(self.expr, self.obj['data'])
        self.state.run()

    def test_initial_partition(self):
        result = self.state.run()
        self.assertEqual(set(result.unknown_indices), {0, 1})
        self.assertEqual(set(result.exclude_indices), {2, 4})
        self.assertEqual(result.keep_indices, [3])

    def test_refill_one_column_only_touches_related_unknowns(self):
        self.obj['data'][0][0] = 1.0
        self.obj['data'][1][0] = 1.0
        result = self.state.reclassify(['a'])

        # Row 1 is now decidable; row 0 still waits on b.
        self.assertEqual(result.keep_indices, [1, 3])
        self.assertEqual(result.unknown_indices, [0])
        self.assertEqual(result.dependencies[0], ['b'])
        # Definite verdicts were not recomputed and did not change.
        self.assertEqual(set(result.exclude_indices), {2, 4})

    def test_unrelated_refill_changes_nothing(self):
        result = self.state.refill('cat')
        self.assertEqual(set(result.unknown_indices), {0, 1})
        self.assertEqual(result.dependencies[0], ['a', 'b'])

    def test_second_refill_resolves_remaining_unknown(self):
        self.obj['data'][0][0] = 1.0
        self.obj['data'][1][0] = 1.0
        self.state.reclassify(['a'])
        self.obj['data'][0][1] = 2.0
        result = self.state.reclassify(['b'])
        self.assertEqual(result.unknown_indices, [])
        self.assertEqual(set(result.keep_indices), {0, 1, 3})

    def test_update_single_record(self):
        label, deps = self.state.update_record(1, [2.0, 3.0, 'x', 'r1'])
        self.assertEqual(label, af.KEEP)
        self.assertEqual(deps, ())
        self.assertNotIn(1, self.state.run().unknown_indices)

    def test_string_expression_requires_attributes(self):
        state = af.FilterState('a > 0', self.obj['data'],
                               attributes=self.obj['attributes'])
        result = state.run()
        # With only the a condition, every row whose a is missing is unknown,
        # including row 4 (whose b = -1 cannot decide this expression).
        self.assertEqual(set(result.unknown_indices), {0, 1, 4})

    def test_string_expression_without_attributes_is_rejected(self):
        with self.assertRaises(af.FilterError):
            af.FilterState('a > 0', self.obj['data'])


class TestCommandLine(unittest.TestCase):
    def test_main_writes_subsets_and_report(self):
        obj = make_obj([
            '1.0,2.0,x,keep',
            '?,2.0,x,unknown',
            '-1.0,2.0,x,exclude',
        ])
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmp:
            source = os.path.join(tmp, 'in.arff')
            with open(source, 'w') as fp:
                arff.dump(obj, fp)
            keep = os.path.join(tmp, 'keep.arff')
            exclude = os.path.join(tmp, 'exclude.arff')
            unknown = os.path.join(tmp, 'unknown.arff')
            report = os.path.join(tmp, 'report.json')

            rc = af.main([
                source, '-e', 'a > 0 and b > 0',
                '--keep', keep, '--exclude', exclude,
                '--unknown', unknown, '--report', report,
            ])
            self.assertEqual(rc, 0)

            with open(report) as fp:
                payload = json.load(fp)
            self.assertEqual(payload['counts'],
                             {af.KEEP: 1, af.EXCLUDE: 1, af.UNKNOWN: 1})
            self.assertEqual(payload['unknown'],
                             [{'index': 1, 'depends_on': ['a']}])

            kept = arff.load(open(keep))
            self.assertEqual(len(kept['data']), 1)
            excluded = arff.load(open(exclude))
            self.assertEqual(len(excluded['data']), 1)
            unknowns = arff.load(open(unknown))
            self.assertEqual(len(unknowns['data']), 1)


if __name__ == '__main__':
    unittest.main()
