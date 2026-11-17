import textwrap
import unittest

import arff
import arff_constraints as constraints


DEVICE_ARFF = textwrap.dedent('''\
@RELATION device

@ATTRIBUTE power {on, off}
@ATTRIBUTE mode {fast, slow, idle}
@ATTRIBUTE cooling {active, inactive}
@ATTRIBUTE bypass {enabled, disabled}
@ATTRIBUTE level NUMERIC

@DATA
on,fast,active,disabled,3
on,?,inactive,disabled,4
on,slow,inactive,enabled,2
off,idle,inactive,disabled,1
off,slow,inactive,disabled,5
''')


RULES = {
    'allowed': [
        {'power': 'on', 'mode': 'fast', 'cooling': 'active'},
        {'power': 'on', 'mode': 'slow', 'cooling': 'inactive'},
        {'power': 'off', 'mode': 'idle', 'cooling': 'inactive'},
    ],
    'implies': [
        {'if': {'power': 'on'}, 'then': {'bypass': 'disabled'}},
    ],
    'excludes': [
        ('mode', 'fast', 'level', '2'),
    ],
}


class BaseConstraintTest(unittest.TestCase):
    def report(self, source=DEVICE_ARFF, rules=RULES, **kwargs):
        return constraints.check_constraints(source, rules, **kwargs)

    def row(self, index, source=DEVICE_ARFF, rules=RULES, **kwargs):
        return self.report(source, rules, **kwargs)['rows'][index]


class TestRowStatus(BaseConstraintTest):
    def test_complete_valid_row(self):
        row = self.row(0)
        self.assertEqual(row['status'], 'complete')
        self.assertTrue(row['has_completion'])
        self.assertEqual(row['missing'], [])
        self.assertEqual(row['possible_values'], {})

    def test_missing_field_is_not_an_error(self):
        row = self.row(1)
        self.assertEqual(row['status'], 'completable')
        self.assertTrue(row['has_completion'])
        self.assertEqual(row['missing'], ['mode'])
        self.assertEqual(row['possible_values'], {'mode': ['slow']})
        self.assertEqual(row['any_value'], {'mode': False})

    def test_known_values_are_preserved(self):
        obj = arff.loads(DEVICE_ARFF)
        obj['data'][1][1] = ''
        report = constraints.check_constraints(
            obj, RULES, include_completions=True)
        row = report['rows'][1]
        self.assertTrue(row['has_completion'])
        self.assertEqual(row['possible_values'], {'mode': ['slow']})
        self.assertEqual(row['values'][1], '')

    def test_range_is_kept_instead_of_picking_a_value(self):
        source = textwrap.dedent('''\
        @RELATION device
        @ATTRIBUTE power {on, off}
        @ATTRIBUTE mode {fast, slow, idle}
        @DATA
        ?,?
        ''')
        rules = {'allowed': [{'power': 'on', 'mode': 'fast'},
                             {'power': 'on', 'mode': 'slow'}]}
        row = self.row(0, source=source, rules=rules,
                       include_completions=True)
        self.assertEqual(row['status'], 'completable')
        self.assertEqual(row['possible_values']['power'], ['on'])
        self.assertEqual(row['possible_values']['mode'], ['fast', 'slow'])
        self.assertFalse(row['any_value']['power'])
        self.assertFalse(row['any_value']['mode'])
        self.assertLessEqual(len(row['completions']), 2)
        for completion in row['completions']:
            self.assertEqual(completion['power'], 'on')
            self.assertIn(completion['mode'], ('fast', 'slow'))

    def test_conflicting_row(self):
        row = self.row(4)
        self.assertEqual(row['status'], 'conflict')
        self.assertFalse(row['has_completion'])
        self.assertEqual(row['conflict']['attributes'], ['power', 'mode'])
        self.assertEqual(row['conflict_fields'], ['power', 'mode'])

    def test_conflict_values_are_the_observed_ones(self):
        row = self.row(4)
        self.assertEqual(row['conflict']['values'],
                         {'power': 'off', 'mode': 'slow'})

    def test_conflict_caused_by_implication(self):
        row = self.row(2)
        self.assertEqual(row['status'], 'conflict')
        self.assertIn('bypass', row['conflict']['attributes'])
        # Every reported field must be needed: dropping any one of them
        # admits a completion.
        self.assert_irredundant(row)

    def test_conflict_caused_by_exclusion(self):
        source = textwrap.dedent('''\
        @RELATION device
        @ATTRIBUTE power {on, off}
        @ATTRIBUTE mode {fast, slow, idle}
        @ATTRIBUTE cooling {active, inactive}
        @ATTRIBUTE bypass {enabled, disabled}
        @ATTRIBUTE level NUMERIC
        @DATA
        on,fast,active,disabled,2
        ''')
        rules = {'allowed': {
            'attributes': ['power', 'mode', 'cooling'],
            'combinations': [['on', 'fast', 'active'],
                             ['off', 'idle', 'inactive']],
        }, 'excludes': [('mode', 'fast', 'level', '2')]}
        report = self.report(source=source, rules=rules)
        row = report['rows'][0]
        self.assertIn('level', row['conflict']['attributes'])
        self.assert_irredundant(row, report, rules)

    def test_minimal_conflict_is_irredundant(self):
        report = self.report()
        conflict_rows = [row for row in report['rows']
                         if row['status'] == 'conflict']
        self.assertTrue(conflict_rows)
        for row in conflict_rows:
            self.assert_irredundant(row)

    def assert_irredundant(self, row, report=None, rules=None):
        """Check inclusion minimality of the reported conflict fields."""
        report = report or self.report()
        rules = RULES if rules is None else rules
        names = [name for name, _ in report['attributes']]
        attributes = row['conflict']['attributes']
        self.assertTrue(attributes)

        def subset(keep):
            return [[value if name in keep else None
                     for name, value in zip(names, row['values'])]]

        obj = {
            'relation': report['relation'],
            'attributes': report['attributes'],
            'data': subset(attributes),
        }
        self.assertFalse(
            constraints.check_constraints(obj, rules)['rows'][0]
            ['has_completion'])
        for attribute in attributes:
            obj['data'] = subset(set(attributes) - {attribute})
            self.assertTrue(
                constraints.check_constraints(obj, rules)['rows'][0]
                ['has_completion'],
                'removing %s should make the conflict fields completable'
                % attribute)


class TestRuleForms(unittest.TestCase):
    SOURCE = textwrap.dedent('''\
    @RELATION device
    @ATTRIBUTE power {on, off}
    @ATTRIBUTE mode {fast, slow}
    @ATTRIBUTE cooling {active, inactive}
    @ATTRIBUTE bypass {enabled, disabled}
    @DATA
    on,fast,active,disabled
    off,slow,inactive,disabled
    ''')

    def check(self, rules):
        return constraints.check_constraints(self.SOURCE, rules)

    def test_group_attributes_form(self):
        rules = {'allowed': {
            'attributes': ['power', 'mode'],
            'combinations': [['on', 'fast'], ['off', 'slow']],
        }}
        report = self.check(rules)
        self.assertTrue(report['satisfiable'])

    def test_group_attributes_rejects_wrong_arity(self):
        rules = {'allowed': {
            'attributes': ['power', 'mode'],
            'combinations': [['on']],
        }}
        with self.assertRaises(constraints.BadRule):
            self.check(rules)

    def test_flat_pairs_allowed_combination(self):
        rules = {'allowed': [('power', 'on', 'mode', 'fast'),
                             ('power', 'off', 'mode', 'slow')]}
        report = self.check(rules)
        self.assertTrue(report['satisfiable'])

    def test_flat_pairs_list_is_one_combination(self):
        rules = {'allowed': [[('power', 'on'), ('mode', 'fast')]]}
        row = constraints.check_constraints(self.SOURCE, rules)['rows'][0]
        self.assertTrue(row['has_completion'])
        row = constraints.check_constraints(self.SOURCE, rules)['rows'][1]
        self.assertFalse(row['has_completion'])

    def test_rule_list_with_type(self):
        rules = [
            {'type': 'allowed', 'power': 'on', 'mode': 'fast'},
            {'type': 'allowed', 'power': 'off', 'mode': 'slow'},
            {'type': 'implies', 'if': {'power': 'on'},
             'then': {'cooling': 'active'}},
            {'type': 'excludes', 'power': 'off', 'bypass': 'enabled'},
        ]
        report = self.check(rules)
        self.assertTrue(report['satisfiable'])

    def test_typed_allowed_groups_are_independent_constraints(self):
        source = textwrap.dedent('''\
        @RELATION device
        @ATTRIBUTE power {on, off}
        @ATTRIBUTE mode {fast, slow}
        @ATTRIBUTE cooling {active, inactive}
        @ATTRIBUTE bypass {enabled, disabled}
        @DATA
        on,fast,active,disabled
        on,fast,active,enabled
        ''')
        rules = [
            {'type': 'allowed', 'attributes': ['power', 'mode'],
             'combinations': [['on', 'fast']]},
            {'type': 'allowed', 'attributes': ['cooling', 'bypass'],
             'combinations': [['active', 'disabled']]},
        ]
        report = constraints.check_constraints(source, rules)
        self.assertEqual([row['has_completion'] for row in report['rows']],
                         [True, False])
        row = report['rows'][1]
        # The group requires bypass=disabled, so that field alone is already
        # contradictory even though cooling is free.
        self.assertEqual(row['conflict']['attributes'], ['bypass'])

    def test_implication_string_form(self):
        rules = {'allowed': [('power', 'on', 'mode', 'fast')],
                 'implies': 'power=on -> cooling=active'}
        row = constraints.check_constraints(
            self.SOURCE, rules)['rows'][0]
        self.assertTrue(row['has_completion'])

    def test_exclusion_pair_form(self):
        rules = {'allowed': [('power', 'on', 'mode', 'fast')],
                 'excludes': [{'if': {'power': 'on'},
                               'excludes': {'cooling': 'inactive'}}]}
        row = constraints.check_constraints(
            self.SOURCE.replace('active', 'inactive'), rules)['rows'][0]
        self.assertFalse(row['has_completion'])

    def test_exclusion_string_form(self):
        rules = {'excludes': 'power=on and bypass=enabled'}
        row = constraints.check_constraints(
            self.SOURCE.replace('disabled', 'enabled'), rules)['rows'][0]
        self.assertFalse(row['has_completion'])

    def test_exclusion_not_form(self):
        rules = {'excludes': [{'not': [{'power': 'on'},
                                       {'bypass': 'enabled'}]}]}
        row = constraints.check_constraints(
            self.SOURCE.replace('disabled', 'enabled'), rules)['rows'][0]
        self.assertFalse(row['has_completion'])


class TestValidation(unittest.TestCase):
    SOURCE = textwrap.dedent('''\
    @RELATION device
    @ATTRIBUTE power {on, off}
    @ATTRIBUTE mode {fast, slow}
    @DATA
    on,fast
    ''')

    def test_unknown_rule_attribute(self):
        with self.assertRaises(constraints.BadRule):
            constraints.check_constraints(
                self.SOURCE, {'implies': [{'if': {'nope': 'on'},
                                           'then': {'mode': 'fast'}}]})

    def test_undeclared_nominal_rule_value(self):
        with self.assertRaises(constraints.BadRule):
            constraints.check_constraints(
                self.SOURCE, {'allowed': [{'power': 'broken'}]})

    def test_bad_numeric_rule_value(self):
        source = textwrap.dedent('''\
        @RELATION device
        @ATTRIBUTE power {on, off}
        @ATTRIBUTE level NUMERIC
        @DATA
        on,4
        ''')
        with self.assertRaises(constraints.BadRule):
            constraints.check_constraints(
                source, {'allowed': [{'level': 'not-a-number'}]})

    def test_unknown_rule_section(self):
        with self.assertRaises(constraints.BadRule):
            constraints.check_constraints(self.SOURCE, {'banana': []})

    def test_wrong_row_length(self):
        with self.assertRaises(constraints.BadObservation):
            constraints.check_constraints(
                self.SOURCE.replace('on,fast', 'on'), {})

    def test_undeclared_input_value_is_conflict(self):
        obj = arff.loads(self.SOURCE)
        obj['data'] = [['on', 'broken']]
        report = constraints.check_constraints(
            obj, {'allowed': [{'power': 'on', 'mode': 'fast'}]})
        row = report['rows'][0]
        self.assertEqual(row['status'], 'conflict')
        self.assertEqual(row['conflict']['invalid'], ['mode'])

    def test_empty_allowed_list_is_rule_conflict(self):
        report = constraints.check_constraints(self.SOURCE, {'allowed': []})
        conflict = report['rows'][0]['conflict']
        self.assertTrue(conflict['rule_conflict'])
        self.assertEqual(conflict['attributes'], [])

    def test_rules_can_be_consistent_but_fail_row(self):
        report = constraints.check_constraints(
            self.SOURCE, {'allowed': [{'power': 'off', 'mode': 'slow'}]})
        self.assertFalse(report['rows'][0]['has_completion'])
        self.assertEqual(report['rows'][0]['conflict']['attributes'],
                         ['power'])

    def test_numeric_domain_keeps_any_value(self):
        source = textwrap.dedent('''\
        @RELATION device
        @ATTRIBUTE power {on, off}
        @ATTRIBUTE level NUMERIC
        @DATA
        on,?
        ''')
        rules = {'allowed': [{'power': 'on', 'level': 5}]}
        row = constraints.check_constraints(source, rules)['rows'][0]
        self.assertEqual(row['possible_values']['level'], [5.0])
        self.assertFalse(row['any_value']['level'])

    def test_unconstrained_numeric_domain_has_any_value(self):
        source = textwrap.dedent('''\
        @RELATION device
        @ATTRIBUTE power {on, off}
        @ATTRIBUTE level NUMERIC
        @DATA
        ?,?
        ''')
        row = constraints.check_constraints(source, {})['rows'][0]
        self.assertEqual(row['possible_values']['power'], ['on', 'off'])
        self.assertEqual(row['possible_values']['level'], [])
        self.assertTrue(row['any_value']['level'])

    def test_dataclass_style_rules(self):
        rules = constraints.ConstraintSet(
            allowed=[{'power': 'on', 'mode': 'fast'}])
        report = constraints.check_constraints(self.SOURCE, rules)
        self.assertTrue(report['satisfiable'])

    def test_reusable_checker(self):
        checker = constraints.ConstraintChecker(
            {'allowed': [{'power': 'on', 'mode': 'fast'}]})
        self.assertTrue(checker.check(self.SOURCE)['satisfiable'])
        self.assertFalse(checker.check(
            self.SOURCE.replace('on,fast', 'off,slow'))['satisfiable'])

    def test_mapping_observation(self):
        obj = arff.loads(self.SOURCE)
        obj['data'] = [{'power': 'on', 'mode': None}]
        report = constraints.check_constraints(
            obj, {'allowed': [{'power': 'on', 'mode': 'fast'}]})
        self.assertTrue(report['rows'][0]['has_completion'])


class TestCommandLine(unittest.TestCase):
    def test_cli_reports_conflict_and_missing_ranges(self):
        import io
        import json
        import os
        import tempfile
        from unittest import mock

        arff_path = os.path.join(self.directory, 'devices.arff')
        rules_path = os.path.join(self.directory, 'rules.json')
        with open(arff_path, 'w') as handle:
            handle.write(DEVICE_ARFF)
        with open(rules_path, 'w') as handle:
            json.dump(RULES, handle)
        output = io.StringIO()
        with mock.patch('sys.stdout', output):
            status = constraints.main([arff_path, rules_path,
                                       '--completions'])
        self.assertEqual(status, 1)
        report = json.loads(output.getvalue())
        self.assertEqual(report['unsatisfiable_rows'], [2, 4])
        self.assertEqual(report['rows'][1]['possible_values'],
                         {'mode': ['slow']})

    def test_cli_returns_zero_when_all_rows_have_completions(self):
        import io
        import json
        import os
        from unittest import mock

        arff_path = os.path.join(self.directory, 'devices.arff')
        rules_path = os.path.join(self.directory, 'rules.json')
        with open(arff_path, 'w') as handle:
            handle.write('@RELATION d\n'
                         '@ATTRIBUTE power {on, off}\n'
                         '@DATA\non\n')
        with open(rules_path, 'w') as handle:
            json.dump({'allowed': [{'power': 'on'}]}, handle)
        output = io.StringIO()
        with mock.patch('sys.stdout', output):
            status = constraints.main([arff_path, rules_path])
        self.assertEqual(status, 0)
        self.assertTrue(json.loads(output.getvalue())['satisfiable'])

    def setUp(self):
        import os
        import tempfile
        self._temp = tempfile.TemporaryDirectory()
        self.directory = self._temp.name

    def tearDown(self):
        self._temp.cleanup()


class TestArffIntegration(unittest.TestCase):
    def test_reads_file_like_object(self):
        import io
        report = constraints.check_constraints(
            io.StringIO(DEVICE_ARFF), RULES)
        self.assertEqual(len(report['rows']), 5)

    def test_sparse_rows_treat_omitted_fields_as_missing(self):
        source = textwrap.dedent('''\
        @RELATION device
        @ATTRIBUTE power {on, off}
        @ATTRIBUTE mode {fast, slow, idle}
        @DATA
        { 0 on }
        ''')
        rules = {'allowed': [{'power': 'on', 'mode': 'fast'},
                             {'power': 'on', 'mode': 'slow'}]}
        row = constraints.check_constraints(source, rules)['rows'][0]
        self.assertTrue(row['has_completion'])
        self.assertEqual(row['missing'], ['mode'])
        self.assertEqual(row['possible_values']['mode'], ['fast', 'slow'])

    def test_dense_row_is_still_read_as_dense(self):
        report = constraints.check_constraints(DEVICE_ARFF, RULES)
        self.assertEqual(len(report['rows']), 5)

    def test_arff_module_exposes_checker(self):
        self.assertTrue(hasattr(arff, 'check_constraints'))
        report = arff.check_constraints(DEVICE_ARFF, RULES)
        self.assertEqual(len(report['rows']), 5)
        self.assertEqual(report['unsatisfiable_rows'],
                         constraints.check_constraints(
                             DEVICE_ARFF, RULES)['unsatisfiable_rows'])
        self.assertIsNone(arff.loads(DEVICE_ARFF)['data'][1][1])


if __name__ == '__main__':
    unittest.main()
