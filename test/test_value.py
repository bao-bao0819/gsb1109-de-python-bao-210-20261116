"""
Tests for SAS custom value formats and grouped views.
"""

# Standard Library
import math
from decimal import Decimal

# Community Packages
import pandas as pd
import pytest

# Xport Modules
import xport
from xport.value import FormatError, IntervalRule, ValueFormat


RISK = """
VALUE RISK
    LOW -< 0  = 'Invalid'
    0   -< 10 = 'Low'
    10  -< 20 = 'Moderate'
    20  -< 30 = 'Moderate'
    30  - HIGH = 'High'
    .   = 'Unknown'
    .A  = 'Not asked'
    OTHER = 'Uncoded';
"""


@pytest.fixture()
def risk():
    """
    A format with two rules sharing a label and several missing rules.
    """
    return ValueFormat.parse(RISK)


class TestParse:
    """
    Verify parsing of SAS VALUE definitions.
    """

    def test_name_and_rules(self, risk):
        assert risk.name == 'RISK'
        assert len(risk) == 8
        assert [rule.number for rule in risk] == [1, 2, 3, 4, 5, 6, 7, 8]

    def test_endpoints(self, risk):
        assert risk.rules[0].evidence == '[LOW, 0)'
        assert risk.rules[2].evidence == '[10, 20)'
        assert risk.rules[4].evidence == '[30, HIGH]'

    def test_line_numbers(self, risk):
        # The VALUE statement is line 2 and rules start on line 3.
        assert risk.rules[0].line == 3
        assert risk.rules[6].line == 9

    @pytest.mark.parametrize(
        'text,evidence',
        [
            ('0 -< 18 = "a"', '[0, 18)'),
            ('0 <- 18 = "a"', '(0, 18]'),
            ('0 - 18 = "a"', '[0, 18]'),
            ('18 = "a"', '[18, 18]'),
            ('LOW - HIGH = "a"', '[LOW, HIGH]'),
            ('-5 -< -1 = "a"', '[-5, -1)'),
        ],
    )
    def test_operators(self, text, evidence):
        assert ValueFormat.parse(text).rules[0].evidence == evidence

    def test_comments_and_whitespace(self):
        fmt = ValueFormat.parse("""
            /* first comment */
            VALUE AGE (NOTSORTED)
                0 <- 18 = 'Child' /* trailing */ ;
                18 - 65 = 'Adult'
            ;
        """)
        assert fmt.name == 'AGE'
        assert [rule.label for rule in fmt] == ['Child', 'Adult']

    def test_quoted_label(self):
        fmt = ValueFormat.parse("0 - 1 = 'O''Brien'; 1 - 2 = \"say 'hi'\"")
        assert [rule.label for rule in fmt] == ["O'Brien", "say 'hi'"]

    def test_invalid_range(self):
        with pytest.raises(FormatError):
            ValueFormat.parse('1,2,3 = "x"')

    def test_reversed_range(self):
        with pytest.raises(FormatError):
            ValueFormat.parse('10 -< 1 = "x"')

    def test_empty_definition(self):
        with pytest.raises(FormatError):
            ValueFormat.parse('   ')

    def test_unterminated_label(self):
        with pytest.raises(FormatError):
            ValueFormat.parse("0 - 1 = 'x")

    def test_multiple_definitions(self):
        with pytest.raises(FormatError, match='one VALUE definition'):
            ValueFormat.parse('VALUE A 0 - 1 = "x"; VALUE B 1 - 2 = "y";')


class TestIntervals:
    """
    Verify interval membership, overlap, and gap detection.
    """

    def test_exclusive_endpoint(self, risk):
        assert risk.assign(0) == 'Low'
        assert risk.assign(10) == 'Moderate'
        # 10 belongs to the second Moderate rule, not the first.
        assert risk.rule_for(10).number == 3

    def test_inclusive_endpoint(self):
        fmt = ValueFormat.parse("0 - 10 = 'Low'; 10 - 20 = 'High'")
        assert fmt.assign(10) == 'Low'
        assert [rule.number for rule in fmt.match_rules(10)] == [1, 2]

    def test_missing_assignment(self, risk):
        assert risk.assign(None) == 'Unknown'
        assert risk.assign(float('nan')) == 'Unknown'
        assert risk.assign(xport.NaN.A) == 'Not asked'
        assert risk.assign(xport.NaN.B) == 'Uncoded'

    def test_other_fallback(self, risk):
        # OTHER is a fallback, not an explicit numeric match.
        assert risk.match_rules(5) == (risk.rules[1],)
        assert risk.match_rules(xport.NaN.Z) == ()
        assert risk.assign(xport.NaN.Z) == 'Uncoded'

    def test_overlap_pair(self):
        fmt = ValueFormat.parse("0 - 10 = 'Low'; 10 - 20 = 'High'")
        (overlap,) = fmt.overlaps()
        assert overlap.left.number == 1
        assert overlap.right.number == 2
        assert overlap.low == overlap.high == Decimal(10)
        assert overlap.low_inclusive and overlap.high_inclusive

    def test_no_overlap_for_open_shared_endpoint(self, risk):
        assert risk.overlaps() == ()

    def test_overlapping_ranges(self):
        fmt = ValueFormat.parse("0 -< 20 = 'A'; 10 -< 30 = 'B'")
        (overlap,) = fmt.overlaps()
        assert overlap.low == Decimal(10)
        assert overlap.high == Decimal(20)

    def test_missing_overlap(self):
        fmt = ValueFormat.parse(". = 'A'; . = 'B'")
        (overlap,) = fmt.overlaps()
        assert overlap.left.kind == 'missing'
        assert overlap.low is None

    def test_gaps(self):
        fmt = ValueFormat.parse("0 -< 10 = 'Low'; 20 -< 30 = 'High'")
        gaps = fmt.gaps()
        assert gaps == ((Decimal(10), True, Decimal(20), False),)
        assert fmt.describe_gaps() == ('[10, 20)',)

    def test_gaps_with_bounds(self):
        fmt = ValueFormat.parse("0 -< 10 = 'Low'")
        assert fmt.describe_gaps((Decimal(0), Decimal(20))) == ('[10, 20]',)

    def test_no_gap_when_neighbours_touch(self, risk):
        assert risk.gaps() == ()

    def test_adjacent_exclusive_neighbours_leave_gap(self):
        fmt = ValueFormat.parse("0 -< 10 = 'A'; 10 <- 20 = 'B'")
        assert fmt.describe_gaps() == ('[10, 10]',)

    def test_diagnose(self):
        fmt = ValueFormat.parse("0 - 10 = 'Low'; 10 - 20 = 'High'; 30 -< 40 = 'Top'")
        diagnostics = fmt.diagnose([0, 5, 10, 15, 35])
        assert len(diagnostics.overlaps) == 1
        assert diagnostics.uncovered == ()
        # 20 is covered by the closed upper endpoint; 30 is excluded by the
        # open lower endpoint of the next rule.
        assert diagnostics.gaps == ((Decimal(20), False, Decimal(30), False),)


class TestGrouping:
    """
    Verify the per-value grouped view and its provenance columns.
    """

    def test_one_row_per_value(self, risk):
        values = [1, 2, 3, 4, 5]
        grouped = risk.group(values)
        assert len(grouped) == len(values)
        assert [row.value for row in grouped] == values

    def test_rule_citation(self, risk):
        row = risk.group([15])[0]
        assert row.label == 'Moderate'
        assert row.rule_number == 3
        assert row.line == 5
        assert row.interval == '[10, 20)'
        assert row.rule_text == "10 -< 20 = 'Moderate'"
    def test_shared_label_keeps_boundaries(self, risk):
        grouped = risk.group([15, 25])
        assert grouped.labels() == ('Moderate', 'Moderate')
        assert [row.rule_number for row in grouped] == [3, 4]
        assert [row.interval for row in grouped] == ['[10, 20)', '[20, 30)']
        by_label = grouped.by_label()
        assert len(by_label['Moderate']) == 2
        assert {entry['interval'] for entry in by_label['Moderate']} == {'[10, 20)', '[20, 30)'}

    def test_summary_counts_per_rule(self, risk):
        summary = risk.group([5, 15, 25, 35]).summary()
        counts = dict(zip(summary['rule_number'], summary['count']))
        assert counts[2] == 1
        assert counts[3] == 1
        assert counts[4] == 1
        assert counts[5] == 1
        assert summary.loc[summary['label'] == 'Moderate', 'rule_number'].tolist() == [3, 4]

    def test_conflicts_expose_overlap(self):
        fmt = ValueFormat.parse("0 - 10 = 'A'; 5 - 15 = 'B'")
        grouped = fmt.group([3, 7, 12])
        conflicts = grouped.conflicts()
        assert [row.value for row in conflicts] == [7]
        assert conflicts[0].matches == (1, 2)
        assert conflicts[0].label == 'A'

    def test_uncovered(self):
        fmt = ValueFormat.parse("0 -< 10 = 'A'")
        grouped = fmt.group([5, 20])
        assert [row.value for row in grouped.uncovered()] == [20]
        assert grouped[1].rule_number is None

    def test_missing_token(self, risk):
        grouped = risk.group([None, xport.NaN.A, xport.NaN.Z])
        assert [row.token for row in grouped] == ['.', '.A', '.Z']
        assert [row.missing for row in grouped] == [True, True, True]
        assert grouped[2].interval == 'OTHER'

    def test_frame(self, risk):
        grouped = risk.group([5, 15], source='SCORE')
        frame = grouped.to_frame()
        assert list(frame.columns) == [
            'value',
            'label',
            'rule_number',
            'line',
            'interval',
            'rule_text',
            'matches',
            'missing',
            'source',
        ]
        assert frame['source'].tolist() == ['SCORE', 'SCORE']
        assert frame['interval'].tolist() == ['[0, 10)', '[10, 20)']

    def test_frame_keeps_index(self, risk):
        index = pd.Index(['a', 'b'], name='SUBJECT')
        grouped = risk.group([5, 15], index=index)
        assert grouped.to_frame().index.tolist() == ['a', 'b']
        assert grouped.to_frame().index.name == 'SUBJECT'

    def test_series_and_dataset(self, risk):
        ds = xport.Dataset({'AGE': [5, 15], 'SEX': ['F', 'M']}, name='TEST')
        grouped = risk.group_variable(ds, 'AGE')
        assert grouped.source == 'AGE'
        assert grouped.labels() == ('Low', 'Moderate')
        frame = grouped.to_frame()
        assert frame['value'].tolist() == [5, 15]

    def test_group_dataframe_keeps_original_columns(self, risk):
        ds = xport.Dataset({'AGE': [5, 15], 'SEX': ['F', 'M']}, name='TEST')
        frame = risk.group_dataframe(ds, 'AGE')
        assert list(frame.columns) == [
            'AGE',
            'SEX',
            'label',
            'rule_number',
            'interval',
            'rule_text',
        ]
        assert frame['AGE'].tolist() == [5, 15]
        assert frame['label'].tolist() == ['Low', 'Moderate']
        assert frame['interval'].tolist() == ['[0, 10)', '[10, 20)']

    def test_group_dataframe_prefix(self, risk):
        ds = pd.DataFrame({'AGE': [5]})
        frame = risk.group_dataframe(ds, 'AGE', prefix='AGE_')
        assert 'AGE_label' in frame
        assert 'AGE_interval' in frame


class TestReclassify:
    """
    Verify incremental reclassification after an interval edit.
    """

    @pytest.fixture()
    def grouped(self, risk):
        return risk.group([-1, 5, 15, 25, 35, None, xport.NaN.A, xport.NaN.Z])

    def test_interval_edit_only_reclassifies_affected(self, grouped):
        new_format = ValueFormat.parse(
            RISK.replace('10  -< 20', '10  -< 22')
        )
        updated = grouped.reclassify(new_format)
        # Widening the rule adds [20, 22), which no sampled value occupies, so
        # no classification changes.  Rows citing the edited rule still get the
        # new interval text, without being re-derived.
        assert updated.reclassified == ()
        assert updated[2].interval == '[10, 22)'
        assert updated[2].label == 'Moderate'
        assert updated[3].interval == '[20, 30)'

    def test_interval_edit_reclassifies_newly_covered_value(self, grouped):
        new_format = ValueFormat.parse(RISK.replace('10  -< 20', '10  -< 26'))
        updated = grouped.reclassify(new_format)
        # 25 joins the edited rule, whose high endpoint now passes 25.
        assert updated.reclassified == (3,)
        assert updated[3].matches == (3, 4)
        assert updated[2].interval == '[10, 26)'

    def test_untouched_rows_are_reused(self, grouped):
        new_format = ValueFormat.parse(RISK.replace('10  -< 20', '10  -< 22'))
        updated = grouped.reclassify(new_format)
        assert updated[0] is grouped[0]
        assert updated[1] is grouped[1]
        assert updated[3] is grouped[3]
        assert updated[4] is grouped[4]
        assert updated[5] is grouped[5]

    def test_label_only_change_is_cheap(self, grouped):
        new_format = ValueFormat.parse(RISK.replace("'Low'", "'Very low'"))
        updated = grouped.reclassify(new_format)
        assert updated.reclassified == ()
        assert updated[1].label == 'Very low'
        assert updated[1].rule_number == 2

    def test_missing_edit_only_reclassifies_missing(self, grouped):
        new_format = ValueFormat.parse(RISK.replace(".   = 'Unknown'", ".   = 'Missing'"))
        updated = grouped.reclassify(new_format)
        assert updated.reclassified == ()
        assert updated[5].label == 'Missing'
        assert updated[6].label == 'Not asked'

    def test_rule_count_change_reclassifies_all(self, grouped):
        new_format = ValueFormat.parse("0 -< 10 = 'A'; 10 - HIGH = 'B'")
        updated = grouped.reclassify(new_format)
        assert updated.reclassified == tuple(range(len(grouped)))

    def test_other_label_change_is_cheap(self, grouped):
        new_format = ValueFormat.parse(RISK.replace("OTHER = 'Uncoded'", "OTHER = 'Other'"))
        updated = grouped.reclassify(new_format)
        assert updated.reclassified == ()
        assert updated[7].label == 'Other'

    def test_changed_rule_creates_new_overlap(self, grouped):
        new_format = ValueFormat.parse(RISK.replace('10  -< 20', '10  -< 26'))
        updated = grouped.reclassify(new_format)
        row = updated[3]
        assert row.matches == (3, 4)
        assert row.label == 'Moderate'
        assert updated[2].interval == '[10, 26)'
        assert updated[3].interval == '[10, 26)'


class TestTable:
    """
    Verify construction from a table of intervals.
    """

    def test_rows(self):
        fmt = ValueFormat.from_table([
            {'low': 0, 'high': 18, 'high_inclusive': False, 'label': 'Under 18'},
            {'low': 18, 'high': 65, 'high_inclusive': False, 'label': '18 to 64'},
            {'low': 65, 'high': 'HIGH', 'label': '65 and over'},
            {'missing': '.', 'label': 'Unknown'},
            {'low': 'OTHER', 'label': 'Uncoded'},
        ], name='AGE')
        assert fmt.name == 'AGE'
        assert [rule.evidence for rule in fmt] == [
            '[0, 18)', '[18, 65)', '[65, HIGH]', '.', 'OTHER'
        ]
        assert fmt.assign(65) == '65 and over'
        assert fmt.assign(None) == 'Unknown'
        assert fmt.assign(xport.NaN.B) == 'Uncoded'

    def test_operators_column(self):
        fmt = ValueFormat.from_table([
            {'low': 0, 'high': 10, 'low_op': '<', 'high_op': '<=', 'label': 'A'},
        ])
        assert fmt.rules[0].evidence == '(0, 10]'

    def test_dataframe(self):
        frame = pd.DataFrame([
            {'low': 0, 'high': 10, 'label': 'Low'},
            {'low': 10, 'high': 20, 'label': 'High'},
        ])
        fmt = ValueFormat.from_table(frame)
        assert len(fmt.overlaps()) == 1
        grouped = fmt.group([10])
        assert grouped[0].matches == (1, 2)
        assert grouped[0].label == 'Low'

    def test_missing_token_column(self):
        fmt = ValueFormat.from_table([{'missing': 'A', 'label': 'Not asked'}])
        assert fmt.rules[0].tag == '.A'
        assert fmt.assign(xport.NaN.A) == 'Not asked'

    def test_bad_row(self):
        with pytest.raises(FormatError):
            ValueFormat.from_table([{'low': 0, 'label': 'No high'}])
        with pytest.raises(FormatError):
            ValueFormat.from_table([{'low': 10, 'high': 0, 'label': 'Reversed'}])
        with pytest.raises(FormatError):
            ValueFormat.from_table([{'low': 0, 'high': 1}])


class TestHelpers:
    """
    Verify display helpers and rule identity.
    """

    def test_endpoint_rendering(self):
        fmt = ValueFormat.parse("0.50 - 1.00 = 'A'")
        assert fmt.rules[0].evidence == '[0.5, 1]'

    def test_rule_equality(self):
        one = IntervalRule(1, 'A', '0 - 1 = "A"', low=Decimal(0), high=Decimal(1))
        two = IntervalRule(1, 'A', '0 - 1 = "A"', low=Decimal(0), high=Decimal(1))
        assert one == two
        assert hash(one) == hash(two)
        assert one != IntervalRule(
            1, 'B', '0 - 1 = "A"', low=Decimal(0), high=Decimal(1)
        )

    def test_same_interval_ignores_label(self):
        one = IntervalRule(1, 'A', '0 - 1 = "A"', low=Decimal(0), high=Decimal(1))
        two = IntervalRule(1, 'B', '0 - 1 = "B"', low=Decimal(0), high=Decimal(1))
        assert one.same_interval(two)

    def test_special_missing_enum(self):
        fmt = ValueFormat.parse(".A = 'Not asked'; ._ = 'Skipped'")
        assert fmt.assign(xport.NaN.A) == 'Not asked'
        assert fmt.assign(xport.NaN._) == 'Skipped'
        assert math.isnan(xport.NaN.A)
