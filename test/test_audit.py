"""
Tests for display-precision audit logic.
"""

# Standard Library
from decimal import Decimal

# Community Packages
import pytest

# Xport Modules
from xport.audit import (
    AuditConclusion,
    DisplayInterpretation,
    RoundingRule,
    ValueInterval,
    audit,
    compare_intervals,
    compatible_with_display,
    display_interval,
    parse_display,
    parse_format,
    stored_interval,
)


class TestParseDisplay:
    """
    Verify display-string parsing and scale detection.
    """

    @pytest.mark.parametrize(
        'text, coefficient, decimals, interpretation',
        [
            ('12.3%', Decimal('12.3'), 1, DisplayInterpretation.PERCENT),
            ('12.30%', Decimal('12.30'), 2, DisplayInterpretation.PERCENT),
            ('123.0‰', Decimal('123.0'), 1, DisplayInterpretation.PERMILLE),
            ('0.12304', Decimal('0.12304'), 5, DisplayInterpretation.NUMBER),
            ('12', Decimal('12'), 0, DisplayInterpretation.NUMBER),
            ('12,300', Decimal('12300'), 0, DisplayInterpretation.NUMBER),
            ('-0.5%', Decimal('-0.5'), 1, DisplayInterpretation.PERCENT),
        ],
    )
    def test_parse(self, text, coefficient, decimals, interpretation):
        parsed = parse_display(text)
        assert parsed.coefficient == coefficient
        assert parsed.decimals == decimals
        assert parsed.interpretation is interpretation

    @pytest.mark.parametrize('text', ['', 'abc', '12.3.4', '12 %x'])
    def test_invalid(self, text):
        with pytest.raises(ValueError):
            parse_display(text)


class TestParseFormat:
    """
    Verify SAS format hints.
    """

    def test_percent_format(self):
        hint = parse_format('PERCENT7.1')
        assert hint.name == 'PERCENT'
        assert hint.decimals == 1
        assert hint.interpretation is DisplayInterpretation.PERCENT

    def test_generic_numeric_format_has_no_inferred_scale(self):
        hint = parse_format('8.2')
        assert hint.name == ''
        assert hint.decimals == 2
        assert hint.interpretation is None

    def test_format_without_decimals(self):
        hint = parse_format('COMMA8.')
        assert hint.decimals is None
        assert hint.interpretation is None

    def test_character_format(self):
        hint = parse_format('$CHAR10.')
        assert hint.name == '$CHAR'
        assert hint.decimals is None

    def test_invalid_format(self):
        with pytest.raises(ValueError):
            parse_format('not a format')


class TestDisplayInterval:
    """
    Verify the pre-image intervals and boundary openness.
    """

    def test_headline_percent_interval(self):
        interval = display_interval('12.3%')
        assert interval.lower == Decimal('0.1225')
        assert interval.upper == Decimal('0.1235')
        assert not interval.lower_inclusive
        assert not interval.upper_inclusive
        assert str(interval) == '(0.1225, 0.1235)'

    def test_half_even_excludes_both_ties(self):
        interval = display_interval('12.3%', rounding=RoundingRule.ROUND_HALF_EVEN)
        assert not interval.contains(Decimal('0.1225'))
        assert interval.contains(Decimal('0.1225000001'))
        assert not interval.contains(Decimal('0.1235'))
        assert interval.contains(Decimal('0.1234999999'))

    def test_half_up_includes_lower_tie_only(self):
        interval = display_interval('12.3%', rounding=RoundingRule.ROUND_HALF_UP)
        assert interval.lower_inclusive
        assert not interval.upper_inclusive
        assert interval.contains(Decimal('0.1225'))
        assert not interval.contains(Decimal('0.1235'))

    def test_round_down_is_half_open_and_sign_aware(self):
        positive = display_interval('12.3%', rounding=RoundingRule.ROUND_DOWN)
        assert (positive.lower, positive.lower_inclusive) == (Decimal('0.123'), True)
        assert (positive.upper, positive.upper_inclusive) == (Decimal('0.124'), False)
        negative = display_interval('-12.3%', rounding=RoundingRule.ROUND_DOWN)
        assert (negative.lower, negative.lower_inclusive) == (Decimal('-0.123'), True)
        assert (negative.upper, negative.upper_inclusive) == (Decimal('-0.122'), False)

    def test_more_decimals_mean_narrower_interval(self):
        one_place = display_interval('12.3%')
        two_places = display_interval('12.30%')
        assert one_place.upper - one_place.lower == Decimal('0.001')
        assert two_places.upper - two_places.lower == Decimal('0.0001')
        assert one_place != two_places

    def test_explicit_decimals_widen_display(self):
        interval = display_interval('12', decimals=2)
        assert interval.lower == Decimal('11.995')
        assert interval.upper == Decimal('12.005')

    def test_explicit_decimals_cannot_be_narrower_than_text(self):
        with pytest.raises(ValueError):
            display_interval('12.3%', decimals=0)

    def test_permille_scale(self):
        interval = display_interval('123.0‰')
        assert interval.lower == Decimal('0.12295')
        assert interval.upper == Decimal('0.12305')

    def test_candidate_union_when_rule_unknown(self):
        interval = display_interval('12.3%', rounding=None)
        assert interval.lower == Decimal('0.1225')
        assert interval.upper == Decimal('0.124')
        assert interval.lower_inclusive
        assert not interval.upper_inclusive

    def test_format_spec_supplies_decimals_and_scale(self):
        interval = display_interval('12.3%', format_spec='PERCENT7.1')
        assert interval.lower == Decimal('0.1225')
        assert interval.upper == Decimal('0.1235')

    def test_integer_display_is_closed_both_sides_under_half_even(self):
        interval = display_interval('12', decimals=0)
        assert (interval.lower, interval.lower_inclusive) == (Decimal('11.5'), True)
        assert (interval.upper, interval.upper_inclusive) == (Decimal('12.5'), True)


class TestIntervalOps:
    """
    Verify containment, overlap, and intersection semantics.
    """

    def test_singleton(self):
        singleton = stored_interval('0.12304')
        assert singleton.is_singleton
        assert singleton.contains(Decimal('0.12304'))

    def test_open_boundary_does_not_overlap_at_touch(self):
        left = ValueInterval(Decimal('0'), True, Decimal('1'), False)
        right = ValueInterval(Decimal('1'), True, Decimal('2'), True)
        assert not left.overlaps(right)
        assert left.intersection(right) is None

    def test_closed_boundary_overlaps_at_touch(self):
        left = ValueInterval(Decimal('0'), True, Decimal('1'), True)
        right = ValueInterval(Decimal('1'), True, Decimal('2'), True)
        overlap = left.intersection(right)
        assert overlap == ValueInterval(Decimal('1'), True, Decimal('1'), True)

    def test_unit_mismatch_refuses_comparison(self):
        left = stored_interval(1, unit='fraction')
        right = stored_interval(1, unit='percent-point')
        with pytest.raises(ValueError):
            compare_intervals(left, right)


class TestAudit:
    """
    Verify end-to-end audit verdicts.
    """

    def test_headline_difference_is_explained_by_precision(self):
        result = audit('12.3%', 0.12304)
        assert result.conclusion is AuditConclusion.COMPATIBLE
        assert (result.left.lower, result.left.lower_inclusive) == (
            Decimal('0.1225'), False
        )
        assert (result.left.upper, result.left.upper_inclusive) == (
            Decimal('0.1235'), False
        )
        assert result.right.lower == Decimal('0.12304')
        assert result.overlap.is_singleton
        assert result.overlap.lower == Decimal('0.12304')

    def test_stored_value_outside_interval_is_incompatible(self):
        result = audit('12.3%', 0.124)
        assert result.conclusion is AuditConclusion.INCOMPATIBLE
        assert result.overlap is None

    def test_equal_display_strings_do_not_establish_equal_originals(self):
        first = audit('12.3%', 0.1226)
        second = audit('12.30%', 0.12301)
        assert first.conclusion is AuditConclusion.COMPATIBLE
        assert second.conclusion is AuditConclusion.COMPATIBLE
        assert first.left != second.left

    def test_same_display_covers_many_distinct_originals(self):
        interval = display_interval('12.3%')
        first, second = Decimal('0.1226'), Decimal('0.1234')
        assert first != second
        assert interval.contains(first) and interval.contains(second)

    def test_unknown_rounding_rule_gives_indeterminate_at_tie(self):
        result = audit('12.3%', Decimal('0.1235'), rounding=None)
        assert result.conclusion is AuditConclusion.INDETERMINATE
        rules = {rule: comp.conclusion for rule, comp in result.candidates}
        assert rules[RoundingRule.ROUND_HALF_EVEN] is AuditConclusion.INCOMPATIBLE
        assert rules[RoundingRule.ROUND_HALF_UP] is AuditConclusion.INCOMPATIBLE
        assert rules[RoundingRule.ROUND_DOWN] is AuditConclusion.COMPATIBLE

    def test_unknown_rounding_rule_agrees_away_from_ties(self):
        result = audit('12.3%', 0.12304, rounding=None)
        assert result.conclusion is AuditConclusion.COMPATIBLE
        assert len(result.candidates) == len(RoundingRule)

    def test_format_spec_drives_the_audit(self):
        result = audit('12.3%', 0.12304, format_spec='PERCENT7.1')
        assert result.conclusion is AuditConclusion.COMPATIBLE
        assert result.left.lower == Decimal('0.1225')

    def test_compatible_helper(self):
        assert compatible_with_display(0.12304, '12.3%')
        assert not compatible_with_display(0.124, '12.3%')

    def test_float_token_keeps_shortest_decimal(self):
        result = audit('12.3%', 0.12304)
        assert result.right.lower == Decimal('0.12304')

    def test_non_finite_stored_value_rejected(self):
        with pytest.raises(ValueError):
            audit('12.3%', float('nan'))
