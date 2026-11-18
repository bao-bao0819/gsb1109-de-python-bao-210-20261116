"""
Tests for control-total substitution rehearsal.
"""

# Standard Library
import math
from fractions import Fraction

# Community Packages
import pandas as pd
import pytest

# Xport Modules
import xport
import xport.control
import xport.v56
from xport.control import Category, ConflictKind


class TestClassify:
    """
    Verify classification of representability problems.
    """

    def test_representable(self):
        for value in [
            0.0,
            -0.0,
            1.0,
            -2.5,
            1e-70,
            xport.control.MIN_REPRESENTABLE,
            xport.control.MAX_REPRESENTABLE,
            -xport.control.MAX_REPRESENTABLE,
            42,
            True,
        ]:
            assert xport.control.classify(value) is None

    def test_missing(self):
        assert xport.control.classify(None) is None
        assert xport.control.classify(float('nan')) is None
        assert xport.control.classify(pd.NA) is None

    def test_special_missing(self):
        for c in '_' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            value = getattr(xport.NaN, c)
            assert xport.control.classify(value) is Category.SPECIAL_MISSING

    def test_overflow(self):
        for value in [1e75, -1e75, 2.0**249, math.inf, -math.inf, 10**400]:
            assert xport.control.classify(value) is Category.OVERFLOW

    def test_underflow(self):
        for value in [1e-80, -1e-80, 1e-300, 2.0**-261, 5e-324, -5e-324]:
            assert xport.control.classify(value) is Category.UNDERFLOW


class TestFindProblems:
    """
    Verify scanning control-total columns for problem fields.
    """

    def test_mapping(self):
        dataset = {
            'A': [1.0, 1e75, float('nan')],
            'B': [1e-80, 2.0, xport.NaN.B],
            'C': ['x', 'y', 'z'],
        }
        problems = xport.control.find_problems(dataset, ['A', 'B'])
        assert [(p.field.column, p.field.row) for p in problems] == [
            ('A', 1),
            ('B', 0),
            ('B', 2),
        ]
        assert [p.category for p in problems] == [
            Category.OVERFLOW,
            Category.UNDERFLOW,
            Category.SPECIAL_MISSING,
        ]

    def test_dataframe_index(self):
        dataset = pd.DataFrame({'A': [1e75, 1.0]}, index=['row1', 'row2'])
        problems = xport.control.find_problems(dataset, ['A'])
        assert problems[0].field == xport.control.Field('A', 'row1')

    def test_special_missing_preserved(self):
        problems = xport.control.find_problems({'A': [xport.NaN.Q]}, ['A'])
        assert problems[0].value is xport.NaN.Q


class TestNeighbor:
    """
    Verify the ``math.nextafter`` stand-in.
    """

    def test_against_math_nextafter(self):
        values = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            5e-324,
            -5e-324,
            xport.control.MAX_REPRESENTABLE,
            -xport.control.MAX_REPRESENTABLE,
            xport.control.MIN_REPRESENTABLE,
            -xport.control.MIN_REPRESENTABLE,
            2.0**-1074,
        ]
        for x in values:
            for toward in (0.0, math.inf, -math.inf, 1.0, -1.0):
                assert xport.control._neighbor(x, toward) == math.nextafter(x, toward)


class TestCandidates:
    """
    Verify replacement candidate generation.
    """

    def test_overflow(self):
        candidates = xport.control._candidates(1e75, Fraction(1e74), 4)
        assert len(candidates) == 4
        assert candidates[0] == xport.control.MAX_REPRESENTABLE
        for candidate in candidates:
            assert candidate <= xport.control.MAX_REPRESENTABLE
            assert abs(Fraction(candidate) - Fraction(1e75)) <= Fraction(1e74)

    def test_overflow_negative(self):
        candidates = xport.control._candidates(-1e75, Fraction(1e74), 2)
        assert candidates[0] == -xport.control.MAX_REPRESENTABLE
        assert all(c < 0 for c in candidates)

    def test_underflow(self):
        candidates = xport.control._candidates(1e-80, Fraction(1e-79), 4)
        assert 0.0 in candidates
        assert xport.control.MIN_REPRESENTABLE not in candidates  # Error too big.

    def test_underflow_negative(self):
        candidates = xport.control._candidates(-1e-80, Fraction(1e-78), 4)
        assert 0.0 in candidates
        assert -xport.control.MIN_REPRESENTABLE in candidates

    def test_no_clamping(self):
        # The tolerance doesn't reach any representable value.
        assert xport.control._candidates(1e75, Fraction(1e70), 4) == ()
        assert xport.control._candidates(1e-80, Fraction(1e-81), 4) == ()

    def test_infinity(self):
        assert xport.control._candidates(math.inf, Fraction(1e99), 4) == ()
        assert xport.control._candidates(-math.inf, Fraction(1e99), 4) == ()

    def test_roundtrip_exact(self):
        # Every candidate must be writable and decode to itself.
        for value in [1e75, -1e75, 1e-80, -1e-80]:
            for candidate in xport.control._candidates(value, Fraction(1e74), 4):
                assert xport.v56.ibm_to_ieee(xport.v56.ieee_to_ibm(candidate)) == candidate


class TestRehearsal:
    """
    Verify end-to-end rehearsal of a two-column control total.
    """

    @pytest.fixture
    def dataset(self):
        return {
            'DEBIT': [1e75, 100.0, xport.NaN.A],
            'CREDIT': [-1e75, -100.0, 0.0],
        }

    def test_feasible(self, dataset):
        rehearsal = xport.control.rehearse(dataset, ['DEBIT', 'CREDIT'], 1e74, 1e58)
        assert rehearsal.feasible
        assert len(rehearsal.groups) == 1
        group = rehearsal.groups[0]
        assert group.category is Category.OVERFLOW
        assert len(group.plans) == 4
        for plan in group.plans:
            assert abs(plan.control_error) <= Fraction(1e58)
            assert plan.max_row_error <= Fraction(1e74)
            for substitution in plan.substitutions:
                candidate = substitution.replacement
                assert xport.v56.ibm_to_ieee(xport.v56.ieee_to_ibm(candidate)) == candidate

    def test_exact_control_error(self, dataset):
        rehearsal = xport.control.rehearse(dataset, ['DEBIT', 'CREDIT'], 1e74, 1e58)
        group = rehearsal.groups[0]
        # Symmetric replacements cancel exactly.
        assert group.suggested.control_error == 0
        for plan in group.plans:
            total = sum((s.error for s in plan.substitutions), Fraction(0))
            assert plan.control_error == total

    def test_disputed(self, dataset):
        rehearsal = xport.control.rehearse(dataset, ['DEBIT', 'CREDIT'], 1e74, 1e58)
        disputed = rehearsal.disputed
        assert len(disputed) == 2
        assert {field.column for field in disputed} == {'DEBIT', 'CREDIT'}

    def test_undisputed(self):
        # Only the nearest candidate satisfies the control tolerance.
        error = Fraction(1e75) - Fraction(xport.control.MAX_REPRESENTABLE)
        rehearsal = xport.control.rehearse({'A': [1e75]}, ['A'], 1e74, error)
        group = rehearsal.groups[0]
        assert len(group.plans) == 1
        assert group.disputed == ()
        assert rehearsal.disputed == ()

    def test_special_missing_independent(self, dataset):
        rehearsal = xport.control.rehearse(dataset, ['DEBIT', 'CREDIT'], 1e74, 1e58)
        assert len(rehearsal.special_missing) == 1
        problem = rehearsal.special_missing[0]
        assert problem.category is Category.SPECIAL_MISSING
        assert problem.field == xport.control.Field('DEBIT', 2)
        # Special missings are never substituted.
        for group in rehearsal.groups:
            for plan in group.plans:
                assert all(s.field != problem.field for s in plan.substitutions)

    def test_categories_independent(self):
        # The overflow group is infeasible, the underflow group is not;
        # rounding the huge value cannot offset losing the tiny one.
        dataset = {
            'BIG': [1e75],
            'SMALL': [1e-80, -1e-80],
        }
        rehearsal = xport.control.rehearse(dataset, ['BIG', 'SMALL'], 1e74, 0)
        assert not rehearsal.feasible
        assert len(rehearsal.groups) == 2
        overflow, underflow = rehearsal.groups
        assert overflow.category is Category.OVERFLOW
        assert overflow.conflict.kind is ConflictKind.CONTROL
        assert underflow.category is Category.UNDERFLOW
        assert underflow.feasible
        assert underflow.suggested.control_error == 0

    def test_zero_control_tolerance(self):
        # Exact cancellation within one category is allowed.
        rehearsal = xport.control.rehearse({'A': [1e75, -1e75]}, ['A'], 1e74, 0)
        group = rehearsal.groups[0]
        assert group.feasible
        assert group.suggested.control_error == 0


class TestConflicts:
    """
    Verify conflict evidence when no legal plan exists.
    """

    def test_row_overflow(self):
        rehearsal = xport.control.rehearse({'A': [1e75]}, ['A'], 1e70, 1e58)
        assert not rehearsal.feasible
        conflict, = rehearsal.conflicts
        assert conflict.kind is ConflictKind.ROW
        assert conflict.category is Category.OVERFLOW
        assert conflict.fields == (xport.control.Field('A', 0), )
        expected = Fraction(1e75) - Fraction(xport.control.MAX_REPRESENTABLE)
        assert conflict.achieved == expected
        assert conflict.best_effort is None

    def test_row_underflow_no_flush_to_zero(self):
        # Zero is outside the per-row tolerance, so no plan is legal.
        rehearsal = xport.control.rehearse({'A': [1e-80]}, ['A'], 1e-81, 1e58)
        conflict, = rehearsal.conflicts
        assert conflict.kind is ConflictKind.ROW
        assert conflict.category is Category.UNDERFLOW
        assert conflict.achieved == Fraction(1e-80)

    def test_row_infinity(self):
        rehearsal = xport.control.rehearse({'A': [math.inf]}, ['A'], 1e99, 1e58)
        conflict, = rehearsal.conflicts
        assert conflict.kind is ConflictKind.ROW
        assert conflict.achieved is None

    def test_control(self):
        rehearsal = xport.control.rehearse({'A': [1e75, 1e75]}, ['A'], 1e74, 1e50)
        assert not rehearsal.feasible
        conflict, = rehearsal.conflicts
        assert conflict.kind is ConflictKind.CONTROL
        assert len(conflict.fields) == 2
        error = Fraction(1e75) - Fraction(xport.control.MAX_REPRESENTABLE)
        assert conflict.achieved == 2 * error
        assert conflict.best_effort.control_error == -2 * error
        assert conflict.best_effort.max_row_error <= Fraction(1e74)

    def test_row_conflict_blocks_enumeration(self):
        # A field without candidates is reported even if siblings are fine.
        rehearsal = xport.control.rehearse({'A': [1e75, 2e75]}, ['A'], 1e74, 1e58)
        conflict, = rehearsal.conflicts
        assert conflict.kind is ConflictKind.ROW
        assert conflict.fields == (xport.control.Field('A', 1), )


class TestValidation:
    """
    Verify argument checking.
    """

    def test_negative_tolerance(self):
        with pytest.raises(ValueError):
            xport.control.rehearse({'A': [1.0]}, ['A'], -1, 1)
        with pytest.raises(ValueError):
            xport.control.rehearse({'A': [1.0]}, ['A'], 1, -1)

    def test_non_numeric_tolerance(self):
        with pytest.raises(ValueError):
            xport.control.rehearse({'A': [1.0]}, ['A'], 'big', 1)
        with pytest.raises(TypeError):
            xport.control.rehearse({'A': [1.0]}, ['A'], object(), 1)
        with pytest.raises(ValueError):
            xport.control.rehearse({'A': [1.0]}, ['A'], float('nan'), 1)

    def test_enumeration_limit(self):
        dataset = {'A': [1e75] * 3}
        with pytest.raises(ValueError, match='max_candidates'):
            xport.control.rehearse(dataset, ['A'], 1e74, 1e58, max_enumerations=10)

    def test_no_problems(self):
        rehearsal = xport.control.rehearse({'A': [1.0, 2.0]}, ['A'], 1, 1)
        assert rehearsal.feasible
        assert rehearsal.groups == ()
        assert rehearsal.disputed == ()
        assert rehearsal.conflicts == ()
