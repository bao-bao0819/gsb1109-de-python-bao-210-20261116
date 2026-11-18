"""
Tests for delivery reconciliation: duplicate imports, stratum totals, and
the minimum review set.
"""

# Community Packages
import pytest

# Xport Modules
import xport
from xport.reconcile import (
    Constraint,
    Observation,
    Tolerance,
    check_constraints,
    collapse_duplicates,
    find_review_set,
    observations_from_dataset,
    reconcile,
)


def make(rows, *constraints, **kwargs):
    return reconcile(rows, constraints, **kwargs)


class TestTolerance:

    def test_symmetric(self):
        t = Tolerance(0.5)
        assert t.contains(10.5, 10)
        assert t.contains(9.5, 10)
        assert not t.contains(10.51, 10)

    def test_asymmetric(self):
        t = Tolerance(lower=0.1, upper=0.9)
        assert t.contains(10.1, 10)
        assert not t.contains(9.85, 10)
        assert t.contains(10.9, 10)

    def test_unbounded_side(self):
        t = Tolerance(lower=None, upper=0.5)
        assert t.contains(-1e9, 10)
        assert not t.contains(10.6, 10)

    def test_negative_rejected(self):
        with pytest.raises(ValueError):
            Tolerance(-0.1)


class TestIdentity:

    def test_duplicate_by_identity(self):
        rows = [
            Observation('a', 'S', 2.0, row='r1'),
            Observation('a', 'S', 2.0, row='r2'),
        ]
        assert collapse_duplicates(rows) == [[0, 1]]

    def test_equal_values_are_not_duplicates(self):
        rows = [
            Observation('a', 'S', 2.0, row='r1'),
            Observation('b', 'S', 2.0, row='r2'),
        ]
        assert collapse_duplicates(rows) == [[0], [1]]

    def test_row_label_does_not_split_identity(self):
        rows = [
            Observation('a', 'S', 2.0, row='first'),
            Observation('a', 'S', 2.0, row='second'),
        ]
        assert collapse_duplicates(rows) == [[0, 1]]

    def test_unhashable_identity(self):
        rows = [
            Observation(['site', 7], 'S', 1.0),
            Observation(['site', 7], 'S', 1.0),
            Observation(['site', 8], 'S', 1.0),
        ]
        assert collapse_duplicates(rows) == [[0, 1], [2]]

    def test_mapping_rows(self):
        rows = [
            {'identity': 'a', 'stratum': 'S', 'weight': 3},
            {'identity': 'a', 'stratum': 'S', 'weight': 3},
        ]
        assert collapse_duplicates(rows) == [[0, 1]]


class TestCheckConstraints:

    def test_stratum_and_global(self):
        rows = [
            Observation('a', 'S1', 2.0),
            Observation('b', 'S2', 3.0),
        ]
        results = check_constraints(rows, [Constraint(5.0), Constraint(2.0, 'S1')])
        assert [r.total for r in results] == [5.0, 2.0]
        assert all(r.pass_ for r in results)

    def test_repeated_import_still_counted(self):
        rows = [
            Observation('a', 'S', 2.0),
            Observation('a', 'S', 2.0),
        ]
        result, = check_constraints(rows, [Constraint(2.0, 'S')])
        assert result.total == 4.0
        assert not result.pass_

    def test_unknown_stratum_rejected(self):
        rows = [Observation('a', 'S', 1.0)]
        with pytest.raises(ValueError):
            check_constraints(rows, [Constraint(1.0, 'missing')])


class TestReconcile:

    def test_nothing_to_do(self):
        rows = [
            Observation('a', 'S', 1.0),
            Observation('b', 'S', 2.0),
        ]
        result = make(rows, Constraint(3.0, 'S'))
        assert result.feasible
        assert result.cost == 0
        assert result.review == ()
        assert result.kept == (0, 1)

    def test_single_duplicate_removed(self):
        rows = [
            Observation('a', 'S', 1.0),
            Observation('a', 'S', 1.0),
            Observation('b', 'S', 2.0),
        ]
        result = make(rows, Constraint(3.0, 'S'))
        assert result.feasible
        # Two occurrences are trusted, so one delivered row is left out.
        assert result.cost == 1
        assert result.solutions == [(0, 2)]
        # The second copy of "a" is an unequivocal duplicate import: the
        # identity says both rows are one sample, so no human review is
        # needed and it is not part of the settled single-copy rows.
        assert result.review == ()
        assert result.disputed == ()
        assert result.certain_duplicates == (1,)
        assert result.kept == (0, 2)

    def test_minimum_set_is_maximum_trusted_set(self):
        rows = [
            Observation('a', 'S', 1.0),
            Observation('b', 'S', 1.0),
            Observation('c', 'S', 1.0),
            Observation('d', 'S', 1.0),
        ]
        # Keep any two of four; all selections are equally good.
        result = make(rows, Constraint(2.0, 'S'))
        assert result.cost == 2
        assert len(result.solutions) == 6
        assert result.review == (0, 1, 2, 3)
        assert result.disputed == (0, 1, 2, 3)

    def test_tie_keeps_all_disputed_members(self):
        rows = [
            Observation('a', 'S', 2.0, row='A'),
            Observation('b', 'S', 2.0, row='B'),
            Observation('c', 'S', 2.0, row='C'),
            Observation('d', 'S', 2.0, row='D'),
        ]
        result = make(rows, Constraint(4.0, 'S'))
        assert result.cost == 2
        assert len(result.solutions) == 6
        assert result.review == (0, 1, 2, 3)
        assert sorted(result.review_labels) == ['A', 'B', 'C', 'D']

    def test_equal_values_different_identity_are_kept_separate(self):
        rows = [
            Observation('a', 'S', 2.0),
            Observation('b', 'S', 2.0),
        ]
        result = make(rows, Constraint(2.0, 'S'))
        assert result.cost == 1
        assert result.review == (0, 1)
        assert len(result.solutions) == 2

    def test_last_copy_never_removed(self):
        rows = [
            Observation('a', 'S', 3.0),
            Observation('b', 'S', 4.0),
        ]
        # Keeping one occurrence alone can never reach a total of zero: the
        # solver may not delete the last copy of every occurrence.
        result = make(rows, Constraint(0.0, 'S'))
        assert not result.feasible
        assert result.message

    def test_empty_selection_is_not_a_reconciliation(self):
        rows = [Observation('a', 'S', 3.0)]
        # A zero control total cannot be met by deleting the whole delivery.
        result = make(rows, Constraint(0.0, 'S'))
        assert not result.feasible
        assert result.message

    def test_per_stratum_constraints(self):
        rows = [
            Observation('a', 'Urban', 2.0),
            Observation('a', 'Urban', 2.0),
            Observation('b', 'Rural', 1.0),
            Observation('b', 'Rural', 1.0),
        ]
        constraints = [Constraint(2.0, 'Urban'), Constraint(1.0, 'Rural')]
        result = make(rows, *constraints)
        # Keeping one copy of each occurrence leaves one row per occurrence
        # out of the trusted table.  The copy doubling is certain.
        assert result.cost == 2
        assert result.solutions == [(0, 2)]
        assert result.review == ()
        assert all(r.pass_ for r in result.constraints)

    def test_global_constraint_added(self):
        rows = [
            Observation('a', 'S1', 2.0),
            Observation('a', 'S1', 2.0),
            Observation('b', 'S2', 3.0),
        ]
        result = make(rows, Constraint(2.0, 'S1'), Constraint(5.0))
        assert result.cost == 1
        assert result.solutions == [(0, 2)]

    def test_tolerance_is_used(self):
        rows = [
            Observation('a', 'S', 3.0),
        ]
        # Same data, same target; only the rounding tolerance differs.
        loose = make(rows, Constraint(3.5, 'S', 0.5))
        assert loose.feasible
        assert loose.cost == 0
        tight = make(rows, Constraint(3.5, 'S', 0.2))
        assert not tight.feasible

    def test_tolerance_changes_feasibility(self):
        rows = [
            Observation('a', 'S', 2.0),
            Observation('b', 'S', 1.0),
        ]
        # Window [2.5, 3.5] accepts keeping both.
        assert make(rows, Constraint(3.0, 'S', 0.5)).cost == 0
        # Window [2.8, 3.2] still accepts their sum of 3.
        assert make(rows, Constraint(3.0, 'S', 0.2)).cost == 0
        # Window [4.8, 5.2] accepts neither sum: infeasible.
        assert not make(rows, Constraint(5.0, 'S', 0.2)).feasible

    def test_asymmetric_tolerance(self):
        rows = [Observation('a', 'S', 2.0, row='A')]
        # lower=1.0, upper=0.5 gives [2.0, 3.5].
        assert make(rows, Constraint(3.0, 'S', Tolerance(lower=1.0, upper=0.5))).cost == 0
        # lower=0.5, upper=1.0 gives [2.5, 4.0].
        assert not make(
            rows, Constraint(3.0, 'S', Tolerance(lower=0.5, upper=1.0))
        ).feasible

    def test_removed_rows_are_not_counted(self):
        rows = [
            Observation('a', 'S', 2.0),
            Observation('a', 'S', 2.0),
            Observation('a', 'S', 2.0),
            Observation('b', 'S', 5.0),
        ]
        result = make(rows, Constraint(7.0, 'S'))
        # Keeping "a" once and "b" once leaves two duplicate copies out.
        assert result.cost == 2
        assert result.solutions == [(0, 3)]
        # Both `a` copies can stay as imported duplicates of one occurrence;
        # neither needs review.
        assert result.review == ()
        assert all(r.pass_ for r in result.constraints)

    def test_unstratified_rows_count_globally_only(self):
        rows = [
            Observation('a', 'S', 2.0),
            Observation('x', None, 1.0),
        ]
        result = make(rows, Constraint(2.0, 'S'), Constraint(3.0))
        assert result.cost == 0

    def test_negative_weight_rejected(self):
        rows = [Observation('a', 'S', -1.0)]
        with pytest.raises(ValueError):
            make(rows, Constraint(0.0))

    def test_negative_contribution_rejected(self):
        rows = [Observation('a', 'S', 1.0, value=-2.0)]
        with pytest.raises(ValueError):
            make(rows, Constraint(0.0))

    def test_negative_weight_allowed(self):
        rows = [
            Observation('a', 'S', -1.0),
            Observation('b', 'S', 2.0),
        ]
        result = make(rows, Constraint(1.0, 'S'), allow_negative_weights=True)
        assert result.feasible
        assert result.cost == 0

    def test_conflicting_copies_go_to_review(self):
        rows = [
            Observation('a', 'S', 1.0),
            Observation('a', 'S', 2.0),
            Observation('b', 'S', 1.0),
        ]
        result = make(rows, Constraint(1.0, 'S'))
        assert result.feasible
        assert result.review == (0, 1)
        assert result.ignored == [[0, 1]]

    def test_conflicting_copies_can_error(self):
        rows = [
            Observation('a', 'S', 1.0),
            Observation('a', 'S', 2.0),
        ]
        with pytest.raises(ValueError):
            make(rows, Constraint(1.0, 'S'), inconsistent='error')

    def test_duplicate_constraint_rejected(self):
        rows = [Observation('a', 'S', 1.0)]
        with pytest.raises(ValueError):
            make(rows, Constraint(1.0, 'S'), Constraint(2.0, 'S'))


class TestOrderIndependence:

    def test_review_set_is_permutation_invariant(self):
        rows = [
            Observation('a', 'S', 2.0),
            Observation('b', 'S', 2.0),
            Observation('c', 'S', 2.0),
            Observation('d', 'S', 2.0),
        ]
        expected = None
        for perm in (
            rows,
            list(reversed(rows)),
            [rows[2], rows[0], rows[3], rows[1]],
        ):
            result = make(perm, Constraint(4.0, 'S'))
            labels = frozenset(result.review_labels)
            if expected is None:
                expected = labels
            assert labels == expected

    def test_solution_sets_match_under_permutation(self):
        rows = [
            Observation('a', 'S', 2.0, row='A'),
            Observation('b', 'S', 2.0, row='B'),
            Observation('c', 'S', 2.0, row='C'),
        ]
        normal = make(rows, Constraint(2.0, 'S'))
        flipped = make(list(reversed(rows)), Constraint(2.0, 'S'))
        normal_sets = {
            frozenset(rows[i].row for i in sol) for sol in normal.solutions
        }
        flipped_rows = list(reversed(rows))
        flipped_sets = {
            frozenset(flipped_rows[i].row for i in sol)
            for sol in flipped.solutions
        }
        assert normal_sets == flipped_sets


class TestDataFrameIntegration:

    def test_observations_from_dataset(self):
        ds = xport.Dataset(
            {
                'IDENTITY': ['a', 'a', 'b'],
                'STRATUM': ['S', 'S', 'S'],
                'WEIGHT': [2.0, 2.0, 3.0],
            },
            name='DELIVERY',
        )
        obs = observations_from_dataset(
            ds, identity='IDENTITY', stratum='STRATUM', weight='WEIGHT'
        )
        assert [o.identity for o in obs] == ['a', 'a', 'b']
        assert [o.contribution for o in obs] == [2.0, 2.0, 3.0]

    def test_value_column_overrides_weight(self):
        ds = xport.Dataset({'ID': ['a'], 'W': [2.0], 'V': [7.0]}, name='D')
        obs = observations_from_dataset(
            ds, identity='ID', stratum=None, weight='W', value='V'
        )
        assert obs[0].contribution == 7.0

    def test_missing_column(self):
        ds = xport.Dataset({'ID': ['a']}, name='D')
        with pytest.raises(ValueError):
            observations_from_dataset(ds, identity='ID', weight='WEIGHT')


class TestFindReviewSet:

    def test_wrapper(self):
        rows = [
            Observation('a', 'S', 2.0),
            Observation('b', 'S', 2.0),
            Observation('c', 'S', 2.0),
        ]
        review = find_review_set(rows, [Constraint(4.0, 'S')])
        assert review == (0, 1, 2)


class TestPackageExport:

    def test_module_and_alias_coexist(self):
        import xport
        import xport.reconcile as module

        assert xport.reconcile_delivery is module.reconcile
        assert xport.Observation is module.Observation


class TestAsDict:

    def test_json_shape(self):
        rows = [
            Observation('a', 'S', 2.0, row='A'),
            Observation('b', 'S', 2.0, row='B'),
        ]
        data = make(rows, Constraint(2.0, 'S')).as_dict()
        assert data['feasible'] is True
        assert data['cost'] == 1
        assert data['review'] == [0, 1]
        assert data['solutions'] == [[0], [1]]


def _brute_force(rows, constraints):
    """
    Reference implementation: enumerate every subset of occurrences.
    """
    groups = collapse_duplicates(rows)
    usable = []
    for gi, g in enumerate(groups):
        values = {(rows[i].stratum, rows[i].contribution) for i in g}
        if len(values) == 1:
            usable.append(gi)

    best = -1
    best_sets = []
    for mask in range(1 << len(usable)):
        if mask == 0:
            continue
        chosen = [usable[i] for i in range(len(usable)) if mask >> i & 1]
        selected = [rows[groups[g][0]] for g in chosen]
        strata = {}
        total = 0.0
        for row in selected:
            total += row.contribution
            if row.stratum is not None:
                strata[row.stratum] = strata.get(row.stratum, 0.0) + row.contribution
        ok = all(
            c.tolerance.contains(
                total if c.stratum is None else strata.get(c.stratum, 0.0),
                c.total,
            )
            for c in constraints
        )
        if not ok:
            continue
        if len(chosen) > best:
            best = len(chosen)
            best_sets = [frozenset(chosen)]
        elif len(chosen) == best:
            best_sets.append(frozenset(chosen))

    if best < 0:
        return None
    kept_anywhere = set().union(*best_sets)
    kept_everywhere = set(best_sets[0])
    for solution in best_sets[1:]:
        kept_everywhere &= solution
    review = set()
    for solution in best_sets:
        for gi, group in enumerate(groups):
            # Only an occurrence that some best selection keeps but another
            # excludes is disputed; one excluded by every selection is a
            # certain removal.
            if gi not in solution and gi in kept_anywhere:
                review.update(group)
    # Inconsistent groups are neither in a solution nor in the oracle's
    # usable set; their copies need review.
    for gi, group in enumerate(groups):
        if gi not in usable:
            review.update(group)
    return best, review, kept_everywhere


class TestAgainstBruteForce:

    def test_randomized_equivalence(self):
        import random

        rng = random.Random(11)
        for _ in range(150):
            k = rng.randint(1, 2)
            present = ['S1', 'S2'][:k]
            n = rng.randint(k, 6)
            strata = present + [rng.choice(present) for _ in range(n - k)]
            rng.shuffle(strata)
            rows = [
                Observation(
                    'abcdef'[i % 6],
                    strata[i],
                    rng.choice([1.0, 2.0, 3.0]),
                    row=str(i),
                )
                for i in range(n)
            ]
            constraints = [
                Constraint(
                    rng.choice([2.0, 4.0, 6.0]),
                    rng.choice(present),
                    rng.choice([0.0, 0.5]),
                )
            ]
            expected = _brute_force(rows, constraints)
            result = reconcile(rows, constraints)
            assert (expected is None) == (not result.feasible)
            if expected is None:
                continue
            best, review, kept_everywhere = expected
            groups = collapse_duplicates(rows)
            representative = result.solutions[0]
            kept = sum(1 for g in groups if any(i in representative for i in g))
            assert kept == best
            assert set(result.review) == review
            certain = set()
            for gi, group in enumerate(groups):
                if len({(rows[i].stratum, rows[i].contribution) for i in group}) == 1 \
                        and gi in kept_everywhere:
                    certain.update(group[1:])
            assert set(result.certain_duplicates) == certain


class TestScale:

    def test_many_rows_with_imported_copies(self):
        import random

        rng = random.Random(3)
        rows = []
        for i in range(400):
            stratum = f'S{i % 30}'
            weight = rng.randint(1, 5) + 0.25
            rows.append(Observation(f'id{i}', stratum, weight))
            if rng.random() < 0.15:
                rows.append(Observation(f'id{i}', stratum, weight))
        result = reconcile(rows, [Constraint(45 * 30, None, 200)])
        assert result.feasible
        assert result.complete
        # Duplicate copies of a trusted occurrence are certain; only
        # occurrences that some best selection drops need review.
        groups = collapse_duplicates(rows)
        duplicated = {i for group in groups if len(group) > 1 for i in group}
        assert not (duplicated & set(result.review))
        assert not (set(result.kept) & set(result.review))
        assert result.cost == len(rows) - len(result.solutions[0])
