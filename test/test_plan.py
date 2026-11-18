"""
Tests for building ingestion plans from XPORT libraries.
"""

# Standard Library
from datetime import datetime

# Community Packages
import pytest

# Xport Modules
import xport
from xport.plan import (
    Dependency,
    MemberRef,
    ROLE_CODE,
    ROLE_DETAIL,
    ROLE_PRIMARY,
    build_plan,
)


def make_dataset(name, data, **metadata):
    """
    Build a named dataset with stable timestamps.
    """
    dataset = xport.Dataset(data, name=name, **metadata)
    dataset.created = dataset.modified = datetime(2026, 11, 18)
    return dataset


def make_library(*datasets, release='2026R1'):
    """
    Build a library whose datasets carry a release label.
    """
    library = xport.Library(list(datasets))
    for dataset in datasets:
        dataset.sas_version = release
    library.sas_version = release
    return library


class TestMemberIdentity:
    """
    Member occurrences keep their identity even when names collide.
    """

    def test_same_name_different_releases_get_distinct_refs(self):
        primary_one = make_dataset('DM', {'USUBJID': ['A', 'B']})
        primary_two = make_dataset('DM', {'USUBJID': ['C']})
        libraries = [
            ('2025R1', make_library(primary_one, release='2025R1')),
            ('2026R1', make_library(primary_two, release='2026R1')),
        ]
        plan = build_plan(libraries, [Dependency('DM', release='2026R1')])

        refs = plan.find('DM')
        assert len(refs) == 2
        assert refs[0] == MemberRef('DM', 0, '2025R1')
        assert refs[1] == MemberRef('DM', 1, '2026R1')

        manifest = plan.get(refs[1])
        assert manifest.complete
        assert plan.unit_for(refs[1]).ready

    def test_dependency_without_release_is_ambiguous(self):
        libraries = [
            ('2025R1', make_library(make_dataset('DM', {'USUBJID': ['A']}),
                                    release='2025R1')),
            ('2026R1', make_library(make_dataset('DM', {'USUBJID': ['B']}),
                                    release='2026R1')),
        ]
        plan = build_plan(libraries, ['DM'])

        assert any('matches more than one release' in issue for issue in plan.issues)
        for ref in plan.find('DM'):
            assert not plan.get(ref).complete


class TestDependencies:
    """
    Dependencies resolve to concrete member occurrences and references.
    """

    def test_references_resolve_to_member_refs(self):
        primary = make_dataset('DM', {'USUBJID': ['A', 'B']})
        detail = make_dataset('EX', {'USUBJID': ['A', 'B'], 'EXTRT': ['X', 'Y']})
        library = make_library(primary, detail)

        plan = build_plan(library, [('EX', ['DM'])], keys={'DM': ['USUBJID'], 'EX': ['USUBJID']})

        ex_ref = plan.find('EX')[0]
        dm_ref = plan.find('DM')[0]
        assert plan.get(ex_ref).references == (dm_ref,)
        assert plan.get(ex_ref).role == ROLE_DETAIL
        assert plan.get(dm_ref).role == ROLE_PRIMARY

        unit = plan.unit_for(ex_ref)
        assert unit.members == (dm_ref, ex_ref)
        assert unit.ready

    def test_mapping_style_dependencies(self):
        primary = make_dataset('DM', {'USUBJID': ['A']})
        detail = make_dataset('EX', {'USUBJID': ['A']})
        plan = build_plan(
            make_library(primary, detail),
            {'EX': {'requires': ['DM']}},
        )
        ex_ref, dm_ref = plan.find('EX')[0], plan.find('DM')[0]
        assert plan.get(ex_ref).references == (dm_ref,)

    def test_missing_reference_blocks_member(self):
        detail = make_dataset('EX', {'USUBJID': ['A']})
        plan = build_plan(make_library(detail), [('EX', ['DM'])])

        ex_ref = plan.find('EX')[0]
        manifest = plan.get(ex_ref)
        assert not manifest.complete
        assert manifest.unresolved == ('DM',)
        assert not plan.unit_for(ex_ref).ready
        with pytest.raises(ValueError, match='DM'):
            plan.verify(ex_ref)

    def test_reference_prefers_same_release(self):
        libraries = [
            ('2025R1', make_library(make_dataset('DM', {'USUBJID': ['A']}),
                                    release='2025R1')),
            ('2026R1', make_library(
                make_dataset('DM', {'USUBJID': ['B']}),
                make_dataset('EX', {'USUBJID': ['B']}),
                release='2026R1',
            )),
        ]
        plan = build_plan(
            libraries,
            [Dependency('EX', release='2026R1', requires=['DM'])],
        )
        ex_ref = next(ref for ref in plan.find('EX') if ref.release == '2026R1')
        dm_new = next(ref for ref in plan.find('DM') if ref.release == '2026R1')
        assert plan.get(ex_ref).references == (dm_new,)

    def test_reference_can_pin_another_release(self):
        libraries = [
            ('2025R1', make_library(make_dataset('DM', {'USUBJID': ['A']}),
                                    release='2025R1')),
            ('2026R1', make_library(
                make_dataset('DM', {'USUBJID': ['B']}),
                make_dataset('EX', {'USUBJID': ['B']}),
                release='2026R1',
            )),
        ]
        plan = build_plan(
            libraries,
            [Dependency('EX', release='2026R1',
                        requires=[{'name': 'DM', 'release': '2025R1'}])],
        )
        ex_ref = next(ref for ref in plan.find('EX') if ref.release == '2026R1')
        dm_old = next(ref for ref in plan.find('DM') if ref.release == '2025R1')
        assert plan.get(ex_ref).references == (dm_old,)

    def test_unrequested_members_still_get_manifests(self):
        primary = make_dataset('DM', {'USUBJID': ['A']})
        code = make_dataset('SEX', {'CODE': ['M', 'F']})
        plan = build_plan(make_library(primary, code), ['DM'])

        assert len(plan.manifests) == 2
        assert plan.find('SEX')

    def test_single_dataset_input(self):
        dataset = make_dataset('DM', {'USUBJID': ['A']})
        plan = build_plan(dataset, ['DM'])

        ref = plan.find('DM')[0]
        assert ref.ordinal == 0
        assert plan.verify(ref) is None


class TestIncompleteDetail:
    """
    An incomplete detail must not publish a seemingly complete association.
    """

    def test_explicitly_flagged_detail_blocks_whole_unit(self):
        primary = make_dataset('DM', {'USUBJID': ['A', 'B']})
        detail = make_dataset('EX', {'USUBJID': ['A', 'B']})
        plan = build_plan(
            make_library(primary, detail),
            [('EX', ['DM'])],
            incomplete=['EX'],
        )

        ex_ref, dm_ref = plan.find('EX')[0], plan.find('DM')[0]
        unit = plan.unit_for(ex_ref)
        assert unit.members == (dm_ref, ex_ref)
        assert not unit.ready
        assert any('flagged incomplete' in issue for issue in unit.issues)

        # Neither the detail nor its primary may be published as a set.
        assert ex_ref not in plan.acceptable
        assert dm_ref not in plan.acceptable
        assert plan.blocked_units == (unit,)
        assert plan.ready_units == ()
        with pytest.raises(ValueError):
            plan.verify(dm_ref)

    def test_missing_key_variable_is_incomplete(self):
        primary = make_dataset('DM', {'USUBJID': ['A']})
        detail = make_dataset('EX', {'OTHER': ['x']})
        plan = build_plan(
            make_library(primary, detail),
            [('EX', ['DM'])],
            keys={'DM': ['USUBJID'], 'EX': ['USUBJID']},
        )
        ex_ref = plan.find('EX')[0]
        assert any('missing key variable' in issue for issue in plan.get(ex_ref).issues)
        assert not plan.unit_for(ex_ref).ready

    def test_null_key_values_are_incomplete(self):
        primary = make_dataset('DM', {'USUBJID': ['A']})
        detail = make_dataset('EX', {'USUBJID': ['A', None]})
        plan = build_plan(
            make_library(primary, detail),
            [('EX', ['DM'])],
            keys={'EX': ['USUBJID']},
        )
        ex_ref = plan.find('EX')[0]
        assert any('missing key value' in issue for issue in plan.get(ex_ref).issues)
        assert not plan.unit_for(ex_ref).ready

    def test_flag_by_name_and_ordinal_targets_one_occurrence(self):
        libraries = [
            ('2025R1', make_library(make_dataset('EX', {'USUBJID': ['A']}),
                                    release='2025R1')),
            ('2026R1', make_library(make_dataset('EX', {'USUBJID': ['B']}),
                                    release='2026R1')),
        ]
        plan = build_plan(
            libraries,
            [Dependency('EX', release='2025R1'), Dependency('EX', release='2026R1')],
            incomplete=[('EX', 1)],
        )

        old_ref, new_ref = plan.find('EX')
        assert plan.get(old_ref).complete
        assert not plan.get(new_ref).complete


class TestCodeTable:
    """
    A complete standalone code table is independently verifiable.
    """

    def test_code_table_accepted_while_another_unit_blocked(self):
        primary = make_dataset('DM', {'USUBJID': ['A', 'B']})
        detail = make_dataset('EX', {'USUBJID': ['A', 'B']})
        code = make_dataset('SEX', {'CODE': ['M', 'F'], 'DECODE': ['Male', 'Female']})
        plan = build_plan(
            make_library(primary, detail, code),
            [('EX', ['DM']), ('SEX', ())],
            roles={'SEX': ROLE_CODE},
            keys={'SEX': ['CODE'], 'DM': ['USUBJID'], 'EX': ['USUBJID']},
            incomplete=['EX'],
        )

        sex_ref = plan.find('SEX')[0]
        assert plan.get(sex_ref).role == ROLE_CODE
        sex_unit = plan.unit_for(sex_ref)
        assert sex_unit.ready
        assert sex_unit.members == (sex_ref,)

        # The code table passes on its own, despite the blocked association.
        assert plan.verify(sex_ref) is None
        assert sex_ref in plan.acceptable
        assert plan.ready_units == (sex_unit,)
        assert len(plan.blocked_units) == 1

    def test_incomplete_code_table_is_not_acceptable(self):
        code = make_dataset('SEX', {'CODE': ['M', None]})
        plan = build_plan(
            make_library(code),
            ['SEX'],
            roles={'SEX': ROLE_CODE},
            keys={'SEX': ['CODE']},
        )
        sex_ref = plan.find('SEX')[0]
        assert not plan.unit_for(sex_ref).ready
        assert sex_ref not in plan.acceptable
        with pytest.raises(ValueError):
            plan.verify(sex_ref)
