"""
Tests for unifying SAS Transport v5 and v8 datasets in one store.
"""

# Standard Library
import datetime

# Community Packages
import pytest

# Xport Modules
import xport
import xport.v56
import xport.v89
from xport.unify import (
    Identity,
    IsolationError,
    TargetSpec,
    UnifiedStore,
)


@pytest.fixture()
def v8_library():
    """
    A v8 dataset matching the v5 ``library`` fixture, plus long metadata.
    """
    ds = xport.Dataset(
        data={
            'VIT_STAT': ['ALIVE'] * 3 + ['DEAD'] * 3,
            'ECON': ['POOR', 'NOT', 'UNK'] * 2,
            'COUNT': [1216, 1761, 2517, 254, 60, 137],
            'TEMP': [98.6, 95.4, 86.7, 93.4, 103.5, 56.7],
        },
        name='ECON',
        label='Blank-padded dataset label',
        dataset_type='',
    )
    ds.created = ds.modified = datetime.datetime(2015, 11, 13, 10, 35, 8)
    ds.sas_os = 'W32_7PRO'
    ds.sas_version = '9.3'
    ds['VIT_STAT'].label = 'Vital status'
    ds['VIT_STAT'].format = '$5.'
    ds['VIT_STAT'].width = 8
    ds['ECON'].label = 'Economic status'
    ds['ECON'].format = xport.Format('$CHAR', 4, 0, xport.FormatAlignment.RIGHT)
    ds['ECON'].width = 8
    ds['COUNT'].label = 'Count'
    ds['COUNT'].format = 'comma8.0'
    ds['COUNT'].width = 8
    ds['TEMP'].label = 'Temperature'
    ds['TEMP'].format = '8.1'
    ds['TEMP'].width = 8
    return xport.Library(
        members=[ds],
        created=ds.created,
        modified=ds.modified,
        sas_os=ds.sas_os,
        sas_version=ds.sas_version,
    )


@pytest.fixture()
def long_v8_bytestring():
    """
    A v8 document with a long variable name and a long label.
    """
    ds = xport.Dataset(
        data={'VeryLongVariableName': [1.0, 2.0, float('nan')]},
        name='LONGDS',
        label='',
        dataset_type='',
    )
    ds.created = ds.modified = datetime.datetime(2021, 11, 11, 22, 33, 22)
    ds.sas_os = 'bsd4.2'
    ds.sas_version = '9.4'
    ds['VeryLongVariableName'].label = (
        'A label much longer than forty characters, only legal in v8.'
    )
    ds['VeryLongVariableName'].width = 8
    return xport.v89.dumps(xport.Library(
        members=[ds],
        created=ds.created,
        modified=ds.modified,
        sas_os=ds.sas_os,
        sas_version=ds.sas_version,
    ))


class TestRoundTrip:
    """
    Recompute each original version and read it back with its own loader.
    """

    def test_v5_byte_identical(self, library_bytestring):
        store = UnifiedStore().ingest(library_bytestring, 5)
        assert store.dump(5) == library_bytestring

    def test_v5_verify(self, library_bytestring):
        store = UnifiedStore().ingest(library_bytestring, 5)
        report = store.verify(5)
        assert report.ok, report.discrepancies

    def test_v8_byte_identical(self, long_v8_bytestring):
        store = UnifiedStore().ingest(long_v8_bytestring, 8)
        assert store.dump(8) == long_v8_bytestring

    def test_v8_verify(self, long_v8_bytestring):
        store = UnifiedStore().ingest(long_v8_bytestring, 8)
        report = store.verify(8)
        assert report.ok, report.discrepancies

    def test_unified_ingest_no_conflicts(self, library_bytestring, v8_library):
        store = UnifiedStore().ingest(library_bytestring, 5)
        store.ingest(xport.v89.dumps(v8_library), 8)
        assert store.conflicts == []
        assert store.verify(5).ok
        assert store.verify(8).ok


class TestPlan:
    """
    The shareable schema and the minimal isolated/sidecar sets.
    """

    def test_shared_schema(self, library_bytestring):
        store = UnifiedStore().ingest(library_bytestring, 5)
        plan = store.plan()
        assert set(plan.shared) == {
            ('ECON', 'VIT_STAT'),
            ('ECON', 'ECON'),
            ('ECON', 'COUNT'),
            ('ECON', 'TEMP'),
        }
        assert plan.isolated == {}
        assert plan.sidecar == []

    def test_long_name_and_label_sidecar(self, long_v8_bytestring):
        store = UnifiedStore().ingest(long_v8_bytestring, 8)
        plan = store.plan()
        assert plan.shared == []
        attributes = {(e.field, e.attribute) for e in plan.sidecar}
        assert ('VeryLongVariableName', 'name') in attributes
        assert ('VeryLongVariableName', 'label') in attributes
        for entry in plan.sidecar:
            assert entry.source == 8
            assert 5 in entry.versions

    def test_identity_mapping_covers_long_name(self, long_v8_bytestring):
        identity = Identity('VeryLongVariableName', {5: 'VERYLONG', 8: 'VeryLongVariableName'})
        store = UnifiedStore(identities=[identity]).ingest(long_v8_bytestring, 8)
        plan = store.plan()
        assert ('VeryLongVariableName', 'name') not in {
            (e.field, e.attribute) for e in plan.sidecar
        }
        bytestring = store.dump(5)
        library = xport.v56.loads(bytestring)
        assert list(library['LONGDS'].columns) == ['VERYLONG']
        report = store.verify(5)
        assert report.ok, report.discrepancies

    def test_restricted_target_spec(self, long_v8_bytestring):
        restricted = TargetSpec(8, label_length=40)
        store = UnifiedStore(target_specs=[restricted]).ingest(long_v8_bytestring, 8)
        plan = store.plan()
        labels = [e for e in plan.sidecar if e.attribute == 'label']
        assert any(8 in e.versions for e in labels)
        bytestring = store.dump(8)
        library = xport.v89.loads(bytestring)
        assert len(library['LONGDS']['VeryLongVariableName'].label) <= 40

    def test_isolation(self):
        ds = xport.Dataset(
            {'WIDE': xport.Variable(['x' * 250], vtype=xport.VariableType.CHARACTER, width=250)},
            name='WIDEDS',
        )
        bytestring = xport.v89.dumps(ds)
        store = UnifiedStore().ingest(bytestring, 8)
        plan = store.plan()
        assert ('WIDEDS', 'WIDE') in plan.isolated
        assert 5 in plan.isolated[('WIDEDS', 'WIDE')]
        with pytest.raises(IsolationError):
            store.dump(5)
        report = store.verify(5, drop_isolated=True)
        assert report.ok, report.discrepancies
        assert store.verify(8).ok


class TestProvenance:
    """
    Long names, labels, formats, and special missings have a source.
    """

    def test_metadata_provenance(self, long_v8_bytestring):
        store = UnifiedStore().ingest(long_v8_bytestring, 8)
        records = {(p.field, p.attribute): p for p in store.provenance}
        assert records[('VeryLongVariableName', 'name')].version == 8
        assert records[('VeryLongVariableName', 'label')].version == 8
        assert records[(None, 'name')].value == 'LONGDS'

    def test_special_missing_provenance_and_restore(self):
        ds = xport.Dataset({'X': xport.Variable([1.0, float('nan')], width=8)}, name='SP')
        bytestring = xport.v56.dumps(ds)
        assert b'.\x00' * 1 in bytestring
        bytestring = bytestring.replace(b'.\x00\x00\x00\x00\x00\x00\x00',
                                        b'A\x00\x00\x00\x00\x00\x00\x00')
        store = UnifiedStore().ingest(bytestring, 5)
        plan = store.plan()
        missing = [e for e in plan.sidecar if e.attribute == 'missing']
        assert len(missing) == 1
        assert missing[0].value == {1: 'A'}
        assert missing[0].source == 5
        dumped = store.dump(5)
        assert b'A\x00\x00\x00\x00\x00\x00\x00' in dumped
        report = store.verify(5)
        assert report.ok, report.discrepancies
        assert any('special missing' in note for note in report.explanations)


class TestConflicts:
    """
    Readable in both versions does not imply mutually compatible.
    """

    def test_value_conflict_detected(self, library_bytestring, v8_library):
        v8_library['ECON']['COUNT'] = [0] * 6
        store = UnifiedStore()
        store.ingest(library_bytestring, 5)
        with pytest.warns(UserWarning, match='Conflict'):
            store.ingest(xport.v89.dumps(v8_library), 8)
        assert store.conflicts
