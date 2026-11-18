from pathlib import Path

import numpy as np
import pytest

import laspy
from laspy.copc import (
    Bounds,
    CopcInfoVlr,
    HierarchyPage,
    NodeStatus,
    QueryManifest,
    VoxelKey,
    load_octree_for_query,
)

COPC_DATA_DIR = Path(__file__).parent / "data"
SIMPLE_COPC_FILE = COPC_DATA_DIR / "simple.copc.laz"
PAGED_COPC_FILE = COPC_DATA_DIR / "simple_with_page.copc.laz"

LAZRS_SELECTIVE_FLAGS = [
    "XY_RETURNS_CHANNEL",
    "Z",
    "CLASSIFICATION",
    "FLAGS",
    "INTENSITY",
    "SCAN_ANGLE",
    "USER_DATA",
    "POINT_SOURCE_ID",
    "GPS_TIME",
    "RGB",
    "NIR",
    "WAVEPACKET",
    "ALL_EXTRA_BYTES",
]


@pytest.fixture
def fake_lazrs(monkeypatch):
    """Make CopcReader constructible without the optional lazrs package."""
    import sys
    import types

    import laspy.copc as copc

    fake = types.ModuleType("lazrs")
    for i, name in enumerate(LAZRS_SELECTIVE_FLAGS):
        setattr(fake, "SELECTIVE_DECOMPRESS_" + name, 1 << i)
    fake.DecompressionSelection = int
    fake.decompress_points_with_chunk_table = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "lazrs", fake)
    monkeypatch.setattr(copc, "lazrs", fake)
    return fake


class NoReadSource:
    """File-like wrapper that fails if a hierarchy page is read."""

    def __init__(self, source):
        self._source = source

    def seek(self, *args, **kwargs):
        raise AssertionError("plan_query must not seek")

    def read(self, *args, **kwargs):
        raise AssertionError("plan_query must not read")

    def tell(self):
        raise AssertionError("plan_query must not call tell")

    def close(self):
        return self._source.close()


def load(path, manifest=None, **kwargs):
    fetch_pages = kwargs.pop("fetch_pages", True)
    manifest = QueryManifest() if manifest is None else manifest
    with open(path, "rb") as f:
        header = laspy.LasHeader.read_from(f)
        info = next(v for v in header.vlrs if isinstance(v, CopcInfoVlr))
        f.seek(info.hierarchy_root_offset)
        root_page = HierarchyPage.from_bytes(f.read(info.hierarchy_root_size))
        nodes = load_octree_for_query(
            f,
            info,
            root_page,
            manifest=manifest,
            page_cache={},
            fetch_pages=fetch_pages,
            **kwargs,
        )
    return manifest, nodes


def test_manifest_records_full_coverage_of_simple_file():
    manifest, nodes = load(SIMPLE_COPC_FILE)

    assert manifest.is_complete
    assert not manifest.pending_pages
    assert manifest.known_point_count == 1065
    assert manifest.selected_point_count == 1065
    assert len(nodes) == len(manifest.covered_nodes)
    assert all(
        manifest.status_of(key) is NodeStatus.POINTS for key in manifest.covered_nodes
    )


def test_manifest_does_not_call_a_populated_region_empty():
    manifest, nodes = load(SIMPLE_COPC_FILE)

    assert not manifest.is_empty_region_confirmed
    assert len(nodes) > 0


def test_offline_plan_reports_pending_page_instead_of_empty():
    manifest, nodes = load(PAGED_COPC_FILE, fetch_pages=False)

    assert not manifest.is_complete
    assert manifest.has_pending_pages
    assert len(manifest.pending_pages) == 1
    assert manifest.known_point_count == 986
    assert manifest.known_point_count < 1065
    assert not manifest.is_empty_region_confirmed

    pending = next(iter(manifest.pending_pages.values()))
    assert pending.key == VoxelKey(level=2, x=0, y=0, z=0)
    assert pending.offset == 33556
    assert pending.byte_size == 160
    assert manifest.status_of(pending.key) is NodeStatus.PENDING


def test_offline_plan_then_query_resolves_earlier_unknown_node():
    plan, _ = load(PAGED_COPC_FILE, fetch_pages=False)
    session = QueryManifest()
    session.merge(plan)

    query_manifest = QueryManifest()
    query_manifest.previously_pending = plan.unresolved_nodes
    load(PAGED_COPC_FILE, manifest=query_manifest)
    session.merge(query_manifest)

    assert session.is_complete
    assert session.known_point_count == 1065
    assert not session.pending_pages
    assert query_manifest.newly_resolved_nodes == {VoxelKey(2, 0, 0, 0)}
    assert not query_manifest.still_unknown_nodes


def test_page_cache_is_reused_across_queries():
    page_cache = {}
    with open(PAGED_COPC_FILE, "rb") as f:
        header = laspy.LasHeader.read_from(f)
        info = next(v for v in header.vlrs if isinstance(v, CopcInfoVlr))
        f.seek(info.hierarchy_root_offset)
        root_page = HierarchyPage.from_bytes(f.read(info.hierarchy_root_size))

        first = QueryManifest()
        load_octree_for_query(
            f,
            info,
            root_page,
            manifest=first,
            page_cache=page_cache,
            fetch_pages=True,
        )
        second = QueryManifest()
        load_octree_for_query(
            f,
            info,
            root_page,
            manifest=second,
            page_cache=page_cache,
            fetch_pages=True,
        )

    assert first.pages_fetched == 1
    assert second.pages_fetched == 0
    assert len(page_cache) == 1
    assert first.known_point_count == second.known_point_count == 1065


def test_manifest_records_nodes_outside_query_bounds():
    manifest, nodes = load(
        SIMPLE_COPC_FILE,
        query_bounds=Bounds(
            np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])
        ),
    )

    assert nodes == []
    assert manifest.is_complete
    assert manifest.out_of_bounds_nodes
    assert manifest.covered_nodes == {}
    assert not manifest.has_pending_pages


def test_absent_keys_are_reported_as_absent_not_pending():
    manifest, _ = load(SIMPLE_COPC_FILE)

    assert manifest.absent_nodes
    assert all(
        manifest.status_of(key) is NodeStatus.ABSENT for key in manifest.absent_nodes
    )
    assert not (manifest.absent_nodes & manifest.unresolved_nodes)


def test_empty_region_is_only_confirmed_with_complete_evidence():
    outside = Bounds(np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))

    manifest, nodes = load(SIMPLE_COPC_FILE, query_bounds=outside)
    assert nodes == []
    assert manifest.is_complete
    assert manifest.is_empty_region_confirmed

    unsigned_page, _ = load(PAGED_COPC_FILE, fetch_pages=False, level_range=range(3, 4))
    assert not unsigned_page.is_complete
    assert not unsigned_page.is_empty_region_confirmed

    unsigned_page.points_fetched = True
    unsigned_page.returned_point_count = 0
    assert not unsigned_page.is_empty_region_confirmed


def test_manifest_is_json_friendly():
    import json

    manifest, _ = load(PAGED_COPC_FILE, fetch_pages=False)
    as_dict = manifest.as_dict()

    assert as_dict["has_pending_pages"] is True
    assert as_dict["pending_pages"] == [[[2, 0, 0, 0], 33556, 160]]
    assert as_dict["known_point_count"] == 986
    assert isinstance(as_dict["covered_nodes"], dict)
    assert as_dict["covered_nodes"]["0,0,0,0"] == 24
    assert as_dict["covered_node_counts"][0] == [[0, 0, 0, 0], 24]

    encoded = json.dumps(as_dict)
    assert '"is_empty_region_confirmed": false' in encoded


def test_level_filter_keeps_unresolved_page_visible():
    manifest, nodes = load(PAGED_COPC_FILE, level_range=range(0, 1))

    assert manifest.is_complete
    assert manifest.selected_nodes == {VoxelKey(0, 0, 0, 0)}
    assert manifest.selected_point_count == 24
    assert len(nodes) == 1
    assert not manifest.is_empty_region_confirmed


def test_reader_plan_query_does_not_read_hierarchy_pages(fake_lazrs):
    with laspy.CopcReader.open(SIMPLE_COPC_FILE) as reader:
        reader.source = NoReadSource(reader.source)
        manifest = reader.plan_query()

    assert manifest.is_complete
    assert manifest.known_point_count == 1065
    assert manifest.pages_fetched == 0


def test_reader_plan_query_lists_page_it_cannot_read(fake_lazrs):
    with laspy.CopcReader.open(PAGED_COPC_FILE) as reader:
        reader.source = NoReadSource(reader.source)
        manifest = reader.plan_query(level=3)

    assert manifest.has_pending_pages
    assert manifest.known_point_count == 986
    assert not manifest.is_empty_region_confirmed
    assert [p.key for p in manifest.pending_pages.values()] == [VoxelKey(2, 0, 0, 0)]


def test_reader_query_resolves_page_once_and_reuses_it(fake_lazrs):
    with laspy.CopcReader.open(PAGED_COPC_FILE) as reader:
        first_manifest = reader.plan_query()
        assert first_manifest.has_pending_pages

        points, resolved = reader.query_with_manifest()
        assert resolved.pages_fetched == 1
        assert resolved.known_point_count == 1065
        assert resolved.is_complete

        _, second = reader.query_with_manifest()
        assert second.pages_fetched == 0
        assert second.known_point_count == 1065
        assert len(reader.page_cache) == 1

        assert resolved.newly_resolved_nodes == {VoxelKey(2, 0, 0, 0)}


def test_reader_manifest_tracks_earlier_conclusions_across_queries(fake_lazrs):
    with laspy.CopcReader.open(PAGED_COPC_FILE) as reader:
        plan = reader.plan_query()
        assert plan.has_pending_pages

        _, resolved = reader.query_with_manifest()
        assert resolved.newly_resolved_nodes == {VoxelKey(2, 0, 0, 0)}
        assert resolved.still_unknown_nodes == set()

        assert reader.manifest.is_complete
        assert reader.manifest.known_point_count == 1065
        assert not reader.manifest.pending_pages


def test_reader_confirms_empty_only_with_complete_evidence(fake_lazrs):
    outside = Bounds(np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))

    with laspy.CopcReader.open(SIMPLE_COPC_FILE) as reader:
        points, manifest = reader.query_with_manifest(bounds=outside)

        assert len(points) == 0
        assert manifest.is_complete
        assert manifest.is_empty_region_confirmed

    with laspy.CopcReader.open(PAGED_COPC_FILE) as reader:
        plan = reader.plan_query(level=3)

        assert not plan.is_complete
        assert not plan.is_empty_region_confirmed


def test_empty_selection_is_not_an_empty_region(fake_lazrs):
    with laspy.CopcReader.open(SIMPLE_COPC_FILE) as reader:
        points, manifest = reader.query_with_manifest(level=99)

    assert len(points) == 0
    assert manifest.is_complete
    assert manifest.selected_nodes == set()
    assert not manifest.is_empty_region_confirmed
