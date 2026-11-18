"""Tests for CRS interpretation, conflict retention and recomputation.

A small pyproj-compatible fake keeps these tests runnable without the optional
``pyproj`` dependency.  Real pyproj behavior is covered by ``test_crs.py``.
"""

import copy
from pathlib import Path

import numpy as np
import pytest

import laspy
from laspy import crs as crs_module
from laspy.crs import (
    CrsConflictError,
    CrsPolicy,
    CrsSource,
    CrsStatus,
    CrsTransformRecord,
    collect_crs_candidates,
    resolve_crs,
)
from laspy.errors import CrsUnresolvedError
from laspy.vlrs.vlrlist import VLRList

CONFLICT_FILE = (
    Path(__file__).parent / "data" / "file_with_both_wkt_and_geotiff_vlrs.las"
)
AUTZEN_FILE = Path(__file__).parent / "data" / "autzen.las"
TEST1_4_FILE = Path(__file__).parent / "data" / "test1_4.las"
SIMPLE_FILE = Path(__file__).parent / "data" / "simple.las"


class FakeCRS:
    """Minimal pyproj.CRS stand-in used only by these unit tests."""

    def __init__(self, code, name=None):
        self.code = int(code)
        self._name = name

    @classmethod
    def from_wkt(cls, wkt):
        from laspy.crs import _wkt_epsg

        code = _wkt_epsg(wkt)
        if code is None:
            # The conflicting fixture's WKT is deliberately a different CRS
            # from its GeoTIFF keys and has no root AUTHORITY node.
            if "NAD83_2011_Nebraska_ft" in wkt:
                return cls(32105, name="NAD83 2011 Nebraska ft")
            raise ValueError("no EPSG code in WKT")
        return cls(code)

    @classmethod
    def from_epsg(cls, code):
        return cls(code)

    @classmethod
    def from_user_input(cls, value):
        if isinstance(value, FakeCRS):
            return value
        if isinstance(value, int):
            return cls(value)
        if isinstance(value, str) and value.lower().startswith("epsg:"):
            return cls(value.split(":", 1)[1])
        raise ValueError(f"unsupported CRS input: {value!r}")

    def to_epsg(self):
        return self.code

    def equals(self, other, ignore_axis_order=False):
        return isinstance(other, FakeCRS) and self.code == other.code

    def __eq__(self, other):
        return self.equals(other)

    def to_string(self):
        return f"EPSG:{self.code}"

    def to_wkt(self):
        return f'PROJCS["fake",AUTHORITY["EPSG","{self.code}"]]'

    @property
    def type_name(self):
        return "Projected CRS"

    @property
    def is_projected(self):
        return True

    @property
    def is_geographic(self):
        return False

    @property
    def is_geocentric(self):
        return False

    @property
    def name(self):
        return self._name or f"Fake EPSG:{self.code}"

    @property
    def geodetic_crs(self):
        return self

    def __str__(self):
        return self.to_string()


class FakeTransformer:
    def __init__(self, source, target):
        self.source = FakeCRS.from_user_input(source)
        self.target = FakeCRS.from_user_input(target)

    @classmethod
    def from_crs(cls, source, target, always_xy=True):
        return cls(source, target)

    def transform(self, x, y, z):
        # Deliberately additive so tests can prove values came from the
        # retained source coordinates rather than from projected values.
        delta = self.target.code - self.source.code
        return (
            np.asarray(x) + delta,
            np.asarray(y) + delta,
            np.asarray(z) + delta,
        )


class FakeProjModule:
    CRS = FakeCRS
    Transformer = FakeTransformer


@pytest.fixture()
def fake_pyproj(monkeypatch):
    monkeypatch.setattr(crs_module, "_pyproj", lambda: FakeProjModule)
    import sys

    monkeypatch.setitem(sys.modules, "pyproj", FakeProjModule)
    return FakeProjModule


def _geotiff_epsg(header):
    for candidate in collect_crs_candidates(header):
        if candidate.source is CrsSource.GEOTIFF:
            return candidate.epsg
    return None


def test_conflict_is_retained_without_projection(fake_pyproj):
    las = laspy.read(CONFLICT_FILE)
    original_xyz = las.xyz.copy()

    interpretation = las.header.resolve_crs()

    assert interpretation.status is CrsStatus.CONFLICT
    assert interpretation.is_conflict
    assert interpretation.has_conflict
    assert interpretation.selected_source is CrsSource.WKT
    assert {candidate.source for candidate in interpretation.candidates} == {
        CrsSource.WKT,
        CrsSource.GEOTIFF,
    }
    # The record keeps both candidates, not just the preferred one.
    assert interpretation.candidate_for(CrsSource.WKT) is not None
    assert interpretation.candidate_for(CrsSource.GEOTIFF) is not None
    assert np.array_equal(las.xyz, original_xyz)


def test_transform_refuses_conflict_and_leaves_coordinates(fake_pyproj):
    las = laspy.read(CONFLICT_FILE)
    original_xyz = las.xyz.copy()

    with pytest.raises(CrsConflictError):
        las.transform_crs(4326, inplace=True)

    assert np.array_equal(las.xyz, original_xyz)


def test_explicit_caller_policy_overrides_version_rule(fake_pyproj):
    las = laspy.read(CONFLICT_FILE)

    wkt = las.header.resolve_crs(policy=CrsPolicy.PREFER_WKT)
    geotiff = las.header.resolve_crs(policy=CrsPolicy.PREFER_GEOTIFF)

    assert wkt.status is CrsStatus.RESOLVED
    assert wkt.selected_source is CrsSource.WKT
    assert wkt.has_conflict, "the rejected candidate must still be recorded"

    assert geotiff.status is CrsStatus.RESOLVED
    assert geotiff.selected_source is CrsSource.GEOTIFF
    assert geotiff.epsg == 32104
    assert geotiff.has_conflict


def test_require_agreement_refuses_disagreement(fake_pyproj):
    las = laspy.read(CONFLICT_FILE)

    interpretation = las.header.resolve_crs(require_agreement=True)

    assert interpretation.status is CrsStatus.CONFLICT
    assert interpretation.crs is None
    assert interpretation.selected is None


def test_non_standard_liblas_wkt_does_not_create_false_conflict(fake_pyproj):
    las = laspy.read(AUTZEN_FILE)

    interpretation = las.header.resolve_crs()

    assert interpretation.status is CrsStatus.RESOLVED
    assert interpretation.selected_source is CrsSource.GEOTIFF
    assert interpretation.epsg == 2994
    assert not interpretation.has_conflict
    # The liblas copy remains visible for audit.
    sources = {candidate.source for candidate in interpretation.candidates}
    assert sources == {CrsSource.WKT, CrsSource.GEOTIFF}
    non_standard = [
        candidate
        for candidate in interpretation.candidates
        if not candidate.is_standard
    ]
    assert len(non_standard) == 1


def test_version_rule_selects_geotiff_for_old_file(fake_pyproj):
    las = laspy.read(AUTZEN_FILE)
    interpretation = las.header.resolve_crs()

    assert interpretation.status is CrsStatus.RESOLVED
    assert interpretation.selected_source is CrsSource.GEOTIFF
    assert "LAS <= 1.3" in interpretation.rule


def test_version_rule_selects_wkt_for_1_4(fake_pyproj):
    las = laspy.read(TEST1_4_FILE)
    interpretation = las.header.resolve_crs()

    assert interpretation.status is CrsStatus.RESOLVED
    assert interpretation.selected_source is CrsSource.WKT
    assert "point format >= 6" in interpretation.rule


def test_missing_crs_is_explicit(fake_pyproj):
    las = laspy.read(SIMPLE_FILE)
    interpretation = las.header.resolve_crs()

    assert interpretation.status is CrsStatus.MISSING
    with pytest.raises(CrsUnresolvedError):
        las.transform_crs(4326, inplace=True)


def test_transform_records_source_and_keeps_input_unchanged(fake_pyproj, tmp_path):
    las = laspy.read(AUTZEN_FILE)
    original_xyz = las.xyz.copy()

    projected = las.transform_crs(4326)

    assert np.array_equal(las.xyz, original_xyz), "default is a copy, not in-place"
    record = projected.crs_transform
    assert isinstance(record, CrsTransformRecord)
    assert record.interpretation.epsg == 2994
    assert record.transformed_count == len(las.points)
    assert np.array_equal(record.source_xyz, original_xyz)
    assert not np.array_equal(projected.xyz, original_xyz)


def test_recompute_uses_original_coordinates_after_metadata_fix(fake_pyproj):
    las = laspy.read(CONFLICT_FILE)
    # Make the file usable via an explicit caller convention: WKT first.
    projected = las.transform_crs(
        5000, policy=CrsPolicy.PREFER_WKT, keep_source_xyz=True
    )
    first_xyz = projected.xyz.copy()

    # Repair the WKT record in a copy of the source header.  Recomputing must
    # replay from the retained source coordinates, not from `first_xyz`.
    corrected_header = projected.crs_transform.source_header.copy()
    wkt_vlr = corrected_header.vlrs.get("WktCoordinateSystemVlr")[0]
    wkt_vlr.string = wkt_vlr.string[:-1] + ',AUTHORITY["EPSG","32106"]]'

    recomputed = projected.recompute_crs(
        source_header=corrected_header, policy=CrsPolicy.PREFER_WKT
    )

    assert not np.array_equal(recomputed.xyz, first_xyz)
    # The replay used the retained original coordinates, not `first_xyz`.
    assert np.array_equal(projected.crs_transform.source_xyz, las.xyz)
    expected = FakeTransformer(FakeCRS(32106), FakeCRS(5000)).transform(
        las.xyz[:, 0], las.xyz[:, 1], las.xyz[:, 2]
    )
    expected = np.vstack(expected).T
    assert np.allclose(recomputed.xyz, expected)


def test_recompute_only_touches_affected_points(fake_pyproj):
    las = laspy.read(CONFLICT_FILE)
    mask = np.zeros(len(las.points), dtype=bool)
    mask[:10] = True

    projected = las.transform_crs(
        5000, policy=CrsPolicy.PREFER_WKT, mask=mask, inplace=True
    )
    untouched = projected.xyz[10:].copy()

    recomputed = projected.recompute_crs(policy=CrsPolicy.PREFER_WKT)

    assert np.array_equal(recomputed.xyz[10:], untouched)
    assert np.array_equal(recomputed.xyz[:10], projected.xyz[:10])


def test_recompute_refuses_metadata_that_is_still_conflicting(fake_pyproj):
    las = laspy.read(CONFLICT_FILE)
    projected = las.transform_crs(5000, policy=CrsPolicy.PREFER_WKT)
    before = projected.xyz.copy()

    with pytest.raises(CrsConflictError):
        projected.recompute_crs(policy=CrsPolicy.VERSION, inplace=True)

    assert np.array_equal(projected.xyz, before)


def test_header_crs_interpretation_json_shape(fake_pyproj):
    las = laspy.read(CONFLICT_FILE)
    payload = las.crs_interpretation().to_dict()

    assert payload["status"] == "conflict"
    assert payload["has_conflict"] is True
    assert len(payload["candidates"]) == 2
    assert payload["candidates"][0]["source"] in {"wkt", "geotiff"}
    assert "wkt" in payload["candidates"][0]


def test_conflict_reports_candidates_needing_validation(fake_pyproj):
    las = laspy.read(CONFLICT_FILE)
    interpretation = las.header.resolve_crs()

    pending = interpretation.needs_validation()

    assert {candidate.source for candidate in pending} == {
        CrsSource.WKT,
        CrsSource.GEOTIFF,
    }
    assert all(candidate.location for candidate in pending)
    payload = interpretation.provenance
    assert payload["candidates"][0]["record_locations"]


def test_apply_crs_header_preserves_non_crs_metadata(fake_pyproj):
    las = laspy.read(AUTZEN_FILE)
    repaired_header = las.header.copy()
    repaired_header.add_crs(FakeCRS(2994))

    before_other_vlrs = [
        vlr
        for vlr in las.vlrs
        if vlr.__class__.__name__
        not in {
            "WktCoordinateSystemVlr",
            "GeoKeyDirectoryVlr",
            "GeoAsciiParamsVlr",
            "GeoDoubleParamsVlr",
        }
    ]

    las.header.apply_crs_header(repaired_header)

    after_other_vlrs = [
        vlr
        for vlr in las.vlrs
        if vlr.__class__.__name__
        not in {
            "WktCoordinateSystemVlr",
            "GeoKeyDirectoryVlr",
            "GeoAsciiParamsVlr",
            "GeoDoubleParamsVlr",
        }
    ]
    assert after_other_vlrs == before_other_vlrs
    assert las.header.resolve_crs().epsg == 2994


def test_crs_in_evlr_participates_in_resolution(fake_pyproj):
    las = laspy.read(CONFLICT_FILE)
    wkt_vlr = las.header.vlrs.extract("WktCoordinateSystemVlr")[0]
    las.evlrs = VLRList([wkt_vlr])

    interpretation = las.header.resolve_crs()

    assert interpretation.status is CrsStatus.CONFLICT
    wkt_candidate = interpretation.candidate_for(CrsSource.WKT)
    assert wkt_candidate.record_locations == (("evlr", 0),)


def test_add_crs_removes_stale_crs_from_evlrs(fake_pyproj):
    las = laspy.read(CONFLICT_FILE)
    old_wkt = las.header.vlrs.extract("WktCoordinateSystemVlr")[0]
    las.evlrs = VLRList([old_wkt])

    las.header.add_crs(FakeCRS(4326))

    assert las.header.resolve_crs().status is CrsStatus.RESOLVED
    assert las.header.resolve_crs().epsg == 4326
    assert las.evlrs.get("WktCoordinateSystemVlr") == []


def test_inplace_repair_and_recompute(fake_pyproj):
    las = laspy.read(CONFLICT_FILE)
    projected = las.transform_crs(5000, policy=CrsPolicy.PREFER_WKT, inplace=True)
    original_xyz = projected.crs_transform.source_xyz.copy()

    # Restore the captured source metadata, repair it, then replay.
    projected.header.apply_crs_header(projected.crs_transform.source_header)
    wkt_vlr = projected.header.vlrs.get("WktCoordinateSystemVlr")[0]
    wkt_vlr.string = wkt_vlr.string[:-1] + ',AUTHORITY["EPSG","32107"]]'

    recomputed = projected.recompute_crs(
        policy=CrsPolicy.PREFER_WKT, inplace=True
    )

    expected = FakeTransformer(FakeCRS(32107), FakeCRS(5000)).transform(
        original_xyz[:, 0], original_xyz[:, 1], original_xyz[:, 2]
    )
    assert np.allclose(recomputed.xyz, np.vstack(expected).T)


def test_parse_crs_info_reports_conflict_without_projection(fake_pyproj):
    las = laspy.read(CONFLICT_FILE)
    crs, interpretation = las.header.parse_crs_info(prefer_wkt=True)

    assert crs is not None, "legacy WKT preference is preserved"
    assert interpretation.selected_source is CrsSource.WKT
    assert interpretation.has_conflict
    assert interpretation.candidates


def test_parse_crs_info_uses_version_rule_when_not_preferred(fake_pyproj):
    las = laspy.read(CONFLICT_FILE)
    _, interpretation = las.header.parse_crs_info(prefer_wkt=None)

    assert interpretation.status is CrsStatus.CONFLICT


def test_add_crs_clears_stale_candidates(fake_pyproj):
    las = laspy.read(CONFLICT_FILE)
    las.header.add_crs(FakeCRS(4326))

    interpretation = las.header.resolve_crs()

    assert interpretation.status is CrsStatus.RESOLVED
    assert not interpretation.has_conflict
    assert interpretation.epsg == 4326


def test_add_crs_geotiff_clears_stale_wkt_flag(fake_pyproj):
    header = laspy.LasHeader(version="1.4", point_format=3)
    header.global_encoding.wkt = True

    header.add_crs(FakeCRS(4326))

    assert header.global_encoding.wkt is False
    assert header.vlrs.get("WktCoordinateSystemVlr") == []
    assert header.vlrs.get("GeoKeyDirectoryVlr")


def test_unparsed_wkt_record_is_retained_when_vlr_is_opaque(fake_pyproj):
    las = laspy.read(CONFLICT_FILE)
    # Simulate an unrecognized WKT VLR payload by keeping only raw bytes.
    wkt_vlr = las.header.vlrs.get("WktCoordinateSystemVlr")[0]
    raw = laspy.VLR(
        user_id=wkt_vlr.user_id,
        record_id=wkt_vlr.record_id,
        description="opaque WKT",
        record_data=wkt_vlr.record_data_bytes(),
    )
    las.header.vlrs.remove(wkt_vlr)
    las.header.vlrs.append(raw)

    candidates = las.header.crs_candidates()
    wkt_candidates = [
        candidate for candidate in candidates if candidate.source is CrsSource.WKT
    ]

    assert len(wkt_candidates) == 1
    assert "NAD83_2011_Nebraska_ft" in wkt_candidates[0].wkt
    assert wkt_candidates[0].record_ids == (2112,)


def test_recompute_uses_object_header_when_it_holds_repaired_source(fake_pyproj):
    las = laspy.read(CONFLICT_FILE)
    projected = las.transform_crs(5000, policy=CrsPolicy.PREFER_WKT)

    # Repair on an edited copy whose header is still the source header.
    repaired = las.copy()
    wkt_vlr = repaired.header.vlrs.get("WktCoordinateSystemVlr")[0]
    wkt_vlr.string = wkt_vlr.string[:-1] + ',AUTHORITY["EPSG","32106"]]'

    recomputed = projected.recompute_crs(
        source_header=repaired.header, policy=CrsPolicy.PREFER_WKT
    )

    expected = FakeTransformer(FakeCRS(32106), FakeCRS(5000)).transform(
        las.xyz[:, 0], las.xyz[:, 1], las.xyz[:, 2]
    )
    assert np.allclose(recomputed.xyz, np.vstack(expected).T)


def test_recompute_falls_back_to_captured_source_header(fake_pyproj):
    las = laspy.read(CONFLICT_FILE)
    projected = las.transform_crs(5000, policy=CrsPolicy.PREFER_WKT)

    # No explicit source_header: the object's header describes the target, so
    # recomputation must use the header captured at transform time.
    recomputed = projected.recompute_crs(policy=CrsPolicy.PREFER_WKT)

    assert np.allclose(recomputed.xyz, projected.xyz)
    assert recomputed.crs_transform.interpretation.epsg == 32105
