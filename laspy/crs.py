"""Coordinate reference system interpretation for LAS files.

A LAS file may describe its CRS in more than one way.  LAS <= 1.3 normally
uses GeoTIFF keys, while LAS >= 1.4 (and point formats >= 6) uses an OGC WKT
record.  Files produced by other software often contain both, sometimes with
different values.

This module does not change coordinates.  It collects every CRS candidate,
applies an explicit resolution policy, and returns an immutable
:class:`CrsInterpretation` explaining what was selected and why.  Coordinate
operations (see :meth:`laspy.LasData.transform_crs`) must consume this record
before doing any projection work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Iterable, Optional, Sequence, Tuple

import numpy as np

from .errors import (
    CrsConflictError,
    CrsMissingError,
    CrsTransformError,
    CrsUnresolvedError,
)
from .vlrs.known import GeoKeyDirectoryVlr, WktCoordinateSystemVlr

if TYPE_CHECKING:
    from .header import LasHeader
    from .lasdata import LasData


class CrsSource(str, Enum):
    """The metadata family a CRS candidate comes from."""

    WKT = "wkt"
    GEOTIFF = "geotiff"


class CrsStatus(str, Enum):
    """The outcome of interpreting all CRS metadata in a file."""

    #: No CRS VLR / EVLR at all.
    MISSING = "missing"
    #: A single usable CRS could be selected.
    RESOLVED = "resolved"
    #: There are CRS records but they disagree and the policy does not
    #: provide a safe way to choose one.
    CONFLICT = "conflict"
    #: CRS records exist but none could be parsed into a usable CRS.
    UNRESOLVED = "unresolved"
    #: CRS records exist but are malformed / empty.
    INVALID = "invalid"
    #: The records are malformed or unusable; kept as a compatibility alias
    #: for callers that only need to know that interpretation failed.
    ERROR = "invalid"


class CrsPolicy(str, Enum):
    """Rules used to select a usable CRS.

    ``VERSION`` follows the LAS specification:

    * LAS <= 1.3 uses GeoTIFF keys.
    * LAS >= 1.4 with point formats >= 6 uses WKT.
    * LAS >= 1.4 with the WKT global-encoding bit uses WKT.
    * Otherwise GeoTIFF is the authoritative representation.

    ``PREFER_WKT`` and ``PREFER_GEOTIFF`` are explicit caller conventions.
    ``REQUIRE_AGREEMENT`` only resolves when every usable source describes the
    same CRS; otherwise the candidates are returned unresolved.

    The caller conventions always override the version rule.  They never
    change coordinates by themselves.
    """

    VERSION = "version"
    PREFER_WKT = "prefer_wkt"
    PREFER_GEOTIFF = "prefer_geotiff"
    REQUIRE_AGREEMENT = "require_agreement"


@dataclass(frozen=True)
class CrsCandidate:
    """One CRS description found in a VLR or EVLR.

    ``crs`` is a ``pyproj.CRS`` when pyproj is installed and the record could
    be parsed; otherwise it is ``None`` and ``error`` explains why.  The raw
    WKT or GeoTIFF key directory is always retained so a caller can inspect and
    repair the metadata without losing the original values.
    """

    source: CrsSource
    crs: Optional[Any] = None
    wkt: Optional[str] = None
    geokeys: Tuple[Tuple[int, int, int, int], ...] = ()
    epsg: Optional[int] = None
    record_locations: Tuple[Tuple[str, int], ...] = ()
    user_ids: Tuple[str, ...] = ()
    record_ids: Tuple[int, ...] = ()
    is_standard: bool = True
    is_valid: bool = True
    error: Optional[str] = None

    @property
    def is_parsed(self) -> bool:
        return self.crs is not None

    @property
    def fingerprint(self) -> str:
        if self.source is CrsSource.WKT:
            return _normalize_wkt(self.wkt or "")
        return repr(self.geokeys)

    @property
    def description(self) -> str:
        if self.source is CrsSource.WKT:
            return "WKT"
        return "GeoTIFF"

    @property
    def location(self) -> str:
        if not self.record_locations:
            return ""
        container, index = self.record_locations[0]
        return f"{container}[{index}]"

    def to_dict(self) -> dict:
        return {
            "source": self.source.value,
            "standard": self.is_standard,
            "valid": self.is_valid,
            "error": self.error,
            "epsg": self.epsg,
            "wkt": self.wkt,
            "geokeys": list(self.geokeys),
            "record_locations": [list(location) for location in self.record_locations],
            "user_ids": list(self.user_ids),
            "record_ids": list(self.record_ids),
        }

    def __repr__(self) -> str:
        state = "valid" if self.is_valid else "invalid"
        return f"<CrsCandidate({self.source.value}, {state}, epsg={self.epsg})>"


@dataclass(frozen=True)
class CrsInterpretation:
    """The explanation record produced for a file's CRS metadata.

    The record is intentionally side-effect free: creating it never changes a
    coordinate, VLR, or header field.  It retains every candidate, including
    the ones that were rejected, so a later metadata correction can be
    re-evaluated and affected coordinates recomputed from the original values.
    """

    status: CrsStatus
    policy: CrsPolicy
    candidates: Tuple[CrsCandidate, ...] = ()
    selected: Optional[CrsCandidate] = None
    crs: Optional[Any] = None
    rule: str = ""
    has_conflict: bool = False
    unverified: bool = False
    version: str = ""
    point_format_id: int = 0
    wkt_flag: bool = False
    diagnostics: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_resolved(self) -> bool:
        return self.status is CrsStatus.RESOLVED and self.crs is not None

    @property
    def selected_crs(self) -> Optional[Any]:
        """The CRS preferred by the policy, even if sources disagree."""
        return None if self.selected is None else self.selected.crs

    @property
    def is_usable(self) -> bool:
        return self.is_resolved

    @property
    def is_conflict(self) -> bool:
        return self.status is CrsStatus.CONFLICT

    @property
    def is_missing(self) -> bool:
        return self.status is CrsStatus.MISSING

    @property
    def selected_source(self) -> Optional[CrsSource]:
        return None if self.selected is None else self.selected.source

    @property
    def conflicts(self) -> Tuple[CrsCandidate, ...]:
        """Valid candidates that were not selected.

        When the interpretation is unresolved these are the retained
        alternatives; when it is resolved they are the rejected values, kept
        for audit and metadata repair.
        """
        return tuple(
            candidate
            for candidate in self.candidates
            if candidate.is_valid and candidate is not self.selected
        )

    @property
    def rejected(self) -> Tuple[CrsCandidate, ...]:
        """Alias for :attr:`conflicts`."""
        return self.conflicts

    def needs_validation(self) -> Tuple[CrsCandidate, ...]:
        """Candidates whose correction would make the metadata unambiguous.

        In a conflict every valid candidate is returned because no source has
        been accepted as authoritative.  In a resolved interpretation the
        rejected candidates are returned for audit.
        """
        if self.status is CrsStatus.CONFLICT:
            return tuple(
                candidate for candidate in self.candidates if candidate.is_valid
            )
        return self.conflicts

    @property
    def provenance(self) -> dict:
        """The interpretation as a JSON-serializable dictionary."""
        return self.to_dict()

    @property
    def epsg(self) -> Optional[int]:
        if self.crs is None:
            return None if self.selected is None else self.selected.epsg
        try:
            return self.crs.to_epsg()
        except Exception:
            return None

    def candidate_for(self, source: CrsSource) -> Optional[CrsCandidate]:
        for candidate in self.candidates:
            if candidate.source is source:
                return candidate
        return None

    def candidates_for(self, source: CrsSource) -> Tuple[CrsCandidate, ...]:
        return tuple(
            candidate
            for candidate in self.candidates
            if candidate.source is source
        )

    def to_dict(self, include_wkt: bool = True) -> dict:
        return {
            "status": self.status.value,
            "policy": self.policy.value,
            "rule": self.rule,
            "has_conflict": self.has_conflict,
            "unverified": self.unverified,
            "version": self.version,
            "point_format_id": self.point_format_id,
            "wkt_flag": self.wkt_flag,
            "selected_source": (
                None if self.selected_source is None else self.selected_source.value
            ),
            "epsg": self.epsg,
            "candidates": [
                candidate.to_dict()
                if include_wkt
                else {
                    key: value
                    for key, value in candidate.to_dict().items()
                    if key != "wkt"
                }
                for candidate in self.candidates
            ],
            "diagnostics": list(self.diagnostics),
        }

    def raise_if_not_usable(self) -> None:
        """Raise a typed error before a coordinate operation is attempted."""
        if self.is_resolved:
            return
        if self.status is CrsStatus.MISSING:
            raise CrsMissingError("the LAS file has no CRS metadata")
        if self.status is CrsStatus.CONFLICT or self.has_conflict:
            raise CrsConflictError(
                "CRS metadata sources disagree; original coordinates and "
                f"candidates were retained ({self.rule})"
            )
        raise CrsUnresolvedError(
            f"no usable CRS could be selected ({self.rule or self.status.value})"
        )

    def __repr__(self) -> str:
        return (
            f"<CrsInterpretation({self.status.value}, "
            f"source={self.selected_source}, epsg={self.epsg})>"
        )


@dataclass
class CrsTransformRecord:
    """Provenance for a coordinate transformation.

    It keeps the interpretation that was used, the target CRS and, optionally,
    the original coordinates.  ``recompute`` can therefore replay the
    transformation after one CRS record was corrected without treating the
    already-projected coordinates as source coordinates.
    """

    interpretation: CrsInterpretation
    target_crs: Any
    source_xyz: Optional[np.ndarray] = None
    target_xyz: Optional[np.ndarray] = None
    affected: Optional[np.ndarray] = None
    source_header: Optional["LasHeader"] = None
    output_header: Optional["LasHeader"] = field(
        default=None, repr=False, compare=False
    )
    header_updated: bool = False

    @property
    def source_crs(self) -> Any:
        if self.interpretation.crs is not None:
            return self.interpretation.crs
        return self.interpretation.selected_crs

    @property
    def transformed_count(self) -> int:
        if self.affected is not None:
            return int(np.count_nonzero(self.affected))
        if self.target_xyz is not None:
            return len(self.target_xyz)
        return 0

    @property
    def source_crs_string(self) -> Optional[str]:
        return _crs_to_string(self.source_crs)

    def to_dict(self) -> dict:
        return {
            "source": self.interpretation.to_dict(),
            "source_crs": self.source_crs_string,
            "target": _crs_to_string(self.target_crs),
            "transformed_count": self.transformed_count,
            "has_source_xyz": self.source_xyz is not None,
        }


CRSCandidate = CrsCandidate
CRSInterpretation = CrsInterpretation
CRSTransformRecord = CrsTransformRecord
CRSSource = CrsSource
CRSStatus = CrsStatus
CRSPolicy = CrsPolicy


_PROJECTED_CS_KEY = 3072
_GEOGRAPHIC_CS_KEY = 2048


def _normalize_wkt(wkt: str) -> str:
    return " ".join(wkt.split()).casefold()


def _decode_wkt(record_data: bytes) -> str:
    return record_data.split(b"\0", 1)[0].decode("utf-8", errors="replace").strip()


def _top_level_authorities(wkt: str) -> Iterable[Tuple[str, int]]:
    """Yield ``(authority_name, code)`` pairs from the outermost WKT node.

    A regex alone would pick nested ``AUTHORITY`` nodes (for example a unit
    or axis authority), which is not the CRS code.  This small scanner tracks
    bracket depth and only inspects keys directly inside the root node.
    """
    depth = 0
    i = 0
    length = len(wkt)
    while i < length:
        char = wkt[i]
        if char == "[":
            depth += 1
            i += 1
            continue
        if char == "]":
            depth = max(0, depth - 1)
            i += 1
            continue
        if depth == 1 and (char.isalpha() or char == "_"):
            start = i
            while i < length and (wkt[i].isalnum() or wkt[i] in "_-"):
                i += 1
            key = wkt[start:i].upper()
            if key not in ("AUTHORITY", "ID"):
                continue
            while i < length and wkt[i].isspace():
                i += 1
            if i >= length or wkt[i] != "[":
                continue
            end = wkt.find("]", i)
            if end == -1:
                return
            parts = [
                part.strip().strip("\"'")
                for part in wkt[i + 1 : end].split(",")
            ]
            if len(parts) >= 2 and parts[0].upper() == "EPSG":
                try:
                    yield parts[0], int(parts[1])
                except ValueError:
                    pass
            i = end + 1
            continue
        i += 1


def _wkt_epsg(wkt: str) -> Optional[int]:
    for _name, code in _top_level_authorities(wkt):
        return code
    return None


def _geotiff_epsg(geokeys: Iterable[Tuple[int, int, int, int]]) -> Optional[int]:
    projected = None
    geographic = None
    for key_id, _location, _count, value in geokeys:
        if key_id == _PROJECTED_CS_KEY and 1024 <= value <= 32766:
            projected = value
        elif key_id == _GEOGRAPHIC_CS_KEY and 1024 <= value <= 32766:
            geographic = value
    return projected if projected is not None else geographic


def _pyproj():
    try:
        import pyproj
    except ModuleNotFoundError:
        return None
    return pyproj


def _crs_to_string(crs: Any) -> Optional[str]:
    if crs is None:
        return None
    try:
        return crs.to_string()
    except Exception:
        return str(crs)


def _record_location(container: str, index: int) -> Tuple[str, int]:
    return (container, index)


def _iter_vlrs(header: "LasHeader") -> Iterable[Tuple[str, int, Any]]:
    for index, vlr in enumerate(header.vlrs):
        yield "vlr", index, vlr
    if header.evlrs is not None:
        for index, vlr in enumerate(header.evlrs):
            yield "evlr", index, vlr


def _is_wkt_vlr(vlr: Any) -> bool:
    if isinstance(vlr, WktCoordinateSystemVlr):
        return True
    return int(getattr(vlr, "record_id", -1)) == 2112


def _is_geotiff_vlr(vlr: Any) -> bool:
    if isinstance(vlr, GeoKeyDirectoryVlr):
        return True
    return (
        int(getattr(vlr, "record_id", -1)) == 34735
        and str(getattr(vlr, "user_id", "")) == GeoKeyDirectoryVlr.official_user_id()
    )


def _wkt_text(vlr: Any) -> str:
    if isinstance(vlr, WktCoordinateSystemVlr):
        return vlr.string.strip()
    return _decode_wkt(vlr.record_data_bytes())


def _geotiff_vlr(vlr: Any) -> GeoKeyDirectoryVlr:
    if isinstance(vlr, GeoKeyDirectoryVlr):
        return vlr
    return GeoKeyDirectoryVlr.from_raw(vlr)


def _parse_wkt_candidate(
    wkt: str,
) -> Tuple[Optional[Any], Optional[int], Optional[str]]:
    if not wkt:
        return None, None, "empty WKT string"

    epsg = _wkt_epsg(wkt)
    pyproj = _pyproj()
    if pyproj is None:
        return None, epsg, "pyproj is required to parse WKT"

    try:
        crs = pyproj.CRS.from_wkt(wkt)
    except Exception as exc:
        return None, epsg, str(exc)

    if epsg is None:
        try:
            epsg = crs.to_epsg()
        except Exception:
            epsg = None
    return crs, epsg, None

def _parse_geotiff_candidate(
    vlr: GeoKeyDirectoryVlr,
) -> Tuple[Optional[Any], Optional[int], Optional[str]]:
    geokeys = tuple(
        (int(key.id), int(key.tiff_tag_location), int(key.count), int(key.value_offset))
        for key in vlr.geo_keys
    )
    epsg = _geotiff_epsg(geokeys)
    if epsg is None:
        return None, None, "no usable ProjectedCSTypeGeoKey/GeographicTypeGeoKey"

    pyproj = _pyproj()
    if pyproj is None:
        return None, epsg, "pyproj is required to parse GeoTIFF keys"

    try:
        crs = vlr.parse_crs()
    except Exception as exc:
        return None, epsg, str(exc)
    if crs is None:
        return None, epsg, "GeoTIFF keys do not describe a CRS"
    return crs, epsg, None


def collect_crs_candidates(header: "LasHeader") -> Tuple[CrsCandidate, ...]:
    """Collect all distinct WKT and GeoTIFF CRS candidates from a header.

    Identical records duplicated in a VLR and an EVLR (a common occurrence in
    files written by libLAS) are merged into one candidate.  The merged record
    keeps every location and user/record id for auditing and repair.
    """
    wkt_buckets: dict[str, dict] = {}
    geotiff_buckets: dict[str, dict] = {}

    for container, index, vlr in _iter_vlrs(header):
        location = _record_location(container, index)

        if _is_wkt_vlr(vlr):
            wkt = _wkt_text(vlr)
            key = _normalize_wkt(wkt)
            bucket = wkt_buckets.setdefault(
                key,
                {
                    "wkt": wkt,
                    "locations": [],
                    "user_ids": [],
                    "record_ids": [],
                },
            )
            bucket["locations"].append(location)
            user_id = str(getattr(vlr, "user_id", ""))
            if user_id and user_id not in bucket["user_ids"]:
                bucket["user_ids"].append(user_id)
            record_id = int(getattr(vlr, "record_id", 2112))
            if record_id not in bucket["record_ids"]:
                bucket["record_ids"].append(record_id)

        elif _is_geotiff_vlr(vlr):
            geotiff_vlr = None
            try:
                geotiff_vlr = _geotiff_vlr(vlr)
                geokeys = tuple(
                    (
                        int(key.id),
                        int(key.tiff_tag_location),
                        int(key.count),
                        int(key.value_offset),
                    )
                    for key in geotiff_vlr.geo_keys
                )
            except Exception:
                geokeys = ()
            key = repr(geokeys)
            bucket = geotiff_buckets.setdefault(
                key,
                {
                    "vlr": geotiff_vlr,
                    "geokeys": geokeys,
                    "locations": [],
                    "user_ids": [],
                    "record_ids": [],
                },
            )
            bucket["locations"].append(location)
            user_id = str(getattr(vlr, "user_id", ""))
            if user_id and user_id not in bucket["user_ids"]:
                bucket["user_ids"].append(user_id)
            record_id = int(getattr(vlr, "record_id", 34735))
            if record_id not in bucket["record_ids"]:
                bucket["record_ids"].append(record_id)

    candidates = []

    for bucket in wkt_buckets.values():
        crs, epsg, error = _parse_wkt_candidate(bucket["wkt"])
        candidates.append(
            CrsCandidate(
                source=CrsSource.WKT,
                crs=crs,
                wkt=bucket["wkt"],
                epsg=epsg,
                record_locations=tuple(bucket["locations"]),
                user_ids=tuple(bucket["user_ids"]),
                record_ids=tuple(bucket["record_ids"]),
                is_standard=WktCoordinateSystemVlr.official_user_id()
                in bucket["user_ids"],
                is_valid=bool(bucket["wkt"])
                and (crs is not None or (error or "").startswith("pyproj")),
                error=error,
            )
        )

    for bucket in geotiff_buckets.values():
        vlr = bucket["vlr"]
        if vlr is None:
            crs, epsg, error = None, None, "could not parse GeoKeyDirectoryVlr"
        else:
            crs, epsg, error = _parse_geotiff_candidate(vlr)
        candidates.append(
            CrsCandidate(
                source=CrsSource.GEOTIFF,
                crs=crs,
                geokeys=bucket["geokeys"],
                epsg=epsg,
                record_locations=tuple(bucket["locations"]),
                user_ids=tuple(bucket["user_ids"]),
                record_ids=tuple(bucket["record_ids"]),
                is_standard=GeoKeyDirectoryVlr.official_user_id()
                in bucket["user_ids"],
                is_valid=bool(bucket["geokeys"]) and epsg is not None,
                error=error,
            )
        )

    return tuple(candidates)


def _same_crs(left: Any, right: Any) -> bool:
    """Best-effort CRS equality that tolerates minimal test doubles."""
    if left is None or right is None:
        return False
    try:
        return bool(left.equals(right))
    except Exception:
        try:
            return bool(left == right)
        except Exception:
            return False


def _crs_equivalence(left: Any, right: Any) -> Optional[bool]:
    """Compare two parsed CRS objects, preferring full CRS equality.

    ``to_epsg()`` may return the same code for CRS that differ in datum or
    realization, so it is only used as a fallback when a CRS cannot be
    compared structurally.
    """
    if left is None or right is None:
        return None
    try:
        return bool(left.equals(right, ignore_axis_order=True))
    except TypeError:
        pass
    except Exception:
        pass
    try:
        return bool(left.equals(right))
    except Exception:
        pass
    try:
        return bool(left == right)
    except Exception:
        return None


def _candidate_equivalence(
    left: CrsCandidate, right: CrsCandidate
) -> Optional[bool]:
    """Return True/False when equivalence is known, None when it is not."""
    if left.crs is not None and right.crs is not None:
        equal = _crs_equivalence(left.crs, right.crs)
        if equal is not None:
            return equal

    # Fall back to EPSG identity only when at least one side could not be
    # parsed structurally.
    if left.epsg is not None and right.epsg is not None:
        return left.epsg == right.epsg

    if left.source is right.source:
        return left.fingerprint == right.fingerprint

    return None


def _decision_candidates(
    candidates: Sequence[CrsCandidate],
) -> Tuple[CrsCandidate, ...]:
    """Return candidates that participate in CRS resolution.

    Within one source family, standard ``LASF_Projection`` records supersede
    older non-standard copies (libLAS also writes record 2112).  If a family
    has no standard record, its non-standard records still participate: they
    are the only evidence for that representation and must not silently
    disappear from a conflict audit.
    """
    selected = []
    for source in (CrsSource.WKT, CrsSource.GEOTIFF):
        family = [
            candidate for candidate in candidates if candidate.source is source
        ]
        standard = [candidate for candidate in family if candidate.is_standard]
        selected.extend(standard or family)
    return tuple(selected)


def _analyse_candidates(
    candidates: Sequence[CrsCandidate],
) -> Tuple[bool, bool, Tuple[str, ...]]:
    valid = [candidate for candidate in candidates if candidate.is_valid]
    known_conflict = False
    unverified = False
    diagnostics = []

    for i, left in enumerate(valid):
        for right in valid[i + 1 :]:
            equal = _candidate_equivalence(left, right)
            if equal is False:
                known_conflict = True
                diagnostics.append(
                    f"{left.source.value} and {right.source.value} "
                    "describe different CRS"
                )
            elif equal is None:
                unverified = True
                diagnostics.append(
                    f"could not verify that {left.source.value} and "
                    f"{right.source.value} describe the same CRS"
                )

    return known_conflict, unverified, tuple(diagnostics)


def _authoritative_source(header: "LasHeader") -> Tuple[CrsSource, str]:
    version = header.version
    point_format_id = header.point_format.id
    wkt_flag = bool(header.global_encoding.wkt)

    if version >= (1, 4) and point_format_id >= 6:
        return (
            CrsSource.WKT,
            "LAS 1.4 point format >= 6 requires an OGC WKT CRS",
        )
    if version >= (1, 4) and wkt_flag:
        return (
            CrsSource.WKT,
            "LAS 1.4 header has the WKT global-encoding bit set",
        )
    return (
        CrsSource.GEOTIFF,
        "LAS <= 1.3 or point format < 6 uses GeoTIFF keys",
    )


def _candidate_for_source(
    candidates: Sequence[CrsCandidate], source: CrsSource
) -> Tuple[Optional[CrsCandidate], Optional[str]]:
    matching = [candidate for candidate in candidates if candidate.source is source]
    if len(matching) == 1:
        return matching[0], None
    if len(matching) > 1:
        return None, f"multiple different {source.value} records are present"
    return None, None


def _policy_from_arguments(
    policy: Optional[object], prefer_wkt: Optional[bool], require_agreement: bool
) -> CrsPolicy:
    if require_agreement:
        return CrsPolicy.REQUIRE_AGREEMENT
    if prefer_wkt is not None:
        return CrsPolicy.PREFER_WKT if prefer_wkt else CrsPolicy.PREFER_GEOTIFF
    if policy is None:
        return CrsPolicy.VERSION
    if isinstance(policy, CrsPolicy):
        return policy
    normalized = str(policy).strip().lower().replace("-", "_")
    aliases = {
        "auto": CrsPolicy.VERSION,
        "version": CrsPolicy.VERSION,
        "version_rule": CrsPolicy.VERSION,
        "wkt": CrsPolicy.PREFER_WKT,
        "prefer_wkt": CrsPolicy.PREFER_WKT,
        "geotiff": CrsPolicy.PREFER_GEOTIFF,
        "prefer_geotiff": CrsPolicy.PREFER_GEOTIFF,
        "agreement": CrsPolicy.REQUIRE_AGREEMENT,
        "require_agreement": CrsPolicy.REQUIRE_AGREEMENT,
        "strict": CrsPolicy.REQUIRE_AGREEMENT,
    }
    try:
        return aliases[normalized]
    except KeyError:
        raise ValueError(f"unknown CRS policy: {policy!r}") from None


def resolve_crs(
    header: "LasHeader",
    *,
    policy: Optional[object] = None,
    prefer_wkt: Optional[bool] = None,
    require_agreement: bool = False,
) -> CrsInterpretation:
    """Interpret all CRS metadata without touching coordinates.

    Parameters
    ----------
    header:
        A :class:`laspy.LasHeader` (or an object exposing the same
        ``version``, ``point_format``, ``global_encoding``, ``vlrs`` and
        ``evlrs`` attributes).
    policy:
        A :class:`CrsPolicy` or one of its string aliases.
    prefer_wkt:
        Compatibility shortcut for ``policy=PREFER_WKT`` /
        ``policy=PREFER_GEOTIFF``.
    require_agreement:
        When true, the result is only resolved if every usable candidate
        describes the same CRS.
    """
    header = getattr(header, "header", header)
    policy = _policy_from_arguments(policy, prefer_wkt, require_agreement)
    candidates = collect_crs_candidates(header)
    decision_candidates = _decision_candidates(candidates)
    valid = tuple(
        candidate for candidate in decision_candidates if candidate.is_valid
    )
    known_conflict, unverified, diagnostics = _analyse_candidates(
        decision_candidates
    )
    has_conflict = known_conflict or unverified

    version = str(header.version)
    point_format_id = int(header.point_format.id)
    wkt_flag = bool(header.global_encoding.wkt)
    authoritative, version_rule = _authoritative_source(header)

    if not candidates:
        return CrsInterpretation(
            status=CrsStatus.MISSING,
            policy=policy,
            rule="no WKT or GeoTIFF CRS records found",
            version=version,
            point_format_id=point_format_id,
            wkt_flag=wkt_flag,
        )

    if not valid:
        errors = tuple(
            f"{candidate.source.value}: {candidate.error or 'invalid record'}"
            for candidate in candidates
        )
        return CrsInterpretation(
            status=CrsStatus.INVALID,
            policy=policy,
            candidates=candidates,
            rule="CRS records exist but none is usable",
            has_conflict=False,
            unverified=False,
            version=version,
            point_format_id=point_format_id,
            wkt_flag=wkt_flag,
            diagnostics=errors,
        )

    selected = None
    rule = ""

    if policy is CrsPolicy.REQUIRE_AGREEMENT and has_conflict:
        if known_conflict:
            rule = "caller required all CRS sources to agree, but they differ"
        else:
            rule = (
                "caller required all CRS sources to agree, "
                "but they could not be verified"
            )
        return CrsInterpretation(
            status=CrsStatus.CONFLICT,
            policy=policy,
            candidates=candidates,
            rule=rule,
            has_conflict=True,
            unverified=unverified,
            version=version,
            point_format_id=point_format_id,
            wkt_flag=wkt_flag,
            diagnostics=diagnostics,
        )

    if policy is CrsPolicy.PREFER_WKT:
        selected, source_error = _candidate_for_source(valid, CrsSource.WKT)
        rule = "caller requested WKT"
        if selected is None and source_error is None:
            selected, _ = _candidate_for_source(valid, CrsSource.GEOTIFF)
            rule = "caller requested WKT, but no WKT record exists; using GeoTIFF"
    elif policy is CrsPolicy.PREFER_GEOTIFF:
        selected, source_error = _candidate_for_source(valid, CrsSource.GEOTIFF)
        rule = "caller requested GeoTIFF"
        if selected is None and source_error is None:
            selected, _ = _candidate_for_source(valid, CrsSource.WKT)
            rule = "caller requested GeoTIFF, but no GeoTIFF record exists; using WKT"
    else:
        source_error = None
        selected, source_error = _candidate_for_source(valid, authoritative)
        rule = version_rule
        if selected is None and source_error is None:
            fallback = (
                CrsSource.GEOTIFF
                if authoritative is CrsSource.WKT
                else CrsSource.WKT
            )
            selected, _ = _candidate_for_source(valid, fallback)
            rule = (
                f"{version_rule}; no {authoritative.value} record exists, "
                f"using {fallback.value}"
            )

    if selected is None:
        rule = source_error or rule or "CRS sources could not be disambiguated"
        return CrsInterpretation(
            status=CrsStatus.CONFLICT,
            policy=policy,
            candidates=candidates,
            rule=rule,
            has_conflict=True,
            unverified=unverified,
            version=version,
            point_format_id=point_format_id,
            wkt_flag=wkt_flag,
            diagnostics=diagnostics,
        )

    if selected.error is not None and selected.crs is None:
        diagnostics = diagnostics + (f"{selected.source.value}: {selected.error}",)
        status = CrsStatus.UNRESOLVED
    elif selected.crs is None:
        status = CrsStatus.UNRESOLVED
    elif policy is CrsPolicy.REQUIRE_AGREEMENT and has_conflict:
        # Handled before selection.
        status = CrsStatus.CONFLICT
    elif known_conflict or unverified:
        if policy in (CrsPolicy.PREFER_WKT, CrsPolicy.PREFER_GEOTIFF):
            # An explicit caller convention is a deliberate decision: it may
            # select a usable CRS, but the disagreement is still recorded.
            status = CrsStatus.RESOLVED
            diagnostics = diagnostics + (
                f"the caller's {policy.value} convention selected "
                f"{selected.source.value} despite disagreeing candidates; "
                "the rejected candidate(s) were retained",
            )
        else:
            status = CrsStatus.CONFLICT
            diagnostics = diagnostics + (
                "the version rule would select "
                f"{selected.source.value}, but the candidates disagree; no CRS "
                "is safe to use without an explicit caller decision",
            )
    else:
        status = CrsStatus.RESOLVED

    return CrsInterpretation(
        status=status,
        policy=policy,
        candidates=candidates,
        selected=selected,
        crs=selected.crs if status is CrsStatus.RESOLVED else None,
        rule=rule,
        has_conflict=has_conflict,
        unverified=unverified,
        version=version,
        point_format_id=point_format_id,
        wkt_flag=wkt_flag,
        diagnostics=diagnostics,
    )


def interpret_crs(
    header: "LasHeader",
    *,
    policy: Optional[object] = None,
    prefer_wkt: Optional[bool] = None,
    require_agreement: bool = False,
) -> CrsInterpretation:
    """Alias for :func:`resolve_crs`."""
    return resolve_crs(
        header,
        policy=policy,
        prefer_wkt=prefer_wkt,
        require_agreement=require_agreement,
    )


def crs_candidates(header: "LasHeader") -> Tuple[CrsCandidate, ...]:
    """Alias for :func:`collect_crs_candidates`."""
    return collect_crs_candidates(header)


class CrsResolver:
    """Small object-oriented facade around :func:`resolve_crs`."""

    def __init__(
        self,
        header: "LasHeader",
        *,
        policy: Optional[object] = None,
        prefer_wkt: Optional[bool] = None,
        require_agreement: bool = False,
    ):
        self.header = header
        self.policy = _policy_from_arguments(policy, prefer_wkt, require_agreement)
        self.require_agreement = require_agreement

    @property
    def candidates(self) -> Tuple[CrsCandidate, ...]:
        return collect_crs_candidates(self.header)

    def resolve(self) -> CrsInterpretation:
        return resolve_crs(
            self.header,
            policy=self.policy,
            require_agreement=self.require_agreement,
        )

    interpret = resolve

    def resolve_with(self, **kwargs) -> CrsInterpretation:
        """Resolve using overrides for this call only."""
        return resolve_crs(self.header, **kwargs)


CRSResolver = CrsResolver


def artifact_crs_header(
    record: Optional[CrsTransformRecord],
    *,
    updated_header: Optional["LasHeader"] = None,
    candidate_header: Optional["LasHeader"] = None,
) -> Optional["LasHeader"]:
    """Return the header to use when re-interpreting a transformed artifact.

    After ``transform_crs(update_crs=True)`` the object's header describes the
    target CRS, while the transform record retains the source header.  A
    recomputation must interpret the source metadata (or the corrected version
    of it), never the target metadata.

    ``updated_header`` takes precedence, followed by ``candidate_header`` (for
    example the header of an edited copy of the artifact), then the header
    captured at transform time.
    """
    if updated_header is not None:
        return updated_header
    if candidate_header is not None:
        return candidate_header
    if record is None:
        return None
    return record.source_header


def _transform_xyz(
    xyz: np.ndarray,
    source_crs: Any,
    target_crs: Any,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    pyproj = _pyproj()
    if pyproj is None:
        raise CrsTransformError("pyproj is required to transform coordinates")

    try:
        target = pyproj.CRS.from_user_input(target_crs)
    except Exception as exc:
        raise CrsTransformError(f"invalid target CRS: {exc}") from exc

    if source_crs is None:
        raise CrsUnresolvedError("the source CRS is not usable")

    try:
        transformer = pyproj.Transformer.from_crs(
            source_crs, target, always_xy=True
        )
    except Exception as exc:
        raise CrsTransformError(f"could not build CRS transformation: {exc}") from exc

    original = np.asarray(xyz, dtype=np.float64)
    result = original.copy()

    if mask is None:
        selected = np.ones(len(original), dtype=bool)
    else:
        selected = np.asarray(mask, dtype=bool)
        if selected.shape != (len(original),):
            raise ValueError("mask must have one boolean per point")

    if np.any(selected):
        x = original[selected, 0]
        y = original[selected, 1]
        z = original[selected, 2]
        try:
            transformed = transformer.transform(x, y, z)
        except Exception as exc:
            raise CrsTransformError(f"CRS transformation failed: {exc}") from exc

        if len(transformed) == 2:
            x_new, y_new = transformed
            z_new = z
        else:
            x_new, y_new, z_new = transformed

        x_new = np.asarray(x_new, dtype=np.float64)
        y_new = np.asarray(y_new, dtype=np.float64)
        z_new = np.asarray(z_new, dtype=np.float64)
        # A 2D-only CRS is common in LAS files; do not replace a valid
        # height with the non-finite sentinel some PROJ versions return.
        z_new = np.where(np.isfinite(z_new), z_new, z)

        if not (np.all(np.isfinite(x_new)) and np.all(np.isfinite(y_new))):
            raise CrsTransformError(
                "CRS transformation produced non-finite coordinates"
            )

        result[selected, 0] = x_new
        result[selected, 1] = y_new
        result[selected, 2] = z_new

    return result


def transform_crs(
    las: "LasData",
    target_crs: Any,
    *,
    policy: Optional[object] = None,
    prefer_wkt: Optional[bool] = None,
    require_agreement: bool = False,
    inplace: bool = False,
    update_crs: bool = True,
    keep_source_xyz: bool = True,
    mask: Optional[Sequence[bool]] = None,
) -> "LasData":
    """Transform coordinates after resolving and recording the source CRS.

    The source metadata is interpreted *before* any coordinate is changed.
    If it is not safely resolved, :class:`laspy.errors.CrsConflictError` or
    :class:`laspy.errors.CrsUnresolvedError` is raised while the original
    coordinates remain untouched.
    """
    interpretation = resolve_crs(
        las.header,
        policy=policy,
        prefer_wkt=prefer_wkt,
        require_agreement=require_agreement,
    )
    interpretation.raise_if_not_usable()

    pyproj = _pyproj()
    if pyproj is None:
        raise CrsTransformError("pyproj is required to transform coordinates")

    try:
        target = pyproj.CRS.from_user_input(target_crs)
    except Exception as exc:
        raise CrsTransformError(f"invalid target CRS: {exc}") from exc

    mask_array = None if mask is None else np.asarray(mask, dtype=bool)
    if mask_array is not None and mask_array.shape != (len(las.points),):
        raise ValueError("mask must have one boolean per point")
    source_xyz = np.array(las.xyz, dtype=np.float64, copy=True)
    # Capture the source metadata before any in-place header rewrite.
    source_header = las.header.copy()
    target_xyz = _transform_xyz(
        source_xyz, interpretation.crs, target, mask_array
    )

    if inplace:
        work = las
    else:
        work = las.copy()

    new_header = None
    if update_crs:
        import copy

        new_header = copy.deepcopy(work.header)
        try:
            new_header.add_crs(target)
        except Exception as exc:
            raise CrsTransformError(
                f"could not store target CRS metadata: {exc}"
            ) from exc

    # Every operation that can fail has completed.  Only now do we commit.
    work.xyz = target_xyz
    if new_header is not None:
        work.header = new_header

    record = CrsTransformRecord(
        interpretation=interpretation,
        target_crs=target,
        source_xyz=source_xyz if keep_source_xyz else None,
        target_xyz=target_xyz.copy(),
        affected=(
            np.ones(len(target_xyz), dtype=bool)
            if mask_array is None
            else mask_array.copy()
        ),
        source_header=source_header,
        output_header=work.header,
        header_updated=update_crs,
    )
    work._crs_transform = record
    return work
