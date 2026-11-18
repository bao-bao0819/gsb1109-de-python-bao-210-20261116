"""Plan and execute batch migrations of LAS point records to a restricted
target version.

This module addresses the situation where several files, potentially using
different (old) point formats, must be migrated to a single restricted target
version.  Classification codes, bit flags and extra dimensions do not have the
same expressible range in every point format (e.g. classification is a 5-bit
field in formats 0-5 but an 8-bit field in formats 6-10), so a migration may
be lossy.

The caller describes, through a :class:`MigrationPolicy`, which degradations
are acceptable for each business field and how much each of them costs per
affected point.  :func:`plan_batch_migration` then selects, among the point
formats allowed by the restricted target version, the single conversion
(target point format + one degradation per business field) that satisfies the
batch-wide constraints with the minimal total loss.  The same business field
is migrated with the same interpretation for every file of the batch.

Points that cannot be migrated under the chosen degradations are not forced
into the target format (which would corrupt them and make other points misuse
the format); they are isolated ("quarantined") into a separate
:class:`laspy.LasData` that keeps the original point format.

The resulting :class:`BatchMigrationResult` reports, per file and per field,
the original codes, the actual loss, and a provable read-back correspondence:
every migrated (and quarantined) file is written to an in-memory buffer and
read again, and the field values are compared one by one.
"""

import copy
import enum
import io
import logging
from typing import Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from . import errors
from .header import Version
from .lasdata import LasData
from .lib import read_las
from .point import dims
from .point.format import PointFormat
from .point.record import PackedPointRecord
from .vlrs.vlrlist import VLRList

logger = logging.getLogger(__name__)

__all__ = [
    "DegradationKind",
    "Degradation",
    "MigrationPolicy",
    "FieldDecision",
    "FieldReport",
    "FileMigrationResult",
    "BatchMigrationResult",
    "MigrationPlan",
    "plan_batch_migration",
]


class MigrationError(errors.LaspyException):
    """Raised when no feasible migration plan satisfies the batch constraints"""


# scan_angle (formats >= 6) is stored in units of 0.006 degrees while
# scan_angle_rank (formats < 6) is stored in whole degrees.
# 0.006 == 3 / 500, so a scan_angle value is exactly representable as a
# whole-degree rank iff it is a multiple of 500.
_SCAN_ANGLE_NUM = 3
_SCAN_ANGLE_DEN = 500
_SCAN_ANGLE_EXACT_STEP = 500
_SCAN_ANGLE_RANK_MIN = -128
_SCAN_ANGLE_RANK_MAX = 127

#: business fields whose values are range-limited by the point format
_RANGE_FIELDS = ("classification", "return_number", "number_of_returns", "scan_angle")

#: fields for which original/stored code tables are reported (bounded code space)
_CODE_REPORT_FIELDS = ("classification", "return_number", "number_of_returns")

#: flags that only exist in point formats >= 6
_NEW_FAMILY_FLAGS = ("overlap", "scanner_channel")

#: presence-constrained standard fields mapped to the dimension(s) backing them
_PRESENCE_FIELD_DIMS: Dict[str, Tuple[str, ...]] = {
    "gps_time": ("gps_time",),
    "red": ("red",),
    "green": ("green",),
    "blue": ("blue",),
    "nir": ("nir",),
    "waveform": dims.WAVEFORM_FIELDS_NAMES,
    "overlap": ("overlap",),
    "scanner_channel": ("scanner_channel",),
}

#: sub-fields that exist in both point format families and are always copied
_SHARED_SUB_FIELDS = (
    "synthetic",
    "key_point",
    "withheld",
    "scan_direction_flag",
    "edge_of_flight_line",
)

#: standard dimensions that are copied verbatim (same type in every format)
_VERBATIM_DIMS = ("X", "Y", "Z", "intensity", "user_data", "point_source_id")

#: dimensions backing a composed (bit-packed) field, never copied directly
_COMPOSED_DIMS = ("bit_fields", "raw_classification", "classification_flags")


class DegradationKind(enum.Enum):
    """The kind of degradation applied to a business field"""

    #: values are stored as-is; points whose values do not fit are quarantined
    PRESERVE = "preserve"
    #: out-of-range values are saturated to the closest representable value
    CLAMP = "clamp"
    #: values are remapped through a caller-provided code mapping
    MAP = "map"
    #: the field is dropped (zeroed / omitted) for every point
    DROP = "drop"


class Degradation(NamedTuple):
    """One acceptable degradation scheme for a business field.

    ``cost`` is the caller-provided price paid for each affected point.
    ``mapping`` is only used by :attr:`DegradationKind.MAP` and maps original
    codes to the codes that must be stored instead.
    """

    kind: "DegradationKind"
    cost: float = 0.0
    mapping: Optional[Mapping[int, int]] = None

    @staticmethod
    def preserve() -> "Degradation":
        """Keep values as-is (cost 0); unrepresentable points are quarantined"""
        return Degradation(DegradationKind.PRESERVE, 0.0)

    @staticmethod
    def clamp(cost: float) -> "Degradation":
        """Saturate out-of-range values to the target range"""
        return Degradation(DegradationKind.CLAMP, float(cost))

    @staticmethod
    def map(cost: float, mapping: Mapping[int, int]) -> "Degradation":
        """Remap codes through ``mapping`` (original code -> stored code)"""
        if mapping is None:
            raise errors.LaspyException("MAP degradation requires a mapping")
        return Degradation(DegradationKind.MAP, float(cost), dict(mapping))

    @staticmethod
    def drop(cost: float) -> "Degradation":
        """Drop the field entirely"""
        return Degradation(DegradationKind.DROP, float(cost))


class MigrationPolicy(NamedTuple):
    """Caller-provided constraints and degradation costs for a batch migration.

    field_policies:
        maps a business field name (e.g. "classification", "gps_time", or an
        extra dimension name) to the degradation schemes the caller accepts
        for it, with their per-point costs.  :meth:`Degradation.preserve` is
        always implicitly allowed with a cost of 0.  Fields without an entry
        only allow PRESERVE.
    max_quarantined_points:
        optional batch-wide budget of points that may be isolated because
        they cannot be migrated.  ``None`` means no limit.
    quarantine_cost:
        optional loss charged per quarantined point, so that isolation can be
        traded off against degradations.
    """

    field_policies: Mapping[str, Tuple[Degradation, ...]] = {}
    max_quarantined_points: Optional[int] = None
    quarantine_cost: float = 0.0

    def degradations_for(self, field: str) -> Tuple[Degradation, ...]:
        allowed = tuple(self.field_policies.get(field, ()))
        for degradation in allowed:
            if not isinstance(degradation, Degradation):
                raise errors.LaspyException(
                    f"Invalid degradation {degradation!r} for field '{field}'"
                )
            if degradation.kind == DegradationKind.MAP and degradation.mapping is None:
                raise errors.LaspyException(
                    f"MAP degradation for field '{field}' requires a mapping"
                )
        return (Degradation.preserve(),) + allowed


class FieldDecision(NamedTuple):
    """The degradation chosen for one business field, valid for the whole batch"""

    field: str
    degradation: Degradation
    #: points whose stored value differs from the original
    affected_points: int
    #: points isolated because of this field
    quarantined_points: int
    #: degradation.cost * affected_points + quarantine cost
    loss: float
    #: sorted unique original codes that are altered (bounded code fields only)
    original_codes: Tuple[int, ...]
    #: stored code corresponding to each entry of ``original_codes``
    stored_codes: Tuple[int, ...]


class FieldReport(NamedTuple):
    """Actual loss and read-back proof for one field of one migrated file"""

    field: str
    degradation: DegradationKind
    cost_per_point: float
    affected_points: int
    quarantined_points: int
    loss: float
    original_codes: Tuple[int, ...]
    stored_codes: Tuple[int, ...]
    #: True if the stored values were read back exactly as written
    readback_verified: bool


class FileMigrationResult(NamedTuple):
    """Result of the migration of one input file"""

    source_index: int
    #: migrated points in the target point format
    migrated: LasData
    #: isolated points, kept in the original point format (None if empty)
    quarantined: Optional[LasData]
    #: indices of the isolated points in the original file
    quarantined_indices: np.ndarray
    field_reports: Tuple[FieldReport, ...]
    #: fields whose values were read back exactly as written
    verified_fields: Tuple[str, ...]
    readback_verified: bool
    quarantined_readback_verified: bool


class BatchMigrationResult(NamedTuple):
    """Result of the migration of a whole batch"""

    plan: "MigrationPlan"
    files: Tuple[FileMigrationResult, ...]
    total_loss: float
    total_quarantined: int
    #: True when every migrated and quarantined file read back exactly
    readback_verified: bool


class _Option(NamedTuple):
    """Internal: evaluation of one degradation for one field"""

    loss: float
    affected: int
    quarantined: int
    original_codes: Tuple[int, ...]
    stored_codes: Tuple[int, ...]


_KIND_ORDER = {
    DegradationKind.PRESERVE: 0,
    DegradationKind.CLAMP: 1,
    DegradationKind.MAP: 2,
    DegradationKind.DROP: 3,
}


def _is_new_family(point_format_id: int) -> bool:
    return point_format_id >= 6


def _range_limit(point_format_id: int, field: str) -> int:
    if field == "classification":
        return 255 if _is_new_family(point_format_id) else 31
    if field in ("return_number", "number_of_returns"):
        return 15 if _is_new_family(point_format_id) else 7
    raise errors.LaspyException(f"'{field}' is not a code range field")


def _needs_range_check(
    source_format_id: int, field: str, target_format_id: int
) -> bool:
    if field == "scan_angle":
        return _is_new_family(source_format_id) and not _is_new_family(target_format_id)
    return _range_limit(source_format_id, field) > _range_limit(target_format_id, field)


def _range_values(las: LasData, field: str) -> np.ndarray:
    if field == "scan_angle":
        if _is_new_family(las.point_format.id):
            return np.asarray(las.points["scan_angle"]).astype(np.int64)
        return np.asarray(las.points["scan_angle_rank"]).astype(np.int64)
    return np.asarray(las.points[field]).astype(np.int64)


def _representable_mask(
    values: np.ndarray, field: str, target_format_id: int
) -> np.ndarray:
    if field == "scan_angle":
        quotient, remainder = np.divmod(values, _SCAN_ANGLE_EXACT_STEP)
        ranks = quotient * _SCAN_ANGLE_NUM
        return (
            (remainder == 0)
            & (ranks >= _SCAN_ANGLE_RANK_MIN)
            & (ranks <= _SCAN_ANGLE_RANK_MAX)
        )
    return values <= _range_limit(target_format_id, field)


def _scan_angle_to_rank(values: np.ndarray) -> np.ndarray:
    ranks = np.rint(values * _SCAN_ANGLE_NUM / _SCAN_ANGLE_DEN).astype(np.int64)
    return np.clip(ranks, _SCAN_ANGLE_RANK_MIN, _SCAN_ANGLE_RANK_MAX)


def _rank_to_scan_angle(ranks: np.ndarray) -> np.ndarray:
    return np.rint(ranks * _SCAN_ANGLE_DEN / _SCAN_ANGLE_NUM).astype(np.int64)


def _code_tables(
    field: str, values: np.ndarray, changed_mask: np.ndarray, stored: np.ndarray
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    if field not in _CODE_REPORT_FIELDS:
        return (), ()
    original = np.unique(values[changed_mask])
    if original.size == 0:
        return (), ()
    stored_codes = []
    for code in original:
        stored_codes.append(int(stored[values == code][0]))
    return tuple(int(c) for c in original), tuple(stored_codes)


def _eval_range_file(
    las: LasData,
    field: str,
    target_format_id: int,
    degradation: Degradation,
    quarantine_cost: float,
) -> Optional[_Option]:
    """Evaluate one degradation for one range field of one file.

    Returns None when the degradation is not applicable / infeasible.
    """
    if not _needs_range_check(las.point_format.id, field, target_format_id):
        return _Option(0.0, 0, 0, (), ())

    values = _range_values(las, field)
    representable = _representable_mask(values, field, target_format_id)

    if degradation.kind == DegradationKind.PRESERVE:
        quarantined = int(np.count_nonzero(~representable))
        return _Option(quarantine_cost * quarantined, 0, quarantined, (), ())

    if degradation.kind == DegradationKind.CLAMP:
        changed = ~representable
        affected = int(np.count_nonzero(changed))
        if field == "scan_angle":
            stored = _scan_angle_to_rank(values)
        else:
            stored = np.minimum(values, _range_limit(target_format_id, field))
        original_codes, stored_codes = _code_tables(field, values, changed, stored)
        return _Option(
            degradation.cost * affected, affected, 0, original_codes, stored_codes
        )

    if degradation.kind == DegradationKind.MAP:
        mapping = degradation.mapping or {}
        if field == "scan_angle":
            valid_targets = all(
                _SCAN_ANGLE_RANK_MIN <= target <= _SCAN_ANGLE_RANK_MAX
                for target in mapping.values()
            )
        else:
            limit = _range_limit(target_format_id, field)
            valid_targets = all(0 <= target <= limit for target in mapping.values())
        if not valid_targets:
            return None
        keys = np.array(list(mapping.keys()), dtype=np.int64)
        mapped = np.isin(values, keys) if keys.size else np.zeros(len(values), bool)
        unmapped_out_of_range = ~representable & ~mapped
        quarantined = int(np.count_nonzero(unmapped_out_of_range))
        stored = values.copy()
        for code, target in mapping.items():
            stored[values == code] = target
        changed = stored != values
        affected = int(np.count_nonzero(changed))
        original_codes, stored_codes = _code_tables(field, values, changed, stored)
        return _Option(
            degradation.cost * affected + quarantine_cost * quarantined,
            affected,
            quarantined,
            original_codes,
            stored_codes,
        )

    if degradation.kind == DegradationKind.DROP:
        changed = values != 0
        affected = int(np.count_nonzero(changed))
        stored = np.zeros_like(values)
        original_codes, stored_codes = _code_tables(field, values, changed, stored)
        return _Option(
            degradation.cost * affected, affected, 0, original_codes, stored_codes
        )

    raise errors.LaspyException(f"Unknown degradation kind {degradation.kind}")


def _field_present_in_target(
    field: str, target_format_id: int, extra_dim_names: Tuple[str, ...]
) -> bool:
    if field in _RANGE_FIELDS or field in _SHARED_SUB_FIELDS:
        return True
    if field in _NEW_FAMILY_FLAGS:
        return _is_new_family(target_format_id)
    if field in _PRESENCE_FIELD_DIMS:
        dim_name = _PRESENCE_FIELD_DIMS[field][0]
        return dim_name in dims.POINT_FORMAT_DIMENSIONS[target_format_id]
    # extra dimension: preservable in any point format unless the name
    # collides with a standard dimension of the target format
    if field in dims.POINT_FORMAT_DIMENSIONS[target_format_id]:
        return False
    return field in extra_dim_names


def _extra_dim_infos(las: LasData) -> Dict[str, dims.DimensionInfo]:
    return {info.name: info for info in las.point_format.extra_dimensions}


def _extra_dim_consistent(lases: Sequence[LasData], name: str) -> bool:
    """True if every file that has the extra dimension `name` defines it identically"""
    reference = None
    for las in lases:
        info = _extra_dim_infos(las).get(name)
        if info is None:
            continue
        if reference is None:
            reference = info
            continue
        if (
            info.kind != reference.kind
            or info.num_bits != reference.num_bits
            or info.num_elements != reference.num_elements
        ):
            return False
        for attr in ("scales", "offsets"):
            mine, theirs = getattr(info, attr), getattr(reference, attr)
            if (mine is None) != (theirs is None):
                return False
            if mine is not None and not np.array_equal(mine, theirs):
                return False
    return True


def _non_default_count(las: LasData, field: str) -> int:
    """Number of points carrying a non-default (non-zero) value for a presence field"""
    if field in _NEW_FAMILY_FLAGS:
        if not _is_new_family(las.point_format.id):
            return 0
        values = np.asarray(las.points[field])
    elif field in _PRESENCE_FIELD_DIMS:
        dim_name = _PRESENCE_FIELD_DIMS[field][0]
        if dim_name not in las.point_format.standard_dimension_names:
            return 0
        values = las.points.array[dim_name]
    else:
        info = _extra_dim_infos(las).get(field)
        if info is None:
            return 0
        values = las.points.array[field]
    if values.ndim > 1:
        return int(np.count_nonzero(np.any(values != 0, axis=-1)))
    return int(np.count_nonzero(values != 0))


def _eval_presence_file(
    las: LasData,
    field: str,
    target_format_id: int,
    degradation: Degradation,
    extra_dim_names: Tuple[str, ...],
) -> Optional[_Option]:
    if degradation.kind == DegradationKind.PRESERVE:
        if not _field_present_in_target(field, target_format_id, extra_dim_names):
            return None
        return _Option(0.0, 0, 0, (), ())
    if degradation.kind == DegradationKind.DROP:
        affected = _non_default_count(las, field)
        return _Option(degradation.cost * affected, affected, 0, (), ())
    # CLAMP / MAP are not meaningful for presence-constrained fields
    return None


def _batch_business_fields(
    lases: Sequence[LasData], policy: MigrationPolicy
) -> List[str]:
    fields: List[str] = list(_RANGE_FIELDS)
    for field in list(_NEW_FAMILY_FLAGS) + list(_PRESENCE_FIELD_DIMS):
        if field in fields:
            continue
        # a presence-constrained field only constrains the format choice if it
        # carries information (dropping all-default values is lossless) or if
        # the caller explicitly cares about it
        carries_data = any(_non_default_count(las, field) > 0 for las in lases)
        if carries_data or field in policy.field_policies:
            fields.append(field)
    extra_names = sorted(
        {name for las in lases for name in las.point_format.extra_dimension_names}
    )
    fields.extend(extra_names)
    return fields


def _evaluate_field(
    lases: Sequence[LasData],
    field: str,
    target_format_id: int,
    policy: MigrationPolicy,
    consistent_extra_dims: Tuple[str, ...],
) -> List[Tuple[Degradation, _Option]]:
    """All feasible (degradation, option) pairs for a field, cheapest first"""
    options: List[Tuple[Degradation, _Option]] = []
    for degradation in policy.degradations_for(field):
        if field in _RANGE_FIELDS:
            per_file = [
                _eval_range_file(
                    las, field, target_format_id, degradation, policy.quarantine_cost
                )
                for las in lases
            ]
        else:
            if (
                degradation.kind == DegradationKind.PRESERVE
                and field not in _PRESENCE_FIELD_DIMS
                and field not in _NEW_FAMILY_FLAGS
                and field not in _RANGE_FIELDS
                and field not in _SHARED_SUB_FIELDS
                and not _extra_dim_consistent(lases, field)
            ):
                # same business field must have one consistent interpretation:
                # conflicting definitions cannot be preserved under a single format
                continue
            per_file = [
                _eval_presence_file(
                    las, field, target_format_id, degradation, consistent_extra_dims
                )
                for las in lases
            ]
        if any(option is None for option in per_file):
            continue
        total = _Option(
            loss=sum(option.loss for option in per_file),
            affected=sum(option.affected for option in per_file),
            quarantined=sum(option.quarantined for option in per_file),
            original_codes=tuple(
                sorted({code for option in per_file for code in option.original_codes})
            ),
            stored_codes=(),
        )
        # rebuild the stored codes aligned with the merged original codes
        stored: Dict[int, int] = {}
        for option in per_file:
            stored.update(zip(option.original_codes, option.stored_codes))
        total = total._replace(
            stored_codes=tuple(stored[code] for code in total.original_codes)
        )
        options.append((degradation, total))
    options.sort(
        key=lambda item: (
            item[1].loss,
            item[1].quarantined,
            item[1].affected,
            _KIND_ORDER[item[0].kind],
        )
    )
    return options


class MigrationPlan:
    """The minimal-loss conversion selected for a batch.

    Use :meth:`execute` to perform the migration and obtain the
    :class:`BatchMigrationResult` with the actual loss and the read-back
    proofs.
    """

    def __init__(
        self,
        lases: Sequence[LasData],
        target_version: str,
        target_point_format_id: int,
        decisions: Tuple[FieldDecision, ...],
        policy: MigrationPolicy,
    ) -> None:
        self.target_version = target_version
        self.target_point_format_id = target_point_format_id
        self.decisions = decisions
        self.total_loss = float(sum(d.loss for d in decisions))
        self.total_quarantined = int(sum(d.quarantined_points for d in decisions))
        self._lases = list(lases)
        self._policy = policy
        self._decision_by_field = {d.field: d for d in decisions}

    @property
    def target_point_format(self) -> PointFormat:
        """The target point format, including the preserved extra dimensions"""
        point_format = PointFormat(self.target_point_format_id)
        preserved = []
        for name in sorted(
            name
            for name, decision in self._decision_by_field.items()
            if decision.degradation.kind != DegradationKind.DROP
        ):
            for las in self._lases:
                info = _extra_dim_infos(las).get(name)
                if info is not None:
                    preserved.append(info)
                    break
        point_format.dimensions.extend(preserved)
        return point_format

    def execute(self) -> BatchMigrationResult:
        files = tuple(
            _execute_file(index, las, self) for index, las in enumerate(self._lases)
        )
        return BatchMigrationResult(
            plan=self,
            files=files,
            total_loss=float(
                sum(report.loss for f in files for report in f.field_reports)
            ),
            total_quarantined=int(sum(len(f.quarantined_indices) for f in files)),
            readback_verified=all(
                f.readback_verified and f.quarantined_readback_verified for f in files
            ),
        )

    def __repr__(self) -> str:
        return (
            f"<MigrationPlan(version={self.target_version}, "
            f"point_format={self.target_point_format_id}, "
            f"loss={self.total_loss}, quarantined={self.total_quarantined})>"
        )


def plan_batch_migration(
    lases: Sequence[LasData],
    target_version: str,
    policy: Optional[MigrationPolicy] = None,
) -> MigrationPlan:
    """Select the minimal-loss conversion of a batch of LAS files to a
    restricted target version.

    A single target point format (compatible with ``target_version``) and a
    single degradation per business field are selected for the whole batch,
    so that the same business field is migrated with the same interpretation
    in every file.  The selected plan satisfies the batch-wide constraints of
    the policy (quarantine budget) with the minimal total loss.

    >>> import laspy
    >>> las = laspy.read('tests/data/simple.las')
    >>> plan = plan_batch_migration([las], target_version='1.2')
    >>> plan.target_point_format_id
    3
    >>> plan.total_loss
    0.0
    >>> result = plan.execute()
    >>> result.readback_verified
    True

    Parameters
    ----------
    lases: sequence of laspy.LasData
        The files to migrate; they may use different point formats.
    target_version: str
        The restricted target version (e.g. "1.2"); only point formats
        compatible with this version are considered.
    policy: MigrationPolicy, optional
        Acceptable degradations, their costs, and the batch constraints.
        By default only PRESERVE is allowed (lossless-or-quarantine).

    Returns
    -------
    MigrationPlan

    Raises
    ------
    MigrationError
        If no point format of the target version yields a feasible plan.
    """
    lases = list(lases)
    if not lases:
        raise errors.LaspyException("Cannot plan a migration of an empty batch")
    if policy is None:
        policy = MigrationPolicy()

    try:
        version = Version.from_str(str(target_version))
    except Exception as exc:
        raise errors.LaspyException(
            f"Invalid target version {target_version!r}"
        ) from exc
    version_str = str(version)
    if version_str not in dims.VERSION_TO_POINT_FMT:
        raise errors.LaspyException(f"Unsupported target version {target_version!r}")

    fields = _batch_business_fields(lases, policy)
    extra_names = tuple(
        sorted({n for las in lases for n in las.point_format.extra_dimension_names})
    )

    best: Optional[MigrationPlan] = None
    for candidate_id in dims.VERSION_TO_POINT_FMT[version_str]:
        decisions = _plan_for_format(lases, fields, candidate_id, policy, extra_names)
        if decisions is None:
            continue
        plan = MigrationPlan(
            lases, version_str, candidate_id, tuple(decisions), policy
        )
        if best is None or _plan_sort_key(plan) < _plan_sort_key(best):
            best = plan

    if best is None:
        raise MigrationError(
            f"No feasible migration plan to version {version_str} for this batch "
            "under the given policy"
        )
    return best


def _plan_sort_key(plan: MigrationPlan) -> Tuple[float, int, int]:
    return (plan.total_loss, plan.total_quarantined, plan.target_point_format_id)


def _plan_for_format(
    lases: Sequence[LasData],
    fields: Sequence[str],
    candidate_id: int,
    policy: MigrationPolicy,
    extra_names: Tuple[str, ...],
) -> Optional[List[FieldDecision]]:
    chosen: Dict[str, Tuple[Degradation, _Option]] = {}
    all_options: Dict[str, List[Tuple[Degradation, _Option]]] = {}
    for field in fields:
        options = _evaluate_field(lases, field, candidate_id, policy, extra_names)
        if not options:
            return None
        all_options[field] = options
        chosen[field] = options[0]

    budget = policy.max_quarantined_points
    if budget is not None:
        total_quarantined = sum(option.quarantined for _, option in chosen.values())
        # greedily swap degradations to get back under the quarantine budget
        while total_quarantined > budget:
            best_swap = None
            for field, options in all_options.items():
                current_degradation, current_option = chosen[field]
                for degradation, option in options:
                    reduction = current_option.quarantined - option.quarantined
                    if reduction <= 0:
                        continue
                    extra_loss = option.loss - current_option.loss
                    ratio = extra_loss / reduction
                    if best_swap is None or (ratio, -reduction) < best_swap[0]:
                        best_swap = ((ratio, -reduction), field, degradation, option)
            if best_swap is None:
                return None
            _, field, degradation, option = best_swap
            total_quarantined -= chosen[field][1].quarantined - option.quarantined
            chosen[field] = (degradation, option)

    return [
        FieldDecision(
            field=field,
            degradation=degradation,
            affected_points=option.affected,
            quarantined_points=option.quarantined,
            loss=float(option.loss),
            original_codes=option.original_codes,
            stored_codes=option.stored_codes,
        )
        for field, (degradation, option) in chosen.items()
    ]


def _scan_angle_stored_ranks(
    values: np.ndarray, decision: Optional[FieldDecision]
) -> np.ndarray:
    """Ranks to store in a format < 6 for scan_angle values kept after degradation"""
    if decision is not None:
        kind = decision.degradation.kind
    else:
        kind = DegradationKind.PRESERVE
    if kind == DegradationKind.DROP:
        return np.zeros_like(values)
    if kind == DegradationKind.CLAMP:
        return _scan_angle_to_rank(values)
    # PRESERVE / MAP: kept values are exactly representable, convert exactly
    ranks = (values // _SCAN_ANGLE_EXACT_STEP) * _SCAN_ANGLE_NUM
    if kind == DegradationKind.MAP:
        for code, target in (decision.degradation.mapping or {}).items():
            ranks[values == code] = target
    return ranks


def _apply_range_degradation(
    values: np.ndarray,
    field: str,
    decision: Optional[FieldDecision],
    target_format_id: int,
) -> np.ndarray:
    """Compute the values to store for a range field of the kept points"""
    if decision is None:
        return values
    kind = decision.degradation.kind
    if kind == DegradationKind.PRESERVE:
        return values
    if kind == DegradationKind.DROP:
        return np.zeros_like(values)
    if kind == DegradationKind.CLAMP:
        if field == "scan_angle":
            return _scan_angle_to_rank(values)
        return np.minimum(values, _range_limit(target_format_id, field))
    if kind == DegradationKind.MAP:
        stored = values.copy()
        for code, target in (decision.degradation.mapping or {}).items():
            stored[values == code] = target
        return stored
    raise errors.LaspyException(f"Unknown degradation kind {kind}")


def _execute_file(index: int, las: LasData, plan: MigrationPlan) -> FileMigrationResult:
    source_format_id = las.point_format.id
    target_format_id = plan.target_point_format_id
    decisions = plan._decision_by_field
    point_count = len(las.points)

    # 1. isolate the points that cannot be migrated under the chosen degradations
    quarantine_mask = np.zeros(point_count, dtype=bool)
    for field in _RANGE_FIELDS:
        decision = decisions.get(field)
        if decision is None or decision.degradation.kind not in (
            DegradationKind.PRESERVE,
            DegradationKind.MAP,
        ):
            continue
        if not _needs_range_check(source_format_id, field, target_format_id):
            continue
        values = _range_values(las, field)
        unrepresentable = ~_representable_mask(values, field, target_format_id)
        if decision.degradation.kind == DegradationKind.MAP:
            keys = list((decision.degradation.mapping or {}).keys())
            mapped = np.isin(values, keys) if keys else np.zeros(point_count, bool)
            unrepresentable &= ~mapped
        quarantine_mask |= unrepresentable

    quarantined_indices = np.nonzero(quarantine_mask)[0]
    keep_mask = ~quarantine_mask
    source_points = las.points[keep_mask]

    # 2. build the migrated record in the target format
    target_format = plan.target_point_format
    new_points = PackedPointRecord.zeros(len(source_points), target_format)

    source_standard = set(las.point_format.standard_dimension_names)
    dropped_fields = {
        field
        for field, decision in decisions.items()
        if decision.degradation.kind == DegradationKind.DROP
    }

    for dim_name in _VERBATIM_DIMS:
        if dim_name in source_standard:
            new_points.array[dim_name] = source_points.array[dim_name]

    for field, dim_names in _PRESENCE_FIELD_DIMS.items():
        if field in dropped_fields:
            continue
        for dim_name in dim_names:
            if (
                dim_name in source_standard
                and dim_name in target_format.standard_dimension_names
            ):
                new_points.array[dim_name] = source_points.array[dim_name]

    for flag in _SHARED_SUB_FIELDS:
        new_points[flag] = np.asarray(source_points[flag])

    if _is_new_family(target_format_id):
        for flag in _NEW_FAMILY_FLAGS:
            if flag in dropped_fields:
                continue
            if _is_new_family(source_format_id):
                new_points[flag] = np.asarray(source_points[flag])

    for field in ("classification", "return_number", "number_of_returns"):
        values = np.asarray(source_points[field]).astype(np.int64)
        stored = _apply_range_degradation(
            values, field, decisions.get(field), target_format_id
        )
        new_points[field] = stored.astype(np.uint8)

    source_new = _is_new_family(source_format_id)
    target_new = _is_new_family(target_format_id)
    scan_angle_decision = decisions.get("scan_angle")
    if "scan_angle" in dropped_fields:
        pass
    elif source_new == target_new:
        dim_name = "scan_angle" if source_new else "scan_angle_rank"
        new_points.array[dim_name] = source_points.array[dim_name]
    elif source_new:
        values = np.asarray(source_points["scan_angle"]).astype(np.int64)
        ranks = _scan_angle_stored_ranks(values, scan_angle_decision)
        new_points.array["scan_angle_rank"] = ranks.astype(np.int8)
    else:
        ranks = np.asarray(source_points["scan_angle_rank"]).astype(np.int64)
        new_points.array["scan_angle"] = _rank_to_scan_angle(ranks).astype(np.int16)

    source_extra = _extra_dim_infos(las)
    for info in target_format.extra_dimensions:
        if info.name in source_extra:
            new_points.array[info.name] = source_points.array[info.name]

    if target_format.has_waveform_packet and source_points._waveform_state is not None:
        new_points._set_waveform_state(source_points._waveform_state)

    # 3. wrap into LasData objects with synchronized headers
    version = Version.from_str(plan.target_version)
    new_header = copy.deepcopy(las.header)
    new_header.set_version_and_point_format(version, copy.deepcopy(target_format))
    migrated = LasData(header=new_header, points=new_points)

    if las.evlrs is not None:
        evlrs = VLRList(las.evlrs.copy())
    else:
        evlrs = None
    if (version.major, version.minor) < (1, 4) and evlrs:
        logger.warning(
            "The source contained %d EVLRs, they will be lost as version %s "
            "does not support them",
            len(evlrs),
            plan.target_version,
        )
    else:
        migrated.evlrs = evlrs
    migrated.update_header()

    quarantined = None
    quarantined_verified = True
    if quarantine_mask.any():
        quarantined_header = copy.deepcopy(las.header)
        quarantined = LasData(
            header=quarantined_header, points=las.points[quarantine_mask]
        )
        if las.evlrs is not None:
            quarantined.evlrs = VLRList(las.evlrs.copy())
        quarantined.update_header()
        quarantined_verified = _verify_readback(
            quarantined,
            quarantined.point_format.id,
            _check_fields(quarantined.point_format.id, las),
        )[0]

    # 4. provable read-back correspondence of the migrated points
    check_fields = _check_fields(target_format_id, las, plan)
    readback_ok, verified_fields, field_checks = _verify_readback(
        migrated, target_format_id, check_fields
    )

    # 5. per-field actual loss report
    reports = []
    for decision in plan.decisions:
        stats = _file_field_stats(las, decision, target_format_id, plan._policy)
        if stats.affected == 0 and stats.quarantined == 0:
            continue
        reports.append(
            FieldReport(
                field=decision.field,
                degradation=decision.degradation.kind,
                cost_per_point=decision.degradation.cost,
                affected_points=stats.affected,
                quarantined_points=stats.quarantined,
                loss=float(stats.loss),
                original_codes=stats.original_codes,
                stored_codes=stats.stored_codes,
                readback_verified=field_checks.get(decision.field, readback_ok),
            )
        )

    return FileMigrationResult(
        source_index=index,
        migrated=migrated,
        quarantined=quarantined,
        quarantined_indices=quarantined_indices,
        field_reports=tuple(reports),
        verified_fields=verified_fields,
        readback_verified=readback_ok,
        quarantined_readback_verified=quarantined_verified,
    )


def _file_field_stats(
    las: LasData,
    decision: FieldDecision,
    target_format_id: int,
    policy: MigrationPolicy,
) -> _Option:
    field = decision.field
    if field in _RANGE_FIELDS:
        option = _eval_range_file(
            las, field, target_format_id, decision.degradation, policy.quarantine_cost
        )
    else:
        option = _eval_presence_file(
            las, field, target_format_id, decision.degradation, ()
        )
    if option is None:
        return _Option(0.0, 0, 0, (), ())
    return option


def _check_fields(
    target_format_id: int, las: LasData, plan: Optional[MigrationPlan] = None
) -> Tuple[str, ...]:
    fields: List[str] = list(_VERBATIM_DIMS)
    fields.extend(_SHARED_SUB_FIELDS)
    fields.extend(
        ("classification", "return_number", "number_of_returns", "scan_angle")
    )
    if _is_new_family(target_format_id):
        fields.extend(_NEW_FAMILY_FLAGS)
    for field, dim_names in _PRESENCE_FIELD_DIMS.items():
        if field in fields:
            continue
        if dim_names[0] in dims.POINT_FORMAT_DIMENSIONS[target_format_id]:
            fields.append(field)
    if plan is not None:
        for info in plan.target_point_format.extra_dimensions:
            fields.append(info.name)
    else:
        fields.extend(las.point_format.extra_dimension_names)
    return tuple(fields)


def _field_values_for_check(points: PackedPointRecord, field: str, format_id: int):
    if field == "scan_angle":
        dim_name = "scan_angle" if _is_new_family(format_id) else "scan_angle_rank"
        return np.asarray(points.array[dim_name])
    if field == "waveform":
        return np.asarray(points.array["wavepacket_index"])
    sub_fields = dims.get_sub_fields_dict(format_id)
    if field in sub_fields:
        return np.asarray(points[field])
    return np.asarray(points.array[field])


def _verify_readback(
    las: LasData, format_id: int, check_fields: Tuple[str, ...]
) -> Tuple[bool, Tuple[str, ...], Dict[str, bool]]:
    """Write `las` to memory, read it back and compare field by field"""
    buffer = io.BytesIO()
    las.write(buffer)
    buffer.seek(0)
    reread = read_las(buffer)

    field_checks: Dict[str, bool] = {}
    for field in check_fields:
        try:
            expected = _field_values_for_check(las.points, field, format_id)
            actual = _field_values_for_check(reread.points, field, format_id)
            field_checks[field] = bool(np.array_equal(expected, actual))
        except (KeyError, ValueError):
            field_checks[field] = False
    verified = tuple(field for field in check_fields if field_checks[field])
    return all(field_checks.values()), verified, field_checks
