"""Auditable repair of LAS return-number / number-of-returns fields.

LAS stores ``return_number`` and ``number_of_returns`` in small bit fields
(3 bits for point formats 0-5, 4 bits for point formats 6-10).  A valid point
cloud must satisfy, for every point, ``1 <= return_number <= number_of_returns``
and both values must fit the fields of the point format that stores them.

Points that belong to the same emitted pulse form a stricter group:

* all points of the pulse carry the same ``number_of_returns``;
* their ``return_number`` values are distinct.

Pulse membership is always supplied explicitly by the caller.  Coordinates are
never inspected, so nearby points are never silently treated as one pulse.

This module solves the following minimum-cost problem:

* every point chooses a ``(return_number, number_of_returns)`` pair from its
  candidate list (or from all pairs that fit the available widths);
* every pulse gets one ``number_of_returns`` value and distinct return numbers;
* optional per-flight-line control totals are matched exactly;
* the total cost is minimal.

When several revisions share the minimum cost, only the field values that occur
in *every* optimal revision are applied.  The other values are returned as
review items together with their possible values.  The original values and all
affected statistics are kept in the report.

The report also keeps one concrete optimal revision (``optimal_*`` fields).
That revision is guaranteed to satisfy every constraint and has the same cost
as the consensus revision; it is a review aid, not the automatically applied
result.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter
from copy import deepcopy
from itertools import combinations
from dataclasses import dataclass
from typing import (
    Any,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from . import errors
from .lasdata import LasData
from .point import PointFormat

__all__ = [
    "ReturnCandidate",
    "ReturnIssue",
    "ReturnRepairPlan",
    "ReturnRepairReport",
    "ReturnStatistics",
    "plan_return_repair",
    "validate_returns",
]

# Large enough to be 'infinite' while still being a float.
_INF = float("inf")

# A safety valve for the number of distinct (n, used returns) options of one
# pulse.  Normal LAS pulses contain a handful of returns; if a caller supplies
# a huge pulse with unrestricted candidates, a clear error is better than
# hanging.
_MAX_PULSE_OPTIONS = 50_000


class ReturnRepairError(errors.LaspyException):
    """Raised when a return-number repair problem is invalid or infeasible."""


@dataclass(frozen=True)
class ReturnCandidate:
    """One allowed ``(return_number, number_of_returns)`` pair for a point.

    Parameters
    ----------
    return_number:
        The candidate return number (1-based).
    number_of_returns:
        The candidate total number of returns for the pulse.
    cost:
        Optional non-negative cost of choosing this candidate.  When omitted,
        the cost is the number of fields that differ from the original point
        (0, 1 or 2).
    """

    return_number: int
    number_of_returns: int
    cost: Optional[float] = None


@dataclass(frozen=True)
class ReturnIssue:
    """A single consistency problem found in a point or a pulse."""

    kind: str
    message: str
    point_index: Optional[int] = None
    pulse_id: Any = None

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "message": self.message,
            "point_index": self.point_index,
            "pulse_id": self.pulse_id,
        }


@dataclass(frozen=True)
class _PulseOption:
    """One pulse-wide choice of ``number_of_returns`` and used return numbers.

    ``values`` is one concrete minimum-cost assignment used for reporting and
    reconstruction.  ``point_values`` records, for each point of the pulse,
    every return number that point can take in *some* assignment with the
    same cost and the same used-return set.
    """

    values: Tuple[Tuple[int, int], ...]
    cost: float
    counts: Tuple[int, ...]
    point_values: Tuple[Tuple[int, ...], ...]
    number_of_returns: int


@dataclass(frozen=True)
class _PointFieldChange:
    point_index: int
    pulse_id: Any
    flightline_id: Any
    field: str
    original: int
    revised: int

    def to_dict(self) -> dict:
        return {
            "point_index": self.point_index,
            "pulse_id": self.pulse_id,
            "flightline_id": self.flightline_id,
            "field": self.field,
            "original": self.original,
            "revised": self.revised,
        }


@dataclass(frozen=True)
class _ReviewItem:
    point_index: int
    pulse_id: Any
    flightline_id: Any
    field: str
    original: int
    possible_values: Tuple[int, ...]
    reason: str

    def to_dict(self) -> dict:
        return {
            "point_index": self.point_index,
            "pulse_id": self.pulse_id,
            "flightline_id": self.flightline_id,
            "field": self.field,
            "original": self.original,
            "possible_values": list(self.possible_values),
            "reason": self.reason,
        }


@dataclass
class ReturnStatistics:
    """Counts and validity information for one state of a point cloud."""

    point_count: int
    invalid_point_indices: Tuple[int, ...]
    return_number_counts: Tuple[int, ...]
    number_of_returns_counts: Tuple[int, ...]
    per_flightline_return_number_counts: Mapping[Any, Tuple[int, ...]]
    per_flightline_number_of_returns_counts: Mapping[Any, Tuple[int, ...]]
    header_number_of_points_by_return: Tuple[int, ...]

    @property
    def invalid_point_count(self) -> int:
        return len(self.invalid_point_indices)

    @property
    def is_valid(self) -> bool:
        return not self.invalid_point_indices

    def to_dict(self) -> dict:
        return {
            "point_count": self.point_count,
            "invalid_point_count": self.invalid_point_count,
            "invalid_point_indices": list(self.invalid_point_indices),
            "return_number_counts": list(self.return_number_counts),
            "number_of_returns_counts": list(self.number_of_returns_counts),
            "per_flightline_return_number_counts": {
                str(_json_scalar(k)): list(v)
                for k, v in self.per_flightline_return_number_counts.items()
            },
            "per_flightline_number_of_returns_counts": {
                str(_json_scalar(k)): list(v)
                for k, v in self.per_flightline_number_of_returns_counts.items()
            },
            "header_number_of_points_by_return": list(
                self.header_number_of_points_by_return
            ),
            "is_valid": self.is_valid,
        }


@dataclass
class ReturnRepairReport:
    """Complete audit trail of a :func:`plan_return_repair` call."""

    optimal_cost: float
    applied_changes: Tuple[_PointFieldChange, ...]
    review_items: Tuple[_ReviewItem, ...]
    original_return_numbers: np.ndarray
    original_number_of_returns: np.ndarray
    revised_return_numbers: np.ndarray
    revised_number_of_returns: np.ndarray
    optimal_return_numbers: np.ndarray
    optimal_number_of_returns: np.ndarray
    stats_before: ReturnStatistics
    stats_after: ReturnStatistics
    stats_optimal: ReturnStatistics
    issues_after: Tuple[ReturnIssue, ...]
    pulse_count: int
    flightline_count: int
    pulse_ids: Tuple[Any, ...]
    flightline_ids: Tuple[Any, ...]
    return_number_limits: Tuple[int, ...]
    number_of_returns_limits: Tuple[int, ...]

    @property
    def has_review(self) -> bool:
        return bool(self.review_items)

    @property
    def is_complete(self) -> bool:
        return not self.review_items

    @property
    def is_valid_after(self) -> bool:
        return not self.issues_after

    def _points(self) -> Iterable[dict]:
        for index in range(len(self.original_return_numbers)):
            yield {
                "point_index": index,
                "pulse_id": _json_scalar(self.pulse_ids[index]),
                "flightline_id": _json_scalar(self.flightline_ids[index]),
                "return_number_max": int(self.return_number_limits[index]),
                "number_of_returns_max": int(
                    self.number_of_returns_limits[index]
                ),
                "original_return_number": int(
                    self.original_return_numbers[index]
                ),
                "original_number_of_returns": int(
                    self.original_number_of_returns[index]
                ),
                "revised_return_number": int(self.revised_return_numbers[index]),
                "revised_number_of_returns": int(
                    self.revised_number_of_returns[index]
                ),
                "optimal_return_number": int(self.optimal_return_numbers[index]),
                "optimal_number_of_returns": int(
                    self.optimal_number_of_returns[index]
                ),
            }

    def to_dict(self) -> dict:
        return {
            "optimal_cost": self.optimal_cost,
            "has_review": self.has_review,
            "is_complete": self.is_complete,
            "is_valid_after": self.is_valid_after,
            "consensus_revision_is_valid": self.stats_after.is_valid,
            "optimal_revision_is_valid": self.stats_optimal.is_valid,
            "pulse_count": self.pulse_count,
            "flightline_count": self.flightline_count,
            "applied_changes": [change.to_dict() for change in self.applied_changes],
            "review_items": [item.to_dict() for item in self.review_items],
            "issues_after": [issue.to_dict() for issue in self.issues_after],
            "statistics": {
                "before": self.stats_before.to_dict(),
                "after": self.stats_after.to_dict(),
                "optimal_template": self.stats_optimal.to_dict(),
            },
            "points": list(self._points()),
        }

    def to_json(self, path: Optional[Union[str, os.PathLike]] = None) -> str:
        """Serializes the report, optionally writing it to *path*."""
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        if path is not None:
            with open(path, "w", encoding="utf-8") as stream:
                stream.write(payload)
        return payload

    def to_csv(self, path: Union[str, os.PathLike]) -> None:
        """Writes a per-point CSV containing original and revised codes."""
        fieldnames = [
            "point_index",
            "pulse_id",
            "flightline_id",
            "return_number_max",
            "number_of_returns_max",
            "original_return_number",
            "original_number_of_returns",
            "revised_return_number",
            "revised_number_of_returns",
            "optimal_return_number",
            "optimal_number_of_returns",
        ]
        with open(path, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for row in self._points():
                writer.writerow(row)


@dataclass
class ReturnRepairPlan:
    """A minimum-cost repair and the subset that can be applied automatically.

    ``revised`` is a copy of the source with every field that all optimal
    revisions agree on already changed.  ``source`` is never modified; call
    :meth:`apply_in_place` if the consensus fields should also be written back.
    """

    source: LasData
    revised: LasData
    report: ReturnRepairReport

    @property
    def optimal_cost(self) -> float:
        return self.report.optimal_cost

    @property
    def has_review(self) -> bool:
        return self.report.has_review

    def to_dict(self) -> dict:
        return self.report.to_dict()

    def to_json(self, path: Optional[Union[str, os.PathLike]] = None) -> str:
        return self.report.to_json(path)

    def to_csv(self, path: Union[str, os.PathLike]) -> None:
        self.report.to_csv(path)

    def apply_in_place(self) -> ReturnRepairReport:
        """Writes the consensus fields back into ``source`` and updates its header.

        Review items are not touched.  The original codes remain available in
        the report.
        """
        if self.report.applied_changes:
            self.source.points["return_number"] = np.array(
                self.report.revised_return_numbers,
                dtype=self.source.points.array["bit_fields"].dtype,
            )
            self.source.points["number_of_returns"] = np.array(
                self.report.revised_number_of_returns,
                dtype=self.source.points.array["bit_fields"].dtype,
            )
            self.source.update_header()
        return self.report


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _format_limit(point_format_id: int, dimension: str) -> int:
    try:
        info = PointFormat(int(point_format_id)).dimension_by_name(dimension)
    except (errors.PointFormatNotSupported, ValueError) as exc:
        raise ReturnRepairError(
            f"Point format {point_format_id!r} does not define {dimension!r}"
        ) from exc
    return int(info.max)


def _as_int_array(values: Any, name: str, expected_length: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or len(array) != expected_length:
        raise ReturnRepairError(
            f"{name} must be a one-dimensional array of length {expected_length}"
        )
    if not np.issubdtype(array.dtype, np.integer):
        if array.dtype == object:
            try:
                array = array.astype(np.int64)
            except (TypeError, ValueError) as exc:
                raise ReturnRepairError(f"{name} must contain integers") from exc
        else:
            raise ReturnRepairError(f"{name} must contain integers")
    return array.astype(np.int64, copy=False)


def _as_id_sequence(values: Any, name: str, expected_length: int) -> list:
    if isinstance(values, np.ndarray) and values.ndim != 1:
        raise ReturnRepairError(
            f"{name} must be one-dimensional of length {expected_length}"
        )
    result = list(values)
    if len(result) != expected_length:
        raise ReturnRepairError(
            f"{name} must have length {expected_length}, got {len(result)}"
        )
    for value in result:
        if isinstance(value, (list, dict, set, np.ndarray)):
            raise ReturnRepairError(f"{name} must contain scalar identifiers")
        try:
            hash(value)
        except TypeError as exc:
            raise ReturnRepairError(
                f"{name} must contain hashable identifiers"
            ) from exc
    return result


def _as_limit_array(
    values: Optional[Union[int, Sequence[int]]],
    name: str,
    point_count: int,
    default: np.ndarray,
) -> np.ndarray:
    """Converts a bit count (3, 4, ...) into the largest code it can store."""
    if values is None:
        return default.astype(np.int64, copy=False)
    array = np.asarray(values)
    if array.ndim == 0:
        array = np.full(point_count, int(array), dtype=np.int64)
    else:
        array = _as_int_array(array, name, point_count)
    if np.any(array < 1) or np.any(array > 32):
        raise ReturnRepairError(f"{name} must contain widths between 1 and 32")
    return (np.left_shift(np.int64(1), array) - 1).astype(np.int64)


def _normalize_counts(
    controls: Optional[Mapping[Any, Union[Sequence[int], Mapping[int, int]]]],
    max_code: int,
    name: str,
) -> Optional[dict]:
    if controls is None:
        return None

    if not isinstance(controls, Mapping):
        raise ReturnRepairError(f"{name} must be a mapping of flight line -> counts")

    normalized: dict[Any, tuple[int, ...]] = {}
    for flightline, spec in controls.items():
        hash(flightline)
        if isinstance(spec, Mapping):
            values = [0] * max_code
            for raw_index, raw_count in spec.items():
                index = int(raw_index)
                if index < 1 or index > max_code:
                    raise ReturnRepairError(
                        f"{name}[{flightline!r}] has index {index}, "
                        f"expected 1..{max_code}"
                    )
                values[index - 1] = int(raw_count)
        else:
            values = list(int(v) for v in spec)
            if len(values) > max_code:
                raise ReturnRepairError(
                    f"{name}[{flightline!r}] has {len(values)} entries, "
                    f"at most {max_code} are allowed"
                )
            values.extend([0] * (max_code - len(values)))

        if any(value < 0 for value in values):
            raise ReturnRepairError(
                f"{name}[{flightline!r}] must contain non-negative counts"
            )
        normalized[flightline] = tuple(values)

    return normalized


def _parse_candidate(raw: Any, point_index: int) -> ReturnCandidate:
    if isinstance(raw, ReturnCandidate):
        candidate = raw
    elif isinstance(raw, Mapping):
        try:
            candidate = ReturnCandidate(
                return_number=int(raw["return_number"]),
                number_of_returns=int(raw["number_of_returns"]),
                cost=raw.get("cost"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ReturnRepairError(
                f"Invalid candidate for point {point_index}: {raw!r}"
            ) from exc
    else:
        try:
            values = tuple(raw)
        except TypeError as exc:
            raise ReturnRepairError(
                f"Invalid candidate for point {point_index}: {raw!r}"
            ) from exc

        if len(values) == 2:
            return_number, number_of_returns = values
            cost = None
        elif len(values) == 3:
            return_number, number_of_returns, cost = values
        else:
            raise ReturnRepairError(
                f"Invalid candidate for point {point_index}: {raw!r}; "
                "expected (return_number, number_of_returns[, cost])"
            )
        try:
            candidate = ReturnCandidate(
                return_number=int(return_number),
                number_of_returns=int(number_of_returns),
                cost=cost,
            )
        except (TypeError, ValueError) as exc:
            raise ReturnRepairError(
                f"Invalid candidate for point {point_index}: {raw!r}"
            ) from exc

    if candidate.cost is not None:
        cost = float(candidate.cost)
        if not math.isfinite(cost) or cost < 0:
            raise ReturnRepairError(
                f"Candidate cost for point {point_index} must be finite and >= 0"
            )
        candidate = ReturnCandidate(
            candidate.return_number, candidate.number_of_returns, cost
        )
    return candidate


def _default_cost(
    original_return_number: int,
    original_number_of_returns: int,
    candidate_return_number: int,
    candidate_number_of_returns: int,
) -> float:
    return float(
        int(candidate_return_number != original_return_number)
        + int(candidate_number_of_returns != original_number_of_returns)
    )


def _candidate_map(
    candidates: Optional[
        Union[
            Mapping[int, Iterable[Any]],
            Sequence[Optional[Iterable[Any]]],
        ]
    ],
    point_count: int,
) -> Optional[dict[int, Any]]:
    if candidates is None:
        return None

    if isinstance(candidates, Mapping):
        result = {}
        for raw_index, raw_candidates in candidates.items():
            index = int(raw_index)
            if index < 0 or index >= point_count:
                raise ReturnRepairError(
                    f"Candidate index {index} is outside the point range"
                )
            result[index] = raw_candidates
        return result

    if len(candidates) != point_count:
        raise ReturnRepairError(
            "A candidate sequence must have the same length as the point count"
        )
    return {
        index: raw
        for index, raw in enumerate(candidates)
        if raw is not None
    }


def _validate_candidate(
    candidate: ReturnCandidate,
    point_index: int,
    return_limit: int,
    count_limit: int,
) -> None:
    if candidate.return_number < 1 or candidate.return_number > return_limit:
        raise ReturnRepairError(
            f"Candidate {candidate!r} for point {point_index} does not fit "
            f"the return_number field (maximum {return_limit})"
        )
    if (
        candidate.number_of_returns < 1
        or candidate.number_of_returns > count_limit
    ):
        raise ReturnRepairError(
            f"Candidate {candidate!r} for point {point_index} does not fit "
            f"the number_of_returns field (maximum {count_limit})"
        )
    if candidate.return_number > candidate.number_of_returns:
        raise ReturnRepairError(
            f"Candidate {candidate!r} for point {point_index} is itself invalid: "
            "return_number must not exceed number_of_returns"
        )


def _all_candidates(
    return_limit: int,
    count_limit: int,
) -> Iterable[ReturnCandidate]:
    for number_of_returns in range(1, count_limit + 1):
        for return_number in range(1, min(return_limit, number_of_returns) + 1):
            yield ReturnCandidate(return_number, number_of_returns)


def _enumerate_pulse_options(
    point_indices: Sequence[int],
    candidates_per_point: Sequence[Sequence[ReturnCandidate]],
    flightline: Any,
    controlled_keys: Sequence[Tuple[str, Any, int]],
    pulse_size_must_match: bool = True,
) -> list[_PulseOption]:
    """Enumerates the distinct ways one pulse can choose its return values.

    The pulse size is ``k``.  All its points share one ``number_of_returns``
    ``n``.  The return numbers must be distinct.  An option is therefore fully
    described by ``(n, used return numbers)``; individual point assignments
    only affect the cost and the per-point value sets.  This avoids
    enumerating every permutation of a large pulse.

    Each option's cost is the minimum over all assignments of those return
    numbers to the pulse's points.
    """
    k = len(point_indices)
    if k == 0:
        return []

    if pulse_size_must_match:
        n_values = [k]
    else:
        n_values = sorted(
            {
                candidate.number_of_returns
                for candidates in candidates_per_point
                for candidate in candidates
            }
        )

    options: dict[Tuple[int, Tuple[int, ...]], _PulseOption] = {}

    for number_of_returns in n_values:
        allowed: list[dict[int, float]] = []
        for candidates in candidates_per_point:
            by_return: dict[int, float] = {}
            for candidate in candidates:
                if candidate.number_of_returns != number_of_returns:
                    continue
                cost = candidate.cost
                if cost is None:
                    raise AssertionError("candidate cost should be resolved")
                previous = by_return.get(candidate.return_number)
                if previous is None or cost < previous:
                    by_return[candidate.return_number] = cost
            if not by_return:
                break
            allowed.append(by_return)
        else:
            union = sorted(set().union(*(set(values) for values in allowed)))
            if len(union) < k:
                continue
            option_count = math.comb(len(union), k)
            if option_count > _MAX_PULSE_OPTIONS:
                raise ReturnRepairError(
                    f"A pulse of {k} points has {option_count} possible "
                    "return-number sets. Supply explicit per-point candidates "
                    "or split the pulse."
                )
            for used_returns in combinations(union, k):
                option = _best_assignment(
                    allowed, used_returns, number_of_returns
                )
                if option is None:
                    continue
                key = (number_of_returns, used_returns)
                counter: Counter[Tuple[str, Any, int]] = Counter()
                for return_number in used_returns:
                    counter[("r", flightline, return_number)] += 1
                counter[("n", flightline, number_of_returns)] += k
                counts = tuple(
                    counter.get(key_name, 0) for key_name in controlled_keys
                )
                options[key] = _PulseOption(
                    option[0],
                    option[1],
                    counts,
                    option[2],
                    number_of_returns,
                )

    if not options:
        pulse_description = ", ".join(str(index) for index in point_indices[:8])
        if len(point_indices) > 8:
            pulse_description += ", ..."
        raise ReturnRepairError(
            "No valid assignment exists for pulse containing points "
            f"{pulse_description}. Check the supplied candidates and field limits."
        )
    return list(options.values())


def _best_assignment(
    allowed: Sequence[Mapping[int, float]],
    used_returns: Tuple[int, ...],
    number_of_returns: int,
) -> Optional[
    Tuple[Tuple[Tuple[int, int], ...], float, Tuple[Tuple[int, ...], ...]]
]:
    """Finds the minimum-cost injective assignment of *used_returns*.

    Returns ``(canonical_values, cost, point_values)`` or ``None`` when the
    return numbers cannot be assigned to the pulse's points.
    """
    k = len(allowed)
    full_mask = (1 << k) - 1
    return_index = {value: index for index, value in enumerate(used_returns)}

    forward = [[_INF] * (1 << k) for _ in range(k + 1)]
    forward[0][0] = 0.0
    for position in range(k):
        row = forward[position]
        next_row = forward[position + 1]
        for mask, accumulated in enumerate(row):
            if accumulated == _INF:
                continue
            for return_number, cost in allowed[position].items():
                index = return_index.get(return_number)
                if index is None:
                    continue
                bit = 1 << index
                if mask & bit:
                    continue
                new_mask = mask | bit
                value = accumulated + cost
                if value < next_row[new_mask]:
                    next_row[new_mask] = value

    optimum = forward[k][full_mask]
    if optimum == _INF:
        return None

    backward = [[_INF] * (1 << k) for _ in range(k + 1)]
    backward[k][full_mask] = 0.0
    for position in range(k - 1, -1, -1):
        row = backward[position]
        next_row = backward[position + 1]
        for mask in range(1 << k):
            if mask.bit_count() != position:
                continue
            best = _INF
            for return_number, cost in allowed[position].items():
                index = return_index.get(return_number)
                if index is None:
                    continue
                bit = 1 << index
                if mask & bit:
                    continue
                value = cost + next_row[mask | bit]
                if value < best:
                    best = value
            row[mask] = best

    point_values: list[Tuple[int, ...]] = []
    for position in range(k):
        values: list[int] = []
        masks = [
            mask
            for mask in range(1 << k)
            if mask.bit_count() == position and forward[position][mask] != _INF
        ]
        for return_number, cost in allowed[position].items():
            index = return_index.get(return_number)
            if index is None:
                continue
            bit = 1 << index
            for mask in masks:
                if mask & bit:
                    continue
                rest = backward[position + 1][mask | bit]
                if rest == _INF:
                    continue
                if _cost_equal(forward[position][mask] + cost + rest, optimum):
                    values.append(return_number)
                    break
        point_values.append(tuple(sorted(values)))

    canonical: list[Tuple[int, int]] = []
    mask = 0
    for position in range(k):
        chosen = None
        for return_number, cost in sorted(allowed[position].items()):
            index = return_index.get(return_number)
            if index is None:
                continue
            bit = 1 << index
            if mask & bit:
                continue
            new_mask = mask | bit
            if not _cost_equal(
                forward[position + 1][new_mask],
                forward[position][mask] + cost,
            ):
                continue
            if not _cost_equal(
                forward[position][mask] + cost + backward[position + 1][new_mask],
                optimum,
            ):
                continue
            chosen = return_number
            mask = new_mask
            break
        if chosen is None:
            raise ReturnRepairError(
                "Internal error: cannot reconstruct a pulse assignment"
            )
        canonical.append((chosen, number_of_returns))

    return tuple(canonical), optimum, tuple(point_values)


def _statistics(
    las: LasData,
    return_numbers: np.ndarray,
    number_of_returns_values: np.ndarray,
    flightlines: Sequence[Any],
    pulse_ids: Sequence[Any],
    return_limits: np.ndarray,
    count_limits: np.ndarray,
) -> ReturnStatistics:
    issues, invalid = _validate(
        return_numbers,
        number_of_returns_values,
        pulse_ids,
        flightlines,
        return_limits,
        count_limits,
    )

    max_width = max(
        1,
        int(return_limits.max(initial=1)),
        int(count_limits.max(initial=1)),
        int(return_numbers.max(initial=0)),
        int(number_of_returns_values.max(initial=0)),
    )
    bins = max(16, max_width + 1)

    return_number_counts = np.bincount(
        np.clip(return_numbers, 0, bins - 1), minlength=bins
    ).tolist()
    count_counts = np.bincount(
        np.clip(number_of_returns_values, 0, bins - 1), minlength=bins
    ).tolist()

    per_flightline_r: dict[int, Tuple[int, ...]] = {}
    per_flightline_n: dict[int, Tuple[int, ...]] = {}
    for flightline in _sorted_ids(set(flightlines)):
        mask = np.asarray(
            [value == flightline for value in flightlines], dtype=bool
        )
        per_flightline_r[_json_scalar(flightline)] = tuple(
            int(v)
            for v in np.bincount(
                np.clip(return_numbers[mask], 0, bins - 1), minlength=bins
            ).tolist()
        )
        per_flightline_n[_json_scalar(flightline)] = tuple(
            int(v)
            for v in np.bincount(
                np.clip(number_of_returns_values[mask], 0, bins - 1),
                minlength=bins,
            ).tolist()
        )

    return ReturnStatistics(
        point_count=len(return_numbers),
        invalid_point_indices=tuple(sorted(invalid)),
        return_number_counts=tuple(int(v) for v in return_number_counts),
        number_of_returns_counts=tuple(int(v) for v in count_counts),
        per_flightline_return_number_counts=per_flightline_r,
        per_flightline_number_of_returns_counts=per_flightline_n,
        header_number_of_points_by_return=tuple(
            int(v) for v in las.header.number_of_points_by_return
        ),
    )


def _validate(
    return_numbers: np.ndarray,
    number_of_returns_values: np.ndarray,
    pulse_ids: Sequence[Any],
    flightlines: Sequence[Any],
    return_limits: np.ndarray,
    count_limits: np.ndarray,
) -> Tuple[list[ReturnIssue], set[int]]:
    issues: list[ReturnIssue] = []
    invalid: set[int] = set()

    for index in range(len(return_numbers)):
        return_number = int(return_numbers[index])
        count_value = int(number_of_returns_values[index])
        if return_number < 1 or return_number > return_limits[index]:
            issues.append(
                ReturnIssue(
                    "return_number_out_of_range",
                    f"return_number {return_number} does not fit "
                    f"the allowed maximum {return_limits[index]}",
                    index,
                )
            )
            invalid.add(index)
        if count_value < 1 or count_value > count_limits[index]:
            issues.append(
                ReturnIssue(
                    "number_of_returns_out_of_range",
                    f"number_of_returns {count_value} does not fit "
                    f"the allowed maximum {count_limits[index]}",
                    index,
                )
            )
            invalid.add(index)
        if return_number > count_value:
            issues.append(
                ReturnIssue(
                    "return_exceeds_total",
                    f"return_number {return_number} exceeds "
                    f"number_of_returns {count_value}",
                    index,
                )
            )
            invalid.add(index)

    pulse_members: dict[Any, list[int]] = {}
    for index, raw_pulse in enumerate(pulse_ids):
        pulse_members.setdefault(raw_pulse, []).append(index)

    for pulse, members in pulse_members.items():
        seen: dict[int, int] = {}
        pulse_counts = {int(number_of_returns_values[i]) for i in members}
        if len(pulse_counts) > 1:
            issues.append(
                ReturnIssue(
                    "pulse_inconsistent_number_of_returns",
                    f"pulse has several number_of_returns values: "
                    f"{sorted(pulse_counts)}",
                    None,
                    pulse,
                )
            )
            invalid.update(members)
        flightline_values = {flightlines[i] for i in members}
        if len(flightline_values) > 1:
            issues.append(
                ReturnIssue(
                    "pulse_mixed_flightlines",
                    f"pulse spans flight lines {_sorted_ids(flightline_values)}",
                    None,
                    pulse,
                )
            )
            invalid.update(members)
        for index in members:
            return_number = int(return_numbers[index])
            if return_number in seen:
                issues.append(
                    ReturnIssue(
                        "pulse_duplicate_return_number",
                        f"return_number {return_number} also used by point "
                        f"{seen[return_number]} in the same pulse",
                        index,
                        pulse,
                    )
                )
                invalid.add(index)
                invalid.add(seen[return_number])
            else:
                seen[return_number] = index

    return issues, invalid


def validate_returns(
    source: LasData,
    pulse_ids: Sequence[Any],
    *,
    flightline_ids: Optional[Sequence[Any]] = None,
    source_point_format_ids: Optional[Sequence[int]] = None,
    return_number_bits: Optional[Union[int, Sequence[int]]] = None,
    number_of_returns_bits: Optional[Union[int, Sequence[int]]] = None,
) -> list[ReturnIssue]:
    """Returns every return-number consistency issue without changing data.

    ``pulse_ids`` are mandatory: pulse membership is never inferred from
    coordinates.  Omit ``flightline_ids`` to use ``point_source_id`` (or a
    single line when the point format does not contain it).
    """
    point_count = len(source.points)
    pulse_array = _as_id_sequence(pulse_ids, "pulse_ids", point_count)
    flightline_array = _resolve_flightlines(
        source, flightline_ids, point_count
    )
    return_limits, count_limits = _resolve_limits(
        source,
        source_point_format_ids,
        return_number_bits,
        number_of_returns_bits,
        point_count,
    )
    issues, _ = _validate(
        np.array(source.points["return_number"]),
        np.array(source.points["number_of_returns"]),
        pulse_array,
        flightline_array,
        return_limits,
        count_limits,
    )
    return issues


def _resolve_flightlines(
    source: LasData,
    flightline_ids: Optional[Sequence[Any]],
    point_count: int,
) -> list:
    if flightline_ids is not None:
        return _as_id_sequence(flightline_ids, "flightline_ids", point_count)
    try:
        return [
            _json_scalar(value)
            for value in np.asarray(source.points["point_source_id"]).tolist()
        ]
    except (ValueError, KeyError):
        return [0] * point_count


def _sorted_ids(values: Iterable[Any]) -> list:
    def key(value: Any) -> Tuple[str, str]:
        return (type(value).__name__, str(value))

    return sorted(values, key=key)


def _resolve_limits(
    source: LasData,
    source_point_format_ids: Optional[Sequence[int]],
    return_number_bits: Optional[Union[int, Sequence[int]]],
    number_of_returns_bits: Optional[Union[int, Sequence[int]]],
    point_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    target_format_id = source.point_format.id
    target_r = _format_limit(target_format_id, "return_number")
    target_n = _format_limit(target_format_id, "number_of_returns")
    default_r = np.full(point_count, target_r, dtype=np.int64)
    default_n = np.full(point_count, target_n, dtype=np.int64)

    if source_point_format_ids is not None:
        formats = _as_int_array(
            source_point_format_ids, "source_point_format_ids", point_count
        )
        default_r = np.array(
            [_format_limit(int(value), "return_number") for value in formats],
            dtype=np.int64,
        )
        default_n = np.array(
            [
                _format_limit(int(value), "number_of_returns")
                for value in formats
            ],
            dtype=np.int64,
        )
        default_r = np.minimum(default_r, target_r)
        default_n = np.minimum(default_n, target_n)

    return_limit = _as_limit_array(
        return_number_bits, "return_number_bits", point_count, default_r
    )
    count_limit = _as_limit_array(
        number_of_returns_bits,
        "number_of_returns_bits",
        point_count,
        default_n,
    )

    # The fields still live inside the current point format.  An explicit
    # per-point width may only narrow what the file can store.
    return_limit = np.minimum(return_limit, default_r)
    count_limit = np.minimum(count_limit, default_n)
    return return_limit, count_limit


def _resolve_candidates(
    candidates: Optional[
        Union[
            Mapping[int, Iterable[Any]],
            Sequence[Optional[Iterable[Any]]],
        ]
    ],
    original_r: np.ndarray,
    original_n: np.ndarray,
    return_limits: np.ndarray,
    count_limits: np.ndarray,
) -> list[list[ReturnCandidate]]:
    point_count = len(original_r)
    supplied = _candidate_map(candidates, point_count)
    result: list[list[ReturnCandidate]] = []

    for point_index in range(point_count):
        if supplied is not None and point_index not in supplied:
            # Points not listed are fixed to their current pair.
            candidate = ReturnCandidate(
                int(original_r[point_index]),
                int(original_n[point_index]),
                0.0,
            )
            _validate_candidate(
                candidate,
                point_index,
                int(return_limits[point_index]),
                int(count_limits[point_index]),
            )
            result.append([candidate])
            continue

        if supplied is None:
            raw_candidates: Iterable[Any] = _all_candidates(
                int(return_limits[point_index]),
                int(count_limits[point_index]),
            )
        else:
            raw_candidates = supplied[point_index]

        candidates_for_point: dict[Tuple[int, int], float] = {}
        for raw in raw_candidates:
            candidate = _parse_candidate(raw, point_index)
            _validate_candidate(
                candidate,
                point_index,
                int(return_limits[point_index]),
                int(count_limits[point_index]),
            )
            if candidate.cost is None:
                cost = _default_cost(
                    int(original_r[point_index]),
                    int(original_n[point_index]),
                    candidate.return_number,
                    candidate.number_of_returns,
                )
            else:
                cost = candidate.cost
            key = (candidate.return_number, candidate.number_of_returns)
            previous = candidates_for_point.get(key)
            if previous is None or cost < previous:
                candidates_for_point[key] = cost

        if not candidates_for_point:
            raise ReturnRepairError(
                f"Point {point_index} has no allowed candidate"
            )
        result.append(
            [
                ReturnCandidate(return_number, number_of_returns, cost)
                for (return_number, number_of_returns), cost in sorted(
                    candidates_for_point.items()
                )
            ]
        )

    return result


def _controlled_keys(
    flightline: Any,
    return_counts: Optional[dict],
    count_counts: Optional[dict],
    width: int,
) -> list[Tuple[str, Any, int]]:
    keys: list[Tuple[str, Any, int]] = []
    if return_counts is not None and flightline in return_counts:
        for index in range(width):
            keys.append(("r", flightline, index + 1))
    if count_counts is not None and flightline in count_counts:
        for index in range(width):
            keys.append(("n", flightline, index + 1))
    return keys


def _target_vector(
    keys: Sequence[Tuple[str, Any, int]],
    return_counts: Optional[dict],
    count_counts: Optional[dict],
) -> Tuple[int, ...]:
    target = []
    for kind, flightline, value in keys:
        source = return_counts if kind == "r" else count_counts
        assert source is not None
        target.append(int(source[flightline][value - 1]))
    return tuple(target)


def _prepare_controls(
    flightlines: Sequence[Any],
    return_number_counts: Optional[
        Mapping[Any, Union[Sequence[int], Mapping[int, int]]]
    ],
    number_of_returns_counts: Optional[
        Mapping[Any, Union[Sequence[int], Mapping[int, int]]]
    ],
    max_code: int,
) -> Tuple[Optional[dict], Optional[dict]]:
    r_controls = _normalize_counts(
        return_number_counts, max_code, "return_number_counts"
    )
    n_controls = _normalize_counts(
        number_of_returns_counts, max_code, "number_of_returns_counts"
    )

    point_count = len(flightlines)
    for controls, name in (
        (r_controls, "return_number_counts"),
        (n_controls, "number_of_returns_counts"),
    ):
        if controls is None:
            continue
        for flightline, values in controls.items():
            actual = sum(1 for value in flightlines if value == flightline)
            if sum(values) != actual:
                raise ReturnRepairError(
                    f"{name}[{flightline!r}] sums to {sum(values)} but the "
                    f"flight line has {actual} points"
                )
        for flightline in set(flightlines):
            if flightline not in controls:
                raise ReturnRepairError(
                    f"{name} has no control total for flight line "
                    f"{flightline!r}"
                )
    if r_controls is not None and n_controls is not None:
        for flightline in r_controls.keys() & n_controls.keys():
            if sum(r_controls[flightline]) != sum(n_controls[flightline]):
                raise ReturnRepairError(
                    f"return_number_counts and number_of_returns_counts "
                    f"disagree for flight line {flightline!r}"
                )
    return r_controls, n_controls


def _solve(
    options_per_pulse: Sequence[Sequence[_PulseOption]],
    target: Tuple[int, ...],
) -> tuple[float, list[set[int]], list[set[int]], list[Tuple[Tuple[int, int], ...]]]:
    """Solves one independent group (normally one flight line).

    The solver uses prefix/suffix dynamic programming over the control-count
    state.  Without controls the state is empty, so pulses are independent.
    With controls the state must match the requested totals exactly.

    The state is a tuple of per-code totals, one block per controlled flight
    line and kind.  Points of different flight lines are never mixed.

    Returns the optimal cost and, for every point, the set of return numbers
    and number-of-returns values that occur in at least one optimal revision.
    A field is consensus exactly when its set has a single value.
    """
    pulse_count = len(options_per_pulse)
    zero = tuple(0 for _ in target)
    dimension = len(target)

    if pulse_count == 0:
        if target != zero:
            raise ReturnRepairError(
                "No revision satisfies all pulse, flight-line and field constraints"
            )
        return 0.0, [], [], []

    ordered_options = [
        sorted(options, key=lambda option: (option.cost, option.values))
        for options in options_per_pulse
    ]

    def add(
        left: Tuple[int, ...], right: Tuple[int, ...]
    ) -> Optional[Tuple[int, ...]]:
        result = tuple(left[i] + right[i] for i in range(dimension))
        if any(result[i] > target[i] for i in range(dimension)):
            return None
        return result

    def subtract(
        left: Tuple[int, ...], right: Tuple[int, ...]
    ) -> Optional[Tuple[int, ...]]:
        if any(left[i] < right[i] for i in range(dimension)):
            return None
        return tuple(left[i] - right[i] for i in range(dimension))

    max_states = 1_000_000

    backward: list[dict[Tuple[int, ...], float]] = [
        {} for _ in range(pulse_count + 1)
    ]
    backward[pulse_count] = {zero: 0.0}
    for pulse_index in range(pulse_count - 1, -1, -1):
        table: dict[Tuple[int, ...], float] = {}
        for option in ordered_options[pulse_index]:
            for state, cost in backward[pulse_index + 1].items():
                new_state = add(state, option.counts)
                if new_state is None:
                    continue
                value = cost + option.cost
                previous = table.get(new_state)
                if previous is None or value < previous - 1e-12:
                    table[new_state] = value
        backward[pulse_index] = table
        if len(table) > max_states:
            raise ReturnRepairError(
                "The flight-line control totals create too many search "
                "states. Split the batch into smaller flight-line groups."
            )

    if target not in backward[0]:
        raise ReturnRepairError(
            "No revision satisfies all pulse, flight-line and field constraints"
        )
    optimal_cost = backward[0][target]

    forward: list[dict[Tuple[int, ...], float]] = [
        {} for _ in range(pulse_count + 1)
    ]
    forward[0] = {target: 0.0}
    for pulse_index in range(pulse_count):
        table: dict[Tuple[int, ...], float] = {}
        for state, cost in forward[pulse_index].items():
            for option in ordered_options[pulse_index]:
                new_state = subtract(state, option.counts)
                if new_state is None or new_state not in backward[pulse_index + 1]:
                    continue
                value = cost + option.cost
                previous = table.get(new_state)
                if previous is None or value < previous - 1e-12:
                    table[new_state] = value
        forward[pulse_index + 1] = table

    possible_r: list[set[int]] = []
    possible_n: list[set[int]] = []
    for options in ordered_options:
        possible_r.extend(set() for _ in options[0].values)
        possible_n.extend(set() for _ in options[0].values)

    offsets = _pulse_offsets(ordered_options)
    for pulse_index in range(pulse_count):
        suffix = backward[pulse_index + 1]
        next_forward = forward[pulse_index + 1]
        start = offsets[pulse_index]
        for state, base_cost in forward[pulse_index].items():
            suffix_cost = backward[pulse_index].get(state)
            if suffix_cost is None:
                continue
            if not _cost_equal(base_cost + suffix_cost, optimal_cost):
                continue
            for option in ordered_options[pulse_index]:
                new_state = subtract(state, option.counts)
                if new_state is None:
                    continue
                rest = suffix.get(new_state)
                if rest is None:
                    continue
                if not _cost_equal(base_cost + option.cost + rest, optimal_cost):
                    continue
                if not _cost_equal(
                    next_forward.get(new_state, _INF), base_cost + option.cost
                ):
                    continue
                for local_index, values in enumerate(option.point_values):
                    possible_r[start + local_index].update(values)
                    possible_n[start + local_index].add(
                        option.number_of_returns
                    )

    for index, values in enumerate(possible_r):
        if not values or not possible_n[index]:
            raise ReturnRepairError(
                f"Internal error: no optimal value found for local point {index}"
            )

    # Reconstruct one concrete optimal revision.  Its cost equals
    # ``optimal_cost`` and it satisfies every control total.
    chosen: list[Tuple[Tuple[int, int], ...]] = []
    state = target
    for pulse_index in range(pulse_count):
        base = forward[pulse_index].get(state)
        if base is None:
            raise ReturnRepairError(
                "Internal error: cannot reconstruct an optimal revision"
            )
        selected = None
        for option in ordered_options[pulse_index]:
            new_state = subtract(state, option.counts)
            if new_state is None:
                continue
            rest = backward[pulse_index + 1].get(new_state)
            if rest is None:
                continue
            if _cost_equal(base + option.cost + rest, optimal_cost):
                selected = option
                state = new_state
                break
        if selected is None:
            raise ReturnRepairError(
                "Internal error: cannot reconstruct an optimal revision"
            )
        chosen.append(selected.values)

    return optimal_cost, possible_r, possible_n, chosen


def _cost_equal(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-9)


def _pulse_offsets(
    options_per_pulse: Sequence[Sequence[_PulseOption]]
) -> list[int]:
    offsets = []
    total = 0
    for options in options_per_pulse:
        offsets.append(total)
        if options:
            total += len(options[0].values)
    return offsets


def plan_return_repair(
    source: LasData,
    pulse_ids: Sequence[Any],
    *,
    flightline_ids: Optional[Sequence[Any]] = None,
    candidates: Optional[
        Union[
            Mapping[int, Iterable[Any]],
            Sequence[Optional[Iterable[Any]]],
        ]
    ] = None,
    return_number_counts: Optional[
        Mapping[Any, Union[Sequence[int], Mapping[int, int]]]
    ] = None,
    number_of_returns_counts: Optional[
        Mapping[Any, Union[Sequence[int], Mapping[int, int]]]
    ] = None,
    source_point_format_ids: Optional[Sequence[int]] = None,
    return_number_bits: Optional[Union[int, Sequence[int]]] = None,
    number_of_returns_bits: Optional[Union[int, Sequence[int]]] = None,
    pulse_size_must_match: bool = False,
) -> ReturnRepairPlan:
    """Plans a minimum-cost repair of the two return fields.

    Parameters
    ----------
    source:
        The point cloud to inspect.  It is not modified.
    pulse_ids:
        Explicit pulse identity for every point.  Points with the same value
        are one pulse; coordinates are deliberately never used.  Pulse ids
        should therefore come from the acquisition system, not from spatial
        clustering.
    flightline_ids:
        Optional flight-line identity for every point.  Defaults to
        ``point_source_id``, or one line when that dimension is unavailable.
    candidates:
        ``{point_index: [(return_number, number_of_returns), ...]}``.  Values
        may be tuples, :class:`ReturnCandidate` instances, or mappings with
        ``return_number`` / ``number_of_returns`` / optional ``cost`` keys.
        Points omitted from a mapping are fixed to their original pair.  When
        ``candidates`` is ``None``, every point may use any pair that fits the
        field widths, with cost equal to the number of changed fields.
    return_number_counts, number_of_returns_counts:
        Optional exact per-flight-line totals.  A value is either a sequence
        (index 0 is return 1) or a ``{return_value: count}`` mapping.  For the
        number-of-returns totals each point contributes one count, so a pulse
        contributes its size to the selected total.
    source_point_format_ids:
        Optional per-point source format ids.  Use this when the batch mixes
        old (3-bit, formats 0-5) and new (4-bit, formats 6-10) records.  The
        effective width is the smaller of the source width and the width of
        the file's current point format.
    return_number_bits, number_of_returns_bits:
        Optional explicit per-point width limits.  They narrow (never widen)
        what the current point format can store.
    pulse_size_must_match:
        When true, every pulse's ``number_of_returns`` must equal the number of
        points carrying that pulse id.  The default is false because a pulse
        may legitimately have emitted more echoes than the batch contains.

    Returns
    -------
    ReturnRepairPlan
        ``plan.revised`` contains the consensus fields, ``plan.report`` has the
        original codes, applied changes, review items and before/after counts.
        Call ``plan.apply_in_place()`` to write the consensus fields back to
        ``source``.
    """
    if not isinstance(source, LasData):
        raise ReturnRepairError("source must be a LasData instance")

    point_count = len(source.points)
    pulse_array = _as_id_sequence(pulse_ids, "pulse_ids", point_count)
    flightline_array = _resolve_flightlines(source, flightline_ids, point_count)

    original_r = np.array(source.points["return_number"], dtype=np.int64)
    original_n = np.array(source.points["number_of_returns"], dtype=np.int64)

    return_limits, count_limits = _resolve_limits(
        source,
        source_point_format_ids,
        return_number_bits,
        number_of_returns_bits,
        point_count,
    )

    # Group in insertion order to keep reports deterministic.
    pulse_members: dict[Any, list[int]] = {}
    for index, raw_pulse in enumerate(pulse_array):
        pulse_members.setdefault(raw_pulse, []).append(index)
    points_per_pulse = list(pulse_members.values())
    pulse_order = list(pulse_members.keys())

    for pulse, members in zip(pulse_order, points_per_pulse):
        if len({flightline_array[index] for index in members}) > 1:
            raise ReturnRepairError(
                f"Pulse {pulse!r} contains points from several flight lines; "
                "a pulse must belong to one flight line"
            )

    candidates_per_point = _resolve_candidates(
        candidates,
        original_r,
        original_n,
        return_limits,
        count_limits,
    )

    max_code = max(
        1,
        int(return_limits.max(initial=1)),
        int(count_limits.max(initial=1)),
    )
    r_controls, n_controls = _prepare_controls(
        flightline_array,
        return_number_counts,
        number_of_returns_counts,
        max_code,
    )

    # Flight lines are independent: their control totals do not interact and
    # costs add up.  Solving each line separately keeps the DP state small
    # and still gives the exact global optimum.
    flightline_order = _sorted_ids(set(flightline_array))
    global_possible_r: list[set[int]] = [set() for _ in range(point_count)]
    global_possible_n: list[set[int]] = [set() for _ in range(point_count)]
    optimal_r = original_r.copy()
    optimal_n = original_n.copy()
    optimal_cost = 0.0

    for flightline in flightline_order:
        members_by_pulse: list[tuple[Any, list[int]]] = [
            (pulse, members)
            for pulse, members in zip(pulse_order, points_per_pulse)
            if flightline_array[members[0]] == flightline
        ]
        keys = _controlled_keys(flightline, r_controls, n_controls, max_code)
        target = _target_vector(keys, r_controls, n_controls)
        options_per_pulse: list[list[_PulseOption]] = []
        for pulse, members in members_by_pulse:
            pulse_candidates = [candidates_per_point[index] for index in members]
            options_per_pulse.append(
                _enumerate_pulse_options(
                    members,
                    pulse_candidates,
                    flightline,
                    keys,
                    pulse_size_must_match,
                )
            )

        group_cost, group_r, group_n, chosen = _solve(
            options_per_pulse, target
        )
        optimal_cost += group_cost

        local_index = 0
        for option_values, (pulse, members) in zip(chosen, members_by_pulse):
            for member, pair in zip(members, option_values):
                global_possible_r[member] = group_r[local_index]
                global_possible_n[member] = group_n[local_index]
                optimal_r[member] = pair[0]
                optimal_n[member] = pair[1]
                local_index += 1

    possible_r, possible_n = global_possible_r, global_possible_n

    applied_changes: list[_PointFieldChange] = []
    review_items: list[_ReviewItem] = []
    revised_r = original_r.copy()
    revised_n = original_n.copy()

    for point_index in range(point_count):
        pulse = pulse_array[point_index]
        flightline = flightline_array[point_index]

        values_r = possible_r[point_index]
        values_n = possible_n[point_index]
        if len(values_r) == 1:
            fixed_r = next(iter(values_r))
            if fixed_r != int(original_r[point_index]):
                applied_changes.append(
                    _PointFieldChange(
                        point_index,
                        pulse,
                        flightline,
                        "return_number",
                        int(original_r[point_index]),
                        int(fixed_r),
                    )
                )
                revised_r[point_index] = fixed_r
        else:
            review_items.append(
                _ReviewItem(
                    point_index,
                    pulse,
                    flightline,
                    "return_number",
                    int(original_r[point_index]),
                    tuple(sorted(values_r)),
                    "optimal revisions disagree",
                )
            )

        if len(values_n) == 1:
            fixed_n = next(iter(values_n))
            if fixed_n != int(original_n[point_index]):
                applied_changes.append(
                    _PointFieldChange(
                        point_index,
                        pulse,
                        flightline,
                        "number_of_returns",
                        int(original_n[point_index]),
                        int(fixed_n),
                    )
                )
                revised_n[point_index] = fixed_n
        else:
            review_items.append(
                _ReviewItem(
                    point_index,
                    pulse,
                    flightline,
                    "number_of_returns",
                    int(original_n[point_index]),
                    tuple(sorted(values_n)),
                    "optimal revisions disagree",
                )
            )

    revised = LasData(deepcopy(source.header), points=source.points.copy())
    revised.points["return_number"] = revised_r
    revised.points["number_of_returns"] = revised_n
    revised.update_header()

    optimal = LasData(deepcopy(source.header), points=source.points.copy())
    optimal.points["return_number"] = optimal_r
    optimal.points["number_of_returns"] = optimal_n
    optimal.update_header()

    after_issues, _ = _validate(
        revised_r,
        revised_n,
        pulse_array,
        flightline_array,
        return_limits,
        count_limits,
    )
    optimal_issues, _ = _validate(
        optimal_r,
        optimal_n,
        pulse_array,
        flightline_array,
        return_limits,
        count_limits,
    )
    if optimal_issues:
        raise ReturnRepairError(
            "Internal error: reconstructed optimal revision is not valid"
        )

    report = ReturnRepairReport(
        optimal_cost=optimal_cost,
        applied_changes=tuple(applied_changes),
        review_items=tuple(review_items),
        original_return_numbers=original_r.copy(),
        original_number_of_returns=original_n.copy(),
        revised_return_numbers=revised_r.copy(),
        revised_number_of_returns=revised_n.copy(),
        optimal_return_numbers=optimal_r.copy(),
        optimal_number_of_returns=optimal_n.copy(),
        stats_before=_statistics(
            source,
            original_r,
            original_n,
            flightline_array,
            pulse_array,
            return_limits,
            count_limits,
        ),
        stats_after=_statistics(
            revised,
            revised_r,
            revised_n,
            flightline_array,
            pulse_array,
            return_limits,
            count_limits,
        ),
        stats_optimal=_statistics(
            optimal,
            optimal_r,
            optimal_n,
            flightline_array,
            pulse_array,
            return_limits,
            count_limits,
        ),
        issues_after=tuple(after_issues),
        pulse_count=len(pulse_order),
        flightline_count=len(set(_json_scalar(value) for value in flightline_array)),
        pulse_ids=tuple(_json_scalar(value) for value in pulse_array),
        flightline_ids=tuple(_json_scalar(value) for value in flightline_array),
        return_number_limits=tuple(int(value) for value in return_limits),
        number_of_returns_limits=tuple(int(value) for value in count_limits),
    )
    return ReturnRepairPlan(source=source, revised=revised, report=report)
