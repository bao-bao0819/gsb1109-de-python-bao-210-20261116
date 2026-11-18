"""Deterministic, provenance-preserving classification cleaning.

The cleaning engine keeps the classification stored in the point record as the
*device original*.  Rules never overwrite it: they compute a separate cleaned
classification from an immutable copy plus an optional correction overlay.

Rules are evaluated against a snapshot of the working classification.  All
changes found in a round are applied together, which makes the result
independent of point iteration order.  When several rules propose different
classes for the same point, an explicit ``(priority, rule_id)`` ordering decides
the winner, so the order in which rules were declared is irrelevant as well.

Only points that may actually be affected by a rule are evaluated.  After the
first round, the candidate set is restricted to changed points and their
neighbourhood (expanded one radius at a time).  The same idea is used by
:class:`ClassificationSession` for incremental corrections: only the
neighbourhood closure of changed seeds is recomputed.

If the fixed point is not reached (cycle or iteration limit), the result is
marked as incomplete and it is never published to a ledger or attached to a
LAS file.
"""

from __future__ import annotations

import hashlib
import json
import math
import zlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from . import errors

__all__ = [
    "ClassificationChange",
    "ClassificationCleaner",
    "ClassificationError",
    "ClassificationHistory",
    "ClassificationLedger",
    "ClassificationNotConverged",
    "ClassificationRule",
    "ClassificationRuleSet",
    "ClassificationSession",
    "ClassificationVersion",
    "CleaningResult",
    "ExtentRule",
    "InvalidClassificationRule",
    "NeighborRule",
    "SpatialExtent",
    "SpatialRule",
    "clean_classification",
    "load_classification_history",
    "store_classification_history",
]


class ClassificationError(errors.LaspyException):
    """Base class for classification cleaning errors."""


class InvalidClassificationRule(ClassificationError):
    """Raised when a rule or rule set is not valid."""


class ClassificationNotConverged(ClassificationError):
    """Raised when an operation needs a complete classification result."""

    def __init__(self, result: "CleaningResult"):
        self.result = result
        if result.status == "cycle":
            detail = "a repeated state was detected"
        else:
            detail = f"the iteration limit ({result.max_iterations}) was reached"
        super().__init__(
            "Classification cleaning did not converge: "
            f"{detail}; no version was published"
        )


def _as_int_class(value: Any, name: str = "class") -> int:
    if isinstance(value, (bool, np.bool_)):
        raise InvalidClassificationRule(f"{name} must be an integer")
    try:
        if isinstance(value, (float, np.floating)) and not float(value).is_integer():
            raise ValueError
        class_value = int(value)
    except (TypeError, ValueError, OverflowError):
        raise InvalidClassificationRule(f"{name} must be an integer") from None
    if not 0 <= class_value <= 255:
        raise InvalidClassificationRule(f"{name} must be in the range 0..255")
    return class_value


def _normalise_classes(
    classes: Iterable[int] | None, name: str
) -> frozenset[int] | None:
    if classes is None:
        return None
    if isinstance(classes, (str, bytes)):
        raise InvalidClassificationRule(f"{name} must be an iterable of integers")
    normalised = frozenset(_as_int_class(value, name) for value in classes)
    if not normalised:
        raise InvalidClassificationRule(f"{name} cannot be empty")
    return normalised


@dataclass(frozen=True)
class SpatialExtent:
    """An explicit, closed, axis-aligned spatial extent.

    ``zmin`` and ``zmax`` may be omitted for 2D data.  Bounds are inclusive.
    """

    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float | None = None
    zmax: float | None = None

    def __post_init__(self) -> None:
        values = [self.xmin, self.xmax, self.ymin, self.ymax]
        if self.zmin is not None or self.zmax is not None:
            if self.zmin is None or self.zmax is None:
                raise InvalidClassificationRule(
                    "zmin and zmax must either both be set or both be omitted"
                )
            values.extend([self.zmin, self.zmax])
        if not all(math.isfinite(float(value)) for value in values):
            raise InvalidClassificationRule("extent bounds must be finite")
        if self.xmin > self.xmax or self.ymin > self.ymax:
            raise InvalidClassificationRule("extent minimum is greater than maximum")
        if self.zmin is not None and self.zmax is not None:
            if self.zmin > self.zmax:
                raise InvalidClassificationRule(
                    "extent minimum is greater than maximum"
                )

    @classmethod
    def from_bounds(cls, bounds: Any) -> "SpatialExtent":
        """Create an extent from a mapping or a ``(xmin, xmax, ymin, ymax)``
        style sequence.

        Mappings may use the keys ``xmin``/``xmax``/``ymin``/``ymax`` (and
        optionally ``zmin``/``zmax``), or ``x``/``y``/``z`` pairs.
        """
        if isinstance(bounds, SpatialExtent):
            return bounds
        if isinstance(bounds, Mapping):
            if "x" in bounds:
                (xmin, xmax) = bounds["x"]
                (ymin, ymax) = bounds["y"]
                z = bounds.get("z")
                if z is None:
                    return cls(xmin, xmax, ymin, ymax)
                return cls(xmin, xmax, ymin, ymax, z[0], z[1])
            return cls(
                bounds["xmin"],
                bounds["xmax"],
                bounds["ymin"],
                bounds["ymax"],
                bounds.get("zmin"),
                bounds.get("zmax"),
            )
        values = tuple(bounds)
        if len(values) == 4:
            return cls(*values)
        if len(values) == 6:
            return cls(*values)
        raise InvalidClassificationRule(
            "an extent must contain xmin, xmax, ymin, ymax "
            "and optionally zmin, zmax"
        )

    def contains(
        self,
        x: NDArray[np.floating],
        y: NDArray[np.floating],
        z: NDArray[np.floating] | None = None,
    ) -> NDArray[np.bool_]:
        mask = (
            (x >= self.xmin)
            & (x <= self.xmax)
            & (y >= self.ymin)
            & (y <= self.ymax)
        )
        if self.zmin is not None:
            if z is None:
                raise InvalidClassificationRule(
                    "this extent constrains z but no z coordinates were supplied"
                )
            mask &= (z >= self.zmin) & (z <= self.zmax)
        return np.asarray(mask, dtype=bool)

    def to_dict(self) -> dict[str, float | None]:
        return {
            "xmin": self.xmin,
            "xmax": self.xmax,
            "ymin": self.ymin,
            "ymax": self.ymax,
            "zmin": self.zmin,
            "zmax": self.zmax,
        }


class ClassificationRule:
    """Base class for classification rules.

    Subclasses implement :meth:`_candidates` and :meth:`_propose`.  User code
    can also subclass this class, but the built-in :class:`ExtentRule` and
    :class:`NeighborRule` cover explicit extents and neighbour references.
    """

    kind = "rule"

    def __init__(
        self,
        rule_id: str,
        target: int,
        *,
        priority: int = 0,
        extent: Any | None = None,
        match_classes: Iterable[int] | None = None,
        description: str = "",
    ) -> None:
        if not isinstance(rule_id, str) or not rule_id:
            raise InvalidClassificationRule("rule_id must be a non-empty string")
        if rule_id.startswith("__"):
            raise InvalidClassificationRule("rule_id cannot start with '__'")
        self.rule_id = rule_id
        self.target = _as_int_class(target, "target")
        self.priority = int(priority)
        self.extent = (
            None if extent is None else SpatialExtent.from_bounds(extent)
        )
        self.match_classes = _normalise_classes(match_classes, "match_classes")
        self.description = str(description)

    @property
    def neighbor_radius(self) -> float:
        """The largest radius this rule needs for propagation."""
        return 0.0

    def applies_to_extent(self, x, y, z) -> NDArray[np.bool_]:
        return self.extent.contains(x, y, z) if self.extent is not None else np.ones(
            len(x), dtype=bool
        )

    def matches_class(self, values: NDArray[np.uint8]) -> NDArray[np.bool_]:
        if self.match_classes is None:
            return np.ones(len(values), dtype=bool)
        return np.isin(values, np.fromiter(self.match_classes, dtype=np.uint8))

    def _candidates(self, context: "_RoundContext") -> NDArray[np.int64]:
        """Return initial candidate indices for a full run."""
        mask = self.applies_to_extent(context.x, context.y, context.z)
        mask &= self.matches_class(context.current)
        return np.flatnonzero(mask)

    def _propose(
        self, context: "_RoundContext", indices: NDArray[np.int64]
    ) -> dict[int, "_Proposal"]:
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "rule_id": self.rule_id,
            "target": self.target,
            "priority": self.priority,
            "extent": None if self.extent is None else self.extent.to_dict(),
            "match_classes": (
                None
                if self.match_classes is None
                else sorted(self.match_classes)
            ),
            "description": self.description,
        }


class ExtentRule(ClassificationRule):
    """Change points in an explicit spatial extent to ``target``.

    ``match_classes`` optionally limits the rule to points whose current class
    is one of the given values.
    """

    kind = "extent"

    def _propose(self, context, indices):
        if indices.size == 0:
            return {}
        mask = self.applies_to_extent(
            context.x[indices], context.y[indices], context.z[indices]
        )
        mask &= self.matches_class(context.current[indices])
        proposals: dict[int, _Proposal] = {}
        for index in indices[mask]:
            index = int(index)
            old_class = int(context.current[index])
            proposals[index] = _Proposal(
                rule=self,
                target=self.target,
                evidence={
                    "kind": self.kind,
                    "old_class": old_class,
                    "target": self.target,
                    "extent": self.extent.to_dict() if self.extent else None,
                },
            )
        return proposals


SpatialRule = ExtentRule


class NeighborRule(ClassificationRule):
    """Reclassify a point based on the classes of its spatial neighbours.

    A point is proposed for reclassification when at least ``min_count``
    neighbours and/or the requested ``min_fraction`` of neighbours belong to
    ``neighbor_classes``.  ``neighbor_classes=None`` means any neighbour.

    Parameters
    ----------
    radius:
        Neighbour search radius in coordinate units.
    neighbor_classes:
        Classes counted as supporting the rule.  ``None`` counts every
        neighbour.
    min_count:
        Minimum number of supporting neighbours.  Defaults to one.
    min_fraction:
        Optional minimum fraction of all neighbours (excluding the point
        itself) that must support the rule.
    neighbor_source:
        ``"current"`` (default) uses the working classification of the round,
        which is what makes corrections propagate.  ``"seed"`` uses the
        original plus manual correction overlay, and ``"original"`` uses only
        the device original.
    """

    kind = "neighbor"

    def __init__(
        self,
        rule_id: str,
        target: int,
        *,
        radius: float,
        neighbor_classes: Iterable[int] | None = None,
        min_count: int = 1,
        min_fraction: float | None = None,
        neighbor_source: str = "current",
        priority: int = 0,
        extent: Any | None = None,
        match_classes: Iterable[int] | None = None,
        description: str = "",
    ) -> None:
        super().__init__(
            rule_id,
            target,
            priority=priority,
            extent=extent,
            match_classes=match_classes,
            description=description,
        )
        radius = float(radius)
        if not math.isfinite(radius) or radius <= 0:
            raise InvalidClassificationRule("radius must be a positive finite number")
        if int(min_count) < 0:
            raise InvalidClassificationRule("min_count cannot be negative")
        if min_fraction is not None:
            min_fraction = float(min_fraction)
            if not math.isfinite(min_fraction) or not 0 <= min_fraction <= 1:
                raise InvalidClassificationRule(
                    "min_fraction must be between 0 and 1"
                )
        if int(min_count) == 0 and min_fraction is None:
            raise InvalidClassificationRule(
                "a neighbour rule needs min_count > 0 or a min_fraction"
            )
        if neighbor_source not in {"current", "seed", "original"}:
            raise InvalidClassificationRule(
                "neighbor_source must be 'current', 'seed' or 'original'"
            )
        self.radius = radius
        self.neighbor_classes = _normalise_classes(
            neighbor_classes, "neighbor_classes"
        )
        self.min_count = int(min_count)
        self.min_fraction = min_fraction
        self.neighbor_source = neighbor_source

    @property
    def neighbor_radius(self) -> float:
        return self.radius

    def _candidates(self, context):
        mask = self.applies_to_extent(context.x, context.y, context.z)
        mask &= self.matches_class(context.current)
        if self.min_count == 0 and (
            self.min_fraction is None or self.min_fraction <= 0
        ):
            # Degenerate configuration: the rule may fire without any
            # supporting neighbour, so every in-extent point is a candidate.
            return np.flatnonzero(mask)

        source = self._source_values(context)
        if self.neighbor_classes is None:
            supporting = np.ones(len(source), dtype=bool)
        else:
            supporting = np.isin(
                source, np.fromiter(self.neighbor_classes, dtype=np.uint8)
            )
        supporting_indices = np.flatnonzero(supporting)
        if supporting_indices.size == 0:
            return np.empty(0, dtype=np.int64)
        # A neighbour rule can only fire within one radius of a point that
        # currently supports it.  Starting from that neighbourhood avoids
        # scanning the whole cloud while remaining exact for the first round;
        # later rounds expand the dirty set by radius around changed points.
        near_support = np.zeros(len(source), dtype=bool)
        near_support[
            np.fromiter(
                context.index.query_many(supporting_indices, self.radius),
                dtype=np.int64,
            )
        ] = True
        mask &= near_support
        return np.flatnonzero(mask)

    def _source_values(self, context) -> NDArray[np.uint8]:
        if self.neighbor_source == "current":
            return context.current
        if self.neighbor_source == "seed":
            return context.seed
        return context.original

    def _propose(self, context, indices):
        if indices.size == 0:
            return {}
        mask = self.applies_to_extent(
            context.x[indices], context.y[indices], context.z[indices]
        )
        mask &= self.matches_class(context.current[indices])
        indices = indices[mask]
        if indices.size == 0:
            return {}

        source = self._source_values(context)
        wanted = (
            None
            if self.neighbor_classes is None
            else np.fromiter(self.neighbor_classes, dtype=np.uint8)
        )
        proposals: dict[int, _Proposal] = {}
        for index in indices:
            index = int(index)
            neighbour_indices = context.index.query(index, self.radius)
            neighbour_indices = neighbour_indices[neighbour_indices != index]
            total = int(neighbour_indices.size)
            if total == 0:
                matching = 0
                class_counts: dict[int, int] = {}
            else:
                neighbour_classes = source[neighbour_indices]
                if wanted is None:
                    matching = total
                    classes, counts = np.unique(
                        neighbour_classes, return_counts=True
                    )
                    class_counts = {
                        int(cls): int(count)
                        for cls, count in zip(classes, counts)
                    }
                else:
                    matching = int(np.count_nonzero(np.isin(neighbour_classes, wanted)))
                    classes, counts = np.unique(
                        neighbour_classes[np.isin(neighbour_classes, wanted)],
                        return_counts=True,
                    )
                    class_counts = {
                        int(cls): int(count)
                        for cls, count in zip(classes, counts)
                    }

            fraction = matching / total if total else 0.0
            if matching < self.min_count:
                continue
            if self.min_fraction is not None and fraction < self.min_fraction:
                continue

            old_class = int(context.current[index])
            proposals[index] = _Proposal(
                rule=self,
                target=self.target,
                evidence={
                    "kind": self.kind,
                    "old_class": old_class,
                    "target": self.target,
                    "radius": self.radius,
                    "neighbor_source": self.neighbor_source,
                    "neighbor_classes": (
                        None
                        if self.neighbor_classes is None
                        else sorted(self.neighbor_classes)
                    ),
                    "matching_neighbors": matching,
                    "total_neighbors": total,
                    "matching_fraction": fraction,
                    "neighbor_class_counts": class_counts,
                    "min_count": self.min_count,
                    "min_fraction": self.min_fraction,
                    "extent": self.extent.to_dict() if self.extent else None,
                },
            )
        return proposals

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update(
            {
                "radius": self.radius,
                "neighbor_classes": (
                    None
                    if self.neighbor_classes is None
                    else sorted(self.neighbor_classes)
                ),
                "min_count": self.min_count,
                "min_fraction": self.min_fraction,
                "neighbor_source": self.neighbor_source,
            }
        )
        return result


@dataclass(frozen=True)
class _Proposal:
    rule: ClassificationRule
    target: int
    evidence: dict[str, Any]


class ClassificationRuleSet:
    """An immutable, named version of a set of classification rules."""

    def __init__(
        self,
        version: str,
        rules: Sequence[ClassificationRule],
        *,
        description: str = "",
    ) -> None:
        if not isinstance(version, str) or not version:
            raise InvalidClassificationRule("version must be a non-empty string")
        self.version = version
        self.rules = tuple(rules)
        self.description = str(description)
        seen: set[str] = set()
        for rule in self.rules:
            if not isinstance(rule, ClassificationRule):
                raise InvalidClassificationRule(
                    "all rules must be ClassificationRule instances"
                )
            if rule.rule_id in seen:
                raise InvalidClassificationRule(
                    f"duplicate rule_id '{rule.rule_id}'"
                )
            seen.add(rule.rule_id)

    def __iter__(self):
        return iter(self.rules)

    def __len__(self) -> int:
        return len(self.rules)

    @property
    def max_neighbor_radius(self) -> float:
        return max((rule.neighbor_radius for rule in self.rules), default=0.0)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ClassificationRuleSet":
        rules: list[ClassificationRule] = []
        for raw_rule in data.get("rules", ()):
            rule_kind = raw_rule.get("kind")
            common = {
                "rule_id": raw_rule["rule_id"],
                "target": raw_rule["target"],
                "priority": raw_rule.get("priority", 0),
                "extent": raw_rule.get("extent"),
                "match_classes": raw_rule.get("match_classes"),
                "description": raw_rule.get("description", ""),
            }
            if rule_kind == "extent":
                rules.append(ExtentRule(**common))
            elif rule_kind == "neighbor":
                rules.append(
                    NeighborRule(
                        **common,
                        radius=raw_rule["radius"],
                        neighbor_classes=raw_rule.get("neighbor_classes"),
                        min_count=raw_rule.get("min_count", 1),
                        min_fraction=raw_rule.get("min_fraction"),
                        neighbor_source=raw_rule.get("neighbor_source", "current"),
                    )
                )
            else:
                raise InvalidClassificationRule(
                    f"unknown rule kind '{rule_kind}'"
                )
        return cls(
            data["version"],
            rules,
            description=data.get("description", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "description": self.description,
            "rules": [
                rule.to_dict()
                for rule in sorted(self.rules, key=lambda item: item.rule_id)
            ],
        }


@dataclass(frozen=True)
class ClassificationChange:
    """One traceable class change (a manual correction or a rule decision)."""

    point_index: int
    round: int
    old_class: int
    new_class: int
    rule_id: str
    rule_version: str
    reason: str
    evidence: Mapping[str, Any] = field(default_factory=dict)
    source: str = "rule"

    def to_dict(self) -> dict[str, Any]:
        return {
            "point_index": self.point_index,
            "round": self.round,
            "old_class": self.old_class,
            "new_class": self.new_class,
            "rule_id": self.rule_id,
            "rule_version": self.rule_version,
            "reason": self.reason,
            "evidence": dict(self.evidence),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ClassificationChange":
        return cls(
            point_index=data["point_index"],
            round=data["round"],
            old_class=data["old_class"],
            new_class=data["new_class"],
            rule_id=data["rule_id"],
            rule_version=data["rule_version"],
            reason=data["reason"],
            evidence=data.get("evidence", {}),
            source=data.get("source", "rule"),
        )


def _digest(array: NDArray[np.uint8]) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _indices_digest(indices: Iterable[int]) -> str:
    values = np.asarray(sorted(set(int(index) for index in indices)), dtype=np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _version_id(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ClassificationVersion:
    """Source and result metadata for one cleaning run.

    ``version_id`` identifies this *run* (it includes the parent, so two
    different update paths produce different lineage rows).
    ``published_id`` identifies the published *content* (rule version plus
    final classification).  Two paths that converge to the same result
    therefore share the same ``published_id`` even though their history rows
    remain distinct.
    """

    version_id: str
    parent_version_id: str | None
    rule_version: str
    status: str
    converged: bool
    rounds: int
    max_iterations: int
    original_digest: str
    seed_digest: str
    classification_digest: str
    corrections: Mapping[int, int]
    changes: tuple[ClassificationChange, ...]
    evaluated_count: int
    recomputed_indices_count: int
    rules: Mapping[str, Any]

    @property
    def published(self) -> bool:
        return self.converged

    @property
    def published_id(self) -> str | None:
        """Content id of the published result, or ``None`` if unpublished."""
        if not self.published:
            return None
        return _version_id(
            {
                "rule_version": self.rule_version,
                "rules": self.rules,
                "classification_digest": self.classification_digest,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "parent_version_id": self.parent_version_id,
            "published_id": self.published_id,
            "rule_version": self.rule_version,
            "status": self.status,
            "converged": self.converged,
            "rounds": self.rounds,
            "max_iterations": self.max_iterations,
            "original_digest": self.original_digest,
            "seed_digest": self.seed_digest,
            "classification_digest": self.classification_digest,
            "corrections": {
                str(index): value for index, value in sorted(self.corrections.items())
            },
            "changes": [change.to_dict() for change in self.changes],
            "evaluated_count": self.evaluated_count,
            "recomputed_indices_count": self.recomputed_indices_count,
            "rules": dict(self.rules),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ClassificationVersion":
        return cls(
            version_id=data["version_id"],
            parent_version_id=data.get("parent_version_id"),
            rule_version=data["rule_version"],
            status=data["status"],
            converged=bool(data["converged"]),
            rounds=int(data["rounds"]),
            max_iterations=int(data["max_iterations"]),
            original_digest=data["original_digest"],
            seed_digest=data["seed_digest"],
            classification_digest=data["classification_digest"],
            corrections={
                int(index): int(value)
                for index, value in data.get("corrections", {}).items()
            },
            changes=tuple(
                ClassificationChange.from_dict(change)
                for change in data.get("changes", ())
            ),
            evaluated_count=int(data.get("evaluated_count", 0)),
            recomputed_indices_count=int(data.get("recomputed_indices_count", 0)),
            rules=data.get("rules", {}),
        )


class ClassificationLedger:
    """Append-only, queryable history of classification versions.

    A failed (non-converged) run is recorded with ``converged=False`` but is
    never marked as published.  Call :meth:`published_versions` to obtain the
    versions that actually form the published lineage.
    """

    def __init__(self, versions: Sequence[ClassificationVersion] = ()) -> None:
        self._versions: list[ClassificationVersion] = list(versions)
        self._by_id = {version.version_id: version for version in self._versions}
        if len(self._by_id) != len(self._versions):
            raise ClassificationError("duplicate version ids in history")

    def __iter__(self):
        return iter(self._versions)

    def __len__(self) -> int:
        return len(self._versions)

    @property
    def versions(self) -> tuple[ClassificationVersion, ...]:
        return tuple(self._versions)

    def append(self, version: ClassificationVersion) -> None:
        if version.version_id in self._by_id:
            existing = self._by_id[version.version_id]
            if existing == version:
                return
            raise ClassificationError(
                f"version id collision for '{version.version_id}'"
            )
        self._versions.append(version)
        self._by_id[version.version_id] = version

    def get(self, version_id: str) -> ClassificationVersion:
        try:
            return self._by_id[version_id]
        except KeyError:
            raise KeyError(f"unknown classification version '{version_id}'") from None

    def history(self, version_id: str) -> tuple[ClassificationVersion, ...]:
        """Return the lineage from the root to ``version_id``."""
        chain: list[ClassificationVersion] = []
        current: str | None = version_id
        seen: set[str] = set()
        while current is not None:
            if current in seen:
                raise ClassificationError("cycle in classification version history")
            seen.add(current)
            version = self.get(current)
            chain.append(version)
            current = version.parent_version_id
        return tuple(reversed(chain))

    def published_versions(self) -> tuple[ClassificationVersion, ...]:
        return tuple(version for version in self._versions if version.published)

    def by_published_id(self, published_id: str) -> tuple[ClassificationVersion, ...]:
        """Return every run that published the same content.

        Different update orders can produce different lineage rows but the
        same final classification.  This lookup makes that equivalence
        explicit and queryable.
        """
        return tuple(
            version
            for version in self._versions
            if version.published_id == published_id
        )

    def latest(self, *, published: bool = True) -> ClassificationVersion | None:
        for version in reversed(self._versions):
            if not published or version.published:
                return version
        return None

    def to_dict(self) -> dict[str, Any]:
        return {"versions": [version.to_dict() for version in self._versions]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ClassificationLedger":
        return cls(
            [ClassificationVersion.from_dict(item) for item in data["versions"]]
        )


# A short alias that reads naturally in application code.
ClassificationHistory = ClassificationLedger


class _SpatialIndex:
    """Small dependency-free uniform grid index for radius queries."""

    def __init__(self, x, y, z, cell_size: float) -> None:
        self.cell_size = float(cell_size)
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.z = np.asarray(z, dtype=np.float64)
        # A tiny cell size can create an excessive number of empty cells for
        # coordinates with a huge offset.  The origin shift keeps the keys
        # bounded without changing distances.
        self._ox = float(self.x.min()) if self.x.size else 0.0
        self._oy = float(self.y.min()) if self.y.size else 0.0
        self._oz = float(self.z.min()) if self.z.size else 0.0
        self._cells = self._build_cells()

    def _cell_coords(self, values, origin):
        return np.floor((values - origin) / self.cell_size).astype(np.int64)

    def _build_cells(self) -> dict[tuple[int, int, int], NDArray[np.int64]]:
        if not len(self.x):
            return {}
        keys = np.column_stack(
            (
                self._cell_coords(self.x, self._ox),
                self._cell_coords(self.y, self._oy),
                self._cell_coords(self.z, self._oz),
            )
        )
        unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
        order = np.argsort(inverse, kind="stable")
        boundaries = np.flatnonzero(np.diff(inverse[order])) + 1
        groups = np.split(order, boundaries)
        return {
            tuple(int(value) for value in key): group.astype(np.int64, copy=False)
            for key, group in zip(unique_keys, groups)
        }

    def _query_cell_candidates(
        self, centres: NDArray[np.int64], radius: float
    ) -> NDArray[np.int64]:
        if not len(centres):
            return np.empty(0, dtype=np.int64)
        span = max(1, int(math.ceil(radius / self.cell_size)))
        centre_keys = np.column_stack(
            (
                self._cell_coords(self.x[centres], self._ox),
                self._cell_coords(self.y[centres], self._oy),
                self._cell_coords(self.z[centres], self._oz),
            )
        )
        unique_centre_keys = np.unique(centre_keys, axis=0)
        offsets = np.array(
            [
                (dx, dy, dz)
                for dx in range(-span, span + 1)
                for dy in range(-span, span + 1)
                for dz in range(-span, span + 1)
            ],
            dtype=np.int64,
        )
        neighbours: list[NDArray[np.int64]] = []
        for key in unique_centre_keys:
            for offset in offsets:
                group = self._cells.get(
                    (
                        int(key[0] + offset[0]),
                        int(key[1] + offset[1]),
                        int(key[2] + offset[2]),
                    )
                )
                if group is not None:
                    neighbours.append(group)
        if not neighbours:
            return np.empty(0, dtype=np.int64)
        return np.unique(np.concatenate(neighbours))

    def query_many(self, indices: Iterable[int], radius: float) -> set[int]:
        """Return all points within ``radius`` of any of the given points."""
        indices = np.asarray(list(indices), dtype=np.int64)
        if indices.size == 0:
            return set()
        candidates = self._query_cell_candidates(indices, radius)
        if candidates.size == 0:
            return set()

        # Chunk the centre points so the temporary boolean matrix stays
        # bounded even for large dirty sets.
        max_pairs = 4_000_000
        chunk = max(1, max_pairs // max(1, candidates.size))
        selected: list[NDArray[np.int64]] = []
        for start in range(0, indices.size, chunk):
            centres = indices[start : start + chunk]
            dx = self.x[candidates, None] - self.x[centres][None, :]
            dy = self.y[candidates, None] - self.y[centres][None, :]
            dz = self.z[candidates, None] - self.z[centres][None, :]
            within = (dx * dx + dy * dy + dz * dz) <= (radius * radius)
            selected.append(candidates[np.any(within, axis=1)])
        if not selected:
            return set()
        return set(int(value) for value in np.unique(np.concatenate(selected)))

    def query(self, index: int, radius: float) -> NDArray[np.int64]:
        candidates = self._query_cell_candidates(
            np.asarray([index], dtype=np.int64), radius
        )
        if candidates.size == 0:
            return np.empty(0, dtype=np.int64)
        dx = self.x[candidates] - self.x[index]
        dy = self.y[candidates] - self.y[index]
        dz = self.z[candidates] - self.z[index]
        mask = (dx * dx + dy * dy + dz * dz) <= (radius * radius)
        return candidates[mask]


@dataclass
class _RoundContext:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    z: NDArray[np.float64]
    original: NDArray[np.uint8]
    seed: NDArray[np.uint8]
    current: NDArray[np.uint8]
    protected: NDArray[np.bool_]
    index: _SpatialIndex

@dataclass
class CleaningResult:
    """The outcome of one deterministic cleaning run.

    ``classification`` is the last computed state.  For an incomplete run it
    must not be published; use :attr:`converged` / :meth:`raise_for_status` and
    :meth:`apply_to_las` enforces this.
    """

    ruleset: ClassificationRuleSet
    original_classification: NDArray[np.uint8]
    seed_classification: NDArray[np.uint8]
    classification: NDArray[np.uint8]
    corrections: Mapping[int, int]
    changes: tuple[ClassificationChange, ...]
    converged: bool
    status: str
    rounds: int
    max_iterations: int
    evaluated_count: int
    recomputed_indices: NDArray[np.int64]
    version: ClassificationVersion
    ledger: ClassificationLedger

    @property
    def rule_version(self) -> str:
        return self.ruleset.version

    @property
    def completed(self) -> bool:
        return self.converged

    @property
    def version_id(self) -> str:
        return self.version.version_id

    @property
    def published_id(self) -> str | None:
        """Content id shared by every path that reaches this exact result."""
        return self.version.published_id

    @property
    def rule_changes(self) -> tuple[ClassificationChange, ...]:
        return tuple(change for change in self.changes if change.source == "rule")

    def explain(self, point_index: int) -> tuple[ClassificationChange, ...]:
        """Return every correction/rule decision that touched a point."""
        if not 0 <= int(point_index) < len(self.original_classification):
            raise IndexError("point_index is out of range")
        index = int(point_index)
        return tuple(
            change for change in self.changes if change.point_index == index
        )

    def raise_for_status(self) -> None:
        if not self.converged:
            raise ClassificationNotConverged(self)

    def summary(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "rule_version": self.rule_version,
            "status": self.status,
            "converged": self.converged,
            "rounds": self.rounds,
            "changes": len(self.rule_changes),
            "evaluated_points": self.evaluated_count,
            "recomputed_points": int(self.recomputed_indices.size),
        }

    def apply_to_las(
        self,
        las,
        *,
        dimension_name: str = "cleaned_classification",
        store_history: bool = True,
    ) -> None:
        """Store the cleaned class in an extra dimension of ``las``.

        The standard ``las.classification`` field is never modified.  A
        non-converged result is rejected before anything is changed.

        ``dimension_name`` must name an extra dimension.  Passing a standard
        dimension name (for example ``"classification"``) raises
        :class:`ClassificationError`, so the device original cannot be
        overwritten by accident.
        """
        self.raise_for_status()
        if len(las.points) != len(self.classification):
            raise ClassificationError(
                "the LAS file and the cleaning result have different point counts"
            )
        if dimension_name in set(las.point_format.standard_dimension_names):
            raise ClassificationError(
                f"'{dimension_name}' is a standard LAS dimension; "
                "the device original must be preserved. Choose an extra "
                "dimension name."
            )
        import laspy

        existing = set(las.point_format.dimension_names)
        if dimension_name not in existing:
            las.add_extra_dim(
                laspy.ExtraBytesParams(
                    name=dimension_name,
                    type="uint8",
                    description=f"cleaned classification ({self.rule_version})",
                )
            )
        las[dimension_name] = self.classification
        if store_history:
            store_classification_history(las, self.ledger)


class ClassificationCleaner:
    """Run a versioned rule set deterministically and incrementally."""

    def __init__(
        self,
        ruleset: ClassificationRuleSet,
        *,
        max_iterations: int = 1024,
    ) -> None:
        if not isinstance(ruleset, ClassificationRuleSet):
            raise TypeError("ruleset must be a ClassificationRuleSet")
        max_iterations = int(max_iterations)
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        self.ruleset = ruleset
        self.max_iterations = max_iterations

    def _prepare_arrays(self, x, y, z, classification):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        z = (
            np.zeros_like(x)
            if z is None
            else np.asarray(z, dtype=np.float64).reshape(-1)
        )
        original = np.asarray(classification)
        if original.ndim != 1:
            original = original.reshape(-1)
        if not (len(x) == len(y) == len(z) == len(original)):
            raise ValueError("coordinates and classification must have equal length")
        if not (
            np.all(np.isfinite(x))
            and np.all(np.isfinite(y))
            and np.all(np.isfinite(z))
        ):
            raise ValueError("coordinates must be finite")
        if original.size and (original.min() < 0 or original.max() > 255):
            raise ValueError("classification values must be in the range 0..255")
        original = np.ascontiguousarray(original, dtype=np.uint8)
        return x, y, z, original

    def _normalise_corrections(
        self,
        corrections: Mapping[int, int] | NDArray[Any] | None,
        point_count: int,
    ) -> dict[int, int]:
        if corrections is None:
            return {}
        if isinstance(corrections, Mapping):
            items = corrections.items()
        else:
            values = np.asarray(corrections)
            if values.ndim != 1 or len(values) != point_count:
                raise ValueError(
                    "a correction array must have one value per point"
                )
            if np.issubdtype(values.dtype, np.floating):
                indices = np.flatnonzero(~np.isnan(values))
            else:
                indices = np.flatnonzero(values >= 0)
            items = ((int(index), values[index]) for index in indices)
        result: dict[int, int] = {}
        for raw_index, raw_value in items:
            index = int(raw_index)
            if not 0 <= index < point_count:
                raise IndexError(f"correction point index {index} is out of range")
            result[index] = _as_int_class(raw_value)
        return result

    def _seed(
        self, original: NDArray[np.uint8], corrections: Mapping[int, int]
    ) -> NDArray[np.uint8]:
        seed = original.copy()
        for index, value in corrections.items():
            seed[index] = value
        return seed

    def _correction_changes(
        self,
        base: NDArray[np.uint8],
        seed: NDArray[np.uint8],
        corrections: Mapping[int, int],
        original: NDArray[np.uint8],
    ) -> list[ClassificationChange]:
        """Describe correction-overlay changes new in this version.

        ``base`` is the previous effective seed (the device original for the
        first version).  A correction that was already carried by the parent
        version is not reported again; the version ledger keeps it traceable
        without duplicating it in every descendant.  Corrections that were
        removed are reported too, because they also change the seed.
        """
        differing = np.flatnonzero(base != seed)
        changes: list[ClassificationChange] = []
        for index in differing:
            index = int(index)
            old_class = int(base[index])
            new_class = int(seed[index])
            removed = index not in corrections
            changes.append(
                ClassificationChange(
                    point_index=index,
                    round=0,
                    old_class=old_class,
                    new_class=new_class,
                    rule_id="__correction__",
                    rule_version=self.ruleset.version,
                    reason=(
                        (
                            f"manual correction removed; reverted "
                            f"{old_class} -> {new_class} while retaining the "
                            f"device original {int(original[index])}"
                        )
                        if removed
                        else (
                            f"manual correction ({old_class} -> {new_class}); "
                            f"device original {int(original[index])} is retained"
                        )
                    ),
                    evidence={
                        "original_class": int(original[index]),
                        "correction_removed": removed,
                    },
                    source="correction",
                )
            )
        return changes

    def _initial_candidates(self, context: _RoundContext) -> set[int]:
        candidates: set[int] = set()
        for rule in self.ruleset:
            candidates.update(int(value) for value in rule._candidates(context))
        return candidates

    def _proposals(
        self, context: _RoundContext, dirty: Iterable[int]
    ) -> dict[int, list[_Proposal]]:
        dirty_array = np.asarray(sorted(set(dirty)), dtype=np.int64)
        if dirty_array.size and context.protected.any():
            dirty_array = dirty_array[~context.protected[dirty_array]]
        proposals: dict[int, list[_Proposal]] = {}
        if dirty_array.size == 0:
            return proposals
        for rule in self.ruleset:
            rule_proposals = rule._propose(context, dirty_array)
            for index, proposal in rule_proposals.items():
                proposals.setdefault(index, []).append(proposal)
        return proposals

    @staticmethod
    def _winner(proposals: Sequence[_Proposal]) -> _Proposal:
        # Higher priority wins.  Ties are resolved by rule id, never by the
        # order in which the rules or points happened to be processed.
        return min(proposals, key=lambda item: (-item.rule.priority, item.rule.rule_id))

    def _reason(self, proposal: _Proposal, old_class: int) -> str:
        rule = proposal.rule
        evidence = proposal.evidence
        if rule.kind == "neighbor":
            matching = evidence["matching_neighbors"]
            total = evidence["total_neighbors"]
            classes = evidence["neighbor_classes"]
            class_text = "any class" if classes is None else f"classes {classes}"
            return (
                f"rule '{rule.rule_id}' changed {old_class} -> {proposal.target}: "
                f"{matching}/{total} neighbours within {evidence['radius']} "
                f"belong to {class_text}"
            )
        return (
            f"rule '{rule.rule_id}' changed {old_class} -> {proposal.target}: "
            f"point lies in the explicit spatial extent"
        )

    def _make_version(
        self,
        *,
        parent_version_id: str | None,
        original: NDArray[np.uint8],
        seed: NDArray[np.uint8],
        classification: NDArray[np.uint8],
        corrections: Mapping[int, int],
        changes: Sequence[ClassificationChange],
        converged: bool,
        status: str,
        rounds: int,
        evaluated_count: int,
        recomputed_indices: NDArray[np.int64],
    ) -> ClassificationVersion:
        payload = {
            "parent_version_id": parent_version_id,
            "rule_version": self.ruleset.version,
            "rules": self.ruleset.to_dict(),
            "original_digest": _digest(original),
            "seed_digest": _digest(seed),
            "classification_digest": _digest(classification),
            "corrections": {
                str(index): int(value) for index, value in sorted(corrections.items())
            },
            "changes": [change.to_dict() for change in changes],
            "status": status,
        }
        version_id = _version_id(payload)
        return ClassificationVersion(
            version_id=version_id,
            parent_version_id=parent_version_id,
            rule_version=self.ruleset.version,
            status=status,
            converged=converged,
            rounds=rounds,
            max_iterations=self.max_iterations,
            original_digest=_digest(original),
            seed_digest=_digest(seed),
            classification_digest=_digest(classification),
            corrections=dict(corrections),
            changes=tuple(changes),
            evaluated_count=evaluated_count,
            recomputed_indices_count=int(recomputed_indices.size),
            rules=self.ruleset.to_dict(),
        )

    def _run(
        self,
        *,
        x,
        y,
        z,
        original,
        corrections,
        working: NDArray[np.uint8],
        dirty: set[int],
        parent_version_id: str | None,
        ledger: ClassificationLedger,
        index: _SpatialIndex | None = None,
        previous_seed: NDArray[np.uint8] | None = None,
    ) -> CleaningResult:
        seed = self._seed(original, corrections)
        parent_seed = original if previous_seed is None else previous_seed
        protected = np.zeros(len(original), dtype=bool)
        if corrections:
            protected[np.fromiter(corrections, dtype=np.int64)] = True
        if index is None:
            index = _SpatialIndex(
                x, y, z, self.ruleset.max_neighbor_radius or 1.0
            )
        changes = self._correction_changes(
            parent_seed, seed, corrections, original
        )
        evaluated: set[int] = set()
        recomputed: set[int] = set()
        seen_states = {(_digest(working), _indices_digest(dirty))}
        status = "max_iterations"
        converged = False
        completed_rounds = 0

        for round_number in range(1, self.max_iterations + 1):
            completed_rounds = round_number
            context = _RoundContext(
                x=x,
                y=y,
                z=z,
                original=original,
                seed=seed,
                current=working.copy(),
                protected=protected,
                index=index,
            )
            dirty = set(dirty)
            evaluated.update(dirty)
            recomputed.update(dirty)
            proposals = self._proposals(context, dirty)
            if not proposals:
                converged = True
                status = "converged"
                break

            next_working = working.copy()
            changed_this_round: set[int] = set()
            for point_index in sorted(proposals):
                old_class = int(working[point_index])
                winner = self._winner(proposals[point_index])
                if winner.target == old_class:
                    continue
                next_working[point_index] = winner.target
                changed_this_round.add(point_index)
                evidence = dict(winner.evidence)
                evidence["alternatives"] = [
                    {
                        "rule_id": proposal.rule.rule_id,
                        "priority": proposal.rule.priority,
                        "target": proposal.target,
                    }
                    for proposal in sorted(
                        proposals[point_index],
                        key=lambda item: (
                            -item.rule.priority,
                            item.rule.rule_id,
                        ),
                    )
                    if proposal is not winner
                ]
                changes.append(
                    ClassificationChange(
                        point_index=point_index,
                        round=round_number,
                        old_class=old_class,
                        new_class=winner.target,
                        rule_id=winner.rule.rule_id,
                        rule_version=self.ruleset.version,
                        reason=self._reason(winner, old_class),
                        evidence=evidence,
                        source="rule",
                    )
                )

            if not changed_this_round:
                converged = True
                status = "converged"
                break

            working = next_working

            # The next round only needs changed points (their own class may
            # now match another static rule) and their neighbourhood within
            # the largest neighbour radius.  This is the incremental part of
            # propagation.
            next_dirty = set(changed_this_round)
            radius = self.ruleset.max_neighbor_radius
            if radius > 0:
                next_dirty.update(index.query_many(changed_this_round, radius))

            # A cycle means both the classification and the set of points
            # that would be evaluated next repeat.
            state_digest = (_digest(working), _indices_digest(next_dirty))
            if state_digest in seen_states:
                status = "cycle"
                break
            seen_states.add(state_digest)
            dirty = next_dirty

        version = self._make_version(
            parent_version_id=parent_version_id,
            original=original,
            seed=seed,
            classification=working,
            corrections=corrections,
            changes=changes,
            converged=converged,
            status=status,
            rounds=completed_rounds,
            evaluated_count=len(evaluated),
            recomputed_indices=np.asarray(sorted(recomputed), dtype=np.int64),
        )
        ledger.append(version)
        return CleaningResult(
            ruleset=self.ruleset,
            original_classification=original.copy(),
            seed_classification=seed.copy(),
            classification=working.copy(),
            corrections=dict(corrections),
            changes=tuple(changes),
            converged=converged,
            status=status,
            rounds=completed_rounds,
            max_iterations=self.max_iterations,
            evaluated_count=len(evaluated),
            recomputed_indices=np.asarray(sorted(recomputed), dtype=np.int64),
            version=version,
            ledger=ledger,
        )

    def clean(
        self,
        x,
        y,
        z,
        classification,
        *,
        corrections: Mapping[int, int] | NDArray[Any] | None = None,
        ledger: ClassificationLedger | None = None,
        parent_version_id: str | None = None,
    ) -> CleaningResult:
        """Clean a point cloud supplied as coordinate and class arrays."""
        x, y, z, original = self._prepare_arrays(x, y, z, classification)
        normalised_corrections = self._normalise_corrections(
            corrections, len(original)
        )
        seed = self._seed(original, normalised_corrections)
        active_ledger = ledger if ledger is not None else ClassificationLedger()
        index = _SpatialIndex(
            x, y, z, self.ruleset.max_neighbor_radius or 1.0
        )
        context = _RoundContext(
            x=x,
            y=y,
            z=z,
            original=original,
            seed=seed,
            current=seed.copy(),
            protected=np.zeros(len(original), dtype=bool),
            index=index,
        )
        dirty = self._initial_candidates(context)
        return self._run(
            x=x,
            y=y,
            z=z,
            original=original,
            corrections=normalised_corrections,
            working=seed.copy(),
            dirty=dirty,
            parent_version_id=parent_version_id,
            ledger=active_ledger,
        )

    def clean_las(
        self,
        las,
        *,
        corrections: Mapping[int, int] | NDArray[Any] | None = None,
        ledger: ClassificationLedger | None = None,
        parent_version_id: str | None = None,
    ) -> CleaningResult:
        """Clean the points of a :class:`laspy.LasData` instance."""
        if not hasattr(las, "xyz") or not hasattr(las, "classification"):
            raise TypeError("clean_las expects a LasData-like object")
        xyz = np.asarray(las.xyz)
        return self.clean(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            np.asarray(las.classification),
            corrections=corrections,
            ledger=ledger,
            parent_version_id=parent_version_id,
        )

    __call__ = clean_las


class ClassificationSession:
    """Incremental cleaning session with an append-only version history.

    ``correct`` / ``apply_corrections`` update only the correction overlay.
    The next :meth:`run` resets and recomputes the neighbourhood closure of the
    changed seeds; untouched clusters are retained from the previous published
    version.
    """

    def __init__(
        self,
        ruleset: ClassificationRuleSet,
        x,
        y,
        z,
        classification,
        *,
        corrections: Mapping[int, int] | NDArray[Any] | None = None,
        max_iterations: int = 1024,
    ) -> None:
        self.cleaner = ClassificationCleaner(
            ruleset, max_iterations=max_iterations
        )
        self.x, self.y, self.z, self.original = self.cleaner._prepare_arrays(
            x, y, z, classification
        )
        self._corrections = self.cleaner._normalise_corrections(
            corrections, len(self.original)
        )
        self.ledger = ClassificationLedger()
        self._last_result: CleaningResult | None = None
        self._index = _SpatialIndex(
            self.x,
            self.y,
            self.z,
            self.cleaner.ruleset.max_neighbor_radius or 1.0,
        )

    @classmethod
    def from_las(
        cls,
        las,
        ruleset: ClassificationRuleSet,
        *,
        corrections: Mapping[int, int] | NDArray[Any] | None = None,
        max_iterations: int = 1024,
    ) -> "ClassificationSession":
        xyz = np.asarray(las.xyz)
        return cls(
            ruleset,
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            np.asarray(las.classification),
            corrections=corrections,
            max_iterations=max_iterations,
        )

    @property
    def corrections(self) -> Mapping[int, int]:
        return dict(self._corrections)

    @property
    def last_result(self) -> CleaningResult | None:
        return self._last_result

    @property
    def original_classification(self) -> NDArray[np.uint8]:
        return self.original.copy()

    def set_correction(self, point_index: int, classification: int) -> None:
        index = int(point_index)
        if not 0 <= index < len(self.original):
            raise IndexError("point_index is out of range")
        value = _as_int_class(classification)
        self._corrections[index] = value

    def clear_correction(self, point_index: int) -> None:
        self._corrections.pop(int(point_index), None)

    def apply_corrections(self, corrections: Mapping[int, int]) -> None:
        normalised = self.cleaner._normalise_corrections(
            corrections, len(self.original)
        )
        for index, value in normalised.items():
            self.set_correction(index, value)

    def _closure(self, seeds: Iterable[int]) -> set[int]:
        radius = self.cleaner.ruleset.max_neighbor_radius
        closure = set(int(index) for index in seeds)
        if radius <= 0:
            return closure
        frontier = set(closure)
        while frontier:
            neighbours = self._index.query_many(frontier, radius)
            new_points = neighbours - closure
            if not new_points:
                break
            closure.update(new_points)
            frontier = new_points
        return closure

    def run(self) -> CleaningResult:
        """Compute the next version, reusing unaffected clusters when possible."""
        seed = self.cleaner._seed(self.original, self._corrections)

        if self._last_result is None:
            return self._full_run(seed)

        previous = self._last_result
        previous_seed = previous.seed_classification
        changed_seeds = {
            index
            for index in set(previous_seed.nonzero()[0])
            | set(seed.nonzero()[0])
            | set(self._corrections)
            | set(previous.corrections)
            if int(previous_seed[index]) != int(seed[index])
        }
        if not changed_seeds:
            # The same seed produces the same fixed point; returning the
            # existing result avoids creating a meaningless new version.
            return previous

        closure = self._closure(changed_seeds)
        working = previous.classification.copy()
        # Reset the whole potentially affected radius closure to the seed, so
        # stale derived decisions from the previous version cannot survive.
        # Points outside the closure are provably independent of the changed
        # seeds and are retained untouched.
        closure_indices = np.asarray(sorted(closure), dtype=np.int64)
        working[closure_indices] = seed[closure_indices]
        result = self.cleaner._run(
            x=self.x,
            y=self.y,
            z=self.z,
            original=self.original,
            corrections=dict(self._corrections),
            working=working,
            dirty=closure,
            parent_version_id=previous.version_id,
            ledger=self.ledger,
            index=self._index,
            previous_seed=previous_seed,
        )
        if result.converged:
            self._last_result = result
        return result

    def _full_run(self, seed: NDArray[np.uint8]) -> CleaningResult:
        context = _RoundContext(
            x=self.x,
            y=self.y,
            z=self.z,
            original=self.original,
            seed=seed,
            current=seed.copy(),
            protected=(self.original != seed),
            index=self._index,
        )
        dirty = self.cleaner._initial_candidates(context)
        result = self.cleaner._run(
            x=self.x,
            y=self.y,
            z=self.z,
            original=self.original,
            corrections=dict(self._corrections),
            working=seed.copy(),
            dirty=dirty,
            parent_version_id=None,
            ledger=self.ledger,
        )
        if result.converged:
            self._last_result = result
        return result


def clean_classification(
    las,
    ruleset: ClassificationRuleSet,
    *,
    corrections: Mapping[int, int] | NDArray[Any] | None = None,
    max_iterations: int = 1024,
    ledger: ClassificationLedger | None = None,
    parent_version_id: str | None = None,
) -> CleaningResult:
    """Convenience wrapper around :meth:`ClassificationCleaner.clean_las`."""
    return ClassificationCleaner(
        ruleset, max_iterations=max_iterations
    ).clean_las(
        las,
        corrections=corrections,
        ledger=ledger,
        parent_version_id=parent_version_id,
    )


# ---------------------------------------------------------------------------
# Persistence in LAS VLRs
# ---------------------------------------------------------------------------

_HISTORY_USER_ID = "laspy.clean"
_HISTORY_RECORD_ID = 0x0100
_HISTORY_MAGIC = b"LSCL1"
_HISTORY_CHUNK_SIZE = 60_000


def _history_records(ledger: ClassificationLedger) -> list[bytes]:
    payload = ledger.to_dict()
    payload["format"] = "laspy-classification-history"
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    compressed = zlib.compress(raw, level=9)
    chunks = [
        compressed[offset : offset + _HISTORY_CHUNK_SIZE]
        for offset in range(0, len(compressed), _HISTORY_CHUNK_SIZE)
    ] or [b""]
    records = []
    for chunk_index, chunk in enumerate(chunks):
        header = (
            _HISTORY_MAGIC
            + len(chunks).to_bytes(4, "little")
            + chunk_index.to_bytes(4, "little")
        )
        records.append(header + chunk)
    return records


def store_classification_history(
    las, ledger: ClassificationLedger
) -> None:
    """Replace classification-history VLRs in ``las`` with ``ledger``.

    The standard classification field is not touched.
    """
    from .vlrs.vlr import VLR

    kept = [
        vlr
        for vlr in las.vlrs
        if not (
            vlr.user_id == _HISTORY_USER_ID
            and vlr.record_id == _HISTORY_RECORD_ID
        )
    ]
    las.vlrs.clear()
    las.vlrs.extend(kept)
    for chunk in _history_records(ledger):
        las.vlrs.append(
            VLR(
                user_id=_HISTORY_USER_ID,
                record_id=_HISTORY_RECORD_ID,
                description="classification history",
                record_data=chunk,
            )
        )


def load_classification_history(las) -> ClassificationLedger:
    """Read the classification history stored by
    :func:`store_classification_history`.

    A file without history returns an empty ledger.
    """
    chunks: dict[int, bytes] = {}
    total_chunks: int | None = None
    for vlr in las.vlrs:
        if (
            vlr.user_id != _HISTORY_USER_ID
            or vlr.record_id != _HISTORY_RECORD_ID
        ):
            continue
        data = vlr.record_data_bytes()
        if not data.startswith(_HISTORY_MAGIC):
            continue
        offset = len(_HISTORY_MAGIC)
        count = int.from_bytes(data[offset : offset + 4], "little")
        chunk_index = int.from_bytes(data[offset + 4 : offset + 8], "little")
        if total_chunks is None:
            total_chunks = count
        elif total_chunks != count:
            raise ClassificationError("inconsistent classification history chunks")
        chunks[chunk_index] = data[offset + 8 :]
    if not chunks:
        return ClassificationLedger()
    if total_chunks is None or set(chunks) != set(range(total_chunks)):
        raise ClassificationError("incomplete classification history")
    compressed = b"".join(chunks[index] for index in range(total_chunks))
    try:
        payload = json.loads(zlib.decompress(compressed).decode("utf-8"))
    except (ValueError, zlib.error) as exc:
        raise ClassificationError(
            "corrupt classification history VLR"
        ) from exc
    if payload.get("format") != "laspy-classification-history":
        raise ClassificationError("unsupported classification history format")
    return ClassificationLedger.from_dict(payload)
