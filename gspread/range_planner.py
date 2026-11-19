"""
gspread.range_planner
~~~~~~~~~~~~~~~~~~~~~

Plan reads for several far-apart regions of one worksheet while respecting
a request budget, a returned-cell budget and regions that must not be read.

The planner is pure geometry: it never talks to the Sheets API. Build a
plan first, inspect :meth:`RangePlan.explain`, then issue the reads with
:meth:`~gspread.worksheet.Worksheet.batch_get` or one request per range.

Coordinates use Google's ``GridRange`` convention: zero-based and half
open. ``start_row``/``start_column`` are inclusive, ``end_row``/
``end_column`` are exclusive.
"""

from dataclasses import dataclass
from typing import (
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)

from .utils import a1_range_to_grid_range, rowcol_to_a1

Rect = Tuple[int, int, int, int]

#: Values accepted by :attr:`RangePlanError.conflicted`.
CONFLICT_REQUESTS = "max_requests"
CONFLICT_CELLS = "max_cells"
CONFLICT_BOTH = "max_requests+max_cells"
CONFLICT_FORBIDDEN = "forbidden"


class RangePlanError(ValueError):
    """No legal plan satisfies every hard constraint.

    :ivar reason: Human-readable explanation naming the binding constraint.
    :ivar conflicted: One of ``"max_requests"``, ``"max_cells"``,
        ``"max_requests+max_cells"`` or ``"forbidden"``.
    :ivar best_plan: The closest candidate plan, when one exists. This plan
        is **not** legal for the given budgets; it is provided for
        diagnostics only.
    """

    def __init__(
        self,
        reason: str,
        conflicted: Optional[str] = None,
        best_plan: Optional["RangePlan"] = None,
    ):
        super().__init__(reason)
        self.reason = reason
        self.conflicted = conflicted
        self.best_plan = best_plan


class InvalidRangePlanError(ValueError):
    """Malformed planner input, such as invalid A1 notation or bounds."""


@dataclass(frozen=True)
class GridRegion:
    """A rectangle on one worksheet, with zero-based half-open bounds.

    A missing bound means "unbounded": for example ``A5:B`` leaves
    ``end_row`` and ``end_column`` unset. Unbounded regions are resolved
    against the worksheet's ``row_count``/``column_count`` before they can
    participate in cell accounting.
    """

    start_row: Optional[int] = None
    end_row: Optional[int] = None
    start_column: Optional[int] = None
    end_column: Optional[int] = None

    def __post_init__(self) -> None:
        for name in ("start_row", "end_row", "start_column", "end_column"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise InvalidRangePlanError(
                    "{} must be >= 0, got {}".format(name, value)
                )
        if (
            self.start_row is not None
            and self.end_row is not None
            and self.start_row > self.end_row
        ):
            raise InvalidRangePlanError(
                "start_row {} is greater than end_row {}".format(
                    self.start_row, self.end_row
                )
            )
        if (
            self.start_column is not None
            and self.end_column is not None
            and self.start_column > self.end_column
        ):
            raise InvalidRangePlanError(
                "start_column {} is greater than end_column {}".format(
                    self.start_column, self.end_column
                )
            )

    def resolve(
        self, row_count: Optional[int] = None, column_count: Optional[int] = None
    ) -> Rect:
        """Return concrete ``(r0, r1, c0, c1)`` bounds.

        Every unbounded side falls back to the sheet size. If the sheet size
        is unknown, a :class:`InvalidRangePlanError` is raised rather than
        pretending the sheet is unlimited.
        """
        start_row = 0 if self.start_row is None else self.start_row
        start_column = 0 if self.start_column is None else self.start_column

        if self.end_row is None:
            if row_count is None:
                raise InvalidRangePlanError("range needs end_row or a known row_count")
            end_row = row_count
        else:
            end_row = self.end_row

        if self.end_column is None:
            if column_count is None:
                raise InvalidRangePlanError(
                    "range needs end_column or a known column_count"
                )
            end_column = column_count
        else:
            end_column = self.end_column

        if row_count is not None and end_row > row_count:
            raise InvalidRangePlanError(
                "range ends at row {} beyond row_count {}".format(end_row, row_count)
            )
        if column_count is not None and end_column > column_count:
            raise InvalidRangePlanError(
                "range ends at column {} beyond column_count {}".format(
                    end_column, column_count
                )
            )
        return (start_row, end_row, start_column, end_column)

    def to_a1(
        self, row_count: Optional[int] = None, column_count: Optional[int] = None
    ) -> str:
        """Render the region in A1 notation.

        Unbounded regions need the sheet size so the label is unambiguous;
        otherwise :class:`InvalidRangePlanError` is raised.
        """
        return _render_rect(self.resolve(row_count, column_count))

    @classmethod
    def from_a1(
        cls,
        range_name: str,
        row_count: Optional[int] = None,
        column_count: Optional[int] = None,
    ) -> "GridRegion":
        """Build a region from A1 notation, optionally qualified by a sheet."""
        a1 = range_name.split("!")[-1]
        if not a1:
            raise InvalidRangePlanError("empty range: {!r}".format(range_name))
        try:
            grid = a1_range_to_grid_range(a1)
        except Exception as exc:
            raise InvalidRangePlanError(
                "invalid A1 range {!r}: {}".format(range_name, exc)
            ) from exc

        region = cls(
            start_row=grid.get("startRowIndex"),
            end_row=grid.get("endRowIndex"),
            start_column=grid.get("startColumnIndex"),
            end_column=grid.get("endColumnIndex"),
        )
        region.resolve(row_count, column_count)
        return region

    def __str__(self) -> str:
        return self.to_a1()


@dataclass(frozen=True)
class PlannedRange:
    """One range that would be requested.

    :ivar range: A1 notation of the range to read.
    :ivar cells: Rectangle area of ``range``.
    :ivar new_cells: Cells not already returned by an earlier planned range.
        Summing this over a plan gives :attr:`RangePlan.unique_cells`, so
        shared cells are counted once.
    :ivar covers: Requested ranges whose cells this range serves.
    :ivar overhead_cells: Nominal cells in ``range`` that are not part of
        any requested range it serves. If planned ranges overlap, this
        nominal value can double-count; :attr:`RangePlan.overhead_cells` is
        the exact total.
    """

    range: str
    cells: int
    rect: Rect
    new_cells: int = 0
    covers: Tuple[str, ...] = ()
    overhead_cells: int = 0

    def __str__(self) -> str:
        return self.range


@dataclass(frozen=True)
class ClippedRange:
    """A requested range that had to be split around forbidden regions."""

    range: str
    kept: Tuple[str, ...]
    kept_cells: int
    removed_cells: int

    def __str__(self) -> str:
        return self.range


@dataclass(frozen=True)
class UnsatisfiedRange:
    """A requested range the plan cannot read at all."""

    range: str
    reason: str

    def __str__(self) -> str:
        return self.range


@dataclass(frozen=True)
class RangePlan:
    """A set of read ranges plus the accounting behind it.

    :ivar ranges: One entry per range that would be requested.
    :ivar unique_cells: Distinct cells returned; overlaps and shared edges
        are counted once.
    :ivar requested_cells: Distinct legal cells in the requested regions.
        ``unique_cells - requested_cells`` is the exact over-read caused by
        merging.
    :ivar alternatives: Pareto-efficient legal plans, ordered by the
        planner's objective. A plan is omitted when another plan uses no
        more requests and no more cells.
    :ivar clipped: Requested ranges split around forbidden regions.
    :ivar unsatisfied: Requested ranges left out entirely, only populated
        when ``allow_partial=True``.
    """

    ranges: Tuple[PlannedRange, ...]
    unique_cells: int
    requested_cells: int = 0
    alternatives: Tuple["RangePlan", ...] = ()
    clipped: Tuple[ClippedRange, ...] = ()
    unsatisfied: Tuple[UnsatisfiedRange, ...] = ()

    @property
    def requests(self) -> int:
        """Requests if every planned range is read separately."""
        return len(self.ranges)

    @property
    def batched_requests(self) -> int:
        """Requests if all planned ranges go through one ``batch_get``."""
        return 1 if self.ranges else 0

    @property
    def overhead_cells(self) -> int:
        """Cells returned beyond the requested regions."""
        return self.unique_cells - self.requested_cells

    @property
    def forbidden_cells_removed(self) -> int:
        """Cells excluded from requested ranges because they are forbidden."""
        return sum(item.removed_cells for item in self.clipped)

    def a1_ranges(self) -> List[str]:
        """The planned ranges, ready for ``Worksheet.batch_get``."""
        return [item.range for item in self.ranges]

    def explain(self) -> str:
        """A short, deterministic, human-readable plan summary."""
        lines = [
            "{} request(s) separately, {} via batch_get, {} unique cell(s); "
            "requested {} cell(s), over-read {}".format(
                self.requests,
                self.batched_requests,
                self.unique_cells,
                self.requested_cells,
                self.overhead_cells,
            )
        ]
        for item in self.ranges:
            details = ["{} cells".format(item.cells)]
            if item.covers and item.covers != (item.range,):
                details.append("serves " + ", ".join(item.covers))
            if item.overhead_cells:
                details.append("{} over-read".format(item.overhead_cells))
            if item.new_cells != item.cells:
                details.append("{} shared".format(item.cells - item.new_cells))
            lines.append("  {} ({})".format(item.range, "; ".join(details)))
        for item in self.clipped:
            lines.append(
                "  clipped {} -> {} ({} cell(s) forbidden)".format(
                    item.range,
                    ", ".join(item.kept),
                    item.removed_cells,
                )
            )
        for item in self.unsatisfied:
            lines.append("  unsatisfied {} ({})".format(item.range, item.reason))
        return "\n".join(lines)


class RangePlanner:
    """Build a legal read plan for several regions.

    :param int max_requests: Maximum number of ranges in the plan. This is
        the number of range expressions charged to the request budget; if
        the caller sends all of them through one ``batch_get`` call, the
        HTTP request count is one but the range expressions are still what
        this budget limits.
    :param int max_cells: Maximum number of distinct returned cells.
    :param forbidden: Regions that must not be read. Planned ranges never
        intersect one, even when a wider range would save a request.
    :param int row_count: Sheet rows, needed to resolve unbounded ranges.
    :param int column_count: Sheet columns, needed for unbounded ranges.
    :param str sheet_title: Optional title used when rendering A1 ranges.
    :param bool allow_partial: When a requested range lies entirely inside
        forbidden regions, report it in :attr:`RangePlan.unsatisfied`
        instead of raising. Partially forbidden ranges are always clipped
        and reported, because a legal combination still exists for them.
    :param str objective: Selection rule among legal plans.

        * ``"requests"`` (default): fewest requests, then fewest over-read
          cells. This matches the common case where the request quota is
          the scarce resource; ``max_cells`` still bounds the return, and
          forbidden regions are still hard exclusions.
        * ``"overhead"``: fewest over-read cells, then fewest requests.
          Prefer this when data minimisation matters more than request
          count.

    Every Pareto-efficient legal plan is available through
    :attr:`RangePlan.alternatives`, so the trade-off is explicit rather than
    hidden in the selection rule.

    For up to eight planned units every set partition is examined exactly,
    so a "no legal plan" verdict is not an artefact of greedy ordering.
    Larger inputs fall back to band and pairwise merging.
    """

    def __init__(
        self,
        max_requests: int,
        max_cells: int,
        forbidden: Iterable[GridRegion] = (),
        row_count: Optional[int] = None,
        column_count: Optional[int] = None,
        sheet_title: Optional[str] = None,
        allow_partial: bool = False,
        objective: str = "requests",
    ):
        if max_requests < 0:
            raise InvalidRangePlanError("max_requests must be >= 0")
        if max_cells < 0:
            raise InvalidRangePlanError("max_cells must be >= 0")
        if objective not in ("overhead", "requests"):
            raise InvalidRangePlanError(
                "objective must be 'overhead' or 'requests', got {!r}".format(objective)
            )
        self.max_requests = max_requests
        self.max_cells = max_cells
        self.row_count = row_count
        self.column_count = column_count
        self.sheet_title = sheet_title
        self.allow_partial = allow_partial
        self.objective = objective
        self.forbidden: Tuple[Rect, ...] = tuple(
            region.resolve(row_count, column_count) for region in forbidden
        )

    def plan(self, requested: Iterable[GridRegion]) -> RangePlan:
        """Return the selected legal plan with its alternatives.

        :raises RangePlanError: if a requested region is entirely forbidden,
            or if no candidate fits both budgets.
            ``RangePlanError.conflicted`` names the binding restriction.
        """
        return self._plan(self._resolve_requested(requested))

    # -- input -----------------------------------------------------------

    def _resolve_requested(
        self, requested: Iterable[GridRegion]
    ) -> List[Tuple[str, Rect]]:
        resolved: List[Tuple[str, Rect]] = []
        for region in requested:
            rect = region.resolve(self.row_count, self.column_count)
            if _is_empty(rect):
                continue
            resolved.append((self._render(rect), rect))
        return resolved

    def _render(self, rect: Rect) -> str:
        label = _render_rect(rect)
        if self.sheet_title:
            return "'{}'!{}".format(self.sheet_title.replace("'", "''"), label)
        return label

    # -- planning --------------------------------------------------------

    def _plan(self, requests: Sequence[Tuple[str, Rect]]) -> RangePlan:
        if not requests:
            raise RangePlanError("no ranges requested", conflicted="input")

        units: List[Tuple[str, Rect]] = []
        clipped: List[ClippedRange] = []
        fully_forbidden: List[UnsatisfiedRange] = []

        for label, rect in requests:
            pieces = self._clip(rect)
            if not pieces:
                fully_forbidden.append(
                    UnsatisfiedRange(label, "entirely inside forbidden regions")
                )
                continue
            kept_cells = sum(_area(piece) for piece in pieces)
            if pieces != [rect]:
                clipped.append(
                    ClippedRange(
                        range=label,
                        kept=tuple(self._render(piece) for piece in pieces),
                        kept_cells=kept_cells,
                        removed_cells=_area(rect) - kept_cells,
                    )
                )
            for piece in pieces:
                units.append((label, piece))

        if fully_forbidden and not self.allow_partial:
            raise RangePlanError(
                "{} requested range(s) lie entirely inside forbidden "
                "regions: {}".format(
                    len(fully_forbidden),
                    ", ".join(item.range for item in fully_forbidden),
                ),
                conflicted=CONFLICT_FORBIDDEN,
            )

        if not units:
            raise RangePlanError(
                "every requested range lies inside a forbidden region",
                conflicted=CONFLICT_FORBIDDEN,
            )

        candidates = self._candidates(units)
        requested_cells = _union_area([rect for _, rect in units])
        feasible = [
            plan
            for plan in candidates
            if plan.requests <= self.max_requests
            and plan.unique_cells <= self.max_cells
        ]

        if not feasible:
            raise self._conflict(
                candidates,
                tuple(clipped),
                tuple(fully_forbidden),
                requested_cells,
            )

        efficient = _pareto(feasible)
        efficient.sort(key=self._selection_key)
        primary = efficient[0]
        return RangePlan(
            ranges=primary.ranges,
            unique_cells=primary.unique_cells,
            requested_cells=requested_cells,
            alternatives=tuple(efficient[1:]),
            clipped=tuple(clipped),
            unsatisfied=tuple(fully_forbidden),
        )

    def _selection_key(self, plan: RangePlan) -> Tuple[int, int, str]:
        if self.objective == "requests":
            key = (plan.requests, plan.unique_cells)
        else:
            key = (plan.overhead_cells, plan.requests)
        return key + (",".join(item.range for item in plan.ranges),)

    def _clip(self, rect: Rect) -> List[Rect]:
        """Subtract every forbidden region, returning disjoint legal pieces."""
        pieces = [rect]
        for cut in self.forbidden:
            next_pieces: List[Rect] = []
            for piece in pieces:
                next_pieces.extend(_subtract(piece, cut))
            pieces = next_pieces
            if not pieces:
                break
        return pieces

    def _candidates(self, units: Sequence[Tuple[str, Rect]]) -> List[RangePlan]:
        """Build deterministic candidate plans from legal units."""
        rects = [rect for _, rect in units]
        plans: Dict[Tuple[Tuple[str, int, Rect], ...], RangePlan] = {}

        def add(rectangles: Sequence[Rect]) -> None:
            plan = self._build(rectangles, units)
            key = tuple((item.range, item.cells, item.rect) for item in plan.ranges)
            plans.setdefault(key, plan)

        if len(rects) <= 8:
            for groups in self._exact_partitions(rects):
                add(groups)
        else:
            add(rects)
            add(self._group_bands(rects))
            for step in self._greedy_merge(rects):
                add(step)

        return list(plans.values())

    def _exact_partitions(self, rects: Sequence[Rect]) -> List[List[Rect]]:
        """Enumerate every legal set partition of the legal units.

        A group is legal only when its bounding rectangle avoids every
        forbidden region. Groups are disjoint subsets of the units; their
        bounding rectangles may overlap, and shared cells are counted once.
        """
        count = len(rects)
        legal_groups: Dict[int, Rect] = {}
        for mask in range(1, 1 << count):
            bounding = _bounding_rect(
                [rects[index] for index in range(count) if mask & (1 << index)]
            )
            assert bounding is not None
            if not self._forbidden_hit(bounding):
                legal_groups[mask] = bounding

        partitions: List[List[Rect]] = []

        def walk(mask: int, groups: List[Rect]) -> None:
            if mask == 0:
                partitions.append(list(groups))
                return
            lowest = mask & -mask
            rest = mask ^ lowest
            submask = rest
            while True:
                group = submask | lowest
                if group in legal_groups:
                    groups.append(legal_groups[group])
                    walk(mask ^ group, groups)
                    groups.pop()
                if submask == 0:
                    break
                submask = (submask - 1) & rest

        walk((1 << count) - 1, [])
        return partitions

    def _group_bands(self, rects: Sequence[Rect]) -> List[Rect]:
        """Merge rectangles sharing identical rows, then identical columns."""
        rows: Dict[Tuple[int, int], List[Rect]] = {}
        for rect in rects:
            rows.setdefault((rect[0], rect[1]), []).append(rect)
        row_merged: List[Rect] = []
        for group in rows.values():
            candidate = (
                group[0][0],
                group[0][1],
                min(rect[2] for rect in group),
                max(rect[3] for rect in group),
            )
            if self._forbidden_hit(candidate):
                row_merged.extend(group)
            else:
                row_merged.append(candidate)

        columns: Dict[Tuple[int, int], List[Rect]] = {}
        for rect in row_merged:
            columns.setdefault((rect[2], rect[3]), []).append(rect)
        column_merged: List[Rect] = []
        for group in columns.values():
            candidate = (
                min(rect[0] for rect in group),
                max(rect[1] for rect in group),
                group[0][2],
                group[0][3],
            )
            if self._forbidden_hit(candidate):
                column_merged.extend(group)
            else:
                column_merged.append(candidate)
        return column_merged

    def _greedy_merge(self, rects: Sequence[Rect]) -> List[List[Rect]]:
        """Merge pairs that reduce requests with the smallest extra cells."""
        current = list(rects)
        snapshots: List[List[Rect]] = [list(current)]
        while len(current) > 1:
            best: Optional[Tuple[int, int, int, int, int]] = None
            best_rect: Optional[Rect] = None
            for i in range(len(current)):
                for j in range(i + 1, len(current)):
                    combined = _bounding_pair(current[i], current[j])
                    if combined is None:
                        continue
                    if self._forbidden_hit(combined):
                        continue
                    added = _area(combined) - _union_area([current[i], current[j]])
                    snapshot = (
                        added,
                        combined[1] - combined[0],
                        combined[3] - combined[2],
                        i,
                        j,
                    )
                    if best is None or snapshot < best:
                        best = snapshot
                        best_rect = combined
            if best_rect is None:
                break
            first, second = best[3], best[4]
            current = [
                rect
                for index, rect in enumerate(current)
                if index not in (first, second)
            ]
            current.append(best_rect)
            snapshots.append(list(current))
        return snapshots

    def _build(
        self, rects: Sequence[Rect], units: Sequence[Tuple[str, Rect]]
    ) -> RangePlan:
        rects = sorted(set(rects))
        unique = _union_area(rects)
        planned: List[PlannedRange] = []
        covered: List[Rect] = []
        for rect in rects:
            assert not self._forbidden_hit(rect)
            overlaps = [
                overlap
                for previous in covered
                for overlap in [_intersection(rect, previous)]
                if overlap is not None
            ]
            new_cells = _area(rect) - _union_area(overlaps)
            served = [
                (label, source) for label, source in units if _contains(rect, source)
            ]
            covers: List[str] = []
            for label, _ in served:
                if label not in covers:
                    covers.append(label)
            planned.append(
                PlannedRange(
                    range=self._render(rect),
                    cells=_area(rect),
                    rect=rect,
                    new_cells=new_cells,
                    covers=tuple(covers),
                    overhead_cells=_area(rect)
                    - _union_area([source for _, source in served]),
                )
            )
            covered.append(rect)
        assert sum(item.new_cells for item in planned) == unique
        return RangePlan(
            ranges=tuple(planned),
            unique_cells=unique,
            requested_cells=_union_area([rect for _, rect in units]),
        )

    def _forbidden_hit(self, rect: Rect) -> bool:
        return any(_intersects(rect, item) for item in self.forbidden)

    def _conflict(
        self,
        candidates: Sequence[RangePlan],
        clipped: Tuple[ClippedRange, ...],
        fully_forbidden: Tuple[UnsatisfiedRange, ...],
        requested_cells: int,
    ) -> RangePlanError:
        if not candidates:
            return RangePlanError(
                "no candidate range plan could be built",
                conflicted=CONFLICT_REQUESTS,
            )

        request_ok = [plan for plan in candidates if plan.requests <= self.max_requests]
        cell_ok = [plan for plan in candidates if plan.unique_cells <= self.max_cells]
        best = min(candidates, key=lambda plan: _violations(plan, self))

        min_requests = min(plan.requests for plan in candidates)
        min_cells = min(plan.unique_cells for plan in candidates)

        if not request_ok and not cell_ok:
            reason = (
                "no plan fits either budget: the legal regions alone cover "
                "{} cell(s), the fewest possible requests is {} "
                "(max_requests={}), and the smallest possible return is {} "
                "cell(s) (max_cells={})".format(
                    requested_cells,
                    min_requests,
                    self.max_requests,
                    min_cells,
                    self.max_cells,
                )
            )
            conflict = CONFLICT_BOTH
        elif not request_ok:
            feasible_requests = (
                min(plan.requests for plan in cell_ok) if cell_ok else min_requests
            )
            reason = (
                "max_requests={} is too small: ignoring the cell budget the "
                "fewest legal requests is {}, but under max_cells={} the "
                "fewest is {} request(s); no merge can get below that "
                "without crossing a forbidden region or over-reading beyond "
                "the cell budget".format(
                    self.max_requests,
                    min_requests,
                    self.max_cells,
                    feasible_requests,
                )
            )
            if cell_ok:
                closest = min(
                    cell_ok, key=lambda plan: (plan.requests, plan.unique_cells)
                )
                reason += "; that plan returns {} cell(s)".format(closest.unique_cells)
            conflict = CONFLICT_REQUESTS
        elif not cell_ok:
            reason = (
                "max_cells={} is too small: the requested legal regions "
                "alone cover {} cell(s), and the smallest possible return "
                "is {} cell(s)".format(self.max_cells, requested_cells, min_cells)
            )
            conflict = CONFLICT_CELLS
        else:
            fewest_requests = min(
                candidates, key=lambda plan: (plan.requests, plan.unique_cells)
            )
            fewest_cells = min(
                candidates, key=lambda plan: (plan.unique_cells, plan.requests)
            )
            reason = (
                "the request and cell budgets cannot be met together: the "
                "fewest-request plan uses {} request(s) but returns {} "
                "cell(s) (max_cells={}), while the smallest-return plan "
                "returns {} cell(s) but needs {} request(s) "
                "(max_requests={})".format(
                    fewest_requests.requests,
                    fewest_requests.unique_cells,
                    self.max_cells,
                    fewest_cells.unique_cells,
                    fewest_cells.requests,
                    self.max_requests,
                )
            )
            conflict = CONFLICT_BOTH

        if clipped:
            reason += (
                "; {} requested range(s) were clipped around forbidden "
                "regions: {}".format(
                    len(clipped),
                    ", ".join(
                        "{} removed {} cell(s)".format(item.range, item.removed_cells)
                        for item in clipped
                    ),
                )
            )
        if fully_forbidden:
            reason += "; entirely forbidden: {}".format(
                ", ".join(item.range for item in fully_forbidden)
            )
        return RangePlanError(reason, conflicted=conflict, best_plan=best)


def plan_ranges(
    requested: Iterable[GridRegion],
    max_requests: int,
    max_cells: int,
    forbidden: Iterable[GridRegion] = (),
    row_count: Optional[int] = None,
    column_count: Optional[int] = None,
    sheet_title: Optional[str] = None,
    allow_partial: bool = False,
    objective: str = "requests",
) -> RangePlan:
    """Plan ranges in one call; see :class:`RangePlanner`."""
    return RangePlanner(
        max_requests=max_requests,
        max_cells=max_cells,
        forbidden=forbidden,
        row_count=row_count,
        column_count=column_count,
        sheet_title=sheet_title,
        allow_partial=allow_partial,
        objective=objective,
    ).plan(requested)


# -- rectangle geometry --------------------------------------------------


def _render_rect(rect: Rect) -> str:
    r0, r1, c0, c1 = rect
    return "{}:{}".format(rowcol_to_a1(r0 + 1, c0 + 1), rowcol_to_a1(r1, c1))


def _area(rect: Rect) -> int:
    return max(0, rect[1] - rect[0]) * max(0, rect[3] - rect[2])


def _is_empty(rect: Rect) -> bool:
    return rect[1] <= rect[0] or rect[3] <= rect[2]


def _intersects(a: Rect, b: Rect) -> bool:
    return a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]


def _contains(outer: Rect, inner: Rect) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] >= inner[1]
        and outer[2] <= inner[2]
        and outer[3] >= inner[3]
    )


def _intersection(a: Rect, b: Rect) -> Optional[Rect]:
    if not _intersects(a, b):
        return None
    return (max(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), min(a[3], b[3]))


def _subtract(rect: Rect, cut: Rect) -> List[Rect]:
    """Split ``rect`` around ``cut``, returning the remaining pieces."""
    overlap = _intersection(rect, cut)
    if overlap is None:
        return [rect]

    r0, r1, c0, c1 = rect
    or0, or1, oc0, oc1 = overlap
    pieces: List[Rect] = []
    if r0 < or0:
        pieces.append((r0, or0, c0, c1))
    if or1 < r1:
        pieces.append((or1, r1, c0, c1))
    if c0 < oc0:
        pieces.append((or0, or1, c0, oc0))
    if oc1 < c1:
        pieces.append((or0, or1, oc1, c1))
    return [piece for piece in pieces if not _is_empty(piece)]


def _bounding_pair(a: Rect, b: Rect) -> Optional[Rect]:
    if a == b:
        return None
    return (
        min(a[0], b[0]),
        max(a[1], b[1]),
        min(a[2], b[2]),
        max(a[3], b[3]),
    )


def _bounding_rect(rects: Sequence[Rect]) -> Optional[Rect]:
    if not rects:
        return None
    return (
        min(rect[0] for rect in rects),
        max(rect[1] for rect in rects),
        min(rect[2] for rect in rects),
        max(rect[3] for rect in rects),
    )


def _union_area(rects: Sequence[Rect]) -> int:
    """Exact area of the union, without materialising individual cells."""
    rects = [rect for rect in rects if not _is_empty(rect)]
    if not rects:
        return 0
    x_edges = sorted({rect[0] for rect in rects} | {rect[1] for rect in rects})
    total = 0
    for left, right in zip(x_edges, x_edges[1:]):
        intervals = sorted(
            (rect[2], rect[3]) for rect in rects if rect[0] <= left and rect[1] >= right
        )
        covered = 0
        current_start: Optional[int] = None
        current_end: Optional[int] = None
        for start, end in intervals:
            if current_start is None:
                current_start, current_end = start, end
            elif start <= current_end:
                current_end = max(current_end, end)
            else:
                covered += current_end - current_start
                current_start, current_end = start, end
        if current_start is not None:
            covered += current_end - current_start
        total += covered * (right - left)
    return total


def _violations(plan: RangePlan, planner: RangePlanner) -> Tuple[int, int, int]:
    """Sort key for the diagnostic best plan: fewer violated budgets first."""
    violated = int(plan.requests > planner.max_requests) + int(
        plan.unique_cells > planner.max_cells
    )
    return (violated, plan.requests, plan.unique_cells)


def _pareto(plans: Sequence[RangePlan]) -> List[RangePlan]:
    """Drop plans dominated on both requests and unique cells."""
    unique: Dict[Tuple[int, int], RangePlan] = {}
    for plan in plans:
        key = (plan.requests, plan.unique_cells)
        unique.setdefault(key, plan)

    ordered = sorted(
        unique.values(), key=lambda plan: (plan.requests, plan.unique_cells)
    )
    efficient: List[RangePlan] = []
    best_cells: Optional[int] = None
    for plan in ordered:
        if best_cells is None or plan.unique_cells < best_cells:
            efficient.append(plan)
            best_cells = plan.unique_cells
    return efficient
