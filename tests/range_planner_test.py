from gspread.range_planner import (
    CONFLICT_BOTH,
    CONFLICT_CELLS,
    CONFLICT_FORBIDDEN,
    CONFLICT_REQUESTS,
    GridRegion,
    InvalidRangePlanError,
    RangePlanError,
    RangePlanner,
    plan_ranges,
)


def region(a1):
    return GridRegion.from_a1(a1)


def test_from_a1_and_back():
    assert region("A1:C2").to_a1() == "A1:C2"
    assert region("'Sheet1'!D4").to_a1() == "D4:D4"
    assert region("A1").to_a1() == "A1:A1"


def test_from_a1_unbounded_needs_sheet_size():
    try:
        GridRegion.from_a1("A5:B")
    except InvalidRangePlanError as exc:
        assert "row_count" in str(exc) or "column_count" in str(exc)
    else:
        raise AssertionError("expected InvalidRangePlanError")

    resolved = GridRegion.from_a1("A5:B", row_count=20, column_count=5)
    assert resolved.to_a1(row_count=20, column_count=5) == "A5:B20"


def test_invalid_bounds():
    for kwargs in (
        {"start_row": -1, "end_row": 2},
        {"start_row": 4, "end_row": 2},
        {"start_column": 5, "end_column": 3},
    ):
        try:
            GridRegion(**kwargs)
        except InvalidRangePlanError:
            pass
        else:
            raise AssertionError("expected InvalidRangePlanError for %r" % kwargs)


def test_tight_cell_budget_forces_split_of_far_ranges():
    requested = [region("A1:C2"), region("A200:C201"), region("Z100:AB120")]
    plan = plan_ranges(requested, max_requests=3, max_cells=100)

    assert plan.requests == 3
    assert plan.unique_cells == 75
    assert plan.requested_cells == 75
    assert plan.overhead_cells == 0
    assert sorted(plan.a1_ranges()) == ["A1:C2", "A200:C201", "Z100:AB120"]


def test_loose_cell_budget_uses_fewest_requests_by_default():
    requested = [region("A1:C2"), region("A200:C201"), region("Z100:AB120")]
    plan = plan_ranges(requested, max_requests=3, max_cells=10_000)

    assert plan.requests == 1
    assert plan.unique_cells == 5628
    assert plan.overhead_cells == 5553


def test_overhead_objective_avoids_the_wide_block():
    requested = [region("A1:C2"), region("A200:C201"), region("Z100:AB120")]
    plan = plan_ranges(
        requested, max_requests=3, max_cells=10_000, objective="overhead"
    )

    assert plan.requests == 3
    assert plan.unique_cells == 75
    assert plan.overhead_cells == 0
    assert plan.alternatives


def test_alternatives_expose_request_cell_tradeoff():
    requested = [region("A1:C2"), region("A200:C201"), region("Z100:AB120")]
    plan = plan_ranges(requested, max_requests=3, max_cells=10_000)

    tradeoff = [(item.requests, item.unique_cells) for item in plan.alternatives]
    assert tradeoff == [(2, 666), (3, 75)]
    assert tradeoff[0][1] > tradeoff[-1][1]


def test_requests_objective_chooses_the_single_block():
    requested = [region("A1:C2"), region("A200:C201"), region("Z100:AB120")]
    plan = plan_ranges(
        requested, max_requests=3, max_cells=10_000, objective="requests"
    )

    assert plan.requests == 1
    assert plan.unique_cells == 5628
    assert plan.overhead_cells == 5553


def test_shared_cells_counted_once():
    plan = plan_ranges(
        [region("A1:D4"), region("C3:F6")],
        max_requests=2,
        max_cells=100,
        objective="overhead",
    )

    assert plan.unique_cells == 28
    assert plan.requested_cells == 28
    assert [(item.range, item.new_cells) for item in plan.ranges] == [
        ("A1:D4", 16),
        ("C3:F6", 12),
    ]


def test_forbidden_region_blocks_the_shortcut():
    requested = [region("A1:C2"), region("A200:C201"), region("Z100:AB120")]
    plan = plan_ranges(
        requested,
        max_requests=3,
        max_cells=10_000,
        forbidden=[region("A3:C199")],
    )

    # The wide A1:AB201 block would cross the forbidden block, so the
    # planner must keep three separate reads rather than over-read it.
    assert plan.requests == 3
    assert plan.unique_cells == 75
    assert plan.a1_ranges() == ["A1:C2", "Z100:AB120", "A200:C201"]


def test_request_conflict_names_the_binding_limit():
    requested = [region("A1:C2"), region("A200:C201"), region("Z100:AB120")]
    try:
        plan_ranges(
            requested,
            max_requests=1,
            max_cells=10_000,
            forbidden=[region("A3:C199")],
        )
    except RangePlanError as exc:
        assert exc.conflicted == CONFLICT_REQUESTS
        assert "max_requests=1" in exc.reason
        assert "fewest legal requests is 3" in exc.reason
        assert exc.best_plan is not None
        assert exc.best_plan.requests == 3
    else:
        raise AssertionError("expected RangePlanError")


def test_cell_conflict_names_the_binding_limit():
    try:
        plan_ranges([region("A1:C10")], max_requests=1, max_cells=10)
    except RangePlanError as exc:
        assert exc.conflicted == CONFLICT_CELLS
        assert "max_cells=10" in exc.reason
        assert "30 cell(s)" in exc.reason
    else:
        raise AssertionError("expected RangePlanError")


def test_clipped_range_reports_conflicting_request_budget():
    # A1:F10 has to be split around D1:D10, so one request is impossible.
    try:
        plan_ranges(
            [region("A1:F10")],
            max_requests=1,
            max_cells=1_000,
            forbidden=[region("D1:D10")],
        )
    except RangePlanError as exc:
        assert exc.conflicted == CONFLICT_REQUESTS
        assert "A1:F10" in exc.reason
        assert "removed 10 cell(s)" in exc.reason
        assert exc.best_plan is not None
        assert exc.best_plan.a1_ranges() == ["A1:C10", "E1:F10"]
    else:
        raise AssertionError("expected RangePlanError")


def test_fully_forbidden_request_raises():
    try:
        plan_ranges(
            [region("D1:D10")],
            max_requests=1,
            max_cells=1_000,
            forbidden=[region("D1:D10")],
        )
    except RangePlanError as exc:
        assert exc.conflicted == CONFLICT_FORBIDDEN
        assert "D1:D10" in exc.reason
    else:
        raise AssertionError("expected RangePlanError")


def test_partial_range_is_clipped_and_kept():
    plan = plan_ranges(
        [region("A1:F10")],
        max_requests=2,
        max_cells=1_000,
        forbidden=[region("D1:D10")],
    )

    assert plan.a1_ranges() == ["A1:C10", "E1:F10"]
    assert plan.unique_cells == 50
    assert plan.forbidden_cells_removed == 10
    assert plan.clipped[0].range == "A1:F10"
    assert plan.clipped[0].kept == ("A1:C10", "E1:F10")


def test_allow_partial_reports_fully_forbidden_request():
    plan = plan_ranges(
        [region("A1:C10"), region("D1:D10")],
        max_requests=1,
        max_cells=1_000,
        forbidden=[region("D1:D10")],
        allow_partial=True,
    )

    assert plan.a1_ranges() == ["A1:C10"]
    assert plan.unique_cells == 30
    assert [item.range for item in plan.unsatisfied] == ["D1:D10"]


def test_empty_request_list_is_an_error():
    try:
        plan_ranges([], max_requests=1, max_cells=1)
    except RangePlanError as exc:
        assert exc.conflicted == "input"
    else:
        raise AssertionError("expected RangePlanError")


def test_invalid_objective_rejected():
    try:
        RangePlanner(max_requests=1, max_cells=1, objective="cell-count")
    except InvalidRangePlanError as exc:
        assert "objective" in str(exc)
    else:
        raise AssertionError("expected InvalidRangePlanError")


def test_planner_is_reusable():
    planner = RangePlanner(max_requests=2, max_cells=100)
    first = planner.plan([region("A1:B2"), region("D1:E2")])
    second = planner.plan([region("A1:B2"), region("D1:E2")])
    assert first.a1_ranges() == second.a1_ranges()


def test_exact_search_finds_split_blocked_by_forbidden():
    # Three regions in a column, with forbidden rows between them.
    requested = [
        region("A1:A2"),
        region("A10:A11"),
        region("A20:A21"),
    ]
    plan = plan_ranges(
        requested,
        max_requests=3,
        max_cells=100,
        forbidden=[region("A3:A9"), region("A12:A19")],
    )
    assert plan.a1_ranges() == ["A1:A2", "A10:A11", "A20:A21"]
    assert plan.unique_cells == 6


def _fake_sheet(title="Sheet1", rows=1000, columns=26):
    from gspread.worksheet import Worksheet

    sheet = Worksheet.__new__(Worksheet)
    sheet._properties = {
        "sheetId": 0,
        "title": title,
        "index": 0,
        "gridProperties": {"rowCount": rows, "columnCount": columns},
    }
    return sheet


def test_worksheet_plan_ranges_uses_sheet_bounds():
    sheet = _fake_sheet(rows=300, columns=10)

    plan = sheet.plan_ranges(
        ["A1:B2", "A200:B201", "H290:J300"],
        max_requests=3,
        max_cells=500,
        forbidden=["A3:B199"],
    )

    assert sorted(plan.a1_ranges()) == ["A1:B2", "A200:B201", "H290:J300"]
    assert plan.unique_cells == 4 + 33 + 4


def test_worksheet_plan_ranges_resolves_unbounded():
    sheet = _fake_sheet(rows=50, columns=5)

    plan = sheet.plan_ranges(["A5:B"], max_requests=1, max_cells=1000)
    assert plan.a1_ranges() == ["A5:B50"]
    assert plan.unique_cells == 46 * 2


def test_worksheet_plan_ranges_surfaces_conflict():
    sheet = _fake_sheet(rows=300, columns=10)

    try:
        sheet.plan_ranges(
            ["A1:B2", "A200:B201", "H290:J300"],
            max_requests=1,
            max_cells=10_000,
            forbidden=["A3:B199", "C1:G289"],
        )
    except RangePlanError as exc:
        assert exc.conflicted == CONFLICT_REQUESTS
    else:
        raise AssertionError("expected RangePlanError")


def test_batched_request_count_is_one_per_batch():
    plan = plan_ranges(
        [region("A1:B2"), region("D1:E2")],
        max_requests=2,
        max_cells=8,
        objective="overhead",
    )
    assert plan.requests == 2
    assert plan.batched_requests == 1


def test_both_budgets_conflict_and_both_are_named():
    try:
        plan_ranges(
            [region("A1:A2"), region("Z1:Z2")],
            max_requests=1,
            max_cells=4,
        )
    except RangePlanError as exc:
        assert exc.conflicted == CONFLICT_BOTH
        assert "max_requests=1" in exc.reason
        assert "max_cells=4" in exc.reason
    else:
        raise AssertionError("expected RangePlanError")


def test_no_planned_range_ever_touches_forbidden_cells():
    import random

    rng = random.Random(7)
    for _ in range(60):
        requested = []
        for _ in range(rng.randint(1, 4)):
            r0 = rng.randint(0, 30)
            c0 = rng.randint(0, 10)
            requested.append(
                GridRegion(
                    start_row=r0,
                    end_row=r0 + rng.randint(1, 8),
                    start_column=c0,
                    end_column=c0 + rng.randint(1, 5),
                )
            )
        forbidden = []
        for _ in range(rng.randint(0, 3)):
            r0 = rng.randint(0, 30)
            c0 = rng.randint(0, 10)
            forbidden.append(
                GridRegion(
                    start_row=r0,
                    end_row=r0 + rng.randint(1, 8),
                    start_column=c0,
                    end_column=c0 + rng.randint(1, 5),
                )
            )

        try:
            plan = plan_ranges(
                requested,
                max_requests=10,
                max_cells=100_000,
                forbidden=forbidden,
                allow_partial=True,
            )
        except RangePlanError:
            continue

        for item in plan.ranges:
            r0, r1, c0, c1 = item.rect
            for cut in forbidden:
                fr0, fr1, fc0, fc1 = cut.resolve()
                assert not (r0 < fr1 and fr0 < r1 and c0 < fc1 and fc0 < c1), (
                    item.range,
                    cut,
                )
