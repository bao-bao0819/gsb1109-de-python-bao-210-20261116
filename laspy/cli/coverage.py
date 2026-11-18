"""``laspy coverage`` commands: missed-scan review lists and strip deltas."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import typer
from rich.console import Console
from rich.table import Table

from laspy.coverage import (
    CoverageComparison,
    CoverageError,
    CoverageResult,
    GridSpec,
    analyze_coverage,
    compare_coverage,
)

app = typer.Typer(help="Build missed-scan review lists from LAS/LAZ strips")
console = Console()
err_console = Console(stderr=True)

MULTI_RETURN_HELP = (
    "Which returns are retained as evidence: all, first, last, single or "
    "first_and_last. Raw pulse support always uses every return."
)


def _parse_floats(value: str) -> List[float]:
    try:
        return [float(item) for item in value.replace(",", " ").split()]
    except ValueError:
        raise typer.BadParameter(
            f"'{value}' is not a list of numbers"
        ) from None


def _parse_bounds(value: Optional[str]) -> Optional[List[float]]:
    if value is None:
        return None
    numbers = _parse_floats(value)
    if len(numbers) != 4:
        raise typer.BadParameter("bounds must contain x_min y_min x_max y_max")
    return numbers


def _grid_from_options(
    cell_size: float,
    grid_bounds: Optional[str],
    coverage_bounds: Optional[str],
) -> Optional[GridSpec]:
    bounds = _parse_bounds(grid_bounds) or _parse_bounds(coverage_bounds)
    if cell_size <= 0:
        raise typer.BadParameter("--cell-size must be positive")
    if bounds is not None:
        try:
            return GridSpec.from_bounds(bounds, cell_size)
        except CoverageError as error:
            raise typer.BadParameter(str(error)) from None
    return None


def _known_coverage(
    bounds: Optional[str], mask_path: Optional[Path]
) -> Any:
    if mask_path is not None:
        return mask_path
    return _parse_bounds(bounds)


def _write_output(
    result: Union[CoverageResult, CoverageComparison],
    output: Optional[Path],
    output_format: Optional[str],
) -> None:
    if output is None:
        return
    suffix = output.suffix.lower()
    fmt = output_format or {
        ".json": "json",
        ".csv": "csv",
    }.get(suffix, "json")
    if fmt == "csv":
        result.to_csv(output)
    elif fmt == "json":
        result.to_json(output)
    else:
        raise typer.BadParameter(
            f"unknown output format {fmt!r}; use json or csv"
        )


def _print_summary(result: CoverageResult) -> None:
    summary = result.summary()
    pulses = summary["pulses"]
    points = summary["points"]
    cells = summary["cells"]

    table = Table(title="Coverage summary", show_header=False, box=None)
    table.add_row("Grid", f"{result.grid.width} x {result.grid.height} cells")
    table.add_row("Cell size", f"{result.grid.cell_size:g}")
    table.add_row("Pulse identity", ", ".join(pulses["identity_methods"]) or "none")
    table.add_row("Input pulses", str(pulses["input"]))
    table.add_row("Input points", str(points["input"]))
    table.add_row("Raw pulse support cells", str(pulses["raw_pulse_support_cells"]))
    table.add_row("Retained pulses", str(pulses["retained"]))
    table.add_row("Retained points", str(points["retained"]))
    table.add_row("Filtered-out points", str(points["filtered_out"]))
    table.add_row("Invalid pulse identity", str(points["invalid_pulse_identity"]))
    table.add_row("Outside grid", str(points["outside_grid"]))
    table.add_row("Covered cells", str(cells["covered"]))
    table.add_row("Gap cells", str(cells["gap"]))
    table.add_row("Low-density cells", str(cells["low_density"]))
    table.add_row("Unconfirmed cells", str(cells["unconfirmed"]))
    console.print(table)


def _print_review(result: CoverageResult, limit: Optional[int]) -> None:
    cells = result.review_list()
    if limit is not None:
        cells = cells[:limit]

    table = Table(title=f"Review list ({len(result.review_list())} candidates)")
    table.add_column("Cell")
    table.add_column("Status")
    table.add_column("Center X", justify="right")
    table.add_column("Center Y", justify="right")
    table.add_column("Raw pulses", justify="right")
    table.add_column("Raw points", justify="right")
    table.add_column("Retained pulses", justify="right")
    table.add_column("Retained points", justify="right")
    table.add_column("Reason")
    for cell in cells:
        center_x, center_y = cell.center
        table.add_row(
            cell.cell_id,
            cell.status,
            f"{center_x:.3f}",
            f"{center_y:.3f}",
            str(cell.raw_pulse_count),
            str(cell.raw_point_count),
            str(cell.retained_pulse_count),
            str(cell.retained_point_count),
            cell.reason,
        )
    console.print(table)


def _print_comparison(comparison: CoverageComparison) -> None:
    delta = comparison.summary()["delta"]
    table = Table(title="Coverage delta", show_header=False, box=None)
    table.add_row("Newly covered cells", str(delta["newly_covered_cells"]))
    table.add_row("Repeat observation cells", str(delta["repeat_observation_cells"]))
    table.add_row("Remaining candidates", str(delta["remaining_candidate_cells"]))
    table.add_row("New coverage points", str(delta["new_coverage_points"]))
    table.add_row("New coverage pulses", str(delta["new_coverage_pulses"]))
    table.add_row(
        "Repeat observation points", str(delta["repeat_observation_points"])
    )
    table.add_row(
        "Repeat observation pulses", str(delta["repeat_observation_pulses"])
    )
    console.print(table)


def _common_analysis_kwargs(
    *,
    cell_size: float,
    grid_bounds: Optional[str],
    coverage_bounds: Optional[str],
    coverage_mask: Optional[Path],
    filter_expression: Optional[str],
    multi_return: str,
    min_pulses: int,
    min_points: int,
    min_density: Optional[float],
    pulse_key: Optional[str],
    gps_time_tolerance: float,
) -> Dict[str, Any]:
    grid = _grid_from_options(cell_size, grid_bounds, coverage_bounds)
    kwargs: Dict[str, Any] = {
        "filter_condition": filter_expression,
        "multi_return": multi_return,
        "min_pulses": min_pulses,
        "min_points": min_points,
        "min_density": min_density,
        "known_coverage": _known_coverage(coverage_bounds, coverage_mask),
        "gps_time_tolerance": gps_time_tolerance,
    }
    if grid is not None:
        kwargs["grid"] = grid
    else:
        kwargs["cell_size"] = cell_size
    if pulse_key is not None:
        kwargs["pulse_key"] = pulse_key
    return kwargs


@app.command("review")
def review(
    input_paths: List[Path] = typer.Argument(
        ..., help="LAS/LAZ file(s) covering the survey"
    ),
    cell_size: float = typer.Option(..., help="Grid cell size"),
    grid_bounds: Optional[str] = typer.Option(
        None, help="Grid extent: 'x_min y_min x_max y_max'"
    ),
    coverage_bounds: Optional[str] = typer.Option(
        None, help="Known coverage extent: 'x_min y_min x_max y_max'"
    ),
    coverage_mask: Optional[Path] = typer.Option(
        None, help="Known coverage mask (.npy, GeoJSON, or CSV)"
    ),
    filter_expression: Optional[str] = typer.Option(
        None, "--filter", help="Point filter expression"
    ),
    multi_return: str = typer.Option("all", help=MULTI_RETURN_HELP),
    min_pulses: int = typer.Option(1, help="Minimum retained pulses per cell"),
    min_points: int = typer.Option(1, help="Minimum retained points per cell"),
    min_density: Optional[float] = typer.Option(
        None, help="Minimum retained points per unit area (e.g. points/m2)"
    ),
    threshold: Optional[int] = typer.Option(
        None, help="Alias for --min-pulses"
    ),
    pulse_key: Optional[str] = typer.Option(
        None, help="Field(s) identifying one emitted pulse"
    ),
    gps_time_tolerance: float = typer.Option(
        1e-6, help="Tolerance used when matching gps_time values"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Write the review list to .json or .csv"
    ),
    output_format: Optional[str] = typer.Option(
        None, help="Output format: json or csv (default: by extension)"
    ),
    limit: Optional[int] = typer.Option(
        None, help="Only print the first N review rows"
    ),
):
    """Build a missed-scan review list.

    Raw pulse support, retained points and unconfirmed cells are reported
    separately.  Several returns of one pulse count as one observation.

    Examples:

        laspy coverage review strips/*.las --cell-size 1.0 \
            --filter "classification in [2, 6]" --min-pulses 2 \
            --output review.csv
    """
    if threshold is not None:
        min_pulses = threshold

    kwargs = _common_analysis_kwargs(
        cell_size=cell_size,
        grid_bounds=grid_bounds,
        coverage_bounds=coverage_bounds,
        coverage_mask=coverage_mask,
        filter_expression=filter_expression,
        multi_return=multi_return,
        min_pulses=min_pulses,
        min_points=min_points,
        min_density=min_density,
        pulse_key=pulse_key,
        gps_time_tolerance=gps_time_tolerance,
    )
    try:
        result = analyze_coverage([str(path) for path in input_paths], **kwargs)
    except (CoverageError, ValueError) as error:
        err_console.print(f"[bold red]Error:[/] {error}")
        raise typer.Exit(code=1) from None

    _print_summary(result)
    _print_review(result, limit)

    if output is not None:
        _write_output(result, output, output_format)
        console.print(f"Wrote {output}")


@app.command("compare")
def compare(
    baseline_paths: List[Path] = typer.Argument(
        ..., help="Baseline LAS/LAZ file(s)"
    ),
    added_paths: List[Path] = typer.Option(
        ..., "--added", help="LAS/LAZ file(s) of the added strip(s)"
    ),
    combined_paths: Optional[List[Path]] = typer.Option(
        None, "--combined", help="Optional LAS/LAZ file(s) of baseline plus added"
    ),
    cell_size: float = typer.Option(..., help="Grid cell size"),
    grid_bounds: Optional[str] = typer.Option(
        None, help="Grid extent: 'x_min y_min x_max y_max'"
    ),
    coverage_bounds: Optional[str] = typer.Option(
        None, help="Known coverage extent: 'x_min y_min x_max y_max'"
    ),
    coverage_mask: Optional[Path] = typer.Option(
        None, help="Known coverage mask (.npy, GeoJSON, or CSV)"
    ),
    filter_expression: Optional[str] = typer.Option(
        None, "--filter", help="Point filter expression"
    ),
    multi_return: str = typer.Option("all", help=MULTI_RETURN_HELP),
    min_pulses: int = typer.Option(1, help="Minimum retained pulses per cell"),
    min_points: int = typer.Option(1, help="Minimum retained points per cell"),
    min_density: Optional[float] = typer.Option(
        None, help="Minimum retained points per unit area (e.g. points/m2)"
    ),
    threshold: Optional[int] = typer.Option(
        None, help="Alias for --min-pulses"
    ),
    pulse_key: Optional[str] = typer.Option(
        None, help="Field(s) identifying one emitted pulse"
    ),
    gps_time_tolerance: float = typer.Option(
        1e-6, help="Tolerance used when matching gps_time values"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Write the delta list to .json or .csv"
    ),
    output_format: Optional[str] = typer.Option(
        None, help="Output format: json or csv (default: by extension)"
    ),
    limit: Optional[int] = typer.Option(
        None, help="Only print the first N delta rows"
    ),
):
    """Compare a baseline survey with the survey after adding strip(s).

    Cells that become covered are reported as new coverage.  Retained
    observations of the added strip in already covered cells are repeat
    observations.

    Examples:

        laspy coverage compare baseline/*.las --added new_strip.las \
            --cell-size 1.0 --output delta.json
    """
    if threshold is not None:
        min_pulses = threshold

    kwargs = _common_analysis_kwargs(
        cell_size=cell_size,
        grid_bounds=grid_bounds,
        coverage_bounds=coverage_bounds,
        coverage_mask=coverage_mask,
        filter_expression=filter_expression,
        multi_return=multi_return,
        min_pulses=min_pulses,
        min_points=min_points,
        min_density=min_density,
        pulse_key=pulse_key,
        gps_time_tolerance=gps_time_tolerance,
    )
    try:
        comparison = compare_coverage(
            [str(path) for path in baseline_paths],
            [str(path) for path in added_paths],
            combined=(
                [str(path) for path in combined_paths]
                if combined_paths is not None
                else None
            ),
            **kwargs,
        )
    except (CoverageError, ValueError) as error:
        err_console.print(f"[bold red]Error:[/] {error}")
        raise typer.Exit(code=1) from None

    _print_comparison(comparison)

    rows = comparison.cells()
    if limit is not None:
        rows = rows[:limit]
    if rows:
        table = Table(title=f"Delta cells ({len(comparison.cells())})")
        table.add_column("Cell")
        table.add_column("Class")
        table.add_column("Center X", justify="right")
        table.add_column("Center Y", justify="right")
        table.add_column("Baseline")
        table.add_column("Combined")
        table.add_column("Added points", justify="right")
        table.add_column("Added pulses", justify="right")
        for row in rows:
            table.add_row(
                row["cell_id"],
                row["classification"],
                f"{row['center_x']:.3f}",
                f"{row['center_y']:.3f}",
                row["baseline_status"],
                row["combined_status"],
                str(row["added_point_count"]),
                str(row["added_pulse_count"]),
            )
        console.print(table)
    else:
        console.print("No newly covered or repeat observation cells.")

    if output is not None:
        _write_output(comparison, output, output_format)
        console.print(f"Wrote {output}")


__all__ = ["app"]
