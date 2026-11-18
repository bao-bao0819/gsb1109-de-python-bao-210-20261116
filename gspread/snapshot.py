"""
gspread.snapshot
~~~~~~~~~~~~~~~~

Helpers for reading a worksheet in several requests while keeping
verifiable evidence of what was read and against which version of the
sheet.

The Google Sheets API gives no snapshot isolation between separate
``values.get`` calls: another editor can insert rows while a batch of
reads is in progress. Stitching the responses together can then produce
a matrix that *looks* complete but silently skips (or duplicates) a row.

The objects in this module never claim completeness on their own. They
record, for every read:

* the A1 range that was requested and the range the server answered
  with (its server-side row coordinates);
* the worksheet version evidence available at read time
  (``gridProperties.rowCount`` from ``spreadsheets.get``);
* the result of every consistency check performed on the evidence.

A batch is only marked :data:`ReadSnapshotStatus.CONSISTENT` when the
version evidence proves nothing changed between the first and last read
and every answered range matches the request. Otherwise the batch is
marked :data:`ReadSnapshotStatus.INCONSISTENT` (or
:data:`ReadSnapshotStatus.UNKNOWN`) and the row regions in which rows
may have been missed or duplicated are reported so they can be re-read.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from .exceptions import GSpreadException
from .utils import (
    a1_range_to_grid_range,
    absolute_range_name,
    finditem,
    get_a1_from_absolute_range,
    is_full_a1_notation,
)

if TYPE_CHECKING is True:
    from .worksheet import Worksheet


class ReadSnapshotStatus(str, Enum):
    """Outcome of the consistency checks run on a read batch.

    * ``UNKNOWN``: no version evidence could be collected (e.g. the
      metadata call failed). Completeness cannot be assessed.
    * ``CONSISTENT``: the available evidence is sufficient to prove the
      reads form a single consistent snapshot of the covered ranges.
    * ``INCONSISTENT``: the evidence shows the worksheet changed (or the
      answered ranges drifted) between reads; the stitched matrix must
      not be trusted as complete.
    """

    UNKNOWN = "UNKNOWN"
    CONSISTENT = "CONSISTENT"
    INCONSISTENT = "INCONSISTENT"


@dataclass
class RangeReadEvidence:
    """Evidence of a single read request inside a batch.

    :ivar str requested_range: A1 range sent to the API (absolute form,
        e.g. ``"'Sheet1'!A1:C100"``).
    :ivar str response_range: range echoed by the server in the
        ``range`` field of the response, expressed in server-side row
        coordinates at the moment of the read.
    :ivar int requested_start_row: one-based first row requested.
    :ivar int requested_end_row: one-based last row requested.
    :ivar Optional[int] response_start_row: one-based first row the
        server answered, or ``None`` when the response carried no range.
    :ivar Optional[int] response_end_row: one-based last row the server
        answered, or ``None`` when the response carried no range.
    :ivar Optional[int] row_count_before: worksheet ``rowCount`` fetched
        immediately before the read, if available.
    :ivar Optional[int] row_count_after: worksheet ``rowCount`` fetched
        immediately after the read, if available.
    :ivar bool range_matches: whether the answered range covers exactly
        the requested rows.
    :ivar list issues: human-readable reasons why this read cannot be
        considered stable.
    """

    requested_range: str
    response_range: str
    requested_start_row: int
    requested_end_row: int
    response_start_row: Optional[int] = None
    response_end_row: Optional[int] = None
    row_count_before: Optional[int] = None
    row_count_after: Optional[int] = None
    range_matches: bool = False
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain ``dict`` suitable for persisting the evidence."""
        return {
            "requested_range": self.requested_range,
            "response_range": self.response_range,
            "requested_start_row": self.requested_start_row,
            "requested_end_row": self.requested_end_row,
            "response_start_row": self.response_start_row,
            "response_end_row": self.response_end_row,
            "row_count_before": self.row_count_before,
            "row_count_after": self.row_count_after,
            "range_matches": self.range_matches,
            "issues": list(self.issues),
        }


@dataclass
class SuspectRegion:
    """A row interval that must be re-read because of a detected change.

    Rows are one-based and the interval is closed:
    ``[first_row, last_row]``.

    :ivar int first_row: first row of the suspect interval.
    :ivar int last_row: last row of the suspect interval.
    :ivar str reason: why the interval is suspect.
    """

    first_row: int
    last_row: int
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "first_row": self.first_row,
            "last_row": self.last_row,
            "reason": self.reason,
        }


@dataclass
class ReadSnapshotReport:
    """Verifiable evidence and verdict for a whole read batch.

    :ivar ReadSnapshotStatus status: verdict of the consistency checks.
    :ivar list reads: one :class:`RangeReadEvidence` per request.
    :ivar list suspect_regions: row intervals that may contain missed or
        duplicated rows.
    :ivar list notes: global findings (e.g. missing version evidence).
    :ivar dict version_evidence: the version markers observed for the
        batch (row counts before/after, plus the raw worksheet
        properties snapshots when available).
    """

    status: ReadSnapshotStatus = ReadSnapshotStatus.UNKNOWN
    reads: List[RangeReadEvidence] = field(default_factory=list)
    suspect_regions: List[SuspectRegion] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    version_evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_consistent(self) -> bool:
        """``True`` only when the batch is proven consistent."""
        return self.status is ReadSnapshotStatus.CONSISTENT

    def raise_if_not_consistent(self) -> None:
        """Raise :class:`~gspread.exceptions.GSpreadException` unless consistent.

        A consistent snapshot returns silently. Both ``INCONSISTENT`` and
        ``UNKNOWN`` batches raise with the evidence attached so callers
        never treat an unverified read as complete by mistake.
        """
        if self.status is not ReadSnapshotStatus.CONSISTENT:
            raise GSpreadException(
                "worksheet read batch is {} and must not be treated as a "
                "complete snapshot: {}".format(
                    self.status.value, self.notes or "no consistency evidence"
                )
            )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation of the whole report."""
        return {
            "status": self.status.value,
            "is_consistent": self.is_consistent,
            "reads": [read.to_dict() for read in self.reads],
            "suspect_regions": [region.to_dict() for region in self.suspect_regions],
            "notes": list(self.notes),
            "version_evidence": _json_safe(self.version_evidence),
        }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _row_bounds(range_name: str) -> Tuple[int, int]:
    a1 = get_a1_from_absolute_range(range_name)
    grid_range = a1_range_to_grid_range(a1)
    start_index = grid_range.get("startRowIndex")
    end_index = grid_range.get("endRowIndex")
    if start_index is None or end_index is None:
        raise GSpreadException(
            "only full A1 ranges with explicit row bounds can be recorded "
            "as snapshot evidence, got: {}".format(range_name)
        )
    return start_index + 1, end_index


def _response_row_bounds(
    response: Mapping[str, Any],
) -> Tuple[Optional[int], Optional[int], str]:
    response_range = response.get("range", "")
    if not response_range:
        return None, None, response_range
    a1 = get_a1_from_absolute_range(response_range)
    if not is_full_a1_notation(a1):
        return None, None, response_range
    grid_range = a1_range_to_grid_range(a1)
    start_index = grid_range.get("startRowIndex")
    end_index = grid_range.get("endRowIndex")
    if start_index is None or end_index is None:
        return None, None, response_range
    return start_index + 1, end_index, response_range


def _worksheet_properties(worksheet: "Worksheet") -> Mapping[str, Any]:
    metadata = worksheet.client.fetch_sheet_metadata(worksheet.spreadsheet_id)
    return finditem(
        lambda sheet: sheet["properties"]["sheetId"] == worksheet.id,
        metadata["sheets"],
    )["properties"]


class SnapshotReadBatch:
    """Accumulator for a multi-request read of one worksheet.

    The batch wraps a fixed target worksheet and records evidence on
    every :meth:`read` call. Call :meth:`finalize` once all reads are
    done to obtain the :class:`ReadSnapshotReport` with the verdict and,
    when inconsistent, the row regions to re-read.

    The Google Sheets API offers no transactional snapshot guarantee, so
    a batch can never upgrade ``INCONSISTENT`` evidence into a clean
    bill of health: consistency is only reported when the evidence
    itself shows no change and no range drift between the first and last
    read.

    :param worksheet: the worksheet being read.
    :param bool verify_before_each_read: when ``True`` (default) the
        worksheet metadata is fetched before and after every read to
        capture ``rowCount`` evidence. Set to ``False`` to skip per-read
        sampling; the batch then lacks evidence covering the read window
        and can only be finalized as ``UNKNOWN``.
    """

    def __init__(
        self,
        worksheet: "Worksheet",
        verify_before_each_read: bool = True,
    ) -> None:
        self.worksheet = worksheet
        self.verify_before_each_read = verify_before_each_read
        self._reads: List[RangeReadEvidence] = []
        self._initial_properties: Optional[Mapping[str, Any]] = None
        self._final_properties: Optional[Mapping[str, Any]] = None
        self._metadata_errors: List[str] = []

    @property
    def reads(self) -> Sequence[RangeReadEvidence]:
        """Evidence collected so far, in read order."""
        return list(self._reads)

    def _fetch_properties(self) -> Optional[Mapping[str, Any]]:
        try:
            return _worksheet_properties(self.worksheet)
        except Exception as error:  # evidence must never break the read
            self._metadata_errors.append(repr(error))
            return None

    def _row_count(
        self, properties: Optional[Mapping[str, Any]]
    ) -> Optional[int]:
        if properties is None:
            return None
        return properties.get("gridProperties", {}).get("rowCount")

    def read(self, range_name: str) -> Mapping[str, Any]:
        """Read one range, recording the request/response evidence.

        The method performs the same ``values.get`` call as
        :meth:`~gspread.worksheet.Worksheet.get` and returns the raw
        API response so the caller can build its values exactly as
        usual; the evidence bookkeeping happens as a side effect.

        :param str range_name: full A1 range with explicit row bounds.
        :returns: the raw ``spreadsheets.values.get`` response.
        :raises GSpreadException: if the range has no explicit row bounds.
        """
        requested_range = _absolute_range(self.worksheet, range_name)
        requested_start_row, requested_end_row = _row_bounds(requested_range)

        properties_before = None
        if self.verify_before_each_read:
            properties_before = self._fetch_properties()
            row_count_before = self._row_count(properties_before)
            if self._initial_properties is None and properties_before:
                self._initial_properties = properties_before
        else:
            row_count_before = None

        response = self.worksheet.client.values_get(
            self.worksheet.spreadsheet_id, requested_range
        )

        properties_after = None
        if self.verify_before_each_read:
            properties_after = self._fetch_properties()
            row_count_after = self._row_count(properties_after)
        else:
            row_count_after = None

        (
            response_start_row,
            response_end_row,
            response_range,
        ) = _response_row_bounds(response)

        evidence = RangeReadEvidence(
            requested_range=requested_range,
            response_range=response_range,
            requested_start_row=requested_start_row,
            requested_end_row=requested_end_row,
            response_start_row=response_start_row,
            response_end_row=response_end_row,
            row_count_before=row_count_before,
            row_count_after=row_count_after,
        )

        evidence.range_matches = (
            response_start_row == requested_start_row
            and response_end_row == requested_end_row
        )
        if not evidence.range_matches:
            evidence.issues.append(
                "response range {!r} does not match requested rows {}-{}".format(
                    response_range, requested_start_row, requested_end_row
                )
            )
        if (
            row_count_before is not None
            and row_count_after is not None
            and row_count_before != row_count_after
        ):
            evidence.issues.append(
                "rowCount changed from {} to {} during this read".format(
                    row_count_before, row_count_after
                )
            )

        self._reads.append(evidence)
        return response

    def finalize(self) -> ReadSnapshotReport:
        """Run all consistency checks and produce the batch report.

        The stitched values are not returned here: callers keep the
        responses returned by :meth:`read` and use the report only to
        decide whether the result may be trusted.
        """
        report = ReadSnapshotReport()
        report.reads = list(self._reads)

        if not self._reads:
            report.status = ReadSnapshotStatus.UNKNOWN
            report.notes.append("no reads were recorded")
            return report

        self._final_properties = self._fetch_properties()

        report.version_evidence = self._collect_version_evidence()
        report.notes.extend(self._metadata_errors)

        row_counts = self._observed_row_counts()
        evidence_complete = (
            self.verify_before_each_read
            and bool(row_counts)
            and not self._metadata_errors
        )

        changed = self._row_count_changed(row_counts)
        drifted = any(not read.range_matches for read in self._reads)

        if not evidence_complete:
            report.status = ReadSnapshotStatus.UNKNOWN
            if not self.verify_before_each_read:
                report.notes.append(
                    "per-read version sampling was disabled, so no evidence "
                    "covers the read window; snapshot completeness cannot be "
                    "asserted"
                )
            else:
                report.notes.append(
                    "no reliable worksheet version evidence available; "
                    "snapshot completeness cannot be asserted"
                )
        elif changed or drifted:
            report.status = ReadSnapshotStatus.INCONSISTENT
            if changed:
                report.notes.append(
                    "worksheet rowCount changed between reads: {}".format(
                        sorted(row_counts)
                    )
                )
        else:
            report.status = ReadSnapshotStatus.CONSISTENT
            report.notes.append(
                "rowCount stable at {} across all evidence points; "
                "all response ranges match the requests".format(
                    next(iter(row_counts))
                )
            )

        if report.status is ReadSnapshotStatus.INCONSISTENT:
            report.suspect_regions = self._suspect_regions(row_counts)

        return report

    def _collect_version_evidence(self) -> Dict[str, Any]:
        per_read = [
            {
                "requested_range": read.requested_range,
                "before": read.row_count_before,
                "after": read.row_count_after,
            }
            for read in self._reads
        ]
        evidence: Dict[str, Any] = {"row_counts_per_read": per_read}
        if self._initial_properties is not None:
            evidence["initial_worksheet_properties"] = _json_safe(
                self._initial_properties
            )
        if self._final_properties is not None:
            evidence["final_worksheet_properties"] = _json_safe(
                self._final_properties
            )
        return evidence

    def _observed_row_counts(self) -> Set[int]:
        counts = set()
        for read in self._reads:
            if read.row_count_before is not None:
                counts.add(read.row_count_before)
            if read.row_count_after is not None:
                counts.add(read.row_count_after)
        if self._initial_properties is not None:
            count = self._initial_properties.get("gridProperties", {}).get(
                "rowCount"
            )
            if count is not None:
                counts.add(count)
        if self._final_properties is not None:
            count = self._final_properties.get("gridProperties", {}).get("rowCount")
            if count is not None:
                counts.add(count)
        return counts

    @staticmethod
    def _row_count_changed(row_counts: Set[int]) -> bool:
        return len(row_counts) > 1

    def _suspect_regions(self, row_counts: Set[int]) -> List[SuspectRegion]:
        """Bound the rows that a concurrent insert could have affected.

        With no server-side change journal we cannot point at the exact
        inserted row. The safe rule is:

        * an insert strictly before the first read's first row shifts
          every read, so the whole covered span is suspect;
        * otherwise the change can only affect rows from the earliest
          read whose evidence disagrees (or whose local rowCount marker
          moved) up to the end of the batch.
        """
        if not self._reads:
            return []

        first_read = self._reads[0]
        last_read = self._reads[-1]

        earliest_unstable = None
        for index, read in enumerate(self._reads):
            unstable = not read.range_matches
            if (
                read.row_count_before is not None
                and read.row_count_after is not None
                and read.row_count_before != read.row_count_after
            ):
                unstable = True
            if unstable:
                earliest_unstable = index
                break

        if earliest_unstable is None:
            first_row = first_read.requested_start_row
        else:
            unstable_read = self._reads[earliest_unstable]
            first_row = unstable_read.requested_start_row
            if unstable_read.response_start_row is not None:
                first_row = min(first_row, unstable_read.response_start_row)

        last_row = last_read.requested_end_row
        for read in self._reads:
            if read.response_end_row is not None:
                last_row = max(last_row, read.response_end_row)

        reason = (
            "rowCount observed values {} do not agree; rows inserted or "
            "deleted by another editor may be missing or duplicated".format(
                sorted(row_counts)
            )
        )
        return [
            SuspectRegion(
                first_row=first_row,
                last_row=last_row,
                reason=reason,
            )
        ]


def _absolute_range(worksheet: "Worksheet", range_name: str) -> str:
    if "!" in range_name:
        return range_name
    return absolute_range_name(worksheet.title, range_name)


def reconcile_ranges(
    requested_ranges: Sequence[str],
    response_ranges: Sequence[str],
) -> List[SuspectRegion]:
    """Pure function locating row drift between requested and answered ranges.

    This is the local-replay building block: feed it the ranges a batch
    asked for and the ``range`` strings the API answered with (in the
    same order) and it returns the row intervals where the server-side
    coordinates shifted. A shift means rows at the boundary were
    inserted or deleted and the stitched values cannot be aligned
    without re-reading.

    :param requested_ranges: A1 ranges that were requested.
    :param response_ranges: A1 ranges echoed by the responses.
    :returns: suspect row intervals (may be empty when nothing drifted).
    :raises GSpreadException: if the two sequences differ in length or a
        range is not a full A1 range.
    """
    if len(requested_ranges) != len(response_ranges):
        raise GSpreadException(
            "cannot reconcile {} requests with {} responses".format(
                len(requested_ranges), len(response_ranges)
            )
        )

    regions: List[SuspectRegion] = []
    for requested_name, response_name in zip(requested_ranges, response_ranges):
        requested_a1 = get_a1_from_absolute_range(requested_name)
        response_a1 = get_a1_from_absolute_range(response_name)
        if not is_full_a1_notation(requested_a1) or not is_full_a1_notation(
            response_a1
        ):
            raise GSpreadException(
                "reconcile_ranges needs full A1 ranges, got {!r} and {!r}".format(
                    requested_name, response_name
                )
            )

        requested_grid = a1_range_to_grid_range(requested_a1)
        response_grid = a1_range_to_grid_range(response_a1)
        requested_start = requested_grid["startRowIndex"] + 1
        requested_end = requested_grid["endRowIndex"]
        response_start = response_grid["startRowIndex"] + 1
        response_end = response_grid["endRowIndex"]

        if (requested_start, requested_end) != (response_start, response_end):
            regions.append(
                SuspectRegion(
                    first_row=min(requested_start, response_start),
                    last_row=max(requested_end, response_end),
                    reason=(
                        "answered range {!r} drifted from requested {!r}".format(
                            response_name, requested_name
                        )
                    ),
                )
            )

    return _merge_regions(regions)


def _merge_regions(regions: Sequence[SuspectRegion]) -> List[SuspectRegion]:
    if not regions:
        return []

    ordered = sorted(regions, key=lambda region: region.first_row)
    merged = [
        SuspectRegion(
            ordered[0].first_row, ordered[0].last_row, ordered[0].reason
        )
    ]
    for region in ordered[1:]:
        current = merged[-1]
        if region.first_row <= current.last_row + 1:
            current.last_row = max(current.last_row, region.last_row)
            current.reason = "{}; {}".format(current.reason, region.reason)
        else:
            merged.append(
                SuspectRegion(region.first_row, region.last_row, region.reason)
            )
    return merged


def suspect_ranges_to_a1(regions: Sequence[SuspectRegion]) -> List[str]:
    """Convert suspect row intervals into row-only A1 ranges.

    :param regions: intervals returned by a report or
        :func:`reconcile_ranges`.
    :returns: strings like ``"12:120"`` describing the rows to re-read.
    """
    return ["{}:{}".format(region.first_row, region.last_row) for region in regions]
