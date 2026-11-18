"""
gspread.importing
~~~~~~~~~~~~~~~~~

Idempotent batch imports for Google Sheets.

The Sheets ``values.append`` endpoint is not idempotent.  If the request times
out the server may have written every row, some of the rows, or none of them.
Re-sending the request can therefore create duplicates, while assuming success
can silently lose rows.

:class:`BatchImporter` makes the outcome verifiable:

* every batch has a business identity (``batch_id``) and a durable manifest;
* every imported row carries a unique receipt token in a receipt column;
* after a timeout the sheet is read back and each input row is classified as
  ``confirmed``, ``unconfirmed`` or ``conflict``;
* only rows classified as ``unconfirmed`` (and safe to retry) are appended.

A business row that was inserted by somebody else is never accepted as our own
receipt.  The receipt token, not the business values, proves ownership.
"""

import hashlib
import json
import os
import tempfile
import threading
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import Timeout as RequestsTimeout

from .exceptions import APIError, GSpreadException
from .http_client import DefaultSerializerType
from .utils import (
    InsertDataOption,
    StrEnum,
    ValueInputOption,
    ValueRenderOption,
    a1_range_to_grid_range,
    get_a1_from_absolute_range,
)

DEFAULT_RECEIPT_COLUMN = "__gspread_import_receipt__"
RECEIPT_PREFIX = "__gimport__:"


class ImportStatus(StrEnum):
    """Outcome of reconciling one input row against the sheet."""

    confirmed = "confirmed"
    unconfirmed = "unconfirmed"
    conflict = "conflict"

    CONFIRMED = "confirmed"
    UNCONFIRMED = "unconfirmed"
    CONFLICT = "conflict"


class ImportConflictError(GSpreadException):
    """Raised when an import cannot be completed because of conflicting rows."""

    def __init__(self, report: "ImportReport") -> None:
        super().__init__(
            "import {} has {} conflicting row(s)".format(
                report.batch_id, len(report.conflicts)
            )
        )
        self.report = report


class ImportStateError(GSpreadException):
    """Raised when a saved import manifest cannot be reconciled safely."""


class ImportTimeout(GSpreadException):
    """Raised when an append timed out and could not be verified.

    The attached report tells the caller exactly which rows are confirmed,
    unconfirmed and conflicting.  Unlike blindly retrying the whole batch,
    ``report.unconfirmed`` (and especially ``report.needs_retry``) can be used
    to decide what, if anything, is safe to do next.
    """

    def __init__(self, report: "ImportReport") -> None:
        super().__init__(
            "import {} timed out; {} row(s) unconfirmed".format(
                report.batch_id, len(report.unconfirmed)
            )
        )
        self.report = report


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _canonical_value(value: Any) -> Any:
    """Normalize a cell value for comparison.

    Sheets may return a value as ``1`` while the caller supplied ``"1"`` (or
    the other way around).  Blank cells are normalized to ``""``.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, (int, float)):
        return value
    return str(value)


def _canonical_key(values: Iterable[Any]) -> Tuple[Any, ...]:
    return tuple(_canonical_value(value) for value in values)


def _keys_equal(left: Iterable[Any], right: Iterable[Any]) -> bool:
    return _values_equal(left, right)


def _values_equal(left: Iterable[Any], right: Iterable[Any]) -> bool:
    left_values = [_canonical_value(value) for value in left]
    right_values = [_canonical_value(value) for value in right]
    if len(left_values) != len(right_values):
        return False

    for left_value, right_value in zip(left_values, right_values):
        if left_value == right_value:
            continue
        # Numbers can arrive as "1" from one API and 1 from another.
        try:
            if float(left_value) == float(right_value):
                continue
        except (TypeError, ValueError):
            pass
        return False
    return True


def _fingerprint(rows: Sequence[Sequence[Any]], headers: Optional[Sequence[str]]) -> str:
    payload = json.dumps(
        {"headers": list(headers) if headers is not None else None, "rows": rows},
        default=_json_default,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_timeout(error: BaseException) -> bool:
    if isinstance(
        error,
        (
            RequestsTimeout,
            RequestsConnectionError,
            TimeoutError,
            ConnectionError,
        ),
    ):
        return True
    if isinstance(error, APIError):
        return error.code in (408, 504)
    return False


def _pad_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


@dataclass
class RowImportResult:
    """Reconciliation result for one input row."""

    index: int
    status: ImportStatus
    business_key: Tuple[Any, ...]
    receipt: str
    values: List[Any]
    row_number: Optional[int] = None
    reason: str = ""
    retryable: bool = False
    observed_values: Optional[List[Any]] = None
    conflicting_row_number: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "status": self.status.value,
            "business_key": list(self.business_key),
            "receipt": self.receipt,
            "values": list(self.values),
            "row_number": self.row_number,
            "reason": self.reason,
            "retryable": self.retryable,
            "observed_values": self.observed_values,
            "conflicting_row_number": self.conflicting_row_number,
        }

    to_dict = as_dict


@dataclass
class ImportReport:
    """Result of :meth:`BatchImporter.run` or :meth:`BatchImporter.reconcile`."""

    batch_id: str
    receipt_column: int
    rows: List[RowImportResult]
    attempts: int = 0
    timed_out: bool = False
    complete: bool = False
    journal_path: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    @property
    def confirmed(self) -> List[RowImportResult]:
        return [row for row in self.rows if row.status == ImportStatus.confirmed]

    @property
    def unconfirmed(self) -> List[RowImportResult]:
        return [row for row in self.rows if row.status == ImportStatus.unconfirmed]

    @property
    def conflicts(self) -> List[RowImportResult]:
        return [row for row in self.rows if row.status == ImportStatus.conflict]

    @property
    def confirmed_rows(self) -> List[RowImportResult]:
        return self.confirmed

    @property
    def unconfirmed_rows(self) -> List[RowImportResult]:
        return self.unconfirmed

    @property
    def conflict_rows(self) -> List[RowImportResult]:
        return self.conflicts

    @property
    def statuses(self) -> Dict[int, str]:
        return {row.index: row.status.value for row in self.rows}

    @property
    def all_confirmed(self) -> bool:
        return not self.unconfirmed and not self.conflicts

    @property
    def needs_retry(self) -> bool:
        return any(row.retryable for row in self.unconfirmed)

    @property
    def requires_attention(self) -> bool:
        return bool(self.conflicts) or bool(
            [row for row in self.unconfirmed if not row.retryable]
        )

    def by_status(self, status: Union[ImportStatus, str]) -> List[RowImportResult]:
        value = ImportStatus(status)
        return [row for row in self.rows if row.status == value]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "receipt_column": self.receipt_column,
            "attempts": self.attempts,
            "timed_out": self.timed_out,
            "complete": self.complete,
            "journal_path": self.journal_path,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "rows": [row.as_dict() for row in self.rows],
        }

    to_dict = as_dict


class ImportJournal:
    """A small JSON journal that survives process restarts.

    A journal can hold several batches.  When ``path`` is ``None`` the journal
    is kept in memory; the receipt tokens in the sheet still make a retry
    reconcilable as long as the same ``batch_id`` and rows are supplied.
    """

    VERSION = 1

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        self._lock = threading.RLock()
        self._batches: Dict[str, Dict[str, Any]] = {}
        if path is not None and os.path.exists(path):
            self._load()

    def _load(self) -> None:
        path = self.path
        if path is None:
            raise ImportStateError("cannot load an in-memory journal")
        try:
            with open(path, "r", encoding="utf-8") as journal_file:
                payload = json.load(journal_file)
        except (OSError, ValueError) as error:
            raise ImportStateError(
                "cannot read import journal {!r}: {}".format(path, error)
            )

        if not isinstance(payload, dict) or not isinstance(
            payload.get("batches"), dict
        ):
            raise ImportStateError(
                "invalid import journal {!r}: missing batches".format(self.path)
            )
        self._batches = payload["batches"]

    def get(self, batch_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            state = self._batches.get(batch_id)
            return (
                json.loads(json.dumps(state, default=_json_default)) if state else None
            )

    def put(self, state: MutableMapping[str, Any]) -> None:
        batch_id = state["batch_id"]
        with self._lock:
            self._batches[batch_id] = json.loads(
                json.dumps(state, default=_json_default)
            )
            self._flush()

    def _flush(self) -> None:
        if self.path is None:
            return

        directory = os.path.dirname(os.path.abspath(self.path))
        if directory and not os.path.isdir(directory):
            os.makedirs(directory, exist_ok=True)

        payload = {"version": self.VERSION, "batches": self._batches}
        fd, temporary_path = tempfile.mkstemp(
            prefix=".import-journal-", suffix=".json", dir=directory or None
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as journal_file:
                json.dump(
                    payload,
                    journal_file,
                    default=_json_default,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                journal_file.flush()
                os.fsync(journal_file.fileno())
            os.replace(temporary_path, self.path)
            try:
                os.chmod(self.path, 0o600)
            except OSError:
                pass
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)


class BatchImporter:
    """Append rows to a worksheet without duplicating them on retry.

    :param worksheet: the target :class:`gspread.worksheet.Worksheet`.
    :param rows: business rows (lists or dictionaries).  May be omitted when
        the batch is already present in the journal.
    :param batch_id: stable business identity of this import.  If omitted a
        random identity is generated and returned in the report.
    :param key_columns: columns that make up the business identity.  Accepts
        header names (when ``headers`` are supplied) or zero-based field
        indexes.  Defaults to all business columns.
    :param receipt_column: sheet column used for the ownership receipt.
        Accepts a one-based column number or a header name.  When omitted the
        receipt is appended after the widest business row.
    :param headers: business column names, excluding the receipt column.
    :param state_path: optional path to a JSON journal.  The manifest is
        written before the first append, so a retry in a new process can rely
        on the persisted batch identity.
    :param journal: an existing :class:`ImportJournal` to use instead of
        ``state_path``.
    """

    def __init__(
        self,
        worksheet: Any,
        rows: Optional[Sequence[Union[Sequence[Any], Mapping[str, Any]]]] = None,
        batch_id: Optional[str] = None,
        key_columns: Optional[Union[str, int, Sequence[Union[str, int]]]] = None,
        receipt_column: Optional[Union[str, int]] = None,
        headers: Optional[Sequence[str]] = None,
        state_path: Optional[str] = None,
        journal: Optional[ImportJournal] = None,
        value_input_option: ValueInputOption = ValueInputOption.raw,
        insert_data_option: Optional[InsertDataOption] = None,
        table_range: Optional[str] = None,
        default_serializer: DefaultSerializerType = None,
    ) -> None:
        self.worksheet = worksheet
        self.batch_id = str(batch_id) if batch_id is not None else uuid.uuid4().hex
        self.key_columns = key_columns
        self.receipt_column = receipt_column
        self.headers = list(headers) if headers is not None else None
        self.value_input_option = value_input_option
        self.insert_data_option = insert_data_option
        self.table_range = table_range
        self.default_serializer = default_serializer

        if journal is not None and state_path is not None:
            raise ValueError("pass either journal or state_path, not both")
        self.journal = journal if journal is not None else ImportJournal(state_path)

        self._state = self._load_or_create(rows)
        self._save()

    def _load_or_create(
        self,
        rows: Optional[Sequence[Union[Sequence[Any], Mapping[str, Any]]]],
    ) -> Dict[str, Any]:
        saved = self.journal.get(self.batch_id)

        if saved is not None:
            if rows is not None:
                reopened_rows = self._strip_saved_receipt(rows, saved)
                prepared_rows, prepared_headers = self._prepare_rows(
                    reopened_rows, saved.get("headers")
                )
                fingerprint = _fingerprint(prepared_rows, prepared_headers)
                if fingerprint != saved["fingerprint"]:
                    raise ImportStateError(
                        "batch {!r} was saved with different rows/headers; "
                        "refusing to treat it as the same business import".format(
                            self.batch_id
                        )
                    )
            saved_table_range = saved.get("table_range")
            if self.table_range is None:
                self.table_range = saved_table_range
            elif self.table_range != saved_table_range:
                raise ImportStateError(
                    "batch {!r} was saved with table_range {!r}; got {!r}".format(
                        self.batch_id, saved_table_range, self.table_range
                    )
                )
            self.headers = saved.get("headers")
            self.receipt_column = saved.get("receipt_column")
            self.key_columns = saved.get("key_columns")
            return saved

        if rows is None:
            raise ImportStateError(
                "unknown import batch {!r}; rows are required".format(self.batch_id)
            )

        source_headers = self.headers
        normalized_source_rows: Sequence[Union[Sequence[Any], Mapping[str, Any]]] = rows
        receipt_name: Optional[str] = None
        receipt_index = -1

        if isinstance(self.receipt_column, str) and source_headers is not None:
            if self.receipt_column in source_headers:
                receipt_name = self.receipt_column
                receipt_index = list(source_headers).index(receipt_name)
                source_headers = [
                    value
                    for position, value in enumerate(source_headers)
                    if position != receipt_index
                ]
                original_width = len(source_headers) + 1
                normalized_source_rows = [
                    row
                    if isinstance(row, Mapping) or len(row) != original_width
                    else list(row)[:receipt_index] + list(row)[receipt_index + 1 :]
                    for row in rows
                ]

        prepared_rows, prepared_headers = self._prepare_rows(
            normalized_source_rows, source_headers
        )
        self.headers = prepared_headers

        if receipt_name is None:
            receipt_index, receipt_name = self._resolve_receipt_column(
                prepared_rows, prepared_headers
            )
        key_indexes = self._resolve_key_columns(prepared_headers, prepared_rows)
        fingerprint = _fingerprint(prepared_rows, prepared_headers)
        tokens = [
            self._make_token(fingerprint, index)
            for index in range(len(prepared_rows))
        ]

        return {
            "version": 1,
            "batch_id": self.batch_id,
            "fingerprint": fingerprint,
            "rows": prepared_rows,
            "headers": prepared_headers,
            "key_columns": self._serializable_key_columns(),
            "key_indexes": key_indexes,
            "receipt_column": receipt_index + 1,
            "receipt_name": receipt_name,
            "column_offset": self._column_offset(),
            "table_range": self.table_range,
            "tokens": tokens,
            "worksheet": self._worksheet_identity(),
            "attempts": [],
            "statuses": [],
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
        }

    def _serializable_key_columns(self) -> Any:
        if isinstance(self.key_columns, (str, int)):
            return self.key_columns
        if self.key_columns is None:
            return None
        return list(self.key_columns)

    def _strip_saved_receipt(
        self,
        rows: Sequence[Union[Sequence[Any], Mapping[str, Any]]],
        saved: Mapping[str, Any],
    ) -> List[Union[Sequence[Any], Mapping[str, Any]]]:
        """Accept a caller who re-supplies full rows including the receipt."""

        offset = int(saved.get("column_offset", 0))
        business_width = len(saved.get("rows", [[]])[0])
        full_width = offset + business_width + 1

        normalized: List[Union[Sequence[Any], Mapping[str, Any]]] = []
        for row in rows:
            if isinstance(row, Mapping) or len(row) != full_width:
                normalized.append(row)
                continue
            normalized.append(list(row[offset : offset + business_width]))
        return normalized

    def _worksheet_identity(self) -> Dict[str, Any]:
        return {
            "spreadsheet_id": getattr(self.worksheet, "spreadsheet_id", None),
            "sheet_id": getattr(self.worksheet, "id", None),
            "title": getattr(self.worksheet, "title", None),
        }

    def _column_offset(self) -> int:
        """Zero-based column where an append writes, derived from table_range."""

        if not self.table_range:
            return 0
        table_range = get_a1_from_absolute_range(self.table_range)
        try:
            grid_range = a1_range_to_grid_range(table_range)
        except Exception:
            return 0
        return grid_range.get("startColumnIndex", 0)

    def _prepare_rows(
        self,
        rows: Sequence[Union[Sequence[Any], Mapping[str, Any]]],
        headers: Optional[Sequence[str]],
    ) -> Tuple[List[List[Any]], Optional[List[str]]]:
        if not rows:
            raise ValueError("rows must not be empty")

        if isinstance(rows[0], Mapping):
            normalized, prepared_headers = self._prepare_mapping_rows(rows, headers)
        else:
            normalized, prepared_headers = self._prepare_sequence_rows(rows, headers)

        width = max(
            [len(prepared_headers or [])] + [len(row) for row in normalized]
        )
        if not width:
            raise ValueError("rows must contain at least one column")
        normalized = [row + [""] * (width - len(row)) for row in normalized]

        return normalized, prepared_headers

    def _prepare_mapping_rows(
        self,
        rows: Sequence[Union[Sequence[Any], Mapping[str, Any]]],
        headers: Optional[Sequence[str]],
    ) -> Tuple[List[List[Any]], Optional[List[str]]]:
        prepared_headers = list(headers) if headers is not None else None
        mapping_rows: List[Mapping[str, Any]] = []

        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("rows must not mix mappings and sequences")
            mapping_rows.append(row)

        if prepared_headers is None:
            prepared_headers = []
            for row in mapping_rows:
                for key in row.keys():
                    if key not in prepared_headers:
                        prepared_headers.append(key)

        normalized = [
            [row.get(header, "") for header in prepared_headers]
            for row in mapping_rows
        ]
        return normalized, prepared_headers

    def _prepare_sequence_rows(
        self,
        rows: Sequence[Union[Sequence[Any], Mapping[str, Any]]],
        headers: Optional[Sequence[str]],
    ) -> Tuple[List[List[Any]], Optional[List[str]]]:
        prepared_headers = list(headers) if headers is not None else None
        normalized: List[List[Any]] = []

        for row in rows:
            if isinstance(row, Mapping):
                raise ValueError("rows must not mix mappings and sequences")
            normalized.append(list(row))

        if prepared_headers is not None:
            for row in normalized:
                if len(row) > len(prepared_headers):
                    raise ValueError(
                        "headers has {} entries but a row has {} values".format(
                            len(prepared_headers), len(row)
                        )
                    )

        return normalized, prepared_headers

    def _resolve_key_columns(
        self,
        headers: Optional[Sequence[str]],
        rows: Sequence[Sequence[Any]],
    ) -> List[int]:
        if self.key_columns is None:
            return list(range(len(rows[0])))

        columns = (
            [self.key_columns]
            if isinstance(self.key_columns, (str, int))
            else list(self.key_columns)
        )
        if not columns:
            raise ValueError("key_columns must not be empty")

        indexes: List[int] = []
        for column in columns:
            if isinstance(column, str):
                if headers is None or column not in headers:
                    raise ValueError(
                        "key column {!r} is not in headers".format(column)
                    )
                indexes.append(list(headers).index(column))
            elif isinstance(column, int):
                if column < 0:
                    raise ValueError("key column indexes must be >= 0")
                indexes.append(column)
            else:
                raise ValueError("key columns must be names or zero-based indexes")

        for index in indexes:
            if index >= len(rows[0]):
                raise ValueError(
                    "key column index {} is outside the row width {}".format(
                        index, len(rows[0])
                    )
                )
        return indexes

    def _resolve_receipt_column(
        self,
        rows: Sequence[Sequence[Any]],
        headers: Optional[Sequence[str]],
    ) -> Tuple[int, Optional[str]]:
        max_width = max(len(row) for row in rows)
        receipt = self.receipt_column

        if receipt is None:
            return max_width, None

        if isinstance(receipt, int):
            if receipt < 1:
                raise ValueError("receipt_column must be a one-based column number")
            relative_index = receipt - 1 - self._column_offset()
            if relative_index < 0:
                raise ValueError(
                    "receipt_column {} is before table_range {!r}".format(
                        receipt, self.table_range
                    )
                )
            return relative_index, None
        if not isinstance(receipt, str):
            raise ValueError("receipt_column must be a column number or header name")

        if headers is not None and receipt in headers:
            raise ValueError(
                "receipt_column must not be in business headers; pass the "
                "receipt header separately as receipt_column and omit it from headers"
            )

        return max_width, receipt

    def _make_token(self, fingerprint: str, index: int) -> str:
        digest = hashlib.sha256(
            "{}:{}:{}".format(self.batch_id, fingerprint, index).encode("utf-8")
        ).hexdigest()[:24]
        return "{}{}:{}".format(RECEIPT_PREFIX, self.batch_id, digest)

    @property
    def state(self) -> Dict[str, Any]:
        return self._state

    @property
    def rows(self) -> List[List[Any]]:
        return self._state["rows"]

    @property
    def tokens(self) -> List[str]:
        return self._state["tokens"]

    @property
    def key_indexes(self) -> List[int]:
        return self._state["key_indexes"]

    @property
    def receipt_index(self) -> int:
        return int(self._state["receipt_column"]) - 1

    @property
    def column_offset(self) -> int:
        return int(self._state.get("column_offset", 0))

    def _save(self) -> None:
        self._state["updated_at"] = _utc_now()
        self.journal.put(self._state)

    def _data_columns(self) -> List[int]:
        receipt_index = self.receipt_index
        return [
            column if column < receipt_index else column + 1
            for column in range(len(self._state["rows"][0]))
        ]

    def _row_with_receipt(self, values: Sequence[Any], token: str) -> List[Any]:
        result = list(values)
        while len(result) < self.receipt_index:
            result.append("")
        result.insert(self.receipt_index, token)
        return result

    def _read_sheet(self) -> List[List[Any]]:
        try:
            values = self.worksheet.get_all_values(
                value_render_option=ValueRenderOption.unformatted,
                pad_values=False,
            )
        except TypeError:
            values = self.worksheet.get_all_values()
        return [list(row) for row in values]

    def _observed_rows(self) -> List[Dict[str, Any]]:
        sheet_rows = self._read_sheet()
        data_columns = self._data_columns()
        receipt_index = self.receipt_index
        field_count = len(self._state["rows"][0])
        business_headers = self._state.get("headers")

        observed: List[Dict[str, Any]] = []
        for position, sheet_row in enumerate(sheet_rows):
            padded = list(sheet_row) + [""] * max(
                0, self.column_offset + receipt_index + 1 - len(sheet_row)
            )
            if all(_canonical_value(value) == "" for value in padded):
                continue

            business_values = [
                padded[self.column_offset + column]
                if self.column_offset + column < len(padded)
                else ""
                for column in data_columns[:field_count]
            ]
            receipt = (
                _pad_string(padded[self.column_offset + receipt_index])
                if self.column_offset + receipt_index < len(padded)
                else ""
            )

            # Do not let a header row masquerade as a business row.
            if (
                business_headers is not None
                and not receipt
                and [str(value) for value in business_values]
                == [str(value) for value in business_headers]
            ):
                continue

            observed.append(
                {
                    "row_number": position + 1,
                    "business_values": business_values,
                    "receipt": receipt,
                }
            )
        return observed

    def reconcile(self) -> ImportReport:
        """Read the sheet and classify every row without appending anything."""

        observed = self._observed_rows()
        receipt_counts = Counter(
            row["receipt"] for row in observed if row["receipt"]
        )
        batch_tokens = set(self._state["tokens"])
        key_indexes = self.key_indexes

        results: List[RowImportResult] = []
        for index, expected_values in enumerate(self._state["rows"]):
            token = self._state["tokens"][index]
            expected_key = _canonical_key(
                expected_values[position] for position in key_indexes
            )
            token_hits = [row for row in observed if row["receipt"] == token]
            same_key_rows = [
                row
                for row in observed
                if _keys_equal(
                    (row["business_values"][position] for position in key_indexes),
                    expected_key,
                )
            ]
            unowned_same_key = [
                row for row in same_key_rows if row["receipt"] not in batch_tokens
            ]

            result = RowImportResult(
                index=index,
                status=ImportStatus.unconfirmed,
                business_key=expected_key,
                receipt=token,
                values=list(expected_values),
            )

            if receipt_counts[token] > 1 or len(token_hits) > 1:
                result.status = ImportStatus.conflict
                result.reason = "receipt token appears more than once"
                result.conflicting_row_number = (
                    token_hits[1]["row_number"] if len(token_hits) > 1 else None
                )
            elif len(token_hits) == 1:
                hit = token_hits[0]
                if not _values_equal(hit["business_values"], expected_values):
                    result.status = ImportStatus.conflict
                    result.reason = "row carrying this receipt has different values"
                    result.conflicting_row_number = hit["row_number"]
                    result.observed_values = hit["business_values"]
                elif unowned_same_key:
                    result.status = ImportStatus.conflict
                    result.reason = (
                        "business row also exists without this batch's receipt"
                    )
                    result.conflicting_row_number = unowned_same_key[0]["row_number"]
                    result.observed_values = unowned_same_key[0]["business_values"]
                    result.row_number = hit["row_number"]
                else:
                    result.status = ImportStatus.confirmed
                    result.row_number = hit["row_number"]
                    result.observed_values = hit["business_values"]
                    result.reason = "receipt found with matching business values"
            elif unowned_same_key:
                result.status = ImportStatus.conflict
                result.reason = (
                    "business row already exists without this batch's receipt"
                )
                result.conflicting_row_number = unowned_same_key[0]["row_number"]
                result.observed_values = unowned_same_key[0]["business_values"]
            else:
                result.status = ImportStatus.unconfirmed
                result.reason = "no matching receipt or business row found"
                result.retryable = True

            results.append(result)

        report = ImportReport(
            batch_id=self.batch_id,
            receipt_column=self.column_offset + self.receipt_index + 1,
            rows=results,
            attempts=len(self._state.get("attempts", [])),
            timed_out=any(
                attempt.get("outcome") == "timeout"
                for attempt in self._state.get("attempts", [])
            ),
            complete=all(
                result.status == ImportStatus.confirmed for result in results
            ),
            journal_path=self.journal.path,
            started_at=self._state.get("created_at"),
            finished_at=_utc_now(),
        )
        self._remember_statuses(results)
        return report

    def _remember_statuses(self, results: Sequence[RowImportResult]) -> None:
        self._state["statuses"] = [result.as_dict() for result in results]
        self._save()

    def _append(self, indexes: Sequence[int]) -> None:
        values = [
            self._row_with_receipt(
                self._state["rows"][index], self._state["tokens"][index]
            )
            for index in indexes
        ]
        self.worksheet.append_rows(
            values,
            value_input_option=self.value_input_option,
            insert_data_option=self.insert_data_option,
            table_range=self.table_range,
            default_serializer=self.default_serializer,
        )

    def _record_attempt(
        self,
        outcome: str,
        indexes: Sequence[int],
        error: Optional[BaseException] = None,
    ) -> None:
        attempt: Dict[str, Any] = {
            "at": _utc_now(),
            "outcome": outcome,
            "row_indexes": list(indexes),
        }
        if error is not None:
            attempt["error"] = "{}: {}".format(type(error).__name__, error)
        self._state.setdefault("attempts", []).append(attempt)
        self._save()

    def _append_without_automatic_retry(self, indexes: Sequence[int]) -> None:
        """Append without letting the HTTP client silently re-POST it.

        ``BackOffHTTPClient`` normally retries 408/5xx responses.  For an
        append that is exactly the dangerous blind retry this module exists to
        prevent, so the retry suppression is scoped to this one request.
        """

        suppress = getattr(
            getattr(self.worksheet, "client", None), "suppress_automatic_retry", None
        )
        if suppress is None:
            self._append(indexes)
            return

        with suppress():
            self._append(indexes)

    def run(
        self,
        max_attempts: int = 3,
        retry: bool = True,
        raise_on_conflict: bool = False,
        raise_on_timeout: bool = False,
    ) -> ImportReport:
        """Reconcile, then append only rows that are safe to append.

        Rows already confirmed by their receipt are never appended again.  A
        row whose business identity exists without this batch's receipt is a
        conflict and is also never appended.

        If an append is acknowledged but its receipts are not immediately
        visible, the importer stops instead of re-sending the batch.  That is
        the safe choice: a successful append that is not yet visible must not
        be repeated.
        """

        if max_attempts < 0:
            raise ValueError("max_attempts must be >= 0")

        while self._should_append(max_attempts, retry):
            self._append_unconfirmed()

        report = self.reconcile()
        if raise_on_conflict and report.conflicts:
            raise ImportConflictError(report)
        if raise_on_timeout and report.timed_out and report.unconfirmed:
            raise ImportTimeout(report)
        return report

    def _should_append(self, max_attempts: int, retry: bool) -> bool:
        if not retry:
            return False
        report = self.reconcile()
        if report.attempts >= max_attempts:
            return False
        if not [row for row in report.unconfirmed if row.retryable]:
            return False
        # If the previous append was acknowledged but its receipts are not
        # visible, retrying could duplicate rows.  Stop and ask for review.
        return not bool(
            self._state.get("attempts")
            and self._state["attempts"][-1].get("outcome") == "success"
        )

    def _append_unconfirmed(self) -> None:
        report = self.reconcile()
        indexes = [
            row.index for row in report.unconfirmed if row.retryable
        ]
        try:
            self._append_without_automatic_retry(indexes)
        except BaseException as error:
            if _is_timeout(error):
                self._record_attempt("timeout", indexes, error)
                return
            self._record_attempt("error", indexes, error)
            raise
        self._record_attempt("success", indexes)

    def retry(
        self,
        max_attempts: int = 3,
        raise_on_conflict: bool = False,
        raise_on_timeout: bool = False,
    ) -> ImportReport:
        """Retry an existing batch using the saved manifest."""

        return self.run(
            max_attempts=max_attempts,
            retry=True,
            raise_on_conflict=raise_on_conflict,
            raise_on_timeout=raise_on_timeout,
        )

    def manifest(self) -> Dict[str, Any]:
        """Return a copy of the persisted batch manifest."""

        return json.loads(json.dumps(self._state, default=_json_default))


def import_rows(
    worksheet: Any,
    rows: Sequence[Union[Sequence[Any], Mapping[str, Any]]],
    **kwargs: Any,
) -> ImportReport:
    """Convenience wrapper around :class:`BatchImporter`.

    ``kwargs`` are forwarded to :class:`BatchImporter`; ``run`` arguments such
    as ``max_attempts`` and ``raise_on_conflict`` are also accepted.
    """

    run_keys = {
        "max_attempts",
        "retry",
        "raise_on_conflict",
        "raise_on_timeout",
    }
    run_kwargs = {key: kwargs.pop(key) for key in list(kwargs) if key in run_keys}
    return BatchImporter(worksheet, rows, **kwargs).run(**run_kwargs)
