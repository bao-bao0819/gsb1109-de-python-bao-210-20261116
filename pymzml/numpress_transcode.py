#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Controlled, round-trip checked (re-)encoding of MS arrays with MS-Numpress.

The Numpress variants implemented in :mod:`pymzml.ms_numpress` are lossy and
the pure python implementation has known edge case bugs (e.g. leading zeros
disappear during PIC encoding, all-zero linear arrays lose their tail).  This
module therefore never treats an encoding as harmless based on equal array
lengths or a smaller encoded payload: every conversion is decoded again and
checked element by element against the source values.

Position (m/z) and intensity arrays use independent error budgets, which may
be absolute, relative or both.  Every round stores its error statistics together
with the identity of the encoding it was converted from, so multi-round
migrations can be judged on their cumulative error against the original data.

Example:

    mz_codec = NumpressTranscoder(original_encoding="64-bit float")
    payload, mz_back, mz_report = mz_codec.transcode(
        mz_values, "linear", {"rtol": 1e-6}
    )
    i_codec = NumpressTranscoder(original_encoding="64-bit float")
    payload, i_back, i_report = i_codec.transcode(
        intensity_values, "pic", {"atol": 0.5}
    )

A report with :attr:`RoundTripReport.within_budget` set to ``True`` means the
round trip stays inside the requested budget, never that the encoding is
lossless.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from .ms_numpress import MSNumpress

#: Numpress methods that can be selected for (re-)encoding.
NUMPRESS_METHODS = ("linear", "pic", "slof")

#: Mapping of method names to their PSI-MS compression term names.
METHOD_ENCODING_NAMES = {
    "linear": "MS-Numpress linear prediction compression",
    "pic": "MS-Numpress positive integer compression",
    "slof": "MS-Numpress short logged float compression",
}

Budget = Union[float, int, dict]


class NumpressRoundtripError(Exception):
    """Raised when an array cannot be represented by the chosen Numpress method."""


def _normalize_budget(budget: Budget) -> Tuple[float, float]:
    """
    Normalize an error budget to ``(absolute tolerance, relative tolerance)``.

    A plain number is interpreted as an absolute tolerance. A dictionary may
    contain the keys ``atol`` and ``rtol`` (missing keys default to zero).
    """
    if isinstance(budget, (int, float)) and not isinstance(budget, bool):
        atol, rtol = float(budget), 0.0
    elif isinstance(budget, dict):
        atol = float(budget.get("atol", 0.0))
        rtol = float(budget.get("rtol", 0.0))
    else:
        raise TypeError(
            "budget must be a number or a dict with 'atol' and/or 'rtol', "
            "got {0!r}".format(type(budget).__name__)
        )
    if atol < 0.0 or rtol < 0.0:
        raise ValueError("error budgets must not be negative")
    return atol, rtol


def _error_stats(
    original: np.ndarray, decoded: np.ndarray
) -> Tuple[float, float]:
    """
    Return the maximum absolute and maximum relative elementwise error.

    Zero elements of ``original`` are excluded from the relative error but
    their absolute error is still measured, so zero peaks have to survive the
    round trip within the absolute tolerance.
    """
    abs_error = np.abs(decoded - original)
    max_abs = float(np.max(abs_error)) if abs_error.size else 0.0
    nonzero = original != 0
    if np.any(nonzero):
        max_rel = float(np.max(abs_error[nonzero] / np.abs(original[nonzero])))
    else:
        max_rel = 0.0
    return max_abs, max_rel


def _within_budget(
    original: np.ndarray, decoded: np.ndarray, atol: float, rtol: float
) -> bool:
    """Elementwise budget check, ``|diff| <= atol + rtol * |original|``."""
    abs_error = np.abs(decoded - original)
    return bool(np.all(abs_error <= atol + rtol * np.abs(original)))


@dataclass
class RoundTripReport:
    """
    Bookkeeping of a single encode/decode round.

    Attributes:
        method: Numpress method used in this round (``linear``, ``pic`` or
            ``slof``).
        round_index: Zero based position of the round in a migration.
        source_encoding: Identity of the encoding the input array carried.
        target_encoding: Identity of the Numpress encoding produced.
        array_length: Number of elements in the source array.
        encoded_size: Size of the Numpress payload in bytes. Recorded for
            information only, it is never used to judge the round trip.
        max_abs_error: Largest absolute element error of the round trip.
        max_rel_error: Largest relative element error (zero sources excluded).
        within_budget: ``True`` if every element stays inside the budget.
        budget_atol: Absolute tolerance applied to this round.
        budget_rtol: Relative tolerance applied to this round.
        reason: Explanation when the budget is violated, else ``None``.
        cumulative_abs_error: Absolute error against the original array of a
            migration; ``None`` for a standalone :meth:`NumpressTranscoder.transcode`.
        cumulative_rel_error: Relative error against the original array of a
            migration; ``None`` for a standalone transcode.
        cumulative_within_budget: Whether the cumulative error is inside the
            budget; ``None`` for a standalone transcode.
    """

    method: str
    round_index: int
    source_encoding: str
    target_encoding: str
    array_length: int
    encoded_size: int
    max_abs_error: float
    max_rel_error: float
    within_budget: bool
    budget_atol: float
    budget_rtol: float
    reason: Optional[str] = None
    cumulative_abs_error: Optional[float] = None
    cumulative_rel_error: Optional[float] = None
    cumulative_within_budget: Optional[bool] = None


@dataclass
class MigrationResult:
    """
    Outcome of a multi-round migration.

    Attributes:
        accepted: ``True`` only if every round and the cumulative error against
            the original array stay inside the budget.
        reports: One :class:`RoundTripReport` per executed round.
        values: Array obtained after the last executed round.
        original_values: Untouched copy of the array the migration started with.
    """

    accepted: bool
    reports: List[RoundTripReport] = field(default_factory=list)
    values: Optional[np.ndarray] = None
    original_values: Optional[np.ndarray] = None


class NumpressTranscoder:
    """
    Round-trip checked Numpress (re-)encoder for a single MS array.

    One instance tracks one array (e.g. either the m/z or the intensity array
    of a spectrum), each with its own error budget.  Use separate instances to
    apply different budgets to position and intensity data.

    Keyword Args:
        original_encoding (str): Identity recorded as ``source_encoding`` of the
            first round, e.g. the CV name of the source compression.
    """

    def __init__(self, original_encoding: str = "64-bit float"):
        self.original_encoding = original_encoding
        #: :class:`RoundTripReport` instances of the latest conversion.
        self.history: List[RoundTripReport] = []

    @staticmethod
    def _validate_method(method: str) -> None:
        if method not in NUMPRESS_METHODS:
            raise ValueError(
                "unknown Numpress method {0!r}, expected one of {1}".format(
                    method, ", ".join(NUMPRESS_METHODS)
                )
            )

    @staticmethod
    def _validate_values(values: np.ndarray, method: str) -> None:
        if values.ndim != 1:
            raise ValueError("only one dimensional arrays can be transcoded")
        if values.size == 0:
            raise NumpressRoundtripError("cannot transcode empty arrays")
        if not np.all(np.isfinite(values)):
            raise ValueError(
                "arrays containing NaN or infinite values are not supported"
            )
        if method in ("pic", "slof") and np.any(values < 0):
            raise ValueError(
                "the {0!r} Numpress method requires non-negative values".format(method)
            )

    def _run_round(
        self,
        values: np.ndarray,
        method: str,
        atol: float,
        rtol: float,
        round_index: int,
        source_encoding: str,
    ) -> Tuple[np.ndarray, bytearray, RoundTripReport]:
        """
        Encode ``values`` with ``method``, decode them again and build the report.
        """
        self._validate_method(method)
        self._validate_values(values, method)

        encoder = MSNumpress(list(values))
        payload = getattr(encoder, "encode_" + method)()
        if not isinstance(payload, bytearray):
            # the pure python linear encoder returns plain ints for arrays of
            # length < 2, which cannot be read back as Numpress payloads
            raise NumpressRoundtripError(
                "method {0!r} cannot encode an array of length {1} "
                "(no bytearray payload produced)".format(method, values.size)
            )

        decoder = MSNumpress(bytearray(payload))
        decoded = np.asarray(
            getattr(decoder, "decode_" + method)(), dtype=np.float64
        )

        target_encoding = METHOD_ENCODING_NAMES[method]
        if decoded.size != values.size:
            report = RoundTripReport(
                method=method,
                round_index=round_index,
                source_encoding=source_encoding,
                target_encoding=target_encoding,
                array_length=values.size,
                encoded_size=len(payload),
                max_abs_error=float("inf"),
                max_rel_error=float("inf"),
                within_budget=False,
                budget_atol=atol,
                budget_rtol=rtol,
                reason=(
                    "length mismatch after round trip: "
                    "{0} elements encoded, {1} elements decoded".format(
                        values.size, decoded.size
                    )
                ),
            )
            return decoded, payload, report

        max_abs, max_rel = _error_stats(values, decoded)
        within = _within_budget(values, decoded, atol, rtol)
        reason = None
        if not within:
            reason = (
                "error budget (atol={0}, rtol={1}) exceeded: "
                "max_abs_error={2}, max_rel_error={3}".format(
                    atol, rtol, max_abs, max_rel
                )
            )
        report = RoundTripReport(
            method=method,
            round_index=round_index,
            source_encoding=source_encoding,
            target_encoding=target_encoding,
            array_length=values.size,
            encoded_size=len(payload),
            max_abs_error=max_abs,
            max_rel_error=max_rel,
            within_budget=within,
            budget_atol=atol,
            budget_rtol=rtol,
            reason=reason,
        )
        return decoded, payload, report

    def transcode(
        self,
        values: Union[Sequence[float], np.ndarray],
        method: str,
        budget: Budget,
    ) -> Tuple[bytearray, np.ndarray, RoundTripReport]:
        """
        Encode ``values`` with the selected Numpress method and read them back.

        Args:
            values: One dimensional array of finite floats.
            method (str): One of :data:`NUMPRESS_METHODS`.
            budget: Absolute tolerance as a number, or a dict with ``atol``
                and/or ``rtol`` keys.

        Returns:
            tuple: ``(payload, decoded_values, report)``.  Check
            ``report.within_budget`` before using ``payload``; a failed round
            trip is reported, not silently accepted.
        """
        atol, rtol = _normalize_budget(budget)
        source = np.asarray(values, dtype=np.float64)
        decoded, payload, report = self._run_round(
            source, method, atol, rtol, 0, self.original_encoding
        )
        report.cumulative_abs_error = report.max_abs_error
        report.cumulative_rel_error = report.max_rel_error
        report.cumulative_within_budget = report.within_budget
        self.history = [report]
        return payload, decoded, report

    def migrate(
        self,
        values: Union[Sequence[float], np.ndarray],
        methods: Sequence[str],
        budget: Budget,
        original_encoding: Optional[str] = None,
    ) -> MigrationResult:
        """
        Run several Numpress migration rounds and judge on cumulative error.

        Each round encodes the output of the previous round and decodes it
        again.  In addition to the per-round error, the error against the
        untouched original array is accumulated; the migration is accepted
        only if every round and every cumulative check stays inside the same
        budget.

        Args:
            values: One dimensional array of finite floats.
            methods: Ordered Numpress methods to apply, one entry per round.
            budget: Budget applied to every per-round and cumulative check.
            original_encoding: Optional identity of the source encoding;
                defaults to the instance's :attr:`original_encoding`.

        Returns:
            MigrationResult: Reports (including per-round and cumulative
            errors and the source encoding of every round) and the values
            after the last executed round.
        """
        methods = list(methods)
        if not methods:
            raise ValueError("at least one Numpress method is required")
        for method in methods:
            self._validate_method(method)
        atol, rtol = _normalize_budget(budget)

        original = np.asarray(values, dtype=np.float64)
        current = original
        source_encoding = original_encoding or self.original_encoding
        reports: List[RoundTripReport] = []
        accepted = True

        for round_index, method in enumerate(methods):
            decoded, _payload, report = self._run_round(
                current, method, atol, rtol, round_index, source_encoding
            )
            if decoded.size == original.size:
                cum_abs, cum_rel = _error_stats(original, decoded)
                cum_within = _within_budget(original, decoded, atol, rtol)
            else:
                cum_abs, cum_rel, cum_within = float("inf"), float("inf"), False
            report.cumulative_abs_error = cum_abs
            report.cumulative_rel_error = cum_rel
            report.cumulative_within_budget = cum_within

            reports.append(report)
            current = decoded
            source_encoding = report.target_encoding
            if not report.within_budget or not cum_within:
                accepted = False
                break

        self.history = reports
        return MigrationResult(
            accepted=accepted,
            reports=reports,
            values=current,
            original_values=original,
        )


def transcode(
    values: Union[Sequence[float], np.ndarray],
    method: str,
    budget: Budget,
    original_encoding: str = "64-bit float",
) -> Tuple[bytearray, np.ndarray, RoundTripReport]:
    """Convenience wrapper around :class:`NumpressTranscoder.transcode`."""
    return NumpressTranscoder(original_encoding=original_encoding).transcode(
        values, method, budget
    )


def migrate(
    values: Union[Sequence[float], np.ndarray],
    methods: Sequence[str],
    budget: Budget,
    original_encoding: str = "64-bit float",
) -> MigrationResult:
    """Convenience wrapper around :class:`NumpressTranscoder.migrate`."""
    return NumpressTranscoder(original_encoding=original_encoding).migrate(
        values, methods, budget
    )
