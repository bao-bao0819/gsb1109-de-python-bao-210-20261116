"""
Batch anonymization of multi-session EDF/BDF recordings.

This module provides :class:`StudyAnonymizer` for anonymizing recordings of a
group of subjects, where each subject may contribute several sessions (e.g.,
multiple nights). In contrast to :meth:`Edf.anonymize`, which anonymizes a
single recording in isolation, the study anonymizer

- shifts the recording dates of each subject by a random offset that is
  identical for all sessions of that subject, preserving intervals between
  sessions,
- assigns a stable pseudonym to each subject, keeping their sessions linkable,
- checks patient fields, recording fields, and annotation texts for identity
  content, and
- ensures that no residual dates remain that would allow inferring the shift.
"""

from __future__ import annotations

import datetime
import hashlib
import hmac
import secrets
from collections.abc import Iterable
from dataclasses import dataclass

from edfio.edf import Bdf, Edf
from edfio.edf_annotations import EdfAnnotation
from edfio.edf_header import AnonymizedDateError, Patient, Recording

_MONTH_NAMES = (
    "JAN",
    "FEB",
    "MAR",
    "APR",
    "MAY",
    "JUN",
    "JUL",
    "AUG",
    "SEP",
    "OCT",
    "NOV",
    "DEC",
)


def _date_renderings(date: datetime.date) -> tuple[str, ...]:
    return (
        date.isoformat(),
        f"{date.day:02}.{date.month:02}.{date.year}",
        f"{date.day:02}.{date.month:02}.{date.year % 100:02}",
        f"{date.day}.{date.month}.{date.year}",
        f"{date.month:02}/{date.day:02}/{date.year}",
        f"{date.month:02}/{date.day:02}/{date.year % 100:02}",
        f"{date.day:02}-{_MONTH_NAMES[date.month - 1]}-{date.year}",
    )


def _contains_any_rendering(text: str, dates: Iterable[datetime.date]) -> bool:
    text_upper = text.upper()
    return any(
        rendering.upper() in text_upper
        for date in dates
        for rendering in _date_renderings(date)
    )


@dataclass(frozen=True)
class ReviewItem:
    """
    A piece of free text removed during anonymization that requires manual review.

    Parameters
    ----------
    source : str
        Where the text was found: `"annotation"`, `"patient"`, or `"recording"`.
    text : str
        The removed text.
    reason : str
        Why the text was removed.
    """

    source: str
    text: str
    reason: str


@dataclass(frozen=True)
class SessionAnonymizationReport:
    """
    Summary of the anonymization of a single recording.

    The report contains the mapping between the original subject identifier and
    the assigned pseudonym as well as the applied date shift. It must therefore
    be stored separately from the anonymized data.

    Parameters
    ----------
    subject_id : str
        The original subject identifier passed to :meth:`StudyAnonymizer.anonymize`.
    pseudonym : str
        The pseudonym assigned to the subject, stored in the patient code.
    shift_days : int
        The date shift applied to the recording, in days.
    startdatetime : datetime.datetime
        The new (shifted) recording start.
    review_items : tuple[ReviewItem, ...]
        Free-text fragments that were removed and require manual review.
    """

    subject_id: str
    pseudonym: str
    shift_days: int
    startdatetime: datetime.datetime
    review_items: tuple[ReviewItem, ...] = ()


class StudyAnonymizer:
    """
    Anonymize multi-session recordings of a group of subjects.

    For each subject, a random date shift (in whole days) is derived once and
    applied to all of the subject's sessions, so intervals between sessions of
    the same subject are preserved. The subject is assigned a stable pseudonym
    that is stored in the patient code, keeping sessions of the same subject
    linkable after anonymization.

    Patient and recording fields are replaced with anonymized values, dropping
    any free-text subfields. Annotation texts are checked against
    `allowed_annotations`; unknown free text is removed and returned in the
    report for manual review. After anonymization, the remaining annotation
    texts are checked for residual renderings of the original recording date or
    birthdate, which would allow inferring the date shift.

    Parameters
    ----------
    max_shift_days : int, default: 365
        Maximum absolute value of the random date shift in days. The shift is
        always non-zero.
    seed : int | str | bytes | None, default: None
        Seed for deriving pseudonyms and date shifts. The same seed and subject
        identifier always yield the same pseudonym and shift, allowing
        reproducible anonymization across runs. If None, a random key is
        generated, and pseudonyms are only stable within this instance.
    keep_sex : bool, default: False
        Whether to keep the sex field in the local patient identification.
    keep_starttime : bool, default: True
        Whether to keep the original start time of day. Since dates are shifted
        by whole days, keeping the time of day preserves circadian information.
    allowed_annotations : Iterable[str], default: ()
        Annotation texts known to be free of identifying information (e.g.,
        sleep stage labels), which are kept without confirmation.
    """

    def __init__(
        self,
        *,
        max_shift_days: int = 365,
        seed: int | str | bytes | None = None,
        keep_sex: bool = False,
        keep_starttime: bool = True,
        allowed_annotations: Iterable[str] = (),
    ) -> None:
        if max_shift_days < 1:
            raise ValueError(f"max_shift_days must be positive, got {max_shift_days}")
        self._max_shift_days = max_shift_days
        if seed is None:
            self._key = secrets.token_bytes(32)
        elif isinstance(seed, bytes):
            self._key = seed
        else:
            self._key = str(seed).encode("utf-8")
        self._keep_sex = keep_sex
        self._keep_starttime = keep_starttime
        self._allowed_annotations = frozenset(allowed_annotations)

    def _subject_params(self, subject_id: str) -> tuple[str, int]:
        digest = hmac.new(
            self._key, subject_id.encode("utf-8"), hashlib.sha256
        ).digest()
        pseudonym = f"SUBJ-{digest[:8].hex()}"
        shift_value = int.from_bytes(digest[8:12], "big") % (2 * self._max_shift_days)
        shift_days = shift_value - self._max_shift_days
        if shift_value >= self._max_shift_days:
            shift_days += 1
        return pseudonym, shift_days

    def _review_annotations(
        self,
        edf: Edf | Bdf,
        sensitive_dates: set[datetime.date],
        confirmed: set[str],
    ) -> tuple[list[EdfAnnotation], list[ReviewItem]]:
        kept = []
        review_items = []
        for annotation in edf.annotations:
            if (
                annotation.text in self._allowed_annotations
                or annotation.text in confirmed
            ):
                kept.append(annotation)
                continue
            if _contains_any_rendering(annotation.text, sensitive_dates):
                reason = "contains a date matching the recording date or birthdate"
            else:
                reason = "unknown free text"
            review_items.append(ReviewItem("annotation", annotation.text, reason))
        return kept, review_items

    def anonymize(
        self,
        edf: Edf | Bdf,
        subject_id: str,
        *,
        confirmed_annotations: Iterable[str] = (),
    ) -> SessionAnonymizationReport:
        """
        Anonymize a single recording of the given subject, in place.

        Parameters
        ----------
        edf : Edf | Bdf
            The recording to anonymize. Must have a known (not previously
            anonymized) startdate.
        subject_id : str
            Identifier grouping sessions of the same subject. All calls with
            the same `subject_id` receive the same pseudonym and date shift.
            The identifier itself is never written to the recording.
        confirmed_annotations : Iterable[str], default: ()
            Additional annotation texts that have been manually reviewed and
            confirmed to be free of identifying information.

        Returns
        -------
        SessionAnonymizationReport
            The subject mapping, applied shift, and free-text fragments removed
            for manual review.

        Raises
        ------
        ValueError
            If the recording startdate is not available (e.g., already
            anonymized), or if a kept annotation contains a residual rendering
            of the original recording date or birthdate.
        """
        pseudonym, shift_days = self._subject_params(subject_id)
        try:
            original_start = edf.startdatetime
        except AnonymizedDateError:
            raise ValueError(
                "Cannot shift dates of a recording whose startdate is not available"
                " (already anonymized?)"
            ) from None
        new_start = original_start + datetime.timedelta(days=shift_days)

        sensitive_dates = {original_start.date()}
        try:
            sensitive_dates.add(edf.patient.birthdate)
        except AnonymizedDateError:
            pass

        review_items = []
        for text in edf.patient.additional:
            review_items.append(
                ReviewItem("patient", text, "free-text patient subfield removed")
            )
        for text in edf.recording.additional:
            review_items.append(
                ReviewItem("recording", text, "free-text recording subfield removed")
            )

        confirmed = set(confirmed_annotations)
        kept_annotations, annotation_review_items = self._review_annotations(
            edf, sensitive_dates, confirmed
        )
        review_items.extend(annotation_review_items)

        for annotation in kept_annotations:
            if _contains_any_rendering(annotation.text, sensitive_dates):
                raise ValueError(
                    f"Kept annotation {annotation.text!r} contains a date matching"
                    " the original recording date or birthdate, which would allow"
                    " inferring the date shift"
                )

        sex = edf.patient.sex if self._keep_sex else "X"
        edf.patient = Patient(code=pseudonym, sex=sex)  # type: ignore[arg-type]
        edf.recording = Recording(startdate=new_start.date())
        if not self._keep_starttime:
            edf.starttime = datetime.time(0, 0, 0)
        if len(kept_annotations) != len(edf.annotations):
            edf.set_annotations(kept_annotations)

        return SessionAnonymizationReport(
            subject_id=subject_id,
            pseudonym=pseudonym,
            shift_days=shift_days,
            startdatetime=edf.startdatetime,
            review_items=tuple(review_items),
        )
