import datetime
from pathlib import Path

import numpy as np
import pytest

from edfio import (
    Edf,
    EdfAnnotation,
    EdfSignal,
    Patient,
    Recording,
    StudyAnonymizer,
)
from edfio.anonymization import ReviewItem, _date_renderings
from edfio.edf_header import AnonymizedDateError

STARTDATE = datetime.date(2024, 3, 10)
STARTTIME = datetime.time(22, 30, 15)
BIRTHDATE = datetime.date(1980, 5, 17)


def _make_edf(
    *,
    startdate: datetime.date = STARTDATE,
    patient: Patient | None = None,
    recording: Recording | None = None,
    annotations: list[EdfAnnotation] | None = None,
) -> Edf:
    if patient is None:
        patient = Patient(
            code="P001",
            sex="F",
            birthdate=BIRTHDATE,
            name="Doe_Jane",
            additional=["ward-4B"],
        )
    if recording is None:
        recording = Recording(
            startdate=startdate,
            hospital_administration_code="PSG-778",
            investigator_technician_code="TECH9",
            equipment_code="EQ3",
            additional=["room_12"],
        )
    signals = [EdfSignal(np.arange(300, dtype=float), 100)]
    if annotations is None:
        return Edf(signals, patient=patient, recording=recording, starttime=STARTTIME)
    return Edf(
        signals,
        patient=patient,
        recording=recording,
        starttime=STARTTIME,
        annotations=annotations,
    )


def test_same_subject_sessions_keep_interval_and_stay_linkable():
    anonymizer = StudyAnonymizer(seed=42)
    edf_night1 = _make_edf(startdate=datetime.date(2024, 3, 10))
    edf_night2 = _make_edf(startdate=datetime.date(2024, 3, 17))

    report1 = anonymizer.anonymize(edf_night1, "subject-A")
    report2 = anonymizer.anonymize(edf_night2, "subject-A")

    assert report1.shift_days == report2.shift_days
    assert report1.shift_days != 0
    assert abs(report1.shift_days) <= 365
    assert report1.pseudonym == report2.pseudonym
    assert edf_night1.patient.code == report1.pseudonym
    assert edf_night2.patient.code == report1.pseudonym
    assert (edf_night2.startdate - edf_night1.startdate).days == 7
    assert edf_night1.startdate == STARTDATE + datetime.timedelta(
        days=report1.shift_days
    )
    assert edf_night1.startdatetime == report1.startdatetime


def test_start_time_of_day_is_kept_by_default():
    anonymizer = StudyAnonymizer(seed=42)
    edf = _make_edf()
    anonymizer.anonymize(edf, "subject-A")
    assert edf.starttime == STARTTIME


def test_patient_and_recording_fields_are_scrubbed():
    anonymizer = StudyAnonymizer(seed=42)
    edf = _make_edf()
    report = anonymizer.anonymize(edf, "subject-A")

    assert edf.patient.code == report.pseudonym
    assert report.pseudonym.startswith("SUBJ-")
    assert edf.patient.sex == "X"
    assert edf.patient.name == "X"
    assert edf.patient.additional == ()
    with pytest.raises(AnonymizedDateError):
        edf.patient.birthdate

    shifted = STARTDATE + datetime.timedelta(days=report.shift_days)
    assert edf.recording.startdate == shifted
    assert edf.recording.hospital_administration_code == "X"
    assert edf.recording.investigator_technician_code == "X"
    assert edf.recording.equipment_code == "X"
    assert edf.recording.additional == ()


def test_free_text_header_subfields_are_removed_and_flagged():
    anonymizer = StudyAnonymizer(seed=42)
    edf = _make_edf()
    report = anonymizer.anonymize(edf, "subject-A")

    assert ReviewItem("patient", "ward-4B", "free-text patient subfield removed") in (
        report.review_items
    )
    assert ReviewItem(
        "recording", "room_12", "free-text recording subfield removed"
    ) in (report.review_items)


def test_unknown_annotations_are_removed_and_flagged_for_review():
    anonymizer = StudyAnonymizer(seed=42, allowed_annotations=["Sleep stage W"])
    edf = _make_edf(
        annotations=[
            EdfAnnotation(0.5, None, "Sleep stage W"),
            EdfAnnotation(1.0, None, "patient called his wife"),
        ]
    )
    report = anonymizer.anonymize(edf, "subject-A")

    assert [a.text for a in edf.annotations] == ["Sleep stage W"]
    assert (
        ReviewItem("annotation", "patient called his wife", "unknown free text")
        in report.review_items
    )


def test_confirmed_annotations_are_kept():
    anonymizer = StudyAnonymizer(seed=42)
    edf = _make_edf(annotations=[EdfAnnotation(0.5, None, "technician note: ok")])
    report = anonymizer.anonymize(
        edf, "subject-A", confirmed_annotations=["technician note: ok"]
    )

    assert [a.text for a in edf.annotations] == ["technician note: ok"]
    assert [item for item in report.review_items if item.source == "annotation"] == []


def test_annotation_matching_original_date_is_removed_and_flagged():
    anonymizer = StudyAnonymizer(seed=42)
    edf = _make_edf(annotations=[EdfAnnotation(0.5, None, "recorded on 2024-03-10")])
    report = anonymizer.anonymize(edf, "subject-A")

    assert edf.annotations == ()
    assert ReviewItem(
        "annotation",
        "recorded on 2024-03-10",
        "contains a date matching the recording date or birthdate",
    ) in (report.review_items)


@pytest.mark.parametrize("text", ["visit on 2024-03-10", "born 17.05.1980"])
def test_kept_annotation_with_residual_date_raises(text: str):
    anonymizer = StudyAnonymizer(seed=42, allowed_annotations=[text])
    edf = _make_edf(annotations=[EdfAnnotation(0.5, None, text)])
    with pytest.raises(ValueError, match="inferring the date shift"):
        anonymizer.anonymize(edf, "subject-A")


def test_anonymize_already_anonymized_startdate_raises():
    edf = _make_edf()
    edf.anonymize()
    with pytest.raises(ValueError, match="startdate is not available"):
        StudyAnonymizer(seed=42).anonymize(edf, "subject-A")


def test_seed_makes_pseudonym_and_shift_reproducible():
    edf1 = _make_edf()
    edf2 = _make_edf()
    report1 = StudyAnonymizer(seed="my-study").anonymize(edf1, "subject-A")
    report2 = StudyAnonymizer(seed="my-study").anonymize(edf2, "subject-A")
    assert report1.pseudonym == report2.pseudonym
    assert report1.shift_days == report2.shift_days
    assert edf1.startdate == edf2.startdate


def test_different_seeds_give_different_pseudonyms():
    report1 = StudyAnonymizer(seed=1).anonymize(_make_edf(), "subject-A")
    report2 = StudyAnonymizer(seed=2).anonymize(_make_edf(), "subject-A")
    assert report1.pseudonym != report2.pseudonym


@pytest.mark.parametrize("seed", [None, 7, "study", b"key"])
def test_supported_seed_types(seed):
    edf = _make_edf()
    report = StudyAnonymizer(seed=seed).anonymize(edf, "subject-A")
    assert report.pseudonym.startswith("SUBJ-")


def test_shifts_are_nonzero_and_within_range():
    anonymizer = StudyAnonymizer(seed=42, max_shift_days=30)
    for i in range(20):
        _, shift_days = anonymizer._subject_params(f"subject-{i}")
        assert 0 < abs(shift_days) <= 30


@pytest.mark.parametrize("max_shift_days", [0, -10])
def test_invalid_max_shift_days_raises(max_shift_days: int):
    with pytest.raises(ValueError, match="max_shift_days must be positive"):
        StudyAnonymizer(max_shift_days=max_shift_days)


def test_keep_sex():
    edf = _make_edf()
    StudyAnonymizer(seed=42, keep_sex=True).anonymize(edf, "subject-A")
    assert edf.patient.sex == "F"


def test_keep_starttime_false():
    edf = _make_edf()
    StudyAnonymizer(seed=42, keep_starttime=False).anonymize(edf, "subject-A")
    assert edf.starttime == datetime.time(0, 0, 0)


def test_patient_without_birthdate():
    edf = _make_edf(patient=Patient(code="P002", sex="M"))
    report = StudyAnonymizer(seed=42).anonymize(edf, "subject-A")
    assert edf.patient.code == report.pseudonym
    with pytest.raises(AnonymizedDateError):
        edf.patient.birthdate


def test_recording_without_annotations_stays_plain_edf():
    edf = _make_edf()
    StudyAnonymizer(seed=42).anonymize(edf, "subject-A")
    assert edf.reserved == ""
    assert edf.annotations == ()


def test_no_residual_dates_in_written_file(tmp_file: Path):
    edf = _make_edf(annotations=[EdfAnnotation(0.5, None, "Sleep stage W")])
    StudyAnonymizer(seed=42, allowed_annotations=["Sleep stage W"]).anonymize(
        edf, "subject-A"
    )
    edf.write(tmp_file)
    content = tmp_file.read_bytes()
    for date in (STARTDATE, BIRTHDATE):
        for rendering in _date_renderings(date):
            assert rendering.encode() not in content
            assert rendering.upper().encode() not in content.upper()
    for identity in (b"P001", b"Doe_Jane", b"PSG-778", b"TECH9", b"ward-4B"):
        assert identity not in content
