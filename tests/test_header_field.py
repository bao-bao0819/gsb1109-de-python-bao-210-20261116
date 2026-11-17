import pytest

from edfio._header_field import (
    decode_date,
    decode_float,
    decode_str,
    decode_time,
    encode_float,
    encode_int,
    is_encodable_float,
)


def test_encode_int_exceeding_field_length_fails():
    with pytest.raises(ValueError, match="exceeds maximum field length"):
        encode_int(100000000000000000, 8)


def test_encode_float_exceeding_field_length_fails():
    with pytest.raises(ValueError, match="exceeds maximum field length"):
        encode_float(123456789.12345)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1.234, True),
        (1.234567, True),
        (12345678, True),
        (0.3333333333333333, False),
        (123456789.0, False),
        (2.25e16, False),
        (0.1, True),
        (50.0, True),
    ],
)
def test_is_encodable_float(*, value: float, expected: bool):
    assert is_encodable_float(value) is expected


def test_is_encodable_float_agrees_with_encode_float():
    for value in (1.234, 12345678, 0.3333333333333333, 123456789.0, 2.25e16):
        if is_encodable_float(value):
            encode_float(value)
        else:
            with pytest.raises(ValueError, match="exceeds maximum field length"):
                encode_float(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1.234, b"1.234   "),
        (1.234567, b"1.234567"),
        (12345678, b"12345678"),
    ],
)
def test_encode_float(value: float, expected: bytes):
    assert encode_float(value) == expected


def test_decode_str_replaces_non_ascii_characters():
    assert decode_str("è".encode("latin-1")) == "�"


def test_decode_str_with_latin_1_encoding():
    assert decode_str("è".encode("latin-1"), "latin-1") == "è"


@pytest.mark.parametrize("field", [b"1E2345  ", b"-1E2345 "])
def test_decode_float_exceeding_float_range_fails(field: bytes):
    with pytest.raises(ValueError, match="outside float range"):
        decode_float(field)


@pytest.mark.parametrize(
    "date",
    [
        b"        ",
        b"2.8.2051",
    ],
)
def test_date_decode_invalid_format(date: bytes):
    with pytest.raises(ValueError, match="Invalid date for format"):
        decode_date(date)


def test_date_decode_invalid_day():
    with pytest.raises(ValueError, match="day is out of range|day 32 must be in range"):
        decode_date(b"32.08.51")


def test_date_decode_invalid_month():
    with pytest.raises(ValueError, match="month must be in 1..12"):
        decode_date(b"02.13.51")


def test_time_decode_invalid_format():
    with pytest.raises(ValueError, match="Invalid time for format"):
        decode_time(b"02_08_51")
