#!/usr/bin/python
# -*- coding: utf-8 -*-
import hashlib
import os

import pytest

from Registry import Registry
from Registry.RegistryParse import (
    DataFragment,
    DataStorage,
    MissingRange,
    SourceRange,
    parse_multisz,
)


def u16(s):
    return s.encode("utf-16le")


def make_storage(chunks, declared_length=None, padding=b""):
    """
    Build a DataStorage from (logical_offset, bytes, physical_offset) tuples.
    """
    chunks = list(chunks)
    data = b"".join(chunk[1] for chunk in chunks)
    if declared_length is None:
        declared_length = len(data)

    storage = DataStorage(declared_length, declared_length)
    storage.storage_kind = "db"
    storage.data = data
    storage.proven_prefix_length = len(data)
    storage.available_length = len(data)
    for logical_offset, chunk, physical_offset in chunks:
        storage.fragments.append(DataFragment(
            logical_offset, len(chunk), SourceRange(physical_offset, len(chunk)),
            "db-indirect", complete=True))
    if padding:
        storage.padding_start = 10000
        storage.padding_data = padding
        storage.padding_ranges.append(SourceRange(10000, len(padding)))
    return storage


def test_terminator_is_not_a_string():
    data = u16("one") + b"\x00\x00" + u16("two") + b"\x00\x00\x00\x00"
    value = parse_multisz(data)
    assert list(value) == ["one", "two"]
    assert value.terminated
    assert value.terminator_start == len(data) - 2
    assert value.terminator_end == len(data)
    assert value.residual == b""
    assert [item.value for item in value.items] == ["one", "two"]


def test_empty_sequence_has_no_items():
    value = parse_multisz(b"\x00\x00")
    assert list(value) == []
    assert value.terminated
    assert value.terminator_start == 0
    assert value.terminator_end == 2


def test_residual_is_not_parsed_as_new_items():
    # Once the empty-string terminator is seen, later text is residual only.
    data = u16("keep") + b"\x00\x00\x00\x00" + u16("must-not-appear") + b"\x00\x00"
    value = parse_multisz(data)
    assert list(value) == ["keep"]
    assert value.residual == u16("must-not-appear") + b"\x00\x00"
    assert "must-not-appear" not in list(value)


def test_missing_terminator_keeps_proven_prefix_and_reports_rest():
    data = u16("one") + b"\x00\x00" + u16("two")
    storage = make_storage([(0, data, 0x1000)])
    value = parse_multisz(storage.data, storage=storage)
    assert list(value) == ["one", "two"]
    assert not value.terminated
    assert value.problems
    assert any("terminator" in problem for problem in value.problems)


def test_odd_trailing_byte_is_unexplained():
    data = u16("one") + b"\x00\x00" + b"Z"
    value = parse_multisz(data)
    assert list(value) == ["one"]
    assert not value.terminated
    assert value.unexplained[-1].offset == len(data) - 1
    assert value.unexplained[-1].length == 1


def test_terminator_split_across_fragments():
    data = u16("alpha") + b"\x00\x00" + u16("beta") + b"\x00\x00\x00\x00"
    split = len(data) - 1
    chunks = [
        (0, data[:split], 0x2000),
        (split, data[split:], 0x4000),
    ]
    storage = make_storage(chunks)
    value = parse_multisz(storage.data, storage=storage)
    assert list(value) == ["alpha", "beta"]
    assert value.terminated
    terminator_ranges = [(r.offset, r.length) for r in value.terminator_ranges]
    assert terminator_ranges == [(0x2000 + split - 1, 1), (0x4000, 1)]


def test_missing_fragment_stops_at_proven_prefix():
    data = u16("alpha") + b"\x00\x00" + u16("beta") + b"\x00\x00"
    split = len(u16("alpha")) + 2 + 2
    storage = make_storage([(0, data[:split], 0x2000)])
    storage.unexplained.append(MissingRange(
        split, len(data) - split, "DB indirect fragment is outside the hive"))
    storage.problems.append("DB indirect fragment is outside the hive")

    value = parse_multisz(storage.data, storage=storage)
    assert list(value) == ["alpha"]
    assert not value.terminated
    assert any(gap.offset == split for gap in value.unexplained)


def test_padding_is_separate_from_value_data():
    data = u16("one") + b"\x00\x00"
    storage = make_storage([(0, data, 0x3000)], padding=b"\x00\x00\x00\x00")
    value = parse_multisz(storage.data, storage=storage)
    assert list(value) == ["one"]
    assert value.residual == b""
    assert value.padding == b"\x00\x00\x00\x00"
    assert value.padding_ranges


def test_repair_only_appends_a_missing_terminator():
    data = u16("one") + b"\x00\x00" + u16("two")
    value = parse_multisz(data)
    repair = value.repair()
    assert repair["applied"]
    assert repair["strings"] == ["one", "two"]
    assert repair["original"] == "6f006e0065000000740077006f00"

    already_terminated = parse_multisz(u16("one") + b"\x00\x00\x00\x00")
    assert not already_terminated.repair()["applied"]


def test_repair_is_refused_when_data_is_incomplete():
    data = u16("one")
    storage = make_storage([(0, data, 0x4000)], declared_length=len(data) + 4)
    storage.unexplained.append(MissingRange(len(data), 4, "fragment is missing"))
    storage.problems.append("declared length exceeds the DB indirect data")
    value = parse_multisz(storage.data, storage=storage)
    assert not value.repair()["applied"]


def test_fixture_multisz_has_no_phantom_terminator():
    path = os.path.join(os.path.dirname(__file__), "reg_samples", "UNICODE_TESTS")
    registry = Registry.Registry(path)
    key = registry.root().subkey("ASCII_KEY_NAME0")
    value = key.value("ASCII_VALUE_NAME8")
    parsed = value.value()
    assert parsed == [
        "ASCII_MULTI_VALUE1",
        "ASCII_MULTI_VALUE2",
        "ASCII_MULTI_VALUE3",
    ]
    assert parsed.terminated
    assert parsed.residual == b""
    assert parsed.storage.declared_length == 116
    assert parsed.items[0].ranges[0].offset == 0x13D4
    assert parsed.items[0].ranges[0].length == 36


def test_fixture_padding_is_reported_outside_declared_length():
    path = os.path.join(os.path.dirname(__file__), "reg_samples", "UNICODE_TESTS")
    registry = Registry.Registry(path)
    key = registry.root().subkey("ASCII_KEY_NAME0")
    parsed = key.value("ASCII_VALUE_NAME9").value()
    assert parsed.terminated
    assert parsed.storage.declared_length == 168
    assert parsed.padding == b"\x00\x00\x00\x00"
    assert parsed.padding_ranges[0].offset == 0x1478 + 4 + 168


def test_large_data_fragments_are_reported():
    path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "testing", "reg_samples", "new_log_1", "SYSTEM")
    if not os.path.exists(path):
        pytest.skip("SYSTEM sample is not available")

    registry = Registry.Registry(path)
    value = registry.open(
        r"ControlSet001\Control\ProductOptions").value("ProductPolicy")
    storage = value.data_storage()
    assert storage.storage_kind == "db"
    assert storage.declared_length == 23712
    assert len(storage.indirect_fragments) == 2
    assert [f.logical_offset for f in storage.indirect_fragments] == [0, 16344]
    assert hashlib.md5(storage.data).hexdigest() == \
        "7e96f10f77e1c44771d7045c24b94024"
    assert storage.padding_ranges
    assert [f.offset for f in storage.indirect_fragments] == [770084, 786468]
    assert [f.cell_offset for f in storage.indirect_fragments] == [770080, 786464]
    assert storage.db_offset == 764768
    assert storage.indirect_cell_offset == 764784
    assert value.raw_data() == storage.data


def test_large_data_reader_does_not_read_cell_headers():
    path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "testing", "reg_samples", "new_log_1", "SYSTEM")
    if not os.path.exists(path):
        pytest.skip("SYSTEM sample is not available")

    registry = Registry.Registry(path)
    value = registry.open(
        r"ControlSet001\Control\ProductOptions").value("ProductPolicy")
    raw = value.raw_data()
    assert raw.startswith(b"\xa0\x5c\x00\x00")
    assert raw[0x3fd8:0x3fd8 + 4] == b"\x98\x00\x82\x00"
    assert not raw.startswith(b"\x00\x08\x00\x00")
