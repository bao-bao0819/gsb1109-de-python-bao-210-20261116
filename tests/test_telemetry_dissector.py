from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from telemetry_dissector import (
    DEMO_DECLARATION,
    DEMO_PACKET,
    BitReader,
    DeclarationError,
    FragmentedReader,
    InsufficientBitsError,
    dissect_packet,
    load_declaration,
    main,
    parse_declaration,
)


def _fields_by_name(result: dict) -> dict:
    return {field["name"]: field for field in result["fields"] if field["kind"] == "field"}


def test_demo_dissection_values() -> None:
    result = dissect_packet(DEMO_DECLARATION, DEMO_PACKET)

    assert result["name"] == "telemetry"
    assert result["complete"] is True
    assert result["next_bit_offset"] == 88

    fields = _fields_by_name(result)
    assert fields["sync"]["value"] == 0x1ACF
    assert fields["sequence"]["value"] == 0x3412
    assert fields["sensor_id"]["value"] == 0xABC
    assert fields["state"]["value"] == 0xD
    assert fields["temperature"]["value"] == -1234
    assert fields["flags"]["value"] == 0b101
    assert fields["payload"]["value"] == "aa55"


def test_demo_bit_ranges() -> None:
    result = dissect_packet(DEMO_DECLARATION, DEMO_PACKET)
    fields = _fields_by_name(result)

    assert fields["sync"]["bit_segments"] == [[0, 16]]
    assert fields["sequence"]["bit_segments"] == [[16, 16]]
    assert fields["sensor_id"]["bit_segments"] == [[32, 12]]
    assert fields["sensor_id"]["bit_offset"] == 32
    assert fields["state"]["bit_segments"] == [[44, 4]]
    assert fields["temperature"]["bit_segments"] == [[48, 16]]
    assert fields["flags"]["bit_segments"] == [[69, 3]]
    assert fields["payload"]["bit_segments"] == [[72, 16]]

    # Bit fields report the storage unit they came from
    assert fields["sensor_id"]["byte_offset"] == 4
    assert fields["sensor_id"]["byte_length"] == 2
    assert fields["state"]["byte_offset"] == 4


def test_cross_byte_bitfield_little_endian_segments() -> None:
    declaration = {
        "endian": "little",
        "fields": [
            {"name": "low", "type": "uint16", "bits": 4},
            {"name": "high", "type": "uint16", "bits": 12},
        ],
    }
    # Storage unit 0xABCD, little endian on the wire; low takes its bottom 4 bits
    result = dissect_packet(declaration, bytes.fromhex("cdab"))

    fields = _fields_by_name(result)
    assert fields["low"]["value"] == 0xD
    assert fields["low"]["bit_segments"] == [[4, 4]]
    # The 12-bit field straddles the byte boundary: its source bits are not contiguous
    assert fields["high"]["value"] == 0xABC
    assert fields["high"]["bit_segments"] == [[0, 4], [8, 8]]
    assert fields["high"]["bit_offset"] is None
    assert fields["high"]["bit_length"] == 12


def test_endianness_is_not_whole_packet_reversal() -> None:
    declaration = {
        "fields": [
            {"name": "a", "type": "uint16", "endian": "big"},
            {"name": "b", "type": "uint16", "endian": "little"},
        ]
    }
    packet = bytes.fromhex("12345678")

    fields = _fields_by_name(dissect_packet(declaration, packet))
    assert fields["a"]["value"] == 0x1234
    assert fields["b"]["value"] == 0x7856

    # Reversing every byte of the packet is not the same as swapping a field's byte order
    reversed_fields = _fields_by_name(dissect_packet(declaration, packet[::-1]))
    assert reversed_fields["a"]["value"] != fields["a"]["value"]
    assert reversed_fields["b"]["value"] != fields["b"]["value"]


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 7, 64])
def test_fragmented_input_matches_one_shot(chunk_size: int) -> None:
    whole = dissect_packet(DEMO_DECLARATION, DEMO_PACKET)
    fragmented = dissect_packet(DEMO_DECLARATION, FragmentedReader(io.BytesIO(DEMO_PACKET), chunk_size))

    # Values, source bit ranges and the next field start are all identical
    assert fragmented == whole
    assert fragmented["next_bit_offset"] == whole["next_bit_offset"] == 88


def test_fragmented_truncated_packet_matches_one_shot() -> None:
    packet = DEMO_PACKET[:5]
    whole = dissect_packet(DEMO_DECLARATION, packet)
    fragmented = dissect_packet(DEMO_DECLARATION, FragmentedReader(io.BytesIO(packet), 1))

    assert fragmented == whole


def test_insufficient_bits_state_is_explicit() -> None:
    # Cut the packet inside the sensor_id/state bit group (needs bytes 4-5, only byte 4 left)
    result = dissect_packet(DEMO_DECLARATION, DEMO_PACKET[:5])

    assert result["complete"] is False
    assert result["next_bit_offset"] == 40

    last = result["fields"][-1]
    assert last["name"] == "sensor_id"
    assert last["status"] == "insufficient-bits"
    assert last["value"] is None
    assert last["expected_bits"] == 16
    assert last["available_bits"] == 8

    # Fields before the truncation point are unaffected
    fields = _fields_by_name(result)
    assert fields["sync"]["value"] == 0x1ACF
    assert fields["sequence"]["value"] == 0x3412


def test_partial_bit_group_reports_padding() -> None:
    # The flags group declares 3 of the 8 bits of its storage unit
    result = dissect_packet(DEMO_DECLARATION, DEMO_PACKET)

    padding = [field for field in result["fields"] if field["kind"] == "padding"]
    assert len(padding) == 1
    assert padding[0]["status"] == "unused"
    assert padding[0]["bits"] == 5
    assert padding[0]["bit_segments"] == [[64, 5]]

    # The next field still starts after the whole storage unit
    fields = _fields_by_name(result)
    assert fields["payload"]["byte_offset"] == 9
    assert result["next_bit_offset"] == 88


def test_full_bit_group_has_no_padding() -> None:
    result = dissect_packet(
        {"fields": [{"name": "nibble", "type": "uint8", "bits": 4}, {"name": "rest", "type": "uint8", "bits": 4}]},
        b"\x5a",
    )

    assert result["complete"] is True
    assert all(field["kind"] == "field" for field in result["fields"])
    fields = _fields_by_name(result)
    assert fields["nibble"]["value"] == 0x5
    assert fields["rest"]["value"] == 0xA


def test_new_storage_unit_on_type_change() -> None:
    declaration = {
        "endian": "big",
        "fields": [
            {"name": "a", "type": "uint8", "bits": 8},
            {"name": "b", "type": "uint16", "bits": 16},
        ],
    }
    result = dissect_packet(declaration, bytes.fromhex("ffabcd"))

    fields = _fields_by_name(result)
    assert fields["a"]["value"] == 0xFF
    assert fields["b"]["value"] == 0xABCD
    assert fields["b"]["byte_offset"] == 1


def test_bit_reader_reports_partial_consumption() -> None:
    reader = BitReader(io.BytesIO(b"\xff"))
    with pytest.raises(InsufficientBitsError) as exc_info:
        reader.read(2)

    assert exc_info.value.expected_bits == 16
    assert exc_info.value.available_bits == 8
    assert exc_info.value.position == 8
    assert reader.bit_position == 8


def test_result_is_json_serializable() -> None:
    result = dissect_packet(DEMO_DECLARATION, DEMO_PACKET)
    assert json.loads(json.dumps(result)) == result


def test_parse_declaration_rejects_straddled_bit_group() -> None:
    with pytest.raises(DeclarationError, match="straddles"):
        parse_declaration(
            {"fields": [{"name": "a", "type": "uint8", "bits": 5}, {"name": "b", "type": "uint8", "bits": 4}]}
        )


def test_parse_declaration_rejects_unknown_type() -> None:
    with pytest.raises(DeclarationError, match="unknown type"):
        parse_declaration({"fields": [{"name": "a", "type": "nonsense"}]})


def test_parse_declaration_rejects_non_integer_type() -> None:
    with pytest.raises(DeclarationError, match="not a fixed-size integer"):
        parse_declaration({"fields": [{"name": "a", "type": "float"}]})


def test_parse_declaration_rejects_invalid_bits() -> None:
    with pytest.raises(DeclarationError, match="'bits'"):
        parse_declaration({"fields": [{"name": "a", "type": "uint8", "bits": 9}]})


def test_load_declaration_from_file_and_string(tmp_path: Path) -> None:
    path = tmp_path / "declaration.json"
    path.write_text(json.dumps(DEMO_DECLARATION), encoding="utf-8")

    assert load_declaration(path) == DEMO_DECLARATION
    assert load_declaration(str(path)) == DEMO_DECLARATION
    assert load_declaration(json.dumps(DEMO_DECLARATION)) == DEMO_DECLARATION
    assert load_declaration(DEMO_DECLARATION) is DEMO_DECLARATION


def test_main_demo(capsys: pytest.CaptureFixture) -> None:
    assert main(["--demo"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["name"] == "telemetry"
    assert out["complete"] is True
    assert out["next_bit_offset"] == 88


def test_main_demo_fragmented_matches_one_shot(capsys: pytest.CaptureFixture) -> None:
    assert main(["--demo"]) == 0
    whole = capsys.readouterr().out

    assert main(["--demo", "--chunk-size", "1"]) == 0
    fragmented = capsys.readouterr().out

    assert fragmented == whole


def test_main_with_declaration_file_and_hex_packet(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    declaration_path = tmp_path / "declaration.json"
    declaration_path.write_text(json.dumps(DEMO_DECLARATION), encoding="utf-8")

    assert main([str(declaration_path), DEMO_PACKET.hex()]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["complete"] is True


def test_main_reports_incomplete_via_exit_code(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    declaration_path = tmp_path / "declaration.json"
    declaration_path.write_text(json.dumps(DEMO_DECLARATION), encoding="utf-8")

    assert main([str(declaration_path), DEMO_PACKET[:5].hex()]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["complete"] is False
