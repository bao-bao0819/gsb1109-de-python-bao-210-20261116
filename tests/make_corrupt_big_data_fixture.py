#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Build a small registry hive whose big-data (db) values are deliberately damaged.

The generated hive contains one root key with four values:

  Small        - an ordinary, fully recoverable inline data cell (RegBin, 16 bytes)
  IntactBig    - a healthy db value spread over three non-adjacent data cells
  BrokenBig    - a db value whose last indirect-block pointer is invalid; only a
                 definite prefix can be recovered
  DupBig       - a db value whose indirect block contains a duplicate reference
                 and is shorter than the length declared by the VK record
  TruncatedBig - a db value whose indirect block runs out of fragment
                 references before the VK-declared length is satisfied

The data cells are never physically adjacent: HBIN headers and filler cells sit
between them, and a decoy cell follows the last fragment. A correct recovery
must follow the indirect-block pointers only and must never continue reading
into the physically neighboring cell.

Re-run with:  .venv/bin/python tests/make_corrupt_big_data_fixture.py
"""
import os
import struct

REGF = 0x1000
HBIN_HEADER_SIZE = 0x20
CELL_DATA_OFFSET = 0x4
MAX_DATA_SEGMENT = 0x3FD8
DATA_CELL_SIZE = MAX_DATA_SEGMENT + CELL_DATA_OFFSET  # 0x3FE0

RegBin = 0x0003


def align8(size):
    return (size + 7) & ~7


class HiveBuilder(object):
    def __init__(self):
        self.pages = []
        self.cell_rel_offsets = {}

    def add_hbin(self, size):
        page = bytearray(size)
        page[0:4] = b"hbin"
        struct.pack_into("<I", page, 0x4, 0)
        struct.pack_into("<I", page, 0x8, size)
        struct.pack_into("<I", page, 0xC, size)
        self.pages.append(page)
        return len(self.pages) - 1

    def add_cell(self, page_index, name, payload):
        page = self.pages[page_index]
        size = align8(CELL_DATA_OFFSET + len(payload))
        cursor = HBIN_HEADER_SIZE
        while cursor < len(page) and page[cursor:cursor + 4] != b"\x00\x00\x00\x00":
            cursor += abs(struct.unpack_from("<i", page, cursor)[0])
        if cursor + size > len(page):
            raise RuntimeError("no room for cell %s in hbin %d" % (name, page_index))
        struct.pack_into("<i", page, cursor, -size)
        page[cursor + CELL_DATA_OFFSET:cursor + CELL_DATA_OFFSET + len(payload)] = payload
        hbin_base = REGF + sum(len(p) for p in self.pages[:page_index])
        rel = (hbin_base - REGF) + cursor
        self.cell_rel_offsets[name] = rel
        return rel

    def build(self):
        hbins_size = sum(len(p) for p in self.pages)
        regf = bytearray(REGF)
        regf[0:4] = b"regf"
        struct.pack_into("<I", regf, 0x4, 1)      # sequence 1
        struct.pack_into("<I", regf, 0x8, 1)      # sequence 2
        struct.pack_into("<I", regf, 0x14, 1)     # major version
        struct.pack_into("<I", regf, 0x18, 5)     # minor version
        struct.pack_into("<I", regf, 0x1C, 0)     # primary file
        struct.pack_into("<I", regf, 0x20, 1)     # format
        struct.pack_into("<I", regf, 0x24, self.cell_rel_offsets["nk_root"])
        struct.pack_into("<I", regf, 0x28, hbins_size)
        struct.pack_into("<I", regf, 0x2C, 1)     # clustering factor
        hive_name = "corrupt_big_data".encode("utf-16le")
        regf[0x30:0x30 + len(hive_name)] = hive_name

        checksum = 0
        for offset in range(0, 0x1FC, 4):
            checksum ^= struct.unpack_from("<I", regf, offset)[0]
        if checksum == 0:
            checksum = 1
        elif checksum == 0xFFFFFFFF:
            checksum = 0xFFFFFFFE
        struct.pack_into("<I", regf, 0x1FC, checksum)

        return bytes(regf + b"".join(bytes(p) for p in self.pages))


def make_nk(name, values_list_rel, values_number):
    payload = bytearray(0x4C)
    payload[0:2] = b"nk"
    struct.pack_into("<H", payload, 0x2, 0x24)   # root key, ascii name
    struct.pack_into("<I", payload, 0x10, 0xFFFFFFFF)
    struct.pack_into("<I", payload, 0x24, values_number)
    struct.pack_into("<I", payload, 0x28, values_list_rel)
    struct.pack_into("<I", payload, 0x2C, 0xFFFFFFFF)
    struct.pack_into("<I", payload, 0x30, 0xFFFFFFFF)
    struct.pack_into("<H", payload, 0x48, len(name))
    payload += name.encode("ascii")
    return bytes(payload)


def make_vk(name, data_length, data_rel, data_type=RegBin):
    payload = bytearray(0x14)
    payload[0:2] = b"vk"
    struct.pack_into("<H", payload, 0x2, len(name))
    struct.pack_into("<I", payload, 0x4, data_length)
    struct.pack_into("<I", payload, 0x8, data_rel)
    struct.pack_into("<I", payload, 0xC, data_type)
    struct.pack_into("<H", payload, 0x10, 0x1)   # ascii name
    payload += name.encode("ascii")
    return bytes(payload)


def make_db(indirect_rel):
    payload = bytearray(0x8)
    payload[0:2] = b"db"
    struct.pack_into("<I", payload, 0x4, indirect_rel)
    return bytes(payload)


def make_indirect(refs):
    payload = bytearray()
    for ref in refs:
        payload += struct.pack("<I", ref)
    return bytes(payload)


def fragment_payload(seed):
    return bytes(((index * 7 + seed) & 0xFF) for index in range(MAX_DATA_SEGMENT))


def main():
    builder = HiveBuilder()
    metadata_page = builder.add_hbin(0x1000)
    frag0_page = builder.add_hbin(0x4000)
    frag1_page = builder.add_hbin(0x4000)
    tail_page = builder.add_hbin(0x8000)

    frag0 = fragment_payload(1)
    frag1 = fragment_payload(2)
    frag2 = fragment_payload(3)
    decoy = (b"DECOY!!" * ((MAX_DATA_SEGMENT // 7) + 1))[:MAX_DATA_SEGMENT]

    builder.add_cell(frag0_page, "frag0", frag0)
    builder.add_cell(frag1_page, "frag1", frag1)
    builder.add_cell(tail_page, "filler", b"NOT-A-FRAGMENT" + b"\x00" * 4)
    builder.add_cell(tail_page, "frag2", frag2)
    builder.add_cell(tail_page, "decoy", decoy)

    full_length = MAX_DATA_SEGMENT * 3
    partial_length = 100
    broken_length = full_length + partial_length

    builder.add_cell(metadata_page, "small_data", b"A" * 16)
    builder.add_cell(metadata_page, "indirect_intact",
                     make_indirect([builder.cell_rel_offsets["frag0"],
                                    builder.cell_rel_offsets["frag1"],
                                    builder.cell_rel_offsets["frag2"]]))
    builder.add_cell(metadata_page, "db_intact",
                     make_db(builder.cell_rel_offsets["indirect_intact"]))

    bad_reference = 0x77770
    builder.add_cell(metadata_page, "indirect_broken",
                     make_indirect([builder.cell_rel_offsets["frag0"],
                                    builder.cell_rel_offsets["frag1"],
                                    builder.cell_rel_offsets["frag2"],
                                    bad_reference]))
    builder.add_cell(metadata_page, "db_broken",
                     make_db(builder.cell_rel_offsets["indirect_broken"]))

    builder.add_cell(metadata_page, "indirect_dup",
                     make_indirect([builder.cell_rel_offsets["frag0"],
                                    builder.cell_rel_offsets["frag0"]]))
    builder.add_cell(metadata_page, "db_dup",
                     make_db(builder.cell_rel_offsets["indirect_dup"]))

    builder.add_cell(metadata_page, "indirect_trunc",
                     make_indirect([builder.cell_rel_offsets["frag0"]]))
    builder.add_cell(metadata_page, "db_trunc",
                     make_db(builder.cell_rel_offsets["indirect_trunc"]))

    value_specs = [
        ("Small", 16, "small_data"),
        ("IntactBig", full_length, "db_intact"),
        ("BrokenBig", broken_length, "db_broken"),
        ("DupBig", full_length, "db_dup"),
        ("TruncatedBig", full_length, "db_trunc"),
    ]
    vk_rels = []
    for value_name, data_length, data_cell in value_specs:
        rel = builder.add_cell(metadata_page, "vk_" + value_name,
                               make_vk(value_name, data_length,
                                       builder.cell_rel_offsets[data_cell]))
        vk_rels.append(rel)

    builder.add_cell(metadata_page, "value_list",
                     b"".join(struct.pack("<I", rel) for rel in vk_rels))
    builder.add_cell(metadata_page, "nk_root",
                     make_nk("CMI", builder.cell_rel_offsets["value_list"], len(vk_rels)))

    blob = builder.build()
    out_path = os.path.join(os.path.dirname(__file__), "reg_samples",
                            "corrupt_big_data.hive")
    with open(out_path, "wb") as output:
        output.write(blob)
    print("wrote %s (%d bytes)" % (out_path, len(blob)))


if __name__ == "__main__":
    main()
