from __future__ import annotations

import io
import json
import re
from typing import TYPE_CHECKING

import pytest

from dissect.cstruct.references import dumpreferences
from tests.util import NonSeekableStream, verify_compiled

if TYPE_CHECKING:
    from dissect.cstruct.cstruct import cstruct


def test_references_self_reference(cs: cstruct, compiled: bool) -> None:
    """A structure that points to itself is a cycle, not an endless expansion."""
    cs.pointer = cs.uint16
    cs.load(
        """
        struct node {
            uint16  value;
            node    *next;
        };
        """,
        compiled=compiled,
    )

    assert verify_compiled(cs.node, compiled)

    # The node starts at offset 4 and points back to itself
    stream = io.BytesIO(b"\x00" * 4 + b"\x2a\x00\x04\x00")
    stream.seek(4)
    obj = cs.node(stream)
    result = dumpreferences(obj, offset=4)

    assert result["root"] == 0
    assert result["address"] == 4
    assert result["bounds"] == [4, 8]
    assert result["truncated"] is False

    assert [(o["id"], o["address"], o["type"], o["bounds"]) for o in result["objects"]] == [
        (0, 4, "node", [4, 8]),
    ]
    assert result["references"] == [
        {
            "source": 0,
            "path": "next",
            "address": 4,
            "type": "node",
            "status": "resolved",
            "target": 0,
            "relation": "cycle",
        },
    ]


def test_references_cycle_and_shared(cs: cstruct, compiled: bool) -> None:
    """Distinguish a reference to an object on the current path from one to an already exported object."""
    cs.pointer = cs.uint16
    cs.load(
        """
        struct node {
            uint16  value;
            node    *next;
        };

        struct root {
            node    *first;
            node    *second;
            node    *again;
        };
        """,
        compiled=compiled,
    )

    assert verify_compiled(cs.node, compiled)
    assert verify_compiled(cs.root, compiled)

    buf = (
        b"\x08\x00\x0c\x00\x08\x00\x00\x00"  # root: first=8, second=12, again=8
        b"\x01\x00\x0c\x00"  # node @ 8: value=1, next=12
        b"\x02\x00\x08\x00"  # node @ 12: value=2, next=8
    )
    result = dumpreferences(cs.root(buf))

    assert [(o["id"], o["address"], o["type"]) for o in result["objects"]] == [
        (0, 0, "root"),
        (1, 8, "node"),
        (2, 12, "node"),
    ]

    references = {(r["source"], r["path"]): r for r in result["references"]}
    assert references[(0, "first")]["relation"] == "reference"
    assert references[(1, "next")]["relation"] == "reference"
    assert references[(2, "next")]["relation"] == "cycle"
    assert references[(2, "next")]["target"] == 1
    assert references[(0, "second")]["relation"] == "shared"
    assert references[(0, "second")]["target"] == 2
    assert references[(0, "again")]["relation"] == "shared"
    assert references[(0, "again")]["target"] == 1


def test_references_same_address_different_types(cs: cstruct, compiled: bool) -> None:
    """Two pointer types to the same address are separate objects with their own type and bounds."""
    cs.pointer = cs.uint16
    cs.load(
        """
        struct pair {
            uint16  a;
            uint16  b;
        };

        struct quad {
            uint8   a;
            uint8   b;
            uint8   c;
            uint8   d;
        };

        struct root {
            pair    *p;
            quad    *q;
        };
        """,
        compiled=compiled,
    )

    assert verify_compiled(cs.root, compiled)

    buf = b"\x04\x00\x04\x00" + b"\x01\x02\x03\x04"
    result = dumpreferences(cs.root(buf))

    assert [(o["id"], o["address"], o["type"], o["size"], o["bounds"]) for o in result["objects"]] == [
        (0, 0, "root", 4, [0, 4]),
        (1, 4, "pair", 4, [4, 8]),
        (2, 4, "quad", 4, [4, 8]),
    ]

    # The same bytes are two facts: the pair has two fields, the quad has four
    assert [field["name"] for field in result["objects"][1]["fields"]] == ["a", "b"]
    assert [field["name"] for field in result["objects"][2]["fields"]] == ["a", "b", "c", "d"]

    references = {(r["source"], r["path"]): r for r in result["references"]}
    assert references[(0, "p")]["target"] == 1
    assert references[(0, "q")]["target"] == 2


def test_references_null_out_of_bounds_and_unresolved(cs: cstruct, compiled: bool) -> None:
    """Null pointers, out-of-bounds targets and targets that can't be obtained are separate statuses."""
    cs.pointer = cs.uint16
    cs.load(
        """
        struct root {
            uint16  *null;
            uint16  *oob;
            uint16  *truncated;
            void    *void_ptr;
        };
        """,
        compiled=compiled,
    )

    assert verify_compiled(cs.root, compiled)

    buf = b"\x00\x00" + b"\x09\x00" + b"\x08\x00" + b"\x01\x00" + b"\xaa"
    result = dumpreferences(cs.root(buf))

    references = {r["path"]: r for r in result["references"]}

    assert references["null"] == {
        "source": 0,
        "path": "null",
        "address": 0,
        "type": "uint16",
        "status": "null",
    }
    assert references["oob"]["status"] == "out_of_bounds"
    assert references["oob"]["available"] == 0
    assert references["oob"]["required"] == 2
    assert references["truncated"]["status"] == "out_of_bounds"
    assert references["truncated"]["available"] == 1
    assert references["truncated"]["required"] == 2
    assert references["void_ptr"]["status"] == "unresolved"
    assert references["void_ptr"]["reason"] == "void"

    # Only the root is an object, none of these pointers resolved to one
    assert [o["type"] for o in result["objects"]] == ["root"]


def test_references_no_stream(cs: cstruct) -> None:
    """A pointer without a stream can't be followed."""
    cs.pointer = cs.uint16
    cs.load("struct test { uint16 value; uint16 *ptr; };")

    obj = cs.test(value=1, ptr=cs._make_pointer(cs.uint16)(8, None, endian=cs.endian))
    result = dumpreferences(obj)

    assert result["references"] == [
        {
            "source": 0,
            "path": "ptr",
            "address": 8,
            "type": "uint16",
            "status": "unresolved",
            "reason": "no_stream",
        },
    ]


def test_references_unreadable_stream(cs: cstruct) -> None:
    """A stream error other than reaching the end is unresolved, not out of bounds."""
    cs.pointer = cs.uint16
    cs.load("struct node { uint16 value; node *next; };")

    obj = cs.node._read(NonSeekableStream(b"\x01\x00\x08\x00"), endian="<")
    result = dumpreferences(obj)

    assert result["references"] == [
        {
            "source": 0,
            "path": "next",
            "address": 8,
            "type": "node",
            "status": "unresolved",
            "reason": "not_seekable",
        },
    ]


def test_references_max_depth(cs: cstruct, compiled: bool) -> None:
    """Recursive structures stop at max_depth instead of expanding until exhaustion."""
    cs.pointer = cs.uint16
    cs.load(
        """
        struct node {
            uint16  value;
            node    *next;
        };
        """,
        compiled=compiled,
    )

    assert verify_compiled(cs.node, compiled)

    buf = (
        b"\x01\x00\x04\x00"  # node @ 0: value=1, next=4
        b"\x02\x00\x08\x00"  # node @ 4: value=2, next=8
        b"\x03\x00\x04\x00"  # node @ 8: value=3, next=4
    )
    obj = cs.node(buf)

    result = dumpreferences(obj, max_depth=0)
    assert result["truncated"] is True
    assert len(result["objects"]) == 1
    assert result["references"] == [
        {
            "source": 0,
            "path": "next",
            "address": 4,
            "type": "node",
            "status": "unresolved",
            "reason": "max_depth",
        },
    ]

    result = dumpreferences(obj, max_depth=1)
    assert result["truncated"] is True
    assert len(result["objects"]) == 2
    assert result["references"][-1] == {
        "source": 1,
        "path": "next",
        "address": 8,
        "type": "node",
        "status": "unresolved",
        "reason": "max_depth",
    }

    result = dumpreferences(obj, max_depth=2)
    assert result["truncated"] is False
    assert [(o["address"], o["type"]) for o in result["objects"]] == [
        (0, "node"),
        (4, "node"),
        (8, "node"),
    ]
    assert result["references"][-1]["relation"] == "cycle"


def test_references_pointer_array(cs: cstruct, compiled: bool) -> None:
    """Pointers in arrays and nested structures get a path relative to their source object."""
    cs.pointer = cs.uint16
    cs.load(
        """
        struct item {
            uint16  value;
        };

        struct container {
            item    *items[2];
        };

        struct root {
            container   nested;
            item        *single;
        };
        """,
        compiled=compiled,
    )

    assert verify_compiled(cs.root, compiled)

    buf = (
        b"\x08\x00\x0a\x00\x0c\x00\x00\x00"  # nested.items[0]=8, nested.items[1]=10, single=12, padding
        b"\x01\x00"  # item @ 8
        b"\x02\x00"  # item @ 10
        b"\x03\x00"  # item @ 12
    )
    result = dumpreferences(cs.root(buf))

    assert [(r["source"], r["path"], r["target"]) for r in result["references"]] == [
        (0, "nested.items[0]", 1),
        (0, "nested.items[1]", 2),
        (0, "single", 3),
    ]
    assert [(o["address"], o["type"]) for o in result["objects"][1:]] == [
        (8, "item"),
        (10, "item"),
        (12, "item"),
    ]


def test_references_union(cs: cstruct, compiled: bool) -> None:
    """Pointers that overlap in a union each produce their own reference."""
    cs.pointer = cs.uint32
    cs.load(
        """
        struct target {
            uint32  value;
        };

        union choice {
            target  *ref;
            uint32  raw;
        };

        struct root {
            choice  value;
        };
        """,
        compiled=compiled,
    )

    assert verify_compiled(cs.root, compiled)

    buf = b"\x04\x00\x00\x00" + b"\x2a\x00\x00\x00"

    # A union parses its members from a copy of its own bytes, so its pointers only observe the union's boundary
    result = dumpreferences(cs.root(buf))
    assert result["references"] == [
        {
            "source": 0,
            "path": "value.ref",
            "address": 4,
            "type": "target",
            "status": "out_of_bounds",
            "available": 0,
            "required": 4,
        },
    ]

    # With the full root data, the same pointer addresses the target in the root stream
    result = dumpreferences(cs.root(buf), buf)
    assert result["references"] == [
        {
            "source": 0,
            "path": "value.ref",
            "address": 4,
            "type": "target",
            "status": "resolved",
            "target": 1,
            "relation": "reference",
        },
    ]
    assert result["objects"][1]["fields"] == [
        {"name": "value", "type": "uint32", "offset": 4, "size": 4, "display": "0x2a"},
    ]


def test_references_dumpstruct_graph(cs: cstruct, compiled: bool) -> None:
    """The graph output of dumpstruct is the same fact set as dumpreferences."""
    from dissect.cstruct.dump import dumpstruct

    cs.pointer = cs.uint16
    cs.load(
        """
        struct node {
            uint16  value;
            node    *next;
        };
        """,
        compiled=compiled,
    )

    buf = b"\x2a\x00\x04\x00"
    obj = cs.node(buf)

    assert dumpstruct(obj, output="graph") == dumpreferences(obj)
    assert json.loads(dumpstruct(obj, output="json"))["type"] == "node"


def test_references_invalid_args(cs: cstruct) -> None:
    message = "Invalid arguments: expected a Structure instance, or a Structure class with data."

    with pytest.raises(ValueError, match=re.escape(message)):
        dumpreferences(42)  # type: ignore[arg-type]

    cs.load("struct test { uint16 value; };")

    with pytest.raises(ValueError, match=re.escape("max_depth must be zero or greater")):
        dumpreferences(cs.test(b"\x01\x00"), max_depth=-1)

    with pytest.raises(ValueError, match=re.escape(message)):
        dumpreferences(cs.test)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=re.escape("Invalid output argument: 'xml'")):
        dumpreferences(cs.test(b"\x01\x00"), output="xml")  # type: ignore[arg-type]
