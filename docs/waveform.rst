Waveform Support
================

laspy can read and write LAS waveform point formats, but the current
implementation intentionally supports only a subset of the LAS 1.4 waveform
specification.

API
---

Use ``fullwave`` when opening a file for reading:

The default is ``fullwave="never"`` (no waveform samples are loaded unless you
opt in).

.. code-block:: python

   import laspy

   with laspy.open("example.laz", fullwave="never") as reader:
       las = reader.read()

   with laspy.open("example.laz", fullwave="lazy") as reader:
       points = reader.read_points(1024)
       waveforms = points["waveform"]

   with laspy.open("example.laz", fullwave="eager") as reader:
       las = reader.read()
       waveforms = las.points["waveform"]

``fullwave="never"``
   Do not load waveform samples.

``fullwave="lazy"``
   Keep waveform samples in the external ``.wdp`` file until
   ``points["waveform"]`` is accessed.

``fullwave="eager"``
   Load waveform samples while reading points.

Writing
-------

When waveform data is present, writing to a filesystem path will emit a sibling
``.wdp`` file next to the ``.las`` or ``.laz`` file.

Writing waveform-enabled data to a file-like object is not supported.

Waveform Objects and Point References
-------------------------------------

Waveform samples are kept separately from the point records. Each point
references a waveform packet through its ``wavepacket_offset``; points that
share the same offset share the same waveform object. Packets are never merged
because their sample bytes happen to be equal: only the packet reference
(offset) determines sharing.

The ``wavepacket_index`` field selects the waveform packet descriptor (VLR
record ID ``100 + index``), and ``wavepacket_size`` must match the packet size
declared by that descriptor. Packet offsets must form fixed-size records for
the descriptor.

Missing Packets
---------------

If the external ``.wdp`` file is truncated or a packet cannot be read, the
reader keeps the point geometry and only marks the points that reference the
missing packet. Reading ``points["waveform"]`` for those points returns an
all-zero sample array, matching the way points without waveform packets are
represented. Writing the data back clears ``wavepacket_index``,
``wavepacket_offset`` and ``wavepacket_size`` for those points, so a new
``.wdp`` file contains only the packets that are actually available.

Current Scope
-------------

The current implementation supports:

- External waveform storage in a sibling ``.wdp`` file.
- Uncompressed waveform packets only.
- Byte-aligned sample widths only.
- A single waveform layout across all waveform descriptors in the file.
- Waveform packets addressed as fixed-size records.

Unsupported LAS 1.4 Features
----------------------------

The LAS 1.4 specification allows more than the current implementation supports.
laspy currently raises ``NotImplementedError`` or ``ValueError`` for these
cases.

- Internal waveform storage in the LAS EVLR area is not supported.
- Non-byte-aligned waveform sample widths are not supported.
- Multiple waveform descriptor layouts in one file are not supported.
- General byte-offset-based packet layouts are not supported when they do not
  behave like fixed-size records.
