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

Sample tables
-------------

``points["waveform"]`` is the raw amplitude array, with one row per point.
For auditing and analysis you usually want one row per *sample*, with the
spatial position and calibrated amplitude of each sample:

.. code-block:: python

   import laspy

   las = laspy.read("example.laz")
   table = las.waveform_samples()

   table.point_index          # owning point for each sample
   table.descriptor_index     # wavepacket_index of that point
   table.sample_index         # sample position inside the packet
   table.raw_amplitude        # integer value as stored
   table.amplitude            # raw * digitizer_gain + digitizer_offset
   table.time                # offset from the return, in ``table.time_unit``
   table.xyz                 # (N, 3) sample coordinates

   table["x"], table["y"], table["z"]
   table["sample_offset_bytes"]         # byte offset relative to its packet
   table["source_sample_byte_offset"]   # byte offset in the source file

Each point is interpreted through the descriptor it references, so packets in
the same file may have different sample counts, spacings, and gain/offset
values.  For sample ``i`` of the packet attached to point ``p`` the LAS
definition is:

.. code-block:: text

   offset_ps = i * temporal_sample_spacing - return_point_wave_location
   sample_xyz = point_xyz + offset_ps * (x_t, y_t, z_t)

The direction vector ``(x_t, y_t, z_t)`` is expressed in metres per
picosecond and normally has a norm close to ``c / 2``.  ``time_unit`` selects
the unit of ``table.time`` (``"ps"`` by default; ``"ns"``, ``"us"``, ``"ms"``,
and ``"s"`` are also accepted).

Points that cannot be converted are not silently dropped and do not abort the
table.  They are reported in ``table.issues`` with a
``WaveformSampleIssueKind``:

- ``missing_packet``: the point has no packet (``wavepacket_index == 0``) or no
  usable byte offset.
- ``unknown_compression``: the descriptor requests a compression laspy cannot
  decode.
- ``incomplete_samples``: the packet or its declared size holds only part of
  the descriptor's sample count.  Complete samples are still retained.
- ``missing_descriptor``: no VLR matches the point's ``wavepacket_index``.
- ``missing_source`` / ``missing_geometry``: the packet bytes or the point's
  coordinates/direction vector are unavailable.

``table.issues_by_kind()`` groups them.  The table is fully materialised, so
it outlives the reader; ``laspy.read_waveform_sample_table(path)`` is the
convenience entry point.

Writing
-------

When waveform data is present, writing to a filesystem path will emit a sibling
``.wdp`` file next to the ``.las`` or ``.laz`` file.

Writing waveform-enabled data to a file-like object is not supported.

Current Scope
-------------

The current implementation supports:

- External waveform storage in a sibling ``.wdp`` file.
- Uncompressed waveform packets only.
- Byte-aligned sample widths only.
- Waveform packets addressed as fixed-size records.
- Mixed waveform descriptors when reading sample tables; the legacy
  ``points["waveform"]`` array still requires one layout per file.

Unsupported LAS 1.4 Features
----------------------------

The LAS 1.4 specification allows more than the current implementation supports.
laspy currently raises ``NotImplementedError`` or ``ValueError`` for these
cases.

- Internal waveform storage is supported for sample tables read from a file
  path, but not by the legacy ``fullwave`` array loader.
- Non-byte-aligned waveform sample widths are not supported.
- Multiple waveform descriptor layouts in one file are not supported.
- General byte-offset-based packet layouts are not supported when they do not
  behave like fixed-size records.
