.. _installation:

============
Installation
============

Short Instructions
==================

Pip
____

.. code-block:: shell

    # Install _without_ LAZ support
    pip install laspy

    # Install with LAZ support via lazrs
    pip install laspy[lazrs]

    # Install with LAZ support via laszip
    pip install laspy[laszip]

    # Install with LAZ support via both lazrs & laszip
    pip install laspy[lazrs,laszip]

Conda
_____

A conda build of laspy is available and maintained (but not by laspy)

.. code-block:: shell

    conda install -c conda-forge laspy

``lazrs`` is also available on conda

.. code-block:: shell

    conda install -c conda-forge lazrs-python


However, ``laszip`` [#f1]_ is not available via conda.


Optional dependencies  / features
=================================


LAZ support
___________

laspy does not support LAZ (.laz) file by itself but can use one of two optional dependencies
to support compressed LAZ files.

The 2 supported options are:

1) `lazrs`_ `[lazrs PyPi]`_

2) `laszip-python`_ (bindings to `laszip`_)

When encountering LAZ data, laspy will try to use one of the backend in the order described above.
(Example: if lazrs is not installed or if it fails during, the process, laspy will try laszip)

`lazrs`_ is a Rust port of the laszip compression and decompression.
Its main advantage is that it is able to compress/decompress using multiple threads which can
greatly speed up things. However it does not supports points with waveforms.

`laszip`_  is the official and original LAZ implementation by Martin Isenburg.
The advantage of the `laszip` backend is that its the official implementation, it supports points
with waveform but does not offer multi-threaded compression/decompression.


Both the laszip bindings and lazrs are available on pip.

To install laspy with one of its supported backend use one of the commands
show in the section above.

.. _lazrs: https://github.com/tmontaigu/laz-rs
.. _laszip-python: https://github.com/tmontaigu/laszip-python
.. _laszip: https://github.com/LASzip/LASzip
.. _[lazrs PyPi]: https://pypi.org/project/lazrs/

CRS / SRS
_________

LAS files allows to define the CRS / SRS in which the points coordinates are.
When `pyproj`_ is installed, you can use the :meth:`.LasHeader.add_crs` to add
CRS information to a file, or you can use :meth:`.LasHeader.parse_crs` to get
`pyproj.CRS`.

A LAS file may contain both GeoTIFF keys and an OGC WKT record, and the two
may disagree.  :meth:`.LasHeader.parse_crs` keeps its historical behavior of
preferring one source.  For ingestion and coordinate operations, use
:meth:`.LasHeader.resolve_crs`, which records the candidates and applies an
explicit rule instead of silently choosing:

.. code-block:: python

    interpretation = las.header.resolve_crs()
    if interpretation.is_conflict:
        print(interpretation.candidates)  # both values are retained
    else:
        print(interpretation.crs, interpretation.rule)

The default ``policy="version"`` follows the LAS specification (GeoTIFF for
LAS <= 1.3 and point formats < 6, WKT for LAS 1.4 point formats >= 6 or when
the WKT global-encoding bit is set).  ``policy="prefer_wkt"``,
``policy="prefer_geotiff"`` and ``policy="require_agreement"`` express caller
conventions.  A conflict is reported as ``status == "conflict"``; no
coordinate is transformed in that state.

When the metadata is usable, :meth:`.LasData.transform_crs` transforms a copy
and records where the source coordinates came from.  After repairing a CRS
VLR, :meth:`.LasData.recompute_crs` replays the transformation from the
retained source coordinates, touching only the affected points:

.. code-block:: python

    projected = las.transform_crs(4326, policy="prefer_wkt")
    projected.crs_transform.to_dict()  # provenance record

    fixed_header = projected.crs_transform.source_header.copy()
    # ... repair the relevant VLR ...
    projected = projected.recompute_crs(source_header=fixed_header,
                                        policy="prefer_wkt")

.. _pyproj: https://pypi.org/project/pyproj/

Cloud Optimized Point Cloud (COPC)
__________________________________

`laspy` supports `COPC`_ files via its :class:`.CopcReader` class.
This features **requires** the optional dependency ``lazrs`` to be installed.

Optionaly, when python package `requests`_ is installed the :class:`.CopcReader`
can handle COPC that are in a HTTP server.


.. _COPC: https://github.com/copcio/copcio.github.io
.. _requests: https://docs.python-requests.org/en/latest


.. rubric:: Footnotes.

.. [#f1] A ``laszip`` package exists on conda-forge, but it only includes the C++ library, not the the Python
         bindings, which means that installing it won't give you a LAZ capable laspy installation.


CLI
---

laspy has an optional command line interface (CLI) with the following commands:

* ``laspy info`` to print informations about a LAS/LAZ file
* ``laspy compress`` / ``laspy decompress`` to compress/decompress one or many file
* ``laspy convert`` to convert file(s) between version and point format
* ``laspy copc query`` to download to a LAS/LAZ file some points from a COPC ressource.

As the cli requires etra dependencies, it has to be installed using
``pip install laspy[cli]``.
