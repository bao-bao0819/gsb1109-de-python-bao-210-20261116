=================
Less Basic Things
=================


Extra Dimensions
================

The LAS Specification version 1.4 defines a standard way to add extra dimensions to
a LAS file.

In laspy you can add extra dimensions using the :meth:`.LasData.add_extra_dim` function


The Allowed base types for an extra dimensions are:

+-------------------------+-------------+-------------+
|       laspy name        | size (bits) |     type    |
+=========================+=============+=============+
|     u1 or uint8         |     8       |  unsigned   |
+-------------------------+-------------+-------------+
|     i1 or int8          |     8       |   signed    |
+-------------------------+-------------+-------------+
|     u2 or uint16        |     16      |   unsigned  |
+-------------------------+-------------+-------------+
|     i2 or int16         |     16      |    signed   |
+-------------------------+-------------+-------------+
|     u4 or uint32        |     32      |   unsigned  |
+-------------------------+-------------+-------------+
|     i4 or int32         |     32      |    signed   |
+-------------------------+-------------+-------------+
|     u8 or uint64        |     64      |   unsigned  |
+-------------------------+-------------+-------------+
|     i8 or int64         |     64      |    signed   |
+-------------------------+-------------+-------------+
|     f4 or float32       |     32      |   floating  |
+-------------------------+-------------+-------------+
|     f8 or float64       |     64      |   floating  |
+-------------------------+-------------+-------------+

You can prepend the number '2' or '3' to one of the above base type to define an extra dimension
that is array of 2 or 3 elements per points.
Example: 3u2 -> each points will have an extra dimension that is an array of 3 * 16 bits


Here we are adding a new dimension called "codification" where each value is stored on a 64 bit unsigned integer
and an array field of 3 doubles for each points.


.. code-block:: python

    import laspy
    import numpy as np

    las = laspy.read("somefile.las")

    las.add_extra_dim(laspy.ExtraBytesParams(
        name="codification",
        type=np.uint64,
        description="More classes available"
    ))

    las.add_extra_dim(laspy.ExtraBytesParams(name="mysterious", type="3f8"))



.. note::

    Although the specification of the ExtraBytesVlr appeared in the 1.4 LAS Spec, laspy allows to
    add new dimensions to file with version < 1.4

.. note::

   If you are adding multiple extra dimensions use :meth:`.LasData.add_extra_dims`
   as it is more efficient (it allows to allocate all the dimensions at once instead
   of re-allocating each time a new dimension is added.


Custom VLRs
===========

Provided you have a valid user_id and record_id (meaning that they are not taken by a VLR described in the LAS specification)
You can add you own VLRs to a file

Fast & Easy way
---------------

The fastest and easiest way to add your custom VLR to a file is to create a :class:`.VLR`,
set its record_data (which must be bytes) and add it to the VLR list.


>>> import laspy
>>> vlr = laspy.vlrs.VLR(user_id='A UserId', record_id=1, description='Example VLR')
>>> vlr
<VLR(user_id: 'A UserId', record_id: '1', data len: 0)>
>>> vlr.record_data = b'somebytes'
>>> vlr
<VLR(user_id: 'A UserId', record_id: '1', data len: 9)>
>>> las = laspy.create()
>>> las.vlrs
[]
>>> las.vlrs.append(vlr)
>>> las.vlrs
[<VLR(user_id: 'A UserId', record_id: '1', data len: 9)>]


Complete & Harder way
---------------------

While the way shown above is quick & easy it might not be perfect for complex VLRs.
Also when reading a file that has custom VLR, laspy won't be able to automatically parse its data
into a better structure, so you will have to find the VLR in the vlrs list and parse it yourself
one laspy is done.

One way to automate this task is to create your own Custom VLR Class that extends
:class:`.BaseKnownVLR` by implementing the missing methods laspy
will be able to automatically parse the VLR when reading the file & write it when saving the file.

>>> class CustomVLR(laspy.vlrs.BaseKnownVLR):
...     def __init__(self):
...         super().__init__()
...         self.numbers = []
...
...     @staticmethod
...     def official_user_id():
...         return "CustomId"
...
...     @staticmethod
...     def official_record_ids():
...         return 1,
...
...     def record_data_bytes(self):
...         return bytes(self.numbers)
...
...     def parse_record_data(self, record_data):
...         self.numbers = [b for b in record_data]
...
...     def __repr__(self):
...         return "<MyCustomVLR>"

>>> import numpy as np
>>> cvlr = CustomVLR()
>>> cvlr.numbers
[]
>>> cvlr.numbers = [1,2, 3]
>>> las = laspy.create()
>>> las.vlrs.append(cvlr)
>>> las.vlrs
[<MyCustomVLR>]
>>> las.x = np.array([1.0, 2.0])
>>> las = laspy.lib.write_then_read_again(las)
>>> las.vlrs
[<MyCustomVLR>]
>>> las.vlrs[0].numbers
[1, 2, 3]



Selective Decompression
=======================

When reading a LAZ file with a version >= 1.4 and a point format id
between 6 and 10 (inclusive) it is possible to ask laspy to only
decompress some of the dimensions, using :class:`.DecompressionSelection`.
This can save a lot of time when only a few dimensions are needed:

.. code-block:: python

    import laspy

    selection = laspy.DecompressionSelection.base().decompress_z()
    las = laspy.read("some_file.laz", decompression_selection=selection)

The dimensions which were not decompressed are still present in the
point format (so that writing the file keeps working), however the
values stored in the array are only placeholders and must not be used.

laspy tracks which dimensions were actually loaded:

.. code-block:: python

    >>> import laspy
    >>> las = laspy.read("tests/data/simple.las")
    >>> "intensity" in las.loaded_dimensions
    True
    >>> las.is_dimension_loaded("intensity")
    True
    >>> las.require_dimensions(["X", "Y", "Z"])

- :attr:`LasData.loaded_dimensions` lists the dimensions holding real values.
- :attr:`LasData.not_loaded_dimensions` lists the dimensions holding placeholders.
- :meth:`LasData.is_dimension_loaded` / :meth:`LasData.dimensions_are_available`
  check the availability of one or more dimensions.
- :meth:`LasData.require_dimensions` raises
  :class:`laspy.errors.DimensionNotLoadedError` if any of the given dimensions
  was not loaded. Use it to make sure a query only relies on available data.

Accessing an unloaded dimension returns a view on the placeholder values.
Comparisons against the placeholders keep working, but aggregations such as
``np.mean`` or ``np.sum`` return ``nan`` so that placeholders never silently
take part in a computation. Converting the view to a plain numpy array emits a
:class:`laspy.point.dims.DimensionNotLoadedWarning`.

Writing points that contain unloaded dimensions raises a
:class:`laspy.errors.DimensionNotLoadedError`, so placeholders are never written
as if they were real zero values. Either decompress the dimensions (use
``DecompressionSelection.all()``), fill them with real values, or remove them
before writing.
