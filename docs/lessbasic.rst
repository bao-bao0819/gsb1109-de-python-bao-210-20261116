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



Auditing return numbers
=======================

``return_number`` and ``number_of_returns`` share a small bit field. A point is
valid when ``1 <= return_number <= number_of_returns`` and both values fit the
point format (3 bits for formats 0-5, 4 bits for formats 6-10). Points that
come from the same emitted pulse must also share one ``number_of_returns`` and
use distinct return numbers.

laspy can report these problems and plan a minimum-cost repair:

.. code-block:: python

    import laspy

    las = laspy.read("some_file.las")
    issues = las.validate_returns(pulse_ids)
    plan = las.plan_return_repair(pulse_ids)

``pulse_ids`` is mandatory for both calls. It is supplied by the caller (for
example from the acquisition system) and is never derived from X/Y/Z, so nearby
points are not silently treated as one pulse.

When several minimum-cost repairs exist, only the field values that all of them
agree on are written by ``plan.apply_in_place()`` (or by
``las.repair_returns(pulse_ids)``). Fields whose optimal value is ambiguous are
left untouched and listed in ``plan.report.review_items`` together with their
possible values. ``plan.report`` also keeps the original codes, the applied
changes, one concrete optimal revision and before/after counts. It can be
serialized with ``plan.to_json()`` or written per point with ``plan.to_csv()``.

``return_number_counts`` and ``number_of_returns_counts`` optionally pin exact
per-flight-line totals. ``source_point_format_ids`` and the ``*_bits`` arguments
narrow the allowed values when a batch mixes old and new point formats. See
:func:`laspy.repair.plan_return_repair` for the full signature.
