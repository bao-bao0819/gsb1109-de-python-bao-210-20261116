API Documentation
=================

.. module:: laspy

Re-exported functions
---------------------

laspy.read
___________

.. autofunction:: read

laspy.open
___________

.. autofunction:: open

laspy.create
____________

.. autofunction:: create

laspy.convert
_____________

.. autofunction:: convert


Re-exported classes
-------------------

 - :class:`.LasHeader`
 - :class:`.LasData`
 - :class:`.PointFormat`
 - :class:`.VLR`
 - :class:`.ExtraBytesParams`
 - :class:`.LasReader`
 - :class:`.LasWriter`
 - :class:`.LasAppender`
 - :class:`.PackedPointRecord`
 - :class:`.ScaleAwarePointRecord`
 - :class:`.CopcReader`


Return repair
-------------

 - :func:`.plan_return_repair`
 - :func:`.validate_returns`
 - :class:`.ReturnCandidate`
 - :class:`.ReturnRepairPlan`
 - :class:`.ReturnRepairReport`
 - :class:`.ReturnStatistics`
 - :class:`.ReturnRepairError`


Submodules
----------

.. toctree::

   laspy.lib
   laspy.header
   laspy.lasreader
   laspy.lasdata
   laspy.copc
   laspy.vlrs.vlrlist
   laspy.vlrs.known
   laspy.vlrs.vlr
   laspy.point.record
   laspy.errors
   laspy.compression
   laspy.point.format
   laspy.lasmmap
   laspy.lasappender
   laspy.laswriter
   laspy.repair


