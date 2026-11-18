.. pymzml_search3d:

.. default-domain:: py

Three-dimensional peak search
=============================

.. automodule:: pymzml.search3d


Index and peaks
---------------

.. autoclass:: pymzml.search3d.Peak3D
    :members:

.. autoclass:: pymzml.search3d.PeakIndex3D
    :members:

.. autoclass:: pymzml.search3d.Projection2D
    :members:

.. autofunction:: pymzml.search3d.extract_peaks3d

.. autofunction:: pymzml.search3d.extract_reader_peaks3d


Query and hits
--------------

.. autoclass:: pymzml.search3d.SearchQuery
    :members:

.. autoclass:: pymzml.search3d.Hit3D
    :members:

.. autoclass:: pymzml.search3d.SearchResult
    :members:

.. autoclass:: pymzml.search3d.DimensionEvidence
    :members:

.. autoclass:: pymzml.search3d.CalibrationBasis
    :members:


Piecewise mobility calibration
------------------------------

.. autoclass:: pymzml.search3d.MobilityCalibrationSegment
    :members:

.. autoclass:: pymzml.search3d.MobilityCalibration
    :members:

.. autoclass:: pymzml.search3d.RecalibrationReport
    :members:

.. autoclass:: pymzml.search3d.ProjectionRow
    :members:


Errors
------

.. autoexception:: pymzml.search3d.Search3DError

.. autoexception:: pymzml.search3d.CalibrationError

.. autoexception:: pymzml.search3d.UncalibratedMobilityError

.. autoexception:: pymzml.search3d.PeakValidationError

.. autoexception:: pymzml.search3d.DuplicatePeakError
