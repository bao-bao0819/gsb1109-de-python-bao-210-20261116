.. pymzml_spec:

.. default-domain:: py


Spectrum and Chromatogram
==========================

.. automodule:: pymzml.spec


Spectrum
--------

.. autoclass:: pymzml.spec.Spectrum
    :members:
    :exclude-members: __repr__, __str__

    .. automethod:: __getitem__
    .. automethod:: __add__
    .. automethod:: __sub__
    .. automethod:: __mul__
    .. automethod:: __truediv__

Chromatogram
------------

.. autoclass:: pymzml.spec.Chromatogram
    :members:
    :exclude-members: __repr__, __str__


MsData
-----------

.. autoclass:: pymzml.msdata.MsData
    :members:

Collision energies
------------------

.. automodule:: pymzml.energy

.. autoclass:: pymzml.energy.AcquisitionEnergy
    :members:

.. autoclass:: pymzml.energy.EnergyBucket
    :members:

.. autofunction:: pymzml.energy.parse_spectrum_element

.. autofunction:: pymzml.energy.comparable_conditions

.. autofunction:: pymzml.energy.bucket_conditions
