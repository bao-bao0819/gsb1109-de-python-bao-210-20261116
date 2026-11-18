# Examples
## Annotations
EDF+ annotations can be created like this:

```python
import numpy as np

from edfio import Edf, EdfAnnotation, EdfSignal

edf = Edf(
    [
        EdfSignal(np.random.randn(30 * 256), sampling_frequency=256, label="EEG Fpz"),
        EdfSignal(np.random.randn(30), sampling_frequency=1, label="Body Temp"),
    ],
    annotations=[
        EdfAnnotation(1, None, "Trial start"),
        EdfAnnotation(1.5, None, "Stimulus"),
        EdfAnnotation(2.5, 2, "Movement"),
        EdfAnnotation(10, None, "Trial end"),
    ]
)
edf.write("example.edf")

```

## Optional header fields
Most header fields are optional and can be set either at instantiation time or afterwards:

```python
import datetime

import numpy as np

from edfio import Edf, EdfSignal, Patient, Recording

edf = Edf(
    [
        EdfSignal(
            np.random.randn(30 * 256),
            sampling_frequency=256,
            label="EEG Fpz-Cz",
            transducer_type="AgAgCl electrode",
            physical_dimension="uV",
            prefiltering="HP:0.1Hz LP:75Hz",
        ),
        EdfSignal(np.random.randn(30), sampling_frequency=1, label="Body Temp"),
    ],
    patient=Patient(
        code="MCH-0234567",
        sex="F",
        birthdate=datetime.date(1951, 5, 2),
        name="Haagse_Harry",
    ),
    starttime=datetime.time(11, 25),
)
edf.signals[1].transducer_type = "Thermistor"
edf.signals[1].physical_dimension = "degC"
edf.recording = Recording(
    startdate=datetime.date(2002, 2, 2),
    hospital_administration_code="EMG561",
    investigator_technician_code="BK/JOP",
    equipment_code="Sony",
)
edf.write("example.edf")
```

## Change a signal's label
```python
edf.get_signal("Body Temp").label = "Body Temperature"
```

## Drop EDF+ annotations
```python
edf.drop_annotations("Movement")
```

## Drop individual signals
```python
# single signal by label
edf.drop_signals("Body Temp")

# multiple signals by label
edf.drop_signals(["C3", "C4"])

# multiple signals by index
edf.drop_signals([4, 5])
```

## Slice a recording in time
```{note}
The upper limit is always exclusive.
```

```python
# by seconds
edf.slice_between_seconds(5, 15)

# by annotation texts
edf.slice_between_annotations("Trial start", "Trial end")
```

## Anonymize a recording
```python
edf = read_edf("example.edf")
edf.anonymize()
edf.write("example_anonymized.edf")
```

## Concatenate recordings with different units and references
Recordings are matched by channel metadata, not by label alone. Units that can be
converted (for example `uV` and `mV`) are converted, and the conversion basis is
recorded on the result and as EDF+ annotations:

```python
from edfio import UnitConversion, concatenate

result = concatenate(
    ["recording_1.edf", "recording_2.edf"],
    unit_conversions=[UnitConversion("mV", "uV", 1000, description="SI prefix")],
)
result[0].write("merged.edf")
```

If the reference scheme cannot be shown to be equivalent, the recordings are
returned as separate outputs instead of raising an error:

```python
result = concatenate(["mastoid_ref.edf", "linked_ears_ref.edf"])
for index, recording in enumerate(result):
    recording.write(f"part_{index}.edf")
```

Use `plan_concatenation` to inspect the concatenable ranges and conversion basis
before writing anything, and `reference_equivalences` to declare references that
should be treated as equivalent.
