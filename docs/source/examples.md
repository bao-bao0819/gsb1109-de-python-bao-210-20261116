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

## Match same-label leads across sessions
When two sessions contain several leads with the same label and the device
changed their physical channel order, match them by metadata instead of by
position:

```python
from edfio import match_leads

first = read_edf("session-1.edf")
second = read_edf("session-2.edf")

result = match_leads(first, second)
result.pairs          # only correspondences forced by metadata/anchors
result.interchangeable  # leads that can still be permuted

# Confirmed anchors are hard evidence: they fix the named pairs and leave
# every other label's candidates untouched.
result = match_leads(first, second, anchors=[(("EEG Fpz-Cz", 0), 3)])
```

Matching only compares header metadata; signal data, labels, and channel order
are never modified.
