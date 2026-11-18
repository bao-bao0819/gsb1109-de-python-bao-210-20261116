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

## Convert a BDF recording to EDF
BDF stores 24-bit samples, EDF only 16-bit samples. `convert_bdf_to_edf`
requantizes each channel in the physical domain instead of truncating the high
bits, using the allowed absolute error to choose the physical range written to
the header:

```python
from edfio import convert_bdf_to_edf, read_bdf

bdf = read_bdf("example.bdf")

# one tolerance per channel, specified by label (or by index)
result = convert_bdf_to_edf(
    bdf,
    tolerances={"EEG Fpz-Cz": 0.5, "Body Temp": 0.01},
    default_tolerance=0.1,
)
edf, report = result  # ConversionResult is also an Edf

if not report.all_satisfied:
    print("unsatisfiable:", report.unsatisfiable_channels.labels)
for channel in report.unsatisfiable_channels:
    print(channel.label, channel.reason, channel.error_bound)
    for sample in channel.saturated_samples:
        print(sample.index, sample.value, sample.decoded_value, sample.error)

edf.write("example.edf")
```

The reported error bound (`channel.error_bound`) covers every sample of the
segment after decoding the EDF values back to physical values. When the
amplitude of a channel requires more physical range than the tolerance allows,
the range is centered on the data and the unavoidable clipping is reported as
saturated samples instead of silently truncating bits.
