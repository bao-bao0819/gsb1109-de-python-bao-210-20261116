## [Unreleased]
### Changed
- input validation errors now raise ValueError, instead of AssertionError
- Allow more values in `Stop.location_type` and in `Fare.transfers` (#27)
- `pygtfs.append_feed` now accepts path-like objects (for example, `pathlib.Path`) in addition to string paths (#92)
- A feed missing `calendar.txt` still loads when `calendar_dates.txt` is present, and a trip whose `service_id` is declared in neither calendar file is kept with a synthesized placeholder `Service` (and a warning), instead of retaining a null service relationship
- Missing both `calendar.txt` and `calendar_dates.txt` now correctly raises `PygtfsException` instead of `NameError`

## [0.1.3]
