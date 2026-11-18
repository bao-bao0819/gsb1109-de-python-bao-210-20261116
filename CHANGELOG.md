## [Unreleased]
### Added
- `pygtfs.service_dates.service_dates` (also available as `Service.service_dates()`): computes the concrete dates a service runs from its date range, weekday flags and `calendar_dates` exceptions. Each fact keeps its source (the regular calendar pattern or the exception that overrides it), conflicting exceptions for the same service and date raise `PygtfsValidationError` instead of silently keeping the last row, and `Service.exceptions` relates a service to its exceptions.

### Changed
- input validation errors now raise ValueError, instead of AssertionError
- Allow more values in `Stop.location_type` and in `Fare.transfers` (#27)
- `pygtfs.append_feed` now accepts path-like objects (for example, `pathlib.Path`) in addition to string paths (#92)

## [0.1.3]
