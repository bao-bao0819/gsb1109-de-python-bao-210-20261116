## [Unreleased]
### Added
- Frequency expansion helpers: `exact_times=1` rows export deterministic
  departure instances, while `exact_times=0` rows export service windows with
  a waiting range instead of invented departures
- `Schedule`, `Trip` and `Frequency` frequency helpers, with half-open
  `[start_time, end_time)` boundaries and overlap conflict reporting

### Changed
- input validation errors now raise ValueError, instead of AssertionError
- Allow more values in `Stop.location_type` and in `Fare.transfers` (#27)
- `pygtfs.append_feed` now accepts path-like objects (for example, `pathlib.Path`) in addition to string paths (#92)

## [0.1.3]
