## [Unreleased]
### Changed
- input validation errors now raise ValueError, instead of AssertionError
- Allow more values in `Stop.location_type` and in `Fare.transfers` (#27)
- `pygtfs.append_feed` now accepts path-like objects (for example, `pathlib.Path`) in addition to string paths (#92)
### Fixed
- `Trip.stop_times` is now ordered by `stop_sequence`, so loop lines that visit the same `stop_id` more than once keep every stop visit in travel order
- Loading a feed with two rows sharing the same primary key no longer aborts the whole import: the first row is retained, the duplicate is skipped with a warning, leaving the conflict for manual adjudication

## [0.1.3]
