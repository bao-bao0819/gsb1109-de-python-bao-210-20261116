## [Unreleased]
### Added
- `pygtfs.interpolation` derives missing `StopTime` arrivals and departures
  from timepoint anchors and `shape_dist_traveled`, exporting exact times,
  point estimates and inclusive ranges while leaving unsupported stops
  unknown.
### Changed
- input validation errors now raise ValueError, instead of AssertionError
- Allow more values in `Stop.location_type` and in `Fare.transfers` (#27)
- `pygtfs.append_feed` now accepts path-like objects (for example, `pathlib.Path`) in addition to string paths (#92)

## [0.1.3]
