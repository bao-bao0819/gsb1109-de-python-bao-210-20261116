## [Unreleased]
### Added
- `pygtfs.block_chains` reconstructs feasible vehicle succession chains from
  `block_id` groups, applying service days, calendar exceptions, stop order
  and timing rules; it keeps every candidate successor, reports conflicts and
  preserves the raw timetable evidence
### Changed
- input validation errors now raise ValueError, instead of AssertionError
- Allow more values in `Stop.location_type` and in `Fare.transfers` (#27)
- `pygtfs.append_feed` now accepts path-like objects (for example, `pathlib.Path`) in addition to string paths (#92)

## [0.1.3]
