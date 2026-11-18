## [Unreleased]
### Changed
- input validation errors now raise ValueError, instead of AssertionError
- Allow more values in `Stop.location_type` and in `Fare.transfers` (#27)
- `pygtfs.append_feed` now accepts path-like objects (for example, `pathlib.Path`) in addition to string paths (#92)

### Added
- `pygtfs.update_feed` and `pygtfs.diff_feed` apply a revised GTFS input
  incrementally: unchanged rows keep their database identity, only removed or
  modified rows are touched, and derived transfers and link tables are rebuilt
  from the revision.  `gtfs2db` gains matching `update` and `diff` commands.

## [0.1.3]
