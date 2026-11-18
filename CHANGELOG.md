## [Unreleased]
### Added
- `pygtfs.transfer` builds sourced `TransferCandidate` objects from
  `transfers.txt`, combining parent-station, route-specific, trip-specific and
  generic stop-to-stop rules. `preferred_transfer` applies precedence (a
  `transfer_type` 3 prohibition wins; trip over route over stop; a rule on the
  queried platform beats its parent station), while `walking_time_seconds`
  returns `None` for unknown or forbidden transfers instead of assuming zero
  minutes. Stops sharing a name but lacking a transfer row produce no
  candidate.
### Changed
- input validation errors now raise ValueError, instead of AssertionError
- Allow more values in `Stop.location_type` and in `Fare.transfers` (#27)
- `pygtfs.append_feed` now accepts path-like objects (for example, `pathlib.Path`) in addition to string paths (#92)

## [0.1.3]
