## [Unreleased]
### Added
- `pygtfs.direction` groups a route's trips into provable direction groups
  for feeds missing `direction_id`, using stop sequences and shapes rather
  than terminals or route short names; verdict is `same`/`opposite`/
  `unknown`, direction anchors propagate across opposite edges, unreachable
  groups stay unassigned, and anchor-driven assignment changes can be
  explained via `DirectionResult.explain_changes`
### Changed
- input validation errors now raise ValueError, instead of AssertionError
- Allow more values in `Stop.location_type` and in `Fare.transfers` (#27)
- `pygtfs.append_feed` now accepts path-like objects (for example, `pathlib.Path`) in addition to string paths (#92)

## [0.1.3]
