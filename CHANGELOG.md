## [Unreleased]
### Added
- `pygtfs.plan_fares` / `pygtfs.FarePlanner` plan legal ticket combinations
  for a timed journey, honouring fare-rule zone/route restrictions, transfer
  limits and transfer duration; one fare covering several legs is charged
  once, and multi-currency costs are reported as separate totals
### Changed
- input validation errors now raise ValueError, instead of AssertionError
- Allow more values in `Stop.location_type` and in `Fare.transfers` (#27)
- `pygtfs.append_feed` now accepts path-like objects (for example, `pathlib.Path`) in addition to string paths (#92)

## [0.1.3]
