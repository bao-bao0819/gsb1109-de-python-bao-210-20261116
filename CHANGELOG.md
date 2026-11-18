## [Unreleased]
### Changed
- input validation errors now raise ValueError, instead of AssertionError
- Allow more values in `Stop.location_type` and in `Fare.transfers` (#27)
- `pygtfs.append_feed` now accepts path-like objects (for example, `pathlib.Path`) in addition to string paths (#92)
- Translations now resolve per GTFS `translations.txt`: rows with
  `table_name`/`field_name`/`record_id` bind to that exact record, while
  rows without them match on field values across agencies, stops, routes
  and trips. Each binding records its `match_basis` (`record_id` or
  `field_value`), and a record-scoped translation no longer leaks to other
  records that share the same field value. Added `Schedule.translation_view`
  to query resolved translations by language and target scope, returning one
  row per candidate when the target is not unique.

## [0.1.3]
