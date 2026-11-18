## [Unreleased]
### Changed
- input validation errors now raise ValueError, instead of AssertionError
- Allow more values in `Stop.location_type` and in `Fare.transfers` (#27)
- `pygtfs.append_feed` now accepts path-like objects (for example, `pathlib.Path`) in addition to string paths (#92)
- Feeds now store their publication version and service validity (`feed_version`, `feed_start_date`, `feed_end_date` from feed_info.txt)
- `pygtfs.append_feed` skips a feed whose revision (name and feed version) was already imported, instead of importing a duplicate
- `pygtfs.overwrite_feed` only replaces the same revision of a feed; other revisions are kept so past publication states remain queryable
- `pygtfs.append_feed` accepts an optional `append_date` to record a publication date other than today

### Added
- `Schedule.feed_as_of` returns the feed revision that was current at a given publication date

## [0.1.3]
