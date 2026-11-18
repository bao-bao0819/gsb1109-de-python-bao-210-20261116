## [Unreleased]
### Added
- Source-scoped entity identity for multi-feed imports. Each imported
  agency/stop/route/trip/service/fare/shape gets an `_entity_identity` row
  keyed by `(feed_id, entity_type, source_id)`, so feeds from different
  agencies no longer mix just because they reuse an id (e.g. two agencies both
  publishing `route_id=1`).
- Explicit cross-feed aliases (`_entity_alias`, plus
  `pygtfs.add_alias`, `pygtfs.add_aliases`,
  `pygtfs.add_aliases_from_file` and the `gtfs2db --aliases` option).
  Identities only share an `identity_key` when an explicit mapping declares
  them equivalent; matching ids, names, or shared stops never cause a merge.
- Feed publication lineage (`_feed_lineage`, plus
  `pygtfs.record_lineage`, `pygtfs.lineage_chain` and the
  `gtfs2db --predecessor-feed-id/--lineage-note` options). A new
  publication is linked to its predecessor without merging entities or
  deleting the older publication, which stays fully reviewable.
### Changed
- input validation errors now raise ValueError, instead of AssertionError
- Allow more values in `Stop.location_type` and in `Fare.transfers` (#27)
- `pygtfs.append_feed` now accepts path-like objects (for example, `pathlib.Path`) in addition to string paths (#92)

## [0.1.3]
