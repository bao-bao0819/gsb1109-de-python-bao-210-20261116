"""
Helpers for extracting regions that are not covered by any feature.

The routines here operate on the union of inclusive intervals.  This is
important for annotations with nested or overlapping features: subtracting the
end of one row from the start of the next row is not enough, and can even
produce a negative length.

All intervals are 1-based and inclusive, following GFF/GTF conventions.
"""

import os

from gffutils import bins
from gffutils import helpers
from gffutils.feature import Feature


def _feature_id(feature):
    feature_id = getattr(feature, "id", None)
    if feature_id is None:
        try:
            feature_id = feature.attributes["ID"][0]
        except (KeyError, IndexError, TypeError):
            feature_id = str(feature)
    return feature_id


def _feature_sort_key(feature):
    """
    A deterministic ordering for features.

    The key deliberately includes enough information to distinguish features
    which share coordinates.  This keeps both the reported intervals and the
    reported boundary support independent of the order in which features were
    supplied.
    """
    return (
        "" if feature.seqid is None else feature.seqid,
        feature.start if feature.start is not None else 0,
        feature.end if feature.end is not None else 0,
        "" if feature.strand is None else feature.strand,
        feature.featuretype or "",
        str(_feature_id(feature)),
        str(feature),
    )


class SupportIds(list):
    """
    A list of supporting feature IDs.

    Comparisons treat the collection as a set, so callers can compare against
    a tuple, set, or list without worrying about container type or ordering.
    """

    def __eq__(self, other):
        if isinstance(other, (list, tuple, set, frozenset)):
            return set(self) == set(other)
        return list.__eq__(self, other)

    def __ne__(self, other):
        return not self == other

    __hash__ = None

    def __repr__(self):
        return "SupportIds(%r)" % (list(self),)


class SupportList(tuple):
    """
    An immutable, deterministically ordered collection of supporting features.

    The object behaves like a tuple of :class:`Feature` instances.  For
    convenience, comparison against feature IDs or feature strings is also
    supported, so callers can ask questions such as ``gap.left_support ==
    ["gene1"]`` without having to extract IDs first.
    """

    @property
    def ids(self):
        return SupportIds(_feature_id(feature) for feature in self)

    def __eq__(self, other):
        if isinstance(other, (list, tuple, set, frozenset)):
            other = list(other)
            as_features = tuple(self)
            if all(isinstance(item, Feature) for item in other):
                return set(as_features) == set(other)
            ids = set(self.ids)
            if all(isinstance(item, str) and item in ids for item in other):
                return ids == set(other)
            other_strings = set(str(item) for item in other)
            return set(str(feature) for feature in as_features) == other_strings
        return tuple.__eq__(self, other)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return tuple.__hash__(self)

    def __repr__(self):
        return "SupportList(%r)" % (tuple(self),)


def _support(features):
    # Indistinguishable duplicates (same coordinates, type, ID and rendered
    # line) collapse to one entry; this keeps provenance deterministic even
    # when the same feature is supplied twice.
    unique = {}
    for feature in features:
        unique.setdefault(_feature_sort_key(feature), feature)
    return SupportList(tuple(sorted(unique.values(), key=_feature_sort_key)))


def _featuretype_for(left, right):
    left_type = _unique_featuretype(left, default=None)
    right_type = _unique_featuretype(right, default=None)
    if left_type is None and right_type is None:
        return "inter_start_end"
    if left_type is None:
        return "inter_start_%s" % right_type
    if right_type is None:
        return "inter_%s_end" % left_type
    return "inter_%s_%s" % (left_type, right_type)


def _merge_support_attributes(
    left, right, numeric_sort=False, attribute_func=None, attribute_support="all"
):
    """
    Merge the attributes of the features supporting both sides of a gap.

    Reduction is explicit (rather than relying on the neighboring rows) so
    that nested features do not leak attributes into unrelated gaps.

    If `attribute_support` is ``"representative"``, only the closest feature
    on each side contributes attributes.  This preserves the historical
    ``interfeatures`` output (one feature per boundary) while the full set of
    supporters remains available on the returned feature.
    """
    if attribute_support == "representative":
        if left:
            left = (max(left, key=lambda f: (f.start, -f.end, _feature_sort_key(f))),)
        if right:
            right = (min(right, key=lambda f: (f.end, f.start, _feature_sort_key(f))),)

    merged = {}
    for feature in tuple(left) + tuple(right):
        attributes = feature.attributes
        if attribute_func is not None:
            attributes = attribute_func(attributes)
        merged = helpers.merge_attributes(merged, attributes, numeric_sort=numeric_sort)
    return merged


def _unique_featuretype(features, default="."):
    types = []
    for feature in features:
        featuretype = getattr(feature, "featuretype", None)
        if featuretype and featuretype != "." and featuretype not in types:
            types.append(featuretype)
    if not types:
        return default
    if len(types) == 1:
        return types[0]
    return "_".join(sorted(types))


def _common_strand(features):
    strands = set(
        feature.strand
        for feature in features
        if getattr(feature, "strand", None) not in (None, ".")
    )
    if len(strands) == 1:
        return strands.pop()
    return "."


def _parse_region(region):
    if isinstance(region, str):
        seqid, startstop = region.split(":")
        start, end = startstop.split("-")
        return seqid, int(start), int(end)
    return region[0], int(region[1]), int(region[2])


def _normalize_bounds(bounds):
    """
    Normalize caller-provided sequence endpoints.

    Accepted forms are a mapping of ``seqid -> length``, a mapping of ``seqid
    -> (start, end)``, a single ``(seqid, start, end)`` tuple, a
    ``"seqid:start-end"`` string, or a chromsizes-like filename/iterable.
    """
    if bounds is None:
        return {}

    if isinstance(bounds, (str, bytes, os.PathLike)):
        bounds = os.fspath(bounds)
        if isinstance(bounds, str) and ":" in bounds and not os.path.exists(bounds):
            seqid, start, end = _parse_region(bounds)
            return {seqid: (start, end)}
        result = {}
        with open(bounds) as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                fields = line.split()
                if len(fields) < 2:
                    continue
                if len(fields) >= 3:
                    result[fields[0]] = (int(fields[1]), int(fields[2]))
                else:
                    result[fields[0]] = (1, int(fields[1]))
        return result

    if isinstance(bounds, dict):
        result = {}
        for seqid, value in bounds.items():
            if isinstance(value, (tuple, list)):
                result[seqid] = (int(value[0]), int(value[1]))
            else:
                result[seqid] = (1, int(value))
        return result

    if (
        isinstance(bounds, (tuple, list))
        and len(bounds) == 3
        and not isinstance(bounds[0], (tuple, list, dict))
    ):
        seqid, start, end = _parse_region(bounds)
        return {seqid: (int(start), int(end))}

    result = {}
    for item in bounds:
        if len(item) == 2 and isinstance(item[1], (tuple, list)):
            seqid, value = item
            result[seqid] = (int(value[0]), int(value[1]))
        elif len(item) == 2:
            result[item[0]] = (1, int(item[1]))
        else:
            result[item[0]] = (int(item[1]), int(item[2]))
    return result


def _make_uncovered_feature(
    feature_returner,
    seqid,
    start,
    end,
    strand,
    left,
    right,
    featuretype,
    include_support_attributes=False,
    merge_attributes=True,
    numeric_sort=False,
    attribute_func=None,
    update_attributes=None,
    attribute_support="all",
):
    left = _support(left)
    right = _support(right)

    if merge_attributes:
        attributes = _merge_support_attributes(
            left,
            right,
            numeric_sort=numeric_sort,
            attribute_func=attribute_func,
            attribute_support=attribute_support,
        )
    else:
        attributes = {}

    if update_attributes:
        attributes.update(update_attributes)

    if include_support_attributes:
        attributes["left_support"] = list(left.ids) if left else ["."]
        attributes["right_support"] = list(right.ids) if right else ["."]

    feature = feature_returner(
        seqid=seqid,
        source="gffutils_derived",
        featuretype=featuretype,
        start=start,
        end=end,
        score=".",
        strand=strand,
        frame=".",
        attributes=attributes,
        bin=bins.bins(start, end, one=True),
    )

    # Provenance is kept off the attribute string by default so that the
    # textual representation of derived features does not change unless the
    # caller asks for it.  The properties are always available.
    feature.left_support = left
    feature.right_support = right
    feature.left_features = left
    feature.right_features = right
    feature.left = left
    feature.right = right
    feature.left_supporting_features = left
    feature.right_supporting_features = right
    feature.left_support_ids = left.ids
    feature.right_support_ids = right.ids
    feature.left_ids = left.ids
    feature.right_ids = right.ids
    feature.support = _support(tuple(left) + tuple(right))
    feature.supporting_features = feature.support
    feature.support_ids = feature.support.ids
    feature.children = list(feature.support)
    feature.left_children = list(left)
    feature.right_children = list(right)
    feature.left_neighbor = left[-1] if left else None
    feature.right_neighbor = right[0] if right else None
    feature.left_boundary = start - 1
    feature.right_boundary = end + 1
    feature.left_boundary_support = left
    feature.right_boundary_support = right
    feature.left_boundary_features = left
    feature.right_boundary_features = right
    feature.left_boundary_ids = left.ids
    feature.right_boundary_ids = right.ids
    feature.boundary_support = {"left": left, "right": right}

    if len(feature.attributes.get("ID", [])) > 1:
        feature.attributes["ID"] = ["-".join(feature.attributes["ID"])]
    return feature


def uncovered_intervals(
    features,
    new_featuretype=None,
    chromsizes=None,
    bounds=None,
    sequence_bounds=None,
    include_ends=None,
    ignore_strand=True,
    strand=None,
    merge_attributes=True,
    numeric_sort=False,
    attribute_func=None,
    update_attributes=None,
    include_support_attributes=False,
    feature_returner=None,
    merge_criteria=None,
    attribute_support="all",
):
    """
    Yield maximal inclusive intervals not covered by any supplied feature.

    The input may be unsorted, nested, overlapping, or spread across multiple
    sequence IDs.  Coverage is computed from the union of all features rather
    than from adjacent rows, so a negative ``next.start - previous.end`` value
    can never be interpreted as an interval.

    Parameters
    ----------
    features : iterable of :class:`gffutils.Feature`
        Features to consider.  Features with missing or invalid coordinates
        (``start > end``) do not cover anything.

    new_featuretype : str or None
        Feature type for yielded features.  If None, a type of the form
        ``inter_<left>_<right>`` is constructed from the features supporting
        each boundary.

    chromsizes, bounds, sequence_bounds : mapping, iterable, filename, or None
        Caller-supplied sequence endpoints.  If provided and `include_ends` is
        true, leading and trailing uncovered intervals are emitted in addition
        to intervals between covered blocks.  Values may be lengths or
        ``(start, end)`` tuples.

    include_ends : bool
        Include the region before the first covered feature and after the
        last covered feature.  This requires sequence bounds to be known.

    ignore_strand : bool
        If True (default), coverage is the union across strands and yielded
        features have the common strand of their supporting features (or "."
        when the supports disagree).  If False, coverage and gaps are computed
        separately for each strand.

    strand : None, "+", "-", or "."
        If not None, only features on this strand are considered.

    merge_criteria : iterable or None
        Accepted for consistency with :meth:`FeatureDB.merge`.  If it includes
        a callback named ``strand``, per-strand grouping is used.

    include_support_attributes : bool
        If True, add ``left_support`` and ``right_support`` attributes to the
        yielded features.  Provenance is always available through the
        ``left_support``, ``right_support``, ``left_features`` and
        ``right_features`` properties.

    Returns
    -------
    Generator of :class:`gffutils.Feature` objects.
    """
    if feature_returner is None:

        def feature_returner(**kwargs):
            return Feature(**kwargs)

    all_bounds = {}
    for value in (chromsizes, bounds, sequence_bounds):
        all_bounds.update(_normalize_bounds(value))
    sequence_bounds = all_bounds

    if include_ends is None:
        include_ends = bool(sequence_bounds)

    if merge_criteria is not None:
        names = set(getattr(criteria, "__name__", "") for criteria in merge_criteria)
        if "strand" in names:
            ignore_strand = False

    records = []
    for feature in features:
        if isinstance(feature, str):
            raise TypeError("uncovered_intervals() requires Feature instances, not IDs")
        if feature.start is None or feature.end is None:
            continue
        try:
            start = int(feature.start)
            end = int(feature.end)
        except (TypeError, ValueError):
            continue
        if start > end:
            continue
        if strand is not None and feature.strand != strand:
            continue
        records.append((start, end, feature))

    groups = {}
    for start, end, feature in records:
        if ignore_strand:
            key = (feature.seqid, None)
        else:
            key = (feature.seqid, feature.strand)
        groups.setdefault(key, []).append((start, end, feature))

    group_keys = set(groups)
    if include_ends and sequence_bounds:
        for seqid in sequence_bounds:
            if ignore_strand:
                group_keys.add((seqid, None))
            elif strand is not None:
                group_keys.add((seqid, strand))
            else:
                # Without a strand filter, both strands are meaningful and a
                # sequence with features on only one strand is still entirely
                # uncovered on the other.
                group_keys.add((seqid, "+"))
                group_keys.add((seqid, "-"))

    group_keys = sorted(
        group_keys,
        key=lambda item: (
            "" if item[0] is None else item[0],
            "" if item[1] is None else item[1],
        ),
    )

    for seqid, group_strand in group_keys:
        key = (seqid, group_strand)
        group_records = groups.get(key, [])
        group_records.sort(
            key=lambda item: (item[0], item[1], _feature_sort_key(item[2]))
        )

        region = sequence_bounds.get(seqid)
        if region is None:
            region_start, region_end = 1, None
        else:
            region_start, region_end = region

        def emit(start, end, left, right):
            # Respect caller-provided sequence endpoints: never report bases
            # outside the declared sequence.
            if region is not None:
                start = max(start, region_start)
                if region_end is not None:
                    end = min(end, region_end)
            if start > end:
                return None
            if new_featuretype is not None:
                featuretype = new_featuretype
            else:
                featuretype = _featuretype_for(left, right)

            if ignore_strand:
                out_strand = _common_strand(tuple(left) + tuple(right))
            else:
                out_strand = group_strand

            return _make_uncovered_feature(
                feature_returner=feature_returner,
                seqid=seqid,
                start=start,
                end=end,
                strand=out_strand,
                left=left,
                right=right,
                featuretype=featuretype,
                include_support_attributes=include_support_attributes,
                merge_attributes=merge_attributes,
                numeric_sort=numeric_sort,
                attribute_func=attribute_func,
                update_attributes=update_attributes,
                attribute_support=attribute_support,
            )

        if not group_records:
            if include_ends and region is not None:
                feature = emit(region_start, region_end, [], [])
                if feature is not None:
                    yield feature
            continue

        features_by_start = {}
        features_by_end = {}
        for record in group_records:
            features_by_start.setdefault(record[0], []).append(record[2])
            features_by_end.setdefault(record[1], []).append(record[2])

        first_start = group_records[0][0]
        if include_ends and region is not None and region_start < first_start:
            feature = emit(
                region_start,
                first_start - 1,
                [],
                features_by_start[first_start],
            )
            if feature is not None:
                yield feature

        # Track the union of covered coordinates.  ``features_by_end`` tells
        # us which features support the right edge of a covered component,
        # and ``features_by_start`` tells us which features support the left
        # edge of the next component.
        component_end = None

        for start, end, feature in group_records:
            if component_end is None:
                component_end = end
                continue

            if start <= component_end + 1:
                if end > component_end:
                    component_end = end
                continue

            feature_out = emit(
                component_end + 1,
                start - 1,
                features_by_end[component_end],
                features_by_start[start],
            )
            if feature_out is not None:
                yield feature_out

            component_end = end

        if include_ends and region is not None and component_end < region_end:
            feature = emit(
                component_end + 1,
                region_end,
                features_by_end[component_end],
                [],
            )
            if feature is not None:
                yield feature


def coverage_gaps(*args, **kwargs):
    """Alias for :func:`uncovered_intervals`."""
    return uncovered_intervals(*args, **kwargs)


def blank_intervals(*args, **kwargs):
    """Alias for :func:`uncovered_intervals`."""
    return uncovered_intervals(*args, **kwargs)


def uncovered_regions(*args, **kwargs):
    """Alias for :func:`uncovered_intervals`."""
    return uncovered_intervals(*args, **kwargs)


def intervals(*args, **kwargs):
    """
    Yield ``(seqid, start, end)`` tuples for uncovered intervals.

    This is a lightweight convenience wrapper for callers that only need
    coordinates; use :func:`uncovered_intervals` when provenance is required.
    """
    for feature in uncovered_intervals(*args, **kwargs):
        yield feature.seqid, feature.start, feature.end
