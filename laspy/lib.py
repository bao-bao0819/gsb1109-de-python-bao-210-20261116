"""'Entry point' of the library, Contains the various functions meant to be
used directly by a user
"""

import copy
import io
import logging
import os
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple, Union

import numpy as np

from .compression import DecompressionSelection, LazBackend
from .errors import LaspyException
from .header import LasHeader, Version
from .lasappender import LasAppender
from .lasdata import LasData
from .lasmmap import LasMMAP
from .lasreader import LasReader
from .laswriter import LasWriter
from .point import ExtraBytesParams, PointFormat, dims, record
from .vlrs.vlrlist import VLRList
from .waveform.mode import WaveformMode

logger = logging.getLogger(__name__)


def open_las(
    source,
    mode="r",
    closefd=True,
    laz_backend=None,
    header=None,
    do_compress=None,
    encoding_errors: str = "strict",
    read_evlrs: bool = True,
    decompression_selection: DecompressionSelection = DecompressionSelection.all(),
    fullwave: str = "never",
) -> Union[LasReader, LasWriter, LasAppender]:
    """The laspy.open opens a LAS/LAZ file in one of the 3 supported
    mode:

     - "r" => Reading => a :class:`laspy.LasReader` will be returned
     - "w" => Writing => a :class:`laspy.LasWriter` will be returned
     - "a" => Appending => a :class:`laspy.LasAppender` will be returned


    When opening a file in 'w' mode, a header (:class:`laspy.LasHeader`)
    is required

        >>> with open_las('tests/data/simple.las') as f:
        ...     print(f.header.point_format.id)
        3


        >>> f = open('tests/data/simple.las', mode='rb')
        >>> with open_las(f,closefd=False) as flas:
        ...     print(flas.header)
        <LasHeader(1.2, <PointFormat(3, 0 bytes of extra dims)>)>
        >>> f.closed
        False
        >>> f.close()
        >>> f.closed
        True


        >>> f = open('tests/data/simple.las', mode='rb')
        >>> with open_las(f) as flas:
        ...    las = flas.read()
        >>> f.closed
        True

    Parameters
    ----------
    source: str or bytes or io.BytesIO
        if source is a str it must be a filename

    mode: Optional, the mode to open the file:
        - "r" for reading (default)
        - "w" for writing
        - "a" for appending

    laz_backend: Optional, laspy.LazBackend, the LAZ backend to use to handle decompression/compression

        By default available backends are detected, see LazBackend to see the
        preference order when multiple backends are available

    header: The header to use when opening in write mode.

    do_compress: optional, bool, only meaningful in writing mode:
        - None (default) guess if compression is needed using the file extension
          or if a laz_backend was explicitely provided
        - True compresses the file
        - False do not compress the file

    closefd: optional, bool, True by default
        Whether the stream/file object shall be closed, this only work
        when using open_las in a with statement. An exception is raised if
        closefd is specified and the source is a filename

    encoding_errors: str, default 'strict'
        Only used in writing and appending mode.
        How encoding errors should be treated.
        Possible values and their explanation can be seen here:
        https://docs.python.org/3/library/codecs.html#error-handlers.

    read_evlrs: bool, default True
            Only applies to 'r' mode.

            If True the evlrs will be read during the __init__ / file opening
            along with the LasHeader.

            It is fine for most of the cases,
            but can be problematic when opening file from a data stream like
            AWS S3 as EVLRs are located at the end of the files, thus
            will require to pull the whole file.

            Does nothing if the input file does not support
            EVLRs

    decompression_selection: DecompressionSelection, default All
        Only applies to 'r' mode and for files which suport selective decompression
        (version >= 1.4 and point format id >= 6), ignored otherwise.

        Allows to select which fields should be decompressed or not, allowing to save time
        by not decompressing unused fields.

        By default all fields are decompressed

    .. versionadded:: 2.4
        The ``read_evlrs`` and ``decompression_selection`` parameters.
    """
    waveform_mode = WaveformMode(fullwave)

    if mode == "r":
        if header is not None:
            raise LaspyException(
                "header argument is not used when opening in read mode, "
                "did you meant to open in write mode ?"
            )
        if do_compress is not None:
            raise LaspyException(
                "do_compress argument is not used when opening in read mode, "
                "did you meant to open in write mode ?"
            )
        if isinstance(source, (str, Path)):
            stream = open(source, mode="rb", closefd=closefd)
        elif isinstance(source, bytes):
            stream = io.BytesIO(source)
        else:
            stream = source
        try:
            return LasReader(
                stream,
                closefd=closefd,
                laz_backend=laz_backend,
                read_evlrs=read_evlrs,
                decompression_selection=decompression_selection,
                waveform_mode=waveform_mode,
            )
        except:
            if closefd:
                stream.close()
            raise
    elif mode == "w":
        if header is None:
            raise ValueError("A header is needed when opening a file for writing")

        if isinstance(source, (str, Path)):
            if do_compress is None:
                do_compress = os.path.splitext(source)[1].lower() == ".laz"
            stream = open(source, mode="wb+", closefd=closefd)
        elif isinstance(source, bytes):
            stream = io.BytesIO(source)
        else:
            assert source.seekable()
            stream = source

        try:
            return LasWriter(
                stream,
                header=header,
                do_compress=do_compress,
                laz_backend=laz_backend,
                closefd=closefd,
                encoding_errors=encoding_errors,
            )
        except:
            if closefd:
                stream.close()
            raise
    elif mode == "a":
        if isinstance(source, (str, Path)):
            stream = open(source, mode="rb+", closefd=closefd)
        elif isinstance(source, bytes):
            stream = io.BytesIO(source)
        else:
            stream = source

        try:
            return LasAppender(
                stream,
                closefd=closefd,
                laz_backend=laz_backend,
                encoding_errors=encoding_errors,
            )
        except:
            if closefd:
                stream.close()
            raise
    else:
        raise ValueError(f"Unknown mode '{mode}'")


def read_las(
    source,
    closefd=True,
    laz_backend=LazBackend.detect_available(),
    decompression_selection: DecompressionSelection = DecompressionSelection.all(),
    encoding_errors: str = "strict",
    fullwave: str = "never",
):
    """Entry point for reading las data in laspy

    Reads the whole file into memory.

    >>> las = read_las("tests/data/simple.las")
    >>> las.classification
    <SubFieldView([1 1 1 ... 1 1 1])>

    Parameters
    ----------
    source : str or io.BytesIO
        The source to read data from

    laz_backend: Optional, the backend to use when the file is as LAZ file.
                 By default laspy will find the backend to use by itself.
                 Use if you want a specific backend to be used

    closefd: bool
            if True and the source is a stream, the function will close it
            after it is done reading

    decompression_selection: DecompressionSelection,
        see :func:`laspy.open`

    encoding_errors: str, default 'strict'
        Only used in writing and appending mode.
        How encoding errors should be treated.
        Possible values and their explanation can be seen here:
        https://docs.python.org/3/library/codecs.html#error-handlers.

    Returns
    -------
    laspy.LasData
        The object you can interact with to get access to the LAS points & VLRs


    .. versionadded:: 2.4
        The ``decompression_selection`` parameter.

    .. versionadded:: 2.6
        The ``encoding_errors`` parameter.
    """
    with open_las(
        source,
        closefd=closefd,
        laz_backend=laz_backend,
        decompression_selection=decompression_selection,
        encoding_errors=encoding_errors,
        fullwave=fullwave,
    ) as reader:
        return reader.read()


def mmap_las(filename):
    """MMap a file, much like laspy did"""
    return LasMMAP(filename)


def create_las(
    *,
    point_format: Optional[Union[int, PointFormat]] = None,
    file_version: Optional[Union[str, Version]] = None,
):
    """Function to create a new empty las data object

    .. note::

        If you provide both point_format and file_version
        an exception will be raised if they are not compatible

    >>> las = create_las(point_format=6,file_version="1.2")
    Traceback (most recent call last):
     ...
    laspy.errors.LaspyException: Point format 6 is not compatible with file version 1.2


    If you provide only the point_format the file_version will automatically
    selected for you.

    >>> las = create_las(point_format=0)
    >>> las.header.version == '1.2'
    True

    >>> las = create_las(point_format=PointFormat(6))
    >>> las.header.version == '1.4'
    True


    Parameters
    ----------
    point_format:
        The point format you want the created file to have

    file_version:
        The las version you want the created las to have

    Returns
    -------
    laspy.lasdatas.base.LasBase
       A new las data object

    """
    header = LasHeader(point_format=point_format, version=file_version)
    return LasData(header=header)


def convert(source_las, *, point_format_id=None, file_version=None):
    """Converts a Las from one point format to another
    Automatically upgrades the file version if source file version is not compatible with
    the new point_format_id


    convert to point format 0

    >>> las = read_las('tests/data/simple.las')
    >>> las.header.version
    Version(major=1, minor=2)
    >>> las = convert(las, point_format_id=0)
    >>> las.header.point_format.id
    0
    >>> str(las.header.version)
    '1.2'

    convert to point format 6, which need version >= 1.4
    then convert back to point format 0, version is not downgraded

    >>> las = read_las('tests/data/simple.las')
    >>> str(las.header.version)
    '1.2'
    >>> las = convert(las, point_format_id=6)
    >>> las.header.point_format.id
    6
    >>> str(las.header.version)
    '1.4'
    >>> las = convert(las, point_format_id=0)
    >>> str(las.header.version)
    '1.4'

    an exception is raised if the requested point format is not compatible
    with the file version

    >>> las = read_las('tests/data/simple.las')
    >>> convert(las, point_format_id=6, file_version='1.2')
    Traceback (most recent call last):
     ...
    laspy.errors.LaspyException: Point format 6 is not compatible with file version 1.2

    Parameters
    ----------
    source_las : laspy.lasdatas.base.LasBase
        The source data to be converted

    point_format_id : int, optional
        The new point format id (the default is None, which won't change the source format id)

    file_version : str, optional,
        The new file version. None by default which means that the file_version
        may be upgraded for compatibility with the new point_format. The file version will not
        be downgraded.

    Returns
    -------
        laspy.lasdatas.base.LasBase
    """
    if point_format_id is None:
        point_format_id = source_las.point_format.id

    if file_version is None:
        file_version = max(
            str(source_las.header.version),
            dims.preferred_file_version_for_point_format(point_format_id),
        )
    else:
        file_version = str(file_version)
        dims.raise_if_version_not_compatible_with_fmt(point_format_id, file_version)

    version = Version.from_str(file_version)

    point_format = PointFormat(point_format_id)
    point_format.dimensions.extend(source_las.point_format.extra_dimensions)

    header = copy.deepcopy(source_las.header)
    header.set_version_and_point_format(version, point_format)

    if source_las.evlrs is not None:
        evlrs = VLRList(source_las.evlrs.copy())
    else:
        evlrs = None

    points = record.PackedPointRecord.from_point_record(
        source_las.points, header.point_format
    )
    las = LasData(header=header, points=points)

    if file_version < "1.4" and evlrs is not None and evlrs:
        logger.warning(
            "The source contained {} EVLRs,"
            " they will be lost as version {} doest not support them".format(
                len(evlrs), file_version
            )
        )
    else:
        las.evlrs = evlrs

    return las


class MergeConflict(NamedTuple):
    """Describes an extra dimension that could not be merged by
    :func:`merge_las`.

    The dimension is left out of the merged result and reported
    here instead of being silently merged (or dropped).
    """

    dimension: str
    """Name of the extra dimension concerned"""

    reason: str
    """Why the dimension could not be merged, one of
    'units_mismatch', 'dtype_overflow' or 'shape_mismatch'"""

    sources: Tuple[int, ...]
    """Indices of the input las files involved in the conflict"""

    message: str
    """Human readable description of the conflict"""


MERGE_SOURCE_DIM_NAME = "source_file_id"
"""Name of the extra dimension added by merge_las to map each
merged point back to the index of the las file it comes from"""


def merge_las(las_files):
    """Merges multiple las files into a single one.

    The points of all the input files are concatenated, and their extra
    dimensions are harmonized using the declarations (scales, offsets,
    no_data, description) found in each file, so that e.g. a 'temperature'
    dimension stored as raw integer codes in one file and as scaled values
    with an invalid (no_data) code in another file still produces a single
    mergeable column:

    - invalid values are recognized on the *raw* codes first (a raw value
      equal to the declared no_data is invalid), then only the valid values
      are converted (``value * scale + offset``).
    - if a dimension is scaled, has a no_data code, or is missing from some
      of the input files, the merged column is float64 and invalid or
      missing values are represented as NaN. A point whose file simply does
      not have the dimension gets NaN, which stays distinguishable from an
      actual value of exactly 0.0.
    - otherwise the merged column keeps the type declared by the first file
      holding the dimension. If the values of another file do not fit in
      that target type, or if the files declare disagreeing units
      (descriptions), the dimension is *not* merged: it is reported as a
      :class:`MergeConflict` instead.

    A 'source_file_id' extra dimension is added to the merged result,
    mapping each point to the index of the input file it comes from.

    The returned :class:`laspy.LasData` also carries a ``conflicts``
    attribute: the list of :class:`MergeConflict` encountered (empty when
    everything could be merged).

    >>> import numpy as np
    >>> las_a = create_las(point_format=3)
    >>> las_a.add_extra_dim(ExtraBytesParams("temperature", "i2", description="celsius"))
    >>> las_a.points = record.ScaleAwarePointRecord.zeros(3, header=las_a.header)
    >>> las_a.temperature = np.array([0, 21, 22], dtype=np.int16)
    >>> las_b = create_las(point_format=3)
    >>> las_b.add_extra_dim(ExtraBytesParams(
    ...     "temperature", "u1", description="celsius",
    ...     scales=[0.5], offsets=[-20.0], no_data=[255]))
    >>> las_b.points = record.ScaleAwarePointRecord.zeros(2, header=las_b.header)
    >>> las_b.points.array["temperature"][:] = [100, 255]
    >>> merged = merge_las([las_a, las_b])
    >>> [float(t) for t in merged.temperature]
    [0.0, 21.0, 22.0, 30.0, nan]
    >>> [int(s) for s in merged.source_file_id]
    [0, 0, 0, 1, 1]
    >>> merged.conflicts
    []

    Parameters
    ----------
    las_files: iterable of laspy.LasData (or a single laspy.LasData)
        The files to merge. They must all have the same point format id.

    Returns
    -------
    laspy.LasData
        The merged las data.
    """
    if isinstance(las_files, LasData):
        return las_files

    las_files = list(las_files)
    if len(las_files) == 0:
        raise LaspyException("Cannot merge 0 las files")
    if len(las_files) == 1:
        return las_files[0]

    point_format_ids = {las.point_format.id for las in las_files}
    if len(point_format_ids) != 1:
        raise LaspyException(
            "Cannot merge las files with different point format ids "
            f"({sorted(point_format_ids)})"
        )

    if any(
        MERGE_SOURCE_DIM_NAME in las.point_format.extra_dimension_names
        for las in las_files
    ):
        raise LaspyException(
            f"Cannot merge: an input file already has a "
            f"'{MERGE_SOURCE_DIM_NAME}' extra dimension"
        )

    per_file_decls = []
    dim_names = []
    for las in las_files:
        decls = {dim.name: dim for dim in las.point_format.extra_dimensions}
        per_file_decls.append(decls)
        for name in decls:
            if name not in dim_names:
                dim_names.append(name)

    conflicts: List[MergeConflict] = []
    merged_dim_params: List[ExtraBytesParams] = []
    merged_dim_converters = []

    for name in dim_names:
        decls = [decls.get(name) for decls in per_file_decls]
        present = [(i, d) for i, d in enumerate(decls) if d is not None]

        # The units must be agreed upon: all the non-empty
        # descriptions must be identical
        descriptions = {d.description for _, d in present if d.description}
        if len(descriptions) > 1:
            conflicts.append(
                MergeConflict(
                    dimension=name,
                    reason="units_mismatch",
                    sources=tuple(i for i, _ in present),
                    message=(
                        f"extra dimension '{name}' has disagreeing "
                        f"unit/description declarations: {sorted(descriptions)}"
                    ),
                )
            )
            continue

        num_elements = {d.num_elements for _, d in present}
        if len(num_elements) != 1:
            conflicts.append(
                MergeConflict(
                    dimension=name,
                    reason="shape_mismatch",
                    sources=tuple(i for i, _ in present),
                    message=(
                        f"extra dimension '{name}' has incompatible "
                        f"element counts: {sorted(num_elements)}"
                    ),
                )
            )
            continue
        num_element = num_elements.pop()

        is_missing_somewhere = any(d is None for d in decls)
        needs_float = is_missing_somewhere or any(
            d.is_scaled or d.no_data is not None for _, d in present
        )

        if needs_float:
            # float64 can hold any converted value as well as
            # NaN for invalid / missing values
            target_base = np.dtype(np.float64)
        else:
            # The target type is the one declared by the first
            # file holding the dimension, every other file's
            # values must fit in it
            target_base = present[0][1].dtype.base
            overflowing = []
            for i, d in present:
                src_base = d.dtype.base
                if src_base == target_base or np.can_cast(
                    src_base, target_base, casting="safe"
                ):
                    continue
                if src_base.kind in "iu" and target_base.kind in "iu":
                    values = np.asarray(las_files[i].points.array[name])
                    if values.size == 0:
                        continue
                    info = np.iinfo(target_base)
                    if values.min() >= info.min and values.max() <= info.max:
                        continue
                overflowing.append(i)
            if overflowing:
                conflicts.append(
                    MergeConflict(
                        dimension=name,
                        reason="dtype_overflow",
                        sources=tuple(overflowing),
                        message=(
                            f"extra dimension '{name}': values from source(s) "
                            f"{overflowing} do not fit in the target "
                            f"type '{target_base}'"
                        ),
                    )
                )
                continue

        if num_element == 1:
            target_dtype = target_base
        else:
            target_dtype = np.dtype((target_base, (num_element,)))

        description = next(iter(descriptions), "")
        merged_dim_params.append(
            ExtraBytesParams(name, target_dtype, description=description)
        )

        def make_converter(decls=decls, target_base=target_base, num_element=num_element):
            def convert(file_index):
                las = las_files[file_index]
                decl = decls[file_index]
                if decl is None:
                    # The file does not have this dimension: NaN,
                    # which must not be confused with an actual 0.0
                    count = len(las.points)
                    shape = (count,) if num_element == 1 else (count, num_element)
                    return np.full(shape, np.nan, dtype=target_base)

                raw = np.asarray(las.points.array[decl.name])
                if target_base.kind == "f":
                    scale = 1.0
                    if decl.scales is not None:
                        scale = np.asarray(decl.scales, dtype=np.float64)
                    offset = 0.0
                    if decl.offsets is not None:
                        offset = np.asarray(decl.offsets, dtype=np.float64)
                    # 1) invalid values are recognized on the raw codes,
                    #    before any conversion
                    invalid = None
                    if decl.no_data is not None:
                        invalid = raw == np.asarray(decl.no_data)
                    # 2) then the values are converted
                    converted = raw.astype(np.float64) * scale + offset
                    if invalid is not None:
                        converted = np.where(invalid, np.nan, converted)
                    return converted

                return raw.astype(target_base)

            return convert

        merged_dim_converters.append(make_converter())

    header = copy.deepcopy(las_files[0].header)
    existing_extra_dims = list(header.point_format.extra_dimension_names)
    if existing_extra_dims:
        header.remove_extra_dims(existing_extra_dims)
    header.add_extra_dims(merged_dim_params)

    num_files = len(las_files)
    if num_files <= np.iinfo(np.uint8).max + 1:
        source_dtype = np.dtype(np.uint8)
    elif num_files <= np.iinfo(np.uint16).max + 1:
        source_dtype = np.dtype(np.uint16)
    else:
        source_dtype = np.dtype(np.uint32)
    header.add_extra_dim(
        ExtraBytesParams(
            MERGE_SOURCE_DIM_NAME,
            source_dtype,
            description="Index of the source las file",
        )
    )

    total_points = sum(len(las.points) for las in las_files)
    points = record.ScaleAwarePointRecord.zeros(total_points, header=header)
    merged = LasData(header=header, points=points)

    start = 0
    for file_index, las in enumerate(las_files):
        end = start + len(las.points)
        sl = slice(start, end)

        # x, y, z go through the scaled views so that files with
        # different header scales/offsets are properly rescaled
        merged.x[sl] = las.x
        merged.y[sl] = las.y
        merged.z[sl] = las.z

        # The other standard fields share the same layout
        # (same point format id) and are copied raw
        extra_names = set(per_file_decls[file_index])
        for field_name in las.points.array.dtype.names:
            if field_name in extra_names or field_name in ("X", "Y", "Z"):
                continue
            merged.points.array[field_name][sl] = las.points.array[field_name]

        for params, convert in zip(merged_dim_params, merged_dim_converters):
            merged.points.array[params.name][sl] = convert(file_index)

        merged.points.array[MERGE_SOURCE_DIM_NAME][sl] = file_index

        start = end

    merged.update_header()
    merged.conflicts = conflicts
    return merged


def write_then_read_again(
    las, do_compress=False, laz_backend=LazBackend.detect_available()
):
    """writes the given las into memory using BytesIO and
    reads it again, returning the newly read file.

    Mostly used for testing purposes, without having to write to disk
    """
    out = io.BytesIO()
    las.write(out, do_compress=do_compress, laz_backend=laz_backend)
    out.seek(0)
    return read_las(out)
