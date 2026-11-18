"""
Unify SAS Transport v5 and v8 datasets in a single store.

This module ingests XPORT documents written in the Version 5 or
Version 8/9 transport format into one :class:`UnifiedStore`, and can
recompute a document for each original version on demand.

The caller supplies two things:

1. A field identity mapping (:class:`Identity`) pairing each canonical
   field with the name it carries in each version's file, because a
   name that is legal in v8 (up to 32 characters) may be illegal in
   v5 (up to 8 characters).
2. A restricted target specification (:class:`TargetSpec`) per version,
   describing the limits each emitted document must satisfy.  Defaults
   come from the format specification; callers may tighten them.

Given those, :meth:`UnifiedStore.plan` computes the schema that can be
shared losslessly across all target versions, plus the minimal set of
items that must be isolated per version or preserved through a side
channel.  Long names, long labels, long format descriptions, and SAS
special missing values each carry provenance recording which source
version supplied them.

Decodability does not imply cross-version compatibility: after writing,
:meth:`UnifiedStore.verify` reads the document back with the version's
own loader and explains every difference in values and metadata.
"""

# Standard Library
import logging
import warnings
from datetime import datetime
from types import SimpleNamespace

# Community Packages
import pandas as pd

# Xport Modules
import xport
import xport.v56
import xport.v89

__all__ = [
    'Identity',
    'TargetSpec',
    'Provenance',
    'SidecarEntry',
    'FieldPlan',
    'UnificationPlan',
    'VerificationReport',
    'IsolationError',
    'UnifiedStore',
]

LOG = logging.getLogger(__name__)

_MODULES = {
    5: xport.v56,
    8: xport.v89,
}

_HEADERS = {
    5: xport.v56.MemberHeader,
    8: xport.v89.MemberHeader,
}

_TEMPLATES = {
    5: xport.v56.Library.template,
    8: xport.v89.Library.template,
}

_SPECIAL_MISSING_TAGS = b'_ABCDEFGHIJKLMNOPQRSTUVWXYZ'


class IsolationError(ValueError):
    """
    A field cannot be represented in the requested target version.
    """


class Identity:
    """
    Caller-provided field identity mapping across XPORT versions.

    ``names`` maps a version number (5 or 8) to the name the field
    carries in that version's document, for example
    ``{5: 'TEMP', 8: 'TemperatureCelsius'}``.  Dataset identities use
    the same structure.
    """

    def __init__(self, canonical, names):
        """
        Initialize an identity mapping for one canonical field.
        """
        self.canonical = canonical
        self.names = {int(version): str(name) for version, name in dict(names).items()}

    def __repr__(self):
        """
        REPL-format string.
        """
        return f'{type(self).__name__}({self.canonical!r}, {self.names!r})'


class TargetSpec:
    """
    Restricted target specification for one XPORT version.

    Limits default to the format specification.  Callers may pass
    tighter limits as keyword overrides, for example
    ``TargetSpec(8, label_length=40)`` to restrict a v8 target to
    v5-style labels.
    """

    DEFAULTS = {
        5: dict(
            name_length=8,
            label_length=40,
            format_length=8,
            dataset_name_length=8,
            dataset_label_length=40,
            character_width=200,
        ),
        8: dict(
            name_length=32,
            label_length=256,
            format_length=8,
            dataset_name_length=32,
            dataset_label_length=40,
            character_width=32767,
        ),
    }

    LIMITS = tuple(next(iter(DEFAULTS.values())).keys())

    def __init__(self, version, **restrictions):
        """
        Initialize a target specification for ``version``.
        """
        try:
            defaults = self.DEFAULTS[int(version)]
        except KeyError:
            raise ValueError(f'Unsupported XPORT version {version!r}')
        unknown = set(restrictions) - set(defaults)
        if unknown:
            raise TypeError(f'Unknown restrictions {sorted(unknown)}')
        self.version = int(version)
        for name, value in {**defaults, **restrictions}.items():
            setattr(self, name, value)

    def __repr__(self):
        """
        REPL-format string.
        """
        limits = ', '.join(f'{name}={getattr(self, name)}' for name in self.LIMITS)
        return f'{type(self).__name__}({self.version}, {limits})'


class Provenance:
    """
    Source of a metadata item: which XPORT version supplied it.
    """

    def __init__(self, dataset, field, attribute, value, version):
        """
        Initialize a provenance record.
        """
        self.dataset = dataset
        self.field = field
        self.attribute = attribute
        self.value = value
        self.version = version

    def __repr__(self):
        """
        REPL-format string.
        """
        where = self.dataset if self.field is None else f'{self.dataset}.{self.field}'
        return f'<{type(self).__name__} {where}.{self.attribute} = {self.value!r} from v{self.version}>'


class SidecarEntry:
    """
    A full-fidelity value preserved beside the shared schema.

    ``attribute`` is one of ``'name'``, ``'label'``, ``'format'``,
    ``'informat'``, or ``'missing'``.  ``source`` is the version that
    supplied the value.  ``versions`` lists the target versions that
    cannot carry the value natively; an empty tuple means all targets
    need the side channel (as for special missing values, which a
    Pandas ``NaN`` cannot express).
    """

    def __init__(self, dataset, field, attribute, value, source, reason, versions=()):
        """
        Initialize a sidecar entry.
        """
        self.dataset = dataset
        self.field = field
        self.attribute = attribute
        self.value = value
        self.source = source
        self.reason = reason
        self.versions = tuple(versions)

    def __repr__(self):
        """
        REPL-format string.
        """
        where = self.dataset if self.field is None else f'{self.dataset}.{self.field}'
        return f'<{type(self).__name__} {where}.{self.attribute} from v{self.source}: {self.reason}>'


class FieldPlan:
    """
    Shareability analysis for one field of the unified schema.
    """

    def __init__(self, dataset, field, identities, shared, isolated, sidecar):
        """
        Initialize a field plan.
        """
        self.dataset = dataset
        self.field = field
        self.identities = identities
        self.shared = shared
        self.isolated = isolated
        self.sidecar = sidecar

    def __repr__(self):
        """
        REPL-format string.
        """
        status = 'shared' if self.shared else 'split'
        return f'<{type(self).__name__} {self.dataset}.{self.field} {status}>'


class UnificationPlan:
    """
    The losslessly shareable schema plus the minimal isolated/sidecar sets.
    """

    def __init__(self, fields, sidecar):
        """
        Initialize a unification plan.
        """
        self.fields = list(fields)
        self.sidecar = list(sidecar)

    @property
    def shared(self):
        """
        Fields losslessly shareable across all target versions.
        """
        return [(f.dataset, f.field) for f in self.fields if f.shared]

    @property
    def isolated(self):
        """
        Fields that must be isolated per version, with reasons.
        """
        return {(f.dataset, f.field): f.isolated for f in self.fields if f.isolated}

    def __repr__(self):
        """
        REPL-format string.
        """
        return (
            f'<{type(self).__name__} shared={len(self.shared)}'
            f' isolated={len(self.isolated)} sidecar={len(self.sidecar)}>'
        )


class VerificationReport:
    """
    Result of reading a recomputed document back with its own loader.

    ``explanations`` accounts for every expected difference between the
    unified store and the read-back (truncations, identity mapping,
    restored special missings).  ``discrepancies`` lists anything that
    could not be explained; the round-trip is faithful only when it is
    empty.
    """

    def __init__(self, version):
        """
        Initialize an empty report for ``version``.
        """
        self.version = version
        self.explanations = []
        self.discrepancies = []

    @property
    def ok(self):
        """
        Whether every value and metadata item read back is explainable.
        """
        return not self.discrepancies

    def __bool__(self):
        """
        Truthy when the round-trip is fully explainable.
        """
        return self.ok

    def __repr__(self):
        """
        REPL-format string.
        """
        status = 'ok' if self.ok else f'{len(self.discrepancies)} discrepancies'
        return f'<{type(self).__name__} v{self.version} {status}>'


def _scan_special_missings(bytestring, version):
    """
    Locate SAS special missing values in an XPORT document.

    Returns a mapping of ``(dataset name, variable name)`` to a mapping
    of row number to special-missing tag (``'A'`` to ``'Z'`` or
    ``'_'``).  Plain missing values (``.``) are not special and are
    skipped.  Pandas represents every missing value as a plain ``NaN``,
    so these tags survive only through this side channel.
    """
    Header = _HEADERS[version]
    mview = memoryview(bytestring)
    matches = list(Header.pattern.finditer(mview))
    found = {}
    for k, mo in enumerate(matches):
        header = Header.from_bytes(mview[mo.start(0):mo.end(0)])
        stop = matches[k + 1].start(0) if k + 1 < len(matches) else len(mview)
        obs = mview[mo.end(0):stop]
        layout = []
        stride = 0
        for namestr in header.values():
            layout.append((namestr.name, namestr.vtype, stride, namestr.length))
            stride += namestr.length
        if stride == 0:
            continue
        sentinel = b' ' * stride
        row = 0
        for i in range(0, len(obs) - stride + 1, stride):
            chunk = bytes(obs[i:i + stride])
            if chunk == sentinel:
                break
            for name, vtype, offset, length in layout:
                if vtype != xport.VariableType.NUMERIC:
                    continue
                raw = chunk[offset:offset + length]
                if raw[:1] in _SPECIAL_MISSING_TAGS and not any(raw[1:]):
                    tag = raw[:1].decode('ascii')
                    found.setdefault((header.name, name), {})[row] = tag
            row += 1
    return found


def _encoded_length(text):
    """
    Length of ``text`` in the metadata encoding, or ``None`` if unencodable.
    """
    try:
        return len(text.encode(xport.v56.TEXT_METADATA_ENCODING))
    except (UnicodeEncodeError, AttributeError):
        return None


class UnifiedStore:
    """
    A single store for datasets ingested from v5 and v8 XPORT documents.

    ``identities`` and ``dataset_identities`` are caller-provided
    :class:`Identity` mappings for variables and datasets.
    ``target_specs`` are caller-provided :class:`TargetSpec` objects;
    any version not specified uses the format defaults.
    """

    def __init__(self, identities=(), target_specs=(), dataset_identities=()):
        """
        Initialize an empty store.
        """
        self.identities = list(identities)
        self.dataset_identities = list(dataset_identities)
        given = {spec.version: spec for spec in target_specs}
        self.target_specs = {version: given.get(version) or TargetSpec(version) for version in (5, 8)}
        self._datasets = {}
        self._base_version = {}
        self._library_meta = {}
        self._provenance = []
        self._missings = {}
        self._missings_source = {}
        self.conflicts = []

    def __repr__(self):
        """
        REPL-format string.
        """
        versions = sorted(self._library_meta)
        return f'<{type(self).__name__} datasets={list(self._datasets)} versions={versions}>'

    def _canonical_dataset(self, name, version):
        for identity in self.dataset_identities:
            if identity.names.get(version) == name:
                return identity.canonical
        return name

    def _canonical_field(self, name, version):
        for identity in self.identities:
            if identity.names.get(version) == name:
                return identity.canonical
        return name

    def _dataset_name(self, canonical, version):
        for identity in self.dataset_identities:
            if identity.canonical == canonical:
                return identity.names.get(version, canonical)
        return canonical

    def _field_name(self, canonical, version):
        for identity in self.identities:
            if identity.canonical == canonical:
                return identity.names.get(version, canonical)
        return canonical

    @property
    def datasets(self):
        """
        The unified datasets, keyed by canonical dataset name.
        """
        return dict(self._datasets)

    @property
    def provenance(self):
        """
        Provenance records for every ingested metadata item.
        """
        return list(self._provenance)

    def _record(self, dataset, field, attribute, value, version):
        self._provenance.append(Provenance(dataset, field, attribute, value, version))

    def ingest(self, bytestring, version):
        """
        Decode an XPORT document of ``version`` into the unified store.

        Special missing values are recovered from the raw bytes and
        preserved through the side channel, because decoding to Pandas
        objects erases their tags.
        """
        version = int(version)
        if version not in _MODULES:
            raise ValueError(f'Unsupported XPORT version {version!r}')
        library = _MODULES[version].loads(bytestring)
        missings = _scan_special_missings(bytestring, version)
        self._library_meta[version] = {
            'created': library.created,
            'modified': library.modified,
            'sas_os': library.sas_os,
            'sas_version': library.sas_version,
        }
        for dataset in library.values():
            self._ingest_dataset(dataset, version, missings)
        LOG.info(f'Ingested {len(library)} dataset(s) from v{version}')
        return self

    def ingest_library(self, library, version):
        """
        Ingest an already-decoded ``xport.Library``.

        Special-missing provenance is unavailable unless the raw bytes
        are supplied to :meth:`ingest` instead.
        """
        version = int(version)
        self._library_meta[version] = {
            'created': library.created,
            'modified': library.modified,
            'sas_os': library.sas_os,
            'sas_version': library.sas_version,
        }
        for dataset in library.values():
            self._ingest_dataset(dataset, version, {})
        return self

    def _ingest_dataset(self, dataset, version, missings):
        ds_key = self._canonical_dataset(dataset.name, version)
        field_keys = {name: self._canonical_field(name, version) for name in dataset.columns}

        self._record(ds_key, None, 'name', dataset.name, version)
        self._record(ds_key, None, 'label', dataset.dataset_label, version)
        self._record(ds_key, None, 'dataset_type', dataset.dataset_type, version)
        self._record(ds_key, None, 'created', dataset.created, version)
        self._record(ds_key, None, 'modified', dataset.modified, version)
        self._record(ds_key, None, 'sas_os', dataset.sas_os, version)
        self._record(ds_key, None, 'sas_version', dataset.sas_version, version)
        for name, variable in dataset.items():
            fk = field_keys[name]
            self._record(ds_key, fk, 'name', name, version)
            self._record(ds_key, fk, 'label', variable.label, version)
            self._record(ds_key, fk, 'format', variable.format, version)
            self._record(ds_key, fk, 'informat', variable.informat, version)
            self._record(ds_key, fk, 'width', variable.width, version)
            self._record(ds_key, fk, 'vtype', variable.vtype, version)

        for (ds_name, var_name), rows in missings.items():
            if ds_name != dataset.name:
                continue
            fk = field_keys.get(var_name, var_name)
            key = (ds_key, fk)
            self._missings.setdefault(key, {}).update(rows)
            self._missings_source[key] = version
            self._record(ds_key, fk, 'missing', dict(rows), version)

        canonical = self._build_canonical(dataset, ds_key, field_keys)
        if ds_key not in self._datasets:
            self._datasets[ds_key] = canonical
            self._base_version[ds_key] = version
        else:
            self._check_conflicts(self._datasets[ds_key], canonical, ds_key, version)
            if version >= self._base_version[ds_key]:
                self._datasets[ds_key] = canonical
                self._base_version[ds_key] = version

    @staticmethod
    def _build_canonical(dataset, ds_key, field_keys):
        data = {}
        for name, variable in dataset.items():
            fk = field_keys[name]
            v = xport.Variable(
                list(variable),
                name=fk,
                label=variable.label,
                width=variable.width,
            )
            v.format = variable.format
            v.informat = variable.informat
            v.vtype = variable.vtype
            data[fk] = v
        return xport.Dataset(
            data,
            name=ds_key,
            label=dataset.dataset_label,
            dataset_type=dataset.dataset_type,
            created=dataset.created,
            modified=dataset.modified,
            sas_os=dataset.sas_os,
            sas_version=dataset.sas_version,
        )

    def _check_conflicts(self, existing, new, ds_key, version):
        base = self._base_version[ds_key]
        for fk in new.columns:
            if fk not in existing.columns:
                continue
            character = new[fk].vtype == xport.VariableType.CHARACTER
            a, b = list(existing[fk]), list(new[fk])
            if len(a) != len(b):
                self._conflict(ds_key, fk, f'{len(a)} rows from v{base}, {len(b)} rows from v{version}')
                continue
            for i, (x, y) in enumerate(zip(a, b)):
                if pd.isna(x) and pd.isna(y):
                    continue
                if character:
                    differ = str(x).rstrip() != str(y).rstrip()
                else:
                    try:
                        differ = float(x) != float(y)
                    except (TypeError, ValueError):
                        differ = True
                if differ:
                    self._conflict(ds_key, fk, f'row {i}: {x!r} from v{base} != {y!r} from v{version}')

    def _conflict(self, ds_key, fk, message):
        note = f'Conflict in {ds_key}.{fk}: {message}'
        LOG.warning(note)
        warnings.warn(note)
        self.conflicts.append(note)

    def plan(self):
        """
        Compute the shareable schema and the minimal isolated/sidecar sets.

        A field is losslessly shareable when its identity, label,
        format, informat, and values satisfy every target specification
        (using the caller's identity mapping where names differ).
        Anything exceeding a target's limits is preserved through the
        side channel with its provenance; values that a target cannot
        represent at all are isolated from that target.
        """
        fields = []
        sidecar = []
        for ds_key, ds in self._datasets.items():
            source = self._base_version[ds_key]
            for version, spec in self.target_specs.items():
                name = self._dataset_name(ds_key, version)
                n = _encoded_length(name)
                if n is None or n > spec.dataset_name_length:
                    sidecar.append(SidecarEntry(
                        ds_key, None, 'name', ds_key, source,
                        f'dataset name {name!r} exceeds v{version} limit'
                        f' {spec.dataset_name_length}',
                        versions=(version,),
                    ))
                label = ds.dataset_label or ''
                n = _encoded_length(label)
                if n is None or n > spec.dataset_label_length:
                    sidecar.append(SidecarEntry(
                        ds_key, None, 'label', label, source,
                        f'dataset label exceeds v{version} limit {spec.dataset_label_length}',
                        versions=(version,),
                    ))
            for fk, variable in ds.items():
                entries = []
                isolated = {}
                identities = {}
                for version, spec in self.target_specs.items():
                    name = self._field_name(fk, version)
                    identities[version] = name
                    n = _encoded_length(name)
                    if n is None or n > spec.name_length:
                        entries.append(SidecarEntry(
                            ds_key, fk, 'name', fk, source,
                            f'variable name {name!r} exceeds v{version} limit'
                            f' {spec.name_length}',
                            versions=(version,),
                        ))
                    label = variable.label or ''
                    n = _encoded_length(label)
                    if n is None or n > spec.label_length:
                        entries.append(SidecarEntry(
                            ds_key, fk, 'label', label, source,
                            f'label exceeds v{version} limit {spec.label_length}',
                            versions=(version,),
                        ))
                    for attribute in ('format', 'informat'):
                        form = getattr(variable, attribute)
                        if form is None or not form.name:
                            continue
                        n = _encoded_length(form.name)
                        if n is None or n > spec.format_length:
                            entries.append(SidecarEntry(
                                ds_key, fk, attribute, str(form), source,
                                f'{attribute} name {form.name!r} exceeds v{version}'
                                f' limit {spec.format_length}',
                                versions=(version,),
                            ))
                    if variable.vtype == xport.VariableType.CHARACTER:
                        width = variable.width or 0
                        if width > spec.character_width:
                            isolated[version] = (
                                f'character width {width} exceeds v{version}'
                                f' limit {spec.character_width}'
                            )
                key = (ds_key, fk)
                if key in self._missings:
                    entries.append(SidecarEntry(
                        ds_key, fk, 'missing', dict(self._missings[key]),
                        self._missings_source[key],
                        'Pandas NaN erases the SAS special-missing tag',
                    ))
                sidecar.extend(entries)
                fields.append(FieldPlan(
                    dataset=ds_key,
                    field=fk,
                    identities=identities,
                    shared=not entries and not isolated,
                    isolated=isolated,
                    sidecar=entries,
                ))
        return UnificationPlan(fields, sidecar)

    @staticmethod
    def _truncate(text, limit):
        if text is None:
            return ''
        return str(text)[:limit]

    @staticmethod
    def _conform_format(form, limit, cls):
        if form is None:
            return None
        name = (form.name or '')[:limit]
        if cls is xport.Format:
            return xport.Format(name, form.length, form.decimals, form.justify)
        return xport.Informat(name, form.length, form.decimals)

    @staticmethod
    def _normalize(variable):
        vtype = variable.vtype
        if vtype is None:
            vtype = (xport.VariableType.CHARACTER if variable.dtype.kind == 'O'
                     else xport.VariableType.NUMERIC)
        if vtype == xport.VariableType.CHARACTER:
            return ['' if pd.isna(v) else str(v) for v in variable]
        return [float('nan') if pd.isna(v) else float(v) for v in variable]

    def _conform_dataset(self, ds_key, version, spec, drop=()):
        ds = self._datasets[ds_key]
        data = {}
        for fk, variable in ds.items():
            if fk in drop:
                continue
            name = self._truncate(self._field_name(fk, version), spec.name_length)
            v = xport.Variable(
                self._normalize(variable),
                name=name,
                label=self._truncate(variable.label, spec.label_length),
                width=variable.width,
            )
            v.format = self._conform_format(variable.format, spec.format_length, xport.Format)
            v.informat = self._conform_format(variable.informat, spec.format_length, xport.Informat)
            v.vtype = variable.vtype
            data[name] = v
        return xport.Dataset(
            data,
            name=self._truncate(self._dataset_name(ds_key, version), spec.dataset_name_length),
            label=self._truncate(ds.dataset_label, spec.dataset_label_length),
            dataset_type=ds.dataset_type,
            created=ds.created,
            modified=ds.modified,
            sas_os=ds.sas_os,
            sas_version=ds.sas_version,
        )

    def _member_bytes(self, ds_key, conformed, version):
        header = _HEADERS[version].from_dataset(conformed)
        rows = [list(t) for t in conformed.itertuples(index=False, name=None)]
        positions = {name: i for i, name in enumerate(conformed.columns)}
        for (d, fk), tags in self._missings.items():
            if d != ds_key:
                continue
            name = self._truncate(self._field_name(fk, version),
                                  self.target_specs[version].name_length)
            if name not in positions:
                continue
            for row, tag in tags.items():
                if row < len(rows):
                    rows[row][positions[name]] = getattr(xport.NaN, tag)
        observations = xport.v56.Observations(iter(rows), header)
        return bytes(header) + bytes(observations)

    def dump(self, version, fp=None, drop_isolated=False):
        """
        Recompute an XPORT document for an original ``version``.

        Fields isolated from ``version`` raise :class:`IsolationError`
        unless ``drop_isolated`` is given, in which case they are
        omitted from that version's document.
        """
        version = int(version)
        if version not in _MODULES:
            raise ValueError(f'Unsupported XPORT version {version!r}')
        spec = self.target_specs[version]
        plan = self.plan()
        blocked = {key: reasons for key, reasons in plan.isolated.items() if version in reasons}
        if blocked and not drop_isolated:
            details = '; '.join(f'{d}.{f}: {r[version]}' for (d, f), r in blocked.items())
            raise IsolationError(f'Cannot represent in v{version}: {details}')
        drops = {}
        if drop_isolated:
            for (d, fk), reasons in blocked.items():
                drops.setdefault(d, set()).add(fk)
        meta = self._library_meta.get(version)
        if meta is None:
            meta = next(iter(self._library_meta.values()), {})
        members = []
        for ds_key in self._datasets:
            conformed = self._conform_dataset(ds_key, version, spec, drop=drops.get(ds_key, ()))
            members.append(self._member_bytes(ds_key, conformed, version))
        namespace = SimpleNamespace(
            sas_version=meta.get('sas_version') or '',
            sas_os=meta.get('sas_os') or '',
        )
        created = meta.get('created') or datetime.now()
        modified = meta.get('modified') or created
        bytestring = _TEMPLATES[version] % {
            b'version': xport.v56.text_encode(namespace, 'sas_version', 8),
            b'os': xport.v56.text_encode(namespace, 'sas_os', 8),
            b'created': xport.v56.strftime(created),
            b'modified': xport.v56.strftime(modified),
            b'members': b''.join(members),
        }
        if fp is not None:
            fp.write(bytestring)
        return bytestring

    def verify(self, version, drop_isolated=False):
        """
        Recompute for ``version`` and read back with its own loader.

        Every difference in values or metadata between the unified
        store and the read-back must be explained by the plan --
        truncation recorded in the side channel, caller-provided
        identity mapping, or restored special missings.  Anything else
        is reported as a discrepancy.
        """
        version = int(version)
        spec = self.target_specs[version]
        bytestring = self.dump(version, drop_isolated=drop_isolated)
        library = _MODULES[version].loads(bytestring)
        report = VerificationReport(version)

        plan = self.plan()
        drops = {}
        if drop_isolated:
            for (d, fk), reasons in plan.isolated.items():
                if version in reasons:
                    drops.setdefault(d, set()).add(fk)
                    report.explanations.append(
                        f'{d}.{fk}: isolated from v{version} ({reasons[version]}); omitted'
                    )
        for entry in plan.sidecar:
            if entry.versions and version not in entry.versions:
                continue
            where = entry.dataset if entry.field is None else f'{entry.dataset}.{entry.field}'
            report.explanations.append(
                f'{where}.{entry.attribute}: {entry.reason};'
                f' full value preserved from v{entry.source}'
            )

        for ds_key in self._datasets:
            expected = self._conform_dataset(ds_key, version, spec, drop=drops.get(ds_key, ()))
            try:
                got = library[expected.name]
            except KeyError:
                report.discrepancies.append(
                    f'dataset {expected.name!r} missing when read back as v{version}'
                )
                continue
            if (got.dataset_label or '') != (expected.dataset_label or ''):
                report.discrepancies.append(
                    f'{ds_key}: dataset label read back as {got.dataset_label!r},'
                    f' expected {expected.dataset_label!r}'
                )
            for name in expected.columns:
                if name not in got.columns:
                    report.discrepancies.append(f'{ds_key}.{name}: missing when read back')
                    continue
                e, g = expected[name], got[name]
                character = e.vtype == xport.VariableType.CHARACTER
                if len(e) != len(g):
                    report.discrepancies.append(
                        f'{ds_key}.{name}: {len(g)} rows read back, expected {len(e)}'
                    )
                    continue
                for i, (x, y) in enumerate(zip(e, g)):
                    if pd.isna(x) and pd.isna(y):
                        continue
                    if character:
                        same = str(x).rstrip() == str(y).rstrip()
                    else:
                        same = float(x) == float(y)
                    if not same:
                        report.discrepancies.append(
                            f'{ds_key}.{name} row {i}: read back {y!r}, expected {x!r}'
                        )
                if (g.label or '') != (e.label or ''):
                    report.discrepancies.append(
                        f'{ds_key}.{name}: label read back as {g.label!r}, expected {e.label!r}'
                    )
                for attribute in ('format', 'informat'):
                    want = getattr(e, attribute)
                    have = getattr(g, attribute)
                    if (str(have) if have else '') != (str(want) if want else ''):
                        report.discrepancies.append(
                            f'{ds_key}.{name}: {attribute} read back as {have}, expected {want}'
                        )

        found = _scan_special_missings(bytestring, version)
        expected_missing = {}
        for (d, fk), tags in self._missings.items():
            if fk in drops.get(d, ()):
                continue
            ds_name = self._truncate(self._dataset_name(d, version), spec.dataset_name_length)
            name = self._truncate(self._field_name(fk, version), spec.name_length)
            expected_missing[(ds_name, name)] = tags
        if found != expected_missing:
            report.discrepancies.append(
                f'special missing values read back as {found}, expected {expected_missing}'
            )
        elif expected_missing:
            n = sum(len(tags) for tags in expected_missing.values())
            report.explanations.append(f'{n} special missing value(s) restored from the side channel')
        return report
