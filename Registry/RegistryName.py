#!/bin/python

#    This file is part of python-registry.
#
#   Copyright 2011, 2012 Willi Ballenthin <william.ballenthin@mandiant.com>
#                    while at Mandiant <http://www.mandiant.com>
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""
Storage-format aware registry names.

A registry name is not just a Unicode string.  The hive records whether the
name was stored in the compressed (ASCII/Windows-1252) form or in the
uncompressed UTF-16LE form, and Windows compares names by upcasing them, not
by lowercasing them.  The two operations are not inverses: ``"\\u212a"``
(KELVIN SIGN) lowercases to ``"k"`` but upcases to itself, while ``"\\u017f"``
(LATIN SMALL LETTER LONG S) upcases to ``"S"`` but lowercases to itself.

This module keeps the exact bytes and the exact decoded text together and
derives a comparison key from the storage format.  It never rewrites the
stored name.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

NAME_KEY = "key"
NAME_VALUE = "value"

FORMAT_ASCII = "ascii"
FORMAT_UTF16 = "utf16"

DEFAULT_DISPLAY_NAME = "(default)"

DEFAULT_VALUE_KEY = (NAME_VALUE, "default")


def normalize_kind(kind):
    """
    Return either NAME_KEY or NAME_VALUE.
    """
    if kind in (NAME_KEY, NAME_VALUE):
        return kind
    if kind in ("keys", "subkey", "subkeys"):
        return NAME_KEY
    if kind in ("values", "value"):
        return NAME_VALUE
    raise ValueError("unknown registry name kind: %r" % (kind,))


def ascii_upcase(text):
    """
    Upcase the ASCII letters A-Z and leave every other code point alone.

    This is the comparison used for names stored in the compressed form.  In
    particular, it does not apply Unicode case mappings to bytes that were
    never ASCII letters.
    """
    out = []
    for ch in text:
        code = ord(ch)
        if 0x61 <= code <= 0x7A:
            out.append(chr(code - 0x20))
        else:
            out.append(ch)
    return "".join(out)


def simple_upcase(text):
    """
    Apply a non-expanding per-code-point upcase.

    Windows' registry upcase table maps one UTF-16 code unit to one UTF-16
    code unit.  Python's ``str.upper`` can expand a character (for example
    ``"\\u00df"`` becomes ``"SS"``), which would silently change the identity
    of a name.  If an upcase would expand, the original code point is kept.
    """
    out = []
    for ch in text:
        up = ch.upper()
        if len(up) == 1:
            out.append(up)
        else:
            out.append(ch)
    return "".join(out)


def upcase(text, is_ascii):
    """
    Upcase `text` according to the format in which the name is stored.
    """
    if is_ascii:
        return ascii_upcase(text)
    return simple_upcase(text)


def decode_name(raw, is_ascii):
    """
    Decode stored name bytes without normalizing or trimming them.
    """
    raw = bytes(raw)
    if is_ascii:
        return raw.decode("windows-1252")
    if len(raw) % 2 != 0:
        raw = raw + b"\x00"
    return raw.decode("utf-16le")


def encode_name(text, is_ascii):
    """
    Encode `text` in the compressed or UTF-16LE form.

    Raises UnicodeEncodeError if the text cannot be represented in the
    requested form.
    """
    if is_ascii:
        return text.encode("windows-1252")
    return text.encode("utf-16le")


def lookup_key(text, kind, is_ascii):
    """
    Build the key Windows uses to find a name at one level.

    Default values do not have a name at all.  They compare as one distinct
    key, regardless of whether the caller spells the default as ``""`` or as
    ``"(default)"``.
    """
    kind = normalize_kind(kind)
    if kind == NAME_VALUE and (text == "" or text == DEFAULT_DISPLAY_NAME):
        return DEFAULT_VALUE_KEY
    return (kind, "name", upcase(text, is_ascii))


def candidate_lookup_keys(text, kind):
    """
    Return the lookup keys a plain string may have in either storage format.

    A plain string carries no storage-format flag.  Callers use this to find
    the stored entries it could refer to rather than assuming a format.
    """
    keys = [lookup_key(text, kind, False), lookup_key(text, kind, True)]
    out = []
    for key in keys:
        if key not in out:
            out.append(key)
    return out


class Name(object):
    """
    An immutable registry name that retains both the decoded text and the
    original stored bytes.

    `raw` is ``None`` for a name supplied by a caller that did not provide
    storage bytes; in that case the name is synthesized when needed and has
    not yet been bound to a stored entry.
    """

    __slots__ = ("text", "raw", "is_ascii", "kind", "present",
                 "format_explicit")

    def __init__(self, text=None, raw=None, is_ascii=None, kind=NAME_KEY,
                 present=True, format_explicit=False):
        self.kind = normalize_kind(kind)

        if not present:
            self.present = False
            self.text = ""
            self.raw = b"" if raw is None else bytes(raw)
            self.is_ascii = True if is_ascii is None else bool(is_ascii)
            self.format_explicit = bool(format_explicit)
            return

        self.present = True

        if raw is not None:
            raw = bytes(raw)
            if is_ascii is None:
                raise ValueError("is_ascii is required when raw name bytes are given")
            self.raw = raw
            self.is_ascii = bool(is_ascii)
            self.text = decode_name(raw, self.is_ascii) if text is None else text
        else:
            if text is None:
                raise ValueError("text or raw name bytes are required")
            self.text = text
            self.is_ascii = bool(is_ascii) if is_ascii is not None else text.isascii()
            self.raw = encode_name(text, self.is_ascii)

        self.format_explicit = bool(format_explicit)

    @classmethod
    def default_value(cls):
        return cls(text="", raw=b"", is_ascii=True, kind=NAME_VALUE,
                   present=False, format_explicit=True)

    @classmethod
    def from_key_record(cls, nkrecord):
        raw = nkrecord.name_bytes()
        return cls(raw=raw, is_ascii=nkrecord.has_ascii_name(), kind=NAME_KEY)

    @classmethod
    def from_value_record(cls, vkrecord):
        if not vkrecord.has_name():
            return cls.default_value()
        raw = vkrecord.name_bytes()
        return cls(raw=raw, is_ascii=vkrecord.has_ascii_name(), kind=NAME_VALUE)

    @classmethod
    def from_entry(cls, entry, kind):
        """
        Coerce a RegistryKey, RegistryValue, Name, or string into a Name.
        """
        kind = normalize_kind(kind)
        if isinstance(entry, Name):
            if entry.kind != kind:
                return cls(text=entry.text, raw=entry.raw,
                           is_ascii=entry.is_ascii, kind=kind,
                           present=entry.present,
                           format_explicit=entry.format_explicit)
            return entry

        if isinstance(entry, bytes):
            return cls(raw=entry, is_ascii=False, kind=kind,
                       present=True, format_explicit=True)

        if isinstance(entry, str):
            if kind == NAME_VALUE and entry in ("", DEFAULT_DISPLAY_NAME):
                return cls.default_value()
            return cls(text=entry, kind=kind)

        # RegistryKey/RegistryValue expose these methods.
        name_bytes = getattr(entry, "name_bytes", None)
        name_is_ascii = getattr(entry, "name_is_ascii", None)
        if callable(name_bytes) and callable(name_is_ascii):
            raw = name_bytes()
            if kind == NAME_VALUE and not raw:
                return cls.default_value()
            return cls(raw=raw, is_ascii=name_is_ascii(), kind=kind)

        raise TypeError("cannot interpret %r as a registry name" % (entry,))

    def lookup_key(self):
        if self.kind == NAME_VALUE and not self.present:
            return DEFAULT_VALUE_KEY
        return lookup_key(self.text, self.kind, self.is_ascii)

    def candidate_lookup_keys(self):
        if self.kind == NAME_VALUE and not self.present:
            return [DEFAULT_VALUE_KEY]
        return candidate_lookup_keys(self.text, self.kind)

    def folded(self):
        if self.kind == NAME_VALUE and not self.present:
            return ""
        return upcase(self.text, self.is_ascii)

    def display(self):
        if self.kind == NAME_VALUE and not self.present:
            return DEFAULT_DISPLAY_NAME
        return self.text

    def is_default(self):
        return self.kind == NAME_VALUE and not self.present

    def with_text(self, text, is_ascii=None, kind=None, format_explicit=True):
        kind = self.kind if kind is None else normalize_kind(kind)
        if kind == NAME_VALUE and text in ("", DEFAULT_DISPLAY_NAME):
            return Name.default_value()
        if is_ascii is None:
            is_ascii = self.is_ascii
        return Name(text=text, is_ascii=is_ascii, kind=kind,
                    format_explicit=format_explicit)

    def sort_key(self):
        return (self.kind, 0 if not self.present else 1,
                1 if self.is_ascii else 0, self.text, self.raw)

    def to_dict(self):
        return {
            "text": self.text,
            "display": self.display(),
            "raw": self.raw.hex(),
            "format": FORMAT_ASCII if self.is_ascii else FORMAT_UTF16,
            "kind": self.kind,
            "present": self.present,
            "lookup": "\\x00".join(part for part in self.lookup_key()),
        }

    def __eq__(self, other):
        if not isinstance(other, Name):
            return NotImplemented
        return (self.text == other.text and self.raw == other.raw and
                self.is_ascii == other.is_ascii and self.kind == other.kind and
                self.present == other.present)

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __hash__(self):
        return hash((self.text, self.raw, self.is_ascii, self.kind,
                     self.present))

    def __repr__(self):
        return "Name(kind=%r, text=%r, raw=%r, is_ascii=%r, present=%r)" % (
            self.kind, self.text, self.raw, self.is_ascii, self.present)


# Short alias, convenient in test code and in callers that build many names.
RegistryName = Name

# Historical spelling used by some callers.
default_value_name = Name.default_value


def query_matches(stored_name, stored_is_ascii, query, kind, present=True):
    """
    Does a user query string match a stored registry name?

    The stored name's own storage format decides how it folds.  The query has
    no storage format of its own, so it is tried in both forms.  The default
    value is matched by both ``""`` and ``"(default)"``, and by nothing else.
    """
    kind = normalize_kind(kind)
    if kind == NAME_VALUE and not present:
        return query in ("", DEFAULT_DISPLAY_NAME)

    if kind == NAME_VALUE and query in ("", DEFAULT_DISPLAY_NAME):
        return False

    stored_key = lookup_key(stored_name, kind, stored_is_ascii)
    for candidate in candidate_lookup_keys(query, kind):
        if stored_key == candidate:
            return True
    return False
