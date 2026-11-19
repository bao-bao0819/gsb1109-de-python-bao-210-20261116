.. _dialects:

Dialects
========
:mod:`gffutils` borrows the idea of "dialects" from Python's :mod:`csv` module.
In this context, a dialect is a dictionary that specifies details about the
format. For GFF and GTF files, most variation in formatting occurs in the
attributes field (column 9), so currently :mod:`gffutils` dialects only
describe details in how the attributes are formatted.

The advantage to using dialects is that the lines that are imported into
a database can be exactly reconstructed after being retrieved from the
database.

While the `GFF3 specification <http://www.sequenceontology.org/gff3.shtml>`_
and `GTF specification <http://mblab.wustl.edu/GTF22.html>`_ documents make
recommendations about how files should be formatted, in practice there is wide
variation.  :mod:`gffutils` tries to accomodate real-world GFF and GTF files by
examining the first N lines of a file and figuring out what "dialect" the file
is using (N=10 by default, but can be changed with the `checklines` kwarg to
:func:`gffutils.create_db`).

The dialect is simply a dictionary of various attributes that have been
empirically found to differ among real-world files.   You can find the current
version of the dialect dictionary, which is used by default and mimics the GFF3
format, in :attr:`gffutils.constants.dialect`.

For example, some files have a trailing semicolon after the last attribute in
column 9.  In this case, the dialect would specify `dialect['trailing
semicolon'] = True`.

A GTF dialect might look like this::

    {'field separator': '; ',
     'fmt': 'gtf',
     'keyval separator': ' ',
     'leading semicolon': False,
     'multival separator': ',',
     'quoted GFF2 values': True,
     'repeated keys': False,
     'trailing semicolon': True,
     'semicolon_in_quotes': False}

In contrast, a GFF dialect might look like this::

    {'field separator': ';',
     'fmt': 'gff3',
     'keyval separator': '=',
     'leading semicolon': False,
     'multival separator': ',',
     'quoted GFF2 values': False,
     'repeated keys': False,
     'trailing semicolon': False,
     'semicolon_in_quotes': False}


As other real-world files are brought to the attention of the developers, it's
likely that more entries will be added to the dialect.

Most dialect entries describe a single, file-wide convention. The
``quoted GFF2 values`` entry says whether values are generally double-quoted,
but real files sometimes mix quoted and unquoted values (even within one
feature, such as ``name "x"; transcriptId 873``). Rather than letting the first
value's convention be applied to the whole file, the boolean
``quoted GFF2 values`` is supplemented by ``quoted value markers``. When the
quoting is consistent this is ``None``; otherwise it maps each attribute key to
``'"'`` (that key's values were quoted) or ``None`` (they were not). For a
repeated key whose own occurrences disagree, the value is an ordered list of
those markers. These markers, together with the fact that percent-decoding
(``%2C`` and friends) happens only after all structural splitting, are what keep
attribute parsing round-trippable.
