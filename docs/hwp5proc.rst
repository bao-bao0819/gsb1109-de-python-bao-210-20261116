``hwp5proc``: HWPv5 processor
=============================

.. argparse::
   :module: hwp5.hwp5proc
   :func: main_argparser
   :prog: hwp5proc
   :nosubcommands:

Subcommands
===========

version
-------

Print the file format version of .hwp files.

.. argparse::
   :module: hwp5.hwp5proc
   :func: main_argparser
   :prog: hwp5proc
   :path: version


header
------

Print file headers of .hwp files.

.. argparse::
   :module: hwp5.hwp5proc
   :func: main_argparser
   :prog: hwp5proc
   :path: header


summaryinfo
-----------

Print summary informations of .hwp files.

.. argparse::
   :module: hwp5.hwp5proc
   :func: main_argparser
   :prog: hwp5proc
   :path: summaryinfo


ls
--
List streams in .hwp files.

.. argparse::
   :module: hwp5.hwp5proc
   :func: main_argparser
   :prog: hwp5proc
   :path: ls


cat
---

Extract out internal streams of .hwp files

.. argparse::
   :module: hwp5.hwp5proc
   :func: main_argparser
   :prog: hwp5proc
   :path: cat

Example::

    $ hwp5proc cat samples/sample-5017.hwp BinData/BIN0002.jpg | file -

    $ hwp5proc cat samples/sample-5017.hwp BinData/BIN0002.jpg > BIN0002.jpg

    $ hwp5proc cat samples/sample-5017.hwp PrvText | iconv -f utf-16le -t utf-8

    $ hwp5proc cat --vstreams samples/sample-5017.hwp PrvText.utf8

    $ hwp5proc cat --vstreams samples/sample-5017.hwp FileHeader.txt

    ccl: 0
    cert_drm: 0
    cert_encrypted: 0
    cert_signature_extra: 0
    cert_signed: 0
    compressed: 1
    distributable: 0
    drm: 0
    history: 0
    password: 0
    script: 0
    signature: HWP Document File
    version: 5.0.1.7
    xmltemplate_storage: 0

unpack
------

Extract out internal streams of .hwp files into a directory.

.. argparse::
   :module: hwp5.hwp5proc
   :func: main_argparser
   :prog: hwp5proc
   :path: unpack

Example::

    $ hwp5proc unpack samples/sample-5017.hwp
    $ ls sample-5017

Example::

    $ hwp5proc unpack --vstreams samples/sample-5017.hwp
    $ cat sample-5017/PrvText.utf8

records
-------

Print the record structure of .hwp file record streams.

.. argparse::
   :module: hwp5.hwp5proc
   :func: main_argparser
   :prog: hwp5proc
   :path: records

Example::

    $ hwp5proc records samples/sample-5017.hwp DocInfo

Example::

    $ hwp5proc records samples/sample-5017.hwp DocInfo --range=0-2

If neither <hwp5file> nor <record-stream> is specified, the record stream is
read from the standard input with an assumption that the input is in the format
version specified by -V option.

Example::

    $ hwp5proc records --raw samples/sample-5017.hwp DocInfo --range=0-2 > tmp.rec
    $ hwp5proc records < tmp.rec

models
-------

Print parsed binary models of .hwp file record streams.

.. argparse::
   :module: hwp5.hwp5proc
   :func: main_argparser
   :prog: hwp5proc
   :path: models

Example::

    $ hwp5proc models samples/sample-5017.hwp DocInfo
    $ hwp5proc models samples/sample-5017.hwp BodyText/Section0

    $ hwp5proc models samples/sample-5017.hwp docinfo
    $ hwp5proc models samples/sample-5017.hwp bodytext/0

Example::

    $ hwp5proc models --simple samples/sample-5017.hwp bodytext/0
    $ hwp5proc models --format='%(level)s %(tagname)s\\n' \\
            samples/sample-5017.hwp bodytext/0

Example::

    $ hwp5proc models --simple --treegroup=1 samples/sample-5017.hwp bodytext/0
    $ hwp5proc models --simple --seqno=4 samples/sample-5017.hwp bodytext/0

If neither <hwp5file> nor <record-stream> is specified, the record stream is
read from the standard input with an assumption that the input is in the format
version specified by -V option.

Example::

    $ hwp5proc cat samples/sample-5017.hwp BodyText/Section0 > Section0.bin
    $ hwp5proc models -V 5.0.1.7 < Section0.bin

find
----

Find record models with specified predicates.

.. argparse::
   :module: hwp5.hwp5proc
   :func: main_argparser
   :prog: hwp5proc
   :path: find

Example: Find paragraphs::

    $ hwp5proc find --model=Paragraph samples/*.hwp
    $ hwp5proc find --tag=HWPTAG_PARA_TEXT samples/*.hwp
    $ hwp5proc find --tag=66 samples/*.hwp

Example: Find and dump records of ``HWPTAG_LIST_HEADER`` which is parsed
incompletely::

    $ hwp5proc find --tag=HWPTAG_LIST_HEADER --incomplete --dump samples/*.hwp

xml
---

Transform .hwp files into an XML.

.. argparse::
   :module: hwp5.hwp5proc
   :func: main_argparser
   :prog: hwp5proc
   :path: xml

Example::

    $ hwp5proc xml samples/sample-5017.hwp > sample-5017.xml
    $ xmllint --format sample-5017.xml

With ``--embedbin`` option, you can embed base64-encoded ``BinData/*`` files in
the output XML.

Example::

    $ hwp5proc xml --embedbin samples/sample-5017.hwp > sample-5017.xml
    $ xmllint --format sample-5017.xml

rawunz
------

Deflate an headerless zlib-compressed stream.

.. argparse::
   :module: hwp5.hwp5proc
   :func: main_argparser
   :prog: hwp5proc
   :path: rawunz

diststream
----------

Decode a distribute document stream.

.. argparse::
   :module: hwp5.hwp5proc
   :func: main_argparser
   :prog: hwp5proc
   :path: diststream

attachments
-----------

Export the binary attachment inventory declared by a document.

.. argparse::
   :module: hwp5.hwp5proc
   :func: main_argparser
   :prog: hwp5proc
   :path: attachments

The inventory is built from ``HWPTAG_BIN_DATA`` records in ``DocInfo``.
Each record yields one entry, so two records are never merged even when they
point at the same stream or hold identical bytes.  ``storage_identity``
records the declared identity (stream id and extension, storage id, or the
external file paths) for every entry.

Every body occurrence is kept in ``references``, including references reached
through a paragraph shape or a BorderFill fill image; ``definition_references``
holds DocInfo-level uses such as a BorderFill fill image.  The top-level
``unresolved_references`` list records body references whose ``bindata-id`` has
no declaration.  ``storage_references`` lists OLE storage references, which
point at a storage identity rather than a single attachment.

``link`` entries keep the absolute and relative paths declared by the
document.  A relative path is resolved against the document directory (or
``--base-dir``); it is not searched for under ``BinData``.  A link whose
declared file was not supplied is reported with ``status`` ``pending`` and
``external.retrieval_required`` ``true``.  It is never satisfied from a
different attachment with the same base name.

A ``storage`` entry declares only a storage id.  Its stream is matched by
that identity and a unique extension; an ambiguous match is reported as
``missing`` instead of picked arbitrarily.

``BinData`` streams that no record declares are listed separately under
``unreferenced_streams`` so shared content remains visible without becoming a
declared attachment.  ``shared.storage_with`` and ``shared.content_with``
report records sharing storage or bytes; ``name_collisions`` reports the same
name denoting different storage identities.

Example::

    $ hwp5proc attachments samples/sample-5017.hwp

    $ hwp5proc attachments --format table samples/sample-5017.hwp
    ID   TYPE       STORAGE ID  NAME             STATUS     REFS
    1    embedding  BIN0002     BIN0002.jpg      available  1
    2    embedding  BIN0003     BIN0003.png      available  1
