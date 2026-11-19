Worksheet identity and references
=================================

Titles are not identities
-------------------------

A worksheet can be renamed, and its old title can then be reused by a new
worksheet. Cross-sheet formulas and developer metadata can keep pointing at
the old object. The identity ledger records stable identities that do not
depend on titles.

.. code:: python

   import gspread

   sh = gc.open_by_key("...")

   # First call bootstraps the ledger from the current worksheets.
   ledger = sh.identity_ledger()

   for identity in ledger.identities:
       print(identity.uid, identity.title, identity.published_at)

A UID has the form ``worksheet:<spreadsheet id>:<sheet id>``. The first
observation of an identity is its immutable ``published_at`` time. Pass
``published_at`` when the true publication time is known:

.. code:: python

   ledger = sh.identity_ledger(published_at=sh.creationTime)

Publishing snapshots
--------------------

Collect and publish a snapshot (titles, developer metadata and formulas) to
detect what changed since the previous publication:

.. code:: python

   result = sh.reconcile_identity()

   for change in result.changes:
       print(change.uid, change.kind, change.previous_title, change.title)

   for change in result.renamed:
       print("renamed:", change.uid)

The possible change kinds are ``new``, ``unchanged``, ``renamed``, ``copied``,
``recreated`` and ``deleted``.

Reconciliation rules
--------------------

Only structural evidence links identities:

- the same ``sheetId`` means the same identity (a rename when the title
  changed);
- an explicit duplicate event with its source sheet links the copy to its
  parent;
- an explicit deletion followed by a worksheet under the old title marks the
  new worksheet as a recreation of the retired one.

Titles, developer metadata and other content that merely looks similar are
reported as ``candidates`` on the change. They never transfer an identity. A
worksheet copied with the same title and the same business metadata therefore
gets its own new identity.

Deleting or duplicating through gspread records the structural event
automatically when a ledger is attached:

.. code:: python

   copied = sheet.duplicate(new_sheet_name="Data copy")
   sh.del_worksheet(sheet)

Historical queries
------------------

Snapshots are immutable. A historical query returns the titles and references
that were valid at that time, never the current title:

.. code:: python

   graph = sh.identity_graph(at="2026-01-01T00:00:00Z")

   for source in graph.sources:
       print(source.uid, source.title, source.revision)

   for edge in graph.edges:
       print(edge.source_uid, edge.cell, edge.target_uid, edge.target_title)

An edge records the target identity and revision that were resolved when the
formula was observed, along with the title as written. Reading the graph later
does not rewrite old edges.

If a sheet ID is deleted and later reused, the ledger distinguishes the
generations with UIDs such as ``worksheet:<spreadsheet id>:<sheet id>#2``.
Explicit sheet-ID references resolve to the generation that was live at the
requested time.

Timestamps must move forward. Publishing a snapshot earlier than the ledger's
last known observation raises
:class:`~gspread.exceptions.IdentityTimelineError`. Events recorded
automatically by gspread are stamped at the latest known observation, so they
never move the timeline into the future.

Stable references
-----------------

Formula references are extracted statically from A1 notation. ``INDIRECT()``
and text inside string literals are ignored.

A formula reference contains only a title. It is resolved while exactly one
worksheet owns that title, and it becomes ``ambiguous`` when the title has
changed hands, instead of guessing.

To make a reference survive renames, store it in the reserved
``gspread.reference`` developer metadata namespace with an explicit target UID
or sheet ID:

.. code:: json

   {
     "metadataKey": "gspread.reference",
     "metadataValue": "{\"targetUid\": \"worksheet:SS:1\", \"targetTitle\": \"Data\"}",
     "location": {"sheetId": 2}
   }

Explicit targets are resolved through the identity ledger, so they continue to
point at the renamed object. Other developer metadata keys are business
content and are never interpreted as references.

Persistence
-----------

The ledger lives with the :class:`~gspread.spreadsheet.Spreadsheet` object.
Persist it yourself between runs:

.. code:: python

   from gspread import IdentityLedger

   with open("ledger.json", "w") as fp:
       fp.write(sh.identity_ledger().dumps())

   ledger = IdentityLedger.loads(open("ledger.json").read())
   sh.attach_identity_ledger(ledger)
