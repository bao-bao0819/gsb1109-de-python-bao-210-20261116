SCHEMA = """

CREATE TABLE features (
    id text,
    seqid text,
    source text,
    featuretype text,
    start int,
    end int,
    score text,
    strand text,
    frame text,
    attributes text,
    extra text,
    bin int,
    primary key (id)
    );

CREATE TABLE relations (
    parent text,
    child text,
    level int,
    primary key (parent, child, level)
    );

CREATE TABLE meta (
    dialect text,
    version text
    );

CREATE TABLE directives (
    directive text
    );

CREATE TABLE autoincrements (
    base text,
    n int,
    primary key (base)
    );

CREATE TABLE duplicates (
    idspecid text,
    newid text,
    primary key (newid)
    );


"""

default_pragmas = {
    "synchronous": "NORMAL",
    "journal_mode": "MEMORY",
    "main.page_size": 4096,
    "main.cache_size": 10000,
}

_keys = [
    "id",
    "seqid",
    "source",
    "featuretype",
    "start",
    "end",
    "score",
    "strand",
    "frame",
    "attributes",
    "extra",
    "bin",
]
_gffkeys = [
    "seqid",
    "source",
    "featuretype",
    "start",
    "end",
    "score",
    "strand",
    "frame",
    "attributes",
]
_gffkeys_extra = _gffkeys + ["extra"]

_SELECT = "SELECT " + ", ".join(_keys) + ", features.rowid as file_order FROM features "

_INSERT = (
    "INSERT INTO features ("
    + ", ".join(_keys)
    + ") VALUES ("
    + ",".join(list("?" * len(_keys)))
    + ")"
)


_update_clause = ",".join(["%s = ?" % i for i in _keys])
_UPDATE = "UPDATE features SET " + _update_clause + " WHERE id = ?"


# TODO: create indexes once profiling figures out which ones work best....
INDEXES = []


# This dictionary keeps track of idiosyncracies to [attempt to] maintain
# invariance of file->db->file round trips.
dialect = {
    # Initial semicolon, e.g.,
    #
    #   ;ID=001;
    # vs
    #   ID=001;
    "leading semicolon": False,
    # Semicolon after the last value, e.g.,
    #
    #   ID=001; Name=gene1;
    # vs
    #   ID=001; Name=gene1
    "trailing semicolon": False,
    # e.g.,
    #
    #   gene_id "GENE1"
    # vs
    #   gene_id GENE1
    #
    # This is the dialect-wide default; see "quoted value markers" for how
    # per-attribute quoting is tracked so that files mixing quoted and
    # unquoted values round-trip.
    "quoted GFF2 values": False,
    # Per-attribute record of how values were quoted in the original file.
    # None means the file was consistent (or a dialect was supplied), so the
    # boolean "quoted GFF2 values" alone is used. Otherwise this is a dict of
    # attribute key -> marker, where the marker is '"' when that key's values
    # were double-quoted and None when they were not. For repeated keys whose
    # occurrences quote inconsistently, the marker is a list aligned with the
    # occurrences. Keeping this per key (rather than guessing one rule from the
    # first value) is what allows mixed-quoting files to be reconstructed
    # faithfully.
    "quoted value markers": None,
    # Sometimes there's extra space surrounding the semicolon, e.g.,
    #
    #   ID=001;Name=gene1
    # vs
    #   ID=001; Name=gene1
    "field separator": ";",
    # Sometimes there are semicolons inside quotes that break things, e.g.,
    #
    #   note "Evidence 1a: Function1, Function2"
    # vs
    #   note "Evidence 1a: Function; PubMedId: 123, 456"
    "semicolon in quotes": False,
    # Usually "=" for GFF3; " " for GTF, e.g.,
    #
    #   gene_id "GENE1"
    # vs
    #   gene_id="GENE1"
    "keyval separator": "=",
    # Usually a comma, e.g.,
    #
    #   Parent=gene1,gene2,gene3
    "multival separator": ",",
    # General GTF or GFF format
    "fmt": "gff3",
    # How multiple values for the same key are handled, e.g.,
    #
    #   Parent=gene1; Parent=gene2;
    # vs
    #   Parent=gene1,gene2;
    #
    # (the first one has repeated keys)
    "repeated keys": False,
    # If these keys exist, then print them in this order.
    "order": ["ID", "Name", "gene_id", "transcript_id"],
}

always_return_list = True
ignore_url_escape_characters = False

# these keyword args are used by iterators.
_iterator_kwargs = (
    "data",
    "checklines",
    "transform",
    "force_dialect_check",
    "dialect",
    "from_string",
)
