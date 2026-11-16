from __future__ import annotations

from ._argument import Argument, SubWikiTextWithArgs
from ._comment_bold_italic import COMMENT_PATTERN
from ._wikitext import WS, rc

COMMENT_SUB = rc(COMMENT_PATTERN).sub

TL_NAME_ARGS_FULLMATCH = rc(rb'[^|}]*+(?#name)(?<arg>\|[^|]*+)*+').fullmatch


class Template(SubWikiTextWithArgs):
    """Convert strings to Template objects.

    The string should start with {{ and end with }}.
    """

    __slots__ = ()

    _name_args_matcher = TL_NAME_ARGS_FULLMATCH
    _first_arg_sep = 124

    def normal_name(
        self,
        rm_namespaces=('Template',),
        *,
        code: str | None = None,
        capitalize=False,
    ) -> str:
        """Return normal form of self.name.

        - Remove comments.
        - Remove language code.
        - Remove namespace ("template:" or any of `localized_namespaces`.
        - Use space instead of underscore.
        - Remove consecutive spaces.
        - Use uppercase for the first letter if `capitalize`.
        - Remove #anchor.

        :param rm_namespaces: is used to provide additional localized
            namespaces for the template namespace. They will be removed from
            the result. Default is ('Template',).
        :param capitalize: If True, convert the first letter of the
            template's name to a capital letter. See
            [[mw:Manual:$wgCapitalLinks]] for more info.
        :param code: is the language code.

        Example:
            >>> Template(
            ...     '{{ eN : tEmPlAtE : <!-- c --> t_1 # b | a }}'
            ... ).normal_name(code='en')
            'T 1'
        """
        # Remove comments
        name = COMMENT_SUB('', self.name).strip(WS)
        # Remove code
        if code:
            head, sep, tail = name.partition(':')
            if not head and sep:
                name = tail.strip(' ')
                head, sep, tail = name.partition(':')
            if code.lower() == head.strip(' ').lower():
                name = tail.strip(' ')
        # Remove namespace
        head, sep, tail = name.partition(':')
        if not head and sep:
            name = tail.strip(' ')
            head, sep, tail = name.partition(':')
        if head:
            ns = head.strip(' ').lower()
            for namespace in rm_namespaces:
                if namespace.lower() == ns:
                    name = tail.strip(' ')
                    break
        # Use space instead of underscore
        name = name.replace('_', ' ')
        if capitalize:
            # Use uppercase for the first letter
            name = name[:1].upper() + name[1:]
        # Remove #anchor
        name, sep, tail = name.partition('#')
        return ' '.join(name.split())

    def get_arg(self, name: str) -> Argument | None:
        """Return the last argument with the given name.

        Return None if no argument with that name is found.
        """
        return super()._get_arg(name, ignore_equals=False)

    def has_arg(self, name: str, value: str | None = None) -> bool:
        """Return true if there is an arg named `name`.

        Also check equality of values if `value` is provided.

        Note: If you just need to get an argument and you want to LBYL, it's
            better to get_arg directly and then check if the returned value
            is None.
        """
        return super()._has_arg(name, value, ignore_equals=False)

    def set_arg(
        self,
        name: str | None,
        value: str,
        positional: bool | None = None,
        before: str | None = None,
        after: str | None = None,
        preserve_spacing: bool = False,
    ) -> None:
        """Set the value for `name` argument. Add it if it doesn't exist.

        - Use `positional`, `before` and `after` keyword arguments only when
            adding a new argument.
        - If `before` is given, ignore `after`.
        - If neither `before` nor `after` are given and it's needed to add a
            new argument, then append the new argument to the end.
        - If `positional` is True, add the given value as a positional
            argument and ignore `preserve_spacing`.
        - If positional is None (the default), mirror the kind of the last
            existing argument when the name is the next positional index
            (positional if the last arg is positional, keyword otherwise).
        """
        super()._set_arg(
            name,
            value,
            positional,
            before,
            after,
            preserve_spacing,
            ignore_equals=False,
        )

    def del_arg(self, name: str) -> None:
        """Delete all arguments with the given name."""
        super()._del_arg(name, ignore_equals=False)

    def import_arguments(self, changes=()):
        """Return every argument occurrence and its effective value.

        Unlike :attr:`arguments`, this view keeps duplicate occurrences and
        distinguishes an explicit ``|1=a`` from the first positional argument.
        Both address parameter ``1``, but they remain separate records and the
        last occurrence wins.

        Pass ``changes`` (delete/set/insert requests or mappings) to apply an
        edit batch.  Every index refers to the occurrence list of the
        *original* template, so deleting one occurrence never changes what
        another request addresses, and inserting a positional occurrence
        renumbers only the result.  See
        :func:`wikitextparser._argument_import.import_arguments`.
        """
        from ._argument_import import import_arguments

        return import_arguments(self, changes)

    def apply_argument_changes(self, changes):
        """Apply delete/set/insert requests to this template's arguments."""
        from ._argument_import import apply_argument_changes

        return apply_argument_changes(self, changes)

    @property
    def argument_occurrences(self):
        """Return every argument occurrence in source order.

        Duplicate names are kept, and an explicit ``|1=a`` is distinct from
        the first positional argument even though both address parameter 1.
        """
        from ._argument_import import import_arguments

        return import_arguments(self).occurrences_before

    @property
    def effective_arguments(self):
        """Return the last-occurrence-wins mapping of parameter names."""
        from ._argument_import import import_arguments

        return import_arguments(self).effective_before

    @property
    def argument_conflicts(self):
        """Return names whose occurrences do not all have the same value."""
        from ._argument_import import import_arguments

        return import_arguments(self).conflicts_before

    @property
    def templates(self) -> list[Template]:
        return super().templates[1:]
