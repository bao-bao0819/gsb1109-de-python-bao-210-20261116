# -*- coding: utf-8 -*-
"""
https://www.w3.org/TR/shacl/#sparql-constraints
"""

import re

import rdflib
from rdflib import XSD

from ..consts import (
    OWL_PFX,
    RDF_PFX,
    RDFS_PFX,
    SH,
    OWL_Ontology,
    RDF_type,
    SH_namespace,
    SH_prefix,
    SH_prefixes,
)
from ..errors import ConstraintLoadError, ReportableRuntimeError, ValidationFailure
from .path_helper import shacl_path_to_sparql_path

SH_declare = SH.declare
invalid_parameter_names = {'this', 'shapesGraph', 'currentShape', 'path', 'PATH', 'value'}


class SPARQLQueryHelper(object):
    bind_this_regex = re.compile(r"([\s{}()])[\$\?]this", flags=re.M)
    bind_value_regex = re.compile(r"([\s{}()])[\$\?]value", flags=re.M)
    bind_path_regex = re.compile(r"([\s{}()])[\$\?]PATH", flags=re.M)
    bind_sg_regex = re.compile(r"([\s{}()])[\$\?]shapesGraph", flags=re.M)
    bind_cs_regex = re.compile(r"([\s{}()])[\$\?]currentShape", flags=re.M)
    has_minus_regex = re.compile(r"^(?:[^#]*|M)(?!#)#?[^\?\$\#]M?INUS[\s\{]", flags=re.M | re.I)
    # Match keyword VALUES without catching predicate names like ex:allowedValues
    # Supports VALUES(?x), VALUES ?x, and VALUES { ... }
    has_values_regex = re.compile(
        r"^(?!\s*#).*?(?<![\w\-\:])VALUES\b(?:\s*(?:\(|[\?\$]\w+|\{|\w|\s))", flags=re.M | re.I
    )
    has_service_regex = re.compile(r"^(?:[^#]*|S)(?!#)#?[^\?\$\#]S?ERVICE[\s\<]", flags=re.M | re.I)
    has_nested_select_regex = re.compile(
        r"SELECT[\s\(\)\$\?\a-z]*\{[^\}]*SELECT\s+((?:(?:[\?\$]\w+\s+)|(?:\*\s+))+)", flags=re.M | re.I
    )
    has_as_var_regex = re.compile(r"[^\w]+AS[\s]+[\$\?](\w+)", flags=re.M | re.I)
    find_msg_subs = re.compile(r"({[\$\?]([^{}]+)})", flags=re.M)

    @staticmethod
    def _iter_active_spans(sparql_text):
        """
        Yield ``(start, end)`` ranges of ``sparql_text`` that are active query code.

        Regions that are not active query code -- string literals (short and long),
        line comments and full IRI references -- are skipped. This makes every later
        lexical check token-aware, so the same wording appearing inside a constant
        (e.g. the text ``$this`` or the keyword ``VALUES`` inside a string) is never
        mistaken for a SPARQL variable or clause and is never rewritten.
        """
        text_len = len(sparql_text)
        i = 0
        span_start = 0
        while i < text_len:
            ch = sparql_text[i]
            if ch == "<":
                # Distinguish an IRI reference (<...>) from comparison operators such as <= and <.
                # Checked before '#' because an IRI may itself contain a fragment marker (http://x/#f).
                j = i + 1
                while j < text_len and sparql_text[j] != ">" and not sparql_text[j].isspace():
                    j += 1
                if j < text_len and sparql_text[j] == ">":
                    if i > span_start:
                        yield span_start, i
                    i = j + 1
                    span_start = i
                else:
                    i += 1
            elif ch == "#":
                if i > span_start:
                    yield span_start, i
                newline = sparql_text.find("\n", i)
                if newline == -1:
                    return
                i = newline
                span_start = i
            elif ch == '"' or ch == "'":
                if i > span_start:
                    yield span_start, i
                quote = ch
                if sparql_text.startswith(quote * 3, i):
                    closing = sparql_text.find(quote * 3, i + 3)
                    i = text_len if closing == -1 else closing + 3
                else:
                    j = i + 1
                    while j < text_len:
                        inner = sparql_text[j]
                        if inner == "\\":
                            j += 2
                            continue
                        if inner == quote or inner == "\n":
                            j += 1
                            break
                        j += 1
                    i = j
                span_start = i
            else:
                i += 1
        if span_start < text_len:
            yield span_start, text_len

    @classmethod
    def _search_active_spans(cls, regex, sparql_text):
        """
        Run a compiled regex against active query code only, ignoring string/comment/IRI regions.

        Each active span is searched as an extracted substring. Several of the clause-detection
        regexes anchor at ``^`` with ``re.MULTILINE``; an active span can begin mid-line (right
        after an inline string), so searching the full text with a mid-line offset would never
        satisfy the anchor and would silently miss a real clause.
        """
        for start, end in cls._iter_active_spans(sparql_text):
            match = regex.search(sparql_text[start:end])
            if match:
                return match
        return None

    @classmethod
    def variable_is_used(cls, sparql_text, variable_regex):
        """True when the variable matched by ``variable_regex`` occurs outside of constants or comments."""
        return cls._search_active_spans(variable_regex, sparql_text) is not None

    @classmethod
    def _substitute_in_active_spans(cls, sparql_text, token_regex, replacement):
        """
        Apply ``token_regex`` substitution only on active query code.

        Hidden regions (string literals, comments, IRIs) are copied through byte-for-byte,
        so a token spelled identically inside a constant can never be rewritten.
        """
        output = []
        position = 0
        for start, end in cls._iter_active_spans(sparql_text):
            output.append(sparql_text[position:start])
            span = sparql_text[start:end]
            rebuilt = []
            span_position = 0
            for match in token_regex.finditer(span):
                rebuilt.append(span[span_position:match.start()])
                rebuilt.append(match.group(1) + replacement)
                span_position = match.end()
            rebuilt.append(span[span_position:])
            output.append("".join(rebuilt))
            position = end
        output.append(sparql_text[position:])
        return "".join(output)

    def __init__(self, shape, node, select_text, parameters=None, messages=None, deactivated=False):
        self._shape = None
        self.node = node
        self.select_text = select_text
        self.unbound_messages = messages or set()
        self.deactivated = deactivated
        self.parameters = [] if parameters is None else parameters
        self.param_bind_map = {}
        self.bound_messages = set()
        self.prefixes = {
            'rdf': RDF_PFX,
            'rdfs': RDFS_PFX,
            'owl': OWL_PFX,
        }
        if shape:
            self.shape = shape

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, newshape):
        self._shape = newshape
        if len(self.parameters) > 0:
            self.bind_params()
            self.bind_messages()

    @property
    def messages(self):
        if len(self.bound_messages) < 1:
            return self.unbound_messages
        return self.bound_messages

    def bind_params(self):
        bind_map = {}
        shape = self.shape
        for p in self.parameters:
            name = p.localname
            if name in invalid_parameter_names:
                # TODO:coverage: No test for this case
                raise ReportableRuntimeError("Parameter name {} cannot be used.".format(name))
            shape_params = set(shape.objects(p.path()))
            if len(shape_params) < 1:
                if not p.optional:
                    # TODO:coverage: No test for this case
                    raise ReportableRuntimeError("Shape does not have mandatory parameter {}.".format(str(p.path())))
                continue
            # TODO: Can shapes have more than one value for the predicate?
            # Just use one for now.
            # TODO: Check for sh:class and sh:nodeKind on the found param value
            bind_map[name] = next(iter(shape_params))
        self.param_bind_map = bind_map

    def bind_messages(self, param_map=None):
        # must call bind_params _before_ bind_messages
        if param_map is None:
            param_map = self.param_bind_map
        var_replacers = {}
        bound_messages = set()
        for m in self.unbound_messages:
            m_val = str(m.value)
            finds = self.find_msg_subs.findall(m_val)
            if len(finds) < 1:
                bound_messages.add(m)
                continue
            for f in finds:
                variable = f[1]
                if variable not in param_map.keys():
                    continue
                try:
                    replacer = var_replacers[variable]
                except KeyError:
                    replacer = re.compile(r"{[\$\?]" + variable + r"}", flags=re.M)
                    var_replacers[variable] = replacer
                m_val = replacer.sub(str(param_map[variable]), m_val, 1)
            bound_messages.add(rdflib.Literal(m_val, lang=m.language, datatype=m.datatype))
        self.bound_messages = bound_messages

    def collect_prefixes(self):
        sg = self.shape.sg.graph
        prefixes_vals = set(sg.objects(self.node, SH_prefixes))
        if len(prefixes_vals) < 1:
            return
        if isinstance(sg, rdflib.Dataset):
            default_graph = sg.default_graph
            g_name = default_graph.identifier
        else:
            g_name = sg.identifier
        if g_name:
            ng_declares = set(sg.objects(g_name, SH_declare))
        else:
            ng_declares = set()
        onts = set(sg.subjects(RDF_type, OWL_Ontology))
        ont_declares = set()
        for o in onts:
            ont_declares.update(set(sg.objects(o, SH_declare)))
        global_declares = ng_declares.union(ont_declares)

        for prefixes_val in iter(prefixes_vals):
            pfx_declares = set(sg.objects(prefixes_val, SH_declare))
            if pfx_declares and prefixes_val in onts:
                all_declares = pfx_declares.union(ng_declares)
            else:
                all_declares = global_declares.union(pfx_declares)
            for dec in iter(all_declares):
                if isinstance(dec, rdflib.Literal):
                    raise ConstraintLoadError(
                        "sh:declare value must be either a URIRef or a BNode.",
                        "https://www.w3.org/TR/shacl/#sparql-prefixes",
                    )
                prefix_vals = set(sg.objects(dec, SH_prefix))
                if len(prefix_vals) < 1 or len(prefix_vals) > 1:
                    raise ConstraintLoadError(
                        "sh:declare must have exactly one sh:prefix predicate.",
                        "https://www.w3.org/TR/shacl/#sparql-prefixes",
                    )
                prefix = next(iter(prefix_vals))
                if not (isinstance(prefix, rdflib.Literal) and isinstance(prefix.value, str)):
                    raise ConstraintLoadError(
                        "sh:prefix value must be an RDF Literal with type xsd:string.",
                        "https://www.w3.org/TR/shacl/#sparql-prefixes",
                    )
                prefix = str(prefix.value)
                namespace_vals = set(sg.objects(dec, SH_namespace))
                if len(namespace_vals) < 1 or len(namespace_vals) > 1:
                    raise ConstraintLoadError(
                        "sh:declare must have exactly one sh:namespace predicate.",
                        "https://www.w3.org/TR/shacl/#sparql-prefixes",
                    )
                namespace = next(iter(namespace_vals))  # type: rdflib.Literal
                if not (isinstance(namespace, rdflib.Literal) and namespace.datatype == XSD.anyURI):
                    if prefix == "sh" and isinstance(namespace.value, str):
                        # Known bug in shacl.ttl https://github.com/w3c/data-shapes/issues/125
                        pass
                    elif (
                        namespace.datatype == XSD.string
                        or namespace.language is not None
                        or isinstance(namespace.value, str)
                    ):
                        # Its now possible for namespace to be xsd:string or string literal
                        pass
                    else:
                        raise ConstraintLoadError(
                            "sh:namespace value must be an RDF Literal with type xsd:anyURI.\nLiteral: {} type={}".format(
                                namespace.value, namespace.datatype or namespace.language
                            ),
                            "https://www.w3.org/TR/shacl/#sparql-prefixes",
                        )
                namespace = rdflib.URIRef(str(namespace.value))
                self.prefixes[prefix] = namespace

    def apply_prefixes(self, sparql):
        prefix_string = ""
        for p, ns in self.prefixes.items():
            prefix_string += "PREFIX {}: <{}>\n".format(str(p), str(ns))
        return "{}\n{}".format(prefix_string, sparql)

    @classmethod
    def _node_to_sparql_text(cls, node):
        if isinstance(node, rdflib.Literal):
            if isinstance(node.value, str):
                node_text = "\"{}\"".format(node.value)
            else:
                node_text = str(node.value)
            if node.language:
                node_text = "{}@{}".format(node_text, str(node.language))
            elif node.datatype:
                node_text = "{}^^{}".format(node_text, cls._node_to_sparql_text(node.datatype))
            return node_text
        elif isinstance(node, rdflib.URIRef):
            return "<{}>".format(str(node))
        elif isinstance(node, rdflib.BNode):
            # I think this works to convert a BNode to its internal id.
            return str(node)
        elif isinstance(node, str):
            return node
        raise NotImplementedError("Cannot turn that kind of node into text.")

    def check_invalid_sparql(self, sparql_text, valuenode=None, extravars=None):
        has_minus = self._search_active_spans(self.has_minus_regex, sparql_text)
        if has_minus:
            raise ValidationFailure("A SPARQL Constraint must not contain a MINUS clause.")
        has_values = self._search_active_spans(self.has_values_regex, sparql_text)
        if has_values:
            raise ValidationFailure("A SPARQL Constraint must not contain a VALUES clause.")
        has_service = self._search_active_spans(self.has_service_regex, sparql_text)
        if has_service:
            raise ValidationFailure("A SPARQL Constraint must not contain a federated query (SERVICE).")
        potentially_prebound_variables = {'this', 'shapesGraph', 'currentShape'}
        if valuenode is not None:
            potentially_prebound_variables.add('value')
        if extravars is not None and len(extravars) > 0:
            potentially_prebound_variables.update(extravars)
        has_nested_select = self._search_active_spans(self.has_nested_select_regex, sparql_text)
        if has_nested_select:
            var_string = has_nested_select.group(1)
            vars = var_string.split()
            if len(vars) == 0:
                raise ValidationFailure("Ill-formed nested SELECT statement found.")
            stripped_vars = [v.lstrip('$?').rstrip() for v in vars]
            if len(stripped_vars) == 1 and stripped_vars[0] == "*":
                raise ValidationFailure(
                    "Using 'SELECT *' in a nested SELECT query does not select potentially pre-bound variables.\n"
                    "See https://github.com/w3c/data-shapes/issues/84."
                )
            for p in potentially_prebound_variables:
                if p not in stripped_vars:
                    # these are optional:
                    if p == "shapesGraph" or p == "currentShape":
                        continue
                    elif p == "this":
                        raise ValidationFailure(
                            "All potentially pre-bound variables must be selected from a nested SELECT query.\n"
                            "Don't forget to include variable `$this` in your SELECT arguments."
                        )
                    else:
                        raise ValidationFailure(
                            "All potentially pre-bound variables must be selected from a nested SELECT query.\n"
                            "Potentially pre-bound variables for this query are: {}.".format(
                                ", ".join(potentially_prebound_variables)
                            )
                        )
        has_as_var = self._search_active_spans(self.has_as_var_regex, sparql_text)
        if has_as_var:
            var_name = has_as_var.group(1)
            if var_name in potentially_prebound_variables:
                raise ValidationFailure(
                    "Cannot use AS to re-bind potentially pre-bound variables such as {}".format(var_name)
                )
        return True

    def pre_bind_variables(self, thisnode, valuenode=None, extravars=None):
        new_query_text = "" + self.select_text
        _ = self.check_invalid_sparql(new_query_text, valuenode=valuenode, extravars=extravars)
        init_bindings = {}
        found_this = self.variable_is_used(new_query_text, self.bind_this_regex)
        if found_this:
            init_bindings['this'] = thisnode

        if valuenode:
            found_value = self.variable_is_used(new_query_text, self.bind_value_regex)
            if found_value:
                init_bindings['value'] = valuenode

        found_cs = self.variable_is_used(new_query_text, self.bind_cs_regex)
        if found_cs:
            init_bindings['currentShape'] = self.shape.node
        path = self.shape.path()
        if path:
            path_string = shacl_path_to_sparql_path(self.shape.sg, path, prefixes=self.prefixes)
            new_query_text = self._substitute_in_active_spans(new_query_text, self.bind_path_regex, path_string)
        else:
            found_path = self.variable_is_used(new_query_text, self.bind_path_regex)
            if found_path:
                raise ReportableRuntimeError(
                    "SPARQL Constraint text has $PATH in it, but no path is known on this Shape."
                )
        # TODO: work out how to get shapesGraph binding from shape.sg
        #  shapes_graph = self.shape.sg
        shapes_graph = False
        if shapes_graph:
            found_sg = self.variable_is_used(new_query_text, self.bind_sg_regex)
            if found_sg:
                init_bindings['shapesGraph'] = shapes_graph
        else:
            found_sg = self.variable_is_used(new_query_text, self.bind_sg_regex)
            if found_sg:
                raise NotImplementedError(
                    "SPARQL Constraint text has $shapesGraph in it, but Shapes Graph is not currently supported."
                )

        return init_bindings, new_query_text
