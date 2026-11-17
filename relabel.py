'''
Relabeling support for ARFF datasets whose nominal label vocabulary changed.

The old vocabulary collapsed several fine-grained classes into a single
``Other`` label while the new vocabulary splits them again.  This module keeps
the possible new labels of every historical sample as a candidate domain and
narrows those domains using:

- the old-label -> new-label candidate mapping,
- human-confirmed samples,
- per-partition quotas controlling how many members of each partition may
  carry a given new label.

A sample that occurs in several partitions owns a single domain: the same
explanation is shared by every occurrence instead of assembling one tag set
per partition.  Only samples whose domain is reduced to a single label are
migrated automatically; everything else keeps its full candidate set for
manual review.  When the controls are mutually inconsistent a
``RelabelConflict`` is raised carrying the offending partition/class
conditions.

The module is pure Python and reuses :mod:`arff` for the file format.
'''

from collections import defaultdict, OrderedDict
import argparse
import sys

import arff


# EXCEPTIONS ==================================================================
class RelabelConflict(Exception):
    '''Raised when the candidate sets and the partition controls have no
    consistent assignment.

    ``conflicts`` is a list of structured diagnostics.  Each item is a dict
    whose ``kind`` entry is one of ``confirmed_not_candidate``,
    ``empty_domain``, ``quota_violation``, ``hall_violation`` or
    ``global_infeasible``; the remaining entries describe the samples,
    partitions and classes involved.
    '''

    def __init__(self, conflicts):
        self.conflicts = conflicts
        super(RelabelConflict, self).__init__(
            '%d conflicting control condition(s)' % len(conflicts))


# CORE SOLVER =================================================================
def _capacitated_matching(variables, domains, capacities):
    '''Find an assignment of every variable to one class in its domain.

    Each class accepts at most ``capacities[class]`` variables; classes absent
    from ``capacities`` are treated as unlimited.  Returns a tuple
    ``(assignment, witness)``.  ``assignment`` is a ``{variable: class}`` dict
    on success or ``None`` on failure, in which case ``witness`` is a Hall-set
    diagnosis with the variables that cannot be placed, the classes reachable
    from them, their demand and the total reachable capacity.
    '''
    variables = list(variables)
    unlimited = len(variables)
    classes = sorted({c for s in variables for c in domains[s]})
    cap = {c: capacities.get(c, unlimited) for c in classes}
    assignment = {}
    class_vars = defaultdict(list)

    def augment(var, seen_classes, seen_vars):
        for c in sorted(domains[var]):
            if c in seen_classes:
                continue
            seen_classes.add(c)
            if len(class_vars[c]) < cap[c]:
                assignment[var] = c
                class_vars[c].append(var)
                return True
            for other in list(class_vars[c]):
                if other in seen_vars:
                    continue
                seen_vars.add(other)
                if augment(other, seen_classes, seen_vars):
                    assignment[var] = c
                    class_vars[c].remove(other)
                    class_vars[c].append(var)
                    return True
        return False

    for s in variables:
        if not augment(s, set(), {s}):
            return None, _hall_witness(variables, domains, assignment,
                                       class_vars, cap)

    return assignment, None


def _hall_witness(variables, domains, assignment, class_vars, cap):
    '''Collect the Hall set behind a failed capacitated matching.'''
    matched_vars = set(assignment)
    reachable_vars = {s for s in variables if s not in matched_vars}
    reachable_classes = set()
    stack = list(reachable_vars)
    while stack:
        s = stack.pop()
        for c in domains[s]:
            if c in reachable_classes:
                continue
            reachable_classes.add(c)
            for other in class_vars.get(c, ()):  # matched class -> its vars
                if other not in reachable_vars:
                    reachable_vars.add(other)
                    stack.append(other)

    return {
        'kind': 'hall_violation',
        'variables': sorted(reachable_vars),
        'classes': sorted(reachable_classes),
        'demand': len(reachable_vars),
        'capacity': sum(cap[c] for c in reachable_classes),
        'class_capacities': {c: cap[c] for c in sorted(reachable_classes)},
    }


def _is_supported(sample, label, members, domains, limits):
    '''Return True when ``sample`` can take ``label`` in a full assignment of
    its partition while respecting every quota.'''
    if label not in domains[sample]:
        return False
    unlimited = len(members)
    if limits.get(label, unlimited) <= 0:
        return False
    capacities = dict(limits)
    capacities[label] = limits.get(label, unlimited) - 1
    others = [s for s in members if s != sample]
    assignment, _ = _capacitated_matching(others, domains, capacities)
    return assignment is not None


def _global_feasible(samples, domains, partitions, partition_of, budget,
                     assume=None):
    '''Backtracking search for one assignment satisfying every partition.

    When ``assume`` is a ``(sample, label)`` pair the search additionally
    forces that binding, which decides whether the label has global support.
    Returns ``True`` for a proven solution, ``False`` for proven infeasibility
    and ``None`` when the shared node ``budget`` is exhausted.
    '''
    remaining = set(samples)
    counters = {}
    unlimited = len(samples)

    def allowed(sample, label):
        for pid in partition_of[sample]:
            used = counters.get((pid, label), 0)
            limit = partitions[pid]['limits'].get(label, unlimited)
            if used + 1 > limit:
                return False
        return True

    def bind(sample, label):
        touched = []
        for pid in partition_of[sample]:
            key = (pid, label)
            counters[key] = counters.get(key, 0) + 1
            touched.append(key)
        remaining.remove(sample)
        return touched

    def unbind(sample, label, touched):
        remaining.add(sample)
        for key in touched:
            counters[key] -= 1

    def dfs():
        if not remaining:
            return True
        budget[0] -= 1
        if budget[0] < 0:
            return None
        sample = min(remaining, key=lambda s: (len(domains[s]), s))
        for label in sorted(domains[sample]):
            if not allowed(sample, label):
                continue
            touched = bind(sample, label)
            outcome = dfs()
            unbind(sample, label, touched)
            if outcome is True:
                return True
            if outcome is None:
                return None
        return False

    if assume is not None:
        sample, label = assume
        if label not in domains[sample] or not allowed(sample, label):
            return False
        touched = bind(sample, label)
        outcome = dfs()
        unbind(sample, label, touched)
        return outcome

    return dfs()


def solve(candidate_sets, old_labels, confirmed, partitions,
          search_budget=200000):
    '''Narrow every historical sample to its still-possible new labels.

    :param candidate_sets: ``{old_label: [new_label, ...]}`` mapping.
    :param old_labels: ``{sample_id: old_label}`` original encoding.
    :param confirmed: ``{sample_id: new_label}`` human-verified samples.
    :param partitions: ``{partition_id: {'members': [sample_id, ...],
        'limits': {new_label: max_count}}}`` control numbers; a missing class
        limit means unlimited.
    :param search_budget: node cap for the global feasibility search.
    :return: a result dict with shared ``domains``, the ``auto`` migration
        map (uniquely provable samples only), per-sample ``status`` and
        feasibility metadata.
    :raises RelabelConflict: when no assignment can satisfy the controls.
    '''
    candidate_sets = {k: list(v) for k, v in candidate_sets.items()}
    confirmed = dict(confirmed)
    partitions = {pid: {'members': list(p['members']),
                        'limits': dict(p['limits'])}
                  for pid, p in partitions.items()}

    samples = list(old_labels)
    conflicts = []

    partition_of = defaultdict(list)
    for pid, p in partitions.items():
        for s in p['members']:
            if s not in old_labels:
                raise ValueError(
                    'partition %r references unknown sample %r' % (pid, s))
            partition_of[s].append(pid)

    vocabulary = set()
    for labels in candidate_sets.values():
        vocabulary.update(labels)
    vocabulary.update(confirmed.values())
    for p in partitions.values():
        unknown = set(p['limits']) - vocabulary
        if unknown:
            raise ValueError(
                'limits reference unknown new labels: %s'
                % ', '.join(sorted(unknown)))

    domains = {}
    for s in samples:
        old = old_labels[s]
        if old not in candidate_sets:
            raise ValueError(
                'sample %r has old label %r without a candidate mapping'
                % (s, old))
        domains[s] = set(candidate_sets[old])

    for s, label in confirmed.items():
        if s not in old_labels:
            raise ValueError('confirmed sample %r is not in history' % s)
        if label not in domains[s]:
            conflicts.append({
                'kind': 'confirmed_not_candidate',
                'sample': s,
                'old_label': old_labels[s],
                'confirmed_label': label,
                'candidates': sorted(domains[s]),
            })
        else:
            domains[s] = {label}

    if conflicts:
        raise RelabelConflict(conflicts)

    # Fixpoint: per-partition capacitated arc consistency over the shared
    # domains.  A value pruned in one partition disappears everywhere, so a
    # sample spanning several partitions always keeps a single explanation.
    while True:
        changed = False
        for pid, p in partitions.items():
            members = p['members']
            limits = p['limits']
            unlimited = len(members)

            forced = defaultdict(list)
            for s in members:
                if len(domains[s]) == 1:
                    forced[next(iter(domains[s]))].append(s)
            for label, forcing in sorted(forced.items()):
                limit = limits.get(label, unlimited)
                if len(forcing) > limit:
                    conflicts.append({
                        'kind': 'quota_violation',
                        'partition': pid,
                        'new_label': label,
                        'limit': limit,
                        'forced_demand': len(forcing),
                        'forcing_samples': sorted(forcing),
                    })

            assignment, witness = _capacitated_matching(
                members, domains, limits)
            if assignment is None:
                witness['partition'] = pid
                witness['limits'] = {
                    c: limits.get(c, unlimited)
                    for c in witness['classes']}
                conflicts.append(witness)

            for s in members:
                for label in sorted(domains[s]):
                    if not _is_supported(s, label, members, domains, limits):
                        domains[s].remove(label)
                        changed = True
                if not domains[s]:
                    conflicts.append({
                        'kind': 'empty_domain',
                        'sample': s,
                        'partitions': list(partition_of[s]),
                        'old_label': old_labels[s],
                    })

        if conflicts:
            raise RelabelConflict(conflicts)
        if not changed:
            break

    # Exact global support.  Local arc consistency cannot see coupling
    # between partitions through a shared sample, so each surviving value is
    # re-checked against a full assignment of every shared domain.  Dropping
    # an unsupported value can invalidate another one, hence the fixpoint.
    budget = [search_budget]
    exhausted = False
    supported = set()
    while True:
        changed = False
        for sample in samples:
            for label in sorted(domains[sample]):
                outcome = _global_feasible(
                    samples, domains, partitions, partition_of, budget,
                    assume=(sample, label))
                if outcome is False:
                    domains[sample].remove(label)
                    supported.discard((sample, label))
                    changed = True
                elif outcome is True:
                    supported.add((sample, label))
                else:
                    exhausted = True
            if not domains[sample]:
                conflicts.append({
                    'kind': 'empty_domain',
                    'sample': sample,
                    'partitions': list(partition_of[sample]),
                    'old_label': old_labels[sample],
                })
        if conflicts:
            raise RelabelConflict(conflicts)
        if not changed:
            break

    if exhausted:
        certified = _global_feasible(
            samples, domains, partitions, partition_of, budget)
        if certified is False:
            conflicts.append({
                'kind': 'global_infeasible',
                'partitions': sorted(partitions),
            })
            raise RelabelConflict(conflicts)
        feasibility_certified = certified is True
    else:
        # Every surviving value participates in at least one complete
        # solution, which also proves the instance as a whole is feasible.
        feasibility_certified = True

    auto = {}
    status = {}
    for s in samples:
        if len(domains[s]) == 1:
            label = next(iter(domains[s]))
            # Only migrate when the singleton is proven; an unproven survivor
            # of a budget-exhausted search stays in the manual review queue.
            if (s, label) in supported:
                auto[s] = label
                status[s] = 'confirmed' if s in confirmed else 'auto'
            else:
                status[s] = 'ambiguous'
        else:
            status[s] = 'ambiguous'

    rank = {label: i for i, label in enumerate(vocabulary)}
    ordered_domains = {
        s: frozenset(domains[s]) for s in samples
    }
    return {
        'domains': ordered_domains,
        'ordered_candidates': {
            s: sorted(domains[s], key=lambda c: rank.get(c, len(rank)))
            for s in samples
        },
        'auto': auto,
        'status': status,
        'confirmed': dict(confirmed),
        'old_labels': dict(old_labels),
        'partitions_of': {s: list(partition_of[s]) for s in samples},
        'feasibility_certified': feasibility_certified,
        'search_budget_exhausted': exhausted,
        'vocabulary': sorted(vocabulary, key=lambda c: rank.get(c, len(rank))),
    }


# ARFF I/O ====================================================================
def _row_dict(attributes, row):
    return {name: value for (name, _), value in zip(attributes, row)}


def _load_objects(fp):
    obj = arff.load(fp)
    return obj['attributes'], [
        _row_dict(obj['attributes'], row) for row in obj['data']
    ]


def load_arff_inputs(history_fp, candidates_fp, confirmed_fp, controls_fp):
    '''Read the four ARFF inputs and build the arguments for :func:`solve`.

    History rows are ``(sample_id, old_label, partition)`` with one row per
    sample/partition occurrence; candidates rows are ``(old_label,
    new_label)``; confirmed rows are ``(sample_id, new_label)``; control rows
    are ``(partition, new_label, max_count)``.
    '''
    _, history_rows = _load_objects(history_fp)
    _, candidate_rows = _load_objects(candidates_fp)
    _, confirmed_rows = _load_objects(confirmed_fp)
    _, control_rows = _load_objects(controls_fp)

    old_labels = OrderedDict()
    partitions = OrderedDict()
    for row in history_rows:
        sid = row['sample_id']
        if sid in old_labels and old_labels[sid] != row['old_label']:
            raise ValueError(
                'sample %r carries conflicting old labels' % sid)
        old_labels.setdefault(sid, row['old_label'])
        pid = row['partition']
        partition = partitions.setdefault(
            pid, {'members': [], 'limits': {}})
        if sid not in partition['members']:
            partition['members'].append(sid)

    candidate_sets = OrderedDict()
    for row in candidate_rows:
        candidate_sets.setdefault(row['old_label'], [])
        if row['new_label'] not in candidate_sets[row['old_label']]:
            candidate_sets[row['old_label']].append(row['new_label'])

    confirmed = OrderedDict()
    for row in confirmed_rows:
        sid = row['sample_id']
        if sid in confirmed and confirmed[sid] != row['new_label']:
            raise ValueError(
                'sample %r has two confirmed labels' % sid)
        confirmed[sid] = row['new_label']

    for row in control_rows:
        pid = row['partition']
        partition = partitions.setdefault(
            pid, {'members': [], 'limits': {}})
        label = row['new_label']
        limit = int(row['max_count'])
        if label in partition['limits']:
            raise ValueError(
                'duplicate limit for (%r, %r)' % (pid, label))
        partition['limits'][label] = limit

    return candidate_sets, old_labels, confirmed, partitions


def result_to_arff(result):
    '''Serialize a solver result, keeping original and inferred labels apart.

    One row is emitted per sample so samples spanning several partitions keep
    a single shared explanation.  ``original_label`` holds the old encoding,
    ``auto_label`` is ``?`` unless the sample is uniquely provable, and
    ``candidate_labels`` lists every still-possible new label.
    '''
    data = []
    for sid, old_label in result['old_labels'].items():
        candidates = ' '.join(result['ordered_candidates'][sid])
        auto_label = result['auto'].get(sid)
        data.append([
            sid,
            old_label,
            ' '.join(result['partitions_of'][sid]),
            candidates,
            auto_label,
            result['status'][sid],
        ])

    return {
        'relation': 'relabel_result',
        'attributes': [
            ('sample_id', 'STRING'),
            ('original_label', 'STRING'),
            ('partitions', 'STRING'),
            ('candidate_labels', 'STRING'),
            ('auto_label', 'STRING'),
            ('status', ['auto', 'ambiguous', 'confirmed']),
        ],
        'data': data,
    }


def write_conflict_report(conflicts, fp):
    '''Write a human-readable report of mutually conflicting controls.'''
    lines = ['Relabeling conflict report',
             '==========================',
             'No assignment satisfies the following conditions:',
             '']
    for index, conflict in enumerate(conflicts, 1):
        kind = conflict['kind']
        if kind == 'confirmed_not_candidate':
            lines.append(
                '%d. confirmed label outside candidate set' % index)
            lines.append('   sample: %s (old label %s)'
                         % (conflict['sample'], conflict['old_label']))
            lines.append('   confirmed: %s, candidates: %s'
                         % (conflict['confirmed_label'],
                            ', '.join(conflict['candidates'])))
        elif kind == 'empty_domain':
            lines.append('%d. no candidate label survives the controls'
                         % index)
            lines.append('   sample: %s, partitions: %s'
                         % (conflict['sample'],
                            ', '.join(conflict['partitions'])))
        elif kind == 'quota_violation':
            lines.append('%d. partition quota exceeded by forced samples'
                         % index)
            lines.append('   partition: %s, new label: %s, limit: %d, '
                         'forced demand: %d'
                         % (conflict['partition'], conflict['new_label'],
                            conflict['limit'], conflict['forced_demand']))
            lines.append('   forcing samples: %s'
                         % ', '.join(conflict['forcing_samples']))
        elif kind == 'hall_violation':
            lines.append('%d. partition capacity below candidate demand'
                         % index)
            lines.append('   partition: %s' % conflict['partition'])
            lines.append('   trapped samples: %s'
                         % ', '.join(conflict['variables']))
            lines.append('   reachable labels: %s'
                         % ', '.join(conflict['classes']))
            lines.append('   demand %d exceeds capacity %d (%s)'
                         % (conflict['demand'], conflict['capacity'],
                            ', '.join('%s=%d' % (c, k) for c, k in
                                      conflict['class_capacities'].items())))
        elif kind == 'global_infeasible':
            lines.append('%d. partitions cannot share one explanation'
                         % index)
            lines.append('   partitions: %s'
                         % ', '.join(conflict['partitions']))
        lines.append('')
    fp.write('\n'.join(lines))


# COMMAND LINE ================================================================
def build_parser():
    parser = argparse.ArgumentParser(
        description='Narrow old ARFF labels to the new vocabulary and '
                    'auto-migrate only uniquely provable samples.')
    parser.add_argument('--history', required=True,
                        help='ARFF with sample_id, old_label, partition rows')
    parser.add_argument('--candidates', required=True,
                        help='ARFF with old_label, new_label candidate rows')
    parser.add_argument('--confirmed', required=True,
                        help='ARFF with sample_id, new_label confirmed rows')
    parser.add_argument('--controls', required=True,
                        help='ARFF with partition, new_label, max_count rows')
    parser.add_argument('--out', required=True,
                        help='output ARFF for the shared per-sample result')
    parser.add_argument('--report',
                        help='optional conflict report path')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    with open(args.history) as history_fp, \
            open(args.candidates) as candidates_fp, \
            open(args.confirmed) as confirmed_fp, \
            open(args.controls) as controls_fp:
        inputs = load_arff_inputs(
            history_fp, candidates_fp, confirmed_fp, controls_fp)

    try:
        result = solve(*inputs)
    except RelabelConflict as exc:
        if args.report:
            with open(args.report, 'w') as fp:
                write_conflict_report(exc.conflicts, fp)
        sys.stderr.write(str(exc) + '\n')
        return 2

    with open(args.out, 'w') as fp:
        arff.dump(result_to_arff(result), fp)
    migrated = sum(1 for s in result['status']
                   if result['status'][s] != 'ambiguous')
    sys.stdout.write(
        '%d samples, %d uniquely resolved, %d left ambiguous\n'
        % (len(result['old_labels']), migrated,
           len(result['old_labels']) - migrated))
    return 0


if __name__ == '__main__':
    sys.exit(main())
