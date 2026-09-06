"""Channels of a generic Jastrow term.

The rules of a term put the pairs of particles it is a function of into
equivalence classes, and a channel is a class of sets of rank_e electrons and
rank_n nuclei that share the whole pattern of pair classes -- their signature.
Every channel carries its own set of linear parameters.
CASINO source: gbasis.f90

Framework for constructing generic Jastrow correlation factors
P. López Ríos, P. Seth, N. D. Drummond, and R. J. Needs
Phys. Rev. E 86, 036703
"""

import itertools

import numpy as np

ALL = -1


def init_groups_ee(nele):
    """One group per pair of spins the system has the particles for, zero where it
    has not: a spin with one electron has no pair of its own.
    """
    groups = np.zeros(shape=(len(nele), len(nele)), dtype=int)
    count = 0
    for i, ni in enumerate(nele):
        if not ni:
            continue
        if ni > 1:
            count += 1
            groups[i, i] = count
        for j in range(i + 1, len(nele)):
            if not nele[j]:
                continue
            count += 1
            groups[i, j] = groups[j, i] = count
    return groups


def init_groups_en(nele, natom):
    """One group per spin and nucleus."""
    groups = np.zeros(shape=(natom, len(nele)), dtype=int)
    count = 0
    for i, ni in enumerate(nele):
        if not ni:
            continue
        for ion in range(natom):
            count += 1
            groups[ion, i] = count
    return groups


def equate(groups, g1, g2):
    """Merge group g1 into g2, closing the gap the merge leaves in the numbering.
    Zero is not a group and is not contagious; -1 is where the removed pairs go.
    """
    if not g1 or not g2 or g1 == g2:
        return
    g1, g2 = max(g1, g2), min(g1, g2)
    equal, above = groups == g1, groups > g1
    groups[equal] = g2
    groups[above] -= 1


def group_by_z(groups_en, atom_numbers, ion=ALL):
    """Make every nucleus of the atomic number of ION equivalent to it, or, for
    ALL, every nucleus equivalent to the first one of its atomic number.
    """
    for i in range(len(atom_numbers)) if ion == ALL else (ion,):
        for j in range(i + 1, len(atom_numbers)):
            if atom_numbers[j] == atom_numbers[i]:
                for spin in range(groups_en.shape[1]):
                    equate(groups_en, groups_en[i, spin], groups_en[j, spin])


def parse_clause(clause, nele, atom_numbers):
    """One clause of a rule: '1-2' a pair of spins, '3-n7' or '3-Z8' a spin and a
    nucleus, '5' a spin, 'n4' or 'Z8' or 'Z' a nucleus.
    :return: the spins it names, the nucleus it names -- None for no nucleus and
        ALL for every one of them -- and whether it names it by number or by
        atomic number, in which case every nucleus of that number goes with it
    """
    spins = []
    ion = None
    match = 'n'
    parts = clause.split('-')
    if len(parts) > 2:
        raise ValueError(f'more than two particles in clause {clause}')
    for part in parts:
        part = part.strip()
        # 'N' names the periodic images of a nucleus, of which an aperiodic
        # system has none, so casino reads it as a plain nucleus number
        if part[:1] in ('n', 'N', 'Z'):
            if ion is not None:
                raise ValueError(f'two nuclei in clause {clause}')
            match = 'Z' if part[0] == 'Z' else 'n'
            if not part[1:]:
                ion = ALL
            elif match == 'Z':
                ion = next((i for i, z in enumerate(atom_numbers) if z == int(part[1:])), None)
                if ion is None:
                    raise ValueError(f'no nucleus of atomic number {part[1:]} in clause {clause}')
            else:
                ion = int(part[1:]) - 1
                if not 0 <= ion < len(atom_numbers):
                    raise ValueError(f'no nucleus {part[1:]} in clause {clause}')
        else:
            spin = int(part) - 1
            if not 0 <= spin < len(nele):
                raise ValueError(f'no particle type {part} in clause {clause}')
            spins.append(spin)
    return sorted(spins), ion, match


def digest(rule, groups_ee, groups_en, has_ee, has_en, nele, atom_numbers):
    """Apply one rule to the groups, a rule that targets pairs the term is not a
    function of being ignored rather than refused.
    """
    rule = rule.strip()
    if rule == 'N':
        # the periodic images of the primitive cell, of which an aperiodic
        # system has none
        return
    if rule == 'Z':
        if has_en:
            group_by_z(groups_en, atom_numbers)
        return
    if '=' in rule:
        digest_equality(rule, groups_ee, groups_en, has_ee, has_en, nele, atom_numbers)
    elif rule.startswith('!'):
        digest_removal(rule[1:], groups_ee, groups_en, has_ee, has_en, nele, atom_numbers)
    else:
        spins, ion, match = parse_clause(rule, nele, atom_numbers)
        if spins or ion is None:
            raise ValueError(f'expected a nucleus grouping rule, found {rule}')
        if match != 'Z':
            raise ValueError(f'a grouping rule names a nucleus by its atomic number, and {rule} does not')
        if has_en:
            group_by_z(groups_en, atom_numbers, ion)


def kind(spins, ion):
    """What a clause is about, which every clause of one rule has to agree on."""
    return len(spins), ion is None, ion == ALL


def digest_removal(rule, groups_ee, groups_en, has_ee, has_en, nele, atom_numbers):
    """Take a pair, or every pair a particle is in, out of the term."""
    spins, ion, match = parse_clause(rule, nele, atom_numbers)
    if not spins and ion is None:
        raise ValueError(f'removal rule {rule} names no particle')
    if ion == ALL:
        raise ValueError(f'removal rule {rule} names every nucleus at once')
    if match == 'Z' and has_en:
        group_by_z(groups_en, atom_numbers, ion)
    if len(spins) == 2:
        if has_ee:
            equate(groups_ee, groups_ee[spins[0], spins[1]], -1)
    elif spins and ion is not None:
        if has_en:
            equate(groups_en, groups_en[ion, spins[0]], -1)
    elif spins:
        if has_ee:
            for spin in range(len(nele)):
                equate(groups_ee, groups_ee[spins[0], spin], -1)
        if has_en:
            for i in range(len(atom_numbers)):
                equate(groups_en, groups_en[i, spins[0]], -1)
    elif has_en:
        for spin in range(len(nele)):
            equate(groups_en, groups_en[ion, spin], -1)


def digest_equality(rule, groups_ee, groups_en, has_ee, has_en, nele, atom_numbers):
    """Make the pairs the clauses of the rule name share one group."""
    clauses = rule.split('=')
    spins1, ion1, match1 = parse_clause(clauses[0], nele, atom_numbers)
    if not spins1 and ion1 is None:
        raise ValueError(f'equality rule {rule} names no particle')
    if not spins1 and ion1 == ALL:
        raise ValueError(f'equality rule {rule} names every nucleus at once')
    if len(spins1) == 2 and not has_ee:
        return
    if ion1 is not None and not has_en:
        return
    if match1 == 'Z' and has_en:
        group_by_z(groups_en, atom_numbers, ion1)
    for clause in clauses[1:]:
        spins2, ion2, match2 = parse_clause(clause, nele, atom_numbers)
        if kind(spins1, ion1) != kind(spins2, ion2):
            raise ValueError(f'clause {clause} does not conform with the first clause of {rule}')
        if match2 == 'Z' and has_en:
            group_by_z(groups_en, atom_numbers, ion2)
        if len(spins1) == 2:
            equate(groups_ee, groups_ee[spins1[0], spins1[1]], groups_ee[spins2[0], spins2[1]])
        elif spins1 and ion1 is not None:
            for i in range(len(atom_numbers)):
                if ion1 == ALL:
                    equate(groups_en, groups_en[i, spins1[0]], groups_en[i, spins2[0]])
                elif i == ion1:
                    equate(groups_en, groups_en[ion1, spins1[0]], groups_en[ion2, spins2[0]])
        elif spins1:
            if has_ee:
                for spin in range(len(nele)):
                    equate(groups_ee, groups_ee[spins1[0], spin], groups_ee[spins2[0], spin])
            if has_en:
                for i in range(len(atom_numbers)):
                    equate(groups_en, groups_en[i, spins1[0]], groups_en[i, spins2[0]])
        else:
            for spin in range(len(nele)):
                equate(groups_en, groups_en[ion1, spin], groups_en[ion2, spin])


def vector(groups_ee, groups_en, spins, ions):
    """The groups of every pair of a set of particles, the e-e pairs in the order
    (1,2), (1,3), (2,3), ... and the e-n pairs in the order (1,n1), (1,n2), ...
    """
    ee = [groups_ee[spins[j], spins[i]] for i in range(len(spins)) for j in range(i + 1, len(spins))]
    en = [groups_en[ions[j], spins[i]] for i in range(len(spins)) for j in range(len(ions))]
    return tuple(ee + en)


def signature(groups_ee, groups_en, spins, ions):
    """The signature of a set of particles: its vector of pair groups, taken in
    the order of the particles that makes the vector smallest, and every
    permutation of the particles that reaches it.

    Which order that is does not matter, only that two sets related by a
    permutation get the same signature, so a plain search over the permutations
    replaces the canonical matrix sort casino does.
    :return: the signature, and the permutations of the electrons and of the
        nuclei that leave it as it is
    """
    best = None
    permutations = []
    for electrons in itertools.permutations(range(len(spins))):
        for nuclei in itertools.permutations(range(len(ions))):
            sig = vector(groups_ee, groups_en, [spins[i] for i in electrons], [ions[i] for i in nuclei])
            if best is None or sig < best:
                best, permutations = sig, [(electrons, nuclei)]
            elif sig == best:
                permutations.append((electrons, nuclei))
    return best, permutations


def model_string(spins, ions):
    """A set of particles as a channel is named after it."""
    return '-'.join([str(spin + 1) for spin in spins] + [f'n{ion + 1}' for ion in ions])


def parse_model(model, rank):
    """The set of particles a channel name is that of."""
    spins = []
    ions = []
    for part in model.split('-'):
        part = part.strip()
        if part.startswith('n'):
            ions.append(int(part[1:]) - 1)
        else:
            spins.append(int(part) - 1)
    if (len(spins), len(ions)) != rank:
        raise ValueError(f'channel {model} names {len(spins)} electrons and {len(ions)} nuclei, not {rank}')
    return spins, ions


class Channels:
    """The channels of one term, and the pair groups its bases and cutoffs are
    defined over.
    """

    def __init__(self, rank, rules, nele, atom_numbers):
        self.rank = rank
        self.nele = nele
        self.atom_numbers = atom_numbers
        self.ee = init_groups_ee(nele)
        self.en = init_groups_en(nele, len(atom_numbers))
        for rule in rules:
            digest(rule, self.ee, self.en, rank[0] > 1, rank[1] > 0, nele, atom_numbers)
        self.signatures = []
        self.names = []
        self.permutations = []
        self.build()

    def build(self):
        """Every set of particles the system holds, sorted into channels by its
        signature, each channel named after the first set that reached it.
        """
        for spins in itertools.combinations_with_replacement(range(len(self.nele)), self.rank[0]):
            if any(spins.count(spin) > self.nele[spin] for spin in set(spins)):
                continue
            for ions in itertools.combinations(range(len(self.atom_numbers)), self.rank[1]):
                sig, permutations = signature(self.ee, self.en, spins, ions)
                # a group of zero is a pair the system does not have, and one of
                # -1 is a pair the rules have taken out of the term
                if any(group <= 0 for group in sig) or sig in self.signatures:
                    continue
                self.signatures.append(sig)
                self.names.append(model_string(spins, ions))
                self.permutations.append(permutations)

    def index(self, model):
        """The channel a 'Channel <model>' name belongs to."""
        spins, ions = parse_model(model, self.rank)
        sig, _ = signature(self.ee, self.en, spins, ions)
        if sig not in self.signatures:
            raise ValueError(f'channel {model} holds a pair of particles this system does not')
        return self.signatures.index(sig)

    def pair_names(self, kind):
        """One name per group of e-e or of e-n pairs, after the first pair in it."""
        groups = self.ee if kind == 'e-e' else self.en
        names = []
        for group in range(1, groups.max(initial=0) + 1):
            if kind == 'e-e':
                names.append(model_string(sorted(np.argwhere(groups == group)[0]), ()))
            else:
                spin, ion = np.argwhere(groups.T == group)[0]
                names.append(model_string((spin,), (ion,)))
        return names

    def pair_index(self, model, kind):
        """The group of pairs a basis or cutoff channel name belongs to."""
        groups = self.ee if kind == 'e-e' else self.en
        spins, ions = parse_model(model, (2, 0) if kind == 'e-e' else (1, 1))
        group = groups[spins[0], spins[1]] if kind == 'e-e' else groups[ions[0], spins[0]]
        if group == 0:
            raise ValueError(f'channel {model} is a pair of particles this system does not have')
        if group < 0:
            raise ValueError(f'channel {model} is a pair the rules have removed')
        return group - 1
