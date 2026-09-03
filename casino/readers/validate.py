"""Validation of a CASINO input file, run before the calculation starts."""

import difflib
import logging
import os
import re

from .keywords import CASINO_KEYWORDS, KEYWORD_TYPE

logger = logging.getLogger(__name__)

# values pycasino implements, a subset of what CASINO accepts
SUPPORTED_VALUES = {
    'runtype': ('vmc', 'vmc_opt', 'vmc_dmc'),
    'atom_basis_type': ('gaussian', 'slater-type'),
    'psi_s': ('slater', 'geminal'),
    'opt_method': ('varmin', 'emin'),
    'emin_method': ('newton', 'linear', 'reconf'),
    # casino withdrew its method 2, 4 is the position dependent time step of pycasino
    'vmc_method': (1, 3, 4),
    'opt_dtvmc': (0, 1, 2),
    'dmc_method': (1, 2),
}

# keywords the reader leaves at None, so the run needs them spelled out
REQUIRED = ('neu', 'ned', 'runtype', 'atom_basis_type', 'vmc_equil_nstep', 'vmc_nstep', 'vmc_nblock')

REQUIRED_BY_RUNTYPE = {
    'vmc': (),
    'vmc_opt': ('vmc_nconfig_write', 'opt_method'),
    'vmc_dmc': (
        'vmc_nconfig_write',
        'dtdmc',
        'dmc_target_weight',
        'dmc_equil_nstep',
        'dmc_equil_nblock',
        'dmc_stats_nstep',
        'dmc_stats_nblock',
    ),
}

ORBITAL_FILE = {'gaussian': 'gwfn.data', 'slater-type': 'stowfn.data'}


class InputError(Exception):
    """Invalid input file."""


def suggestion(keyword):
    match = difflib.get_close_matches(keyword, KEYWORD_TYPE, n=1)
    if match:
        return f', did you mean {match[0]}?'
    else:
        return ''


def valid_value(keyword_type, value):
    if keyword_type == 'bool':
        return value in ('T', 'F')
    elif keyword_type == 'int':
        return re.fullmatch(r'[+-]?[0-9]+', value) is not None
    elif keyword_type in ('float', 'physical'):
        # a physical value may carry a unit, as in 'max_cpu_time : 12 hours'
        try:
            float(value.split()[0])
        except (IndexError, ValueError):
            return False
        return True
    else:
        return True


def parse(file_path):
    """Keyword lines as (line number, keyword, value) with keyword None if the line is not one,
    and blocks as (line number, block name). Block bodies are left to whoever reads the block.
    """
    keywords, blocks = [], []
    block = None
    with open(file_path, 'r') as f:
        for number, line in enumerate(f, 1):
            line = line.partition('#')[0].strip()
            if not line:
                continue
            if line.startswith('%endblock'):
                block = None
            elif line.startswith('%block'):
                block = line.split()[1].lower()
                blocks.append((number, block))
            elif block is None:
                keyword, separator, value = line.partition(':')
                if separator:
                    keywords.append((number, keyword.strip().lower(), value.strip()))
                else:
                    keywords.append((number, None, line))
    return keywords, blocks


def check_file(file_path):
    """Keyword names and value types, before the values are read. Returns the keywords found."""
    keywords, blocks = parse(file_path)
    errors = []
    seen = {}
    for number, keyword, value in keywords:
        if keyword is None:
            # esdf also separates a keyword from its value by '=' or a space, pycasino does not
            errors.append(f'line {number}: {value!r} is not a "keyword : value" line')
            continue
        keyword_type = KEYWORD_TYPE.get(keyword)
        if keyword_type is None:
            errors.append(f'line {number}: unknown keyword {keyword}{suggestion(keyword)}')
        elif keyword_type == 'block':
            errors.append(f'line {number}: {keyword} is a block, write %block {keyword} ... %endblock {keyword}')
        elif not valid_value(keyword_type, value):
            errors.append(f'line {number}: {keyword} is {keyword_type}, got {value!r}')
        elif keyword in seen:
            logger.warning(' Warning: %s on line %i is overridden on line %i', keyword, seen[keyword], number)
        seen[keyword] = number
    for number, block in blocks:
        if CASINO_KEYWORDS.get(block) != 'block':
            errors.append(f'line {number}: unknown block {block}{suggestion(block)}')
    if errors:
        raise InputError('\n'.join([f'{file_path}:'] + errors))
    return set(seen) | {block for _, block in blocks}


def check_input(input, base_path, file_keywords):
    """Values, their combinations and the files the run needs, after the input is read."""
    errors = []
    for keyword in REQUIRED + REQUIRED_BY_RUNTYPE.get(input.runtype, ()):
        if getattr(input, keyword, None) is None:
            errors.append(f'{keyword} is required for runtype {input.runtype}')
    for keyword, values in SUPPORTED_VALUES.items():
        value = getattr(input, keyword, None)
        if value is not None and value not in values:
            errors.append(f'{keyword} : {value} is not implemented, pycasino has {", ".join(map(str, values))}')
    if (
        input.runtype == 'vmc_dmc'
        and None not in (input.vmc_nconfig_write, input.dmc_target_weight)
        and input.vmc_nconfig_write < input.dmc_target_weight
    ):
        errors.append(f'vmc_nconfig_write {input.vmc_nconfig_write} is below dmc_target_weight {input.dmc_target_weight}')
    if input.opt_backflow and not input.backflow:
        errors.append('opt_backflow needs backflow : T')
    if input.opt_jastrow and not (input.use_jastrow or input.use_gjastrow):
        errors.append('opt_jastrow needs use_jastrow : T')
    if input.opt_geminal and input.psi_s != 'geminal':
        errors.append('opt_geminal needs psi_s : geminal')
    if input.psi_s == 'geminal':
        if input.ned > input.neu:
            errors.append(f'psi_s : geminal pairs the {input.ned} down-spin electrons with up-spin ones, of which there are {input.neu}')
        if input.backflow:
            # the local energy would come from the determinant while the drift comes from the
            # geminal, as geminal has no hessian for the backflow branch of kinetic_energy
            errors.append('psi_s : geminal with backflow : T is not implemented')

    orbital_file = ORBITAL_FILE.get(input.atom_basis_type)
    if orbital_file is not None and not os.path.isfile(os.path.join(base_path, orbital_file)):
        errors.append(f'atom_basis_type {input.atom_basis_type} needs {orbital_file}')
    if (input.backflow or (input.use_jastrow and not input.use_gjastrow)) and not os.path.isfile(os.path.join(base_path, 'correlation.data')):
        errors.append('use_jastrow/backflow need correlation.data')
    if (input.use_gjastrow or input.psi_s == 'geminal') and not os.path.isfile(os.path.join(base_path, 'parameters.casl')):
        errors.append('use_gjastrow/psi_s : geminal need parameters.casl')
    if errors:
        raise InputError('\n'.join([f'{input.file_path}:'] + errors))

    ignored = sorted(file_keywords - input.keywords)
    if ignored:
        logger.warning(' Warning: keywords not read by pycasino: %s\n', ' '.join(ignored))
    if 'use_tmove' in file_keywords and input.use_tmove and not input.ppotential:
        logger.warning(' Warning: use_tmove has no effect without a pseudopotential\n')
