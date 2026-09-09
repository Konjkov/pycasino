"""CASINO serialization language.
CASINO source: casl.f90

CASL looks like YAML but is not one: a line that carries no name is an item of
the block it is indented into, written without any marker and named '%u<n>' by
the reader, which YAML instead folds into a multi-line scalar. Values carry no
type until a consumer asks for one, so they are kept as strings here as well.
"""

import math
import os

CLOSING = {'[': ']', '(': ')', '{': '}'}
MAX_LINE = 79
TABSTOP = 2
TABCONT = 5


def unique(name):
    """CASL matches item names in lowercase and without spaces."""
    return name.lower().replace(' ', '')


def to_float(value):
    """CASL scalar as a float, Fortran exponents included."""
    return float(value.lower().replace('d', 'e'))


def to_casl(value):
    """CASL scalar of a float, as fortran list-directed output writes one: 17
    significant digits, and an exponent of three digits below 0.1 and above 1e17.
    """
    if math.isinf(value):
        return '+Inf' if value > 0 else '-Inf'
    exponent = math.floor(math.log10(abs(value))) if value else 0
    if -1 <= exponent < 17:
        return f'{value:.{16 - exponent}f}'
    mantissa, exponent = f'{value:.16E}'.split('E')
    return f'{mantissa}E{exponent[0]}{exponent[1:]:0>3}'


def to_bool(value):
    """CASL scalar as a bool, as get_casl_item_L projects it."""
    return unique(value) in ('.true.', 'true', 't', 'yes', 'y', '1')


class Block(dict):
    """A CASL block, keeping the order its items were written in.
    :param inline: written as '[ item, item ]' rather than as indented lines
    """

    def __init__(self, items=(), inline=False):
        super().__init__(items)
        self.inline = inline

    def add(self, name, value):
        """Append an item, naming it '%u<n>' if it has no name of its own.
        An item named '%!<name>' is commented out and is dropped by prune().
        """
        if name is None:
            name = '%u{}'.format(sum(key.startswith('%u') for key in self) + 1)
        elif name.startswith('%!'):
            name = '%!{}'.format(sum(key.startswith('%!') for key in self) + 1)
        elif name.startswith('%'):
            raise ValueError(f'"%" starts a CASL directive, and {name} is none')
        elif any(unique(key) == unique(name) for key in self):
            raise ValueError(f'item {name} found twice')
        self[name] = value

    def item(self, name):
        """Item by CASL name matching, None if the block has no such item."""
        for key, value in self.items():
            if unique(key) == unique(name):
                return value
        return None

    def implicit(self):
        """Values of the items that carry no name of their own, in order."""
        return [value for name, value in self.items() if name.startswith('%u')]


def inline(*values):
    """An inline block of unnamed items, written as '[ value, value ]'."""
    block = Block(inline=True)
    for value in values:
        block.add(None, value)
    return block


def scan(line, char, start=-1):
    """Position of the first CHAR outside brackets and quotes after START,
    -1 if there is none and -2 if a bracket or a quote is left open.
    """
    depth = 0
    quote = False
    for i in range(start + 1, len(line)):
        c = line[i]
        if quote:
            quote = c != '"'
        elif c == char and not depth:
            return i
        elif c == '"':
            quote = True
        elif c in CLOSING:
            depth += 1
        elif c in CLOSING.values():
            if not depth:
                raise ValueError(f'unmatched {c} in {line}')
            depth -= 1
    return -2 if depth or quote else -1


def fold(text):
    """Comments and empty lines out, continuation lines joined to their own.
    A line indented deeper than the one before it continues that line unless it
    ended at its ':', which is the only way a value can span several lines.
    :return: indentation, line number and text of every line left
    """
    folded = []
    for number, raw in enumerate(text, 1):
        line = raw.replace('\t', ' ')
        comment = line.find('#')
        if comment >= 0:
            line = line[:comment]
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(' '))
        line = line.strip()
        if folded and indent > folded[-1][0] and scan(folded[-1][2], ':') != len(folded[-1][2]) - 1:
            folded[-1][2] += ' ' + line
        else:
            folded.append([indent, number, line])
    return folded


def split_name(line, number=0):
    """Name and value of a line, the name ending at the first ':' that a space
    or the end of the line follows. The name is None if the item has none.
    """
    colon = -1
    while True:
        colon = scan(line, ':', colon)
        if colon < 0:
            return None, line
        if colon == len(line) - 1 or line[colon + 1] == ' ':
            name = line[:colon].strip()
            if ':' in name:
                raise ValueError(f'missing space after ":" at line {number}')
            return name, line[colon + 1 :].strip()


def parse_inline(value, number=0):
    """A '[ ... ]' block, whose items are separated by a comma and a space and
    whose named items may be inline blocks themselves.
    """
    end = scan(value, ']', 0)
    if end < 0 or end != len(value) - 1:
        raise ValueError(f'unterminated inline block at line {number}')
    block = Block(inline=True)
    body = value[1:end].strip()
    while body:
        comma = -1
        while True:
            comma = scan(body, ',', comma)
            if comma < 0 or comma == len(body) - 1 or body[comma + 1] == ' ':
                break
        if comma < 0:
            item, body = body, ''
        else:
            item, body = body[:comma], body[comma + 1 :].strip()
        colon = item.find(':')
        if colon < 0:
            block.add(None, item.strip())
        else:
            name, item = item[:colon].strip(), item[colon + 1 :].strip()
            block.add(name, parse_inline(item, number) if item.startswith('[') else item)
    return block


def prune(block):
    """Drop the items a '%!' directive comments out, and everything under them."""
    for name in [name for name in block if name.startswith('%!')]:
        del block[name]
    for value in block.values():
        if isinstance(value, Block):
            prune(value)
    return block


def parse(text):
    """Parse CASL text into a tree of blocks whose values are strings."""
    root = Block()
    stack = [[None, root]]
    folded = fold(text)
    for i, (indent, number, line) in enumerate(folded):
        while stack[-1][0] is not None and indent < stack[-1][0]:
            stack.pop()
        if stack[-1][0] is None:
            stack[-1][0] = indent
        elif indent != stack[-1][0]:
            raise ValueError(f'bad indentation at line {number}: no matching sibling')
        name, value = split_name(line, number)
        if name is not None and value.startswith('['):
            child = parse_inline(value, number)
        elif name is not None and not value and i + 1 < len(folded) and folded[i + 1][0] > indent:
            child = Block()
        else:
            child = value
        stack[-1][1].add(name, child)
        if isinstance(child, Block) and not child.inline:
            stack.append([None, child])
    return prune(root)


def read(base_path, file_name='parameters.casl'):
    """Read a CASL file, None if it is not there."""
    file_path = os.path.join(base_path, file_name)
    if not os.path.isfile(file_path):
        return None
    with open(file_path, 'r') as f:
        return parse(f.readlines())


def dump_inline(line, level, name, block, out):
    """Write an inline block, folding at the width CASINO folds at."""

    def append(text):
        nonlocal line
        if len(line) + len(text) > MAX_LINE:
            out.append(line)
            line = ' ' * (TABSTOP * level + TABCONT) + text.lstrip(' ')
        else:
            line += text

    append(f'{name}: [ ' if name else '[ ')
    for i, (key, value) in enumerate(block.items(), 1):
        comma = ', ' if i < len(block) else ''
        if isinstance(value, Block):
            line = dump_inline(line, level, '' if key.startswith('%u') else key, value, out)
            append(f' ]{comma}')
        elif key.startswith('%u'):
            append(value + comma)
        else:
            append(f'{key}: {value}{comma}')
    return line


def dump_item(name, value, level, out):
    """Write one item and, if it is a block, everything under it."""
    indent = ' ' * TABSTOP * level
    if not isinstance(value, Block):
        out.append(indent + (value if name.startswith('%u') else f'{name}: {value}'))
    elif not value:
        out.append(f'{indent}{name}: [ ]')
    elif value.inline:
        out.append(dump_inline(indent, level, name, value, out) + ' ]')
    else:
        out.append(f'{indent}{name}:')
        for key, item in value.items():
            dump_item(key, item, level + 1, out)


def dumps(tree):
    """CASL text of a tree, a blank line between its top level items."""
    out = []
    for i, (name, value) in enumerate(tree.items(), 1):
        dump_item(name, value, 0, out)
        if i < len(tree):
            out.append('')
    return '\n'.join(out) + '\n'


def write(tree, base_path, file_name='parameters.casl'):
    """Write a tree of blocks to a CASL file."""
    with open(os.path.join(base_path, file_name), 'w') as f:
        f.write(dumps(tree))
