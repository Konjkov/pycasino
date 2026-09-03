import re
import subprocess
import textwrap
from pathlib import Path

SRC = Path.home() / 'bin/CASINO/src'
OUT = Path('/home/vladimir/PycharmProjects/PyCasino/.claude/skills/casino-run/references/keywords.md')

kws = Path('/tmp/claude-1000/-home-vladimir-PycharmProjects-PyCasino/e4fdb603-08a2-4608-94eb-d384aa8bcbf1/scratchpad/kws.txt').read_text().split()

# defaults from the esdf_* call sites
defaults = {}
pat = re.compile(r"esdf_(boolean|integer|double|string|physical|longint)\('([a-z0-9_]+)'\s*,\s*([^)]*?)\s*\)")
for f in SRC.glob('*.f90'):
    for m in pat.finditer(f.read_text(errors='replace')):
        kind, kw, dflt = m.groups()
        dflt = dflt.strip().rstrip(',')
        dflt = re.sub(r"_i64|_dp|_sp", '', dflt)
        defaults.setdefault(kw, (dflt, f.name))

blocks = []
for kw in kws:
    out = subprocess.run(['casinohelp', kw], capture_output=True, text=True).stdout
    title = typ = lvl = ''
    desc = []
    in_desc = False
    for line in out.splitlines():
        s = line.strip()
        if s.startswith('Title'):
            title = s.split(':', 1)[1].strip()
        elif s.startswith('Type'):
            typ = s.split(':', 1)[1].strip()
        elif s.startswith('Level'):
            lvl = s.split(':', 1)[1].strip()
        elif s.startswith('----'):
            in_desc = True
        elif in_desc:
            desc.append(s)
    desc = ' '.join(w for w in ' '.join(desc).split())
    dflt = defaults.get(kw)
    head = f"### {kw}\n\n*{title}* — {typ}, {lvl}"
    if dflt:
        head += f", default `{dflt[0]}` ({dflt[1]})"
    blocks.append(head + '\n\n' + '\n'.join(textwrap.wrap(desc, 92)) + '\n')

header = """# CASINO input keywords

Full dump of `casinohelp <keyword>` for all %d keywords of CASINO %s, plus the default
value read from the `esdf_*()` call site in `src/` (file given in brackets; a symbolic
default such as `no_default` or a variable name means it is computed or mandatory).

Regenerate with `.claude/skills/casino-run/references/gen_keywords.py`.
Types: `Logical` = T/F, `Physical` = number + optional unit, `Block` = `%%block name ... %%endblock name`.

""" % (len(kws), re.sub(r'^VERSION=|"', '', (Path.home() / 'bin/CASINO/VERSION').read_text().strip().splitlines()[-1]))

OUT.write_text(header + '\n'.join(blocks))
print(OUT, len(blocks))
