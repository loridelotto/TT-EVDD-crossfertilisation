"""Sec. 4.3 of Hillmich et al. on a Dicke state.

Sweeps the target fidelity, checks the guarantee, and draws every distinct
outcome next to the exact diagram.
"""

import itertools
from pathlib import Path

import numpy as np

from .hillmich_approx import (from_state, approx_hillmich, size, fidelity,
                              contributions, nodes_by_level)
from .draw_EVDDD import to_tikz, to_document

N, K = 6, 3
SWEEP = (0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3)
PER_ROW = 3
OUT = Path("dicke.tex")


def dicke_state(n, k):
    """Uniform superposition of every bit string of Hamming weight k."""
    v = np.zeros(2 ** n)
    for ones in itertools.combinations(range(n), k):
        v[sum(1 << (n - 1 - b) for b in ones)] = 1
    return v                               # from_state normalizes it


def level_counts(evdd):
    return {lv: len(ids) for lv, ids in nodes_by_level(evdd).items()}


det_evdd = from_state(dicke_state(N, K))
c = contributions(det_evdd)
before = level_counts(det_evdd)

print(f"D({N},{K}): {size(det_evdd)} nodes")
for lv, ids in sorted(nodes_by_level(det_evdd).items()):
    print(f"  level {lv}: {sorted(round(c[i], 4) for i in ids)}")
print()

figures = [to_tikz(det_evdd, labels={i: f"{c[i]:.2f}" for i in c})]
captions = [f"$D^{{{N}}}_{{{K}}}$ exact: {size(det_evdd)} nodes"]
seen = {det_evdd[1][1]}                    # root ids already drawn

for f in SWEEP:
    app_evdd = approx_hillmich(det_evdd, f=f)
    F = fidelity(det_evdd, app_evdd)
    assert F >= f - 1e-9, (f, F)

    # which levels lost nodes, and how many
    after = level_counts(app_evdd)
    changed = ", ".join(f"level {lv}: ${before[lv]} \\to {after.get(lv, 0)}$"
                        for lv in sorted(before) if after.get(lv, 0) != before[lv])

    root = app_evdd[1][1]
    dup = "   (same diagram as above)" if root in seen else ""
    print(f"  target {f:<5} -> {size(app_evdd):>2} nodes, attained F = {F:.4f}{dup}")
    if root in seen:
        continue                           # nothing new to draw
    seen.add(root)

    figures.append(to_tikz(app_evdd))
    captions.append(f"target $f={f}$: {size(app_evdd)} nodes, $F={F:.2f}$ ({changed})")

OUT.write_text(to_document(
    [figures[i:i + PER_ROW] for i in range(0, len(figures), PER_ROW)],
    [captions[i:i + PER_ROW] for i in range(0, len(captions), PER_ROW)],
))
print(f"\nwrote {OUT} ({len(figures)} figures)")