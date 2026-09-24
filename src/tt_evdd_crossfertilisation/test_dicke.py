# src/tt_evdd_crossfertilisation/test_dicke.py
"""Sec. 4.3 of Hillmich et al. on a Dicke state.

Sweeps the target fidelity, checks the guarantee against the fidelity measured
on the state vector, and draws every distinct outcome next to the exact
diagram.
"""

import itertools
from pathlib import Path

import numpy as np

from .evdd import from_state, to_vector, contributions, nodes_by_level
from .draw_dd import to_tikz, to_document
from .hillmich_approx import candidates, rebuild

N, K = 6, 3
SWEEP = (0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3)
PER_ROW = 3
OUT = Path("dicke.tex")


def dicke_state(n, k):
    """Uniform superposition of every bit string of Hamming weight k."""
    v = np.zeros(2 ** n, dtype=complex)
    for ones in itertools.combinations(range(n), k):
        v[sum(1 << (n - 1 - b) for b in ones)] = 1
    return v / np.linalg.norm(v)


def live_size(dd, edge):
    """Nodes actually reachable -- dd also holds orphans from older versions."""
    return sum(len(ids) for ids in nodes_by_level(dd, edge).values())


def level_counts(dd, edge):
    return {lv: len(ids) for lv, ids in nodes_by_level(dd, edge).items()}


def unit(edge):
    """Renormalize: the rebuilt root carries the norm of the truncated state."""
    w, t = edge
    return (w / abs(w), t) if w else edge


def best_level(dd, root, c, f):
    """Sec. 4.3: dry-run every level, keep the one that removes the most nodes.

    Returns (removed, level, dropped ids, new root edge, guaranteed fidelity).
    """
    n0 = live_size(dd, root)
    best = None
    for lv in sorted(nodes_by_level(dd, root)):
        drop, _ = candidates(dd, root, lv, 1 - f, c)
        new = rebuild(dd, root, drop)
        n = live_size(dd, new)
        if best is None or n0 - n > best[0]:
            best = (n0 - n, lv, drop, new, abs(new[0]) ** 2)
    return best


psi = dicke_state(N, K)
dd, root = from_state(psi)
c = contributions(dd, root)
n0 = live_size(dd, root)
before = level_counts(dd, root)

print(f"D({N},{K}): {n0} nodes")
for lv, ids in sorted(nodes_by_level(dd, root).items()):
    print(f"  level {lv}: {sorted(round(c[i], 4) for i in ids)}")
print()

figures = [to_tikz(dd, root, labels={i: f"{c[i]:.2f}" for i in c})]
captions = [f"$D^{{{N}}}_{{{K}}}$ exact: {n0} nodes"]
seen = set()

for f in SWEEP:
    _, lv, drop, new, fid = best_level(dd, root, c, f)

    # the guarantee, read off the rebuilt root weight, against the fidelity
    # measured on the state vector -- these must agree exactly
    meas = abs(np.vdot(psi, to_vector(dd, unit(new)))) ** 2
    assert abs(fid - meas) < 1e-9, (f, fid, meas)
    assert fid >= f - 1e-9, (f, fid)

    key = (lv, frozenset(drop))
    dup = "   (same cut as above)" if key in seen else ""
    print(f"  f = {f:<5} -> level {lv}, {live_size(dd, new):>2} nodes, "
          f"F = {fid:.4f}{dup}")
    if key in seen or not drop:
        continue                      # nothing new to draw

    seen.add(key)
    after = level_counts(dd, new)
    word = "node" if len(drop) == 1 else "nodes"
    figures.append(to_tikz(dd, unit(new)))
    captions.append(
        f"$f={f}$: {len(drop)} {word} cut from level {lv} "
        f"(${before[lv]} \\to {after.get(lv, 0)}$), "
        f"{live_size(dd, new)} nodes in total, $F={fid:.2f}$")

OUT.write_text(to_document(
    [figures[i:i + PER_ROW] for i in range(0, len(figures), PER_ROW)],
    [captions[i:i + PER_ROW] for i in range(0, len(captions), PER_ROW)],
))
print(f"\nwrote {OUT} ({len(figures)} figures)")