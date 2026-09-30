"""Read a tt2dd.nd-evdd diagram and turn it into the deterministic EVDD 
Typical use:
    det_evdd = load_nd("state.json")
    app_evdd = approx_hillmich(det_evdd, f=0.8)
"""

import json
import warnings

from .hillmich_approx import TERM, ZERO, TOL, new_dd, mk_node, _key_w


def load_nd(path):
    """The deterministic EVDD of the nd-EVDD saved in a tt2dd JSON file."""
    with open(path) as fh:
        return from_nd(json.load(fh))


def from_nd(obj):
    # The deterministic EVDD of an nd-EVDD given as a tt2dd JSON object.
    nd = _parse(obj)
    dd, (w, t) = _determinize(nd)
    if w == 0:
        raise ValueError("the nd-EVDD represents the zero vector")
    if abs(abs(w) - 1) > 1e-6:
        warnings.warn(f"the nd-EVDD has norm {abs(w):.6g}, not 1: normalizing")
    return (dd, (w / abs(w), t))


def _parse(obj):
    if obj.get("format") != "tt2dd.nd-evdd":
        raise ValueError(f"not a tt2dd.nd-evdd diagram: {obj.get('format')!r}")

    def edge(e):
        re, im = e["weight"]
        return complex(re, im), e["target"]

    n, term = obj["num_levels"], obj["terminal_id"]
    level = {term: n}                  # the terminal sits below the last level
    edges = {}
    for node in obj["nodes"]:
        level[node["id"]] = node["level"]
        edges[node["id"]] = ([edge(e) for e in node["edges_0"]],
                             [edge(e) for e in node["edges_1"]])
    return {"num_levels": n, "terminal": term, "level": level,
            "edges": edges, "root": edge(obj["root_edge"])}


def _determinize(nd):
    n, term = nd["num_levels"], nd["terminal"]
    dd = new_dd(n)
    memo = {}

    def branch(combo, lv, x):
        out = {}
        for v, a in combo.items():
            if nd["level"][v] > lv:        # edge skipped this level: don't care
                out[v] = out.get(v, 0) + a
            else:
                for w, t in nd["edges"][v][x]:
                    out[t] = out.get(t, 0) + a * w
        return {v: a for v, a in out.items() if abs(a) > TOL}

    def build(combo, lv):
        if not combo:
            return ZERO
        if lv == n:
            a = combo[term]                # only the terminal is left
            return ZERO if abs(a) < TOL else (a, TERM)

        # scale the combination so that proportional ones share a memo entry
        items = sorted(combo.items())
        a0 = items[0][1]
        key = (lv,) + tuple((v, _key_w(a / a0)) for v, a in items)
        if key not in memo:
            unit = {v: a / a0 for v, a in items}
            memo[key] = mk_node(dd, lv,
                                build(branch(unit, lv, 0), lv + 1),
                                build(branch(unit, lv, 1), lv + 1))
        w, t = memo[key]
        return (a0 * w, t)

    w_root, t_root = nd["root"]
    return (dd, build({t_root: w_root}, 0))