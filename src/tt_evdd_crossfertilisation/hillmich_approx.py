"""Deterministic edge-valued decision diagrams (EVDDs) for quantum states,
and the approximation of Hillmich et al., Sec. 4.3.

A diagram is the pair (dd, root_edge)+
Typical use:
    evdd = from_state(psi)
    app = approx_hillmich(evdd, f=0.8)
    print(size(evdd), size(app), fidelity(evdd, app))
"""

import math
import numpy as np

TERM = 0                 # id of the terminal
ZERO = (0, TERM)         # the dead edge
TOL = 1e-12              # below this a number counts as zero

def new_dd(num_levels):
    """An empty store for diagrams over num_levels qubits."""
    return {
        "num_levels": num_levels,
        "level": {},       # id -> level
        "edges_0": {},     # id -> (weight, child id) of branch 0
        "edges_1": {},     # id -> (weight, child id) of branch 1
        "_unique": {},     # key -> id: the unique table
        "_next_id": 1,     # 0 is the terminal
    }


def _key_w(z):
    """A complex weight as a hashable key, rounded to the tolerance."""
    return (round(z.real / TOL), round(z.imag / TOL))


def mk_node(dd, level, e0, e1):
    """The edge pointing to the node with branches e0, e1.

    Normalizes the two outgoing weights to unit 2-norm, pushes the scale and
    phase up into the returned edge, and reuses an equal node if the store
    already has one.
    """
    (w0, t0), (w1, t1) = e0, e1
    mag2 = (abs(w0) ** 2, abs(w1) ** 2)

    if mag2[0] < TOL and mag2[1] < TOL:          # both branches dead
        return ZERO
    if mag2[1] < TOL:                            # only branch 0 alive
        edges, top = ((1, t0), ZERO), w0
    elif mag2[0] < TOL:                          # only branch 1 alive
        edges, top = (ZERO, (1, t1)), w1
    else:                                        # both alive
        w, child = (w0, w1), (t0, t1)
        a = 0 if mag2[0] + TOL >= mag2[1] else 1     # the larger branch
        b = 1 - a
        norm = math.sqrt(mag2[0] + mag2[1])
        top = w[a] * (norm / math.sqrt(mag2[a]))
        out = [None, None]
        out[a] = (math.sqrt(mag2[a]) / norm + 0j, child[a])
        out[b] = (w[b] / top, child[b])
        edges = tuple(out)

    key = (level, edges[0][1], _key_w(edges[0][0]),
                  edges[1][1], _key_w(edges[1][0]))
    i = dd["_unique"].get(key)
    if i is None:
        i = dd["_next_id"]
        dd["_next_id"] += 1
        dd["level"][i] = level
        dd["edges_0"][i] = edges[0]
        dd["edges_1"][i] = edges[1]
        dd["_unique"][key] = i
    return (top, i)


# ------------------------------------------------- vector <-> diagram

def _build(dd, vec, level):
    if len(vec) == 1:
        w = complex(vec[0])
        return ZERO if abs(w) < TOL else (w, TERM)
    half = len(vec) // 2
    return mk_node(dd, level,
                   _build(dd, vec[:half], level + 1),
                   _build(dd, vec[half:], level + 1))


def from_state(psi):
    """The deterministic EVDD of a state vector of length 2**n.

    The vector need not be normalized: the diagram represents psi / ||psi||.
    """
    v = np.asarray(psi, dtype=complex).ravel()
    if v.size == 0 or v.size & (v.size - 1):
        raise ValueError(f"length {v.size}: must be a power of 2")
    dd = new_dd(v.size.bit_length() - 1)
    w, t = _build(dd, v, 0)
    if w == 0:
        raise ValueError("the zero vector is not a quantum state")
    # the nodes below the root have unit norm, so |w| is ||psi||:
    # dividing it out normalizes the state and keeps its global phase
    return (dd, (w / abs(w), t))


def _to_vector(dd, edge, length):
    w, t = edge
    if w == 0:
        return np.zeros(length, dtype=complex)
    if t == TERM:
        return np.array([w], dtype=complex)
    half = length // 2
    return w * np.concatenate([_to_vector(dd, dd["edges_0"][t], half),
                               _to_vector(dd, dd["edges_1"][t], half)])


def to_vector(evdd):
    """The state vector a diagram represents. Costs 2**n: for checking only."""
    dd, root = evdd
    return _to_vector(dd, root, 2 ** dd["num_levels"])

def nodes_by_level(evdd):
    """The nodes reachable from the root, grouped by level."""
    dd, root = evdd
    seen, stack = set(), [root[1]]
    while stack:
        i = stack.pop()
        if i == TERM or i in seen:
            continue
        seen.add(i)
        stack.append(dd["edges_0"][i][1])
        stack.append(dd["edges_1"][i][1])
    out = {}
    for i in seen:
        out.setdefault(dd["level"][i], []).append(i)
    for lv in out:
        out[lv].sort()
    return out


def size(evdd):
    """Number of nodes of the diagram. Counts only what is reachable: the
    store also keeps nodes left over from other diagrams."""
    return sum(len(ids) for ids in nodes_by_level(evdd).values())


def contributions(evdd):
    """Def. 2: c(v) = sum over all paths root -> v of |product of weights|^2."""
    dd, (w_root, t_root) = evdd
    by_level = nodes_by_level(evdd)
    c = {i: 0.0 for ids in by_level.values() for i in ids}
    if t_root == TERM:
        return c
    c[t_root] = abs(w_root) ** 2
    for lv in sorted(by_level):
        for i in by_level[lv]:
            for w, t in (dd["edges_0"][i], dd["edges_1"][i]):
                if t != TERM:
                    c[t] += c[i] * abs(w) ** 2
    return c


def fidelity(a, b):
    """|<a|b>|^2 between the two states, each taken normalized.
    Goes through the vectors, so it costs 2**n."""
    va, vb = to_vector(a), to_vector(b)
    return abs(np.vdot(va, vb)) ** 2 / (np.vdot(va, va).real * np.vdot(vb, vb).real)


# ------------------------------------------- Hillmich et al., Sec. 4.3

def _rebuild(dd, edge, drop, memo):
    """The diagram below `edge` with every node in `drop` replaced by ZERO.
    Nodes are not modified: mk_node rebuilds them bottom-up, so normalization
    and sharing come for free. The weight of the returned edge is the norm of
    the truncated state.
    """
    w, t = edge
    if w == 0 or t == TERM:
        return edge
    if t in drop:
        return ZERO
    if t not in memo:
        memo[t] = mk_node(dd, dd["level"][t],
                          _rebuild(dd, dd["edges_0"][t], drop, memo),
                          _rebuild(dd, dd["edges_1"][t], drop, memo))
    w_local, t_new = memo[t]
    return (w * w_local, t_new)


def _candidates(ids, c, budget):
    """The nodes of one level to remove: cheapest first, while the removed
    contributions add up to at most the budget."""
    drop, used = set(), 0.0
    for i in sorted(ids, key=lambda i: c[i]):
        if used + c[i] > budget + TOL:
            break
        used += c[i]
        drop.add(i)
    return drop


def approx_hillmich(evdd, f):
    """Hillmich et al., Sec. 4.3: approximation with target fidelity f.

    On every level the nodes are sorted by norm contribution and removed,
    cheapest first, while the removed contributions stay within 1 - f. Every
    level is tried (the paper's dry run) and the one that makes the most nodes
    disappear is kept. The result is renormalized.
    Guarantee: fidelity(evdd, result) >= f.
    """
    if not 0 < f <= 1:
        raise ValueError(f"target fidelity must be in (0, 1], got {f}")
    dd, (w, t) = evdd
    if w == 0 or t == TERM:
        return evdd

    unit = (dd, (w / abs(w), t))               # work on the normalized state
    c = contributions(unit)
    n0 = size(unit)

    best, best_removed = unit[1], 0
    for lv, ids in sorted(nodes_by_level(unit).items()):
        drop = _candidates(ids, c, 1 - f)
        if not drop:
            continue
        new = _rebuild(dd, unit[1], drop, {})
        removed = n0 - size((dd, new))
        if removed > best_removed:
            best, best_removed = new, removed

    w, t = best
    return (dd, (w / abs(w), t))