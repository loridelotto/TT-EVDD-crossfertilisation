import math
import numpy as np

TERM = 0
ZERO = (0, TERM)
TOL = 1e-12

def new_dd(num_levels):
    return {                                # Example for Bell State (1/sqrt(2), 0, 0 , 1/sqrt(2))
        "num_levels": num_levels,           # n = 2
        "level": {},                        # dd['level']   = {1: 1, 2: 1, 3: 0}
        "edges_0": {},                      # dd['edges_0'] = {1: (1+0j, 0), 2: (0j, 0), 3: (0.7071+0j, 1)}
        "edges_1": {},                      # dd['edges_1'] = {1: (0j, 0), 2: (1+0j, 0), 3: (0.7071+0j, 2)}
        "_unique": {},
        "_next_id": 1,      
    }



def _key_w(z):
    return (round(z.real / TOL), round(z.imag / TOL)) # constructs a key for imaginary numbers

def mk_node(dd, level, e0, e1): # lo and hi are the edges with parameters (weight, id_child)

    (w0, t0) = e0
    (w1, t1) = e1
    mag2 = (abs(w0) ** 2, abs(w1) ** 2)

    # both edges are dead
    if mag2[0] < TOL and mag2[1] < TOL:
        return ZERO

    # only one of the two edges is dead
    
    if mag2[1] < TOL: #if e1 has weight 0, then e0 must have weight 1 
        edges, top = ((1 , t0), ZERO), w0
    elif mag2[0] < TOL:
        edges, top = (ZERO, (1, t1)), w1

    # general case

    else:
        w, child = (w0, w1), (t0, t1)
        a = 0 if mag2[0] + TOL >= mag2[1] else 1 #index of the edge with biggest weight
        b = 1-a # index of the other edge
        norm = math.sqrt(mag2[0] + mag2[1])
        top = w[a] * (norm / math.sqrt(mag2[a]))
        out = [None, None]
        out[a] = (math.sqrt(mag2[a]) / norm + 0j, child[a])
        out[b] = (w[b] / top, child[b])
        edges = tuple(out)

    # build unique id for the node we are building

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

def build(dd, vec, level=0):    # takes in a vector of amplitudes and returns the incoming edge
    if len(vec) == 1:
        w = complex(vec[0])
        return ZERO if abs(w) < TOL else (w, TERM)
    half = len(vec) // 2
    return mk_node(dd, level,
                   build(dd, vec[:half], level + 1),
                   build(dd, vec[half:], level + 1))

def from_state(vec):    # takes in a 
    v = np.asarray(vec, dtype=complex).ravel()
    if v.size == 0 or v.size & (v.size - 1):
        raise ValueError(f"lunghezza {v.size}: deve essere una potenza di 2")
    dd = new_dd(v.size.bit_length() - 1)

    return dd, build(dd, v)

def to_vector(dd, edge, size_=None):
    if size_ is None:
        size_ = 2 ** dd["num_levels"]
    w, t = edge
    if w == 0:
        return np.zeros(size_, dtype=complex)
    if t == TERM:
        return np.array([w], dtype=complex)
    half = size_ // 2
    return w * np.concatenate([to_vector(dd, dd["edges_0"][t], half),
                               to_vector(dd, dd["edges_1"][t], half)])

def nodes_by_level(dd, root_edge):
    """The nodes reachable from root_edge, grouped by level."""
    seen, stack = set(), [root_edge[1]]
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


def contributions(dd, root_edge):
    """Def. 2: c(v) = sum over all paths root -> v of |product of weights|^2."""
    by_level = nodes_by_level(dd, root_edge)
    c = {i: 0.0 for ids in by_level.values() for i in ids}
    w_root, t_root = root_edge
    if t_root == TERM:
        return c
    c[t_root] = abs(w_root) ** 2
    for lv in sorted(by_level):
        for i in by_level[lv]:
            for w, t in (dd["edges_0"][i], dd["edges_1"][i]):
                if t != TERM:
                    c[t] += c[i] * abs(w) ** 2
    return c

def size(dd):
    return len(dd["level"])

