"""Draw decision diagrams as TikZ figures.

    from .draw_EVDD import to_tikz, nd_to_tikz, to_document

    to_tikz(det_evdd)    # deterministic: the result of from_state / from_nd / approx_hillmich
    nd_to_tikz(graph)    # non-deterministic: the colleague's ndEVDDGraph, or its JSON dict

Both return a TikZ string. To turn it into a .tex file:

    from pathlib import Path
    Path("diagram.tex").write_text(to_document([to_tikz(det_evdd)]))

then compile it with:  pdflatex diagram.tex
Several pictures in the same list end up side by side.
"""

import math
from fractions import Fraction

from .hillmich_approx import TERM, nodes_by_level, contributions

EPS = 1e-9          # tolerance for rendering, not for the structure


def _real(x):
    """A real number in LaTeX. Recognises square roots of rationals with a
    small denominator, which is the form almost every weight comes in."""
    if abs(x - round(x)) < EPS:
        return f"{round(x):d}"
    sign = "-" if x < 0 else ""
    f = Fraction(x * x).limit_denominator(144)
    if abs(float(f) - x * x) < EPS:
        p, q = f.numerator, f.denominator
        rp, rq = math.isqrt(p), math.isqrt(q)
        num = str(rp) if rp * rp == p else f"\\sqrt{{{p}}}"
        if rq * rq == q:                     # rational denominator
            return f"{sign}{num}" if rq == 1 else f"{sign}\\tfrac{{{num}}}{{{rq}}}"
        if rp * rp == p:                     # only the numerator is rational
            return f"{sign}\\tfrac{{{rp}}}{{\\sqrt{{{q}}}}}"
        return f"{sign}\\sqrt{{\\tfrac{{{p}}}{{{q}}}}}"
    return f"{x:.4g}"


def _weight(z):
    """Label for an edge weight. None when it is 1, which is left unlabelled."""
    z = complex(z)
    if abs(z - 1) < EPS:
        return None
    if abs(z.imag) < EPS:
        return _real(z.real)
    if abs(z.real) < EPS:
        if abs(z.imag - 1) < EPS:
            return "i"
        if abs(z.imag + 1) < EPS:
            return "-i"
        return _real(z.imag) + "i"
    sign = "+" if z.imag > 0 else "-"
    return f"{_real(z.real)}{sign}{_real(abs(z.imag))}i"


def _edge_label(lab, contrib):
    """The text on an edge: its weight (None when it is 1) and, when given,
    its norm contribution in brackets."""
    if contrib is None:
        return f"${lab}$" if lab else None
    tag = f"{{\\color{{red!65!black}}[{contrib:.3g}]}}"
    return f"${lab}$\\,{tag}" if lab else tag


def _order(dd, by_level):
    """Order the nodes of each level left to right by their smallest path.

    The path is the string of branch labels read from the root, so the key is
    a property of what the node means, not of the order it happened to be
    built in. Two diagrams of the same state -- before and after an
    approximation -- therefore place corresponding nodes in the same order.
    """
    path = {}
    for lv in sorted(by_level):
        for i in by_level[lv]:
            if lv == min(by_level):
                path[i] = ""
                continue
            for j in by_level.get(lv - 1, []):
                for b, (w, t) in enumerate((dd["edges_0"][j], dd["edges_1"][j])):
                    if w != 0 and t == i:
                        cand = path[j] + str(b)
                        if i not in path or cand < path[i]:
                            path[i] = cand
    return {lv: sorted(ids, key=lambda i: (path.get(i, ""), i))
            for lv, ids in by_level.items()}


def to_tikz(evdd, labels=None, edge_contrib=False, dx=2.4, dy=2.0):
    """Render a diagram as a TikZ picture, ready to paste into a LaTeX file.

    Needs \\usepackage{amsmath,tikz} and \\usetikzlibrary{arrows.meta,positioning}.

    Only the nodes reachable from the root are drawn: the store also keeps
    nodes of other diagrams. Dashed edge = branch 0, solid edge = branch 1.
    Weights equal to 1 are left unlabelled and dead branches are not drawn.

    labels: optional dict id -> string printed next to each node, e.g.
            {i: f"{c[i]:.2f}" for i in c} to show the norm contributions.
    edge_contrib: also print on every edge, in brackets, its norm
            contribution: the probability of the paths through it,
            c(source) * |w|^2 since every node has unit norm below it.
    """
    dd, root_edge = evdd
    c = contributions(evdd) if edge_contrib else None
    by_level = _order(dd, nodes_by_level(evdd))
    n_levels = max(by_level, default=-1) + 1

    out = [r"\begin{tikzpicture}[",
           r"    every node/.style={font=\small},",
           r"    nd/.style={circle, draw, minimum size=7.5mm, inner sep=0pt},",
           r"    tm/.style={rectangle, draw, minimum size=6mm, inner sep=2pt},",
           r"    e/.style={-{Stealth[length=2mm]}},",
           r"    lbl/.style={font=\scriptsize, inner sep=1.5pt, fill=white,",
           r"                fill opacity=0.85, text opacity=1},",
           r"    idl/.style={font=\tiny, inner sep=1pt, gray},",
           r"  ]"]

    pos = {}
    for lv in sorted(by_level):
        ids = by_level[lv]
        for j, i in enumerate(ids):
            x = (j - (len(ids) - 1) / 2) * dx
            y = -lv * dy
            pos[i] = (x, y)
            out.append(f"  \\node[nd] (n{i}) at ({x:.2f}, {y:.2f}) {{$x_{{{lv}}}$}};")
            if labels and i in labels:
                out.append(f"  \\node[idl, above right=1pt of n{i}] {{{labels[i]}}};")
    out.append(f"  \\node[tm] (term) at (0, {-n_levels * dy:.2f}) {{$1$}};")

    # the incoming edge of the root
    w_root, t_root = root_edge
    if t_root != TERM:
        rx, ry = pos[t_root]
        out.append(f"  \\coordinate (in) at ({rx:.2f}, {ry + 1.0:.2f});")
        lab = _weight(w_root)
        mid = f" node[lbl, right] {{${lab}$}}" if lab else ""
        out.append(f"  \\draw[e] (in) --{mid} (n{t_root});")

    # the two branches are labelled at different fractions along the edge, so
    # that crossing edges do not stack their labels on top of each other
    for i in sorted(pos):
        e0, e1 = dd["edges_0"][i], dd["edges_1"][i]
        # two live branches into the same node would lie on top of each other
        twin = e0[0] != 0 and e1[0] != 0 and e0[1] == e1[1]
        for branch, style, side, along, bend in (
                (e0, "dashed", "left", 0.34, "bend left=45"),
                (e1, "solid", "right", 0.66, "bend right=45")):
            w, t = branch
            if w == 0:
                continue                      # dead branch: not drawn
            dest = "term" if t == TERM else f"n{t}"
            text = _edge_label(_weight(w), c[i] * abs(w) ** 2 if c else None)
            mid = f" node[lbl, {side}, pos={along}] {{{text}}}" if text else ""
            link = f"to[{bend}]" if twin else "--"
            out.append(f"  \\draw[e, {style}] (n{i}) {link}{mid} ({dest});")

    out.append(r"\end{tikzpicture}")
    return "\n".join(out)


def to_document(rows, captions=None, gap="1.4cm", vgap="10pt"):
    """Wrap TikZ pictures into a compilable standalone document.

    rows is either a list of pictures (one row) or a list of such lists (a
    grid). captions has the same shape and is printed under each picture.
    """
    if rows and isinstance(rows[0], str):
        rows, captions = [rows], [captions] if captions else None
    width = max(len(r) for r in rows)
    cols = ("@{\\hskip " + gap + "}").join("c" for _ in range(width))

    out = [r"\documentclass[border=12pt]{standalone}",
           r"\usepackage{amsmath}",
           r"\usepackage{tikz}",
           r"\usetikzlibrary{arrows.meta,positioning}",
           r"\begin{document}",
           r"\begin{tabular}{" + cols + "}"]
    for r, row in enumerate(rows):
        pad = [""] * (width - len(row))
        out.append("\n&\n".join(list(row) + pad))
        if captions and captions[r]:
            out.append(r"\\[4pt] " + " & ".join(list(captions[r]) + pad))
        if r != len(rows) - 1:
            out.append(r"\\[" + vgap + "]")
    out += [r"\end{tabular}", r"\end{document}"]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Non-deterministic diagrams, as written by the tt2dd converter               #
# --------------------------------------------------------------------------- #

def _nd_order(nodes, by_level, root):
    """Left-to-right order of the nd nodes of each level, by smallest path from
    the root, as _order does for deterministic diagrams."""
    path = {root: ""}
    for lv in sorted(by_level):
        for i in sorted(by_level[lv], key=lambda i: (path.get(i, "~"), i)):
            if i not in path:
                continue                           # not reachable from the root
            for b, key in enumerate(("edges_0", "edges_1")):
                for e in nodes[i][key]:
                    cand = path[i] + str(b)
                    t = e["target"]
                    if t not in path or cand < path[t]:
                        path[t] = cand
    return {lv: sorted(ids, key=lambda i: (path.get(i, "~"), i))
            for lv, ids in by_level.items()}


def nd_edge_contributions(obj):
    """Norm contribution of every edge of a non-deterministic EVDD (the JSON
    object of the tt2dd.nd-evdd format), keyed by (source, branch, index in
    the branch).

    The paths through an edge s -> t on branch b form the vector
    L_s (x) w (x) R_t, where L_s sums every path root -> s and R_t every path
    t -> terminal; its squared norm ||L_s||^2 |w|^2 ||R_t||^2 is the
    contribution. Paths of different edges may interfere, so unlike the
    deterministic case the contributions of a level need not add up to 1.
    ||L||^2 and ||R||^2 come from the Gram matrices <L_u, L_v>, <R_u, R_v>
    of the nodes of each level, built top-down and bottom-up.
    """
    term = obj["terminal_id"]
    nodes = {v["id"]: v for v in obj["nodes"]}
    by_level = {}
    for v in obj["nodes"]:
        by_level.setdefault(v["level"], []).append(v["id"])
    keys = ("edges_0", "edges_1")

    def pairs(u, v):
        """Edges of u and v on the same branch, two by two."""
        for k in keys:
            for e in nodes[u][k]:
                for f in nodes[v][k]:
                    yield complex(*e["weight"]), e["target"], complex(*f["weight"]), f["target"]

    R = {(term, term): 1.0}
    for lv in sorted(by_level, reverse=True):
        for u in by_level[lv]:
            for v in by_level[lv]:
                R[u, v] = sum(we.conjugate() * wf * R.get((te, tf), 0)
                              for we, te, wf, tf in pairs(u, v))

    root = obj["root_edge"]["target"]
    L = {(root, root): abs(complex(*obj["root_edge"]["weight"])) ** 2}
    for lv in sorted(by_level):
        for u in by_level[lv]:
            for v in by_level[lv]:
                if (u, v) in L:
                    for we, te, wf, tf in pairs(u, v):
                        L[te, tf] = L.get((te, tf), 0) + we.conjugate() * wf * L[u, v]

    return {(i, k, j): abs(L.get((i, i), 0)) * abs(complex(*e["weight"])) ** 2
                       * abs(R.get((e["target"], e["target"]), 0))
            for i in nodes for k in keys for j, e in enumerate(nodes[i][k])}


def nd_to_tikz(nd, labels=None, edge_contrib=False, dx=3.0, dy=2.2, spread=35):
    """Render a non-deterministic EVDD as a TikZ picture.

    nd is the colleague's diagram: either the JSON object of the
    tt2dd.nd-evdd format (json.load of his file, or graph.to_dict()), or his
    ndEVDDGraph itself.

    Same conventions as to_tikz: dashed = branch 0, solid = branch 1, weights
    equal to 1 unlabelled. A branch may hold several edges; when two or more
    edges join the same pair of nodes they are bent apart by `spread` degrees,
    each with its label on the outer side of its curve.

    labels: optional dict id -> string printed next to each node.
    edge_contrib: also print on every edge, in brackets, its norm
            contribution (see nd_edge_contributions).
    """
    obj = nd.to_dict() if hasattr(nd, "to_dict") else nd
    if obj.get("format") != "tt2dd.nd-evdd":
        raise ValueError(f"not a tt2dd.nd-evdd diagram: {obj.get('format')!r}")
    c = nd_edge_contributions(obj) if edge_contrib else None
    n, term = obj["num_levels"], obj["terminal_id"]
    nodes = {v["id"]: v for v in obj["nodes"]}
    by_level = {}
    for v in obj["nodes"]:
        by_level.setdefault(v["level"], []).append(v["id"])
    root = obj["root_edge"]["target"]
    by_level = _nd_order(nodes, by_level, root)

    out = [r"\begin{tikzpicture}[",
           r"    every node/.style={font=\small},",
           r"    nd/.style={circle, draw, minimum size=7.5mm, inner sep=0pt},",
           r"    tm/.style={rectangle, draw, minimum size=6mm, inner sep=2pt},",
           r"    e/.style={-{Stealth[length=2mm]}},",
           r"    lbl/.style={font=\scriptsize, inner sep=1.5pt, fill=white,",
           r"                fill opacity=0.85, text opacity=1},",
           r"    idl/.style={font=\tiny, inner sep=1pt, gray},",
           r"  ]"]

    def name(i):
        return "term" if i == term else f"n{i}"

    pos = {}
    for lv in sorted(by_level):
        ids = by_level[lv]
        for j, i in enumerate(ids):
            x = (j - (len(ids) - 1) / 2) * dx
            pos[i] = (x, -lv * dy)
            out.append(f"  \\node[nd] (n{i}) at ({x:.2f}, {-lv * dy:.2f}) {{$x_{{{lv}}}$}};")
            if labels and i in labels:
                where = "left" if x < 0 else "right"
                out.append(f"  \\node[idl, {where}=2pt of n{i}] {{{labels[i]}}};")
    out.append(f"  \\node[tm] (term) at (0, {-n * dy:.2f}) {{$1$}};")

    # the incoming edge of the root
    w_root = complex(*obj["root_edge"]["weight"])
    if root != term and w_root != 0:
        rx, ry = pos[root]
        out.append(f"  \\coordinate (in) at ({rx:.2f}, {ry + 1.0:.2f});")
        lab = _weight(w_root)
        mid = f" node[lbl, right] {{${lab}$}}" if lab else ""
        out.append(f"  \\draw[e, solid] (in) --{mid} ({name(root)});")

    # all edges, grouped by (source, target) so that parallel ones can be
    # spread out: branch 0 first, then branch 1
    for lv in sorted(by_level):
        for i in by_level[lv]:
            groups = {}
            for key, style in (("edges_0", "dashed"), ("edges_1", "solid")):
                for j, e in enumerate(nodes[i][key]):
                    groups.setdefault(e["target"], []).append(
                        (style, complex(*e["weight"]), c[i, key, j] if c else None))
            for t, edges in groups.items():
                k = len(edges)
                for j, (style, w, con) in enumerate(edges):
                    angle = (j - (k - 1) / 2) * spread
                    text = _edge_label(_weight(w), con)
                    if angle == 0:
                        link = "--"
                        place = "left" if style == "dashed" else "right"
                    else:
                        # going down, "bend right" bulges west and "bend left"
                        # east: the label goes on the bulging side
                        link = f"to[bend {'right' if angle < 0 else 'left'}={abs(angle):g}]"
                        place = "auto, swap" if angle < 0 else "auto"
                    mid = f" node[lbl, {place}, pos=0.5] {{{text}}}" if text else ""
                    out.append(f"  \\draw[e, {style}] (n{i}) {link}{mid} ({name(t)});")

    out.append(r"\end{tikzpicture}")
    return "\n".join(out)