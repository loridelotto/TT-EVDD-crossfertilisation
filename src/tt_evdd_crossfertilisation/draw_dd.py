"""Draw a decision diagram as a TikZ figure. Drawing only: nothing here knows
about quantum states, contributions or approximation."""

import math
from fractions import Fraction

from .evdd import TERM, nodes_by_level

EPS = 1e-9          # tolerance for rendering, not for the structure

def _order(dd, by_level):
    """Order the nodes of each level left to right by their smallest path.

    The path is the string of branch labels read from the root, so the key is
    a property of what the node *means*, not of the order it happened to be
    built in. Two diagrams over the same state -- before and after an
    approximation, say -- therefore place corresponding nodes in the same
    order, which is what makes them comparable side by side.
    """
    path = {}
    for lv in sorted(by_level):
        for i in by_level[lv]:
            if lv == 0:
                path[i] = ""
                continue
            for j in by_level[lv - 1]:
                for b, (w, t) in enumerate((dd["edges_0"][j], dd["edges_1"][j])):
                    if w != 0 and t == i:
                        cand = path[j] + str(b)
                        if i not in path or cand < path[i]:
                            path[i] = cand
    return {lv: sorted(ids, key=lambda i: (path.get(i, ""), i))
            for lv, ids in by_level.items()}

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


def to_tikz(dd, root_edge, labels=None, dx=2.4, dy=2.0):
    """Render as a TikZ picture the diagram reachable from root_edge.

    Only reachable nodes are drawn: dd also holds nodes left over from earlier
    versions of the diagram, and drawing those would superimpose two diagrams.

    Dashed edge = branch 0, solid edge = branch 1. Weights equal to 1 are left
    unlabelled and dead branches are not drawn, as is usual in the literature.

    labels: optional dict id -> string, printed next to each node. Pass
            {i: f"{c[i]:.2f}" for i in c} to show the norm contributions.
    """
    by_level = _order(dd, nodes_by_level(dd, root_edge))
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
            lab = _weight(w)
            mid = f" node[lbl, {side}, pos={along}] {{${lab}$}}" if lab else ""
            link = f"to[{bend}]" if twin else "--"
            out.append(f"  \\draw[e, {style}] (n{i}) {link}{mid} ({dest});")

    out.append(r"\end{tikzpicture}")
    return "\n".join(out)


def to_document(rows, captions=None, gap="1.4cm", vgap="10pt"):
    """Wrap TikZ pictures into a compilable standalone document.

    rows is either a list of pictures (one row) or a list of such lists (a
    grid). captions has the same shape and is printed under each picture.

    amsmath is needed as well as tikz: the weight labels use \\tfrac.
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