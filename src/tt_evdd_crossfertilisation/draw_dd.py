"""Draw a decision diagram as a TikZ figure."""

import math
from fractions import Fraction

from .evdd import TERM

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


def to_tikz(dd, root_edge, labels=None, dx=2.4, dy=2.0):
    """Render the diagram as a TikZ picture, ready to paste into a LaTeX file.

    Requires \\usepackage{tikz} and \\usetikzlibrary{arrows.meta,positioning}.

    Dashed edge = branch 0, solid edge = branch 1. Weights equal to 1 are left
    unlabelled and dead branches are not drawn, following the usual convention
    in the decision-diagram literature.

    labels: optional dict id -> string, printed next to each node. Pass
            {i: str(i) for i in dd["level"]} to see the node ids, or the norm
            contributions once you compute them.
    """
    by_level = {}
    for i, lv in dd["level"].items():
        by_level.setdefault(lv, []).append(i)
    for lv in by_level:
        by_level[lv].sort()
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
        nodes = by_level[lv]
        for j, i in enumerate(nodes):
            x = (j - (len(nodes) - 1) / 2) * dx
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
    for i in sorted(dd["level"]):
        for branch, style, side, along in ((dd["edges_0"][i], "dashed", "left", 0.34),
                                           (dd["edges_1"][i], "solid", "right", 0.66)):
            w, t = branch
            if w == 0:
                continue                      # dead branch: not drawn
            dest = "term" if t == TERM else f"n{t}"
            lab = _weight(w)
            mid = f" node[lbl, {side}, pos={along}] {{${lab}$}}" if lab else ""
            out.append(f"  \\draw[e, {style}] (n{i}) --{mid} ({dest});")

    out.append(r"\end{tikzpicture}")
    return "\n".join(out)

