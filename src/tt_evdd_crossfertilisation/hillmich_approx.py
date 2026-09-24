"""Hillmich et al., Sec. 4.3: approximation with a target fidelity."""

from .evdd import TERM, ZERO, mk_node, contributions, nodes_by_level


def rebuild(dd, edge, drop, memo=None):
    """Rebuild the diagram reachable from `edge`, replacing every node in
    `drop` by the zero stub.

    Nodes are not mutated: mk_node re-derives each one bottom-up, so
    normalization and sharing are redone for free. The weight of the returned
    edge is the norm of the truncated state -- the diagram below it is a unit
    vector, so the scale has nowhere else to go.
    """
    if memo is None:
        memo = {}
    w, t = edge
    if w == 0 or t == TERM:
        return edge
    if t in drop:
        return ZERO
    if t not in memo:
        memo[t] = mk_node(dd, dd["level"][t],
                          rebuild(dd, dd["edges_0"][t], drop, memo),
                          rebuild(dd, dd["edges_1"][t], drop, memo))
    w_local, t_new = memo[t]
    return (w * w_local, t_new)


def candidates(dd, root_edge, level, budget, c=None):
    """The nodes of `level` that Sec. 4.3 would remove for a loss budget.

    Sorted by contribution ascending, accumulated while the running sum stays
    within the budget. The node that would push the sum past it is kept.
    """
    if c is None:
        c = contributions(dd, root_edge)
    ids = nodes_by_level(dd, root_edge).get(level, [])
    drop, used = set(), 0.0
    for i in sorted(ids, key=lambda i: c[i]):
        if used + c[i] > budget + 1e-12:
            break
        used += c[i]
        drop.add(i)
    return drop, used