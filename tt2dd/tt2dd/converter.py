"""Constructive translation of a tensor train into an nd-EVDD.

Implements the linear-time construction of Quist et al., *From Tensor
Networks to Tractable Circuits, and back*, Sec. 3.1 (Theorem 1):

* For every core ``A^(r)`` with shape ``(chi_{r-1}, 2, chi_r)`` and every
  incoming virtual index ``s in [chi_{r-1}]`` there is a node ``v_s^(r)``
  labelled with variable ``x_r``.
* For every ``t in [chi_r]`` and ``b in {0, 1}`` there is a ``b``-edge
  ``v_s^(r) -> v_t^(r+1)`` with weight ``A^(r)[s, b, t]``.
* ``v_1^(1)`` is the root (``chi_0 = 1``) and ``v_1^(n+1)`` the terminal
  (``chi_n = 1``).

Every node ``v_s^(r)`` then represents the partially contracted function
``f_s^(r)(x_r, ..., x_{n-1}) = sum_{t} A^(r)[s, x_r, t] f_t^(r+1)(x_{r+1}, ...)``,
i.e. the ``s``-th row of the right environment; the induction of the paper
carries over verbatim.

On top of the bare bipartite construction the converter performs the
reduction steps listed in the project brief:

1. zero-weight edges are dropped (``|w| <= tolerance``);
2. a scalar is pushed out of every node towards the root (see
   :class:`~tt2dd.structures.ndEVDDGraph` for the normalisation rules);
3. nodes are hash-consed bottom-up, so equal sub-diagrams (e.g. rows of a
   core that coincide up to a scalar, or bond indices that a truncated SVD
   left linearly dependent *and* structurally equal) are shared; parallel
   edges that arise from such merges are summed;
4. rows that represent the zero function collapse to a 0-edge, and
   (optionally) redundant "don't care" nodes are skipped.

The construction visits every core entry exactly once, so it runs in
``O(sum_r chi_{r-1} * 2 * chi_r)`` -- linear in the size of the tensor train.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .structures import (
    TERMINAL_ID,
    EVDDEdge,
    Normalization,
    TensorTrainState,
    ndEVDDGraph,
)

__all__ = ["convert_tt_to_evdd", "convert_cores_to_evdd", "check_equivalence"]


def convert_tt_to_evdd(
    tt: TensorTrainState,
    tolerance: float = 1e-12,
    normalization: Normalization = "l2",
    eliminate_redundant: bool = False,
) -> ndEVDDGraph:
    """Translate a :class:`TensorTrainState` into an :class:`ndEVDDGraph`.

    Parameters
    ----------
    tt:
        Input tensor train.  Exact and SVD-truncated trains are handled
        identically: only the core shapes and values are used.
    tolerance:
        Edge weights with modulus at most ``tolerance`` are discarded, and
        weights are compared at this precision when sharing nodes.  Use a
        larger value (e.g. ``1e-8``) for heavily truncated inputs whose
        cores carry numerical noise.
    normalization:
        ``"l2"`` (default), ``"first"`` or ``"none"``; see
        :class:`ndEVDDGraph`.
    eliminate_redundant:
        Skip nodes whose two branches are the same single edge (fully reduced
        diagram).  Default keeps the diagram quasi-reduced.

    Returns
    -------
    ndEVDDGraph
        Diagram with ``tt.n_sites`` levels whose ``evaluate(bits)`` equals
        ``tt.amplitude(bits)`` for every bit string, up to rounding.
    """
    n = tt.n_sites
    graph = ndEVDDGraph(
        num_levels=n,
        tolerance=tolerance,
        normalization=normalization,
        eliminate_redundant=eliminate_redundant,
        variable_order=tt.site_order,
    )

    # ``below[t]`` is the edge representing bond index ``t`` of the level
    # currently being processed, i.e. the sub-diagram for f_t^(r+1).  For the
    # last core chi_n = 1 and the single bond index is the terminal.
    below: List[EVDDEdge] = [EVDDEdge(1.0 + 0j, TERMINAL_ID)]

    for level in range(n - 1, -1, -1):
        core = tt.cores[level]  # (chi_l, 2, chi_r)
        chi_l, _, chi_r = core.shape
        if chi_r != len(below):
            raise ValueError(  # defensive: TensorTrainState already validates this
                f"Core {level} has chi_r={chi_r} but {len(below)} edges were built below it."
            )
        below_w = np.fromiter((e.weight for e in below), dtype=np.complex128, count=chi_r)
        below_t = [e.target_id for e in below]

        current: List[EVDDEdge] = []
        for s in range(chi_l):
            branches: List[List[EVDDEdge]] = []
            for b in (0, 1):
                # Effective weight of the edge v_s^(r) --b--> (sub-diagram t):
                # core entry times the scalar pushed out of that sub-diagram.
                weights = core[s, b, :] * below_w
                nz = np.flatnonzero(np.abs(weights) > tolerance)
                branches.append([EVDDEdge(complex(weights[t]), below_t[t]) for t in nz])
            current.append(graph.make_node(level, branches[0], branches[1]))
        below = current

    assert len(below) == 1, "chi_0 must be 1 (validated by TensorTrainState)"
    graph.root_edge = below[0].scaled(tt.norm)
    return graph


def convert_cores_to_evdd(
    cores: Sequence[np.ndarray],
    site_order: Optional[Sequence[int]] = None,
    norm: complex = 1.0,
    **kwargs,
) -> ndEVDDGraph:
    """Convenience wrapper: build the :class:`TensorTrainState` and convert it.

    ``kwargs`` are forwarded to :func:`convert_tt_to_evdd`.
    """
    return convert_tt_to_evdd(TensorTrainState(cores, site_order=site_order, norm=norm), **kwargs)


def check_equivalence(
    tt: TensorTrainState,
    graph: ndEVDDGraph,
    bitstrings: Optional[Iterable[Sequence[int]]] = None,
    num_samples: int = 256,
    atol: float = 1e-9,
    rtol: float = 1e-9,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[bool, float]:
    """Compare ``graph.evaluate`` with ``tt.amplitude`` on a set of bit strings.

    If ``bitstrings`` is ``None`` all ``2**n`` strings are checked when
    ``n <= 12``, otherwise ``num_samples`` uniformly random strings.

    A bit string passes when ``|dd - tt| <= atol + rtol * |tt|``.
    Returns ``(all_pass, max_abs_error)``.
    """
    n = tt.n_sites
    if graph.num_levels != n:
        raise ValueError("Graph and tensor train have different numbers of variables.")
    if bitstrings is None:
        if n <= 12:
            bitstrings = (
                [(k >> (n - 1 - i)) & 1 for i in range(n)] for k in range(2**n)
            )
        else:
            rng = np.random.default_rng() if rng is None else rng
            bitstrings = (list(row) for row in rng.integers(0, 2, size=(num_samples, n)))

    worst = 0.0
    all_pass = True
    for bits in bitstrings:
        ref = tt.amplitude(bits)
        err = abs(graph.evaluate(bits) - ref)
        worst = max(worst, err)
        if err > atol + rtol * abs(ref):
            all_pass = False
    return all_pass, worst
