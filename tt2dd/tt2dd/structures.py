"""Core data structures for the ``tt2dd`` bridge.

Two families of classes are defined in this script.

* class:`TensorTrainState`, a tensor train/matrix product state (MPS),
  stored as a list of 3-index NumPy cores ``A[i]`` of shape
  ``(chi_{i-1}, 2, chi_i)`` with open boundaries ``chi_0 = chi_n = 1``.
* class:`EVDDEdge`, :class:`EVDDNode`, :class:`ndEVDDGraph` a
  non-deterministic edge-valued decision diagram (nd-EVDD) in the sense of
  Quist et al.(Section 2).

nd-EVDD
------------------------
Every node ``v`` is labelled with a variable level and carries a
*list* of 0-edges and a *list* of 1-edges (this is what makes the diagram
non-deterministic since a deterministic EVDD has at most one edge per branch).
The pseudo-Boolean function represented by an edge ``e = (w, v)`` is

    f_e(x) = w * f_v(x)

and the function of a node ``v`` at level ``l`` is

    f_v(x) = sum_{e in edges_{x_l}(v)}  f_e(x_{l+1}, ..., x_{n-1}),

with the terminal node representing the constant ``1``. ``f(x)`` is the sum 
over all root-to-terminal paths consistent with ``x`` ofthe product of edge 
weights along the path.  An edge that skips ``k`` levels
(only made when redundant-node elimination is enabled) is interpreted as
passing through ``k`` implicit "don't care" nodes with weight-1 edges.

Conventions
-----------
* Levels are 0-based and run from the root (level 0) down to the terminal
  (level ``num_levels``).  Level ``i`` of the diagram corresponds to core
  ``i`` of the tensor train.
* Bit strings are always given in *level order* (index ``i`` of the bit
  string is the value of the variable at level ``i``).  The optional
  ``variable_order``/``site_order`` metadata records which original
  qubit / site each level corresponds to, but is not used for evaluation.
* State vectors are big-endian in level order: the amplitude of bit string
  ``x_0 x_1 ... x_{n-1}`` sits at flat index ``sum_i x_i * 2**(n-1-i)``.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "TERMINAL_ID",
    "Normalization",
    "TensorTrainState",
    "EVDDEdge",
    "EVDDNode",
    "ndEVDDGraph",
]

#Node id reserved for the terminal node.
TERMINAL_ID: int = 0

#Supported edge-weight normalisation rules, see `ndEVDDGraph.make_node`.
Normalization = Literal["l2", "first", "none"]

Bits = Sequence[int]


# --------------------------------------------------------------------------- #
# Tensor train                                                                #
# --------------------------------------------------------------------------- #
class TensorTrainState:
    """A tensor train (matrix product state) with binary physical indices.

    Parameters
    ----------
    cores:
        Sequence of arrays.  Core ``i`` must have shape ``(chi_{i-1}, 2, chi_i)``;
        the boundary cores can also be passed as 2-D arrays of shape
        ``(2, chi_1)`` and ``(chi_{n-1}, 2)`` and are reshaped to 3-D.  Bond
        dimensions must chain consistently and ``chi_0 = chi_n = 1``!
    site_order:
        Which original site(/qubit) each core acts on (defaults to
        ``range(n)``).  Pure metadata: it is propagated to the decision
        diagram as ``variable_order``.
    norm:
        Global scalar prefactor.  The represented function is
        ``norm * contraction(cores)``.
    copy:
        If ``True`` (default) the cores are copied and cast to ``complex128``.

    Notes
    -----
    The class deliberately knows nothing about *how* the cores were obtained
    it only validates shapes. 
    """

    def __init__(
        self,
        cores: Sequence[np.ndarray],
        site_order: Optional[Sequence[int]] = None,
        norm: complex = 1.0,
        copy: bool = True,
    ) -> None:
        if len(cores) == 0:
            raise ValueError("A tensor train needs at least one core.")

        fixed: List[np.ndarray] = []
        n = len(cores)
        for i, raw in enumerate(cores):
            a = np.array(raw, dtype=np.complex128, copy=True) if copy else np.asarray(raw)
            if a.ndim == 2:
                # Accept 2-D boundary cores (as in the paper).
                if i == 0 and n > 1:
                    a = a[None, :, :]
                elif i == n - 1 and n > 1:
                    a = a[:, :, None]
                else:#2-D?
                    raise ValueError(f"Core {i} is 2-D but is not a boundary core.")
            elif a.ndim == 1 and n == 1:
                a = a[None, :, None]
            if a.ndim != 3: # check if it is 3-D
                raise ValueError(f"Core {i} must be 3-D (chi_l, 2, chi_r); got shape {a.shape}.")
            if a.shape[1] != 2:
                raise ValueError(
                    f"Core {i} has physical dimension {a.shape[1]}; only qubits (d=2) are supported."
                )
            fixed.append(a)

        for i in range(n - 1):
            if fixed[i].shape[2] != fixed[i + 1].shape[0]:
                raise ValueError(
                    f"Bond mismatch between core {i} (chi_r={fixed[i].shape[2]}) and "
                    f"core {i + 1} (chi_l={fixed[i + 1].shape[0]})."
                )
        if fixed[0].shape[0] != 1 or fixed[-1].shape[2] != 1:
            raise ValueError(
                "Open boundary conditions required: the first core must have chi_0 = 1 and the last core chi_n = 1."
            )

        self.cores: List[np.ndarray] = fixed
        self.site_order: List[int] = (
            list(range(n)) if site_order is None else [int(s) for s in site_order]
        )
        if len(self.site_order) != n:
            raise ValueError("site_order must have one entry per core.")
        self.norm: complex = complex(norm)

    # -- constructors ------------------------------------------------------ #
    @classmethod
    def from_numpy(
        cls,
        cores: Sequence[np.ndarray],
        site_order: Optional[Sequence[int]] = None,
        norm: complex = 1.0,
    ) -> "TensorTrainState":
        """Build from a list of NumPy cores (3-D, or 2-D at the boundaries)."""
        return cls(cores, site_order=site_order, norm=norm)

    @classmethod
    def from_quimb(cls, mps: Any, site_order: Optional[Sequence[int]] = None) -> "TensorTrainState":
        """Build from a ``quimb.tensor.MatrixProductState`` (or compatible 1-D TN).

        The physical index of site ``i`` is taken from ``mps.site_ind(i)`` and
        the virtual bonds from ``mps.bond(i, i+1)``; each tensor is transposed
        to ``(left bond, physical, right bond)`` before extraction, so this is
        independent of quimb's internal index ordering.  Periodic MPS are not
        supported.
        """
        try:
            n = int(mps.L)
        except AttributeError as exc:  # pragma: no cover - depends on quimb
            raise TypeError("from_quimb expects a quimb MatrixProductState-like object.") from exc
        if getattr(mps, "cyclic", False):
            raise ValueError("Periodic (cyclic) MPS are not supported; only open boundaries.")

        cores: List[np.ndarray] = []
        for i in range(n):
            tensor = mps[i]
            phys = mps.site_ind(i)
            left = mps.bond(i - 1, i) if i > 0 else None
            right = mps.bond(i, i + 1) if i < n - 1 else None
            order = [ix for ix in (left, phys, right) if ix is not None]
            arr = np.asarray(tensor.transpose(*order).data, dtype=np.complex128)
            if left is None:
                arr = arr[None, ...]
            if right is None:
                arr = arr[..., None]
            cores.append(arr)
        return cls(cores, site_order=site_order, copy=False)

    @classmethod
    def from_state_vector(
        cls,
        psi: np.ndarray,
        max_bond: Optional[int] = None,
        cutoff: float = 0.0,
        site_order: Optional[Sequence[int]] = None,
    ) -> "TensorTrainState":
        """Exact (or truncated) TT-SVD decomposition of a dense state vector.

        ``psi`` must have length ``2**n`` and is interpreted big-endian (site 0
        is the most significant bit).  With ``max_bond``/``cutoff`` the sweep
        keeps at most ``max_bond`` singular values per bond and drops those
        below ``cutoff * s_max``; this yields the standard truncated TT.
        """
        psi = np.asarray(psi, dtype=np.complex128).ravel()
        n = int(round(math.log2(psi.size)))
        if 2**n != psi.size:
            raise ValueError("State vector length must be a power of two.")
        cores: List[np.ndarray] = []
        rest = psi.reshape(1, -1)  # (chi_{i-1}, 2**(n-i))
        for _ in range(n - 1):
            chi_l = rest.shape[0]
            mat = rest.reshape(chi_l * 2, -1)
            u, s, vh = np.linalg.svd(mat, full_matrices=False)
            k = _rank_to_keep(s, max_bond, cutoff)
            cores.append(u[:, :k].reshape(chi_l, 2, k))
            rest = s[:k, None] * vh[:k, :]
        cores.append(rest.reshape(rest.shape[0], 2, 1))
        return cls(cores, site_order=site_order, copy=False)

    # -- properties -------------------------------------------------------- #
    @property
    def n_sites(self) -> int:
        """Number of physical sites / qubits."""
        return len(self.cores)

    @property
    def bond_dims(self) -> List[int]:
        """``[chi_0, chi_1, ..., chi_n]`` (boundaries included, both equal 1)."""
        return [self.cores[0].shape[0]] + [c.shape[2] for c in self.cores]

    @property
    def max_bond_dim(self) -> int:
        return max(self.bond_dims)

    @property
    def num_parameters(self) -> int:
        """Total number of stored core entries (the "size" of the TT)."""
        return int(sum(c.size for c in self.cores))

    # -- evaluation -------------------------------------------------------- #
    def amplitude(self, bits: Bits) -> complex:
        """Contract the TT on a single bit string given in core (level) order."""
        bits = tuple(int(b) for b in bits)
        if len(bits) != self.n_sites:
            raise ValueError(f"Expected {self.n_sites} bits, got {len(bits)}.")
        vec = np.ones(1, dtype=np.complex128)
        for core, b in zip(self.cores, bits):
            vec = vec @ core[:, b, :]
        return complex(self.norm * vec[0])

    def to_state_vector(self) -> np.ndarray:
        """Full contraction to a dense ``2**n`` vector (big-endian, core order).

        Exponential in ``n``; intended for testing on small instances.
        """
        acc = self.cores[0][0]  # (2, chi_1)
        for core in self.cores[1:]:
            acc = np.tensordot(acc, core, axes=([acc.ndim - 1], [0]))
        return self.norm * acc.reshape(-1)

    # -- manipulation ------------------------------------------------------ #
    def truncate(self, max_bond: Optional[int] = None, cutoff: float = 0.0) -> "TensorTrainState":
        """Return a new TT with bond dimensions reduced by SVD truncation.

        The train is first brought into right-canonical form (QR sweep from
        the right) so that the subsequent left-to-right SVD sweep performs
        optimal (Schmidt-value) truncation at every bond.  This mirrors what
        Student 2's pipeline produces and is handy for generating test inputs.
        """
        cores = [c.copy() for c in self.cores]
        n = len(cores)
        # Right-canonicalise: A[i] = R Q with Q having orthonormal rows.
        for i in range(n - 1, 0, -1):
            chi_l, d, chi_r = cores[i].shape
            mat = cores[i].reshape(chi_l, d * chi_r)
            q_t, r_t = np.linalg.qr(mat.T)  # mat.T = q_t r_t  =>  mat = r_t.T q_t.T
            q, r = q_t.T, r_t.T
            cores[i] = q.reshape(q.shape[0], d, chi_r)
            cores[i - 1] = np.einsum("axb,bc->axc", cores[i - 1], r)
        # Left-to-right SVD sweep with truncation.
        for i in range(n - 1):
            chi_l, d, chi_r = cores[i].shape
            mat = cores[i].reshape(chi_l * d, chi_r)
            u, s, vh = np.linalg.svd(mat, full_matrices=False)
            k = _rank_to_keep(s, max_bond, cutoff)
            cores[i] = u[:, :k].reshape(chi_l, d, k)
            cores[i + 1] = np.einsum("ab,bxc->axc", s[:k, None] * vh[:k, :], cores[i + 1])
        return TensorTrainState(cores, site_order=self.site_order, norm=self.norm, copy=False)

    def __repr__(self) -> str:
        return (
            f"TensorTrainState(n_sites={self.n_sites}, bond_dims={self.bond_dims}, "
            f"norm={self.norm!r})"
        )


def _rank_to_keep(s: np.ndarray, max_bond: Optional[int], cutoff: float) -> int:
    """Number of singular values to keep given an absolute cap and a relative cutoff."""
    if s.size == 0:
        return 1
    k = int(np.count_nonzero(s > cutoff * s[0])) if s[0] > 0 else 1
    k = max(k, 1)
    if max_bond is not None:
        k = min(k, int(max_bond))
    return k


# --------------------------------------------------------------------------- #
# Decision diagram                                                            #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class EVDDEdge:
    """A weighted edge pointing at ``target_id``.

    The edge represents the function ``weight * f_target``.
    """

    weight: complex
    target_id: int

    def is_zero(self, tolerance: float = 0.0) -> bool:
        return abs(self.weight) <= tolerance

    def scaled(self, factor: complex) -> "EVDDEdge":
        return EVDDEdge(complex(factor) * self.weight, self.target_id)

    def to_dict(self) -> Dict[str, Any]:
        w = complex(self.weight)
        return {"weight": [w.real, w.imag], "target": self.target_id}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EVDDEdge":
        re, im = d["weight"]
        return cls(complex(re, im), int(d["target"]))


@dataclass(slots=True)
class EVDDNode:
    """An internal nd-EVDD node.

    ``edges_0`` / ``edges_1`` are the (possibly empty, possibly multi-element)
    tuples of 0-edges and 1-edges.  A deterministic EVDD is the special case
    where each tuple has at most one element.  Inside a :class:`ndEVDDGraph`
    the tuples are kept sorted by ``target_id`` and contain no duplicate
    targets, which makes the node's hash-consing key canonical.
    """

    node_id: int
    var_level: int
    edges_0: Tuple[EVDDEdge, ...]
    edges_1: Tuple[EVDDEdge, ...]

    def branch(self, x: int) -> Tuple[EVDDEdge, ...]:
        """The outgoing edges for physical value ``x`` (0 or 1)."""
        return self.edges_1 if x else self.edges_0

    @property
    def is_terminal(self) -> bool:
        return self.node_id == TERMINAL_ID

    @property
    def out_degree(self) -> int:
        return len(self.edges_0) + len(self.edges_1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.node_id,
            "level": self.var_level,
            "edges_0": [e.to_dict() for e in self.edges_0],
            "edges_1": [e.to_dict() for e in self.edges_1],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EVDDNode":
        return cls(
            node_id=int(d["id"]),
            var_level=int(d["level"]),
            edges_0=tuple(EVDDEdge.from_dict(e) for e in d["edges_0"]),
            edges_1=tuple(EVDDEdge.from_dict(e) for e in d["edges_1"]),
        )


# Canonical hash-consing key of a node: (level, branch0 key, branch1 key) with
# each branch a tuple of (target_id, rounded re, rounded im).
_BranchKey = Tuple[Tuple[int, float, float], ...]
_NodeKey = Tuple[int, _BranchKey, _BranchKey]


class ndEVDDGraph:
    """Class for non-deterministic edge-valued decision diagram.

    Parameters
    ----------
    num_levels:
        Number of Boolean variables ``n``.  Nodes live on levels
        ``0..n-1``; the terminal node has level ``n`` and id :data:`TERMINAL_ID`.
    tolerance:
        Weights with modulus ``<= tolerance`` are treated as zero (edges are
        dropped), and weights are quantised to roughly ``tolerance * 100``
        when computing hash-consing keys, so nodes that are equal up to this
        precision are shared.
    normalization:
        Rule used by :meth:`make_node` to push a scalar out of every node so
        that structurally equal sub-diagrams that differ by a scalar are
        merged:

        * ``"l2"``   -- the vector of all outgoing weights gets unit 2-norm and
          the first (canonically ordered) weight is made real and positive.
          Numerically robust; matches the convention used by MQT's vector DDs.
        * ``"first"`` -- the first non-zero outgoing weight is set to exactly
          ``1`` (classic EVDD / QMDD normalisation).
        * ``"none"`` -- no normalisation (nodes are still deduplicated, but
          only when they are literally equal).
    eliminate_redundant:
        If ``True``, a node whose 0- and 1-branch consist of the *same single*
        edge is not created; its parent points directly at the child (the
        edge then skips a level).  This yields a fully reduced diagram.  The
        default ``False`` keeps the diagram quasi-reduced (every root-to-
        terminal path visits every level), which is what MQT's DD package and
        the TT<->nd-EVDD correspondence of Quist et al. assume.
    variable_order:
        Optional list mapping each level to an original qubit / site label.
    """

    def __init__(
        self,
        num_levels: int,
        tolerance: float = 1e-12,
        normalization: Normalization = "l2",
        eliminate_redundant: bool = False,
        variable_order: Optional[Sequence[int]] = None,
    ) -> None:
        if num_levels < 1:
            raise ValueError("num_levels must be >= 1.")
        if normalization not in ("l2", "first", "none"):
            raise ValueError(f"Unknown normalization {normalization!r}.")
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative.")

        self.num_levels: int = int(num_levels)
        self.tolerance: float = float(tolerance)
        self.normalization: Normalization = normalization
        self.eliminate_redundant: bool = bool(eliminate_redundant)
        self.variable_order: List[int] = (
            list(range(num_levels)) if variable_order is None else [int(v) for v in variable_order]
        )
        if len(self.variable_order) != self.num_levels:
            raise ValueError("variable_order must have one entry per level.")

        self.nodes: Dict[int, EVDDNode] = {
            TERMINAL_ID: EVDDNode(TERMINAL_ID, self.num_levels, (), ())
        }
        self.root_edge: EVDDEdge = EVDDEdge(0j, TERMINAL_ID)

        self._unique: Dict[_NodeKey, int] = {}
        self._next_id: int = TERMINAL_ID + 1
        # decimals used when quantising weights for the unique-table key
        self._key_decimals: int = (
            max(0, int(round(-math.log10(self.tolerance))) - 2) if self.tolerance > 0 else 15
        )

    # -- basic queries ----------------------------------------------------- #
    @property
    def terminal(self) -> EVDDNode:
        return self.nodes[TERMINAL_ID]

    @property
    def root(self) -> EVDDNode:
        return self.nodes[self.root_edge.target_id]

    @property
    def num_nodes(self) -> int:
        """Number of internal (non-terminal) nodes."""
        return len(self.nodes) - 1

    @property
    def num_edges(self) -> int:
        return sum(n.out_degree for n in self.nodes.values())

    def nodes_per_level(self) -> List[int]:
        """Internal node count per level (length ``num_levels``)."""
        counts = [0] * self.num_levels
        for node in self.nodes.values():
            if not node.is_terminal:
                counts[node.var_level] += 1
        return counts

    @property
    def max_width(self) -> int:
        return max(self.nodes_per_level()) if self.num_nodes else 0

    @property
    def is_deterministic(self) -> bool:
        return all(len(n.edges_0) <= 1 and len(n.edges_1) <= 1 for n in self.nodes.values())

    # -- construction ------------------------------------------------------ #
    def _canonical_branch(self, edges: Iterable[EVDDEdge]) -> Tuple[EVDDEdge, ...]:
        """Merge parallel edges to the same target, drop zeros, sort by target."""
        merged: Dict[int, complex] = {}
        for e in edges:
            merged[e.target_id] = merged.get(e.target_id, 0j) + complex(e.weight)
        return tuple(
            EVDDEdge(w, t) for t, w in sorted(merged.items()) if abs(w) > self.tolerance
        )

    def _branch_key(self, edges: Tuple[EVDDEdge, ...]) -> _BranchKey:
        d = self._key_decimals
        # +0.0 avoids distinguishing -0.0 from 0.0 in the key
        return tuple(
            (e.target_id, round(e.weight.real, d) + 0.0, round(e.weight.imag, d) + 0.0)
            for e in edges
        )

    def _normalizer(self, edges_0: Tuple[EVDDEdge, ...], edges_1: Tuple[EVDDEdge, ...]) -> complex:
        """Scalar ``c`` to divide all outgoing weights by (never zero)."""
        first = (edges_0 or edges_1)[0].weight
        if self.normalization == "none":
            return 1.0 + 0j
        if self.normalization == "first":
            return complex(first)
        # "l2"
        norm = math.sqrt(sum(abs(e.weight) ** 2 for e in edges_0 + edges_1))
        phase = first / abs(first)
        return complex(norm * phase)

    def make_node(
        self,
        var_level: int,
        edges_0: Iterable[EVDDEdge],
        edges_1: Iterable[EVDDEdge],
    ) -> EVDDEdge:
        """Create (or look up) a node and return a normalised edge to it.

        This is the only way nodes should be added during construction.  The
        returned edge carries the scalar that was pushed out of the node, so
        that ``returned_edge`` represents exactly ``sum_x |x> (x) branch_x``.

        Steps: merge parallel edges and drop zero weights; detect the zero
        function (returns a 0-weight edge to the terminal); optionally
        eliminate a redundant node; normalise weights; hash-cons.
        """
        if not 0 <= var_level < self.num_levels:
            raise ValueError(f"var_level {var_level} out of range [0, {self.num_levels}).")

        e0 = self._canonical_branch(edges_0)
        e1 = self._canonical_branch(edges_1)
        for e in e0 + e1:
            tgt = self.nodes.get(e.target_id)
            if tgt is None:
                raise KeyError(f"Edge target {e.target_id} does not exist.")
            if tgt.var_level <= var_level:
                raise ValueError("Edges must point to strictly lower levels (ordered DD).")

        if not e0 and not e1:
            return EVDDEdge(0j, TERMINAL_ID)

        if (
            self.eliminate_redundant
            and len(e0) == 1
            and len(e1) == 1
            and self._branch_key(e0) == self._branch_key(e1)
        ):
            return e0[0]

        c = self._normalizer(e0, e1)
        inv = 1.0 / c
        e0 = tuple(e.scaled(inv) for e in e0)
        e1 = tuple(e.scaled(inv) for e in e1)

        key: _NodeKey = (var_level, self._branch_key(e0), self._branch_key(e1))
        node_id = self._unique.get(key)
        if node_id is None:
            node_id = self._next_id
            self._next_id += 1
            self.nodes[node_id] = EVDDNode(node_id, var_level, e0, e1)
            self._unique[key] = node_id
        return EVDDEdge(c, node_id)

    # -- evaluation -------------------------------------------------------- #
    def evaluate(self, bits: Bits) -> complex:
        """Value of the represented function on a bit string (level order).

        Linear in the number of nodes and edges (memoised traversal).
        """
        bits = tuple(int(b) for b in bits)
        if len(bits) != self.num_levels:
            raise ValueError(f"Expected {self.num_levels} bits, got {len(bits)}.")
        memo: Dict[int, complex] = {TERMINAL_ID: 1.0 + 0j}

        def value(node_id: int) -> complex:
            cached = memo.get(node_id)
            if cached is not None:
                return cached
            node = self.nodes[node_id]
            total = 0j
            for e in node.branch(bits[node.var_level]):
                total += e.weight * value(e.target_id)
            memo[node_id] = total
            return total

        return complex(self.root_edge.weight * value(self.root_edge.target_id))

    def to_state_vector(self) -> np.ndarray:
        """Dense ``2**num_levels`` vector (big-endian in level order).

        Exponential in the number of levels; for testing on small instances.
        Level-skipping edges are expanded with implicit "don't care" nodes.
        """
        n = self.num_levels
        memo: Dict[int, np.ndarray] = {TERMINAL_ID: np.ones(1, dtype=np.complex128)}

        def lifted(e: EVDDEdge, parent_level: int) -> np.ndarray:
            child = dense(e.target_id)
            gap = self.nodes[e.target_id].var_level - parent_level - 1
            if gap > 0:
                child = np.kron(np.ones(2**gap, dtype=np.complex128), child)
            return e.weight * child

        def dense(node_id: int) -> np.ndarray:
            cached = memo.get(node_id)
            if cached is not None:
                return cached
            node = self.nodes[node_id]
            half = 2 ** (n - node.var_level - 1)
            parts = []
            for x in (0, 1):
                acc = np.zeros(half, dtype=np.complex128)
                for e in node.branch(x):
                    acc += lifted(e, node.var_level)
                parts.append(acc)
            vec = np.concatenate(parts)
            memo[node_id] = vec
            return vec

        return lifted(self.root_edge, -1)

    # -- reduction / canonicalisation -------------------------------------- #
    def reachable_ids(self) -> List[int]:
        """Ids of all nodes reachable from the root (terminal included)."""
        seen = {self.root_edge.target_id}
        stack = [self.root_edge.target_id]
        while stack:
            node = self.nodes[stack.pop()]
            for e in node.edges_0 + node.edges_1:
                if e.target_id not in seen:
                    seen.add(e.target_id)
                    stack.append(e.target_id)
        return sorted(seen)

    def reduce(self, **overrides: Any) -> "ndEVDDGraph":
        """Rebuild the diagram bottom-up through :meth:`make_node`.

        Returns a new graph in which unreachable nodes are removed, parallel
        edges merged, zero edges dropped, weights re-normalised and equal
        sub-diagrams shared.  Useful after manual graph surgery (e.g. node
        elimination) or after loading an un-normalised diagram from JSON.
        Keyword ``overrides`` replace constructor settings (``tolerance``,
        ``normalization``, ``eliminate_redundant``).
        """
        settings: Dict[str, Any] = dict(
            tolerance=self.tolerance,
            normalization=self.normalization,
            eliminate_redundant=self.eliminate_redundant,
            variable_order=self.variable_order,
        )
        settings.update(overrides)
        out = ndEVDDGraph(self.num_levels, **settings)

        # edge that represents old node ``i`` inside ``out``
        mapping: Dict[int, EVDDEdge] = {TERMINAL_ID: EVDDEdge(1.0 + 0j, TERMINAL_ID)}
        ids = [i for i in self.reachable_ids() if i != TERMINAL_ID]
        for nid in sorted(ids, key=lambda i: -self.nodes[i].var_level):
            node = self.nodes[nid]
            new_0 = [mapping[e.target_id].scaled(e.weight) for e in node.edges_0]
            new_1 = [mapping[e.target_id].scaled(e.weight) for e in node.edges_1]
            mapping[nid] = out.make_node(node.var_level, new_0, new_1)

        out.root_edge = mapping[self.root_edge.target_id].scaled(self.root_edge.weight)
        return out

    # -- serialisation ----------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        """Plain-Python representation (JSON-serialisable).

        Nodes are listed top-down (by level, then id); the terminal node is
        implicit and identified by ``terminal_id``.
        """
        internal = sorted(
            (n for n in self.nodes.values() if not n.is_terminal),
            key=lambda n: (n.var_level, n.node_id),
        )
        return {
            "format": "tt2dd.nd-evdd",
            "version": 1,
            "num_levels": self.num_levels,
            "variable_order": list(self.variable_order),
            "tolerance": self.tolerance,
            "normalization": self.normalization,
            "eliminate_redundant": self.eliminate_redundant,
            "terminal_id": TERMINAL_ID,
            "root_edge": self.root_edge.to_dict(),
            "nodes": [n.to_dict() for n in internal],
        }

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, path: str, indent: Optional[int] = 2) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_json(indent=indent))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ndEVDDGraph":
        """Inverse of :meth:`to_dict`.  Nodes are loaded verbatim (no re-normalisation)."""
        if d.get("format") != "tt2dd.nd-evdd":
            raise ValueError("Not a tt2dd nd-EVDD dictionary.")
        if int(d.get("terminal_id", TERMINAL_ID)) != TERMINAL_ID:
            raise ValueError(f"Unsupported terminal id {d['terminal_id']}; expected {TERMINAL_ID}.")
        g = cls(
            num_levels=int(d["num_levels"]),
            tolerance=float(d.get("tolerance", 1e-12)),
            normalization=d.get("normalization", "l2"),
            eliminate_redundant=bool(d.get("eliminate_redundant", False)),
            variable_order=d.get("variable_order"),
        )
        for nd in d["nodes"]:
            node = EVDDNode.from_dict(nd)
            if node.node_id == TERMINAL_ID:
                raise ValueError("Internal node may not use the terminal id.")
            g.nodes[node.node_id] = node
            key = (node.var_level, g._branch_key(node.edges_0), g._branch_key(node.edges_1))
            g._unique[key] = node.node_id
            g._next_id = max(g._next_id, node.node_id + 1)
        g.root_edge = EVDDEdge.from_dict(d["root_edge"])
        if g.root_edge.target_id not in g.nodes:
            raise ValueError("root_edge points at an unknown node.")
        return g

    @classmethod
    def from_json(cls, text: str) -> "ndEVDDGraph":
        return cls.from_dict(json.loads(text))

    @classmethod
    def load_json(cls, path: str) -> "ndEVDDGraph":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_json(fh.read())

    # -- misc -------------------------------------------------------------- #
    def summary(self) -> str:
        kind = "deterministic" if self.is_deterministic else "non-deterministic"
        return (
            f"ndEVDDGraph(levels={self.num_levels}, nodes={self.num_nodes}, "
            f"edges={self.num_edges}, width={self.nodes_per_level()}, {kind}, "
            f"normalization={self.normalization!r})"
        )

    __repr__ = summary
