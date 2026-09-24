# tt2dd — Tensor Trains → non-deterministic Edge-Valued Decision Diagrams

`tt2dd` is a small, Python package that translates a **tensor
train** (TT, a.k.a. matrix product state / MPS) into a **non-deterministic
edge-valued decision diagram** (nd-EVDD) in linear time, following the
work from the paper

> A.-J. Quist, M. Farreras Bartra, A. de Colnet, J. van de Wetering, A. Laarman,
> *From Tensor Networks to Tractable Circuits, and back*, arXiv:2605.00106, Sec. 3.1.

It is the bridge module of our project *"Bridging Tensor Networks and Decision
Diagrams"*: it accepts both exact TTs (extracted from OpenQASM circuits) and
SVD-truncated TTs, and emits one standardised nd-EVDD structure that can be inputted in the other parts of the project.

---

## Contents

1. [Pipeline](#pipeline)
2. [Quick Background Jargon](#quick-background-jargon)
3. [Installation](#installation)
4. [Quick start](#quick-start)
5. [Conventions](#conventions)
6. [Output format](#output-format)
7. [Module reference](#module-reference)
   - [`tt2dd/__init__.py`](#tt2dd__init__py)
   - [`tt2dd/structures.py`](#tt2ddstructurespy)
   - [`tt2dd/converter.py`](#tt2ddconverterpy)
8. [Algorithm notes](#algorithm-notes)
9. [Project layout](#project-layout)
10. [Roadmap](#roadmap)

---
## Quick background jargon
**Cores $A^{(i)}$** are 3-index tensors with entries `cores[i][s,b,t]`. They are essentially a pair of matrices per physical bit (`b`).\
**Amplitude** is essentially the ordered product of those matrices. `amplitude()` contracts the TT to one  bitstring.\
**Bond Dimension $\chi_i$** is the ize of the virtual index on the bond between site `i-1` and `i`. (Same as a Schmidt rank mathematically). This means that the minimal possible $\chi_i$ for a given $\Psi$ is its rank. `from_state_vector` computes this, it reshapes, takes the SVD keeps $k$ singular values and $k$ becomes $\chi_i$. So it can be a measure of entanglement across the cut, a product state has $\chi_i =1$

## Pipeline

```

 OpenQASM circuit ──► exact TT  SVD truncation ──► truncated TT
        │                                │
        └──────────────┬─────────────────┘
                       ▼
          TensorTrainState            (tt2dd/structures.py)
          list of cores (χ_{i-1}, 2, χ_i), site_order, norm
                       │
                       ▼
          convert_tt_to_evdd(tt)      (tt2dd/converter.py)
          bottom-up e construction + zero pruning
          + weight normalisation + hash-consing
                       │
                       ▼
          ndEVDDGraph                 (tt2dd/structures.py)
          root_edge, nodes{id → EVDDNode}, terminal
              │              │                 │
              ▼              ▼                 ▼
        to_json()      to_mqt_dd()     to_qsylvan_format()
        (node          (mqt.core /     (Q-Sylvan C++ verification)
         elimination)   mqt.ddsim)       
               [planned tt2dd/exporters.py(?)]
```

Verification loop used throughout: `check_equivalence(tt, dd)` compares
`dd.evaluate(bits)` with `tt.amplitude(bits)` for every (or a sample of) bit
string(s), and `dd.to_state_vector()` can be compared with the dense state.

---

## Installation

Python ≥ 3.10 and NumPy are the only hard requirements.

```bash
python -m venv .venv
.venv\Scripts\activate            #
pip install -e .                  # core package
pip install -e ".[test]"          # + pytest
pip install -e ".[quimb]"         # + quimb (for TensorTrainState.from_quimb)
pip install -e ".[mqt]"           # + mqt.core / mqt.ddsim (for the planned MQT exporter)
pip install -e ".[dev]"           # everything
```

`setup.py` is only a shim. All metadata lives in `pyproject.toml`.

---

## Quick start

```python
import numpy as np
from tt2dd import TensorTrainState, convert_tt_to_evdd, check_equivalence

# 1. Build (or receive) a tensor train.  Here: exact TT-SVD of a 3-qubit GHZ state.
psi = np.zeros(8, complex); psi[0] = psi[7] = 1 / np.sqrt(2)
tt = TensorTrainState.from_state_vector(psi)          # bond dims [1, 2, 2, 1]

#    ...or from raw cores handed over by another tool
# tt = TensorTrainState.from_numpy(cores, site_order=[0, 1, 2], norm=1.0)
#    ...or from quimb:
# tt = TensorTrainState.from_quimb(mps)

# 2. Convert.
dd = convert_tt_to_evdd(tt, tolerance=1e-12)          # ndEVDDGraph
print(dd.summary())
# ndEVDDGraph(levels=3, nodes=5, edges=6, width=[1, 2, 2], deterministic, normalization='l2')

# 3. Use / verify / export.
dd.evaluate([1, 1, 1])                                 # (0.7071+0j)
check_equivalence(tt, dd)                              # (True, 0.0)
np.allclose(dd.to_state_vector(), psi)                 # True
dd.save_json("ghz3.evdd.json")

# 4. A truncated TT goes through the very same path.
tt_trunc = tt.truncate(max_bond=1)
dd_trunc = convert_tt_to_evdd(tt_trunc)
```

---

## Conventions

These rules are shared by every object in the package; the JSON output follows
them too.

**Levels.** A diagram over `n` variables has levels `0 … n-1` for
nodes and level `n` for the single terminal. Level `i` corresponds to TT core
`i` and to variable `x_i`. Level 0 is the root.

**Bit strings are in level order.** `bits[i]` is the value of the variable at
level `i` (= TT core `i`). `variable_order[i]` / `site_order[i]` records which
original qubit that is, but is metadata only meaning our evaluation never looks at it.

**State vectors are big-endian ($q_0$ MSB) in level order.** The amplitude of
`x_0 x_1 … x_{n-1}` sits at flat index `Σ_i x_i · 2^(n-1-i)`. This holds for
`TensorTrainState.from_state_vector`, `TensorTrainState.to_state_vector` and
`ndEVDDGraph.to_state_vector`.

**Semantics of a diagram.** An edge `e = (w, v)` represents $w · f_v$. A node
`v` at level `l` with 0-edges `edges_0` and 1-edges `edges_1` represents\
    $f_v(x_l, …, x_{n-1}) = Σ_{e ∈ edges_{x_l}(v)}  w_e · f_{target}(e)(x_{l+1}, …, x_{n-1})$

and the terminal represents the constant 1. Likewise, `f(x)` is the sum
over all root→terminal paths consistent with `x` of the product of the edge
weights along the path. A branch may have **zero** edges (that branch is the
zero function) or **several** edges (non-determinism). An edge that skips `k`
levels which are only produced when `eliminate_redundant=True`, stands for `k`
implicit "don't care" nodes joined by weight-1 edges.

**Canonical form** (enforced by `ndEVDDGraph.make_node`):

1. Parallel edges to the same target are summed; weights with `|w| ≤ tolerance`
   are dropped; a node left with no edges collapses to the edge
   `(0, TERMINAL)`.
2. Within a branch, edges are sorted by `target_id`.
3. A scalar is pushed out of every node onto its incoming edge
   (`normalization="l2"` by default: outgoing weights have unit 2-norm and the
   first one is real-positive; `"first"`: first weight = 1; `"none"`: no
   normalisation).
4. Nodes are hash-consed on `(level, edges_0, edges_1)` with weights rounded to
   about `100 · tolerance`, so equal sub-diagrams are shared.

---

## Output format

`convert_tt_to_evdd` returns an `ndEVDDGraph`. Its interchange form is the
JSON produced by `to_json()` / `save_json()` and read by `from_json()` /
`load_json()`:

```json
{
  "format": "tt2dd.nd-evdd",
  "version": 1,
  "num_levels": 2,
  "variable_order": [0, 1],
  "tolerance": 1e-12,
  "normalization": "l2",
  "eliminate_redundant": false,
  "terminal_id": 0,
  "root_edge": {"weight": [0.7071, 0.0], "target": 3},
  "nodes": [
    {"id": 3, "level": 0,
     "edges_0": [{"weight": [0.7071, 0.0], "target": 1}],
     "edges_1": [{"weight": [0.7071, 0.0], "target": 2}]},
    {"id": 1, "level": 1, "edges_0": [{"weight": [1.0, 0.0], "target": 0}], "edges_1": []},
    {"id": 2, "level": 1, "edges_0": [], "edges_1": [{"weight": [1.0, 0.0], "target": 0}]}
  ]
}
```

| Field | Meaning |
|---|---|
| `format`, `version` | string and schema version |
| `num_levels` | number of variables `n` |
| `variable_order` | original site(/qubit) label of each level |
| `tolerance`, `normalization`, `eliminate_redundant` | settings used to build the diagram, so my function `reduce()` can remake it canonically |
| `terminal_id` | always `0`, for now the terminal is **not** listed under `nodes` |
| `root_edge` | the single entry edge, its weight carries the global scalar |
| `nodes` | nodes, sorted top-down by `(level, id)` where the ids are positive integers |
per node:
| `edges_0` / `edges_1` | lists of 0- and 1-edges, sorted by `target`, no duplicate targets, it can be empty or multi-element |
| `weight` | complex number saved as `[real, imag]` |
| `target` | id of a node on a lower level, or `0` for the terminal |

---

## Module reference

### `tt2dd/__init__.py`

Package entry point. Re-exports the public API so that everything can be
imported from `tt2dd` directly, and defines `__version__ = "0.1.0"`.

| Name | From |
|---|---|
| `TERMINAL_ID`, `Normalization`, `TensorTrainState`, `EVDDEdge`, `EVDDNode`, `ndEVDDGraph` | `structures` |
| `convert_tt_to_evdd`, `convert_cores_to_evdd`, `check_equivalence` | `converter` |

---

### `tt2dd/structures.py`

Data structures for both sides of the bridge. No knowledge of the
translation here, the module only validates, stores, evaluates,
canonicalises and serialises.

#### Module-level objects

| Object | Description |
|---|---|
| `TERMINAL_ID: int = 0` | Node id reserved for the unique terminal node. |
| `Normalization` | `Literal["l2", "first", "none"]` — the accepted normalisation rules. |
| `Bits` | Type alias `Sequence[int]` for bit strings. |
| `_rank_to_keep(s, max_bond, cutoff) -> int` | Private helper: number of singular values to keep given an absolute cap `max_bond` and a relative cutoff (`s_j > cutoff · s_0`); always ≥ 1. Used by `from_state_vector` and `truncate`. |

#### `class TensorTrainState`

A tensor train with binary physical indices and open boundaries.

**Intended use:** the single input type of the converter. We input cores and this class validates them and gives a few
contraction helpers for testing. It deliberately does not know how the cores
were produced (exact vs. truncated).

Constructor `TensorTrainState(cores, site_order=None, norm=1.0, copy=True)`

- `cores` — sequence of arrays. Core `i` has shape `(χ_{i-1}, 2, χ_i)`.
  Boundary cores may be passed as 2-D arrays `(2, χ_1)` and `(χ_{n-1}, 2)` and
  are reshaped. Checks: 3-D after reshaping, physical dimension 2, bond
  dimensions chain, `χ_0 = χ_n = 1`. Cores are cast to `complex128`.
- `site_order` — which original qubit / site each core acts on (default
  `range(n)`). Metadata only; propagated to the diagram's `variable_order`.
- `norm` — global scalar prefactor; the represented function is
  `norm · contraction(cores)`.
- `copy` — copy the cores (`True`) or keep the arrays passed in.

Attributes: `cores: List[np.ndarray]`, `site_order: List[int]`, `norm: complex`.

| Member | Description |
|---|---|
| `from_numpy(cores, site_order=None, norm=1.0)` *(classmethod)* | Thin alias of the constructor for plain NumPy input. |
| `from_quimb(mps, site_order=None)` *(classmethod)* | Build from a `quimb.tensor.MatrixProductState`-like object. Uses `mps.L`, `mps[i]`, `mps.site_ind(i)` and `mps.bond(i, i+1)` and transposes every tensor to `(left bond, physical, right bond)`, so it is independent of quimb's internal index order. Cyclic MPS are rejected. *(quimb is not installed in this environment; untested.)* |
| `from_state_vector(psi, max_bond=None, cutoff=0.0, site_order=None)` *(classmethod)* | TT-SVD of a dense length-`2^n` vector (big-endian). With `max_bond` / `cutoff` it produces a truncated TT directly. Mainly for building benchmark and test inputs. |
| `n_sites` *(property)* | Number of cores / qubits. |
| `bond_dims` *(property)* | `[χ_0, χ_1, …, χ_n]` including both boundary 1s. |
| `max_bond_dim` *(property)* | `max(bond_dims)`. |
| `num_parameters` *(property)* | Total number of stored core entries — the "size" of the TT, the quantity the conversion is linear in. |
| `amplitude(bits) -> complex` | Contract the TT on one bit string (level order). O(Σ χ²). |
| `to_state_vector() -> np.ndarray` | Full contraction to a dense `2^n` vector. Exponential; for small tests. |
| `truncate(max_bond=None, cutoff=0.0) -> TensorTrainState` | Return a new TT with reduced bond dimensions: right-canonicalise by a QR sweep, then a left-to-right SVD sweep keeping at most `max_bond` Schmidt values above `cutoff · s_max` per bond. This essentially does the SVD truncation but is only used for testing purposes. |
| `__repr__` | `TensorTrainState(n_sites=…, bond_dims=[…], norm=…)`. |

#### `class EVDDEdge` *(frozen dataclass, slots)*

A weighted edge `(weight: complex, target_id: int)` representing
`weight · f_target`.

| Member | Description |
|---|---|
| `is_zero(tolerance=0.0) -> bool` | `abs(weight) <= tolerance`. |
| `scaled(factor) -> EVDDEdge` | New edge with `weight · factor`, same target. |
| `to_dict()` / `from_dict(d)` | `{"weight": [re, im], "target": id}` ↔ edge. |

Being frozen makes edges hashable and safe to share between nodes.

#### `class EVDDNode` *(dataclass, slots)*

An internal node: `node_id`, `var_level`, `edges_0: Tuple[EVDDEdge, ...]`,
`edges_1: Tuple[EVDDEdge, ...]`. The tuples are the complete sets of 0-edges
and 1-edges (this is where non-determinism lives; a deterministic EVDD has at
most one edge per tuple). Inside a graph they are sorted by target and free
of duplicate targets. The terminal is also stored as an `EVDDNode` with
`node_id = 0`, `var_level = n` and empty edge tuples.

| Member | Description |
|---|---|
| `branch(x) -> Tuple[EVDDEdge, ...]` | `edges_1` if `x` else `edges_0`. |
| `is_terminal` *(property)* | `node_id == TERMINAL_ID`. |
| `out_degree` *(property)* | `len(edges_0) + len(edges_1)`. |
| `to_dict()` / `from_dict(d)` | `{"id", "level", "edges_0": [...], "edges_1": [...]}` ↔ node. |

#### `class ndEVDDGraph`

Container for an ordered nd-EVDD together with its unique table
(hash-consing), evaluation, reduction and serialisation.

**Intended use:** the single output type of the converter. New nodes should only ever be created through
`make_node`, which guarantees the canonical form, after manual surgery call
`reduce()` to restore it.

Constructor `ndEVDDGraph(num_levels, tolerance=1e-12, normalization="l2", eliminate_redundant=False, variable_order=None)`

- `num_levels` — number of variables `n` (≥ 1).
- `tolerance` — weights with `|w| ≤ tolerance` are treated as zero; weights are
  quantised to ~`100 · tolerance` in the unique-table key.
- `normalization` — `"l2"` (unit 2-norm of outgoing weights, first weight
  real-positive; numerically robust, MQT-like), `"first"` (first non-zero
  weight = 1; classic EVDD/QMDD) or `"none"`.
- `eliminate_redundant` — if `True`, a node whose 0- and 1-branch are the same
  single edge is not created; the parent points at the child directly (fully
  reduced, level-skipping). Default `False` keeps the diagram quasi-reduced
  (every path visits every level), matching MQT and the paper's construction.
- `variable_order` — original qubit label per level (default `range(n)`).

Attributes: `num_levels`, `tolerance`, `normalization`, `eliminate_redundant`,
`variable_order: List[int]`, `nodes: Dict[int, EVDDNode]` (terminal included
under id 0), `root_edge: EVDDEdge` (initially the zero edge `(0, TERMINAL)`).
Private: `_unique` (key → node id), `_next_id`, `_key_decimals`.

| Member | Description |
|---|---|
| `terminal` *(property)* | The terminal `EVDDNode`. |
| `root` *(property)* | The node `root_edge` points at. |
| `num_nodes` *(property)* | Number of internal nodes (terminal excluded). |
| `num_edges` *(property)* | Total number of edges over all nodes. |
| `nodes_per_level() -> List[int]` | Internal node count per level — the DD analogue of the bond dimensions. |
| `max_width` *(property)* | `max(nodes_per_level())`. |
| `is_deterministic` *(property)* | `True` iff every branch has ≤ 1 edge. |
| `make_node(var_level, edges_0, edges_1) -> EVDDEdge` | **The node constructor.** Canonicalises both branches (merge parallel edges, drop zeros, sort), validates that targets exist on lower levels, returns `(0, TERMINAL)` if both branches are empty, optionally skips a redundant node, pushes out the normaliser `c`, hash-conses, and returns the edge `(c, node_id)` that represents exactly the requested node. |
| `evaluate(bits) -> complex` | Value of the function on a bit string (level order); memoised traversal, linear in the graph size. Handles level-skipping edges. |
| `to_state_vector() -> np.ndarray` | Dense `2^n` vector, big-endian in level order; exponential, for tests. Level-skipping edges are expanded with `kron(ones, ·)`. |
| `reachable_ids() -> List[int]` | Ids of all nodes reachable from the root (terminal included). |
| `reduce(**overrides) -> ndEVDDGraph` | Rebuild bottom-up through `make_node` into a **new** graph: drops unreachable nodes, merges parallel edges, drops zeros, re-normalises and re-shares. `overrides` may change `tolerance`, `normalization`, `eliminate_redundant`. Use after node elimination or after loading un-normalised JSON. |
| `to_dict() -> dict` / `to_json(indent=2) -> str` / `save_json(path)` | Serialise to the [output format](#output-format). |
| `from_dict(d)` / `from_json(text)` / `load_json(path)` *(classmethods)* | Inverse of the above. Nodes are loaded verbatim (not re-normalised) and the unique table is rebuilt from them. |
| `summary() -> str` (also `__repr__`) | One-line description: levels, nodes, edges, width per level, deterministic or not, normalisation. |

Private helpers: `_canonical_branch(edges)` (merge / prune / sort),
`_branch_key(edges)` (quantised hashing key), `_normalizer(edges_0, edges_1)`
(the scalar `c` for the chosen rule; never zero).

---

### `tt2dd/converter.py`

The translation itself plus a verification helper.

#### `convert_tt_to_evdd(tt, tolerance=1e-12, normalization="l2", eliminate_redundant=False) -> ndEVDDGraph`

**Intended use:** the main input point of the tool

Implements Quist et al. Sec. 3.1: one node `v_s^(r)` per incoming bond index
`s` of core `r`, and one `b`-edge `v_s^(r) → v_t^(r+1)` of weight
`A^(r)[s, b, t]` for each outgoing bond index `t`. Cores are processed from
the **last to the first** so that every node's children already exist when
the node is created (bottom-up hash-consing):

```
below = [edge(1, TERMINAL)]                      # χ_n = 1
for level = n-1 … 0:
    for each row s of core[level]:
        for b in {0, 1}:
            edges_b = [ (A[s,b,t] · below[t].weight, below[t].target)  for t with |·| > tolerance ]
        current[s] = graph.make_node(level, edges_0, edges_1)
    below = current
root_edge = below[0] · tt.norm                   # χ_0 = 1
```

Every core entry is touched once → `O(Σ_r χ_{r-1} · 2 · χ_r)`, linear
in the size of the TT. Because `make_node` returns edges that already carry
the pushed-out scalar, bond indices whose sub-diagrams turn out to be equal
map to the same node, and the parallel edges this creates are summed.

Parameters: `tolerance` (zero threshold and node-sharing precision; raise it,
e.g. to `1e-8`, for noisy truncated inputs), `normalization` and
`eliminate_redundant` as in `ndEVDDGraph`. The result has `tt.n_sites` levels
and `variable_order = tt.site_order`.

#### `convert_cores_to_evdd(cores, site_order=None, norm=1.0, **kwargs) -> ndEVDDGraph`

Convenience wrapper: wraps raw NumPy cores in a `TensorTrainState` and calls
`convert_tt_to_evdd(**kwargs)`.

#### `check_equivalence(tt, graph, bitstrings=None, num_samples=256, atol=1e-9, rtol=1e-9, rng=None) -> (bool, float)`

**Intended use:** Cross-checking the Q-sylvan results

Compares `graph.evaluate(bits)` with `tt.amplitude(bits)`. If `bitstrings`
is not given, all `2^n` strings are checked for `n ≤ 12`, otherwise
`num_samples` uniformly random strings (from `rng`). A string passes when
`|dd − tt| ≤ atol + rtol · |tt|`. Returns `(all_pass, max_abs_error)`.

---

## Notes

- **Bottom-up.** Hash-consing needs children before parents, processing
  cores from the right also means the scalar pushed out of a node is
  immediately absorbed into the core entries of the level above, so no second
  pass is needed.
- **What gets reduced.** (I) Core entries below `tolerance` never become
  edges. (II) A row `A^(r)[s, :, :]` that is entirely zero
  becomes the zero edge and produces no node. (III) Two rows that are equal up
  to a scalar produce one shared node. (IV) Wrong singular values from an
  SVD (e.g. `1e-17`) are pruned, which is why an exact TT-SVD of a W state
  with bond dims `[1, 2, 3, 2, 1]` gives us a diagram of width `[1, 2, 2, 2]`.
- **Truncated inputs.** Nothing in the converter depends on canonical form or
  on how the cores were produced, a truncated TT simply has smaller `χ` and
  gives a narrower diagram that represents the truncated state exactly.


---

## Project layout

`tt2dd/` is a self-contained sub-project inside the
`TT-EVDD-crossfertilisation` repository; the importable package is the inner
`tt2dd/tt2dd/`.

```
TT-EVDD-crossfertilisation/     # repository root
├── src/tt_evdd_crossfertilisation/   # other sub-projects (state -> EVDD, Hillmich, ...)
├── pyproject.toml                    # repository-level project file
└── tt2dd/                      # <- this sub-project
    ├── README.md               # this file
    ├── pyproject.toml          # package metadata, dependencies, extras, pytest config
    ├── setup.py                # shim for legacy tooling
    ├── .gitignore
    ├── tt2dd/                  # <- the importable package
    │   ├── __init__.py         # public API re-exports, __version__
    │   ├── structures.py       # TensorTrainState, EVDDEdge, EVDDNode, ndEVDDGraph
    │   └── converter.py        # convert_tt_to_evdd, convert_cores_to_evdd, check_equivalence
    └── tests/                  # pytest suite (Task 4, to be written)
```

Install it with `pip install -e .` from the `tt2dd/` directory; nothing outside
that directory is needed to build or run the package.

The state-vector convention used here (big-endian, site 0 = most significant
bit = top of the diagram = first MPS site) is the same one the repository
README specifies, so dense states can be exchanged with the other
sub-projects without reindexing.

---

## Roadmap

| Task | Status |
|---|---|
| 1. Core data structures (`tt2dd/structures.py`) | done |
| 2. Constructive converter (`tt2dd/converter.py`) | done |
| 3. Exporters (`tt2dd/exporters.py`): `to_mqt_dd` (mqt.core / mqt.ddsim handle), `to_qsylvan_format` (C-readable text for Q-Sylvan) | To-Do |
| 4. Test suite (`tests/test_converter.py`): benchmark states, exhaustive equivalence, truncated-TT compatibility | To-Do |
