# TT-EVDD-crossfertilisation

Cross-fertilisation between **tensor networks** and **decision diagrams** for quantum states.

A tensor train (TT, a.k.a. matrix product state, MPS) and a non-deterministic
edge-valued decision diagram (nd-EVDD) are two views of the same object
(Quist et al., *From Tensor Networks to Tractable Circuits, and back*,
arXiv:2605.00106). This repository collects the tools that move a state
between the two interpretations: build or truncate a TT, translate it to a decision
diagram, approximate the diagram by removing nodes, then verify and draw the
result.


Legend used below: ✔ works · ⚠ partial / unverified · ✗ missing.

## Concepts in one minute

- **TT / MPS**: a state on `n` qubits stored as `n` cores `A[i][s, b, t]` with
  bond dimensions `χ_i`. The amplitude of a bit string is the ordered product
  of the matrices `A[i][:, x_i, :]`. `χ` measures entanglement across a cut.
  See [`tt2dd/README.md`](tt2dd/README.md#quick-background-jargon).
- **EVDD**: a rooted DAG with one node per (variable, sub-function); each edge
  carries a complex weight and the amplitude of a bit string is the product of
  the weights along the path it selects. In a **deterministic** EVDD every node
  has at most one 0-edge and at most one 1-edge. In an **nd-EVDD** a node may
  have several 0-edges and several 1-edges and the amplitude is the *sum* over
  all consistent paths. A TT is exactly an nd-EVDD in which the level of core
  `i` has one node per value of that core's left bond index.
- **Approximation**: drop the nodes that contribute least to the norm so the
  diagram gets smaller while the fidelity `F = |⟨ψ|ψ'⟩|²` stays above a
  target. It plays the role that bond-dimension truncation plays for a TT; the
  Quist et al. paper notes that it is not yet known what truncation means on
  the DD side, which is what the proposed comparison experiment below is for.

## Repository map

| Path | What it is | Status |
|---|---|---|
| [`tt2dd/`](tt2dd/) | Standalone package: `TensorTrainState` → `convert_tt_to_evdd` → `ndEVDDGraph`, with JSON export. Linear-time construction of Quist et al. Sec. 3.1. Full API reference in [`tt2dd/README.md`](tt2dd/README.md). | ✔ converter, ✗ not tested yet|
| [`src/tt_evdd_crossfertilisation/hillmich_approx.py`](src/tt_evdd_crossfertilisation/hillmich_approx.py) | The repo's **deterministic** EVDD: dense state → EVDD (`from_state`), evaluation (`to_vector`), node contributions, `fidelity`, `size`, and the approximation of Hillmich et al. Sec. 4.3 (`approx_hillmich`). | ✔ (deterministic diagrams only) |
| [`src/tt_evdd_crossfertilisation/draw_dd.py`](src/tt_evdd_crossfertilisation/draw_dd.py) | Renders a deterministic EVDD as a TikZ picture (`to_tikz`) or a standalone LaTeX document (`to_document`). Drawing only, imports the diagram format from `hillmich_approx.py`. | ✔ (deterministic diagrams only) |
| [`src/tt_evdd_crossfertilisation/test_dicke.py`](src/tt_evdd_crossfertilisation/test_dicke.py) | Demo/experiment: Dicke state D(6,3), sweeps the target fidelity, asserts the guarantee `F ≥ f`, draws every distinct outcome. | ✔ (run as a module) |
| [`tests/dicke.tex`](tests/dicke.tex) | Committed output of the demo above (standalone LaTeX, needs `amsmath`, `tikz`). | output file |
| `mps_utils.py`, `quimb_intro.ipynb` (branch **`mps_utils`**, not merged) | quimb helpers: `create_ghz_state`, `create_high_entangled_state` (random brickwork circuit), `evaluate_truncation_error` (compress an MPS to a maximum bond dimension and report overlap, infidelity and norm distance). The notebook is exploratory work. | ⚠ unmerged |
| `pyproject.toml`, `uv.lock`, `.python-version` | uv project for `src/` (Python 3.12, numpy). | ⚠ template metadata |

## State convention

A state on n qubits is a numpy array of `2**n` complex128 amplitudes with norm 1.

The amplitude of the bit string `x_0 x_1 ... x_{n-1}` is stored at index

```
i = sum_k  x_k * 2**(n-1-k)
```

so site 0 is the most significant bit: the top of the decision diagram, and the first site of the MPS.

In Python:

```python
import numpy as np

psi = np.array([1, 0, 0, 1], dtype=complex)   # Bell state, n = 2
psi /= np.linalg.norm(psi)

np.save("bell.npy", psi)
psi = np.load("bell.npy")
```

Both code bases already follow this: `tt2dd` (`from_state_vector`,
`to_state_vector`, `evaluate`) and `hillmich_approx` (`from_state`,
`to_vector`) agree on it, and both number levels `0 … n-1` from the root with
terminal id `0`.

## Pipeline

The intended flow, with the state of every arrow:

```
 SOURCES
   OpenQASM/Qiskit circuit ──► exact TT                                     ✗ Not done yet
   mps_utils (quimb): GHZ / random brickwork ──► exact MPS           ⚠ Needs to be merged
     └─ evaluate_truncation_error: SVD-compress to χ_max + metrics   ⚠ Needs to be merged
   dense state ψ (2^n) ──► TensorTrainState.from_state_vector()      ✔ small test inputs only
                       └─► hillmich_approx.from_state()              ✔ deterministic EVDD baseline

        quimb MatrixProductState / plain numpy cores
                        │  TensorTrainState.from_numpy()             ✔
                        │  TensorTrainState.from_quimb()             ⚠ do not run on a real quimb MPS
                        ▼
                TensorTrainState                     tt2dd/structures.py
                        │  convert_tt_to_evdd()      tt2dd/converter.py      ✔ linear in TT size
                        ▼
                ndEVDDGraph  (nd-EVDD)               tt2dd/structures.py
                  │        │        │
                  │        │        └─ to_qsylvan_format() ─► Q-Sylvan fidelity check   ✗
                  │        └─ to_mqt_dd() ─► mqt.core / mqt.ddsim                        ✗
                  └─ to_json() / save_json()                                             ✔
                  │
                  │   ✗ ADAPTER IN PROGRESS
                  ▼
           (dd, root_edge)  deterministic EVDD       hillmich_approx.py
                  │  approx_hillmich(evdd, f)        ✔ node elimination, guarantee F ≥ f
                  ▼
           approximated EVDD ── size(), fidelity()   ✔ (fidelity goes through 2^n vectors)
                  │  draw_dd.to_tikz / to_document   ✔
                  ▼
                 .tex
```

### Two diagram formats 

| | `hillmich_approx` EVDD | `tt2dd` `ndEVDDGraph` |
|---|---|---|
| Representation | tuple `(dd, root_edge)`; `dd` is a dict of dicts (`level`, `edges_0`, `edges_1`: id → `(weight, child_id)`) plus a unique table | dataclasses `EVDDNode`, `EVDDEdge`; `nodes: Dict[int, EVDDNode]` and `root_edge` |
| Edges per branch | exactly one; a dead branch is `ZERO = (0, TERM)` | any number (0, 1 or several); several = non-determinism |
| Built from | a dense vector, cost `2^n` | TT cores, linear in the number of core entries |
| Weight normalisation | unit 2-norm; the larger-magnitude branch gets a real non-negative weight | `"l2"` (default): unit 2-norm, first weight real positive; also `"first"` and `"none"` |
| Tolerance | global constant `TOL = 1e-12` | per-graph `tolerance` |
| Size measure | `size()` = number of nodes | `num_nodes`, `num_edges`, `nodes_per_level()` |
| Serialisation | none | JSON (`to_json` / `from_json`) |

The two diagram formats are built with different purposes so we are keeping both formats and sharing the JSON format.

## How the pieces fit together

Each row is one interface between two parts, with what was actually checked.

| # | From → To | Interface | Status | Notes |
|---|---|---|---|---|
| 1 | TT source → `TensorTrainState` | `from_numpy(cores)`, `from_quimb(mps)` | ✔ numpy · ⚠ quimb | `from_quimb` is written against quimb's documented API but has not been run on a real `MatrixProductState`.|
| 2 | `TensorTrainState` → `ndEVDDGraph` | `convert_tt_to_evdd` | ✔ | Amplitudes match `tt.amplitude` on Bell, GHZ, W, random and truncated TTs; a 30-site χ=64 random TT converts in ~0.5 s. |
| 3 | `ndEVDDGraph` → `(dd, root_edge)` | — | ✗ | ADAPTER IS IN PROGRESS. See [the main integration gap](#the-main-integration-gap). |
| 4 | `ndEVDDGraph` → TikZ | — | ✗ | `draw_dd` assumes exactly one edge per branch. But should work once adapter is done|
| 5 | `ndEVDDGraph` → MQT / Q-Sylvan | `exporters.py` | ✗ | Not written;dropped (?) from the project for now|
| 6 | dense ψ → diagram, two routes | `from_state` vs `from_state_vector` + `convert_tt_to_evdd` | ✔ | Same nodes per level on Bell, GHZ-4, W-4, Dicke(6,3) and Dicke(8,4): a useful cross-check for tests. |
| 7 | `hillmich_approx` → `draw_dd` → `test_dicke` | direct imports | ✔ | `python -m tt_evdd_crossfertilisation.test_dicke` reproduces the committed `tests/dicke.tex` byte for byte. |
| 8 | `mps_utils` → `TensorTrainState` | quimb MPS | ⚠ | Same open point as row 1, plus the branch is not merged yet. |

### The main integration gap

`approx_hillmich` works on **deterministic** EVDDs; `tt2dd` produces
**non-deterministic** ones. 

## To-do

### A. Integration (needed for an end-to-end pipeline)

- [ ] **Merge branch `mps_utils`** and declare its dependencies in
  `pyproject.toml`. `mps_utils.py` imports `quimb`, which is not listed (only
  `numpy` is). The notebook additionally uses `networkx`.
- [ ] **Verify `TensorTrainState.from_quimb` on real quimb output**
  (`create_ghz_state`, `create_high_entangled_state`, and the truncated MPS
  returned by `evaluate_truncation_error`) and turn it into a regression test.
- [ ] **Write the adapter `ndEVDDGraph → (dd, root_edge)`** for deterministic
  diagrams. It must rebuild through `mk_node` (so the two normalisation
  conventions do not leak) and raise on non-deterministic input.
- [ ] **Draw nd-EVDDs**: extend `draw_dd.py` (or add a second drawer) to
  handle several edges per branch. Until then an nd diagram can only be
  inspected as JSON.

### B. Missing parts of the project brief

- [ ] **`to_mqt_dd` exporter** (`mqt.core` / `mqt.ddsim`).
- [ ] **`tt2dd/tests/test_converter.py`**: Bell/GHZ/W on 2–4 qubits, exhaustive
  amplitude equivalence against the TT, and truncated-TT inputs. The folder
  currently holds only `.gitkeep`.
- [ ] **Comparison experiment (proposition).** For one state, compare the TT route
  (truncate to `χ`, then `convert_tt_to_evdd`) with the DD route (`from_state`,
  then `approx_hillmich`): fidelity against size. Needs one shared size measure
  (edges or stored numbers rather than nodes) and one shared fidelity
  definition.

### C. Known issues in existing code

- [ ] **`from_state_vector` keeps numerically-zero singular values.** The
  default `cutoff=0.0` inflates the bond dimensions of "exact" TTs
  (Dicke(6,3): `[1,2,3,6,4,2,1]` instead of the Schmidt ranks
  `[1,2,3,4,3,2,1]`; Dicke(8,4): middle bond 16 instead of 5). The converter
  prunes them so the diagram is unchanged, but TT size and fan-out are larger
  than necessary. `cutoff=1e-12` fixes it: change the default, or document it,
  and correct the "minimal bond dimension" wording in `tt2dd/README.md`.


### D. Housekeeping

- [ ] **Replace the uv template leftovers**: `__init__.py` still contains
  `main()` printing "Hello from ...", wired as the console script
  `tt-evdd-crossfertilisation`, and the `pyproject.toml` description is still
  "Add your description here".
- [ ] **Unify the two Python projects.** The root uses `uv_build`, Python ≥ 3.12,
  numpy; `tt2dd/` uses setuptools, Python ≥ 3.10, numpy; there is no link
  between them, so installing the root does not install `tt2dd`. We should pick one
  layout: a uv workspace, a path dependency, or moving `tt2dd` under `src/`.

## Getting started

```bash
# root package (uv project, Python 3.12)
uv sync                      # or: pip install -e .

# TT -> nd-EVDD converter (separate package)
pip install -e ./tt2dd

# demo: Dicke state + Hillmich approximation sweep, writes dicke.tex
cd tests && python -m tt_evdd_crossfertilisation.test_dicke
```

Deterministic EVDD from a dense state, then approximate it:

```python
from tt_evdd_crossfertilisation.hillmich_approx import from_state, approx_hillmich, size, fidelity

evdd = from_state(psi)                 # psi: 2**n amplitudes, see State convention
app = approx_hillmich(evdd, f=0.8)     # guarantees fidelity(evdd, app) >= 0.8
print(size(evdd), size(app), fidelity(evdd, app))
```

Tensor train to nd-EVDD (details in [`tt2dd/README.md`](tt2dd/README.md)):

```python
from tt2dd import TensorTrainState, convert_tt_to_evdd, check_equivalence

tt = TensorTrainState.from_numpy(cores)        # cores[i].shape == (chi_{i-1}, 2, chi_i)
dd = convert_tt_to_evdd(tt)                    # ndEVDDGraph
assert check_equivalence(tt, dd)[0]            # dd.evaluate(bits) == tt.amplitude(bits)
dd.save_json("state.evdd.json")
```

## References

- A.-J. Quist, M. Farreras Bartra, A. de Colnet, J. van de Wetering, A. Laarman,
  *From Tensor Networks to Tractable Circuits, and back*, arXiv:2605.00106.
  TT ⇔ nd-EVDD (Sec. 3.1 is what `tt2dd` implements).
- Hillmich et al., Sec. 4.3, as cited in `hillmich_approx.py`: approximation of
  decision diagrams with a target fidelity.
