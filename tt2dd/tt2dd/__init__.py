"""tt2dd -- bridge tensor trains (MPS) and non-deterministic edge-valued DDs.

Typical use::

    from tt2dd import TensorTrainState, convert_tt_to_evdd

    tt = TensorTrainState.from_numpy(cores)          # exact or truncated cores
    dd = convert_tt_to_evdd(tt, tolerance=1e-12)     # nd-EVDD (Quist et al.)
    dd.evaluate([0, 1, 1])                           # amplitude of |011>
    dd.to_json()                                     # hand-off to other tools
"""
from .structures import (
    TERMINAL_ID,
    EVDDEdge,
    EVDDNode,
    Normalization,
    TensorTrainState,
    ndEVDDGraph,
)
from .converter import check_equivalence, convert_cores_to_evdd, convert_tt_to_evdd

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "TERMINAL_ID",
    "EVDDEdge",
    "EVDDNode",
    "Normalization",
    "TensorTrainState",
    "ndEVDDGraph",
    "convert_tt_to_evdd",
    "convert_cores_to_evdd",
    "check_equivalence",
]
