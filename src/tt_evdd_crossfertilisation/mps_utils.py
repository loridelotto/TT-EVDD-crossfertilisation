from typing import Optional, Tuple, Dict, Any

import numpy as np
import quimb as qu
import quimb.tensor as qtn

def create_ghz_state(n_qubits: int, max_bond: int = 2)->qtn.MatrixProductState:
    """
    Generates an n-qubit Greenberger-Horne-Zeilinger (GHZ) state:
    (|00...0> + |11...1>) / sqrt(2)
    
    The exact bond dimension across any bipartite cut is at most 2.
    """

    mps = qtn.MPS_computational_state(n_qubits * "0")
    mps.gate_(qu.hadamard(), where=0, contract=True)

    for i in range(n_qubits - 1):
        mps.gate_split(qu.CNOT(), where=(i, i+1), inplace=True, max_bond=max_bond)

    mps.normalize()
    return mps

def create_high_entangled_state(n_qubits:int,
                                max_bond: Optional[int],
                                depth: int = 6,
                                seed:Optional[int] = None
                                ) ->qtn.MatrixProductState:
    """
    Builds a random brickwork circuit generating volume-law entanglement.
    Calculated exactly if max_bond is not defined.
    """

    if seed is not None:
        np.random.seed(seed)

    mps = qtn.MPS_computational_state(n_qubits * "0")

    for d in range(depth):
        offset = d % 2
        for i in range(offset, n_qubits - 1, 2):
            u_gate = qu.rand_uni(4).reshape(2,2,2,2)
            mps.gate_(u_gate, where=(i, i+1), contract="split", 
                      max_bond=max_bond, cutoff=0.0)

    mps.normalize()
    return mps

def evaluate_truncation_error(
    mps_exact: qtn.MatrixProductState,
    max_bond: int,
    cutoff: float = 0.0
) -> Tuple[qtn.MatrixProductState, Dict[str, Any]]:
    """
    Compresses an exact MPS to a target max_bond and calculates truncation metrics:
      - overlap: <psi_exact | psi_trunc>
      - infidelity: 1 - |<psi_exact | psi_trunc>|^2
      - norm_distance: ||psi_exact - psi_trunc||_2
    """
    mps_trunc = mps_exact.copy()
    mps_trunc.compress(max_bond=max_bond, cutoff=cutoff)
    mps_trunc.normalize()

    overlap = mps_exact.H @ mps_trunc # inner prooduct between the stetes
    infidelity = float(1.0 - np.abs(overlap) ** 2)
    norm_dist = float((mps_exact - mps_trunc).norm())

    metrics = {
        "target_max_bond": max_bond,
        "exact_max_bond": mps_exact.max_bond(),
        "truncated_max_bond": mps_trunc.max_bond(),
        "overlap": overlap,
        "infidelity": infidelity,
        "norm_distance": norm_dist,
        "exact_bonds": mps_exact.bond_sizes(),
        "truncated_bonds": mps_trunc.bond_sizes(),
    }
    return mps_trunc, metrics