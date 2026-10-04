from typing import Optional, Tuple, Dict, Any

import numpy as np
import quimb as qu
import quimb.tensor as qtn

def convert_to_canonical_TT(psi: np.array,
                            canonize: Optional[str] = "right",
                            max_bond: Optional[int] = None,
                            cutoff: float = 0.0) -> qtn.MatrixProductState:
    """
    Convert a quantum state of complex amplitudes into a canonical Tensor Train

    - psi: np.array()
        vector of dtype=complex which represents the complex amplitude.
    - canonize: std, optional
        to decide in which of the possible way to canonize (right by default)
    - max_bond: int, optional
        to decide if approximate the TT while building it with SVD algorithm 
    """
    # converts any input into a contiguous array of complex data
    amplitude_vector = np.asarray(psi, dtype=complex).ravel()

    norm = np.linalg.norm(amplitude_vector)
    amplitude_vector = amplitude_vector / norm # normalization before transformation.

    # returns the number of bits to represent the integer size
    dimension = amplitude_vector.size.bit_length() - 1

    # converts into mps using quimb
    # dims is defined as [2]*dimesnion to create a list of 2s of dimension=dimension.
    mps = qtn.MatrixProductState.from_dense(amplitude_vector, dims=[2]*dimension,
                                  max_bond=max_bond, cutoff=cutoff)

    if canonize == "left":
        mps.left_canonicalize(normalize=True, inplace=True)
    elif canonize == "right" or canonize is None:
        mps.right_canonicalize(normalize=True, inplace=True)

    return mps

def create_ghz_state(n_qubits: int, canonize: Optional[str] = None,
                      max_bond: int = 2)->qtn.MatrixProductState:
    """
    Generates an n-qubit Greenberger-Horne-Zeilinger (GHZ) state:
    (|00...0> + |11...1>) / sqrt(2)
    
    The exact bond dimension across any bipartite cut is at most 2.
    """

    mps = qtn.MPS_computational_state(n_qubits * "0")
    mps.gate_(qu.hadamard(), where=0, contract=True)

    for i in range(n_qubits - 1):
        mps.gate_split(qu.CNOT(), where=(i, i+1), inplace=True, max_bond=max_bond)

    if canonize == "left":
        mps.left_canonicalize(normalize=True, inplace=True)
    if canonize == "right" or canonize is None:
        mps.right_canonicalize(normalize=True, inplace=True)
    
    return mps

def create_high_entangled_state(n_qubits:int,
                                max_bond: Optional[int],
                                canonize: Optional[str] = None,
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

    if canonize == "left":
        mps.left_canonicalize(normalize=True, inplace=True)
    if canonize == "right" or canonize is None:
        mps.right_canonicalize(normalize=True, inplace=True)
    return mps

def evaluate_truncation_error(
    mps_exact: qtn.MatrixProductState,
    max_bond: int,
    canonize: Optional[str] = None,
    cutoff: float = 0.0
) -> Tuple[qtn.MatrixProductState, Dict[str, Any]]:
    """
    Compresses an exact MPS to a target max_bond and calculates truncation metrics:
      - overlap: <psi_exact | psi_trunc>
      - fidelity: |<psi_exact | psi_trunc>|^2
      - norm_distance: ||psi_exact - psi_trunc||_2
    """
    mps_trunc = mps_exact.copy()
    mps_trunc.compress(max_bond=max_bond, cutoff=cutoff)

    # apply the normalization without destroying the canon form:
    #   - if == left all the weights are inside the last tensor (the last dot)
    #     so I can just hard code it.
    #   - if == right on the most left tesnotr of the TT.
    if canonize == "left":
        mps_trunc.left_canonicalize(normalize=True, inplace=True) 
    if canonize == "right" or canonize is None:
        mps_trunc.right_canonicalize(normalize=True, inplace=True)
                                                                

    overlap = mps_exact.H @ mps_trunc # inner prooduct between the stetes
    fidelity = float(np.abs(overlap) ** 2)
    norm_dist = float((mps_exact - mps_trunc).norm())

    metrics = {
        "target_max_bond": max_bond,
        "exact_max_bond": mps_exact.max_bond(),
        "truncated_max_bond": mps_trunc.max_bond(),
        "overlap": overlap,
        "fidelity": fidelity,
        "norm_distance": norm_dist,
        "exact_bonds": mps_exact.bond_sizes(),
        "truncated_bonds": mps_trunc.bond_sizes(),
    }
    return mps_trunc, metrics