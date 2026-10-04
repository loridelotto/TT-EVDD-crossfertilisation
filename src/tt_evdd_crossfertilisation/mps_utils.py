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

def random_state_vector(n_qubits: int, seed: int = None) -> np.ndarray:
    """Generate a random state vector of 2**n complex amplitudes (complex128),

    sampled uniformly according to the Haar measure and normalized to unit norm.
    """
    rng = np.random.default_rng(seed)
    dim = 2**n_qubits

    # Real and imaginary parts sampled from a standard normal distribution N(0, 1)
    real_part = rng.normal(0.0, 1.0, size=dim)
    imag_part = rng.normal(0.0, 1.0, size=dim)
    psi = (real_part + 1j * imag_part).astype(np.complex128)

    # Normalize to ensure ||psi||_2 = 1
    psi /= np.linalg.norm(psi)
    return psi

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

def evaluate_truncation_error_fidelity(mps_exact: qtn.MatrixProductState,
                                       target_fidelity: Optional[float] = None,
                                       canonize: Optional[str] = "right",
                                       max_bond: Optional[int] = None
                                       )->Tuple[qtn.MatrixProductState, 
                                                Dict[str,Any]]:
    """
    Truncate by giving in input the fidelity and optionally the max_bond

    Parameters
    -----------
    - mps_exact: qtn.MatrixProductState
        the exact MPS that will be truncated
    - target_fidelity: float, optional
        the minimun fidelity to reach while truncating. It must be (0, 1]
    - canonize: std, optional
        "right", "left". (By default is "right")
    - max_bond: int, optional
        max chi value supported between tensors. None by default.

    ***
    # The math
    For this approach, we need a normalized MPS in such a way to use a sequence 
    of orthogonal projections. It's useful for the following relationship: if 
    we cut the weight at bond k (the sum of squared discarded singular values),
    the fidelity F = |<Psi|Psi_trunc>|^2 >= 1 - sum_k delta_k, by consequences, 
    delta_k <= (1 - fidelity_target)/(n-1) with n-1 bonds.
    """

    mps_trunc = mps_exact.copy()
    n = mps_trunc.L # length i.e. number of sites
    mps_trunc.normalize() # weigths must sum to 1

    if target_fidelity is not None:
        cut_value = (1-target_fidelity)/max(n-1, 1)
    else:
        cut_value = 0.0

    mps_trunc.compress(form=canonize, 
                       max_bond=max_bond,
                        cutoff=cut_value,
                        cutoff_mode="sum2")

    # apply the normalization without destroying the canon form:
    #   - if == left all the weights are inside the last tensor (the last dot)
    #     so I can just hard code it.
    #   - if == right on the most left tesnotr of the TT.
    if canonize == "left":
        mps_trunc.left_canonicalize(normalize=True, inplace=True) 
    if canonize == "right" or canonize is None:
        mps_trunc.right_canonicalize(normalize=True, inplace=True)

    overlap = mps_exact.H @ mps_trunc
    norm_exact_sq = np.real(mps_exact.H @ mps_exact)
    norm_trunc_sq = np.real(mps_trunc.H @ mps_trunc)
    
    fidelity = float(np.abs(overlap)**2 / (norm_exact_sq * norm_trunc_sq))

    metrics = {
        "exact_max_bond": mps_exact.max_bond(),
        "truncated_max_bond": mps_trunc.max_bond(),
        "exact_bonds": mps_exact.bond_sizes(),
        "truncated_bonds": mps_trunc.bond_sizes(),
        "target_reached": fidelity,
        "per_bond_cutoff": cut_value
    }

    return mps_trunc, metrics