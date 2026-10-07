"""Write the test states of the pipeline as .npy files next to this script.

Every file holds 2**n complex amplitudes with norm 1, site 0 = most
significant bit (see "State convention" in the README). Run it with:

    uv run python states/make_states.py

then feed a file to the pipeline:

    uv run tt-evdd-pipeline states/dicke_5_2.npy
"""

import itertools
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def basis_index(ones, n):
    """Index of the bit string with ones at the given sites."""
    return sum(1 << (n - 1 - b) for b in ones)


def dicke(n, k):
    """Uniform superposition of every bit string of Hamming weight k."""
    psi = np.zeros(2 ** n, dtype=complex)
    for ones in itertools.combinations(range(n), k):
        psi[basis_index(ones, n)] = 1
    return psi


def ghz(n):
    """(|0...0> + |1...1>) / sqrt(2)."""
    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = psi[-1] = 1
    return psi


def w(n):
    """Dicke state with a single excitation."""
    return dicke(n, 1)


def grover(n, marked, iterations=None):
    """Grover's search on n qubits after the given number of iterations
    (default: half of the optimal one, so that the marked string and the
    rest have comparable weight). With theta = arcsin(1/sqrt(N)) the
    marked amplitude is sin((2k+1) theta), every other one
    cos((2k+1) theta) / sqrt(N-1).
    """
    N = 2 ** n
    theta = np.arcsin(1 / np.sqrt(N))
    if iterations is None:
        iterations = int(np.pi / (4 * theta)) // 2
    angle = (2 * iterations + 1) * theta
    psi = np.full(N, np.cos(angle) / np.sqrt(N - 1), dtype=complex)
    psi[marked] = np.sin(angle)
    return psi


def random_state(n, seed):
    """Haar-random state."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)


STATES = {
    "dicke_5_2": dicke(5, 2),
    "dicke_6_3": dicke(6, 3),
    "ghz_5": ghz(5),
    "w_5": w(5),
    "random_5": random_state(5, seed=1),
    "random_6": random_state(6, seed=1),
    "grover_12": grover(12, marked=0b101101011010),
    "dicke_12_6": dicke(12, 6),
}


if __name__ == "__main__":
    for name, psi in STATES.items():
        path = HERE / f"{name}.npy"
        np.save(path, psi / np.linalg.norm(psi))
        print(f"wrote {path.name} ({psi.size} amplitudes)")
