"""Benchmark circuits from the literature (Qiskit) and their state vectors.

Run once to write the input states of the pipeline::

    python -m tt_evdd_crossfertilisation.benchmark_circuits benchmarks/

For every benchmark this writes

* ``<name>.npy``  the state vector: complex128, length 2**n, norm 1, qubit 0 = most
  significant bit (the convention of the repository README; Qiskit's own vector is
  little endian, the conversion is done here), and
* ``<name>.txt``  the same vector as text, one amplitude per line as ``real imag``
  (index = sum_k x_k 2**(n-1-k)); only up to ``--text-max-qubits`` qubits (default 16)
  because text files get large.

Defaults come from the papers we use:

* Hillmich et al.: Google random circuits ``rqc_AxB_C_D`` (4x4 and 4x5 grids) and Shor's
  algorithm ``shor_N_a``.
* Q-Sylvan / FTDD (MQT Bench): GHZ, graph state, QFT, entangled QFT, Deutsch-Jozsa, QPE
  (exact / inexact), W state, QAOA and the VQE ansaetze, rebuilt with Qiskit from MQT
  Bench's own definitions (seeds included).
* This repository's tests: Bell, GHZ, W and Dicke states.

Every circuit function below returns a Qiskit ``QuantumCircuit``; use it directly, or
``circuit_state(qc)`` to get the vector.

Limits: the Google circuits are generated with the published rules, not the original GRCS
instance files (read those with ``load_grcs``).  Shor is emulated (modular exponentiation as a
permutation), so it has no circuit.  MQT Bench's graph state is unseeded; here it is seeded.
"""
from __future__ import annotations

import argparse
import functools
import itertools
import math
import random
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import ParameterVector
    from qiskit.circuit.library import (GraphStateGate, QFTGate, efficient_su2, iSwapGate,
                                        n_local, real_amplitudes)
    from qiskit.quantum_info import Statevector
    from qiskit.synthesis import synth_qft_full
except ImportError as exc:  # pragma: no cover
    raise ImportError("benchmark_circuits builds its circuits with Qiskit: pip install qiskit") from exc

SEED = 10   # the seed MQT Bench uses everywhere
_BASIS = ["h", "x", "y", "z", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "p", "cx", "cz", "cp", "swap"]


# --------------------------------------------------------------------------- #
# Circuit -> state vector                                                     #
# --------------------------------------------------------------------------- #
def circuit_state(qc: QuantumCircuit, max_qubits: int = 26) -> np.ndarray:
    """State vector of ``qc`` applied to ``|0...0>``: complex128, norm 1, qubit 0 = MSB."""
    n = qc.num_qubits
    if n > max_qubits:
        raise ValueError(f"{n} qubits need a dense vector of {16 * 2.0 ** n / 2**30:.1f} GiB "
                         f"(limit {max_qubits})")
    # Decompose first: Statevector would otherwise build the full 2**n x 2**n matrix of a QFTGate.
    plain = transpile(qc.remove_final_measurements(inplace=False), basis_gates=_BASIS, optimization_level=0)
    vec = Statevector(plain).data.reshape((2,) * n).transpose(list(range(n - 1, -1, -1))).reshape(-1)
    return np.ascontiguousarray(vec / np.linalg.norm(vec))


# --------------------------------------------------------------------------- #
# MQT Bench circuits, rebuilt with Qiskit                                     #
# --------------------------------------------------------------------------- #
def bell() -> QuantumCircuit:
    qc = QuantumCircuit(2, name="bell")
    qc.h(0)
    qc.cx(0, 1)
    return qc


def ghz(n: int = 10) -> QuantumCircuit:
    qc = QuantumCircuit(n, name="ghz")
    qc.h(n - 1)
    for i in range(1, n):
        qc.cx(n - i, n - i - 1)
    return qc


def wstate(n: int = 10) -> QuantumCircuit:
    qc = QuantumCircuit(n, name="wstate")
    qc.x(n - 1)
    for m in range(1, n):
        theta = math.acos(math.sqrt(1 / (n - m + 1)))
        qc.ry(-theta, n - m - 1)
        qc.cz(n - m, n - m - 1)
        qc.ry(theta, n - m - 1)
    for k in reversed(range(1, n)):
        qc.cx(k - 1, k)
    return qc


def _random_regular_graph(n: int, degree: int, rng: np.random.Generator) -> List[Tuple[int, int]]:
    if degree >= n or (n * degree) % 2:
        raise ValueError(f"no {degree}-regular graph on {n} vertices")
    for _ in range(10000):
        stubs = np.repeat(np.arange(n), degree)
        rng.shuffle(stubs)
        edges = {tuple(sorted((int(a), int(b)))) for a, b in stubs.reshape(-1, 2)}
        if len(edges) == n * degree // 2 and all(a != b for a, b in edges):
            return sorted(edges)
    raise RuntimeError("could not sample a regular graph")


def graphstate(n: int = 10, degree: int = 2, seed: int = SEED) -> QuantumCircuit:
    adjacency = np.zeros((n, n))
    for a, b in _random_regular_graph(n, degree, np.random.default_rng(seed)):
        adjacency[a, b] = adjacency[b, a] = 1
    qc = QuantumCircuit(n, name="graphstate")
    qc.compose(GraphStateGate(adjacency), inplace=True)
    return qc


def qft(n: int = 10) -> QuantumCircuit:
    qc = QuantumCircuit(n, name="qft")
    qc.compose(QFTGate(num_qubits=n), inplace=True)
    return qc


def qftentangled(n: int = 10) -> QuantumCircuit:
    qc = QuantumCircuit(n, name="qftentangled")
    qc.h(n - 1)
    for i in range(1, n):
        qc.cx(n - i, n - i - 1)
    qc.compose(QFTGate(num_qubits=n), inplace=True)
    return qc


def dj(n: int = 10, balanced: bool = True) -> QuantumCircuit:
    """Deutsch-Jozsa on ``n`` qubits in total (``n-1`` inputs and one ancilla)."""
    m = n - 1
    rng = np.random.default_rng(SEED)
    qc = QuantumCircuit(n, name="dj")
    qc.x(m)
    qc.h(m)
    qc.h(range(m))
    if balanced:
        bits = [int(rng.integers(0, 2)) for _ in range(m)]
        for q in range(m):
            if bits[q]:
                qc.x(q)
        for q in range(m):
            qc.cx(q, m)
        for q in range(m):
            if bits[q]:
                qc.x(q)
    elif int(rng.integers(2)) == 1:
        qc.x(m)
    qc.h(range(m))
    return qc


def _qpe(n: int, exact: bool) -> QuantumCircuit:
    m = n - 1                                        # counting qubits; qubit m holds the eigenstate
    random.seed(SEED)
    theta = 0
    if exact:
        while theta == 0:
            theta = random.getrandbits(m)
        indices, mask = range(m), (lambda i: 1 << (m - i - 1))
    else:
        while theta == 0 or (theta & 1) == 0:
            theta = random.getrandbits(m + 1)
        indices, mask = range(m + 1), (lambda i: 1 << (m - i))
    lam = Fraction(0, 1)
    for i in indices:
        if theta & mask(i):
            lam += Fraction(1, 1 << i)
    qc = QuantumCircuit(n, name="qpeexact" if exact else "qpeinexact")
    qc.x(m)
    qc.h(range(m))
    for i in range(m):
        angle = (lam * (1 << i)) % 2
        if angle > 1:
            angle -= 2
        if angle != 0:
            qc.cp(float(angle) * math.pi, m, i)
    qc.compose(synth_qft_full(num_qubits=m, inverse=True), qubits=list(range(m)), inplace=True)
    return qc


def qpeexact(n: int = 10) -> QuantumCircuit:
    return _qpe(n, True)


def qpeinexact(n: int = 10) -> QuantumCircuit:
    return _qpe(n, False)


def _bind_random(qc: QuantumCircuit, seed: int = SEED) -> QuantumCircuit:
    """Bind every parameter to uniform(0, 2 pi), drawn in ``qc.parameters`` order (as MQT Bench)."""
    rng = np.random.default_rng(seed)
    return qc.assign_parameters({p: float(rng.uniform(0, 2 * np.pi)) for p in qc.parameters})


def qaoa(n: int = 10, reps: int = 2, seed: int = SEED) -> QuantumCircuit:
    """MaxCut QAOA on a random graph (MQT Bench's construction)."""
    adjacency = np.triu(np.random.default_rng(seed).integers(0, 2, size=(n, n)), 1)
    gamma, beta = ParameterVector("g", reps), ParameterVector("b", reps)
    qc = QuantumCircuit(n, name="qaoa")
    qc.h(range(n))
    for layer in range(reps):
        for i in range(n):
            for j in range(i + 1, n):
                if adjacency[i, j] != 0:
                    qc.rzz(2 * gamma[layer], i, j)
        for q in range(n):
            qc.rx(2 * beta[layer], q)
    return _bind_random(qc)


def vqe_real_amp(n: int = 10, reps: int = 3, entanglement: str = "reverse_linear") -> QuantumCircuit:
    qc = real_amplitudes(n, entanglement=entanglement, reps=reps)
    qc.name = "vqe_real_amp"
    return _bind_random(qc)


def vqe_su2(n: int = 10, reps: int = 3, entanglement: str = "reverse_linear") -> QuantumCircuit:
    qc = efficient_su2(n, entanglement=entanglement, reps=reps)
    qc.name = "vqe_su2"
    return _bind_random(qc)


def vqe_two_local(n: int = 10, reps: int = 3, entanglement: str = "full") -> QuantumCircuit:
    qc = n_local(n, rotation_blocks="ry", entanglement_blocks="cx", entanglement=entanglement, reps=reps)
    qc.name = "vqe_two_local"
    return _bind_random(qc)


# --------------------------------------------------------------------------- #
# Google random quantum circuits                                              #
# --------------------------------------------------------------------------- #
def _rqc_layers(rows: int, cols: int) -> List[List[Tuple[int, int]]]:
    """Eight CZ layers; horizontal and vertical ones alternate (as in GRCS ``cz_v2``)."""
    q = lambda r, c: r * cols + c
    horizontal, vertical = [], []
    for a, b in [(0, 0), (1, 1), (1, 0), (0, 1)]:
        horizontal.append([(q(r, c), q(r, c + 1)) for r in range(rows) for c in range(cols - 1)
                           if c % 2 == a and r % 2 == b])
        vertical.append([(q(r, c), q(r + 1, c)) for r in range(rows - 1) for c in range(cols)
                         if r % 2 == a and c % 2 == b])
    return [layer for pair in zip(horizontal, vertical) for layer in pair]


def rqc(rows: int = 4, cols: int = 4, depth: int = 10, instance: int = 0) -> QuantumCircuit:
    """A random circuit built with the rules of the Google (GRCS ``cz_v2``) circuits.

    Hadamards at cycle 0 and at cycle ``depth``; cycles ``1..depth-1`` apply one CZ layer
    each, then a random non-diagonal gate (X^1/2 or Y^1/2, never the same twice in a row) on
    idle qubits that just had a CZ, and a T on idle qubits after a non-diagonal gate.  The
    layer order of the original instances is not reproduced; see ``load_grcs`` for those.
    Qubits are numbered row-major.
    """
    n = rows * cols
    rng = np.random.default_rng([rows, cols, depth, instance])
    layers = _rqc_layers(rows, cols)
    qc = QuantumCircuit(n, name=f"rqc_{rows}x{cols}_{depth}_{instance}")
    qc.h(range(n))
    last = ["h"] * n                      # "h", "cz", "nd" (non-diagonal) or "t"
    last_nd: List[Optional[str]] = [None] * n
    for cycle in range(1, depth):
        busy = set()
        for a, b in layers[(cycle - 1) % 8]:
            qc.cz(a, b)
            busy.update((a, b))
            last[a] = last[b] = "cz"
        for q in range(n):
            if q in busy:
                continue
            if last[q] == "cz":
                options = [g for g in ("x_1_2", "y_1_2") if g != last_nd[q]]
                g = options[int(rng.integers(len(options)))]
                (qc.rx if g == "x_1_2" else qc.ry)(math.pi / 2, q)
                last[q], last_nd[q] = "nd", g
            elif last[q] in ("h", "nd"):
                qc.t(q)
                last[q] = "t"
    qc.h(range(n))
    return qc


def load_grcs(path: str) -> QuantumCircuit:
    """Read an original GRCS instance file (``inst_4x4_10_0.txt``): first line = number of
    qubits, then ``cycle gate qubit [qubit]`` with gates h, t, x_1_2, y_1_2, cz, is."""
    p = Path(path)
    lines = [ln.split() for ln in p.read_text().splitlines() if ln.strip()]
    qc = QuantumCircuit(int(lines[0][0]), name=p.stem)
    for _, gate, *qs in lines[1:]:
        qs = [int(q) for q in qs]
        if gate == "x_1_2":
            qc.rx(math.pi / 2, qs[0])
        elif gate == "y_1_2":
            qc.ry(math.pi / 2, qs[0])
        elif gate == "is":
            qc.append(iSwapGate(), qs)
        elif gate in ("h", "t"):
            getattr(qc, gate)(qs[0])
        elif gate == "cz":
            qc.cz(*qs)
        else:
            raise ValueError(f"unknown GRCS gate {gate!r}")
    return qc


# --------------------------------------------------------------------------- #
# States that have no circuit                                                 #
# --------------------------------------------------------------------------- #
def shor_qubits(N: int) -> int:
    return 3 * N.bit_length()


def shor_state(N: int = 33, a: int = 5) -> np.ndarray:
    """Final state of Shor's algorithm with ``2n`` counting and ``n`` work qubits, ``n = bits(N)``.

    Counting register = qubits ``0..2n-1`` (qubit 0 most significant), work register = the rest.
    Modular exponentiation is applied as a permutation: before the inverse QFT the state is
    ``sum_x |x>|a^x mod N> / sqrt(2^2n)``; the inverse QFT is done with an FFT.
    """
    if math.gcd(a, N) != 1 or not 1 < a < N:
        raise ValueError(f"need 1 < a < N and gcd(a, N) = 1, got a={a}, N={N}")
    n = N.bit_length()
    M = 1 << (2 * n)
    y = np.empty(M, dtype=np.int64)
    y[0] = 1
    k = 1
    while k < M:                          # y[k:2k] = y[:k] * a^k mod N
        y[k:2 * k] = (y[:k] * pow(a, k, N)) % N
        k *= 2
    psi = np.zeros((M, 1 << n), dtype=np.complex128)
    psi[np.arange(M), y] = 1.0 / math.sqrt(M)
    return np.fft.fft(psi, axis=0, norm="ortho").reshape(-1)


def dicke_state(n: int = 6, k: int = 3) -> np.ndarray:
    """Uniform superposition of all bit strings of Hamming weight ``k`` (as in test_dicke)."""
    psi = np.zeros(2 ** n)
    for ones in itertools.combinations(range(n), k):
        psi[sum(1 << (n - 1 - b) for b in ones)] = 1.0
    return (psi / np.linalg.norm(psi)).astype(np.complex128)


# --------------------------------------------------------------------------- #
# The default list and the script                                             #
# --------------------------------------------------------------------------- #
Item = Tuple[str, int, Callable[[], Union[QuantumCircuit, np.ndarray]]]   # (name, qubits, build)

MQT_CIRCUITS = {"ghz": ghz, "graphstate": graphstate, "qft": qft, "qftentangled": qftentangled, "dj": dj,
                "qpeexact": qpeexact, "qpeinexact": qpeinexact, "wstate": wstate, "qaoa": qaoa,
                "vqe_real_amp": vqe_real_amp, "vqe_su2": vqe_su2, "vqe_two_local": vqe_two_local}


def default_benchmarks(sizes: Sequence[int] = (10, 14, 18)) -> List[Item]:
    """The literature benchmarks as ``(name, number of qubits, builder)`` triples."""
    items: List[Item] = []
    for rows, cols, depths in ((4, 4, (10, 15)), (4, 5, (10, 11, 12, 13, 14, 15))):   # Hillmich et al.
        for depth in depths:
            for inst in (0, 1):
                items.append((f"rqc_{rows}x{cols}_{depth}_{inst}", rows * cols,
                              functools.partial(rqc, rows, cols, depth, inst)))
    for N, a in ((33, 5), (55, 2), (69, 2), (69, 4), (221, 4), (323, 8), (629, 8), (1157, 8)):
        items.append((f"shor_{N}_{a}", shor_qubits(N), functools.partial(shor_state, N, a)))
    for name, build in MQT_CIRCUITS.items():                                           # MQT Bench
        for n in sizes:
            items.append((f"{name}_{n}", n, functools.partial(build, n)))
    items.append(("bell", 2, bell))                                                    # repository tests
    for n in (2, 3, 4):
        items += [(f"ghz_{n}", n, functools.partial(ghz, n)), (f"wstate_{n}", n, functools.partial(wstate, n))]
    items += [("dicke_6_3", 6, functools.partial(dicke_state, 6, 3)),
              ("dicke_8_4", 8, functools.partial(dicke_state, 8, 4))]
    unique = {}
    for item in items:                                   # ghz_10 etc. could appear twice
        unique.setdefault(item[0], item)
    return list(unique.values())


def to_state(built: Union[QuantumCircuit, np.ndarray]) -> np.ndarray:
    """Vector of a built benchmark: simulate a circuit, or normalise a ready-made state."""
    if isinstance(built, QuantumCircuit):
        return circuit_state(built)
    psi = np.asarray(built, dtype=np.complex128)
    return psi / np.linalg.norm(psi)


def save_state(psi: np.ndarray, stem: Union[str, Path], text: bool = True) -> List[Path]:
    """Write ``<stem>.npy`` (and ``<stem>.txt``, one ``real imag`` line per amplitude)."""
    stem = Path(stem)
    n = int(round(math.log2(psi.size)))
    np.save(stem.with_suffix(".npy"), psi)
    written = [stem.with_suffix(".npy")]
    if text:
        np.savetxt(stem.with_suffix(".txt"), np.column_stack([psi.real, psi.imag]), fmt="%.17e",
                   header=f"n_qubits={n}; one amplitude per line: real imag; "
                          f"index = sum_k x_k 2^(n-1-k), qubit 0 = most significant bit")
        written.append(stem.with_suffix(".txt"))
    return written


def load_state(path: Union[str, Path]) -> np.ndarray:
    """Read a state written by :func:`save_state` (``.npy`` or ``.txt``)."""
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path)
    data = np.loadtxt(path, ndmin=2)
    return data[:, 0] + 1j * data[:, 1]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Write the state vectors of the benchmark circuits.")
    ap.add_argument("outdir", nargs="?", default="benchmarks")
    ap.add_argument("--sizes", default="10,14,18", help="qubit counts of the MQT Bench circuits")
    ap.add_argument("--max-qubits", type=int, default=22, help="skip benchmarks with more qubits")
    ap.add_argument("--text-max-qubits", type=int, default=16, help="write .txt only up to this many qubits")
    ap.add_argument("--only", default=None, help="only benchmarks whose name contains this text")
    args = ap.parse_args(argv)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    done = skipped = 0
    for name, n, build in default_benchmarks([int(s) for s in args.sizes.split(",")]):
        if args.only and args.only not in name:
            continue
        if n > args.max_qubits:
            print(f"skip  {name}: {n} qubits (limit {args.max_qubits}; raise --max-qubits)")
            skipped += 1
            continue
        t = time.perf_counter()
        psi = to_state(build())
        files = save_state(psi, out / name, text=n <= args.text_max_qubits)
        print(f"wrote {name:<20} {n:2d} qubits  {time.perf_counter() - t:5.1f}s  "
              + " ".join(f.suffix for f in files))
        done += 1
    print(f"{done} written, {skipped} skipped -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
