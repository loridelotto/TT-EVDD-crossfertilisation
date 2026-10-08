"""Tests for ``benchmark_circuits``: conventions, fidelity to MQT Bench, file formats.

Run with the package importable (``pip install -e .`` or ``PYTHONPATH=src``), from a directory
other than the repository root if you want the tt2dd integration test to run (see the skip
message):  ``python -m pytest tests/test_benchmark_circuits.py``
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("qiskit")
from qiskit import QuantumCircuit  # noqa: E402

from tt_evdd_crossfertilisation import benchmark_circuits as B  # noqa: E402
from tt_evdd_crossfertilisation.benchmark_circuits import (  # noqa: E402
    MQT_CIRCUITS,
    circuit_state,
    default_benchmarks,
    dicke_state,
    load_grcs,
    load_state,
    rqc,
    save_state,
    shor_qubits,
    shor_state,
    to_state,
)

DATA = Path(__file__).parent / "data" / "mqt_reference_n4.json"


def signature(qc):
    return [(i.operation.name, tuple(qc.find_bit(q).index for q in i.qubits),
             tuple(round(float(x), 12) for x in i.operation.params)) for i in qc.data]


# --------------------------------------------------------------------------- #
# The state convention of the repository                                      #
# --------------------------------------------------------------------------- #
def test_qubit_zero_is_the_most_significant_bit():
    qc = QuantumCircuit(3)
    qc.x(0)
    assert circuit_state(qc)[0b100] == pytest.approx(1)      # index 4; Qiskit's own vector says 1
    qc = QuantumCircuit(3)
    qc.x(2)
    assert circuit_state(qc)[0b001] == pytest.approx(1)


def test_every_default_benchmark_up_to_twelve_qubits_is_a_unit_complex128_vector():
    checked = 0
    for name, n, build in default_benchmarks(sizes=(4, 6)):
        if n > 12:
            continue
        psi = to_state(build())
        assert psi.dtype == np.complex128 and psi.shape == (2**n,), name
        assert np.linalg.norm(psi) == pytest.approx(1.0, abs=1e-12), name
        checked += 1
    assert checked > 30


def test_known_analytic_states():
    from tt_evdd_crossfertilisation.benchmark_circuits import bell, ghz, qft, qpeexact, wstate

    np.testing.assert_allclose(circuit_state(bell()), [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)], atol=1e-12)
    g = circuit_state(ghz(4))
    assert abs(g[0]) == pytest.approx(1 / math.sqrt(2)) and abs(g[15]) == pytest.approx(1 / math.sqrt(2))
    w = circuit_state(wstate(5))
    assert {i for i in range(32) if abs(w[i]) > 1e-9} == {1, 2, 4, 8, 16}
    np.testing.assert_allclose(circuit_state(qft(4)), np.full(16, 0.25), atol=1e-12)   # QFT|0..0> is uniform
    assert np.count_nonzero(np.abs(circuit_state(qpeexact(6))) > 1e-9) == 1            # exact phase: basis state


def test_dicke_matches_the_repository_definition():
    psi = dicke_state(6, 3)
    assert all((abs(psi[i]) > 0) == (bin(i).count("1") == 3) for i in range(64))


# --------------------------------------------------------------------------- #
# Fidelity to MQT Bench (reference vectors frozen from Qiskit)                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("family", sorted(json.loads(DATA.read_text())["states"]))
def test_state_equals_the_mqt_bench_reference(family):
    data = json.loads(DATA.read_text())
    ref = np.array([complex(re, im) for re, im in data["states"][family]])
    np.testing.assert_allclose(circuit_state(MQT_CIRCUITS[family](data["n"])), ref, atol=1e-9)


def test_qft_circuits_work_beyond_the_size_where_a_matrix_would_fit():
    # Qiskit's Statevector builds the full 2**n x 2**n matrix of a QFTGate (8 TiB at 20 qubits);
    # circuit_state decomposes first.
    psi = circuit_state(B.qft(20))
    assert np.linalg.norm(psi) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Google random circuits                                                      #
# --------------------------------------------------------------------------- #
def test_rqc_structure():
    rows, cols, depth = 4, 5, 10
    c = rqc(rows, cols, depth, instance=3)
    n = c.num_qubits
    assert n == rows * cols
    gates = signature(c)
    assert [g[0] for g in gates[:n]] == ["h"] * n and [g[0] for g in gates[-n:]] == ["h"] * n
    for name, qs, _ in gates:                      # every CZ joins nearest neighbours of the row-major grid
        if name == "cz":
            (r1, c1), (r2, c2) = divmod(qs[0], cols), divmod(qs[1], cols)
            assert abs(r1 - r2) + abs(c1 - c2) == 1
    assert sum(1 for g in gates if g[0] == "cz") > 0
    assert {g[0] for g in gates} <= {"h", "cz", "t", "rx", "ry"}


def test_rqc_layers_never_touch_a_qubit_twice_and_cover_every_edge():
    layers = B._rqc_layers(4, 5)
    for layer in layers:
        qubits = [q for pair in layer for q in pair]
        assert len(qubits) == len(set(qubits))
    assert len({tuple(sorted(e)) for layer in layers for e in layer}) == 4 * 4 + 3 * 5


def test_rqc_is_reproducible_and_instances_differ():
    assert signature(rqc(3, 3, 8, 0)) == signature(rqc(3, 3, 8, 0))
    assert signature(rqc(3, 3, 8, 0)) != signature(rqc(3, 3, 8, 1))


def test_grcs_file_parser(tmp_path):
    path = tmp_path / "inst_2x1_2_0.txt"
    path.write_text("2\n0 h 0\n0 h 1\n1 cz 0 1\n2 x_1_2 0\n2 y_1_2 1\n2 t 0\n")
    c = load_grcs(str(path))
    assert c.name == "inst_2x1_2_0" and c.num_qubits == 2
    h = np.array([[1, 1], [1, -1]]) / math.sqrt(2)
    rx = np.array([[1, -1j], [-1j, 1]]) / math.sqrt(2)      # X^1/2 of GRCS = rotation by pi/2
    ry = np.array([[1, -1], [1, 1]]) / math.sqrt(2)
    t = np.diag([1, np.exp(1j * math.pi / 4)])
    psi = np.diag([1, 1, 1, -1]) @ (np.kron(h, h) @ np.array([1, 0, 0, 0]))
    psi = np.kron(t @ rx, ry) @ psi
    np.testing.assert_allclose(circuit_state(c), psi, atol=1e-12)


# --------------------------------------------------------------------------- #
# Shor                                                                        #
# --------------------------------------------------------------------------- #
def test_shor_qubit_counts_are_those_of_hillmich_et_al():
    expected = {33: 18, 55: 18, 69: 21, 221: 24, 323: 27, 629: 30, 1157: 33}
    for N, qubits in expected.items():
        assert shor_qubits(N) == qubits


def test_shor_state_has_peaks_at_multiples_of_m_over_r():
    psi = to_state(shor_state(15, 2))                       # the order of 2 modulo 15 is 4
    prob = (np.abs(psi.reshape(2**8, 2**4)) ** 2).sum(axis=1)
    assert set(np.flatnonzero(prob > 1e-9)) == {0, 64, 128, 192}
    assert prob.sum() == pytest.approx(1.0)


def test_shor_rejects_bad_inputs():
    with pytest.raises(ValueError):
        shor_state(15, 5)                                   # gcd(5, 15) != 1


# --------------------------------------------------------------------------- #
# The default list and the files                                              #
# --------------------------------------------------------------------------- #
def test_default_list_has_the_sets_of_the_papers():
    names = {name for name, _, _ in default_benchmarks()}
    rqc_names = {n for n in names if n.startswith("rqc_")}
    assert len(rqc_names) == 2 * 2 + 6 * 2                  # 4x4 depth 10,15 and 4x5 depth 10..15, two instances
    assert {n for n in names if n.startswith("shor_")} == {
        f"shor_{N}_{a}" for N, a in [(33, 5), (55, 2), (69, 2), (69, 4), (221, 4), (323, 8), (629, 8), (1157, 8)]}
    for family in MQT_CIRCUITS:
        assert f"{family}_10" in names
    assert {"bell", "dicke_6_3", "dicke_8_4"} <= names


def test_text_and_npy_files_agree(tmp_path):
    psi = to_state(B.qftentangled(6))
    files = save_state(psi, tmp_path / "qftentangled_6")
    assert [f.suffix for f in files] == [".npy", ".txt"]
    np.testing.assert_array_equal(load_state(files[0]), psi)
    np.testing.assert_allclose(load_state(files[1]), psi, atol=1e-15)
    first = files[1].read_text().splitlines()[0]
    assert first.startswith("#") and "n_qubits=6" in first
    assert len(files[1].read_text().splitlines()) == 1 + 64            # header + one line per amplitude


def test_command_line_writes_the_files(tmp_path):
    assert B.main([str(tmp_path), "--only", "ghz_", "--sizes", "6,8"]) == 0
    assert (tmp_path / "ghz_6.npy").exists() and (tmp_path / "ghz_6.txt").exists()
    assert np.linalg.norm(load_state(tmp_path / "ghz_8.npy")) == pytest.approx(1.0)


def test_command_line_respects_the_size_limits(tmp_path, capsys):
    B.main([str(tmp_path), "--only", "ghz_", "--sizes", "20", "--max-qubits", "12"])
    assert "skip" in capsys.readouterr().out
    assert not (tmp_path / "ghz_20.npy").exists()
    B.main([str(tmp_path), "--only", "wstate_", "--sizes", "18", "--text-max-qubits", "10"])
    assert (tmp_path / "wstate_18.npy").exists() and not (tmp_path / "wstate_18.txt").exists()


# --------------------------------------------------------------------------- #
# Plugs into the rest of the repository                                       #
# --------------------------------------------------------------------------- #
def test_states_are_valid_input_for_the_deterministic_evdd():
    hillmich = pytest.importorskip("tt_evdd_crossfertilisation.hillmich_approx")
    psi = circuit_state(B.qftentangled(6))
    np.testing.assert_allclose(hillmich.to_vector(hillmich.from_state(psi)), psi, atol=1e-10)


def test_states_are_valid_input_for_tt2dd(tmp_path):
    tt2dd = pytest.importorskip("tt2dd")
    if not hasattr(tt2dd, "TensorTrainState"):
        pytest.skip("`import tt2dd` found the project folder tt2dd/ (a namespace package) instead of the "
                    "package: run pytest from another directory, or from tt2dd/")
    nd2det = pytest.importorskip("tt_evdd_crossfertilisation.nd_EVDD_2_EVDD")
    hillmich = pytest.importorskip("tt_evdd_crossfertilisation.hillmich_approx")
    files = save_state(circuit_state(B.vqe_su2(6)), tmp_path / "vqe_su2_6")
    for f in files:                                                    # the .npy and the .txt both work
        psi = load_state(f)
        tt = tt2dd.TensorTrainState.from_state_vector(psi, cutoff=1e-12)
        det = nd2det.from_nd(tt2dd.convert_tt_to_evdd(tt).to_dict())
        np.testing.assert_allclose(hillmich.to_vector(det), psi, atol=1e-9)
