"""Run the pipeline on every benchmark state and write all the tables to one text file.

Every <name>.npy in the benchmark folder is a state written by benchmark_circuits;
in the output file each table is headed by that name and by the circuit that
produced the state.  A table is appended as soon as its state is done, so an
interrupted run keeps what it has done, and --append resumes it.

    python -m tt_evdd_crossfertilisation.run_benchmarks --out results.txt
    python -m tt_evdd_crossfertilisation.run_benchmarks --out results.txt --append
    python -m tt_evdd_crossfertilisation.run_benchmarks --out ghz.txt --only ghz --max-qubits 14
"""

import argparse
import contextlib
import re
import time
import traceback
from pathlib import Path

import numpy as np

from .pipeline import CUTOFF, chis_for_fidelities, fmt_time, log, print_table, run_pipeline

ROOT = Path(__file__).resolve().parents[2]        # the repository: src/tt_evdd_crossfertilisation/ -> root
BENCH_DIR = ROOT / "benchmarks"
RESULTS_DIR = ROOT / "results benchmark"
FIDELITIES = [0.999, 0.995, 0.99, 0.98, 0.95, 0.90, 0.85, 0.80, 0.70, 0.50]

FAMILIES = {
    "bell": "Bell state",
    "ghz": "GHZ state",
    "wstate": "W state",
    "dicke": "Dicke state",
    "graphstate": "graph state (random 2-regular graph, MQT Bench)",
    "qft": "quantum Fourier transform (MQT Bench)",
    "qftentangled": "QFT applied to a GHZ state (MQT Bench)",
    "dj": "Deutsch-Jozsa, balanced oracle (MQT Bench)",
    "qpeexact": "quantum phase estimation, exact phase (MQT Bench)",
    "qpeinexact": "quantum phase estimation, inexact phase (MQT Bench)",
    "qaoa": "MaxCut QAOA, 2 layers (MQT Bench)",
    "vqe_real_amp": "VQE RealAmplitudes ansatz, 3 reps (MQT Bench)",
    "vqe_su2": "VQE EfficientSU2 ansatz, 3 reps (MQT Bench)",
    "vqe_two_local": "VQE TwoLocal ansatz, 3 reps (MQT Bench)",
    "rqc": "Google random circuit (Hillmich et al.)",
    "shor": "Shor's algorithm, final state (Hillmich et al.)",
}


def describe(name):
    """The circuit behind a benchmark name, e.g. 'Dicke state (n=6, k=3)'."""
    m = re.fullmatch(r"rqc_(\d+)x(\d+)_(\d+)_(\d+)", name)
    if m:
        rows, cols, depth, inst = m.groups()
        return f"{FAMILIES['rqc']} (grid {rows}x{cols}, depth {depth}, instance {inst})"
    m = re.fullmatch(r"shor_(\d+)_(\d+)", name)
    if m:
        return f"{FAMILIES['shor']} (N={m.group(1)}, a={m.group(2)})"
    m = re.fullmatch(r"dicke_(\d+)_(\d+)", name)
    if m:
        return f"{FAMILIES['dicke']} (n={m.group(1)}, k={m.group(2)})"
    m = re.fullmatch(r"(.+?)(?:_(\d+))?", name)
    return FAMILIES.get(m.group(1), "unknown circuit")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dir", type=Path, default=BENCH_DIR, help="folder with the .npy states")
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "benchmark_results.txt", help="text file with all the tables")
    p.add_argument("--only", help="only states whose name contains this text")
    p.add_argument("--min-qubits", type=int, help="skip states with fewer qubits")
    p.add_argument("--max-qubits", type=int, help="skip states with more qubits")
    p.add_argument("--chis", type=int, nargs="+", help="bond dimensions to try (default: chi_max-1 .. 1)")
    p.add_argument("--pow2", action="store_true", help="try only the powers of 2 below chi_max")
    p.add_argument("--fidelities", type=float, nargs="*",
                   help="try only the smallest chi reaching each of these fidelities "
                        f"(no values: {' '.join(map(str, FIDELITIES))})")
    p.add_argument("--cutoff", type=float, default=CUTOFF, help="SVD cutoff of the exact MPS")
    p.add_argument("--append", action="store_true",
                   help="add to --out instead of overwriting it, skipping the states already in it")
    p.add_argument("--quiet", action="store_true", help="do not print the progress of each state")
    args = p.parse_args(argv)

    # smallest states first
    files = sorted(args.dir.glob("*.npy"), key=lambda f: (np.load(f, mmap_mode="r").size, f.stem))
    if args.only:
        files = [f for f in files if args.only in f.stem]

    chis = args.chis
    if args.pow2:
        chis = lambda mps: [2 ** k for k in reversed(range(mps.max_bond().bit_length())) if 2 ** k < mps.max_bond()]
    if args.fidelities is not None:
        targets = args.fidelities or FIDELITIES
        chis = lambda mps: chis_for_fidelities(mps, targets)

    done = set()
    if args.append and args.out.exists():
        # a state is done when its table is complete, i.e. its "time:" line was written
        name = None
        for line in args.out.read_text().splitlines():
            m = re.match(r"(\S+): ", line)
            if m and m.group(1) not in ("State", "Exact", "time", "WARNING"):
                name = m.group(1)
            elif name and (line.startswith("time:") or line.startswith("ERROR")):
                done.add(name)
    else:
        args.out.write_text("")

    start = time.perf_counter()
    for i, f in enumerate(files, 1):
        if f.stem in done:
            log(f"[{i}/{len(files)}] skip {f.stem}: already in {args.out}")
            continue
        psi = np.load(f)
        n = int(round(np.log2(psi.size)))
        if (args.min_qubits and n < args.min_qubits) or (args.max_qubits and n > args.max_qubits):
            log(f"[{i}/{len(files)}] skip {f.stem}: {n} qubits")
            continue

        log(f"[{i}/{len(files)}] {f.stem} ({n} qubits)")
        with args.out.open("a") as out, contextlib.redirect_stdout(out):
            print("=" * 100)
            print(f"{f.stem}: {describe(f.stem)}")
            print("=" * 100)
            try:
                result = run_pipeline(psi, chis=chis,cutoff=args.cutoff, verbose=not args.quiet)
                print_table(result)
                print(f"time: {fmt_time(result['time_total'])}")
            except Exception as exc:  # keep going with the other states
                print(f"ERROR {type(exc).__name__}: {exc}")
                log(traceback.format_exc())
            print()

    log(f"done in {fmt_time(time.perf_counter() - start)} -> {args.out}")


if __name__ == "__main__":
    main()
