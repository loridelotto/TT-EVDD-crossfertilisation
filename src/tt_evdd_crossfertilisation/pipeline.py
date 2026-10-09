"""TT truncation vs Hillmich approximation on the same state.

From a state vector psi (2**n amplitudes, site 0 = most significant bit):

    TT route:  psi -> MPS -> truncate to chi -> nd-EVDD -> deterministic EVDD
    DD route:  psi -> deterministic EVDD -> approx_hillmich(f = fidelity of chi)

and the two deterministic diagrams are compared for every chi.

Typical use:
    result = run_pipeline(psi)
    print_table(result)

or from the command line:
    python -m tt_evdd_crossfertilisation.pipeline state.npy --out results.json --tex results.tex

The diagrams are kept only when asked for (a dict passed as diagrams, or
--tex): on 16+ qubits they take tens of GB.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from tt2dd import TensorTrainState, convert_tt_to_evdd

from .mps_utils import convert_to_canonical_TT, evaluate_truncation_error
from .hillmich_approx import from_state, approx_hillmich, size, fidelity, nodes_by_level
from .nd_EVDD_2_EVDD import from_nd
from .draw_EVDDD import to_tikz, nd_to_tikz, to_document

CUTOFF = 1e-12      # drops numerically-zero singular values: chi_max is the Schmidt rank


def level_counts(evdd, n):
    """Number of nodes on each level 0 .. n-1."""
    by_level = nodes_by_level(evdd)
    return [len(by_level.get(lv, [])) for lv in range(n)]


def fmt_time(seconds):
    """A duration as 0.42 s, 3 min 05 s or 1 h 02 min."""
    if seconds < 60:
        return f"{seconds:.2f} s"
    m, s = divmod(round(seconds), 60)
    if m < 60:
        return f"{m} min {s:02d} s"
    h, m = divmod(m, 60)
    return f"{h} h {m:02d} min"


def log(msg):
    """Progress goes to stderr, so that the table on stdout stays clean."""
    print(msg, file=sys.stderr, flush=True)


def chis_for_fidelities(mps, targets):
    """For every target fidelity, the smallest chi whose truncation reaches it.

    The fidelity of the truncation grows with chi, so each target is a binary
    search over 1 .. chi_max that only truncates the MPS (no diagram is built).
    Targets that only chi_max reaches (no truncation at all) are dropped, and
    targets that land on the same chi give one chi.  Returns the chis, largest first.
    """
    chi_max = mps.max_bond()
    F = {chi_max: 1.0}

    def fid(chi):
        if chi not in F:
            F[chi] = evaluate_truncation_error(mps, max_bond=chi)[1]["target_reached"]
        return F[chi]

    found = {}
    for target in targets:
        lo, hi = 1, chi_max
        while lo < hi:
            mid = (lo + hi) // 2
            if fid(mid) >= target:
                hi = mid
            else:
                lo = mid + 1
        found[target] = lo
    log("chi per target: " + ", ".join(
        f"F>={t} -> {'none (needs chi_max)' if c == chi_max else c}" for t, c in found.items()))
    return sorted({c for c in found.values() if c < chi_max}, reverse=True)


def run_pipeline(psi, chis=None, cutoff=CUTOFF, verbose=False, diagrams=None):
    """Run both routes for every chi in chis (default: chi_max - 1 down to 1);
    chis can also be a function of the exact MPS that returns the list.

    Returns the results as a JSON-serialisable dict. The diagrams of each chi
    are dropped once measured, unless a dict is passed as diagrams: it gets
    the exact EVDD under "exact", the nd-EVDD of the untruncated TT under
    "exact_nd" and, for every chi, the triple
    (nd-EVDD, EVDD from the TT, EVDD from Hillmich).
    With verbose, prints the progress, an estimate of the time left and the
    total time.
    """
    start = time.perf_counter()
    psi = np.asarray(psi, dtype=complex).ravel()
    psi = psi / np.linalg.norm(psi)
    det = from_state(psi)
    mps = convert_to_canonical_TT(psi, cutoff=cutoff)
    n = mps.L
    chi_max = mps.max_bond()
    if chis is None:
        chis = range(chi_max - 1, 0, -1)
    elif callable(chis):
        chis = chis(mps)
    chis = list(chis)

    result = {
        "n": n,
        "exact_bonds": [int(b) for b in mps.bond_sizes()],
        "chi_max": int(chi_max),
        "size_exact": size(det),
        "levels_exact": level_counts(det, n),
        "rows": [],
    }
    if diagrams is not None:
        diagrams["exact"] = det
        diagrams["exact_nd"] = convert_tt_to_evdd(TensorTrainState.from_quimb(mps))
    setup = time.perf_counter() - start
    if verbose:
        log(f"exact EVDD and MPS built in {fmt_time(setup)}; {len(chis)} values of chi to go")

    loop_start = time.perf_counter()
    for k, chi in enumerate(chis, 1):
        step_start = time.perf_counter()
        mps_trunc, metrics = evaluate_truncation_error(mps, max_bond=chi)
        F = metrics["target_reached"]

        # TT route
        tt = TensorTrainState.from_quimb(mps_trunc)
        nd = convert_tt_to_evdd(tt)
        tt_det = from_nd(nd.to_dict())

        # DD route, at the fidelity the TT reached: approx_hillmich wants f in (0, 1]
        hill = approx_hillmich(det, f=min(max(F, 1e-12), 1.0))

        result["rows"].append({
            "chi": int(chi),
            "bonds": [int(b) for b in metrics["truncated_bonds"]],
            "tt_params": tt.num_parameters,
            "F_tt": F,
            "nd_nodes": nd.num_nodes,
            "nd_edges": nd.num_edges,
            "nd_is_det": nd.is_deterministic,
            "tt_det_size": size(tt_det),
            "tt_det_levels": level_counts(tt_det, n),
            "F_tt_det": fidelity(det, tt_det),
            "hill_size": size(hill),
            "hill_levels": level_counts(hill, n),
            "F_hill": fidelity(det, hill),
            "F_between": fidelity(tt_det, hill),
            "time": time.perf_counter() - step_start,
        })
        if diagrams is not None:
            diagrams[chi] = (nd, tt_det, hill)

        if verbose:
            # remaining steps at the mean pace so far: a rough estimate, steps differ in cost
            elapsed = time.perf_counter() - loop_start
            left = elapsed / k * (len(chis) - k)
            log(f"[{k}/{len(chis)}] chi={chi} done in {fmt_time(result['rows'][-1]['time'])}"
                f" | elapsed {fmt_time(time.perf_counter() - start)}"
                f" | ~{fmt_time(left)} left")

    result["time_setup"] = setup
    result["time_total"] = time.perf_counter() - start
    if verbose:
        log(f"total time: {fmt_time(result['time_total'])}\n")
    return result


def print_table(result):
    """The results as a table: one row per chi, the TT route on the left,
    the Hillmich route on the right, nodes as a count and as a % of the
    exact EVDD."""
    exact = result["size_exact"]
    print(f"State: {result['n']} qubits, chi_max = {result['chi_max']}, exact EVDD = {exact} nodes")
    print(f"Exact bonds: {result['exact_bonds']}")
    print()

    def pct(nodes):
        return f"{nodes} ({100 * nodes / exact:.0f}%)"

    # (title, width) of every column, grouped as chi | TT route | Hillmich | both
    groups = [("", [("chi", 4)]),
              ("TT truncation", [("fidelity", 9), ("nd nodes", 9), ("nd edges", 9), ("det nodes", 12)]),
              ("Hillmich (same target)", [("fidelity", 9), ("det nodes", 12)]),
              ("", [("F(TT, Hill)", 12)])]

    def line(cells):
        return " | ".join(cells)

    print(line(f"{g:^{sum(w for _, w in cols) + 2 * (len(cols) - 1)}}" for g, cols in groups))
    print(line("  ".join(f"{t:>{w}}" for t, w in cols) for _, cols in groups))
    print("-+-".join("--".join("-" * w for _, w in cols) for _, cols in groups))

    warnings = []
    for r in result["rows"]:
        values = [[f"{r['chi']}"],
                  [f"{r['F_tt']:.4f}", f"{r['nd_nodes']}", f"{r['nd_edges']}", pct(r["tt_det_size"])],
                  [f"{r['F_hill']:.4f}", pct(r["hill_size"])],
                  [f"{r['F_between']:.4f}"]]
        print(line("  ".join(f"{v:>{w}}" for v, (_, w) in zip(vals, cols))
                   for vals, (_, cols) in zip(values, groups)))
        if abs(r["F_tt_det"] - r["F_tt"]) > 1e-6:
            warnings.append(f"chi={r['chi']}: the det EVDD from the TT has F = {r['F_tt_det']:.6f},"
                            f" not F_tt = {r['F_tt']:.6f}")

    for w in warnings:
        print(f"WARNING {w}")


def draw(result, diagrams):
    """A standalone LaTeX document: the exact EVDD and the nd-EVDD of the
    untruncated TT, then for every chi the EVDD from the TT, the nd-EVDD it
    came from, and the Hillmich EVDD. Every edge carries its weight and, in
    brackets, its norm contribution."""
    nd = diagrams["exact_nd"]
    nd_kind = "" if nd.is_deterministic else " (non-deterministic)"
    figures = [[to_tikz(diagrams["exact"], edge_contrib=True), nd_to_tikz(nd, edge_contrib=True)]]
    captions = [[f"exact: {result['size_exact']} nodes",
                 f"exact nd-EVDD from TT{nd_kind}: {nd.num_nodes} nodes, {nd.num_edges} edges"]]
    for r in result["rows"]:
        nd, tt_det, hill = diagrams[r["chi"]]
        nd_kind = "" if r["nd_is_det"] else " (non-deterministic)"
        figures.append([to_tikz(tt_det, edge_contrib=True), nd_to_tikz(nd, edge_contrib=True),
                        to_tikz(hill, edge_contrib=True)])
        captions.append([
            f"$\\chi={r['chi']}$, TT $\\to$ det: {r['tt_det_size']} nodes, $F={r['F_tt_det']:.3f}$",
            f"nd-EVDD from TT{nd_kind}: {r['nd_nodes']} nodes, {r['nd_edges']} edges",
            f"Hillmich $f={r['F_tt']:.3f}$: {r['hill_size']} nodes, $F={r['F_hill']:.3f}$",
        ])
    return to_document(figures, captions)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("state", help=".npy file with the 2**n amplitudes")
    p.add_argument("--chis", type=int, nargs="+", help="bond dimensions to try (default: chi_max-1 .. 1)")
    p.add_argument("--cutoff", type=float, default=CUTOFF, help="SVD cutoff of the exact MPS")
    p.add_argument("--out", type=Path, help="write the results as JSON")
    p.add_argument("--tex", type=Path, help="write the diagrams as a LaTeX document")
    p.add_argument("--quiet", action="store_true", help="do not print progress and timing")
    args = p.parse_args(argv)

    diagrams = {} if args.tex else None
    result = run_pipeline(np.load(args.state), chis=args.chis, cutoff=args.cutoff,
                          verbose=not args.quiet, diagrams=diagrams)
    print_table(result)
    if args.out:
        args.out.write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.out}")
    if args.tex:
        args.tex.write_text(draw(result, diagrams))
        print(f"wrote {args.tex}")


if __name__ == "__main__":
    main()
