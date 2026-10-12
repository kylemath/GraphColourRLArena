"""Print a search-budget curve and one learned run.

From this directory, with the project virtualenv:

    ../../.venv/bin/python -m bitter_prover
    ../../.venv/bin/python -m bitter_prover --budget 256 --rounds 8 --synth-depth 6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bitter_prover.iterate import (
    DEFAULT_BUDGET,
    DEFAULT_ROUNDS,
    DEFAULT_WIDTH,
    run_once,
    run_sweep,
    search_curve,
)
from bitter_prover.kernel import walk
from bitter_prover.theory import EXAM, OPEN_ATOM, SYNTH_DEPTH


def main() -> None:
    parser = argparse.ArgumentParser(description="Scale-knob checker-trained prover")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--synth-depth", type=int, default=SYNTH_DEPTH)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    args = parser.parse_args()

    print("Scale knobs: budget, rounds, synth-depth, width")
    print("Unfitted search, one seed, base theory (no lemmas, no training)")
    curve = search_curve(seed=args.seed, width=args.width)
    for row in curve:
        status = "proved" if row.proved else "unproved"
        print(f"  budget {row.budget:5d}: {status} ({row.expansions} visits)")

    report = run_once(
        seed=args.seed,
        budget=args.budget,
        rounds=args.rounds,
        synth_depth=args.synth_depth,
        width=args.width,
    )
    print(
        f"Learned loop: budget {report.budget}, rounds requested {report.rounds}, "
        f"synth depth {report.synth_depth}, width {report.width}"
    )
    print(f"Synthetic theorems: {report.synthetic}. Lemmas installed: {report.lemmas}.")
    for row in report.rows:
        status = "proved" if row.exam_proved else "unproved"
        print(
            f"  round {row.index}: exam {status} "
            f"({row.expansions} visits, {row.lemmas} lemmas)"
        )
    if report.exam_proof is None:
        print(f"Exam `{EXAM}`: unproved after {report.exam_expansions} visits on the last round")
    else:
        names = [node.clause for node in walk(report.exam_proof)]
        print(
            f"Exam `{EXAM}`: proved on round {report.exam_round} "
            f"({report.exam_expansions} visits, checker accepted, "
            f"{len(names)} nodes in the returned tree)"
        )
    print(
        f"Open atom `{OPEN_ATOM}`: "
        f"{'proved' if report.open_proved else 'unproved'} "
        f"({report.open_expansions} visits)"
    )

    sweep = run_sweep(
        seeds=args.seeds,
        budget=args.budget,
        rounds=args.rounds,
        synth_depth=args.synth_depth,
        width=args.width,
    )
    print(
        f"Sweep of {sweep.seeds} seeds at these knobs: "
        f"exam proved {sweep.trained_exam}/{sweep.seeds}, "
        f"open atom proved {sweep.open_atom}/{sweep.seeds}"
    )
    print("A flat exam rate is a measurement. This run does not prove the Four Colour Theorem.")

    payload = {
        "exam": EXAM,
        "open_atom": OPEN_ATOM,
        "seed": report.seed,
        "budget": report.budget,
        "rounds": report.rounds,
        "synth_depth": report.synth_depth,
        "width": report.width,
        "synthetic": report.synthetic,
        "lemmas": report.lemmas,
        "search_curve": [
            {"budget": row.budget, "proved": row.proved, "expansions": row.expansions}
            for row in curve
        ],
        "learned_rounds": [
            {
                "index": row.index,
                "exam_proved": row.exam_proved,
                "expansions": row.expansions,
                "lemmas": row.lemmas,
            }
            for row in report.rows
        ],
        "exam_proved": report.exam_proof is not None,
        "exam_round": report.exam_round,
        "exam_expansions": report.exam_expansions,
        "exam_clauses": (
            [node.clause for node in walk(report.exam_proof)] if report.exam_proof else []
        ),
        "open_proved": report.open_proved,
        "sweep": {
            "seeds": sweep.seeds,
            "trained_exam": sweep.trained_exam,
            "open_atom": sweep.open_atom,
        },
        "four_colour_theorem": "not claimed",
    }
    out = Path(__file__).resolve().parent.parent / "last-run.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
