"""Print an Init-only Lean search curve and one typed run.

From this directory, with the project virtualenv:

    ../../.venv/bin/python -m bitter_prover.typed_demo

``search_budget_curve`` rows expose ``budget``, ``proved``, and ``attempts``.
``run_typed`` exposes ``nat_add_zero_accepted``, ``exact_nat_add_zero_accepted``,
``mismatch_rejected``, and ``open_unproved`` (``open_proved`` is the opposite
bit). An unproved open goal is a finished run: this process exits 0.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from bitter_prover.typed_loop import run_typed, search_budget_curve

_CURVE_BUDGETS: tuple[int, ...] = (0, 1, 2, 4)

_ATTEMPTS = ("attempts", "expansions", "visits")


def _has(obj: Any, name: str) -> bool:
    if isinstance(obj, dict):
        return name in obj
    return hasattr(obj, name)


def _lookup(obj: Any, names: Sequence[str]) -> Any:
    for name in names:
        if _has(obj, name):
            return obj[name] if isinstance(obj, dict) else getattr(obj, name)
    if isinstance(obj, dict):
        visible: Iterable[str] = obj.keys()
    else:
        visible = [name for name in dir(obj) if not name.startswith("_")]
    raise AttributeError(
        "need one of {0}; {1} has {2}".format(
            ", ".join(names), type(obj).__name__, ", ".join(visible)
        )
    )


def _curve_rows() -> list[dict[str, Any]]:
    curve = search_budget_curve(budgets=_CURVE_BUDGETS, seed=0)
    if not isinstance(curve, (list, tuple)) and _has(curve, "rows"):
        curve = curve.rows
    rows: list[dict[str, Any]] = []
    for row in curve:
        if isinstance(row, (list, tuple)):
            budget, proved, attempts = row[0], row[1], row[2]
        else:
            budget = _lookup(row, ("budget",))
            proved = _lookup(row, ("proved",))
            attempts = _lookup(row, _ATTEMPTS)
        rows.append(
            {"budget": int(budget), "proved": bool(proved), "attempts": int(attempts)}
        )
    return rows


def main() -> None:
    print(
        "This demo's constraint is the Lean kernel on Init-only files. "
        "It is not a proof of the Four Colour Theorem. mathlib is not built."
    )
    print("Unfitted search, seed 0")
    curve_rows = _curve_rows()
    for row in curve_rows:
        status = "proved" if row["proved"] else "unproved"
        print(
            "  budget {budget:5d}: {status} ({attempts} attempts)".format(
                budget=row["budget"], status=status, attempts=row["attempts"]
            )
        )

    report = run_typed(seed=0, budget=4, rounds=1, width=32)
    add_zero = report.provable_accepted
    corollary = report.lemma_use_accepted
    mismatch = not report.bad_link_accepted
    open_unproved = not report.open_accepted

    print(
        "Nat add-zero goal: {0}".format("accepted" if add_zero else "not accepted")
    )
    print(
        "exact nat_add_zero for the matching corollary: {0}".format(
            "accepted" if corollary else "not accepted"
        )
    )
    print(
        "mismatched goal `m + 1 = m`: {0}".format(
            "rejected" if mismatch else "not rejected"
        )
    )
    print("open false goal: {0}".format("unproved" if open_unproved else "proved"))

    payload = {
        "seed": 0,
        "budget": 4,
        "rounds": 1,
        "width": 32,
        "search_curve": curve_rows,
        "nat_add_zero_accepted": add_zero,
        "exact_nat_add_zero_accepted": corollary,
        "mismatch_rejected": mismatch,
        "open_unproved": open_unproved,
        "four_colour_theorem": "not claimed",
        "mathlib": "not built",
    }
    out = Path(__file__).resolve().parent.parent / "typed-last-run.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("Wrote {0}".format(out))


if __name__ == "__main__":
    main()
