"""Parallel Lean search sized for a multi-core machine.

Each candidate is a snippet. Lean accepts or rejects it. ``workers`` Lean
processes run at once, each with ``LEAN_NUM_THREADS=1``. ``budget`` is how
many scripts are offered to one goal. ``rounds`` refits script weights from
scripts Lean has accepted and retries goals still open. A checked proof is
reused only by a later snippet Lean also accepts. A false goal has no
accepted script in the pool.

Nothing here is a proof of the Four Colour Theorem. The files are Init-only,
plus ``import Lean`` on the ``omega`` script.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from bitter_prover.lean_kernel import CheckResult, check_many
from bitter_prover.policy import stable_hash

# Exact tactic text. The unfitted order is a shuffle, not this listing.
# ``omega`` is the only script that needs a preamble.
_SCRIPTS: tuple[tuple[str, str, str], ...] = (
    ("sorry", "sorry", ""),
    ("rfl", "rfl", ""),
    ("simp", "simp", ""),
    ("add_zero", "exact Nat.add_zero n", ""),
    ("zero_add", "exact Nat.zero_add n", ""),
    ("mul_zero", "exact Nat.mul_zero n", ""),
    ("mul_one", "exact Nat.mul_one n", ""),
    ("one_mul", "exact Nat.one_mul n", ""),
    ("sub_self", "exact Nat.sub_self n", ""),
    ("add_comm_same", "exact Nat.add_comm n n", ""),
    ("true_intro", "exact True.intro", ""),
    ("omega", "omega", "import Lean\n"),
)


@dataclass(frozen=True)
class Script:
    """One proof script. ``preamble`` is source placed above the theorem."""

    label: str
    body: str
    preamble: str = ""


SCRIPTS: tuple[Script, ...] = tuple(Script(label, body, preamble) for label, body, preamble in _SCRIPTS)


@dataclass(frozen=True)
class Goal:
    """A theorem Lean will be asked to close. ``open_exam`` goals are false."""

    name: str
    binders: str
    statement: str
    open_exam: bool


@dataclass(frozen=True)
class GoalResult:
    """Lean's answer for one goal after one round."""

    name: str
    open_exam: bool
    accepted: bool
    script: str | None
    attempts: int


@dataclass
class ScaleReport:
    """One finished run. Counts are Lean answers."""

    workers: int
    goals: int
    budget: int
    rounds: int
    width: int
    seed: int
    proved: int
    open_goals: int
    open_proved: int
    lemma_uses_accepted: int
    bad_links_accepted: int
    checks: int
    seconds: float
    results: tuple[GoalResult, ...]


def cpu_count() -> int:
    """Logical CPUs, at least 1. On Apple Silicon this is the pool size to pass."""
    count = os.cpu_count() or 1
    return max(1, count)


def make_goals(count: int) -> tuple[Goal, ...]:
    """``count`` theorems. Every fifth one is false and is an open exam.

    The true families are definitional or standard ``Nat`` lemmas in Init.
    Repeating them is the workload: each copy is a separate Lean file.
    """
    true_family = (
        ("add_zero", "{v} + 0 = {v}"),
        ("zero_add", "0 + {v} = {v}"),
        ("mul_zero", "{v} * 0 = 0"),
        ("mul_one", "{v} * 1 = {v}"),
        ("one_mul", "1 * {v} = {v}"),
        ("sub_self", "{v} - {v} = 0"),
    )
    false_family = (
        ("add_one_eq", "{v} + 1 = {v}"),
        ("mul_one_zero", "{v} * 1 = 0"),
    )
    goals: list[Goal] = []
    for index in range(count):
        if index % 5 == 4:
            family, template = false_family[(index // 5) % len(false_family)]
            open_exam = True
        else:
            family, template = true_family[index % len(true_family)]
            open_exam = False
        goals.append(
            Goal(
                name=f"{family}_{index}",
                binders="(n : Nat)",
                statement=template.format(v="n"),
                open_exam=open_exam,
            )
        )
    return tuple(goals)


def run_scaled(
    *,
    goals: int = 32,
    budget: int = 12,
    rounds: int = 1,
    workers: int | None = None,
    width: int = 256,
    seed: int = 0,
    output: Path | None = None,
) -> ScaleReport:
    """Search ``goals`` theorems with ``workers`` Lean processes.

    ``workers`` defaults to :func:`cpu_count`. ``width`` is recorded and is
    the modulus for a hash of script labels in the fitted weights; the
    weights themselves come only from scripts Lean has accepted.
    """
    if goals < 1:
        raise ValueError("goals must be positive")
    if budget < 0:
        raise ValueError("budget must be non-negative")
    if rounds < 1:
        raise ValueError("rounds must be positive")
    if width < 1:
        raise ValueError("width must be positive")
    pool = cpu_count() if workers is None else workers
    if pool < 1:
        raise ValueError("workers must be positive")

    work = make_goals(goals)
    weights: dict[str, float] = {}
    by_name: dict[str, GoalResult] = {}
    checks = 0
    lemma_uses = 0
    bad_links = 0
    started = time.monotonic()

    for round_index in range(rounds):
        pending = [goal for goal in work if goal.name not in by_name or not by_name[goal.name].accepted]
        if pending:
            found, used = _search_round(
                pending, budget, seed, pool, weights, width, round_index
            )
            checks += used
            for result in found:
                previous = by_name.get(result.name)
                if previous is not None and previous.accepted:
                    continue
                by_name[result.name] = result
                if result.accepted and result.script is not None:
                    key = _weight_key(result.script, width)
                    weights[key] = weights.get(key, 0.0) + 1.0
        proved_now = [goal for goal in work if by_name.get(goal.name) and by_name[goal.name].accepted and not goal.open_exam]
        if proved_now and round_index == rounds - 1:
            uses, bads, used = _check_links(proved_now, by_name, pool)
            checks += used
            lemma_uses += uses
            bad_links += bads

    elapsed = time.monotonic() - started
    ordered = tuple(by_name[goal.name] for goal in work if goal.name in by_name)
    report = ScaleReport(
        workers=pool,
        goals=goals,
        budget=budget,
        rounds=rounds,
        width=width,
        seed=seed,
        proved=sum(1 for item in ordered if item.accepted and not item.open_exam),
        open_goals=sum(1 for goal in work if goal.open_exam),
        open_proved=sum(1 for item in ordered if item.accepted and item.open_exam),
        lemma_uses_accepted=lemma_uses,
        bad_links_accepted=bad_links,
        checks=checks,
        seconds=elapsed,
        results=ordered,
    )
    if output is not None:
        _write_report(output, report, weights)
    return report


def _search_round(
    goals: list[Goal],
    budget: int,
    seed: int,
    workers: int,
    weights: dict[str, float],
    width: int,
    round_index: int,
) -> tuple[list[GoalResult], int]:
    """Check a budget-prefix for every goal. The earliest accepted script wins."""
    jobs: list[tuple[str, int, str, str]] = []
    prefixes: dict[str, list[Script]] = {}
    for goal in goals:
        ranked = _rank(goal.name, seed + round_index, weights, width)
        chosen = ranked[:budget]
        prefixes[goal.name] = chosen
        for index, script in enumerate(chosen):
            jobs.append((goal.name, index, script.label, _render(goal, script)))
    if not jobs:
        return [
            GoalResult(goal.name, goal.open_exam, False, None, 0) for goal in goals
        ], 0

    sources = [job[3] for job in jobs]
    done = 0

    def _progress(index: int, result: CheckResult) -> None:
        del index, result
        nonlocal done
        done += 1
        if done % 50 == 0 or done == len(sources):
            print(f"  checks {done}/{len(sources)}", file=sys.stderr)

    results = check_many(sources, workers=workers, on_result=_progress)
    accepted_at: dict[str, tuple[int, str]] = {}
    for (name, index, label, _source), result in zip(jobs, results):
        if not result.accepted:
            continue
        previous = accepted_at.get(name)
        if previous is None or index < previous[0]:
            accepted_at[name] = (index, label)
    found: list[GoalResult] = []
    for goal in goals:
        hit = accepted_at.get(goal.name)
        found.append(
            GoalResult(
                name=goal.name,
                open_exam=goal.open_exam,
                accepted=hit is not None,
                script=None if hit is None else hit[1],
                attempts=len(prefixes[goal.name]),
            )
        )
    return found, len(jobs)


def _check_links(
    proved: list[Goal],
    by_name: dict[str, GoalResult],
    workers: int,
) -> tuple[int, int, int]:
    """Ask Lean to use each accepted theorem, once at its type and once at a mismatch.

    The mismatch is a single shared false statement, checked against the first
    proved theorem only. The matching uses are one per proved theorem.
    """
    jobs: list[tuple[str, str]] = []
    for goal in proved:
        result = by_name[goal.name]
        if result.script is None:
            continue
        script = _script_by_label(result.script)
        jobs.append(("use", _render_link(goal, script, goal.statement)))
    if proved:
        first = proved[0]
        result = by_name[first.name]
        if result.script is not None:
            script = _script_by_label(result.script)
            jobs.append(("bad", _render_link(first, script, "m + 1 = m")))
    if not jobs:
        return 0, 0, 0
    checked = check_many([source for _kind, source in jobs], workers=workers)
    uses = sum(1 for (kind, _), result in zip(jobs, checked) if kind == "use" and result.accepted)
    bads = sum(1 for (kind, _), result in zip(jobs, checked) if kind == "bad" and result.accepted)
    return uses, bads, len(jobs)


def _rank(goal_name: str, seed: int, weights: dict[str, float], width: int) -> list[Script]:
    """Shuffle, then stable-sort by weight. Equal weights keep the shuffle.

    The weight key is a hash bucket of width ``width``. At a large width each
    script has its own bucket. At a small width scripts share a bucket and
    the fit cannot tell them apart.
    """
    rng = random.Random(f"{seed}:{goal_name}")
    ordered = list(SCRIPTS)
    rng.shuffle(ordered)
    ordered.sort(key=lambda script: weights.get(_weight_key(script.label, width), 0.0), reverse=True)
    return ordered


def _weight_key(label: str, width: int) -> str:
    return str(stable_hash(label) % width)


def _script_by_label(label: str) -> Script:
    for script in SCRIPTS:
        if script.label == label:
            return script
    raise KeyError(label)


def _render(goal: Goal, script: Script) -> str:
    return (
        f"{script.preamble}"
        f"theorem {goal.name} {goal.binders} : {goal.statement} := by\n"
        f"  {script.body}\n"
    )


def _render_link(goal: Goal, script: Script, use_statement: str) -> str:
    """Accepted theorem, then ``exact`` of that name at ``use_statement``.

    ``use_statement`` is written with ``m`` as the bound variable.
    """
    use = use_statement.replace("n", "m")
    return (
        f"{script.preamble}"
        f"theorem {goal.name} {goal.binders} : {goal.statement} := by\n"
        f"  {script.body}\n"
        f"\n"
        f"theorem {goal.name}_use (m : Nat) : {use} := by\n"
        f"  exact {goal.name} m\n"
    )


def _write_report(output: Path, report: ScaleReport, weights: dict[str, float]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "workers": report.workers,
        "goals": report.goals,
        "budget": report.budget,
        "rounds": report.rounds,
        "width": report.width,
        "seed": report.seed,
        "proved": report.proved,
        "open_goals": report.open_goals,
        "open_proved": report.open_proved,
        "lemma_uses_accepted": report.lemma_uses_accepted,
        "bad_links_accepted": report.bad_links_accepted,
        "checks": report.checks,
        "seconds": round(report.seconds, 3),
        "checks_per_second": round(report.checks / report.seconds, 2) if report.seconds else None,
        "cpu_count": cpu_count(),
        "machine": platform.machine(),
        "system": platform.system(),
        "script_weights": weights,
        "four_colour_theorem": "not claimed",
        "results": [
            {
                "name": item.name,
                "open_exam": item.open_exam,
                "accepted": item.accepted,
                "script": item.script,
                "attempts": item.attempts,
            }
            for item in report.results
        ],
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Parallel Lean search across CPU cores")
    parser.add_argument("--goals", type=int, default=32)
    parser.add_argument("--budget", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Lean processes at once. 0 means one per logical CPU.",
    )
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/scaled"),
        help="Directory for summary.json",
    )
    args = parser.parse_args(argv)
    workers = cpu_count() if args.workers == 0 else args.workers
    print(
        f"Parallel Lean search: {args.goals} goals, budget {args.budget}, "
        f"{args.rounds} rounds, {workers} workers, width {args.width}"
    )
    print("Init-only Nat facts, plus omega. This is not the Four Colour Theorem.")
    report = run_scaled(
        goals=args.goals,
        budget=args.budget,
        rounds=args.rounds,
        workers=workers,
        width=args.width,
        seed=args.seed,
        output=args.output,
    )
    rate = report.checks / report.seconds if report.seconds else 0.0
    print(
        f"Proved {report.proved} true goals. "
        f"Open exams {report.open_goals}, of which proved {report.open_proved}. "
        f"Lemma uses accepted {report.lemma_uses_accepted}. "
        f"Bad links accepted {report.bad_links_accepted}."
    )
    print(
        f"{report.checks} Lean checks in {report.seconds:.1f}s "
        f"({rate:.1f} checks/s) on {report.workers} workers."
    )
    print(f"Wrote {args.output / 'summary.json'}")


if __name__ == "__main__":
    main()
