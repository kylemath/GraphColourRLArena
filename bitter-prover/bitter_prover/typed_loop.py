"""Init-only Lean loop. The knobs are amounts of compute.

``budget`` is how many candidate scripts Lean is asked to check on one
goal. ``rounds`` is how many times an accepted proof may be installed and
the open exam retried. ``width`` is recorded and passed to the policy. It
is not a curriculum: candidates are exact tactic strings, and a weight is
a number on that whole string.

An unfitted policy has every weight 0, so the order is a seeded shuffle.
One shuffle is fixed for a call. Budget ``b`` may try only the first ``b``
scripts of that order, and it stops at the first script Lean accepts, so a
larger budget tries a superset. ``sorry`` is not acceptance. The exam
``n + 1 = n`` is false.

Nothing here is a proof of the Four Colour Theorem.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import NamedTuple

from bitter_prover.lean_kernel import CheckResult, check_snippet

# Fixed scripts. ``sorry`` and the ``intro`` lines are rejected. ``rfl`` is a
# proof of ``n + 0 = n`` because addition on the right by zero is the
# definition. The order below is not a ranking; the policy shuffles it.
CANDIDATES: tuple[str, ...] = (
    "sorry",
    "intro n; rfl",
    "rfl",
    "intro n; exact Nat.add_zero n",
)

_USE_STATEMENT = "m + 0 = m"
_BAD_STATEMENT = "m + 1 = m"


@dataclass(frozen=True)
class _Goal:
    """Theorem binders are ``(n : Nat)``. ``statement`` is the type after the colon."""

    name: str
    statement: str


PROVABLE_GOAL = _Goal("nat_add_zero", "n + 0 = n")
OPEN_EXAM = _Goal("open_goal", "n + 1 = n")


class Policy:
    """Rank tactic strings. Unfitted weights are 0, so rank is a seeded shuffle.

    ``width`` is stored and reported. It does not bin or rewrite a candidate:
    each weight key is the exact script text.
    """

    def __init__(self, seed: int = 0, width: int = 32) -> None:
        if width < 1:
            raise ValueError("width must be positive")
        self.seed = seed
        self.width = width
        self.weights: dict[str, float] = {}

    def rank(self, candidates: tuple[str, ...]) -> tuple[str, ...]:
        """Highest weight first. Equal weights keep the shuffle order."""
        ordered = list(candidates)
        random.Random(self.seed).shuffle(ordered)
        ordered.sort(key=lambda text: self.weights.get(text, 0.0), reverse=True)
        return tuple(ordered)


@dataclass(frozen=True)
class _Hit:
    """One budgeted pass over a prefix of a fixed candidate order."""

    attempts: int
    proof: str | None

    @property
    def accepted(self) -> bool:
        return self.proof is not None


@dataclass(frozen=True)
class RoundRow:
    """One pass: search the provable goal, install if Lean accepted, retry the exam."""

    index: int
    provable_accepted: bool
    provable_attempts: int
    open_accepted: bool
    open_attempts: int
    lemma_use_accepted: bool
    bad_link_accepted: bool


@dataclass(frozen=True)
class TypedReport:
    """One seed. Every boolean is a Lean answer, never a budget cutoff dressed up as a proof."""

    seed: int
    budget: int
    rounds: int
    width: int
    provable_accepted: bool
    provable_attempts: int
    open_accepted: bool
    open_attempts: int
    lemma_use_accepted: bool
    bad_link_accepted: bool
    provable_proof: str | None
    rows: tuple[RoundRow, ...]


class CurvePoint(NamedTuple):
    """Unfitted search of the provable goal at one budget."""

    budget: int
    proved: bool
    attempts: int


def run_typed(
    seed: int = 0,
    budget: int = 4,
    rounds: int = 1,
    width: int = 32,
    policy: Policy | None = None,
) -> TypedReport:
    """Search, install one accepted lemma, and retry the open exam for ``rounds`` passes.

    The candidate order is shuffled once. A provided policy supplies that
    order; otherwise the policy is unfitted at ``seed`` and ``width``.
    """
    if budget < 0:
        raise ValueError("budget must be non-negative")
    if rounds < 0:
        raise ValueError("rounds must be non-negative")
    if width < 1:
        raise ValueError("width must be positive")
    if policy is None:
        policy = Policy(seed=seed, width=width)
    order = policy.rank(CANDIDATES)

    provable_accepted = False
    provable_attempts = 0
    provable_proof: str | None = None
    lemma_use_accepted = False
    bad_link_accepted = False
    rows: list[RoundRow] = []

    for index in range(rounds):
        if not provable_accepted:
            hit = _search(PROVABLE_GOAL, order, budget)
            provable_attempts = hit.attempts
            provable_proof = hit.proof
            provable_accepted = hit.accepted
            if hit.proof is not None:
                lemma_use_accepted = _lean_accepts(_linked(hit.proof, _USE_STATEMENT))
                bad_link_accepted = _lean_accepts(_linked(hit.proof, _BAD_STATEMENT))
        opened = _search(OPEN_EXAM, order, budget)
        rows.append(
            RoundRow(
                index=index,
                provable_accepted=provable_accepted,
                provable_attempts=provable_attempts,
                open_accepted=opened.accepted,
                open_attempts=opened.attempts,
                lemma_use_accepted=lemma_use_accepted,
                bad_link_accepted=bad_link_accepted,
            )
        )

    return TypedReport(
        seed=seed,
        budget=budget,
        rounds=rounds,
        width=width,
        provable_accepted=provable_accepted,
        provable_attempts=provable_attempts,
        open_accepted=any(row.open_accepted for row in rows),
        open_attempts=rows[-1].open_attempts if rows else 0,
        lemma_use_accepted=lemma_use_accepted,
        bad_link_accepted=bad_link_accepted,
        provable_proof=provable_proof,
        rows=tuple(rows),
    )


def search_budget_curve(budgets: tuple[int, ...], seed: int) -> tuple[CurvePoint, ...]:
    """Unfitted search of the provable goal at each budget, one seed, one shuffle.

    The provable goal only. No lemma is installed and the open exam is not
    attempted. Each budget reads a prefix of the same permutation, so a
    success is kept by every larger budget.
    """
    order = Policy(seed=seed).rank(CANDIDATES)
    points: list[CurvePoint] = []
    for budget in budgets:
        hit = _search(PROVABLE_GOAL, order, budget)
        points.append(CurvePoint(budget, hit.accepted, hit.attempts))
    return tuple(points)


def _search(goal: _Goal, order: tuple[str, ...], budget: int) -> _Hit:
    """Try ``order[:budget]`` and stop at the first script Lean accepts."""
    if budget < 0:
        raise ValueError("budget must be non-negative")
    attempts = 0
    for candidate in order[:budget]:
        attempts += 1
        if _lean_accepts(_script(goal, candidate)):
            return _Hit(attempts, candidate)
    return _Hit(attempts, None)


def _lean_accepts(source: str) -> bool:
    """True only when Lean accepts ``source`` and it has no sorry, admit, or axiom."""
    result: CheckResult = check_snippet(source)
    return result.accepted


def _script(goal: _Goal, candidate: str) -> str:
    """Init theorem. The candidate is the proof script after ``by``."""
    return f"theorem {goal.name} (n : Nat) : {goal.statement} := by\n  {candidate}\n"


def _linked(proof: str, statement: str) -> str:
    """The accepted proof of ``nat_add_zero``, then a typed use of that name.

    ``proof`` is a script Lean already accepted. A rejected or ``sorry``
    script is never passed here.
    """
    return (
        "theorem nat_add_zero (n : Nat) : n + 0 = n := by\n"
        f"  {proof}\n"
        "\n"
        f"theorem uses_lemma (m : Nat) : {statement} := by\n"
        "  exact nat_add_zero m\n"
    )
