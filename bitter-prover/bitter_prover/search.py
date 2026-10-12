"""Budgeted backward search.

At the current atom the policy orders the clauses that conclude it. The
first clause whose premises are all derived within the budget wins. Trap
chains die when they reach an atom that no clause concludes. A cycle is a
failure, not a loop. Running out of visits aborts the attempt and does not
invent a proof.

Atoms proved before the budget ran out are returned even when the root is
unproved. That dictionary is the hindsight corpus.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from bitter_prover.kernel import Proof, Theory
from bitter_prover.policy import Policy


class _BudgetExhausted(Exception):
    """Internal signal. Callers see a missing proof, not this exception."""


@dataclass(frozen=True)
class SearchResult:
    """Outcome of one attempt. ``hindsight`` holds derivations found along the way."""

    proof: Proof | None
    hindsight: dict[str, Proof]
    expansions: int


def prove(theory: Theory, goal: str, policy: Policy, budget: int, seed: int) -> SearchResult:
    """Search for a derivation of ``goal``. ``budget`` counts atom visits."""
    rng = random.Random(seed)
    cache: dict[str, Proof] = {}
    failed: set[str] = set()
    expansions = 0

    def solve(atom: str, stack: frozenset[str]) -> Proof | None:
        nonlocal expansions
        if atom in cache:
            return cache[atom]
        if atom in failed or atom in stack:
            return None
        expansions += 1
        if expansions > budget:
            raise _BudgetExhausted
        candidates = theory.by_head.get(atom, ())
        if not candidates:
            failed.add(atom)
            return None
        for clause in policy.rank(atom, candidates, rng):
            if atom in clause.premises or any(premise in stack for premise in clause.premises):
                continue
            children: list[Proof] = []
            ok = True
            for premise in clause.premises:
                child = solve(premise, stack | {atom})
                if child is None:
                    ok = False
                    break
                children.append(child)
            if not ok:
                continue
            proof = Proof(conclusion=atom, clause=clause.name, premises=tuple(children))
            cache[atom] = proof
            return proof
        failed.add(atom)
        return None

    try:
        found = solve(goal, frozenset())
    except _BudgetExhausted:
        found = None
    return SearchResult(proof=found, hindsight=dict(cache), expansions=expansions)
