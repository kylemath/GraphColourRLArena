"""Forward synthesis of accepted proofs.

Apply every clause whose premises are already derived, and keep the
conclusion when the resulting derivation is within ``max_depth``. The
returned proofs are exactly the cold-start corpus: each one passes the
kernel because it was built by applying clauses of the theory.
"""

from __future__ import annotations

from bitter_prover.kernel import Proof, Theory, proof_depth


def synthesize(theory: Theory, max_depth: int) -> dict[str, Proof]:
    """Derive every atom reachable at depth at most ``max_depth``."""
    proved: dict[str, Proof] = {}
    progressed = True
    while progressed:
        progressed = False
        for clause in theory.clauses:
            if clause.conclusion in proved:
                continue
            if not all(premise in proved for premise in clause.premises):
                continue
            children = tuple(proved[premise] for premise in clause.premises)
            candidate = Proof(
                conclusion=clause.conclusion,
                clause=clause.name,
                premises=children,
            )
            if proof_depth(candidate) > max_depth:
                continue
            proved[clause.conclusion] = candidate
            progressed = True
    return proved
