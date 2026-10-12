"""Horn-clause kernel.

A clause says: from these premise atoms, conclude this atom. A proof is a
tree of clause applications. ``check`` accepts the tree exactly when every
node names a clause of the theory and the children prove that clause's
premises. Search is not an input to the checker.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Clause:
    """One Horn clause. Empty premises make a fact."""

    name: str
    premises: tuple[str, ...]
    conclusion: str


@dataclass(frozen=True)
class Proof:
    """A candidate derivation. Validity is decided by :func:`check`."""

    conclusion: str
    clause: str
    premises: tuple[Proof, ...]


class CheckError(Exception):
    """Raised by :func:`explain` when a tree is not a derivation."""


@dataclass(frozen=True)
class Theory:
    """Clauses, plus shortcut lemmas whose certificates the kernel rechecks.

    A lemma is a new fact-clause. Its certificate is a derivation the kernel
    already accepted. Using the lemma is one search step. ``explain`` unfolds
    the certificate, so a shortcut cannot prove more than the derivation did.
    """

    clauses: tuple[Clause, ...]
    by_head: dict[str, tuple[Clause, ...]]
    by_name: dict[str, Clause]
    certificates: dict[str, Proof]

    @staticmethod
    def from_clauses(
        clauses: list[Clause],
        certificates: dict[str, Proof] | None = None,
    ) -> Theory:
        by_name: dict[str, Clause] = {}
        by_head: dict[str, list[Clause]] = {}
        for clause in clauses:
            if clause.name in by_name:
                raise ValueError(f"duplicate clause name {clause.name}")
            by_name[clause.name] = clause
            by_head.setdefault(clause.conclusion, []).append(clause)
        return Theory(
            clauses=tuple(clauses),
            by_head={head: tuple(group) for head, group in by_head.items()},
            by_name=by_name,
            certificates={} if certificates is None else dict(certificates),
        )


def check(proof: Proof, theory: Theory) -> bool:
    """Return whether ``proof`` is a derivation in ``theory``."""
    try:
        explain(proof, theory)
    except CheckError:
        return False
    return True


def explain(proof: Proof, theory: Theory) -> None:
    """Accept ``proof`` or raise :class:`CheckError` at the first bad node."""
    _explain(proof, theory, ())


def _explain(proof: Proof, theory: Theory, stack: tuple[str, ...]) -> None:
    clause = theory.by_name.get(proof.clause)
    if clause is None:
        raise CheckError(f"unknown clause {proof.clause}")
    if clause.conclusion != proof.conclusion:
        raise CheckError(
            f"clause {clause.name} concludes {clause.conclusion}, "
            f"not {proof.conclusion}"
        )
    if len(clause.premises) != len(proof.premises):
        raise CheckError(
            f"clause {clause.name} has {len(clause.premises)} premises, "
            f"the tree has {len(proof.premises)}"
        )
    for expected, child in zip(clause.premises, proof.premises):
        if child.conclusion != expected:
            raise CheckError(
                f"clause {clause.name} expects premise {expected}, "
                f"got {child.conclusion}"
            )
        _explain(child, theory, stack)
    certificate = theory.certificates.get(proof.clause)
    if certificate is None:
        return
    if proof.clause in stack:
        raise CheckError(f"lemma cycle at {proof.clause}")
    if certificate.conclusion != proof.conclusion:
        raise CheckError(
            f"certificate for {proof.clause} concludes {certificate.conclusion}"
        )
    _explain(certificate, theory, stack + (proof.clause,))


def lemma_name(conclusion: str) -> str:
    """Clause name of the shortcut for an accepted derivation of ``conclusion``."""
    return f"lemma:{conclusion}"


def with_lemma(theory: Theory, proof: Proof) -> Theory:
    """Return ``theory`` plus a one-step clause for a derivation it already accepts.

    A second call for the same conclusion returns ``theory`` unchanged. The
    action space grows only when the kernel has accepted a new atom.
    """
    if not check(proof, theory):
        raise CheckError(f"refusing a lemma for an unchecked proof of {proof.conclusion}")
    if not proof.premises:
        return theory
    name = lemma_name(proof.conclusion)
    if name in theory.by_name:
        return theory
    clause = Clause(name, (), proof.conclusion)
    certificates = dict(theory.certificates)
    certificates[name] = proof
    return Theory.from_clauses([*theory.clauses, clause], certificates)


def proof_depth(proof: Proof) -> int:
    """Depth of a derivation. A fact has depth 1."""
    if not proof.premises:
        return 1
    return 1 + max(proof_depth(child) for child in proof.premises)


def walk(proof: Proof) -> list[Proof]:
    """Preorder listing of a derivation, sharing a node only once."""
    seen: set[int] = set()
    order: list[Proof] = []

    def visit(node: Proof) -> None:
        if id(node) in seen:
            return
        seen.add(id(node))
        order.append(node)
        for child in node.premises:
            visit(child)

    visit(proof)
    return order


def format_proof(proof: Proof, indent: int = 0) -> str:
    """Render a derivation as an indented clause trace."""
    pad = "  " * indent
    line = f"{pad}{proof.conclusion} via {proof.clause}"
    children = [format_proof(child, indent + 1) for child in proof.premises]
    return "\n".join([line, *children])
