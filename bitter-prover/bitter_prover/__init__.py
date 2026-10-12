"""Checker-trained proof search, miniature kernel.

The loop trains a policy on proofs a kernel has accepted, then searches
again. See SPEC.md. Nothing here is a proof of the Four Colour Theorem.
"""

from bitter_prover.iterate import RunReport, run_once, run_sweep, search_curve
from bitter_prover.kernel import CheckError, Clause, Proof, Theory, check, with_lemma

__all__ = [
    "CheckError",
    "Clause",
    "Proof",
    "RunReport",
    "Theory",
    "check",
    "run_once",
    "run_sweep",
    "search_curve",
    "with_lemma",
]
