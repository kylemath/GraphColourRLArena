"""Kernel and synthesis tests. No search budget is involved."""

from __future__ import annotations

import unittest

from bitter_prover.kernel import Proof, Theory, check, explain, proof_depth, with_lemma
from bitter_prover.kernel import Clause
from bitter_prover.synthesize import synthesize
from bitter_prover.theory import EXAM, OPEN_ATOM, SYNTH_DEPTH, build_theory


class KernelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.theory = build_theory()

    def test_deep_synthesis_proves_the_exam_and_checks(self) -> None:
        proved = synthesize(self.theory, max_depth=10**6)
        self.assertIn(EXAM, proved)
        self.assertTrue(check(proved[EXAM], self.theory))
        self.assertGreater(proof_depth(proved[EXAM]), SYNTH_DEPTH)
        self.assertNotIn(OPEN_ATOM, proved)
        for proof in proved.values():
            self.assertTrue(check(proof, self.theory))
            self.assertFalse(proof.clause.startswith("trap_"))

    def test_shallow_synthesis_stops_before_the_exam(self) -> None:
        proved = synthesize(self.theory, SYNTH_DEPTH)
        self.assertNotIn(EXAM, proved)
        self.assertIn("exists_deg_at_most_5", proved)
        self.assertNotIn("induction_setup", proved)
        for proof in proved.values():
            self.assertLessEqual(proof_depth(proof), SYNTH_DEPTH)

    def test_fact_checks_and_tampered_trees_do_not(self) -> None:
        fact = Proof(conclusion="sphere", clause="sphere", premises=())
        self.assertTrue(check(fact, self.theory))
        missing = Proof(conclusion=EXAM, clause="assemble", premises=())
        self.assertFalse(check(missing, self.theory))
        unknown = Proof(conclusion="sphere", clause="no_such_clause", premises=())
        self.assertFalse(check(unknown, self.theory))
        swapped = Proof(
            conclusion="euler_characteristic",
            clause="from_sphere",
            premises=(Proof(conclusion="tetrahedron", clause="tetrahedron", premises=()),),
        )
        self.assertFalse(check(swapped, self.theory))

    def test_explain_names_the_broken_node(self) -> None:
        with self.assertRaises(Exception) as caught:
            explain(Proof(conclusion="sphere", clause="missing", premises=()), self.theory)
        self.assertIn("unknown clause", str(caught.exception))

    def test_a_forged_lemma_certificate_is_rejected(self) -> None:
        forged = Theory.from_clauses(
            [*self.theory.clauses, Clause("lemma:five_colour", (), EXAM)],
            {"lemma:five_colour": Proof("sphere", "sphere", ())},
        )
        shortcut = Proof(conclusion=EXAM, clause="lemma:five_colour", premises=())
        self.assertFalse(check(shortcut, forged))

    def test_with_lemma_rejects_an_unchecked_proof(self) -> None:
        bogus = Proof(conclusion=EXAM, clause="assemble", premises=())
        with self.assertRaises(Exception):
            with_lemma(self.theory, bogus)

    def test_duplicate_clause_name_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Theory.from_clauses(
                [Clause("same", (), "a"), Clause("same", (), "b")]
            )


if __name__ == "__main__":
    unittest.main()
