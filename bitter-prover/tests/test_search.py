"""Search controls: cycles, the open atom, and a policy that has seen a proof."""

from __future__ import annotations

import unittest

from bitter_prover.kernel import Clause, Proof, Theory, check, lemma_name, with_lemma
from bitter_prover.policy import Example, Policy, examples_from
from bitter_prover.search import prove
from bitter_prover.synthesize import synthesize
from bitter_prover.theory import EXAM, OPEN_ATOM, build_theory


class SearchTests(unittest.TestCase):
    def test_cycle_returns_unproved(self) -> None:
        theory = Theory.from_clauses(
            [
                Clause("a_from_b", ("b",), "a"),
                Clause("b_from_a", ("a",), "b"),
            ]
        )
        result = prove(theory, "a", Policy(seed=0), budget=50, seed=0)
        self.assertIsNone(result.proof)
        self.assertLessEqual(result.expansions, 50)

    def test_open_atom_has_no_derivation_even_with_a_large_budget(self) -> None:
        theory = build_theory()
        result = prove(theory, OPEN_ATOM, Policy(seed=1), budget=10_000, seed=1)
        self.assertIsNone(result.proof)
        for proof in result.hindsight.values():
            self.assertTrue(check(proof, theory))
            self.assertNotEqual(proof.conclusion, OPEN_ATOM)

    def test_imitation_of_a_known_proof_prefers_the_real_clause(self) -> None:
        theory = build_theory()
        proved = synthesize(theory, max_depth=10**6)
        policy = Policy(seed=0)
        examples = []
        seen: set[str] = set()
        for proof in proved.values():
            for example in examples_from(proof):
                if example.goal in seen or example.goal.startswith("trap_"):
                    continue
                seen.add(example.goal)
                examples.append(example)
        policy.fit(examples, theory)
        candidates = theory.by_head["euler_characteristic"]
        real = next(clause for clause in candidates if clause.name == "from_sphere")
        traps = [clause for clause in candidates if clause.name.startswith("trap_")]
        self.assertGreater(len(traps), 0)
        self.assertGreater(
            policy.logit("euler_characteristic", real),
            max(policy.logit("euler_characteristic", trap) for trap in traps),
        )
        found = prove(theory, EXAM, policy, budget=50_000, seed=0)
        self.assertIsNotNone(found.proof)
        assert found.proof is not None
        self.assertTrue(check(found.proof, theory))

    def test_a_checked_lemma_is_a_new_action(self) -> None:
        theory = build_theory()
        proved = synthesize(theory, max_depth=10**6)
        atom = "exists_deg_at_most_5"
        bare = prove(theory, atom, Policy(seed=0), budget=3, seed=0)
        self.assertIsNone(bare.proof)
        extended = with_lemma(theory, proved[atom])
        shortcut = Proof(conclusion=atom, clause=lemma_name(atom), premises=())
        self.assertTrue(check(shortcut, extended))
        policy = Policy(seed=0, width=128)
        policy.fit([Example(goal=atom, chosen=lemma_name(atom))], extended)
        found = prove(extended, atom, policy, budget=3, seed=0)
        self.assertIsNotNone(found.proof)
        assert found.proof is not None
        self.assertEqual(found.proof.clause, lemma_name(atom))
        self.assertTrue(check(found.proof, extended))
        self.assertLessEqual(found.expansions, 3)


if __name__ == "__main__":
    unittest.main()
