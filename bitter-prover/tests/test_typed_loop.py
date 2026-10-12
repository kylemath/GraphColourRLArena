"""Scale knobs for the Init-only Lean loop.

These tests call Lean. A run may leave the open equation unproved.
Solve rates are not targets.
"""

from __future__ import annotations

import unittest

from bitter_prover.typed_loop import CANDIDATES, Policy, run_typed, search_budget_curve


class TypedLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.full = run_typed(seed=0, budget=4, rounds=1, width=32)
        cls.again = run_typed(seed=0, budget=4, rounds=1, width=32)

    def test_sorry_does_not_count_as_proving(self) -> None:
        # The weight only puts the sorry script first. It is not a scale fit.
        policy = Policy(seed=0, width=32)
        policy.weights["sorry"] = 1.0
        report = run_typed(seed=0, budget=1, rounds=1, width=32, policy=policy)
        self.assertFalse(report.provable_accepted)
        self.assertIsNone(report.provable_proof)
        self.assertEqual(report.provable_attempts, 1)
        self.assertFalse(report.lemma_use_accepted)
        self.assertFalse(report.bad_link_accepted)
        self.assertFalse(report.open_accepted)

    def test_budget_zero_proves_nothing(self) -> None:
        report = run_typed(seed=0, budget=0, rounds=1, width=32)
        self.assertFalse(report.provable_accepted)
        self.assertIsNone(report.provable_proof)
        self.assertEqual(report.provable_attempts, 0)
        self.assertFalse(report.open_accepted)
        self.assertEqual(report.open_attempts, 0)
        self.assertFalse(report.lemma_use_accepted)
        self.assertFalse(report.bad_link_accepted)

    def test_seed_zero_full_budget_accepts_a_proof_that_is_not_sorry(self) -> None:
        report = self.full
        self.assertEqual(report.seed, 0)
        self.assertEqual(report.budget, 4)
        self.assertEqual(report.rounds, 1)
        self.assertEqual(report.width, 32)
        self.assertTrue(report.provable_accepted)
        self.assertGreater(report.provable_attempts, 0)
        self.assertLessEqual(report.provable_attempts, 4)
        self.assertIn(report.provable_proof, CANDIDATES)
        self.assertNotEqual(report.provable_proof, "sorry")
        self.assertEqual(len(report.rows), 1)
        self.assertTrue(report.rows[0].provable_accepted)

    def test_search_budget_curve_is_monotone(self) -> None:
        curve = search_budget_curve((0, 1, 4), seed=0)
        self.assertEqual(tuple(point.budget for point in curve), (0, 1, 4))
        self.assertFalse(curve[0].proved)
        self.assertEqual(curve[0].attempts, 0)
        self.assertTrue(curve[-1].proved)
        proved = [point.proved for point in curve]
        self.assertEqual(proved, sorted(proved))
        for earlier, later in zip(curve, curve[1:]):
            if earlier.proved:
                self.assertTrue(later.proved)
                self.assertEqual(later.attempts, earlier.attempts)
            else:
                self.assertGreaterEqual(later.attempts, earlier.attempts)

    def test_installed_lemma_is_usable_when_the_goal_was_proved(self) -> None:
        report = self.full
        self.assertTrue(report.provable_accepted)
        self.assertTrue(report.lemma_use_accepted)
        self.assertTrue(report.rows[0].lemma_use_accepted)

    def test_bad_link_is_refused(self) -> None:
        self.assertFalse(self.full.bad_link_accepted)
        self.assertFalse(self.full.rows[0].bad_link_accepted)

    def test_open_exam_stays_open_at_full_budget(self) -> None:
        report = self.full
        self.assertEqual(report.budget, 4)
        self.assertFalse(report.open_accepted)
        self.assertEqual(report.open_attempts, len(CANDIDATES))
        self.assertFalse(report.rows[0].open_accepted)

    def test_same_seed_matches(self) -> None:
        self.assertEqual(self.full, self.again)


if __name__ == "__main__":
    unittest.main()
