"""Scale knobs and soundness of the learned loop.

Solve rates are not targets. A run may leave the exam open.
"""

from __future__ import annotations

import unittest

from bitter_prover.iterate import run_once, run_sweep, search_curve
from bitter_prover.kernel import check, walk
from bitter_prover.theory import EXAM


class IterateTests(unittest.TestCase):
    def test_search_budget_is_monotone_for_a_fixed_seed(self) -> None:
        curve = search_curve(seed=0)
        proved = [row.proved for row in curve]
        self.assertFalse(proved[0])
        self.assertTrue(proved[-1])
        self.assertEqual(proved, sorted(proved))
        self.assertGreater(curve[-1].budget, curve[0].budget)
        last = curve[-1]
        self.assertTrue(last.proved)

    def test_large_search_proof_checks_and_the_open_atom_stays_open(self) -> None:
        report = run_once(seed=0, budget=50_000, rounds=1, synth_depth=1)
        self.assertIsNotNone(report.exam_proof)
        assert report.exam_proof is not None
        self.assertTrue(check(report.exam_proof, report.theory))
        self.assertEqual(report.exam_proof.conclusion, EXAM)
        self.assertFalse(report.open_proved)
        self.assertTrue(all(not node.clause.startswith("trap_") for node in walk(report.exam_proof)))

    def test_deep_synthesis_is_allowed_to_reach_the_exam(self) -> None:
        report = run_once(seed=0, budget=50_000, rounds=1, synth_depth=10**6)
        self.assertIsNotNone(report.exam_proof)
        assert report.exam_proof is not None
        self.assertTrue(check(report.exam_proof, report.theory))
        self.assertGreater(report.lemmas, 0)
        self.assertFalse(report.open_proved)

    def test_same_seed_is_deterministic(self) -> None:
        first = run_once(seed=2, rounds=2)
        second = run_once(seed=2, rounds=2)
        self.assertEqual(first.exam_round, second.exam_round)
        self.assertEqual(first.exam_expansions, second.exam_expansions)
        self.assertEqual(first.lemmas, second.lemmas)
        self.assertEqual(first.open_proved, second.open_proved)
        self.assertEqual(
            tuple(row.exam_proved for row in first.rows),
            tuple(row.exam_proved for row in second.rows),
        )

    def test_sweep_never_proves_the_open_atom(self) -> None:
        sweep = run_sweep(seeds=3, rounds=2)
        self.assertEqual(sweep.open_atom, 0)
        for report in sweep.reports:
            self.assertFalse(report.open_proved)
            if report.exam_proof is not None:
                self.assertTrue(check(report.exam_proof, report.theory))


if __name__ == "__main__":
    unittest.main()
