"""Parallel Lean search: same answers at two pool sizes, false goals stay open."""

from __future__ import annotations

import unittest

from bitter_prover.scaled import run_scaled


class ScaledTests(unittest.TestCase):
    def test_two_workers_match_one_and_false_goals_stay_open(self) -> None:
        single = run_scaled(goals=6, budget=12, rounds=1, workers=1, seed=0)
        double = run_scaled(goals=6, budget=12, rounds=1, workers=2, seed=0)
        self.assertEqual(single.open_proved, 0)
        self.assertEqual(double.open_proved, 0)
        self.assertGreater(single.open_goals, 0)
        self.assertEqual(
            [(item.name, item.accepted, item.script) for item in single.results],
            [(item.name, item.accepted, item.script) for item in double.results],
        )
        self.assertGreater(single.proved, 0)
        self.assertEqual(single.bad_links_accepted, 0)
        self.assertEqual(single.lemma_uses_accepted, single.proved)

    def test_larger_budget_keeps_every_proof_a_smaller_budget_found(self) -> None:
        small = run_scaled(goals=8, budget=1, rounds=1, workers=2, seed=1)
        large = run_scaled(goals=8, budget=12, rounds=1, workers=2, seed=1)
        small_proved = {item.name for item in small.results if item.accepted}
        large_proved = {item.name for item in large.results if item.accepted}
        self.assertTrue(small_proved <= large_proved)
        self.assertEqual(large.open_proved, 0)
        self.assertGreater(large.proved, small.proved)


if __name__ == "__main__":
    unittest.main()
