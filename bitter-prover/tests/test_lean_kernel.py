"""Lean kernel adapter tests. These call the real ``lean`` binary."""

from __future__ import annotations

import os
import unittest

from bitter_prover.lean_kernel import CheckResult, check_snippet

_NAT_ADD_ZERO = "theorem nat_add_zero (n : Nat) : n + 0 = n := by rfl"
_USES_LEMMA = "theorem uses_lemma (m : Nat) : m + 0 = m := by exact nat_add_zero m"
_BAD_LINK = "theorem bad_link (m : Nat) : m + 1 = m := by exact nat_add_zero m"


class LeanKernelTests(unittest.TestCase):
    def test_nat_add_zero_is_accepted(self) -> None:
        result = check_snippet(_NAT_ADD_ZERO)
        self.assertIsInstance(result, CheckResult)
        self.assertTrue(result.accepted, result.message)
        self.assertFalse(result.has_sorry)
        self.assertEqual(result.returncode, 0)

    def test_sorry_is_rejected(self) -> None:
        result = check_snippet("theorem bad (n : Nat) : n + 0 = n := by sorry")
        self.assertFalse(result.accepted)
        self.assertTrue(result.has_sorry)

    def test_false_equation_is_rejected(self) -> None:
        result = check_snippet("theorem bad (n : Nat) : n + 1 = n := by intro n; rfl")
        self.assertFalse(result.accepted, result.message)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(result.has_sorry)

    def test_typed_link_is_accepted(self) -> None:
        source = f"{_NAT_ADD_ZERO}\n\n{_USES_LEMMA}\n"
        result = check_snippet(source)
        self.assertTrue(result.accepted, result.message)
        self.assertFalse(result.has_sorry)
        self.assertEqual(result.returncode, 0)

    def test_mistyped_link_is_rejected(self) -> None:
        source = f"{_NAT_ADD_ZERO}\n\n{_BAD_LINK}\n"
        result = check_snippet(source)
        self.assertFalse(result.accepted, result.message)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(result.has_sorry)

    def test_admit_is_rejected(self) -> None:
        result = check_snippet("theorem bad (n : Nat) : n + 0 = n := by admit")
        self.assertFalse(result.accepted)
        self.assertTrue(result.has_sorry)

    def test_fresh_axiom_is_rejected(self) -> None:
        result = check_snippet("axiom forged : True\n")
        self.assertFalse(result.accepted, result.message)
        self.assertFalse(result.has_sorry)
        self.assertEqual(result.returncode, 0)

    def test_redundant_intro_is_rejected_as_lean_wrote_it(self) -> None:
        result = check_snippet(
            "theorem nat_add_zero (n : Nat) : n + 0 = n := by intro n; rfl"
        )
        self.assertFalse(result.accepted, result.message)
        self.assertNotEqual(result.returncode, 0)

    def test_necessary_intro_is_accepted(self) -> None:
        source = "theorem forall_refl : ∀ n : Nat, n = n := by intro n; rfl\n"
        result = check_snippet(source)
        self.assertTrue(result.accepted, result.message)
        self.assertEqual(result.returncode, 0)

    def test_missing_binary_is_rejected(self) -> None:
        previous = os.environ.get("LEAN_BIN")
        os.environ["LEAN_BIN"] = "/nonexistent/lean-binary"
        try:
            result = check_snippet("theorem t : True := trivial")
        finally:
            if previous is None:
                os.environ.pop("LEAN_BIN", None)
            else:
                os.environ["LEAN_BIN"] = previous
        self.assertFalse(result.accepted)
        self.assertEqual(result.returncode, -1)
        self.assertIn("not found", result.message)


if __name__ == "__main__":
    unittest.main()
