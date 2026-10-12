"""Linear policy over generic clause tokens.

Each clause name has its own weight, so the table grows when new lemmas
appear. Goal and premise tokens also land in a hash table of size ``width``.
``width`` is a capacity knob. There is no hand-built curriculum feature.
Training is cross-entropy against clauses that fired in kernel-accepted
proofs. An unfitted policy scores every clause at 0, and :meth:`Policy.rank`
shuffles those ties, so the cold policy is a uniform random order.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from bitter_prover.kernel import Clause, Proof, Theory, walk


@dataclass(frozen=True)
class Example:
    """One accepted decision: at ``goal``, the clause ``chosen`` fired."""

    goal: str
    chosen: str


class Policy:
    """Premise selector. Weights start at 0. ``width`` is the hash-table size."""

    def __init__(self, seed: int = 0, width: int = 256) -> None:
        if width < 1:
            raise ValueError("width must be positive")
        self.seed = seed
        self.width = width
        self.weights: dict[str, float] = {}
        self._fits = 0

    def logit(self, goal: str, clause: Clause) -> float:
        return sum(self.weights.get(key, 0.0) for key in feature_keys(goal, clause, self.width))

    def rank(self, goal: str, clauses: tuple[Clause, ...] | list[Clause], rng: random.Random) -> list[Clause]:
        """Highest logit first. Equal logits keep a freshly shuffled order."""
        ordered = list(clauses)
        rng.shuffle(ordered)
        ordered.sort(key=lambda clause: self.logit(goal, clause), reverse=True)
        return ordered

    def fit(
        self,
        examples: list[Example],
        theory: Theory,
        epochs: int = 40,
        lr: float = 0.5,
    ) -> None:
        """Minimise cross-entropy of the accepted clause among the candidates."""
        if not examples:
            return
        rng = random.Random(self.seed + 1000 + self._fits)
        self._fits += 1
        data = list(examples)
        for _ in range(epochs):
            rng.shuffle(data)
            for example in data:
                candidates = list(theory.by_head[example.goal])
                if len(candidates) == 1:
                    continue
                chosen = _index_of(candidates, example.chosen)
                logits = [self.logit(example.goal, clause) for clause in candidates]
                probabilities = _softmax(logits)
                for index, clause in enumerate(candidates):
                    grad = probabilities[index] - (1.0 if index == chosen else 0.0)
                    if grad == 0.0:
                        continue
                    step = lr * grad
                    for key in feature_keys(example.goal, clause, self.width):
                        self.weights[key] = self.weights.get(key, 0.0) - step


def feature_keys(goal: str, clause: Clause, width: int) -> tuple[str, ...]:
    """Clause name, plus hashed tokens of the goal and the premises.

    The clause name is exact, so a newly accepted lemma gets its own weight.
    The hash bins are shared across clauses. Which of them matter is left to
    the fit. ``width`` bounds how many bins exist.
    """
    keys = [f"clause:{clause.name}"]
    tokens = [f"goal:{goal}", f"arity:{len(clause.premises)}"]
    for premise in clause.premises:
        tokens.append(f"prem:{premise}")
        tokens.append(f"pair:{goal}>{premise}")
    for token in tokens:
        keys.append(f"bin:{stable_hash(token) % width}")
    return tuple(keys)


def stable_hash(text: str) -> int:
    """Process-stable hash. ``hash()`` is salted per interpreter."""
    value = 2166136261
    for byte in text.encode():
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def examples_from(proof: Proof) -> list[Example]:
    """Clause choices along an accepted derivation, one per concluded atom."""
    seen: set[str] = set()
    examples: list[Example] = []
    for node in walk(proof):
        if node.conclusion in seen:
            continue
        seen.add(node.conclusion)
        examples.append(Example(goal=node.conclusion, chosen=node.clause))
    return examples


def _index_of(candidates: list[Clause], chosen: str) -> int:
    for index, clause in enumerate(candidates):
        if clause.name == chosen:
            return index
    raise KeyError(chosen)


def _softmax(logits: list[float]) -> list[float]:
    peak = max(logits)
    shifted = [math.exp(value - peak) for value in logits]
    total = sum(shifted)
    return [value / total for value in shifted]
