"""Expert iteration whose knobs are amounts of compute.

``budget`` is search visits. ``rounds`` is how many times accepted proofs
are folded back in. ``synth_depth`` caps forward synthesis. ``width`` is
the policy's hash-table size. Accepted derivations are installed as lemmas,
so later search can use them as single steps. The kernel rechecks the
certificate.

Nothing here orders goals by a proof the synthesizer has already finished.
The exam is attempted as itself. Atoms proved along the way become lemmas
only after ``check`` accepts them. A run that leaves the exam open is a
finished measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

from bitter_prover.kernel import Proof, Theory, check, lemma_name, with_lemma
from bitter_prover.policy import Example, Policy, examples_from
from bitter_prover.search import SearchResult, prove
from bitter_prover.synthesize import synthesize
from bitter_prover.theory import EXAM, OPEN_ATOM, SYNTH_DEPTH, build_theory

# Defaults are small so a laptop run finishes immediately. They are not
# fitted to a target solve rate. Raise them to spend more compute.
DEFAULT_BUDGET = 64
DEFAULT_ROUNDS = 4
DEFAULT_WIDTH = 256

# Unfitted search curve. The top end is large on purpose: it is "more
# search", not the smallest budget that happens to succeed.
CURVE_BUDGETS: tuple[int, ...] = (8, 32, 128, 512, 2048, 20000)


@dataclass(frozen=True)
class RoundRow:
    """One exam attempt after the library and the policy have been updated."""

    index: int
    exam_proved: bool
    expansions: int
    lemmas: int


@dataclass
class RunReport:
    """One seed. An exam proof is present only when ``check`` accepts it."""

    seed: int
    budget: int
    rounds: int
    synth_depth: int
    width: int
    synthetic: int
    lemmas: int
    exam_proof: Proof | None
    exam_expansions: int
    exam_round: int | None
    open_proved: bool
    open_expansions: int
    memory: int
    rows: tuple[RoundRow, ...]
    theory: Theory


@dataclass(frozen=True)
class CurveRow:
    """Unfitted search at one budget. Same seed, same theory, no training."""

    budget: int
    proved: bool
    expansions: int


@dataclass(frozen=True)
class SweepReport:
    """Learned-loop outcomes over seeds. Rates are measurements, not targets."""

    seeds: int
    budget: int
    trained_exam: int
    open_atom: int
    reports: tuple[RunReport, ...]


def run_once(
    seed: int = 0,
    budget: int = DEFAULT_BUDGET,
    rounds: int = DEFAULT_ROUNDS,
    synth_depth: int = SYNTH_DEPTH,
    width: int = DEFAULT_WIDTH,
) -> RunReport:
    """Synthesize, install lemmas, then attempt the exam for ``rounds`` passes."""
    theory = build_theory()
    synthetic = synthesize(theory, synth_depth)
    memory: dict[tuple[str, str], Example] = {}
    theory = _install(theory, synthetic.values(), memory)
    policy = Policy(seed=seed, width=width)
    policy.fit(list(memory.values()), theory)

    rows: list[RoundRow] = []
    exam_proof: Proof | None = None
    exam_expansions = 0
    exam_round: int | None = None
    for index in range(rounds):
        attempt = prove(theory, EXAM, policy, budget, seed + index * 1009)
        _require_checked(attempt, theory)
        exam_expansions = attempt.expansions
        if attempt.proof is not None:
            exam_proof = attempt.proof
            exam_round = index
            theory = _install(theory, [attempt.proof], memory)
        else:
            theory = _install(theory, attempt.hindsight.values(), memory)
        policy.fit(list(memory.values()), theory)
        rows.append(
            RoundRow(
                index=index,
                exam_proved=exam_proof is not None,
                expansions=attempt.expansions,
                lemmas=_lemma_count(theory),
            )
        )
        if exam_proof is not None:
            break

    opened = prove(theory, OPEN_ATOM, policy, budget, seed + 17)
    _require_checked(opened, theory)
    return RunReport(
        seed=seed,
        budget=budget,
        rounds=rounds,
        synth_depth=synth_depth,
        width=width,
        synthetic=len(synthetic),
        lemmas=_lemma_count(theory),
        exam_proof=exam_proof,
        exam_expansions=exam_expansions,
        exam_round=exam_round,
        open_proved=opened.proof is not None,
        open_expansions=opened.expansions,
        memory=len(memory),
        rows=tuple(rows),
        theory=theory,
    )


def search_curve(
    budgets: tuple[int, ...] = CURVE_BUDGETS,
    seed: int = 0,
    width: int = DEFAULT_WIDTH,
) -> tuple[CurveRow, ...]:
    """Same unfitted policy and the same seed, at each budget.

    No lemmas and no training. This is the search knob alone.
    """
    theory = build_theory()
    policy = Policy(seed=seed, width=width)
    rows: list[CurveRow] = []
    for budget in budgets:
        result = prove(theory, EXAM, policy, budget, seed)
        if result.proof is not None and not check(result.proof, theory):
            raise RuntimeError("curve returned an unchecked exam proof")
        rows.append(
            CurveRow(
                budget=budget,
                proved=result.proof is not None,
                expansions=result.expansions,
            )
        )
    return tuple(rows)


def run_sweep(
    seeds: int = 8,
    budget: int = DEFAULT_BUDGET,
    rounds: int = DEFAULT_ROUNDS,
    synth_depth: int = SYNTH_DEPTH,
    width: int = DEFAULT_WIDTH,
) -> SweepReport:
    """Repeat :func:`run_once`. The solve count is an observation."""
    reports = tuple(
        run_once(
            seed=seed,
            budget=budget,
            rounds=rounds,
            synth_depth=synth_depth,
            width=width,
        )
        for seed in range(seeds)
    )
    return SweepReport(
        seeds=seeds,
        budget=budget,
        trained_exam=sum(report.exam_proof is not None for report in reports),
        open_atom=sum(report.open_proved for report in reports),
        reports=reports,
    )


def _install(
    theory: Theory,
    proofs: object,
    memory: dict[tuple[str, str], Example],
) -> Theory:
    """Remember accepted steps and install a lemma for each derived atom."""
    for proof in proofs:
        _remember(memory, proof)
        if not proof.premises:
            continue
        name = lemma_name(proof.conclusion)
        if name in theory.by_name:
            continue
        theory = with_lemma(theory, proof)
        _remember(memory, Proof(conclusion=proof.conclusion, clause=name, premises=()))
    return theory


def _remember(memory: dict[tuple[str, str], Example], proof: Proof) -> None:
    for example in examples_from(proof):
        memory[(example.goal, example.chosen)] = example


def _require_checked(result: SearchResult, theory: Theory) -> None:
    for proof in result.hindsight.values():
        if not check(proof, theory):
            raise RuntimeError(f"search cached an unchecked proof of {proof.conclusion}")
    if result.proof is not None and not check(result.proof, theory):
        raise RuntimeError(f"search returned an unchecked proof of {result.proof.conclusion}")


def _lemma_count(theory: Theory) -> int:
    return sum(1 for name in theory.by_name if name.startswith("lemma:"))
