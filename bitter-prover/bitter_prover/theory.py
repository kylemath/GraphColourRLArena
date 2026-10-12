"""Miniature Horn theory used as the phase-0 library.

Real clauses are an uninterpreted dependency sketch of the five-colour
induction: Euler, a vertex of degree at most 5, the degree-4 extension,
one compressed degree-5 step, then induction. The checker confirms that
these names compose. It does not replay the graph argument.

``five_colour`` is the exam inside the Horn closure. ``four_colour`` has
trap clauses only, so a sound search leaves it unproved. Trap chains give
every real head several legal moves that fail after a long walk.

Nothing in this module is a formalisation of the Four Colour Theorem.
"""

from __future__ import annotations

from bitter_prover.kernel import Clause, Theory

# Visits spent inside one wrong clause before the search gives up on it.
TRAP_LENGTH = 12
# Wrong clauses attached to each real conclusion, and to ``four_colour``.
TRAP_COUNT = 3

# Default cap on forward synthesis, in derivation depth. A scale knob:
# raising it spends more compute. If the cap is high enough for the exam
# to fall out, forward search has found a proof.
SYNTH_DEPTH = 4

EXAM = "five_colour"
OPEN_ATOM = "four_colour"

# Atoms of the real five-colour sketch, shallowest first is computed later.
FACTS: tuple[str, ...] = (
    "sphere",
    "maximal_planar",
    "palette_of_five",
    "tetrahedron",
)

# (name, premises, conclusion). Each line is an uninterpreted axiom.
REAL_STEPS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("from_sphere", ("sphere",), "euler_characteristic"),
    ("charge_from_euler", ("euler_characteristic", "maximal_planar"), "charge_sum_12"),
    ("low_degree", ("charge_sum_12",), "exists_deg_at_most_5"),
    ("colour_base", ("tetrahedron", "palette_of_five"), "base_coloured"),
    (
        "induction_ready",
        ("exists_deg_at_most_5", "base_coloured"),
        "induction_setup",
    ),
    (
        "apply_ih",
        ("induction_setup", "palette_of_five"),
        "smaller_graph_coloured",
    ),
    ("split_deg4", ("smaller_graph_coloured",), "deg_at_most_4_case"),
    (
        "missing_colour",
        ("deg_at_most_4_case", "palette_of_five"),
        "extend_deg_at_most_4",
    ),
    ("split_deg5", ("smaller_graph_coloured",), "deg_5_link"),
    (
        "pick_pair",
        ("deg_5_link", "maximal_planar"),
        "nonadjacent_neighbours",
    ),
    (
        "kempe_identify",
        ("nonadjacent_neighbours", "palette_of_five"),
        "deg_5_reducible",
    ),
    (
        "join_cases",
        ("extend_deg_at_most_4", "deg_5_reducible"),
        "inductive_step",
    ),
    ("assemble", ("induction_setup", "inductive_step"), EXAM),
)

SIDE_COUNT = 6


def build_theory() -> Theory:
    """Return the phase-0 theory: real sketch, side lemmas, and traps."""
    clauses: list[Clause] = [Clause(name, (), name) for name in FACTS]
    clauses.extend(
        Clause(name, premises, conclusion) for name, premises, conclusion in REAL_STEPS
    )
    for index in range(SIDE_COUNT):
        fact = f"side_fact_{index}"
        lemma = f"side_lemma_{index}"
        clauses.append(Clause(fact, (), fact))
        clauses.append(Clause(f"side_step_{index}", (fact,), lemma))

    heads = [clause.conclusion for clause in clauses]
    heads.append(OPEN_ATOM)
    serial = 0
    trap_clauses: list[Clause] = []
    for head in heads:
        trap_clauses.extend(_traps(head, serial))
        serial += TRAP_COUNT
    clauses.extend(trap_clauses)
    return Theory.from_clauses(clauses)


def real_conclusions() -> tuple[str, ...]:
    """Facts and real-step conclusions, exam included, side theory excluded."""
    return FACTS + tuple(step[2] for step in REAL_STEPS)


def _traps(conclusion: str, serial_start: int) -> list[Clause]:
    """``TRAP_COUNT`` chains that may be selected for ``conclusion`` and then die."""
    clauses: list[Clause] = []
    serial = serial_start
    for _ in range(TRAP_COUNT):
        atoms = tuple(f"trap_{serial}_atom_{i}" for i in range(TRAP_LENGTH))
        clauses.append(Clause(f"trap_{serial}_enter", (atoms[0],), conclusion))
        for index in range(TRAP_LENGTH - 1):
            clauses.append(
                Clause(
                    f"trap_{serial}_step_{index}",
                    (atoms[index + 1],),
                    atoms[index],
                )
            )
        serial += 1
    return clauses
