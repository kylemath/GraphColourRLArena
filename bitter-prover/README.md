# Checker-trained prover

Read this, then `SPEC.md`. The spec wins if they disagree.

This directory is a search loop whose only judge is a proof checker. The long-run exam is a kernel-accepted proof of the Four Colour Theorem. That exam is allowed to stay open. The work now is to make more compute change what the checker is asked to accept, on theorems whose proofs are already known and are **withheld from the search**.

## What an agent must not do

- Do not put the human proof script into the candidate list. The Nat runner does this (`rfl`, `exact Nat.add_zero`, `omega`). That measures the process pool. It does not measure search.
- Do not add a feature so that a small run climbs. In particular, do not add a bit that means "every premise is already proved", and do not sort a curriculum by a proof the synthesizer has already built.
- Do not edit a snippet after Lean rejects it and check the edited text. Lean judges the text it was given.
- Do not report a finite sweep, or a closed Nat goal, as evidence for the Four Colour Theorem.
- Do not retune `budget`, trap length, or the script list after seeing whether the exam closed.
- Do not treat `sorry`, `admit`, or a new `axiom` as acceptance.

## Three layers, in the order they were built

| Layer | What it checks | What it is for | Command |
|---|---|---|---|
| Horn sketch | Name-matching of uninterpreted clauses (`kernel.py`) | The training loop: budget, rounds, synthesis depth, width, lemmas | `python -m bitter_prover` |
| Init-only Lean | The `lean` binary on the text it was given | Type-correct use of a lemma, and many checks at once | `python -m bitter_prover.typed_demo`, `python -m bitter_prover.scaled` |
| Known theorems, proof withheld | Lean, through Pantograph, against `mathlib4-planemap` | The first measurement of search. **Not built yet.** Spec §6 | — |

The Four Colour Theorem is not a goal in any of these layers. Gonthier's checked proof is in Coq (`rocq-community/fourcolor`). It is not a script to imitate.

## Layout

| Path | Role |
|---|---|
| `SPEC.md` | Design contract. Read §1, §5, and §6 before writing code |
| `TYPED.md` | How to run the Init-only Lean demos |
| `bitter_prover/kernel.py` | Horn checker. Does not know mathematics |
| `bitter_prover/iterate.py` | Horn expert iteration |
| `bitter_prover/lean_kernel.py` | `check_snippet`, `check_many`. Calls `lean` |
| `bitter_prover/typed_loop.py` | One Nat goal, a false goal, a typed `exact`, a mismatched `exact` |
| `bitter_prover/scaled.py` | One Lean process per worker, `LEAN_NUM_THREADS=1` |
| `run-studio.sh` | 2000 goals, workers = `hw.ncpu` |
| `tests/` | What is locked. Solve rates are not targets |
| `runs/` | Outputs. Gitignored |

Lean on this machine is invoked through elan. The checker pins an installed toolchain when elan has no default. It does not change the user's elan default. The planemap checkout is `/Users/fulkanjou/mathlib4-planemap` (Lean 4.35). The Kempe sources are `lean4/KempeReconfiguration/`.

## Run

From this directory, with the project virtualenv at `../../.venv` (stdlib only):

```bash
../../.venv/bin/python -m unittest discover -s tests -q
../../.venv/bin/python -m bitter_prover.typed_demo
../../.venv/bin/python -m bitter_prover.scaled --workers 2 --goals 8 --budget 4
./run-studio.sh
```

`run-studio.sh` is a throughput run of planted Nat scripts. It is the shape of the pool. It is not the next research run.

## Next task

Spec §6. Freeze `exists_missing_colour`, then `five_color_degree_at_most_four`. Search with the proof text withheld. Publish solve rate against budget. If the first theorem does not fall, do not point the loop at the Four Colour Theorem.
