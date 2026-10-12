# Typed demo

Agents start at `README.md`, then `SPEC.md`. This page is only how to run the Init-only Lean checks.

Minimum of the scale spec in `SPEC.md`, with Lean as the kernel. The Nat scripts are planted. That measures the process pool. The search measurement is spec §6, and it is not this demo.

Lean typechecks every candidate. Budget is a knob. A checked proof can be used at its type. A use at the wrong type is refused. A false goal stays open.

The printed run is unfitted search at seed $0$ for budgets $0, 1, 2, 4$, then one call `run_typed(seed=0, budget=4, rounds=1, width=32)`. That call records whether the `Nat` add-zero goal was accepted, whether `exact nat_add_zero` was accepted for the matching corollary, whether the mismatched goal $m + 1 = m$ was rejected, and whether the open false goal stayed unproved. An unproved open goal is a successful demo: the process exits $0$. This is not a proof of the Four Colour Theorem. mathlib is not built.

## Run

From `compute/bitter-prover`, with the project virtualenv:

```bash
cd /Users/fulkanjou/GraphColour/compute/bitter-prover && /Users/fulkanjou/GraphColour/.venv/bin/python -m bitter_prover.typed_demo
```

The module writes `typed-last-run.json` beside this file.

## Limit

The facts are Init-only `Nat` statements. This demo does not load the Kempe files and does not call Pantograph, because those need a mathlib build. That wiring is the next scale step, not this demo.

Feasibility of this Init-only slice: **High**. Feasibility of the mathlib wiring: **Medium**.

## Many cores

`bitter_prover.scaled` runs one Lean process per worker. Each process is single-threaded, so a Studio's cores stay on separate files. The shell script reads `hw.ncpu` and runs 2000 goals:

```bash
./run-studio.sh
```

A longer run is `--goals` or `--budget` on `python -m bitter_prover.scaled`. The summary is `runs/studio/summary.json`. False goals are in the batch. A run that proves one of them is a kernel failure, not a success.
