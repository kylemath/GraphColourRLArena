# Checker-trained proof search

Design contract for agents working in `compute/bitter-prover/`. The README is the map. This file is the rule. If a run and this file disagree, fix the run or change this file in the same commit. Do not leave both stories standing.

Claim labels follow `SolvingFrameworkPlan/START-HERE.md`. A derivation a kernel accepts is **[computed]** relative to that kernel. It is not a status word for the Four Colour Theorem. Nothing in this directory is **[compiled]** in the planemap module audit.

## 1. Priority

Scaling comes before a clean result on a fixed budget.

A change earns its place when one of these becomes a larger number and the kernel is still the only judge:

- search visits (`budget`)
- passes that fold accepted proofs back into the library (`rounds`)
- how far forward synthesis is allowed to run (`synth_depth`), on a kernel where synthesis can form new steps
- the policy's capacity (`width`)
- how many checked lemmas the library holds
- how many checker processes run at once (`workers`)

The exam solve rate, and the size of any accepted proof, are reported as functions of those numbers. A flat curve is a finished measurement. The numbers are not retuned after seeing whether the exam closed.

These are not scale knobs: a hand-built "one frontier step" feature, a curriculum sorted by a proof already in hand, a budget chosen so an untrained run dies and a trained run fits, a candidate list that already contains the human script.

## 2. Goal

Produce a proof of the Four Colour Theorem that a kernel accepts, together with the compute settings of the search that found it.

The proof term is the result. The model that guided the search is not part of the proof and it is not evidence. Finite checks of coloured graphs are not proofs.

The search is not given a route. Vacancy induction, discharging, Kempe chains, belts, and rank functions may appear inside a found proof. They are not the action script, and they are not to be copied out of `SolvingFrameworkPlan/` into the candidate list.

## 3. What "a link" is

Two pieces follow one from another only when the kernel says the type fits.

On the Horn layer, a child must prove the exact premise atom named by the clause. That is name-matching. The clauses themselves are uninterpreted axioms. A finished Horn tree is a checked assembly of those names. It is not a proof that the mathematics follows.

On the Lean layer, a tactic or a term is accepted only when the Lean binary accepts the text it was given. `exact nat_add_zero m` is a link when the goal is `m + 0 = m`. The same term on `m + 1 = m` is refused. `sorry`, `admit`, and a fresh `axiom` are refused even if the process exits 0. The checker does not delete a failing tactic and try the remainder.

The layer that is not built yet uses Lean's elaborator through [Pantograph](https://arxiv.org/html/2410.16429): a proposed step returns new goals or an error. A `have` counts only when its type checks. That is the link between lemmas. Reimplementing a type theory in Python is out of scope.

## 4. Reward and data

For a goal $G$ and a finished trace $\tau$,

$$
R(G, \tau) =
\begin{cases}
1 & \text{if the kernel accepts \(\tau\) as a proof of \(G\)}, \\
0 & \text{otherwise}.
\end{cases}
$$

There is no label for the correct next step except a trace the kernel has already accepted. Three sources:

1. **Library traces.** Proofs the kernel has accepted. They may be used as imitation data. They are not inserted as the only successful action for a goal the search is being graded on.
2. **Forward synthesis.** Legal steps, up to a depth or visit cap. If the exam falls out, that is a result.
3. **Hindsight.** A search aimed at a hard goal that closed some other goal. Install that goal as a lemma only after the kernel accepts it. Unfolding the lemma must recover a derivation the kernel accepts, so the shortcut cannot prove more than the derivation did.

The Four Colour Theorem is a standing exam, not the training distribution. It pays reward only if a search closes it.

## 5. What is running

### Horn sketch

`bitter_prover/kernel.py`, `iterate.py`, `python -m bitter_prover`.

Uninterpreted clauses for a five-colour outline. `five_colour` is in the closure. `four_colour` has trap clauses only and must stay unproved. Tests lock monotonicity of budget on one seed, rejection of a forged lemma, and refusal of the open atom. They do not lock a solve rate.

**[computed]** Unfitted search on seed 0: budgets 8, 32, and 128 miss `five_colour`; budgets 512 and above find it, in 317 visits. The curve is flat after 512 because the Horn closure is finite. That flatness is the signal to stop enriching this theory.

### Init-only Lean

`lean_kernel.py`, `typed_loop.py`, `scaled.py`. Details of the commands are in `TYPED.md`.

`check_snippet` writes a temp file and runs `lean`. `check_many` uses one process per worker and sets `LEAN_NUM_THREADS=1` in each, so cores spend themselves on separate files. `run-studio.sh` sets workers from `hw.ncpu` and runs 2000 goals.

The candidate list contains the scripts that prove the true goals (`rfl`, `simp`, `exact Nat.*`, `omega`). A larger budget walks further down a seeded shuffle of that list. False goals (`n + 1 = n`, `n * 1 = 0`) stay open. A use of an accepted theorem at a matching type is accepted. A use at `m + 1 = m` is rejected.

**[computed]** On 14 workers, 200 goals, budget 12: 2561 checks in 24 seconds, about 107 checks a second. All 160 true goals accepted, all 160 matching uses accepted, 0 of 40 false goals accepted, the bad link rejected. This is a pool measurement. It is not a search measurement, because the successful scripts were planted.

## 6. The next build: a known proof, withheld

Do this before any run aimed at the Four Colour Theorem.

### Suite

Freeze these names. Each must compile in the planemap checkout with no `sorry` before it is a search goal. If it does not compile, it is not in the suite. Do not repair it by adding an axiom.

| Order | Name | File | Why this one |
|---|---|---|---|
| 1 | `exists_missing_colour` | `lean4/KempeReconfiguration/KempeReconfiguration/FiveColor.lean` | Short. A `Finset (Fin 5)` of size at most 4 omits a colour |
| 2 | `five_color_degree_at_most_four` | same file | Uses the colouring lemmas. Still a known proof |
| 3 | `five_color_degree_five` | `FiveColorDeg5.lean` | Only after 1 and 2 move under search, and only if the file has no `sorry` |

The statement given to search is the type. The script in the file is not an action and is not a training target for that goal. Other accepted library lemmas may be premises the elaborator can apply. That is the library. It is not the answer script.

### Kernel

Pantograph against `/Users/fulkanjou/mathlib4-planemap`, or a Lean version pin that this checkout actually builds. A tactic call goes to Lean's elaborator. Failure returns messages and no next goals. Success returns goals. The run ends when no goals remain and the kernel accepts the term.

`have` is allowed. It counts only when its type checks. The resulting subgoals are ordinary goals.

### Knobs

The same as §1, plus `workers`. One Lean process per worker until a measurement shows the machine is out of memory. Publish solve rate against budget for each frozen name. A name that stays open at the stated cap is reported open.

### Done

- `exists_missing_colour` has a published curve, and the human script was not in the action list.
- A candidate that is ill-typed scores 0.
- `sorry` does not score 1.
- A false strengthening, if one is included as a control, stays open.
- The Four Colour Theorem is not among the goals of this build.

### If the first name does not fall

Stop. Do not add the human script. Do not point the same loop at `five_color_degree_five` or at the Four Colour Theorem. Report the curve and the cap. The next change is more compute or a larger legal action space, and it has to be declared as such.

## 7. What this plan refuses to treat as progress

- A parameter changed after seeing the exam, in order to open a gap between two runs.
- A feature added so the policy generalises one inductive step and no further.
- A colouring model, however accurate, until a kernel accepts a proof.
- A finite sweep reported as evidence that the exam is true.
- A flat curve answered by writing the missing script into the candidate list.
- The Nat pool, or the Horn closure, reported as a search for a proof.
- A Lean port of Gonthier's Coq proof treated as a proof this search found. That development, where it exists, is a formalisation of the known census argument. The maintained machine-checked proof is `rocq-community/fourcolor`.

## 8. Feasibility

- Keeping the Horn tests and the Init-only pool green: **High**.
- Recovering `exists_missing_colour` by tactic search with the script withheld: **Medium**. The mathematics is short. The work is the Pantograph wiring and an action space that does not contain the file's proof.
- Recovering `five_color_degree_at_most_four` the same way, after the first curve moves: **Medium**.
- A kernel-accepted proof of the Four Colour Theorem from this loop: **Low** on any near horizon, **Medium** as a programme that may emit a very large proof or a flat curve at a stated cap.
- A short human narrative falling out of the search: **Low**. The plan does not require one.

## 9. Agent checklist

1. Read this file and `README.md`. Run `python -m unittest discover -s tests -q` from this directory before changing the loop.
2. Do not edit another team's pages under `SolvingFrameworkPlan/`. This directory is the prover.
3. A new goal is a type plus a kernel. If you need a statement that is not already a compiling theorem, say so and stop.
4. Record the command, the commit, the budget, the worker count, and the solve rate. A capped miss is unproved.
5. Leave the Four Colour Theorem off the goal list until §6.1 has a curve.
