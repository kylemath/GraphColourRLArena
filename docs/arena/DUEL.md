# Arena duel

The agent desk and the map desk. `duel.js` is the rule. A colouring score is not a proof. If this file and `bitter-prover/SPEC.md` both seem to speak for the prover, the prover spec wins.

## Desks

| Desk | Generator proposes | Critic returns |
|---|---|---|
| Agent | a `mazearena-rule-1` packet | `success`, `partial`, `fail`, or a `harder` request to the map desk |
| Map | a graph grown from a blockage | `reject`, `discard`, `incomplete`, `train`, `test`, or `unrealizable` |

Generators talk to each other. A critic talks to its own generator.

- `rule-gen` sends `map-gen` the train-map blockages where the admitted roster played a swap that did not lower `d`.
- `map-gen` sends `rule-gen` those blockages' `good[]` chains, which are the swaps that do lower `d`.
- Census test keys `c30`, `p32r11`, `p34r28`, `c60`, `six`, `trap17`, `r210` may be scored. They are never splice parents. Their dead ends stay on the agent desk.

## Round

The bars are fixed before the run. Defaults: `agree2 >= 0.95` and, when `n3 > 0`, `agree3 >= 0.9`. They are arguments of the run, not numbers refit after the scores.

The champion is the row with the higher `agree2`, then the higher `agree3`. `before` is the champion of the admitted roster. `after` is the champion once the candidate is added.

The roster is at the bar only when that champion has `agree2` at or above its bar and, when `n3 > 0`, `agree3` at or above its bar. A perfect `agree3` on a few deep states does not finish the round.

| Roster before | Roster after | Message |
|---|---|---|
| under the bar | at or above the bar | `success` to `rule-gen` |
| under the bar | no higher | `fail` to `rule-gen` |
| under the bar | higher, still under the bar | `partial` to `rule-gen` |
| at or above the bar | anything | `harder` to `map-gen` |

`partial` is the gap between "crossed the bar" and "did not move." The packet is a parent for the next rule. It is not the champion.

Search agents (`twoply`, and later a net via `netAgent`) sit in the roster beside packets. A packet does not get to replace `twoply` by declaration.

## What a blockage is

On a sampled state with `d >= 2`, the agent returns a move index into `movesOf`. The blockage stores the played chain and every chain whose swap makes `distOf` drop by `1`:

`{ map, heldOut, canon, d, picked: { pair, size, rimBits, rimMask }, lowered, good[] }`

The map generator's next graph is a completion of a `good[]` chain the roster did not play. On the first train slice those chains often have `rimBits = 0`: the swap that lowers `d` never touches the rim, and a rule that prefers a rim hit plays the decoy. A random triangulation is not a proposal.

## Map critic

| Check | Code |
|---|---|
| `checkTriangulation` fails, including `E != 3n-6` | `reject` |
| parent key is a census test map | `reject` |
| builder was exhaustive and every branch failed | `unrealizable` (a local lemma for the rule desk, not a map) |
| builder stopped and was not exhaustive | `incomplete` |
| planted `rimMask` changed | `discard` |
| admitted roster already clears the map | `discard` |
| old roster misses it and the new rule clears it | `train` |
| roster is under the bar here | `test` |

`unrealizable` is a disproof of that skeleton only. The search has to have covered every completion the rules allow. A builder that gives up has an incomplete proposal.

## Rule packet

`docs/arena/rules/shortest-double-rim.json` is the shape. Clauses are tried in order. The first clause with a hit picks the earliest `movesOf` index among its hits, and the shortest chain when `shortest` is true.

Allowed picks: `fill` (the swap wins now), `rim` (the chain meets at least `minBits` rim vertices), `any`.

Allowed packet fields: `format`, `name`, `admits`, `clauses`, `statement`, `reads`. A `script` or `tactic` field is rejected. `sorry`, `admit`, and `axiom` are rejected.

`admits` is `arena-only` or `lemma-candidate`.

## What the prover has to implement before a packet can be a lemma

Not this round. `bitter-prover/SPEC.md` §6 still comes first: `exists_missing_colour`, script withheld, solve rate against budget.

When that curve exists, a lemma enters the arena only as a packet with:

- `admits: "lemma-candidate"`
- `statement`: the type name that already compiles, with no `sorry`
- `clauses`: the hypotheses of that type, written as `fill` / `rim` / `any` picks
- no script, no tactic text, no weights

The arena runs the clauses. The kernel is the only thing that can accept the type. An arena `success` does not accept it. The Four Colour Theorem is not a statement this duel can close.

A translator inside the prover, later, may read `good[]` and propose a clause. It may not append that clause to the prover's candidate list, and it may not mark the exam solved.
