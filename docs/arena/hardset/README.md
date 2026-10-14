# Hard set v1: maps that are hard for the proof (exploratory)

`hardset-v1.json` is a separate pool, apart from the held-out census test maps, which are unchanged. It holds maps
where the four-colour **proof** struggles, which is not the same as being hard for players. Every map is a sphere
triangulation with a degree-5 hole, in the arena's map format (`key, name, n, hole, link, edges, rot`). Each map also
has `family`, `source`, a one-line `why`, `meta`, a `split` (train/eval, 80/20 within each family, deterministic) and
exact distances `dist` = {N states, unfilled, hist of distance to a fill, maxD, unreachable, hardness}.

| family | maps (train/eval) | n | max distance to a fill | why hard |
|---|---|---|---|---|
| `census-tight` | 150 (120/30) | 27–31 | 2 (67), 3 (83) | a Kempe class sits exactly on the quarter floor F/N = 1/4 (frame census, Census29) |
| `glued-tight` | 50 (40/10) | 42–45 | 2 (5), 3 (45) | glued along a separating 4/5-cycle (Studio gluing run), class exactly on the quarter floor |
| `R0` | 2 (2/0) | 22, 27 | 3 | planar counterexamples to Conjecture R0 (p27#4710 hole 14; fullerene C40#14 dual hole 0) |
| `2ball` | 100 (80/20) | 30–32 | 2 (30), 3 (70) | hole word 66666 whose 2-ball (hole, neighbours, second ring) is **not D-reducible** (Studio Job 9) |

**Distances are small (≤ 3).** These maps are hard for reducibility and discharging arguments, not deep mazes.
Their value for RL is as a test of whether a policy's shortcuts survive on structurally hard holes.

**Provenance.**
- `tools/hardset_select.py GRAPHCOLOUR_DIR out.json` picks the maps from a GraphColourDeductiveSolver checkout.
- `node tools/arena_hardset.js cands.json out.json` validates each map (`checkTriangulation`) and enumerates its state
  space (`stateSpace`). The distances for all 251 census maps equal those of GraphColourDeductiveSolver
  `boundary/studio/bnd.cpp` (state count, unfilled count, max distance).
- `node tools/arena_check.js --quick` passes.
- No model has been retrained on this pool yet.
