# GraphColourRLArena

🚀 **[Live Demo](https://kylemath.github.io/GraphColourRLArena)** 🚀

The Maze Arena, copied out of [4Square](https://github.com/kylemath/4Square). It trains one map-independent graph network on Kempe-swap mazes and scores it on maps it never trained on. The 4Square game is not in this repo.

Open `docs/index.html` from a local server (a worker and the model files cannot load from a `file://` page):

```
python3 -m http.server 8765
```

Then visit http://127.0.0.1:8765/docs/.

The page, the worker, and the Node scripts all read `docs/levels.js` (the census of maps and exact distances to a fill), a copy of 4Square's `levels.js`. Everything the page needs lives under `docs/`, which GitHub Pages serves. To pick up new 4Square levels, copy its rebuilt `levels.js` over `docs/levels.js`.

| path | role |
|---|---|
| `docs/index.html` | the arena: panels, controls, leaderboard |
| `docs/arena/engine.js` | puzzle, network, training, testbench |
| `docs/arena/worker.js` | training off the page thread |
| `docs/arena/models/` | committed models and `index.json` |
| `docs/core.js` | the game rules, used only by the check script |
| `docs/levels.js` | maps, levels, and census distances |
| `tools/arena_check.js` | tests (`node tools/arena_check.js`, `--quick` skips the short RL run) |
| `tools/arena_train.js` | headless pretraining and RL |
| `tools/arena_hardgen.js` | hard training pool; writes `docs/arena/cache/` (not committed) |
| `docs/arena/hardset/` | proof-hard map pool v1 (floor-tight, R0 maps, non-D-reducible 2-balls); see its README |
| `tools/arena_scale.js` | multi-core training on that pool |
| `docs/arena/PRETRAINING.md` | what scaling the training takes |

`tools/arena_hardgen.js` can also read small triangulations from GraphColourDeductiveSolver at `/home/user/GraphColourDeductiveSolver/compute/data/triangulations_n4_11.json`. That file is not part of this copy. The census maps in `levels.js` are.
