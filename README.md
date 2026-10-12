# GraphColourRLArena

The Maze Arena, copied out of [4Square](https://github.com/kylemath/4Square). It trains one map-independent graph network on Kempe-swap mazes and scores it on maps it never trained on. The 4Square game is not in this repo.

Open `lab/MazeArena.html` from a local server (a worker and the model files cannot load from a `file://` page):

```
python3 -m http.server 8765
```

Then visit http://127.0.0.1:8765/lab/MazeArena.html.

The page, the worker, and the Node scripts all read `levels.js` (the census of maps and exact distances to a fill). Paths are the same as in 4Square, so nothing in the scripts was rewritten.

| path | role |
|---|---|
| `lab/MazeArena.html` | the arena: panels, controls, leaderboard |
| `lab/arena/engine.js` | puzzle, network, training, testbench |
| `lab/arena/worker.js` | training off the page thread |
| `lab/arena/models/` | committed models and `index.json` |
| `lab/core.js` | the game rules, used only by the check script |
| `levels.js` | maps, levels, and census distances |
| `tools/arena_check.js` | tests (`node tools/arena_check.js`, `--quick` skips the short RL run) |
| `tools/arena_train.js` | headless pretraining and RL |
| `tools/arena_hardgen.js` | hard training pool; writes `lab/arena/cache/` (not committed) |
| `tools/arena_scale.js` | multi-core training on that pool |
| `lab/arena/PRETRAINING.md` | what scaling the training takes |

`tools/arena_hardgen.js` can also read small triangulations from GraphColourDeductiveSolver at `/home/user/GraphColourDeductiveSolver/compute/data/triangulations_n4_11.json`. That file is not part of this copy. The census maps in `levels.js` are.
