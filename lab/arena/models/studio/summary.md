# Tabula-rasa track at scale (2026-10-12/13)

Three recurrent nets (raw colours + hole flag, no Kempe-chain features) trained for 3 hours each on a 2x hard pool,
then evaluated with `node tools/arena_scale.js --eval <name> --save`.

- Machine: MacBook Pro, Apple M4 Pro (10 performance + 4 efficiency cores, 48 GB). Not a Mac Studio.
- Corpus: `levels.js` at commit `b706b3b`. New levels added since then are not in the census or the pool.
- PyTorch is not installed here, so the optional MPS port (step 5) was skipped.

## Hard pool

`node tools/arena_hardgen.js --workers 10 --minutes 30 --keep 480 --seed 2` (`--keep` is new; it was a fixed 160).
36,163 maps evaluated, 12,515 distinct maps with d >= 3; 478 train maps (old pool 222) and 117 held out.

| all train states | d=1 | d=2 | d=3 | d=4 | d=5 | d=6 |
|---|---|---|---|---|---|---|
| old pool (4 cores, 12 min) | 3,178,156 | 1,390,696 | 130,336 | 502 | 21 | 7 |
| new pool (10 cores, 30 min) | 11,985,765 | 4,809,986 | 1,013,176 | 284,630 | 676 | 1 |

Training records: 3,203,882 (old 811,592). d=4 is 567x and d=5 32x, but d=6 fell from 7 to 1: the search score
rewards d=4 so heavily that the population converged on d=4-rich maps near n=39-42 and lost the rare d=6 maps.
Full numbers in `pool-summary.json`.

## Throughput (`--bench`, rec H=32 T=16)

| workers | 4 | 8 | 10 |
|---|---|---|---|
| samples/s | 541 | 931 | 1,064 |

The reference run (4 cores) managed about 83 samples/s.

## Runs

All with `--mix 0.15,0.35,0.3,0.2 --wval 1 --wrank 0.5 --wpol 1 --minutes 180 --ckpt-min 15`, run in parallel.

| run | arch | workers | seed | params | samples seen | wall |
|---|---|---|---|---|---|---|
| a `studio-rec-h32-t16` | H32 T16 tmin 8 | 4 | 11 | 11,781 | 5,070,048 | 3 h |
| b `studio-rec-h64-t24` | H64 T24 tmin 12 | 3 | 12 | 46,085 | 780,978 | 3 h |
| c `studio-rec-h32-t32` | H32 T32 tmin 16 | 3 | 13 | 11,781 | 2,016,729 | 3 h |
| reference `tr-rec-h32-t16` | H32 T16 | 2 | | 11,781 | 306,336 | 1 h |

Loss (value / rank / policy), from the training curve:

| run | start | 1 h | end |
|---|---|---|---|
| a | 0.564 / 1.848 / 1.224 | 0.108 / 1.098 / 0.447 | 0.083 / 0.992 / 0.350 |
| b | 1.022 / 1.916 / 1.249 | 0.151 / 1.358 / 0.756 | 0.102 / 1.116 / 0.439 |
| c | 0.735 / 1.872 / 1.242 | 0.129 / 1.222 / 0.556 | 0.101 / 1.088 / 0.459 |
| reference | 1.097 / 1.928 / 0.915 | 0.314 / 1.800 / 0.805 (end) | |

The value head now learns: value loss 0.08-0.10 against 0.31 for the reference.

## Results

Census test maps (7 maps, n3 = 49, n4 = 9). Hard pool = 30 held-out pool maps. **The hard pool changed with the new
pool**, so the reference models were re-evaluated on the new held-out maps (second number) next to their committed
numbers on the old pool (first number).

### Value agent (afterstate scoring)

| model | agree2 | agree3 | agree4 | rollout success | hard-pool agree3 | deep starts within 1x / 3x par |
|---|---|---|---|---|---|---|
| a H32 T16 | 0.840 | 0.755 | 0.889 | 0.934 | 0.907 | 0 / 0 |
| b H64 T24 | 0.754 | 0.816 | 0.778 | 0.945 | 0.936 | 0.25 / 0.25 |
| c H32 T32 | 0.727 | 0.816 | 0.667 | 0.890 | 0.919 | 0.25 / 0.50 |
| reference tr-rec-h32-t16 | 0.676 | 0.674 | 0.556 | 0.923 | 0.624 / **0.320** | 0.25 / 1.00 |
| priors gnn-h16-k3-pool | 0.990 | 0.898 | 0.667 | 0.989 | 0.974 / **0.555** | 0.75 / 0.75 |

### Policy agent (click region, pick colour)

| model | agree2 | agree3 | agree4 | rollout success | hard-pool agree3 |
|---|---|---|---|---|---|
| a H32 T16 | 0.759 | 0.837 | 0.444 | 0.769 | 0.922 |
| b H64 T24 | 0.773 | 0.837 | 0.667 | 0.692 | 0.913 |
| c H32 T32 | 0.709 | 0.816 | 0.667 | 0.648 | 0.916 |
| reference tr-rec-h32-t16 | 0.778 | 0.857 | 0.667 | 0.363 | 0.683 / **0.369** |

Recurrence at 2T (value agent, test agree3 / hard agree3): a 0.816 / 0.884, b 0.816 / 0.933, c 0.796 / 0.919.

## Reading

- Against the brief's reference (policy agree2 77.8 / agree3 85.7): the policy head is unchanged on the census test
  maps (75.9-77.3 / 81.6-83.7). The gain is in the value head (census agree2 0.68 -> 0.84 for run a) and on the hard
  pool, where every new model scores 0.91-0.94 agree3 against 0.32-0.37 for the reference rec net and 0.56 for the
  priors GNN on the same new held-out maps. The priors GNN is still far ahead on census agree2 (0.99).
- More samples (a) beat more width (b) and more recurrence depth (c) on census agree2; b and c are slightly better at
  agree3 and on the hard pool. Doubling T at test time does not help. With n <= 42, T=16 may already reach every
  region; recurrence depth should be retested on bigger (padded) maps.
- **Deep starts regressed, and it is the rollout, not the net.** The testbench's greedy rollout has no memory, and a
  Kempe swap undoes itself. On the deep level starts the new nets play 1-3 correct swaps and then enter a 2-cycle,
  e.g. run a on `deepest-pit`: d = 6, 5, then back to 6 and around again. Run c solves `deepest-pit` in par (6 steps)
  and run b solves `second-pit` in par. A no-immediate-undo rule in `netAgent` would probably recover most of these.
  The benchmark was left unchanged so the numbers stay comparable.

## Files

- `studio-rec-*.json`: the three final models (also in `lab/arena/models/` and listed in `index.json`).
- `pool-summary.json`: the new pool's summary.
- Checkpoints and the pool itself are in `lab/arena/cache/` (not committed).
