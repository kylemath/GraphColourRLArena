# Pretraining a Kempe-swap foundation model: what runs here, and what scaling would take

The Maze Arena (`lab/MazeArena.html`, engine `lab/arena/engine.js`) pretrains its graph network on random maps in
plain JavaScript: in the page's Web Worker, or headless with `node tools/arena_train.js --pretrain-minutes N`. That is
real pretraining, but at laptop scale: about 10^4 maps and 2 x 10^5 labelled samples in two minutes on one CPU core.
"Billions of graphs" needs a tensor library and accelerators. This note records what was measured, what the data and
objectives are, and what a PyTorch/JAX port would need. No Python script ships with it: torch is not installed in
this environment, and nothing was run that could not be checked here.

## Measured here (Node 22, one core of a 4-core container, `node tools/arena_check.js --speed-only`)

| what | rate |
|---|---|
| random triangulation + blocked 4-colouring, n = 14 / 18 / 24 / 34 | ~6,400 / 2,800 / 7,500 / 4,700 per s |
| the same plus exact distances (BFS over the colouring's whole Kempe class, cap 3,000 states), n = 14 / 18 | ~6,300 / 2,700 labelled maps per s |
| exact labels at n = 20 / 22 / 24 / 26 (mean class size 47 / 70 / 108 / 173 states) | ~2,000 / 1,300 / 740 / 460 maps per s |
| GNN forward + backward, H = 8 / 16 / 32 (2.2k / 7.8k / 28k params), n = 34 | ~4,600 / 1,800 / 450 samples per s |
| pretraining end to end, H = 16, all four objectives | ~1,950 samples per s (226k samples, 9.4k maps in 116 s) |
| RL with afterstate scoring, H = 16, n = 34 | ~380 env steps per s acting only, ~250 with learning |

(The n = 18 data rate is lower than n = 24 because the n = 24 row has no BFS labels.) Distances on random maps are
almost all 1 or 2: over 10^5 labelled states at n = 14-26, about 73% are at distance 1, 27% at distance 2 and fewer
than 1 in 10^4 at distance 3. The level census looks the same (deep states are rare by nature), so a pretraining set
that teaches deep planning has to search for deep states on purpose (see the curriculum below).

So one CPU core makes labelled data at ~10^3-10^4 maps per second. A billion maps is ~1-3 x 10^5 core-seconds, about a
day on a 64-core machine, and embarrassingly parallel. Training is the bottleneck: at ~2 x 10^3 samples per second
per core in JS, 10^9 samples would take about 6 core-days *per pass* with H = 16, and far longer at useful widths.
A batched GPU implementation should get to 10^5-10^6 samples per second for graphs this small.

## Data generator (as implemented in `engine.js`)

1. `generateGraph(n)`: start from the icosahedron (12 vertices, all of degree 5), insert vertices into random faces
   until n vertices, then make about 6n random edge flips (degrees kept >= 3 and <= 11, no multi-edges). Faces are a
   rotation system, so the result is a sphere triangulation by construction. `checkTriangulation` checks it: E = 3n - 6,
   every face a triangle, 2n - 4 faces. Insertion alone would give stacked triangulations, which have a unique
   4-colouring, so the flips matter.
2. Hole: a random degree-5 vertex. The link (rim) is its cyclic neighbour order, reversed half the time and started at
   a random rim region, so the net sees both orientations.
3. `randomBlocked`: put the rim pattern D s1 D s2 s3 at a random rotation with random colour names (rejected if a
   chord makes it improper), then complete the colouring by randomized most-constrained-first backtracking.
4. Labels: `exactDistances` runs a BFS over the colouring's Kempe class (non-web swaps, fills are leaves), then a
   multi-source BFS back from the fills, for maps up to n = 26 (cap 3,000 states). It matches the census on every
   level map tested (`tools/arena_check.js`, section 2). Bigger maps get only the self-supervised objectives.

## Architecture and objectives (as implemented)

A message-passing GNN over the map's adjacency (width H, K rounds of h' = relu(Ws h + Wn mean(neighbours) + b))
with a readout of [mean pool, max pool, the 5 rim embeddings in relative rim order, the hole, 3 globals], an MLP of
width 2H, and one small head per objective:

| objective | head | label | loss |
|---|---|---|---|
| (a) distance to a fill | scalar on the MLP | exact BFS distance | Huber |
| (b) good swap | per candidate chain: MLP on [mean of the chain's embeddings, MLP output, chain features] | the swap lowers the exact distance by one | BCE |
| (c) masked colour | per region on h_K | relative colour class of 15% hidden regions | cross-entropy |
| (d) chain structure | per region on h_K | size and rim reach of the region's chain for each partner colour, with every chain input hidden | MSE |

The encoding is map-independent: colours relative to the rim pattern (D -> 0, then the singles in rim order from the
reference D), rim positions relative to the same reference, so renaming colours or rotating the link list leaves the
input unchanged (tested). After pretraining the value head starts as minus the distance head and RL fine-tunes it.

## What pretraining did here (single seeds; differences of about 1 point are noise)

| model | params | pretrain | RL | held-out agree2 | agree3 | success |
|---|---|---|---|---|---|---|
| nano-rl | 679 | none | 60 s | 99.0% | 89.8% | 98.9% |
| mid-zero-shot | 7,759 | 116 s, 8.8k random maps | none | 93.3% | 59.2% | 97.8% |
| mid-rl | 7,759 | none | 180 s | 99.2% | 91.8% | 97.8% |
| mid-pretrained | 7,759 | 116 s, 9.4k maps | 120 s | 98.8% | 87.8% | 98.9% |
| colours-only-rl | 30,383 | none | 240 s | 91.7% | 77.5% | 94.5% |
| colours-only-pretrained | 30,383 | 106 s, 2.6k maps | 180 s | 92.5% | 81.6% | 96.7% |
| baselines: random / hand rule / two-ply | 0 | | | 63.2 / 68.5 / 96.4% | 70.6 / 76.3 / 74.7% | 56.7 / 95.4 / 100% |

- With no level map ever seen, pretraining on random maps transfers: 93% held-out agree2 zero-shot (a second seed gave
  98% on the quick bench), well above random (63%) and the hand rule (69%), a little under the two-ply search (96%).
- With the hand-built chain features, RL alone is already at 99% after a few minutes. Even the 679-parameter H = 4,
  K = 1 net `nano-rl` (1 minute of RL) reaches 99.0% held-out agree2 and 89.8% agree3. Those features (each region's chain
  sizes and which rim positions each chain reaches) make "is there a one-swap fill from here?" nearly linear, so at
  this testbench pretraining has nothing left to add.
- Without chain features (colours only, the net must find chains by message passing) the task is genuinely harder
  (about 92%), and there pretraining gave a small gain for less RL (+0.8 agree2, +4 agree3, +2 success).
- The policy head trained by objective (b) on random maps is weak on the level maps (62-78% agree2). The afterstate
  value head is the one that transfers.
- Still unsolved by every model: the deepest held-out starts (six-ring-trap, par 4; second-pit, par 5) are not filled
  within 3x par, and agree3 (states 3+ swaps out) stays under 92%.

## A PyTorch / JAX port

1. **Exchange format.** Keep the model JSON (`mazearena-model-1`): `arch` (H, K, H1, Hp, F = 42, GD = 3, CF = 12,
   chainFeats), a `shapes` table and base64 float32 weights in that order (Win, bin, Ws1, Wn1, b1, ..., W1, b1, wv,
   bv, wd, bd, Wp1, bp1, wp2, bp2, Wc, bc, Wch, bch; row-major, y = W x). A model trained in Python loads in the page
   unchanged, and the page's testbench and leaderboard rank it next to the JS models.
2. **Features.** Port `encode()` exactly (or call the JS through a small Node worker pool that writes feature shards).
   The tests in `tools/arena_check.js` (invariance, census agreement) are the acceptance check.
3. **Batching.** Pack a batch of maps as one disjoint graph (`torch_scatter` / `jax.ops.segment_sum` for the
   neighbour mean and the pools; the rim and hole embeddings are gathers by index). Chains for the policy head are a
   second segment index. No padding needed.
4. **Data pipeline.** Generator processes (Node or a C/Rust port of the four functions above) write sharded records:
   adjacency, link, colouring, exact distance, good-swap mask. Labelled maps up to n = 26 are cheap; for larger maps
   use the self-supervised objectives (c) and (d), plus labels from a bounded BFS that searches for deep states.
5. **Curriculum.** Grow n from 12-18 to 30-60. Oversample states with d >= 2 (and search for d >= 3 states: walk away
   from fills with a BFS that keeps the deepest), since uniform sampling is 73% d = 1. Keep (c) and (d) on throughout,
   and train a colours-only variant so the network has to learn chain structure itself.
6. **Width and scale.** At these graph sizes the measured gains stop at small widths when chain features are given;
   the colours-only nets are where width (H = 64-256, K = 6-10, residual connections and layer norm) should be tried.
7. **Evaluation.** Score every checkpoint with the same held-out testbench (`ARENA.Testbench`, all 16 four-colour level
   maps, census labels) and report agree2, agree3 and success against the random, hand-rule and two-ply rows in
   `models/index.json`.

## Round 2: a hard data pool, a tabula-rasa track, one hour on 4 cores

**Tracks.** The round-1 nets are now the *priors* track: they are given rim-relative colours, degree, rim position
and per-chain features (which is most of the Kempe-chain reasoning, built in). The main track of round 2 is *tabula
rasa*: inputs are the adjacency, RAW colour one-hots and the hole flag, nothing else. Colour symmetry is not built
in: training applies a random colour permutation to every sample. The rules (which swaps are legal, when the hole
fills) stay with the environment, as chess rules do in AlphaZero. The net (`arch.kind = 'rec'` in `engine.js`) is
one weight-tied message-passing layer iterated T times with the input injected every iteration ("deep thinking"
style): h_t = tanh(Ws h + Wn Σ neighbours / 6 + Wx x + b). Chains can be 30 regions long, so depth comes from T
rather than from features. Two agents share it: the afterstate value (uses the simulator to look one swap ahead),
and a "click a region, pick a colour" policy head that needs no simulator (the env turns the click into the swap).

**Data (`node tools/arena_hardgen.js`, 4 worker threads, 12 min + 6 min writing).** Every state of every census map
(whole state spaces by enumeration; they equal the census exactly), 106 single-splice variants of the training maps,
and an evolutionary hardness search: 3,371 maps evaluated in total; parents mutated by 1-3 splices; score = Σ over
states with d ≥ 3 of 8^(d-3), divided by √(unfilled states / 1000); 1,615 distinct maps with deep states, the
hardest 160 kept. 20% of the splice/evolved maps are held out as the *hard pool* test set. These maps descend from
training maps, so the hard pool tests near-distribution generalisation; the census test maps are the
out-of-distribution test. The census test maps and anything derived from them are never used for training. The small
triangulations of GraphColourDeductiveSolver (n ≤ 11, 1,587 graph/hole pairs) were checked and are at most 2 swaps
deep, so they were not used. Mirrors are applied at sampling time and not stored. For the tabula-rasa net a mirror
is the same input (no orientation anywhere); rim rotations and vertex relabelling are no-ops for both tracks.

| unfilled states by depth d | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| round 1 (census of the 9 training maps) | 11,468 | 3,028 | 49 | 5 | 3 | 1 |
| round 2 pool, all states (222 training maps) | 3,178,156 | 1,390,696 | 130,336 | 502 | 21 | 7 |
| round 2 training records (d = 1 and 2 capped per map) | 262,028 | 261,562 | 130,336 | 502 | 21 | 7 |
| round 2 hard pool (53 held-out maps), all states | 784,776 | 346,203 | 30,941 | 63 | 6 | 2 |

The search found many states 3 swaps deep (2,660× round 1) but few deeper ones (d ≥ 4: 530 vs 9). Its best
depth-4-rich map has 3,563 states at d = 3 and 85 at d = 4 (n = 42). Depth 5-6 states still come only from r178 and its
close variants: deep traps are rare even among hard maps.

**Training (`node tools/arena_scale.js`).** The labels are exact, so the main lever is supervised distillation rather
than RL. Per sample (deep states oversampled 15/35/30/20% for d = 1/2/3/4+) the losses are:
- value regression on −d(s);
- for one good and one bad swap, regression of both afterstates plus a logistic ranking loss (good above bad);
- (tabula rasa) −log P(good click) over all legal (region, colour) clicks.

RL fine-tuning was not run. With exact labels for every state, TD targets are noisier versions of the same labels,
and at about 80 samples/s for the recurrent net an RL phase would have cost more than it could add. The training is
synchronous data parallelism with `worker_threads`: weights in one SharedArrayBuffer read directly by the workers,
one gradient slot per worker in another, and the main thread sums the slots and applies AdamW. Recurrent nets train
with T drawn from [T/2, T] per sample.

| throughput (samples/s; one sample = 1-3 forward+backward passes) | 1 worker | 2 workers | 4 workers |
|---|---|---|---|
| tabula rasa rec H32 T16 | 48 | 92 | 156 (3.2×) |
| priors GNN H16 K3 | 701 | – | 2,080 (3.0×) |

Encoding is not the bottleneck. Building the raw input plus the move list (chain extraction) takes about 11 µs, against
about 8 ms for a forward+backward pass at H32 T16, so caching feature tensors would save under 1%. Unrolling the
recurrent loops gave no measurable gain.

**The one-hour run.** Three models ran concurrently on the 4 cores:
- tabula rasa rec H32 T16, 2 workers: 306k samples;
- tabula rasa rec H16 T24, 1 worker: 371k samples;
- priors GNN H16 K3, 1 worker: 2.49M samples.

Learning curves:
- The tabula-rasa value loss fell from 1.10 to 0.42 in the first 5 minutes and then only to 0.31 by minute 60.
- The tabula-rasa ranking loss stayed near 1.8 (2.13 = no discrimination); the policy loss went from 0.92 to 0.80.
- The priors GNN reached value loss 0.036 and ranking loss 0.66.
- A sanity check confirms the tabula-rasa pipeline can learn: the same net reaches 100% on tidy-trap (48 states)
  after about 50k samples. On 222 maps it needs far more data and compute than an hour of CPU gives.

| model (track) | params | held-out agree2 / agree3 / agree4+ (49 and 9 states) | hard pool agree2 / agree3 | deep starts within par | success |
|---|---|---|---|---|---|
| priors-gnn-h16-k3-pool (priors, hard-pool distillation) | 7,759 | 99.0 / 89.8 / 66.7 | **98.7 / 97.4** | 3/4 | 98.9 |
| mid-rl (priors, round 1 RL) | 7,759 | 99.2 / 91.8 / 66.7 | 86.3 / 73.2 | 2/4 | 97.8 |
| nano-rl (priors, round 1 RL) | 679 | 99.0 / 89.8 / 66.7 | 84.9 / 70.6 | 3/4 | 98.9 |
| colours-only-pretrained (priors without chain features) | 30,383 | 92.5 / 81.6 / 55.6 | 80.7 / 74.8 | 3/4 | 96.7 |
| tr-rec-h32-t16 click policy (tabula rasa, no simulator) | 11,781 | 77.8 / **85.7** / 66.7 | 64.9 / 68.3 | 0/4 | 36.3 |
| tr-rec-h32-t16 afterstate value (tabula rasa) | 11,781 | 67.6 / 67.3 / 55.6 | 57.2 / 62.4 | 1/4 | 92.3 |
| tr-rec-h16-t24 click policy (tabula rasa) | 3,077 | 72.3 / 77.5 / 66.7 | 67.0 / 72.9 | 0/4 | 28.6 |
| two-ply count (search baseline) | 0 | 97.5 / 81.6 / 66.7 | 76.0 / 53.9 | 0/4 | 100 |
| hand rule | 0 | 68.3 / 71.4 / 55.6 | 65.3 / 67.0 | 1/4 | 96.7 |
| random | 0 | 62.5 / 63.3 / 66.7 | 52.8 / 57.2 | 0/4 | 68.1 |

Deep starts are six-ring-trap (par 4), second-pit (par 5), deepest-pit (par 6) and depth-5 (par 5). The last two are
on r178, a training map, so they are not held out. Within 3× par is too loose to separate anything (random gets 3/4),
so the table shows within par. agree4+ rests on 9 states, so differences there are noise.

**Iterations at test time** (train T, test 2T):

| model | T | value agree3 test / hard | policy agree3 test / hard |
|---|---|---|---|
| tr-rec-h32-t16 | 16 | 67.3 / 62.4 | 85.7 / 68.3 |
| tr-rec-h32-t16 | 32 | 75.5 / 62.7 | 85.7 / 68.3 |
| tr-rec-h16-t24 | 24 | 51.0 / 66.3 | 77.5 / 72.9 |
| tr-rec-h16-t24 | 48 | 51.0 / 66.3 | 77.5 / 73.2 |

The tanh recurrence is stable at 2T: nothing degrades and the value agent gains a little, but extra iterations do not
create skill the net has not learned.

**What this says.**
- **Hard-pool distillation is the round's clear win for the priors track.** Held-out hard-pool agree3 rose from
  about 73% for every round-1 model to 97.4%. Two-ply search scores 53.9% there, below random: the evolved maps are
  traps for greedy lookahead. Its census test scores did not move (agree3 89.8%).
- **Tabula rasa learns something real but far less in an hour.** The click policy picks a distance-lowering swap on
  85.7% of the held-out census d ≥ 3 states, above the two-ply search (81.6%) and random (63.3%). But its agree2
  (77.8%) is far below two-ply (97.5%), and it is weaker on the hard pool (68%, roughly the hand rule).
- **The tabula-rasa value head barely beats random** (ranking loss never moved); the policy head is the one that learned.
- **The click policy fails on rollouts** because it does not always take an available fill (36% success), while the
  value agent fills whenever it can (92%).
- **Honest reading.** Without the built-in chain features, an hour of 4-core CPU (300-370k samples) is far from
  enough. The trap17 sanity run suggests the architecture can learn this, but sample needs scale with the number of
  distinct maps. The bitter-lesson path is the PyTorch/JAX port above, with this pool generator as the data source and
  10^8-10^9 samples.
