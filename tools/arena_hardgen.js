// node tools/arena_hardgen.js [--minutes 12] [--workers 4] [--seed 1] : the Maze Arena's hard data pool.
//
// Writes lab/arena/cache/ (gitignored; regenerate any time) and a small summary lab/arena/models/pool-summary.json:
//   1. every state of every census TRAINING map (the full state space by enumeration, which equals the census);
//   2. single-splice variants of the training maps (an edge flip away from the hole, a vertex inserted into a face
//      without the hole, a degree-3/4 vertex not next to the hole removed and its link retriangulated), kept only if
//      the result is a sphere triangulation with a degree-5 hole whose state space still has states 3+ swaps deep;
//   3. an evolutionary hardness search on worker threads: mutate a parent by 1-3 random splices, enumerate the child's
//      whole state space (every proper 4-colouring up to renaming, swap graph, multi-source BFS from the fills, fills
//      terminal), score hardness = sum over states with d >= 3 of 8^(d-3) / sqrt(unfilled states / 1000) (engine
//      hardness()), keep the hardest distinct maps;
//   4. the small triangulations of GraphColourDeductiveSolver (n <= 11, every degree-5 vertex as the hole), checked
//      for depth (they are too small to be deep, see the summary).
// The census TEST maps (c30 p32r11 p34r28 c60 six trap17 r210) and anything derived from them are never used. 20% of
// the splice/evolved maps are held out (hardpool test set; they share ancestry with training maps, so they measure
// near-distribution generalisation, unlike the census test maps).
//
// Mirror images are not stored: a mirror has the same adjacency and the same labels, only the rim's cyclic order is
// reversed. For the tabula-rasa net (adjacency + raw colours + hole flag) a mirror is literally the same input; for
// the priors GNN (rim-relative encoding) it is a different input, applied as augmentation at training time.
// Rim rotations are no-ops for both encodings (tested), and so is relabelling vertex ids (message passing is
// permutation-equivariant).
//
// Training records (cache/pool-train.bin, cache/pool-holdout.bin): per state: n colour bytes (255 = hole), d, m,
// then m afterstate distances in movesOf order (0 = fill, 255 = no fill reachable). Kept per map: every state with
// d >= 3, up to 1,500 with d = 2 and up to 1,500 with d = 1 (at least 300).
'use strict';
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');
const fs = require('fs'), path = require('path');
const root = path.join(__dirname, '..');
global.window = global;
require(path.join(root, 'levels.js'));
const A = require(path.join(root, 'lab/arena/engine.js'));
const GAME = window.GAME;
const TRAIN_MAPS = ['w9', 'w13', 'w18', 'ico', 'p22', 'r1', 'r178', 'p32r111', 'p31r12'];
const CAP = 120000, NMAX = 42;

function evaluate(spec) {
  const G = A.makeGraph(spec);
  if (A.checkTriangulation(G)) return null;
  const S = A.stateSpace(G, CAP); if (!S) return null;
  const h = A.hardness(S.hist);
  return { hist: S.hist, unreachable: S.unreachable, N: S.N, score: h.score, maxD: h.maxD };
}
function mutate(spec, R, k) {
  const tri = new A.Tri(A.makeGraph(spec)), ops = [];
  for (let i = 0; i < k; i++) { const op = A.spliceRandom(tri, R); if (op) ops.push(op); }
  if (!ops.length) return null;
  const G = tri.toGraph(spec.key);
  if (G.n > NMAX) return null;
  return { spec: A.graphSpec(G), ops };
}

if (!isMainThread) {
  const R = A.rng(workerData.seed);
  parentPort.on('message', job => {
    const out = [];
    for (const j of job.items) {
      const m = mutate(j.spec, R, j.k); if (!m) { out.push(null); continue; }
      const ev = evaluate(m.spec); out.push(ev ? { spec: m.spec, ops: m.ops, parent: j.id, ev } : null);
    }
    parentPort.postMessage(out);
  });
  return;
}

const argv = process.argv.slice(2), flag = (k, d) => { const i = argv.indexOf('--' + k); return i >= 0 ? +argv[i + 1] : d; };
const MINUTES = flag('minutes', 12), NW = flag('workers', 4), SEED = flag('seed', 1);
const CACHE = path.join(root, 'lab/arena/cache');
fs.mkdirSync(CACHE, { recursive: true });
fs.writeFileSync(path.join(CACHE, '.gitignore'), '*\n');
const R = A.rng(SEED * 977 + 1), t0 = Date.now(), log = (...a) => console.log(`[${((Date.now() - t0) / 1000).toFixed(0)}s]`, ...a);
const histAdd = (H, h) => { for (const d in h) H[d] = (H[d] || 0) + h[d]; return H; };

// ---- 0. small triangulations (n <= 11)
let small = { graphs: 0, holes: 0, hist: {}, maxD: 0, note: '' };
try {
  const J = JSON.parse(fs.readFileSync('/home/user/GraphColourDeductiveSolver/compute/data/triangulations_n4_11.json', 'utf8'));
  for (const nk in J.graphs) for (const edges of J.graphs[nk]) {
    const n = +nk; small.graphs++;
    const nb = Array.from({ length: n }, () => []); edges.forEach(([a, b]) => { nb[a].push(b); nb[b].push(a); });
    for (let h = 0; h < n; h++) {
      if (nb[h].length !== 5) continue;
      // the link of h is a 5-cycle; walk it for the rim order
      const L = nb[h], link = [L[0]]; while (link.length < 5) { const last = link[link.length - 1], nx = L.find(x => !link.includes(x) && nb[last].includes(x)); if (nx === undefined) break; link.push(nx); }
      if (link.length < 5) continue;
      const G = A.makeGraph({ n, hole: h, link, edges }), S = A.stateSpace(G, CAP); if (!S) continue;
      small.holes++; histAdd(small.hist, S.hist); small.maxD = Math.max(small.maxD, A.hardness(S.hist).maxD);
    }
  }
  small.note = 'max depth ' + small.maxD + ' over every (triangulation, degree-5 hole) with n <= 11: too shallow to seed hard cases';
} catch (e) { small.note = 'not read: ' + e.message; }
log('small triangulations:', small.graphs, 'graphs,', small.holes, 'degree-5 holes, depth histogram', JSON.stringify(small.hist), '-', small.note);

// ---- 1. census training maps
const maps = [];   // { id, source, parent, split, spec, ev }
for (const key of TRAIN_MAPS) { const G = A.graphFromLevels(GAME, key), ev = evaluate(A.graphSpec(G)); maps.push({ id: key, source: 'census', parent: null, split: 'train', spec: A.graphSpec(G), ev }); }
log('census training maps:', maps.length);

// ---- workers
const workers = Array.from({ length: NW }, (_, i) => new Worker(__filename, { workerData: { seed: SEED * 1000 + i } }));
let evals = 0;
function runBatch(items) {   // spread items over the workers, resolve with results in order
  const per = Math.ceil(items.length / NW);
  return Promise.all(workers.map((w, i) => new Promise(res => {
    const mine = items.slice(i * per, (i + 1) * per); if (!mine.length) return res([]);
    w.once('message', res); w.postMessage({ items: mine });
  }))).then(rs => { const out = [].concat(...rs); evals += out.length; return out; });
}
const sig = m => m.spec.n + ':' + m.spec.edges.length + ':' + JSON.stringify(m.ev.hist) + ':' + m.ev.unreachable;

(async () => {
  // ---- 2. single-splice variants of the hard training maps
  const seen = new Set(maps.map(sig));
  const hardSeeds = maps.filter(m => m.spec.n >= 18);
  const items = []; for (const m of hardSeeds) for (let i = 0; i < 48; i++) items.push({ id: m.id, spec: m.spec, k: 1 });
  const sv = await runBatch(items); let splices = 0;
  sv.forEach((c, i) => { if (!c || c.ev.maxD < 3) return; const m = { id: 'splice-' + items[i].id + '-' + i, source: 'splice', parent: items[i].id, ops: c.ops, spec: c.spec, ev: c.ev };
    const s = sig(m); if (seen.has(s)) return; seen.add(s); maps.push(m); splices++; });
  log('single-splice variants: tried', items.length, 'kept', splices, '(valid, still has d >= 3 states)');

  // ---- 3. evolution
  const pop = maps.filter(m => m.ev.maxD >= 3).slice();
  for (let i = 0; i < 6; i++) { const G = A.generateGraph(22 + R.int(12), R); pop.push({ id: 'rand' + i, source: 'random', spec: A.graphSpec(G), ev: { score: 0, maxD: 0, hist: {} } }); }
  const archive = [];
  let gen = 0;
  const pickParent = () => { let best = null; for (let t = 0; t < 4; t++) { const c = R.pick(pop); if (!best || c.ev.score > best.ev.score) best = c; } return best; };
  while (Date.now() - t0 < MINUTES * 60000) {
    gen++;
    const kids = []; for (let i = 0; i < NW * 6; i++) { const p = pickParent(); kids.push({ id: p.id, spec: p.spec, k: 1 + R.int(3) }); }
    const res = await runBatch(kids);
    res.forEach((c, i) => {
      if (!c || c.ev.maxD < 3) return;
      const m = { id: 'evo-g' + gen + '-' + i, source: 'evolved', parent: kids[i].id, ops: c.ops, spec: c.spec, ev: c.ev }, s = sig(m);
      if (seen.has(s)) return; seen.add(s); archive.push(m); pop.push(m);
    });
    pop.sort((a, b) => b.ev.score - a.ev.score); pop.length = Math.min(pop.length, 64);
    if (gen % 10 === 0) { const b = pop[0]; log(`gen ${gen}: ${evals} maps evaluated, archive ${archive.length}, best score ${b.ev.score} maxD ${b.ev.maxD} n ${b.spec.n} hist ${JSON.stringify(b.ev.hist)}`); }
  }
  workers.forEach(w => w.terminate());
  // keep the hardest distinct evolved maps
  archive.sort((a, b) => b.ev.score - a.ev.score);
  const evolved = archive.slice(0, 160); evolved.forEach(m => maps.push(m));
  log('evolution:', gen, 'generations,', evals, 'maps evaluated in total,', archive.length, 'distinct maps with d >= 3, kept', evolved.length);

  // ---- split and write
  const extra = maps.filter(m => m.source !== 'census'); A.shuffle(extra, R);
  const nHold = Math.round(extra.length * 0.2); extra.forEach((m, i) => { m.split = i < nHold ? 'holdout' : 'train'; });
  const files = { train: [], holdout: [] }, holdoutDist = {};
  const totals = { train: {}, holdout: {} }, kept = { train: {}, holdout: {} };
  let recs = 0;
  for (const m of maps) {
    const G = A.makeGraph(m.spec), S = A.stateSpace(G, CAP), hist = S.hist; histAdd(totals[m.split], hist);
    const deep = [], two = [], one = [];
    for (const i of S.unfilled) { const d = S.dist[i]; if (d >= 3) deep.push(i); else if (d === 2) two.push(i); else if (d === 1) one.push(i); }
    A.shuffle(two, R); A.shuffle(one, R);
    const two2 = two.slice(0, 1500), keep = deep.concat(two2, one.slice(0, Math.max(300, Math.min(1500, two2.length))));
    const bufs = [];
    for (const i of keep) {
      const col = S.cols[i], mv = A.movesOf(G, col), b = Buffer.alloc(G.n + 2 + mv.length);
      for (let u = 0; u < G.n; u++) b[u] = u === G.hole ? 255 : col[u];
      b[G.n] = S.dist[i]; b[G.n + 1] = mv.length;
      mv.forEach((K, k) => { const j = S.idx.get(A.canon(G, A.flip(col, K))), d = S.won[j] ? 0 : S.dist[j]; b[G.n + 2 + k] = d < 0 ? 255 : d; });
      bufs.push(b); kept[m.split][S.dist[i]] = (kept[m.split][S.dist[i]] || 0) + 1; recs++;
    }
    m.records = keep.length; m.bytes = bufs.reduce((s, b) => s + b.length, 0); m._buf = Buffer.concat(bufs);
    if (m.split === 'holdout') { const D = {}; for (const i of S.unfilled) if (S.dist[i] > 0) D[S.keys[i]] = S.dist[i]; holdoutDist[m.id] = D; }
  }
  for (const split of ['train', 'holdout']) {
    let off = 0; const parts = [];
    for (const m of maps.filter(x => x.split === split)) { m.offset = off; off += m.bytes; parts.push(m._buf); }
    fs.writeFileSync(path.join(CACHE, 'pool-' + split + '.bin'), Buffer.concat(parts));
  }
  maps.forEach(m => { delete m._buf; });
  fs.writeFileSync(path.join(CACHE, 'pool.json'), JSON.stringify({ format: 'mazearena-pool-1', date: new Date().toISOString(), maps }));
  fs.writeFileSync(path.join(CACHE, 'holdout-dist.json'), JSON.stringify(holdoutDist));
  // round 1 reference: the census of the 9 training maps (round 1 sampled these, 75% of them d = 1)
  const r1 = {}; for (const key of TRAIN_MAPS) histAdd(r1, maps.find(m => m.id === key).ev.hist);
  const count = s => maps.filter(m => m.split === s).length;
  const summary = { format: 'mazearena-pool-summary-1', date: new Date().toISOString(), minutes: MINUTES, workers: NW, mapsEvaluated: evals + maps.length,
    round1TrainingCensus: r1, round2: { trainMaps: count('train'), holdoutMaps: count('holdout'), bySource: maps.reduce((o, m) => { o[m.source + ':' + m.split] = (o[m.source + ':' + m.split] || 0) + 1; return o; }, {}),
      allStates: totals, trainingRecords: kept, records: recs }, smallTriangulations: small,
    hardestMaps: maps.slice().sort((a, b) => b.ev.score - a.ev.score).slice(0, 12).map(m => ({ id: m.id, source: m.source, split: m.split, n: m.spec.n, hist: m.ev.hist, unreachable: m.ev.unreachable })) };
  fs.mkdirSync(path.join(root, 'lab/arena/models'), { recursive: true });
  fs.writeFileSync(path.join(root, 'lab/arena/models/pool-summary.json'), JSON.stringify(summary, null, 1) + '\n');
  log('pool written:', count('train'), 'train maps,', count('holdout'), 'holdout maps,', recs, 'records');
  log('depth histogram, all states: train', JSON.stringify(totals.train), 'holdout', JSON.stringify(totals.holdout));
  log('round 1 training census:', JSON.stringify(r1));
})();
