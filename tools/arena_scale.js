// node tools/arena_scale.js : Maze Arena training at scale on the hard data pool (tools/arena_hardgen.js), data
// parallel over worker threads, plus the round-2 evaluation.
//
// Training (supervised distillation of the exact distances, then optional RL fine-tuning elsewhere):
//   --name tr-h32t16 --arch rec --H 32 --T 16 [--tmin 8]     tabula-rasa recurrent net (raw colours + hole flag only)
//   --arch gnn --H 16 --K 3                                   the priors GNN (rim-relative colours + chain features)
//   --workers 2 --minutes 60 --batch 32 --lr 1e-3 --wd 1e-5 --seed 1
//   --mix 0.15,0.35,0.3,0.2   sampling weights for states at d = 1, 2, 3, 4+ (deep states oversampled)
//   --wval 1 --wrank 0.5 --wpol 1    loss weights: value regression, afterstate ranking, (region, colour) policy
//   --ckpt-min 5              checkpoint + quick eval every N minutes (docs/arena/cache/ckpt-<name>.json)
// Each worker: sample a state from the pool (deep states oversampled), apply a random colour permutation (rec) or
// a random mirror (gnn), and compute
//   value loss  Huber(V(s) + d(s))
//   ranking     for one good swap g (lowers the exact distance) and one bad swap b of s: Huber(V(g) + d(g)),
//               Huber(V(b) + d(b)) and softplus(2 (V(b) - V(g)) + 2), so the afterstate the agent should pick scores higher
//   policy      (rec) -log P(a good click) over every legal (region, colour) click
// into its own gradient slot of a SharedArrayBuffer; the main thread sums the slots and applies AdamW to the weights,
// which live in another SharedArrayBuffer the workers read directly (synchronous data parallelism).
//
// Evaluation:  --eval <model.json or ckpt name> [--save]   census testbench (agree2/3/4+), hard held-out pool, deep
//              level starts within 1x and 3x par; value agent and (rec) policy agent; rec nets at T and 2T
//              --baselines   the same metrics for random / hand rule / two-ply, written into models/index.json
//              --bench       samples/s with 1, 2 and 4 workers (and a single-thread round-1 style loop)
'use strict';
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');
const fs = require('fs'), path = require('path');
const root = path.join(__dirname, '..');
global.window = global;
require(path.join(root, 'docs/levels.js'));
const A = require(path.join(root, 'docs/arena/engine.js'));
const GAME = window.GAME;
const CACHE = path.join(root, 'docs/arena/cache'), MODELS = path.join(root, 'docs/arena/models');
const ALL = A.levelMaps(GAME).map(m => m.key);
const TRAIN_MAPS = ['w9', 'w13', 'w18', 'ico', 'p22', 'r1', 'r178', 'p32r111', 'p31r12',
  // Census33/34 batch (4square commit 207ad1b): training only, so the held-out test maps stay the same as before
  'p33r240', 'p34r51', 'p34r189', 'p34r10', 'p34r84', 'p33r115', 'p34r2_856229', 'p33r57', 'p33r37', 'p33r116', 'p33r219'];
const TEST_MAPS = ALL.filter(k => TRAIN_MAPS.indexOf(k) < 0);

// ---------------------------------------------------------------- the pool
function loadPool(split) {
  const P = JSON.parse(fs.readFileSync(path.join(CACHE, 'pool.json'), 'utf8')), buf = fs.readFileSync(path.join(CACHE, 'pool-' + split + '.bin'));
  const maps = P.maps.filter(m => m.split === split).map(m => {
    const G = A.makeGraph(m.spec), M = { id: m.id, source: m.source, G, mirror: A.mirrorGraph(G), base: m.offset, bytes: m.bytes };
    return M;
  });
  const cls = [[], [], [], []];   // (map index, byte offset) per depth class d = 1, 2, 3, 4+
  maps.forEach((M, mi) => {
    let o = M.base; const end = M.base + M.bytes, n = M.G.n;
    while (o < end) { const d = buf[o + n], m = buf[o + n + 1]; cls[Math.min(3, d - 1)].push(mi, o); o += n + 2 + m; }
  });
  return { maps, buf, cls: cls.map(a => Int32Array.from(a)) };
}
function readRecord(pool, mi, o) {
  const M = pool.maps[mi], n = M.G.n, b = pool.buf, col = new Int8Array(n);
  for (let u = 0; u < n; u++) col[u] = b[o + u] === 255 ? -1 : b[o + u];
  const d = b[o + n], m = b[o + n + 1], after = b.subarray(o + n + 2, o + n + 2 + m);
  return { M, col, d, after };
}

// ---------------------------------------------------------------- worker: gradients on a minibatch
if (!isMainThread) {
  const { arch, seed, paramsSAB, gradSAB, slot, size, cfg } = workerData;
  const pool = loadPool('train'), R = A.rng(seed);
  const net = A.netCreate(arch, 1); net.params = new Float32Array(paramsSAB);
  const grad = new Float32Array(gradSAB, slot * size * 4, size);
  const netG = A.netShare(net), netB = A.netShare(net);
  const mix = cfg.mix, cum = []; mix.reduce((s, x, i) => (cum[i] = s + x), 0);
  const isRec = net.arch.kind === 'rec';
  const hub = e => Math.abs(e) <= 1 ? [0.5 * e * e, e] : [Math.abs(e) - 0.5, Math.sign(e)];
  function sample() {
    let c = 3, x = R() * cum[3]; for (let i = 0; i < 4; i++) if (x < cum[i]) { c = i; break; }
    while (!pool.cls[c].length) c = (c + 3) % 4;
    const k = R.int(pool.cls[c].length / 2); return readRecord(pool, pool.cls[c][2 * k], pool.cls[c][2 * k + 1]);
  }
  function one(stats) {
    const r = sample(), T = isRec ? cfg.tmin + R.int(net.arch.T - cfg.tmin + 1) : 0;
    let G = r.M.G, opt;
    if (isRec) { const perm = A.shuffle([0, 1, 2, 3], R); opt = { kind: 'rec', perm }; }
    else { if (R() < 0.5) G = r.M.mirror; opt = net.arch; }
    const enc = A.encode(G, r.col, opt), m = enc.moves.length;
    // the state itself: value (+ policy)
    const out = A.netForward(net, enc, { policy: isRec && cfg.wpol > 0, T });
    const [lv, gv] = hub(out.v + r.d); stats.v += lv;
    const g = { v: cfg.wval * gv };
    const good = new Uint8Array(m); const gi = [], bi = [];
    for (let k = 0; k < m; k++) { if (r.after[k] === r.d - 1) { good[k] = 1; gi.push(k); } else bi.push(k); }
    if (isRec && cfg.wpol > 0) {
      const pl = A.policyLoss(out.logits, A.actionTable(enc), good, enc.n); stats.p += pl.loss;
      for (let k = 0; k < pl.grad.length; k++) pl.grad[k] *= cfg.wpol; g.logits = pl.grad;
    }
    A.netBackward(net, out, g, grad);
    // ranking between one good and one bad afterstate
    if (cfg.wrank > 0 && gi.length && bi.length) {
      const kg = R.pick(gi), kb = R.pick(bi), cg = A.flip(r.col, enc.moves[kg].K), cb = A.flip(r.col, enc.moves[kb].K);
      const after = isRec ? Object.assign({}, opt, { noMoves: true }) : opt;
      const db = r.after[kb] === 255 ? r.d + 2 : r.after[kb];
      const ob = A.netForward(netB, A.encode(G, cb, after), { T });
      let og = null, vg = 0;
      if (r.after[kg] > 0) { og = A.netForward(netG, A.encode(G, cg, after), { T }); vg = og.v; }
      const z = 2 * (ob.v - vg) + 2, sp = z > 30 ? z : Math.log1p(Math.exp(z)), sg = 1 / (1 + Math.exp(-z)); stats.r += sp;
      const [lb, gb] = hub(ob.v + db);
      A.netBackward(netB, ob, { v: cfg.wval * gb + cfg.wrank * 2 * sg }, grad);
      if (og) { const [lg, gg] = hub(og.v + r.after[kg]); stats.v2 += lg + lb; A.netBackward(netG, og, { v: cfg.wval * gg - cfg.wrank * 2 * sg }, grad); }
      else stats.v2 += lb;
      stats.pairs++;
    }
    stats.n++;
  }
  parentPort.on('message', job => {
    const stats = { n: 0, v: 0, v2: 0, r: 0, p: 0, pairs: 0 };
    for (let i = 0; i < job.count; i++) one(stats);
    parentPort.postMessage(stats);
  });
  return;
}

// ---------------------------------------------------------------- main
const argv = process.argv.slice(2), flags = {};
for (let i = 0; i < argv.length; i++) { const a = argv[i]; if (!a.startsWith('--')) continue; const v = argv[i + 1] && !argv[i + 1].startsWith('--') ? argv[++i] : true; flags[a.slice(2)] = v; }
const num = (k, d) => flags[k] === undefined ? d : +flags[k], str = (k, d) => flags[k] === undefined ? d : String(flags[k]);
const t0 = Date.now(), log = (...a) => console.log(`[${((Date.now() - t0) / 1000).toFixed(0)}s]`, ...a);
const r4 = x => x == null ? null : +x.toFixed(4);

// ---- evaluation (round 2)
let _tb = null, _hp = null;
function benches(opt) {
  if (!_tb) _tb = new A.Testbench(GAME, ALL, { maxDeep: 40, maxHard: 60, maxEasy: 20, rollouts: 12 });
  if (!_hp) {
    const pool = JSON.parse(fs.readFileSync(path.join(CACHE, 'pool.json'), 'utf8')), dist = JSON.parse(fs.readFileSync(path.join(CACHE, 'holdout-dist.json'), 'utf8'));
    const R = A.rng(77), hold = A.shuffle(pool.maps.filter(m => m.split === 'holdout'), R).slice(0, (opt && opt.hpMaps) || 30);
    const extra = hold.map(m => { const G = A.makeGraph(Object.assign({}, m.spec, { key: m.id, dist: dist[m.id] })); return { key: m.id, G, starts: [], census: Object.keys(dist[m.id]), levels: [] }; });
    _hp = new A.Testbench(GAME, [], { maxDeep: 12, maxHard: 8, maxEasy: 0, rollouts: 4, extra });
    _hp.keys = hold.map(m => m.id);
  }
  return { tb: _tb, hp: _hp };
}
function evalAgent(agent) {
  const { tb, hp } = benches(), per = tb.run(agent), hper = hp.run(agent);
  const rnd = {}; for (const k in per) { rnd[k] = {}; for (const f in per[k]) { const v = per[k][f]; rnd[k][f] = typeof v === 'number' ? r4(v) : v; } }
  return { train: A.aggregate(per, TRAIN_MAPS), test: A.aggregate(per, TEST_MAPS), hardpool: A.aggregate(hper, hp.keys), deepStarts: A.deepStarts(per), perMap: rnd, bench: 'full-v2' };
}
function quickBench() {   // every d >= 3 census state of the test maps + a few hard-pool states
  if (quickBench.S) return quickBench.S;
  const S = [];
  for (const key of TEST_MAPS) { const G = A.graphFromLevels(GAME, key); for (const k in G.dist) if (G.dist[k] >= 3) S.push({ G, col: A.decodeCanon(G, k), d: G.dist[k], set: 'test3' }); }
  const { hp } = benches({ hpMaps: 30 }), R = A.rng(5);
  hp.maps.forEach(M => { const deep = M.samples.filter(s => s.d >= 3); A.shuffle(deep, R).slice(0, 3).forEach(s => S.push({ G: M.G, col: s.col, d: s.d, set: 'hard3' })); });
  return (quickBench.S = S);
}
function quickEval(agent) {
  const S = quickBench(), o = {}, c = {};
  for (const s of S) { const mv = A.movesOf(s.G, s.col), i = agent(s.G, s.col), ok = i >= 0 && A.distOf(s.G, A.flip(s.col, mv[i])) === s.d - 1; o[s.set] = (o[s.set] || 0) + ok; c[s.set] = (c[s.set] || 0) + 1; }
  for (const k in o) o[k] = r4(o[k] / c[k]); return o;
}
function readIndex() { try { return JSON.parse(fs.readFileSync(path.join(MODELS, 'index.json'), 'utf8')); } catch (_) { return { format: 'mazearena-index-1', models: [], baselines: null }; } }
function writeIndexLocked(fn) {
  const lock = path.join(MODELS, '.index.lock');
  for (let i = 0; i < 600; i++) { try { fs.mkdirSync(lock); break; } catch (_) { Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 100); } }
  try { const ix = readIndex(); fn(ix); fs.writeFileSync(path.join(MODELS, 'index.json'), JSON.stringify(ix, null, 1) + '\n'); } finally { try { fs.rmdirSync(lock); } catch (_) {} }
}

if (flags.baselines) {
  const out = {};
  for (const b of ['random', 'hand', 'twoply']) {
    const R = A.rng(1001); out[b] = evalAgent((G, c) => A.baselines[b](G, c, R)); delete out[b].perMap;
    log(b, JSON.stringify({ test: out[b].test, hardpool: out[b].hardpool, deepStarts: out[b].deepStarts.within3 }));
  }
  writeIndexLocked(ix => { ix.baselines = Object.assign({ date: new Date().toISOString(), split: { train: TRAIN_MAPS, test: TEST_MAPS }, bench: 'full-v2' }, out); });
  process.exit(0);
}

if (flags.eval) {
  let M; const src = String(flags.eval);
  if (fs.existsSync(src)) M = JSON.parse(fs.readFileSync(src, 'utf8'));
  else M = JSON.parse(fs.readFileSync(path.join(CACHE, 'ckpt-' + src + '.json'), 'utf8')).model;
  const net = A.modelFromJSON(M), a = net.arch, gamma = (M.rl && M.rl.gamma) || 1, et0 = Date.now();
  const metrics = evalAgent(A.netAgent(net, 'value', gamma));
  log('value agent:', JSON.stringify({ test: metrics.test, hardpool: metrics.hardpool, deep: metrics.deepStarts }));
  if (a.kind === 'rec') {
    const pol = evalAgent(A.netAgent(net, 'policy')); delete pol.perMap; metrics.policyHead = pol;
    log('policy agent (click region, pick colour):', JSON.stringify({ test: pol.test, hardpool: pol.hardpool, deep: pol.deepStarts }));
    const n2 = A.modelFromJSON(M); n2.arch.T = 2 * a.T;
    const at2 = evalAgent(A.netAgent(n2, 'value', gamma)); delete at2.perMap;
    const pol2 = evalAgent(A.netAgent(n2, 'policy')); delete pol2.perMap;
    metrics.atT2 = { T: 2 * a.T, value: at2, policy: pol2 };
    log(`at T=${2 * a.T}: value ${JSON.stringify({ test: at2.test.agree3, hard: at2.hardpool.agree3 })} policy ${JSON.stringify({ test: pol2.test.agree3, hard: pol2.hardpool.agree3 })}`);
  }
  metrics.evalSeconds = Math.round((Date.now() - et0) / 1000);
  M.metrics = metrics; M.rl = M.rl || { gamma };
  if (flags.save) {
    const file = M.name.replace(/[^\w.-]+/g, '_') + '.json';
    fs.writeFileSync(path.join(MODELS, file), JSON.stringify(M));
    writeIndexLocked(ix => { ix.models = (ix.models || []).filter(m => m.file !== file);
      ix.models.push({ file, name: M.name, date: M.date, paramCount: M.paramCount, arch: { kind: a.kind || 'gnn', H: a.H, K: a.K, T: a.T, chainFeats: a.kind === 'rec' ? false : a.chainFeats },
        track: a.kind === 'rec' ? 'tabula rasa' : 'priors', pretrain: !!(M.training && M.training.pretrain), test: metrics.test, train: metrics.train, hardpool: metrics.hardpool, deepStarts: metrics.deepStarts,
        policyHead: metrics.policyHead ? { test: metrics.policyHead.test, hardpool: metrics.policyHead.hardpool, deepStarts: metrics.policyHead.deepStarts } : null,
        atT2: metrics.atT2 ? { T: metrics.atT2.T, test: metrics.atT2.value.test, hardpool: metrics.atT2.value.hardpool, policyTest: metrics.atT2.policy.test, policyHard: metrics.atT2.policy.hardpool } : null }); });
    log('saved docs/arena/models/' + file);
  }
  process.exit(0);
}

// ---- training
const archCfg = str('arch', 'rec') === 'rec' ? { kind: 'rec', H: num('H', 32), T: num('T', 16) } : { H: num('H', 16), K: num('K', 3), chainFeats: !flags['colours-only'] };
const cfg = { mix: str('mix', '0.15,0.35,0.3,0.2').split(',').map(Number), wval: num('wval', 1), wrank: num('wrank', 0.5), wpol: num('wpol', 1), tmin: num('tmin', Math.max(1, Math.round((archCfg.T || 2) / 2))) };
const NW = num('workers', 2), B = num('batch', 32), LR = num('lr', 1e-3), WD = num('wd', 1e-5), SEED = num('seed', 1), MIN = num('minutes', 60), name = str('name', 'scale-' + Date.now().toString(36));
const net = A.netCreate(archCfg, SEED), size = net.size;
const paramsSAB = new SharedArrayBuffer(size * 4), gradSAB = new SharedArrayBuffer(size * 4 * NW);
const P = new Float32Array(paramsSAB); P.set(net.params); net.params = P;
const grads = Array.from({ length: NW }, (_, i) => new Float32Array(gradSAB, i * size * 4, size)), gsum = new Float32Array(size);
let opt = A.adamCreate(net), step = 0, samples = 0, curve = [], prevWall = 0;
const ckptFile = path.join(CACHE, 'ckpt-' + name + '.json');
if (flags.resume && fs.existsSync(ckptFile)) {
  const C = JSON.parse(fs.readFileSync(ckptFile, 'utf8')), n2 = A.modelFromJSON(C.model); P.set(n2.params);
  opt.m = Float32Array.from(C.adam.m); opt.v = Float32Array.from(C.adam.v); opt.t = C.adam.t; step = C.step; samples = C.samples; curve = C.curve; prevWall = C.wall;
  log('resumed', name, 'at step', step);
}
log(`training ${name}: ${net.arch.kind === 'rec' ? 'tabula rasa rec H=' + net.arch.H + ' T=' + net.arch.T + ' (train T in ' + cfg.tmin + '..' + net.arch.T + ')' : 'priors gnn H=' + net.arch.H + ' K=' + net.arch.K} params ${size}, ${NW} workers, batch ${B}, lr ${LR}, mix ${cfg.mix}`);
const workers = Array.from({ length: NW }, (_, i) => new Worker(__filename, { workerData: { arch: net.arch, seed: SEED * 100 + i + step, paramsSAB, gradSAB, slot: i, size, cfg } }));
const per = Math.ceil(B / NW);
const meta = () => ({ name, rl: { gamma: 1, note: 'supervised distillation of exact distances; afterstate score = -1 + V(after)' },
  training: { method: 'supervised distillation on the hard pool (value + afterstate ranking' + (net.arch.kind === 'rec' ? ' + region/colour policy' : '') + ')', pool: 'docs/arena/cache/pool.json (tools/arena_hardgen.js)',
    trainMaps: TRAIN_MAPS, testMaps: TEST_MAPS, steps: step, samples, batch: B, lr: LR, wd: WD, workers: NW, mix: cfg.mix, weights: { value: cfg.wval, rank: cfg.wrank, policy: cfg.wpol }, tmin: cfg.tmin,
    augment: net.arch.kind === 'rec' ? 'random colour permutation' : 'random mirror', wallSeconds: Math.round(prevWall + (Date.now() - t0) / 1000), curve, pretrain: null, tool: 'tools/arena_scale.js ' + argv.join(' ') } });
function checkpoint() {
  const M = A.modelToJSON(net, meta());
  fs.writeFileSync(ckptFile + '.tmp', JSON.stringify({ model: M, adam: { m: Array.from(opt.m), v: Array.from(opt.v), t: opt.t }, step, samples, curve, wall: prevWall + (Date.now() - t0) / 1000 }));
  fs.renameSync(ckptFile + '.tmp', ckptFile);
}
let acc = { n: 0, v: 0, v2: 0, r: 0, p: 0, pairs: 0 }, lastCk = Date.now(), lastLog = Date.now(), lastSamples = samples, lastT = Date.now();
const ckMs = num('ckpt-min', 5) * 60000, stopAt = t0 + MIN * 60000;
function stepOnce() {
  return Promise.all(workers.map(w => new Promise(res => { w.once('message', res); w.postMessage({ count: per }); }))).then(rs => {
    gsum.fill(0); let n = 0;
    rs.forEach((s, i) => { n += s.n; for (const k in acc) acc[k] += s[k]; const g = grads[i]; for (let j = 0; j < size; j++) { gsum[j] += g[j]; g[j] = 0; } });
    A.adamStep(net, opt, gsum, n, { lr: LR * Math.min(1, (step + 1) / 200), wd: WD, clip: 5 });
    step++; samples += n;
  });
}
(async () => {
  if (flags.bench) {   // samples/s for this worker count
    const tb = Date.now(); let s0 = samples; while (Date.now() - tb < num('bench', 20) * 1000) await stepOnce();
    log(`bench ${NW} workers: ${((samples - s0) / ((Date.now() - tb) / 1000)).toFixed(1)} samples/s`); workers.forEach(w => w.terminate()); return;
  }
  while (Date.now() < stopAt) {
    await stepOnce();
    if (Date.now() - lastLog > 60000) {
      const sps = (samples - lastSamples) / ((Date.now() - lastT) / 1000); lastSamples = samples; lastT = Date.now(); lastLog = Date.now();
      log(`step ${step} samples ${samples} (${sps.toFixed(1)}/s) loss value ${(acc.v / acc.n).toFixed(3)} after ${(acc.v2 / Math.max(1, acc.pairs)).toFixed(3)} rank ${(acc.r / Math.max(1, acc.pairs)).toFixed(3)}${net.arch.kind === 'rec' ? ' policy ' + (acc.p / acc.n).toFixed(3) : ''}`);
      curve.push({ step, samples, minutes: +((prevWall + (Date.now() - t0) / 1000) / 60).toFixed(1), sps: +sps.toFixed(1), value: r4(acc.v / acc.n), rank: r4(acc.r / Math.max(1, acc.pairs)), policy: net.arch.kind === 'rec' ? r4(acc.p / acc.n) : null });
      acc = { n: 0, v: 0, v2: 0, r: 0, p: 0, pairs: 0 };
    }
    if (Date.now() - lastCk > ckMs) {
      lastCk = Date.now(); const q = quickEval(A.netAgent(net, 'value', 1)), qp = net.arch.kind === 'rec' ? quickEval(A.netAgent(net, 'policy')) : null;
      if (curve.length) { curve[curve.length - 1].quick = q; if (qp) curve[curve.length - 1].quickPolicy = qp; }
      checkpoint(); log('checkpoint; quick d>=3 agreement: value', JSON.stringify(q), qp ? 'policy ' + JSON.stringify(qp) : '');
    }
  }
  checkpoint(); log('done:', step, 'steps,', samples, 'samples; checkpoint', ckptFile);
  workers.forEach(w => w.terminate());
})();
