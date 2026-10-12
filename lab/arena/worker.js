/* lab/arena/worker.js: the Maze Arena's training loop, off the page's main thread.
 *
 * One network (the learner) is shared by every mouse (the actors) in every panel. Mice in panels tagged "train" explore
 * (epsilon-greedy or softmax) and feed the replay buffer; mice in panels tagged "test" walk their held-out mazes
 * greedily and never learn, so the panels show generalisation live. The worker steps the mice in time slices, learns
 * every `updateEvery` steps, posts mouse positions and stats a few times a second ("passively in parallel"), runs the
 * quick testbench every `evalEvery` episodes, and answers save / pretrain / size-sweep / watch requests.
 *
 * Messages in:  init | start {cfg} | pause | reset {cfg} | reshape {cfg} (panels, mice: keeps the network) | rl {rl} (live)
 *               | speed {speed} | heat {mode} | save {name} | pretrain {cfg, pre}
 *               sweep {cfg, configs, episodes} | watch {model} | load {model} | stopSweep
 * Messages out: ready {levels, maps} | panels {panels} | tick {...} | chart {point} | model {json, purpose} | status {text}
 *               | error {message}
 */
'use strict';
self.window = self;
importScripts('../../levels.js', 'engine.js');
const A = self.ARENA, GAME = self.GAME;
const MAPS = A.levelMaps(GAME), ALLKEYS = MAPS.map(m => m.key);
const LEVELS = MAPS.flatMap(m => m.levels.map(L => ({ id: L.id, num: String(L.num), g: L.g, par: L.par, n: GAME.graphs[L.g].n, mapName: m.name })));
const post = (type, data, tr) => self.postMessage(Object.assign({ type }, data || {}), tr || []);

let cfg = null, T = null, P = null, panels = [], running = false, phase = 'idle', speed = 'fast', heatMode = 'value';
let quick = null, full = null, nextEval = 0, lastPost = 0, lastHeat = 0, heatPanel = 0, sweep = null, watchName = null;
let wsteps = 0, stepsAtLast = 0, tAtLast = 0, sps = 0, chartPts = 0, preTarget = 0;
const mazeCache = new Map();

function mazeFor(levelId) {
  if (mazeCache.has(levelId)) return mazeCache.get(levelId);
  const L = GAME.levels.find(l => l.id === levelId), env = A.makeEnv(GAME, L.g), start = Int8Array.from(L.start); start[env.G.hole] = -1;
  const SG = A.stateGraph(env.G, start, { R: L.par + 2, cap: 300, whole: 160 }), pos = A.layoutStateGraph(SG, A.hashStr('arena:' + levelId));
  const rooms = []; SG.nodes.forEach((v, i) => { if (!v.won && v.dist != null) rooms.push(i); });
  const M = { L, env, SG, pos, rooms, start };
  mazeCache.set(levelId, M); return M;
}
function splitKeys(c) {
  const trainMaps = [...new Set(c.trainLevels.map(id => GAME.levels.find(l => l.id === id).g))];
  const testMaps = [...new Set(c.testLevels.map(id => GAME.levels.find(l => l.id === id).g))].filter(k => trainMaps.indexOf(k) < 0);
  return { trainMaps, testMaps };
}
function buildPanels() {
  const c = cfg, N = c.panels, train = c.trainLevels.slice(), test = c.testLevels.filter(id => splitKeys(c).testMaps.indexOf(GAME.levels.find(l => l.id === id).g) >= 0);
  let nTest = test.length && N > 1 ? Math.max(1, Math.round(N / 3)) : 0; if (!train.length) nTest = N;
  const nTrain = N - nTest;
  panels = [];
  for (let i = 0; i < N; i++) {
    const isTest = i >= nTrain, list = isTest ? test : train;
    if (!list.length) break;
    const k = isTest ? i - nTrain : i, id = list[k % list.length];
    panels.push({ i, tag: isTest ? 'test' : 'train', list, slot: k, id, mice: [], eps: 0, ok: 0, hist: [], rotateAt: c.rotateEvery || 0 });
  }
  panels.forEach(setPanelLevel);
  post('panels', { panels: panels.map(panelInfo) });
}
function setPanelLevel(p) {
  const M = mazeFor(p.id); p.M = M; p.visits = new Uint32Array(M.SG.nodes.length); p.heated = false; p.eps = 0; p.hist = [];
  const learn = p.tag === 'train' && phase !== 'watch';
  p.mice = [];
  if (!M.mirrorEnv) M.mirrorEnv = Object.assign({}, M.env, { G: A.mirrorGraph(M.env.G) });
  for (let m = 0; m < cfg.mice; m++) p.mice.push({ env: learn && cfg.mirror && m % 2 ? M.mirrorEnv : M.env, col: null, learn, greedy: !learn, room: -1, prev: -1,
    startFn: R => (R() < cfg.rl.startMix || !M.rooms.length ? M.start.slice() : M.SG.nodes[M.rooms[R.int(M.rooms.length)]].c.slice()) });
}
function panelInfo(p) {
  const SG = p.M.SG, N = SG.nodes.length, E = new Int32Array(SG.edges.length * 2);
  SG.edges.forEach(([a, b], i) => { E[2 * i] = a; E[2 * i + 1] = b; });
  const won = new Uint8Array(N), dist = new Int8Array(N); SG.nodes.forEach((v, i) => { won[i] = v.won ? 1 : 0; dist[i] = v.won ? 0 : v.dist == null ? -1 : v.dist; });
  const L = p.M.L;
  return { i: p.i, tag: p.tag, id: L.id, num: String(L.num), par: L.par, map: L.g, name: GAME.graphs[L.g].name, rooms: N, whole: SG.whole, pos: p.M.pos, edges: E, won, dist };
}
// splice variants of the training maps (single flip / insert / remove away from the hole) that keep states 3+ deep,
// each with its whole state space; walked by hidden mice that start in the deepest states
let hidden = [];
function buildSplices() {
  hidden = []; if (!cfg.splices || phase === 'watch') return;
  const R = A.rng(cfg.seed * 31 + 7), keys = splitKeys(cfg).trainMaps.filter(k => GAME.graphs[k].n >= 18); if (!keys.length) return;
  let tries = 0;
  while (hidden.length < 2 * cfg.splices && tries++ < cfg.splices * 12) {
    const tri = new A.Tri(A.graphFromLevels(GAME, R.pick(keys))); if (!A.spliceRandom(tri, R)) continue;
    const G = tri.toGraph('splice' + tries); if (A.checkTriangulation(G)) continue;
    const S = A.stateSpace(G, 60000); if (!S || A.hardness(S.hist).maxD < 3) continue;
    const dist = {}, deep = []; for (const i of S.unfilled) if (S.dist[i] > 0) { dist[S.keys[i]] = S.dist[i]; if (S.dist[i] >= 3) deep.push(S.keys[i]); }
    G.dist = dist; const env = { G, key: G.key, name: 'splice', levels: [], starts: [], census: Object.keys(dist) };
    for (let k = 0; k < 2; k++) hidden.push({ env, col: null, learn: true, greedy: false, startFn: r => A.decodeCanon(G, r() < 0.7 ? r.pick(deep) : r.pick(env.census)) });
  }
}
function newTrainer(c, net) {
  const T2 = new A.Trainer({ arch: c.arch, rl: c.rl, seed: c.seed, net });
  return T2;
}
function setupBench() {
  const { trainMaps, testMaps } = splitKeys(cfg);
  quick = new A.Testbench(GAME, testMaps.concat(trainMaps), { maxDeep: 6, maxHard: 14, maxEasy: 2, rollouts: 0 });
  quick.trainMaps = trainMaps; quick.testMaps = testMaps;
}
function quickEval() {
  const r = quick.run(A.netAgent(T.net, T.mode, T.rl.gamma));
  return { train: A.aggregate(r, quick.trainMaps), test: A.aggregate(r, quick.testMaps) };
}
function fullEval(net, mode, gamma) {
  if (!full) full = new A.Testbench(GAME, ALLKEYS, { maxDeep: 40, maxHard: 60, maxEasy: 20, rollouts: 12 });
  const { trainMaps, testMaps } = splitKeys(cfg), perMap = full.run(A.netAgent(net, mode || 'value', gamma));
  const rnd = {}; for (const k in perMap) { rnd[k] = {}; for (const f in perMap[k]) { const v = perMap[k][f]; rnd[k][f] = typeof v === 'number' ? +v.toFixed(4) : v; } }
  return { train: A.aggregate(perMap, trainMaps), test: A.aggregate(perMap, testMaps), deepStarts: A.deepStarts(perMap), perMap: rnd, bench: 'full-v2 (no hard pool in the browser)' };
}
function reset(c) {
  cfg = c; phase = 'idle'; running = false; sweep = null; watchName = null; P = null;
  T = newTrainer(cfg); nextEval = cfg.evalEvery; chartPts = 0; T.wall = 0;
  setupBench(); buildPanels(); buildSplices(); postTick(true);
  const sk = splitKeys(cfg);
  post('status', { text: 'new ' + (T.net.arch.kind === 'rec' ? 'tabula-rasa' : 'priors') + ' network: ' + sk.trainMaps.length + ' training maps, ' + sk.testMaps.length + ' held-out maps, ' + panels.length + ' panels x ' + cfg.mice + ' mice' + (hidden.length ? ', ' + hidden.length / 2 + ' splice variants (' + hidden.length + ' hidden mice)' : '') });
}

// ---- the loop
function stepMice(rounds) {
  if (phase === 'train') for (let r = 0; r < rounds; r++) for (const m of hidden) { T.actorStep(m); if (T.steps % T.rl.updateEvery === 0) T.learn(); }
  for (let r = 0; r < rounds; r++) for (const p of panels) for (const m of p.mice) {
    const ev = T.actorStep(m); if (!m.learn) wsteps++;
    if (m.learn && T.steps % T.rl.updateEvery === 0 && phase === 'train') T.learn();
    const idx = p.M.SG.idx.get(A.canon(p.M.env.G, ev.to));
    m.prev = m.room; m.room = ev.ended ? -2 : idx === undefined ? -1 : idx;   // -2: just ended (a fill or the cap)
    if (m.room >= 0) p.visits[m.room]++;
    if (ev.ended) { p.eps++; p.hist.push(ev.ok ? 1 : 0); if (p.hist.length > 50) p.hist.shift(); m.lastOk = ev.ok; if (ev.ok) { const w = p.M.SG.idx.get(A.canon(p.M.env.G, ev.to)); m.fill = w === undefined ? -1 : w; } }
    else m.fill = -1;
  }
}
function rotatePanels() {
  let changed = false;
  for (const p of panels) {
    if (!cfg.rotateEvery || p.list.length <= 1 || p.eps < cfg.rotateEvery) continue;
    const step = panels.filter(q => q.tag === p.tag).length;
    p.slot = (p.slot + step) % p.list.length; const id = p.list[p.slot];
    if (id !== p.id) { p.id = id; setPanelLevel(p); p.fresh = true; changed = true; }
    else p.eps = 0;
  }
  if (changed) post('panels', { panels: panels.filter(p => p.fresh).map(panelInfo), partial: true });
  panels.forEach(p => { p.fresh = false; });
}
function maybeEval() {
  if (phase !== 'train' || T.episodes < nextEval) return;
  nextEval = T.episodes + cfg.evalEvery;
  const q = quickEval(), rec = T.recent.filter(r => r.learn).slice(-300), sr = rec.length ? rec.filter(r => r.ok).length / rec.length : null;
  chartPts++;
  post('chart', { point: { episodes: T.episodes, steps: T.steps, loss: T.loss, trainSuccess: sr, trainAgree2: q.train.agree2, testAgree2: q.test.agree2, testAgree3: q.test.agree3, run: sweep ? sweep.label : 'run' } });
}
function heatOf(p) {
  const SG = p.M.SG, N = SG.nodes.length, h = new Float32Array(N);
  if (heatMode === 'visits') { let mx = 1; for (let i = 0; i < N; i++) mx = Math.max(mx, p.visits[i]); for (let i = 0; i < N; i++) h[i] = Math.log1p(p.visits[i]) / Math.log1p(mx); return h; }
  // value: the network's learned steps to a fill, -V, scaled 0..4 -> 0..1 (fills are 0)
  for (let i = 0; i < N; i++) { const v = SG.nodes[i]; h[i] = v.won ? 0 : Math.min(1, Math.max(0, -A.netValue(T.net, A.encode(p.M.env.G, v.c, T.arch)) / 4)); }
  return h;
}
function postTick(force) {
  const t = A.now();
  const stepsNow = phase === 'watch' ? wsteps : T.steps;   // watch mode: no learning mice, count the greedy ones
  if (t - tAtLast > 500) { sps = Math.max(0, stepsNow - stepsAtLast) / ((t - tAtLast) / 1000); stepsAtLast = stepsNow; tAtLast = t; }
  const mice = panels.map(p => { const a = new Int16Array(p.mice.length * 3); p.mice.forEach((m, k) => { a[3 * k] = m.room; a[3 * k + 1] = m.prev; a[3 * k + 2] = m.fill == null ? -1 : m.fill; }); return a; });
  const pstats = panels.map(p => ({ eps: p.eps, succ: p.hist.length ? p.hist.reduce((x, y) => x + y, 0) / p.hist.length : null }));
  let heat = null;
  if (heatMode !== 'none' && (force || t - lastHeat > (heatMode === 'visits' ? 700 : 1500)) && panels.length) {
    lastHeat = t; const cold = panels.findIndex(q => !q.heated);   // panels that just got a new maze first
    heatPanel = cold >= 0 ? cold : (heatPanel + 1) % panels.length; const p = panels[heatPanel]; p.heated = true;
    heat = { panel: heatPanel, values: heatOf(p) };
  }
  const pre = P ? { step: P.step, target: preTarget, graphs: P.graphs, samples: P.samples, loss: P.loss, n: P.nRange(), gps: P.graphs / Math.max(0.01, P.wall) } : null;
  post('tick', { mice, pstats, heat, phase, running, watchName, sweep: sweep ? { i: sweep.i, n: sweep.configs.length, label: sweep.label } : null, pre,
    stats: { episodes: phase === 'watch' ? panels.reduce((a, p) => a + p.eps, 0) : T.episodes, steps: phase === 'watch' ? wsteps : T.steps, updates: T.updates, loss: T.loss, aux: T.auxLoss, eps: T.eps(), sps, params: T.net.size, replay: T.replay.length, wall: T.wall } });
  lastPost = t;
}
function pump() {
  if (!running) return;
  const t0 = A.now();
  try {
    if (phase === 'pretrain') {
      while (A.now() - t0 < 40 && P.step < preTarget) P.trainStep();
      P.handover();   // mice walk with V = -dist head while the trunk pretrains
      if (A.now() - lastPost > 300) { stepMice(1); postTick(); }
      if (P.step >= preTarget) finishPretrain();
    } else if (speed === 'slow') {
      stepMice(1); maybeEval(); rotatePanels(); postTick();
    } else {
      let r = 0;
      while (A.now() - t0 < 45) { stepMice(1); r++; maybeEval(); }
      rotatePanels();
      if (A.now() - lastPost > (speed === 'turbo' ? 1000 : 120)) postTick();
      if (sweep && T.episodes >= sweep.episodes) sweepNext();
    }
    T.wall += (A.now() - t0) / 1000;
  } catch (e) { running = false; post('error', { message: String(e && e.stack || e) }); return; }
  setTimeout(pump, speed === 'slow' && phase !== 'pretrain' ? 160 : 0);
}
function start() { if (!running) { running = true; if (phase === 'idle') phase = 'train'; tAtLast = A.now(); stepsAtLast = T.steps; pump(); } }

// ---- models
function meta(name, extra) {
  const { trainMaps, testMaps } = splitKeys(cfg);
  return { name, rl: Object.assign({}, T.rl, { seed: cfg.seed, panels: cfg.panels, mice: cfg.mice }),
    training: { trainMaps, testMaps, trainLevels: cfg.trainLevels, episodes: T.episodes, steps: T.steps, updates: T.updates, wallSeconds: +T.wall.toFixed(1),
      pretrain: T.pretrain || null, tool: 'lab/MazeArena.html' + (extra ? ' ' + extra : '') } };
}
function saveModel(name, purpose, extra) {
  post('status', { text: 'running the full testbench on all ' + ALLKEYS.length + ' maps…' });
  const t0 = A.now(), metrics = fullEval(T.net, 'value', T.rl.gamma);
  const M = A.modelToJSON(T.net, Object.assign(meta(name, extra), { metrics }));
  post('model', { json: M, purpose });
  post('status', { text: 'testbench done in ' + ((A.now() - t0) / 1000).toFixed(1) + ' s: held-out agree2 ' + fmt(metrics.test.agree2) + ', train ' + fmt(metrics.train.agree2) });
}
const fmt = x => x == null ? '–' : (100 * x).toFixed(1) + '%';

// ---- pretraining
function startPretrain(pre) {
  phase = 'pretrain';
  const envs = splitKeys(cfg).trainMaps.map(k => A.makeEnv(GAME, k));
  P = new A.Pretrainer(T, pre, envs); preTarget = P.cfg.steps;
  running = true; tAtLast = A.now(); stepsAtLast = T.steps; pump();
}
function finishPretrain() {
  P.handover(); T.pretrain = P.summary();
  const q = quickEval();
  post('chart', { point: { episodes: T.episodes, steps: T.steps, loss: null, trainSuccess: null, trainAgree2: q.train.agree2, testAgree2: q.test.agree2, testAgree3: q.test.agree3, run: 'pretrained', marker: 'pretrained' } });
  post('status', { text: 'pretrained ' + P.step + ' steps on ' + P.graphs + ' random maps (' + P.labelled + ' with exact distances) in ' + P.wall.toFixed(0) + ' s; zero-shot held-out agree2 ' + fmt(q.test.agree2) + '. Fine-tuning with RL…' });
  P = null; phase = 'train'; nextEval = T.episodes + cfg.evalEvery;
}

// ---- size sweep: the same RL budget for each (H, K); each result goes to the leaderboard
function sweepNext() {
  if (sweep.i >= 0) saveModel('sweep ' + sweep.label + ' ' + new Date().toISOString().slice(5, 16).replace('T', ' '), 'sweep', 'size sweep');
  sweep.i++;
  if (sweep.i >= sweep.configs.length) { post('status', { text: 'size sweep done: ' + sweep.configs.length + ' models on the leaderboard' }); sweep = null; running = false; phase = 'idle'; postTick(true); return; }
  const [H, K] = sweep.configs[sweep.i]; sweep.label = H + 'x' + K;
  cfg.arch = cfg.arch.kind === 'rec' ? { kind: 'rec', H, T: K } : Object.assign({}, cfg.arch, { H, K, H1: 0, Hp: 0 });
  T = newTrainer(cfg); nextEval = cfg.evalEvery; phase = 'train';
  panels.forEach(p => { setPanelLevel(p); });
  post('status', { text: 'size sweep ' + (sweep.i + 1) + '/' + sweep.configs.length + ': H=' + H + (cfg.arch.kind === 'rec' ? ' T=' : ' K=') + K + ' (' + T.net.size + ' params), ' + sweep.episodes + ' episodes' });
}

self.onmessage = e => {
  const d = e.data;
  try {
    if (d.type === 'init') post('ready', { levels: LEVELS, maps: MAPS.map(m => ({ key: m.key, name: m.name, n: m.n, levels: m.levels.map(L => L.id) })) });
    else if (d.type === 'reset') reset(d.cfg);
    else if (d.type === 'start') { if (!T) reset(d.cfg); if (phase === 'watch') { phase = 'train'; reset(d.cfg); } start(); }
    else if (d.type === 'pause') { running = false; postTick(true); }
    else if (d.type === 'speed') speed = d.speed;
    else if (d.type === 'reshape') { if (!T) reset(d.cfg); else { cfg.panels = d.cfg.panels; cfg.mice = d.cfg.mice; cfg.rotateEvery = d.cfg.rotateEvery; buildPanels(); postTick(true); } }
    else if (d.type === 'rl') { if (T) { const keep = { replay: T.rl.replay }; Object.assign(T.rl, d.rl, keep); cfg.rl = Object.assign(cfg.rl || {}, d.rl); cfg.evalEvery = d.evalEvery || cfg.evalEvery; } }
    else if (d.type === 'heat') { heatMode = d.mode; lastHeat = 0; }
    else if (d.type === 'save') saveModel(d.name, 'save');
    else if (d.type === 'pretrain') { reset(d.cfg); startPretrain(d.pre); }
    else if (d.type === 'sweep') { reset(d.cfg); sweep = { configs: d.configs, episodes: d.episodes, i: -1, label: '' }; sweepNext(); start(); }
    else if (d.type === 'stopSweep') { sweep = null; running = false; phase = 'idle'; postTick(true); }
    else if (d.type === 'watch' || d.type === 'load') {
      const net = A.modelFromJSON(d.model);
      cfg = d.cfg; cfg.arch = net.arch.kind === 'rec' ? { kind: 'rec', H: net.arch.H, T: net.arch.T, H1: net.arch.H1, Hq: net.arch.Hq, holeRead: net.arch.holeRead }
        : Object.assign({}, cfg.arch, { kind: 'gnn', H: net.arch.H, K: net.arch.K, H1: net.arch.H1, Hp: net.arch.Hp, chainFeats: net.arch.chainFeats });
      if (d.model.rl && d.model.rl.gamma != null) cfg.rl.gamma = d.model.rl.gamma;
      sweep = null; P = null; running = false;
      T = newTrainer(cfg, net); T.wall = 0; nextEval = cfg.evalEvery; T.pretrain = d.model.training && d.model.training.pretrain || null;
      setupBench();
      if (d.type === 'watch') { phase = 'watch'; watchName = d.model.name; T.rl.epsStart = T.rl.epsEnd = 0; }
      else { phase = 'idle'; watchName = null; }
      buildPanels();
      if (d.type === 'watch') start(); else postTick(true);
      post('status', { text: (d.type === 'watch' ? 'watching ' : 'loaded ') + d.model.name + ' (' + net.size + ' params)' + (d.type === 'watch' ? ': every mouse plays greedily, no learning' : ': press Start to keep training it') });
    }
  } catch (err) { post('error', { message: String(err && err.stack || err) }); }
};
