// node tools/arena_train.js [flags] : headless Maze Arena training (pretraining + RL), writes a model into
// lab/arena/models/<name>.json and lists it in lab/arena/models/index.json (the committed leaderboard).
//
//   --name tiny-rl              model name (file name too)
//   --H 16 --K 3 --H1 0 --Hp 0  network size (H1, Hp default 2H and H); --colours-only drops the chain features
//   --lr 1e-3 --gamma 0.97 --eps-start 0.5 --eps-end 0.05 --eps-decay 3000 --batch 32 --replay 20000
//   --target-sync 200 --nstep 2 --backup td|max --aux 0 --cap 20 --update-every 4 --start-mix 0.3 --explore egreedy|softmax
//   --seed 1 --episodes 20000 --minutes 5      RL budget: stops at whichever comes first (0 = no limit)
//   --actors 4                 mice per training map (all share one network)
//   --train w9,w13,...         training maps (graph keys); default: the warm-ups + depth ladder split below
//   --test c30,c60,...          held-out maps; default: every other 4-colour map
//   --pretrain-steps 0 --pretrain-minutes 0 --pretrain-source generated|maps|both --pretrain-batch 16 --pretrain-lr 2e-3
//   --pretrain-n 12,18,30 (min, start max, end max) --pretrain-exact 26 --pretrain-weights 1,1,0.5,0.5 (dist, good swap,
//   masked colour, chain)
//   --eval-every 2000          quick held-out check (episodes), logged only; the final model is the last one
//   --baselines                 only evaluate the random / hand-rule / two-ply baselines and store them in index.json
//   --no-save                   do not write files
'use strict';
const fs = require('fs'), path = require('path');
const root = path.join(__dirname, '..');
global.window = global;
require(path.join(root, 'levels.js'));
const A = require(path.join(root, 'lab/arena/engine.js'));
const GAME = window.GAME;
const MODELS = path.join(root, 'lab/arena/models');

const argv = process.argv.slice(2), flags = {};
for (let i = 0; i < argv.length; i++) {
  const a = argv[i]; if (!a.startsWith('--')) continue;
  const k = a.slice(2), v = argv[i + 1] && !argv[i + 1].startsWith('--') ? argv[++i] : true; flags[k] = v;
}
const num = (k, d) => flags[k] === undefined ? d : +flags[k];
const str = (k, d) => flags[k] === undefined ? d : String(flags[k]);

const ALL = A.levelMaps(GAME).map(m => m.key);
const DEFAULT_TRAIN = ['w9', 'w13', 'w18', 'ico', 'p22', 'r1', 'r178', 'p32r111', 'p31r12'];
const trainKeys = flags.train ? String(flags.train).split(',') : DEFAULT_TRAIN;
const testKeys = flags.test ? String(flags.test).split(',') : ALL.filter(k => trainKeys.indexOf(k) < 0);
const log = (...a) => console.log(...a);

function readIndex() { try { return JSON.parse(fs.readFileSync(path.join(MODELS, 'index.json'), 'utf8')); } catch (_) { return { format: 'mazearena-index-1', models: [], baselines: null }; } }
function writeIndex(ix) { fs.mkdirSync(MODELS, { recursive: true }); fs.writeFileSync(path.join(MODELS, 'index.json'), JSON.stringify(ix, null, 1) + '\n'); }

function fullBench() { return new A.Testbench(GAME, ALL, { maxHard: 60, maxEasy: 20, rollouts: 12 }); }
function summarise(perMap) { return { train: A.aggregate(perMap, trainKeys), test: A.aggregate(perMap, testKeys) }; }
function round(perMap) { const o = {}; for (const k in perMap) { o[k] = {}; for (const f in perMap[k]) { const v = perMap[k][f]; o[k][f] = typeof v === 'number' ? +v.toFixed(4) : v; } } return o; }

if (flags.baselines) {
  const TB = fullBench(), out = {};
  for (const b of ['random', 'hand', 'twoply']) {
    // average the randomised baselines over 5 seeds
    const runs = [1, 2, 3, 4, 5].map(s => { const R = A.rng(1000 + s); return TB.run((G, c) => A.baselines[b](G, c, R)); });
    const perMap = {};
    for (const k of ALL) { perMap[k] = {}; for (const f in runs[0][k]) { const vals = runs.map(r => r[k][f]).filter(v => v != null); perMap[k][f] = vals.length ? vals.reduce((x, y) => x + y, 0) / vals.length : null; } }
    out[b] = { perMap: round(perMap), ...summarise(perMap) };
    log(b.padEnd(7), 'train', JSON.stringify(out[b].train), '\n        test ', JSON.stringify(out[b].test));
  }
  if (!flags['no-save']) { const ix = readIndex(); ix.baselines = { date: new Date().toISOString(), split: { train: trainKeys, test: testKeys }, ...out }; writeIndex(ix); log('wrote baselines to index.json'); }
  process.exit(0);
}

const arch = { H: num('H', 16), K: num('K', 3), H1: num('H1', 0), Hp: num('Hp', 0), chainFeats: !flags['colours-only'] };
const rl = A.rlDefaults({ lr: num('lr', 1e-3), gamma: num('gamma', 0.97), epsStart: num('eps-start', 0.5), epsEnd: num('eps-end', 0.05), epsDecay: num('eps-decay', 3000),
  batch: num('batch', 32), replay: num('replay', 20000), targetSync: num('target-sync', 200), nstep: num('nstep', 2), backup: str('backup', 'td'),
  auxWeight: num('aux', 0), episodeCap: num('cap', 20), updateEvery: num('update-every', 4), startMix: num('start-mix', 0.3), explore: str('explore', 'egreedy') });
const seed = num('seed', 1), name = str('name', 'model-' + Date.now().toString(36));
const T = new A.Trainer({ arch, rl, seed });
log(`model ${name}: H=${T.arch.H} K=${T.arch.K} H1=${T.arch.H1} Hp=${T.arch.Hp} params ${T.net.size}${arch.chainFeats ? '' : ' (colours only)'}`);
log('train maps', trainKeys.join(','), '| held-out', testKeys.join(','));
const envs = trainKeys.map(k => A.makeEnv(GAME, k)).filter(Boolean);
const quick = new A.Testbench(GAME, testKeys.concat(trainKeys), { maxHard: 20, maxEasy: 4, rollouts: 0 });
const quickEval = () => { const r = quick.run(A.netAgent(T.net, T.mode, T.rl.gamma)); return { train: A.aggregate(r, trainKeys), test: A.aggregate(r, testKeys) }; };

// ---- pretraining
let pre = null;
const preSteps = num('pretrain-steps', 0), preMin = num('pretrain-minutes', 0);
if (preSteps > 0 || preMin > 0) {
  const nr = str('pretrain-n', '12,18,30').split(',').map(Number), wts = str('pretrain-weights', '1,1,0.5,0.5').split(',').map(Number);
  const P = new A.Pretrainer(T, { steps: preSteps || 1e9, batch: num('pretrain-batch', 16), lr: num('pretrain-lr', 2e-3), nMin: nr[0], nMax: nr[1], nMaxEnd: nr[2],
    exactMax: num('pretrain-exact', 26), source: str('pretrain-source', 'generated'), wA: wts[0], wB: wts[1], wC: wts[2], wD: wts[3], seed: seed + 100 }, envs);
  const t0 = Date.now(), lim = preMin > 0 ? preMin * 60000 : Infinity, target = preSteps || Infinity;
  if (preMin > 0 && !preSteps) P.cfg.steps = Math.max(1000, Math.round(preMin * 60 * 8));   // curriculum length estimate
  let lastLog = 0;
  while (P.step < target && Date.now() - t0 < lim) {
    P.trainStep();
    if (Date.now() - lastLog > 30000) {
      lastLog = Date.now(); const q = quickEval();
      log(`  pretrain step ${P.step} graphs ${P.graphs} samples ${P.samples} n<=${P.nRange()[1]} loss dist ${P.loss.a.toFixed(3)} good ${P.loss.b.toFixed(3)} ssl ${P.loss.c.toFixed(3)} | policy-head test agree2 ${A.aggregate(quick.run(A.netAgent(T.net, 'policy')), testKeys).agree2}`);
    }
  }
  P.cfg.steps = P.step;
  P.handover();
  pre = P.summary();
  const qv = quickEval(), qp = A.aggregate(quick.run(A.netAgent(T.net, 'policy')), testKeys);
  log(`pretraining done: ${pre.steps} steps, ${pre.graphs} graphs (${pre.labelledGraphs} with exact labels), ${pre.samples} samples, ${pre.wallSeconds}s; ` +
    `held-out agree2 (value from -dist head) ${qv.test.agree2}, (policy head) ${qp.agree2}`);
}

// ---- RL
const maxEp = num('episodes', 20000), maxMin = num('minutes', 0), evalEvery = num('eval-every', 2000);
const actors = []; for (const env of envs) for (let i = 0; i < num('actors', 4); i++) actors.push({ env, col: null, learn: true });
const t0 = Date.now(); let nextEval = evalEvery, k = 0;
const curve = [];
if (maxEp > 0) {
  while (T.episodes < maxEp && (maxMin <= 0 || Date.now() - t0 < maxMin * 60000)) {
    T.actorStep(actors[k]); k = (k + 1) % actors.length;
    if (T.steps % rl.updateEvery === 0) T.learn();
    if (T.episodes >= nextEval) {
      nextEval += evalEvery; const q = quickEval(), rec = T.recent.slice(-400), sr = rec.filter(r => r.ok).length / Math.max(1, rec.length);
      curve.push({ episodes: T.episodes, steps: T.steps, loss: +T.loss.toFixed(4), trainSuccess: +sr.toFixed(3), trainAgree2: q.train.agree2, testAgree2: q.test.agree2 });
      log(`  ep ${T.episodes} steps ${T.steps} eps ${T.eps().toFixed(2)} loss ${T.loss.toFixed(3)} success ${sr.toFixed(2)} | quick agree2 train ${q.train.agree2} test ${q.test.agree2} | ${((Date.now() - t0) / 1000).toFixed(0)}s`);
    }
  }
}
const wall = (Date.now() - t0) / 1000;
log(`RL done: ${T.episodes} episodes, ${T.steps} steps, ${T.updates} updates, ${wall.toFixed(0)}s (${(T.steps / Math.max(1, wall)).toFixed(0)} env steps/s incl. learning)`);

// ---- full testbench
const tb0 = Date.now(), TB = fullBench(), perMap = TB.run(A.netAgent(T.net, 'value', rl.gamma)), S = summarise(perMap);
let policyMetrics = null;
if (pre && pre.weights.goodSwap > 0) { const pm = TB.run(A.netAgent(T.net, 'policy')); policyMetrics = summarise(pm); }
log(`testbench (${((Date.now() - tb0) / 1000).toFixed(0)}s):\n  train ${JSON.stringify(S.train)}\n  test  ${JSON.stringify(S.test)}`);
if (policyMetrics) log(`  policy head: train agree2 ${policyMetrics.train.agree2} test agree2 ${policyMetrics.test.agree2}`);

if (!flags['no-save']) {
  const M = A.modelToJSON(T.net, {
    name, rl: Object.assign({}, rl, { seed, actorsPerMap: num('actors', 4) }),
    training: { trainMaps: trainKeys, testMaps: testKeys, episodes: T.episodes, steps: T.steps, updates: T.updates, wallSeconds: +wall.toFixed(1), pretrain: pre, curve, tool: 'tools/arena_train.js ' + argv.join(' ') },
    metrics: { train: S.train, test: S.test, perMap: round(perMap), policyHead: policyMetrics ? { train: policyMetrics.train, test: policyMetrics.test } : null, bench: 'full-v1' },
  });
  fs.mkdirSync(MODELS, { recursive: true });
  const file = name.replace(/[^\w.-]+/g, '_') + '.json';
  fs.writeFileSync(path.join(MODELS, file), JSON.stringify(M));
  const ix = readIndex(); ix.models = (ix.models || []).filter(m => m.file !== file);
  ix.models.push({ file, name, date: M.date, paramCount: M.paramCount, arch: { H: M.arch.H, K: M.arch.K, chainFeats: M.arch.chainFeats }, pretrain: !!pre, test: S.test, train: S.train });
  writeIndex(ix);
  log('wrote lab/arena/models/' + file + ' (' + (JSON.stringify(M).length / 1024).toFixed(0) + ' KB) and index.json');
}
