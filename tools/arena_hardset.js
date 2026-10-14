// node tools/arena_hardset.js CANDIDATES.json OUT.json [--shard i/k] [--cap 1500000]
//   Studio Job 10 (14 Oct 2026, exploratory): exact census distances for the proof-hard map pool.
//   For each candidate (map spec + source + why, from tools/hardset_select.py): build the graph with the arena engine,
//   check it is a sphere triangulation with a degree-5 hole (checkTriangulation), enumerate the whole state space
//   (stateSpace: every proper 4-colouring of T - hole up to renaming, swap graph, BFS from the fills) and record the
//   distance histogram of unfilled states, max distance, unreachable count and the engine's hardness score.
//   Maps whose state space exceeds --cap get distances "unknown". Shards are merged by tools/arena_hardset_merge.js.
'use strict';
const fs = require('fs'), path = require('path');
const root = path.join(__dirname, '..');
global.window = global;
require(path.join(root, 'docs/levels.js'));
const A = require(path.join(root, 'docs/arena/engine.js'));
const argv = process.argv.slice(2);
const flag = (k, d) => { const i = argv.indexOf('--' + k); return i >= 0 ? argv[i + 1] : d; };
const [inF, outF] = argv;
const CAP = +flag('cap', 1500000);
const [si, sk] = flag('shard', '0/1').split('/').map(Number);
const C = JSON.parse(fs.readFileSync(inF, 'utf8'));
const out = [];
C.forEach((c, i) => {
  if (i % sk !== si) return;
  const t0 = Date.now();
  const G = A.makeGraph(c);
  const bad = A.checkTriangulation(G);
  const rec = Object.assign({}, c);
  if (bad) { rec.check = bad; rec.dist = 'unknown'; out.push(rec); return; }
  const S = A.stateSpace(G, CAP);
  if (!S) { rec.dist = 'unknown'; rec.distNote = 'state space > ' + CAP; }
  else {
    const h = A.hardness(S.hist);
    rec.dist = { N: S.N, unfilled: S.unfilled.length, hist: S.hist, maxD: h.maxD, unreachable: S.unreachable, hardness: h.score };
  }
  rec.sec = (Date.now() - t0) / 1000;
  out.push(rec);
  console.log(i, c.key, c.n, rec.dist === 'unknown' ? 'unknown' : ('N=' + rec.dist.N + ' maxD=' + rec.dist.maxD + ' hist=' + JSON.stringify(rec.dist.hist)), rec.sec.toFixed(1) + 's');
});
fs.writeFileSync(outF, JSON.stringify(out));
