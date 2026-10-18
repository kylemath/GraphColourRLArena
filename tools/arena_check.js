// node tools/arena_check.js : checks the Maze Arena engine (docs/arena/engine.js).
//   1. the puzzle core against docs/core.js on several levels: chains (webs flagged), swaps, canonical forms, census
//      distances and good moves, walking random swap sequences;
//   2. exactDistances (BFS over the Kempe class) reproduces the census on the small maps;
//   3. the encoding is map-independent: renaming the colours or rotating the link list does not change it;
//   4. a gradient check of the GNN (every head, Float64 copy, central differences);
//   5. the random-map generator: sphere triangulations (Euler, all faces triangles), a degree-5 hole, proper
//      colourings with a blocked hole;
//   6. a model file round trip (both architectures);
//   9. round 2: the tabula-rasa recurrent net (gradient check with the policy head, colour-permutation equivariance of
//      the inputs), whole state spaces equal the census on every map, mirror images keep every label, single splices
//      give valid triangulations with a degree-5 hole and proper colourings, the (region, colour) action table;
//  10. optional spectral inputs (arch.spectral): eigenpairs solve L v = l v, HKS/WKS do not depend on how regions
//      are numbered, both nets pass a gradient check with them on, model files keep the setting, and off is unchanged;
//   7. a short RL run on tidy-trap improves optimal-move agreement there;
//   8. speeds: environment steps / s, forward+backward samples / s, generator graphs / s.
// Flags: --quick (skip the RL run), --speed-only.
'use strict';
const path = require('path');
const root = path.join(__dirname, '..');
global.window = global; global.location = { search: '' };
require(path.join(root, 'docs/levels.js'));
require(path.join(root, 'docs/core.js'));
const A = require(path.join(root, 'docs/arena/engine.js'));
const GAME = window.GAME, LAB = window.LAB;
const args = process.argv.slice(2), QUICK = args.includes('--quick'), SPEED = args.includes('--speed-only');
let bad = 0;
const fail = msg => { bad++; console.log('  FAIL ' + msg); };
const ok = (c, msg) => { if (!c) fail(msg); };

if (!SPEED) {
  // 1. core semantics
  console.log('1. engine vs docs/core.js');
  for (const id of ['tidy-trap', 'kempes-double', 'six-ring-trap', 'depth-4', 'cage', 'round-and-round', 'twin-loops']) {
    const g = LAB.game(id), L = g.level, G = A.graphFromLevels(GAME, L.g), R = A.rng(A.hashStr(id));
    let checks = 0;
    for (let walk = 0; walk < 6; walk++) {
      g.reset();
      for (let step = 0; step < 12 && !g.won(); step++) {
        const col = Int8Array.from(g.col);
        const mine = A.chainsOf(G, col), theirs = g.chains;
        const sig = K => (K.web ? 'W' : '') + Math.min(K.a, K.b) + Math.max(K.a, K.b) + ':' + Array.from(K.vs).sort((x, y) => x - y).join('.');
        const sig2 = K => (K.web ? 'W' : '') + K.pair[0] + K.pair[1] + ':' + K.vs.join('.');
        const s1 = mine.map(sig).sort().join('|'), s2 = theirs.map(sig2).sort().join('|');
        ok(s1 === s2, id + ' chains differ at step ' + step);
        ok(A.canon(G, col) === g.canon(), id + ' canon differs');
        ok(A.distOf(G, col) === g.dist(), id + ' dist ' + A.distOf(G, col) + ' vs ' + g.dist());
        const goodMine = A.movesOf(G, col).filter(K => A.distOf(G, A.flip(col, K)) === g.dist() - 1).length;
        ok(goodMine === g.good().length, id + ' good moves ' + goodMine + ' vs ' + g.good().length);
        ok(A.won(G, col) === g.won(), id + ' won differs');
        const live = theirs.filter(K => !K.web); if (!live.length) break;
        const K = live[R.int(live.length)];
        const mk = mine.find(M => sig(M) === sig2(K));
        const after = A.flip(col, mk); g.swap(K);
        ok(A.canon(G, after) === g.canon(), id + ' swap differs');
        checks++;
      }
    }
    console.log('   ' + id.padEnd(16) + ' ' + checks + ' states checked');
  }
  // 2. exact distances
  console.log('2. exactDistances vs census');
  for (const key of ['trap17', 'c30', 'six', 'p22']) {
    const G = A.graphFromLevels(GAME, key), L = GAME.levels.find(l => l.g === key), st = Int8Array.from(L.start); st[G.hole] = -1;
    const ex = A.exactDistances(G, st, 20000);
    let agree = 0, n = 0; for (const [k, d] of ex.map) { n++; if (G.dist[k] === d) agree++; }
    ok(agree === n && n > 0, key + ' exact distances disagree with the census ' + agree + '/' + n);
    console.log('   ' + key.padEnd(8) + ' class of the level start: ' + ex.states + ' states, ' + n + ' unfilled, all match the census: ' + (agree === n));
  }
  // 3. invariance of the encoding
  console.log('3. encoding invariance');
  {
    const G = A.graphFromLevels(GAME, 'six'), L = GAME.levels.find(l => l.id === 'six-ring-trap'), col = Int8Array.from(L.start); col[G.hole] = -1;
    const e1 = A.encode(G, col);
    const perm = [2, 0, 3, 1], col2 = col.map(c => c < 0 ? -1 : perm[c]), e2 = A.encode(G, col2);
    const spec = GAME.graphs.six, link = spec.link.slice(2).concat(spec.link.slice(0, 2));
    const G3 = A.makeGraph({ n: spec.n, hole: spec.hole, link, edges: spec.edges }), e3 = A.encode(G3, col);
    const same = (a, b) => a.X.every((x, i) => x === b.X[i]) && a.rimOrd.every((x, i) => x === b.rimOrd[i]);
    ok(same(e1, e2), 'encoding changes when colours are renamed'); ok(same(e1, e3), 'encoding changes when the link list is rotated');
    const r = e1.rel, rim = Array.from(e1.rimOrd).map(u => r[col[u]]);
    ok(rim.join('') === '01023', 'relative rim pattern is ' + rim.join('') + ', expected 01023');
    console.log('   rename colours: same; rotate link list: same; relative rim pattern ' + rim.join(''));
  }
  // 4. gradient check
  console.log('4. gradient check (Float64, central differences)');
  {
    const G = A.graphFromLevels(GAME, 'trap17'), L = GAME.levels.find(l => l.id === 'tidy-trap'), col = Int8Array.from(L.start); col[G.hole] = -1;
    const net = A.netCreate({ H: 6, K: 2, H1: 7, Hp: 5 }, 3, Float64Array);
    for (let i = 0; i < net.size; i++) net.params[i] += 0.05 * Math.sin(i * 1.7);   // break ties at zero biases
    const enc = A.encode(G, col);
    const mask = new Uint8Array(G.n); mask[2] = mask[5] = 1;
    const encM = A.encode(G, col, { mask, dropChains: true });
    const R = A.rng(9), m = enc.moves.length;
    const cv = R(), cd = R(), cl = Array.from({ length: m }, () => R() - 0.5), cc = Array.from({ length: G.n * 4 }, () => R() - 0.5), ch = Array.from({ length: G.n * 8 }, () => R() - 0.5);
    const lossOf = () => {
      const o = A.netForward(net, enc, { policy: true });
      let L = cv * o.v + cd * o.d; for (let c = 0; c < m; c++) L += cl[c] * o.logits[c];
      const o2 = A.netForward(net, encM, { nodes: true });
      for (let i = 0; i < G.n * 4; i++) L += cc[i] * o2.colL[i]; for (let i = 0; i < G.n * 8; i++) L += ch[i] * o2.chP[i];
      return L;
    };
    const grad = new Float64Array(net.size);
    const o = A.netForward(net, enc, { policy: true }); A.netBackward(net, o, { v: cv, d: cd, logits: Float64Array.from(cl) }, grad);
    const o2 = A.netForward(net, encM, { nodes: true }); A.netBackward(net, o2, { colL: Float64Array.from(cc), chP: Float64Array.from(ch) }, grad);
    let worst = 0, checked = 0;
    for (let i = 0; i < net.size; i++) {
      const h = 1e-6, x = net.params[i];
      net.params[i] = x + h; const lp = lossOf(); net.params[i] = x - h; const lm = lossOf(); net.params[i] = x;
      const num = (lp - lm) / (2 * h), err = Math.abs(num - grad[i]) / Math.max(1e-4, Math.abs(num) + Math.abs(grad[i]));
      worst = Math.max(worst, err); checked++;
    }
    ok(worst < 1e-4, 'gradient check worst relative error ' + worst);
    console.log('   ' + checked + ' parameters, worst relative error ' + worst.toExponential(2));
  }
  // 5. generator
  console.log('5. random maps');
  {
    const R = A.rng(42); let maps = 0, cols = 0, labelled = 0, d2 = 0;
    for (const n of [12, 14, 16, 18, 20, 24, 30, 40]) for (let t = 0; t < 6; t++) {
      const G = A.generateGraph(n, R), why = A.checkTriangulation(G);
      ok(!why, 'generated n=' + n + ': ' + why); maps++;
      const col = A.randomBlocked(G, R); if (!col) { fail('no blocked colouring for n=' + n); continue; }
      cols++;
      for (const [u, w] of G.edges) if (u !== G.hole && w !== G.hole) ok(col[u] !== col[w], 'improper colouring');
      ok(A.rimCount(G, col) === 4, 'hole not blocked'); ok(G.deg[G.hole] === 5, 'hole degree');
      if (n <= 18) { const ex = A.exactDistances(G, col, 3000); if (ex) { labelled++; for (const d of ex.map.values()) if (d >= 2) d2++; } }
    }
    console.log('   ' + maps + ' triangulations valid, ' + cols + ' blocked colourings proper, ' + labelled + ' small classes labelled (' + d2 + ' states with d >= 2)');
  }
  // 6. model round trip
  console.log('6. model file');
  {
    const net = A.netCreate({ H: 8, K: 2 }, 5), J = JSON.parse(JSON.stringify(A.modelToJSON(net, { name: 't' }))), net2 = A.modelFromJSON(J);
    ok(net2.params.every((x, i) => x === net.params[i]), 'weights differ after a round trip');
    const nr = A.netCreate({ kind: 'rec', H: 8, T: 6 }, 5), nr2 = A.modelFromJSON(JSON.parse(JSON.stringify(A.modelToJSON(nr, { name: 'r' }))));
    ok(nr2.arch.kind === 'rec' && nr2.params.every((x, i) => x === nr.params[i]), 'rec weights differ after a round trip');
    console.log('   ' + net.size + ' params, JSON ' + JSON.stringify(J).length + ' bytes, round trip exact');
  }
}

// 9. round 2
if (!SPEED) {
  console.log('9. round 2: recurrent net, state spaces, mirrors, splices');
  {
    const G = A.graphFromLevels(GAME, 'trap17'), L = GAME.levels.find(l => l.id === 'tidy-trap'), col = Int8Array.from(L.start); col[G.hole] = -1;
    const net = A.netCreate({ kind: 'rec', H: 5, T: 4, H1: 6, Hq: 4 }, 3, Float64Array);
    for (let i = 0; i < net.size; i++) net.params[i] += 0.05 * Math.sin(i * 1.3);
    const enc = A.encode(G, col, { kind: 'rec', perm: [2, 0, 3, 1] }), R = A.rng(4), cv = R(), cl = Array.from({ length: G.n * 4 }, () => R() - 0.5);
    const lossOf = () => { const o = A.netForward(net, enc, { policy: true }); let L = cv * o.v; for (let k = 0; k < G.n * 4; k++) L += cl[k] * o.logits[k]; return L; };
    const grad = new Float64Array(net.size), o = A.netForward(net, enc, { policy: true }); A.netBackward(net, o, { v: cv, logits: Float64Array.from(cl) }, grad);
    let worst = 0; for (let i = 0; i < net.size; i++) { const x = net.params[i], h = 1e-6; net.params[i] = x + h; const a = lossOf(); net.params[i] = x - h; const b = lossOf(); net.params[i] = x; const num = (a - b) / 2 / h; worst = Math.max(worst, Math.abs(num - grad[i]) / Math.max(1e-4, Math.abs(num) + Math.abs(grad[i]))); }
    ok(worst < 1e-4, 'rec gradient check ' + worst);
    // inputs are raw: permuting the colours of the state and the input frame together leaves the input unchanged
    const c2 = col.map(c => c < 0 ? -1 : [1, 3, 0, 2][c]), e2 = A.encode(G, c2, { kind: 'rec', perm: [2, 0, 3, 1].map((_, k) => [2, 0, 3, 1][[1, 3, 0, 2].indexOf(k)]) });
    ok(e2.X.every((x, i) => x === enc.X[i]), 'rec input not equivariant under colour renaming');
    // the action table: every legal click is a non-web chain through the clicked region
    const tab = A.actionTable(enc); let clicks = 0;
    for (let u = 0; u < G.n; u++) for (let k = 0; k < 4; k++) { const mi = tab[u * 4 + k]; if (mi < 0) continue; clicks++; ok(Array.from(enc.moves[mi].K.vs).includes(u), 'click not on its chain'); }
    console.log('   rec gradient check worst ' + worst.toExponential(2) + ' (' + net.size + ' params); ' + clicks + ' legal clicks map to ' + enc.moves.length + ' swaps');
  }
  {
    let maps = 0;
    for (const m of A.levelMaps(GAME)) {
      const G = A.graphFromLevels(GAME, m.key), S = A.stateSpace(G), Gm = A.mirrorGraph(G);
      for (const i of S.unfilled) ok(G.dist[S.keys[i]] === S.dist[i], m.key + ' state space differs from the census');
      ok(S.unfilled.length === Object.keys(G.dist).length, m.key + ' state count');
      ok(!A.checkTriangulation(Gm), 'mirror of ' + m.key + ' invalid');
      const Sm = A.stateSpace(Gm); ok(Sm.unfilled.every(i => Sm.dist[i] === S.dist[S.idx.get(Sm.keys[i])]), 'mirror of ' + m.key + ' changes labels');
      maps++;
    }
    console.log('   ' + maps + ' maps: whole state spaces equal the census; mirrors valid with identical labels');
  }
  {
    const R = A.rng(11), ops = {}; let valid = 0, tot = 0, deep = 0;
    for (const key of ['r178', 'r1', 'p31r12', 'p22']) for (let t = 0; t < 12; t++) {
      const tri = new A.Tri(A.graphFromLevels(GAME, key)); let op = A.spliceRandom(tri, R);
      if (t % 3 === 0) { for (let v = 0; v < tri.nxt.length; v++) if (tri.remove(v, t & 1)) { op = 'remove'; break; } }
      ops[op] = (ops[op] || 0) + 1; const G = tri.toGraph(key + '*'), why = A.checkTriangulation(G); tot++;
      ok(!why, 'splice ' + op + ' of ' + key + ': ' + why); if (why) continue; valid++;
      const S = A.stateSpace(G, 100000); if (!S) continue;
      for (const i of S.unfilled.slice(0, 50)) { const c = S.cols[i]; for (const [u, w] of G.edges) if (u !== G.hole && w !== G.hole && c[u] === c[w]) { fail('improper colouring after splice'); break; } }
      if (A.hardness(S.hist).maxD >= 3) deep++;
    }
    console.log('   splices: ' + valid + '/' + tot + ' valid triangulations with a degree-5 hole ' + JSON.stringify(ops) + ', ' + deep + ' still have states 3+ deep');
  }
}

// 10. spectral inputs
if (!SPEED) {
  console.log('10. optional spectral inputs');
  {
    const G = A.graphFromLevels(GAME, 'r178'), n = G.n, { w, V } = A.laplacianEigen(G);
    let res = 0;
    for (let k = 0; k < n; k++) for (let u = 0; u < n; u++) {
      let lv = (G.adjStart[u + 1] - G.adjStart[u]) * V[u * n + k];
      for (let e = G.adjStart[u]; e < G.adjStart[u + 1]; e++) lv -= V[G.adjList[e] * n + k];
      res = Math.max(res, Math.abs(lv - w[k] * V[u * n + k]));
    }
    ok(res < 1e-8 && Math.abs(w[0]) < 1e-9 && w[1] > 1e-6, 'Laplacian eigenpairs: residual ' + res + ', l0 ' + w[0] + ', l2 ' + w[1]);
    // renumber the regions: HKS/WKS of region u must equal those of its new number
    const R = A.rng(9), pi = A.shuffle(Array.from({ length: n }, (_, i) => i), R);
    const Gp = A.makeGraph({ n, hole: pi[G.hole], link: Array.from(G.rim, u => pi[u]), edges: G.edges.map(([a, b]) => [pi[a], pi[b]]) });
    const S1 = A.spectralFeatures(G, 'hks,wks'), S2 = A.spectralFeatures(Gp, 'hks,wks'), D = A.spectralDim('hks,wks');
    let dev = 0; for (let u = 0; u < n; u++) for (let j = 0; j < D; j++) dev = Math.max(dev, Math.abs(S1[u * D + j] - S2[pi[u] * D + j]));
    ok(dev < 1e-4, 'HKS/WKS change when regions are renumbered: ' + dev);
    let mean = 0; for (let u = 0; u < n; u++) mean += S1[u * D] / n;
    ok(Math.abs(mean - 1) < 1e-4, 'HKS not normalised to mean 1: ' + mean);
    const col = A.decodeCanon(G, Object.keys(G.dist)[5]);
    // off is the old net: same parameter count and identical encoding
    ok(A.netCreate({ H: 8, K: 2 }, 1).size === A.netCreate({ H: 8, K: 2, spectral: '' }, 1).size, 'spectral off changes the GNN');
    const e0 = A.encode(G, col), e1 = A.encode(G, col, { spectral: 'hks,wks,pe' });
    ok(e1.F === A.F + 20 && e1.X.length === n * (A.F + 20), 'spectral encoding width');
    let same = true; for (let u = 0; u < n; u++) for (let j = 0; j < A.F; j++) if (e0.X[u * A.F + j] !== e1.X[u * e1.F + j]) same = false;
    ok(same, 'spectral on changes the usual features');
    let threw = false; try { A.netForward(A.netCreate({ H: 8, K: 2, spectral: 'hks' }, 1), e0); } catch (e) { threw = true; }
    ok(threw, 'a spectral net accepted an encoding without spectral features');
    // gradient checks with spectral on
    const gc = (net, enc, need, gsig) => {
      for (let i = 0; i < net.size; i++) net.params[i] += 0.05 * Math.sin(i * 1.3);
      const Rr = A.rng(4), cv = Rr(), m = enc.moves.length, cl = Array.from({ length: Math.max(m, n * 4) }, () => Rr() - 0.5);
      const lossOf = () => { const o = A.netForward(net, enc, need); let L = cv * o.v; const k = net.arch.kind === 'rec' ? n * 4 : m; for (let c = 0; c < k; c++) L += cl[c] * o.logits[c]; return L; };
      const grad = new Float64Array(net.size), o = A.netForward(net, enc, need);
      A.netBackward(net, o, { v: cv, logits: Float64Array.from(cl.slice(0, net.arch.kind === 'rec' ? n * 4 : m)) }, grad);
      let worst = 0;
      for (let i = 0; i < net.size; i++) { const x = net.params[i], h = 1e-6; net.params[i] = x + h; const a = lossOf(); net.params[i] = x - h; const b = lossOf(); net.params[i] = x; const num = (a - b) / 2 / h; worst = Math.max(worst, Math.abs(num - grad[i]) / Math.max(1e-4, Math.abs(num) + Math.abs(grad[i]))); }
      return worst;
    };
    const ng = A.netCreate({ H: 5, K: 2, H1: 6, Hp: 4, spectral: 'hks,wks,pe' }, 3, Float64Array);
    const wg = gc(ng, A.encode(G, col, ng.arch), { policy: true });
    const nr = A.netCreate({ kind: 'rec', H: 5, T: 3, H1: 6, Hq: 4, spectral: 'hks,wks' }, 3, Float64Array);
    const wr = gc(nr, A.encode(G, col, { kind: 'rec', spectral: nr.arch.spectral }), { policy: true });
    ok(wg < 1e-4 && wr < 1e-4, 'spectral gradient checks: gnn ' + wg + ', rec ' + wr);
    // model files keep the setting; old files (no spectral key) load as off
    const J = JSON.parse(JSON.stringify(A.modelToJSON(ng, { name: 's' }))), n2 = A.modelFromJSON(J);
    ok(n2.arch.spectral === 'hks,wks,pe' && n2.size === ng.size, 'spectral setting lost in the model file');
    const Jo = JSON.parse(JSON.stringify(A.modelToJSON(A.netCreate({ H: 8, K: 2 }, 5), { name: 'o' })));
    ok(!('spectral' in Jo.arch) && A.modelFromJSON(Jo).arch.spectral === '', 'spectral-off model file changed');
    console.log('   eigen residual ' + res.toExponential(1) + ', renumbering deviation ' + dev.toExponential(1) + ', gradient checks gnn ' + wg.toExponential(2) + ' rec ' + wr.toExponential(2) + ', model files keep the setting');
  }
}

// 8. speeds
console.log('8. speeds (this machine, one thread)');
for (const [H, K] of [[8, 2], [16, 3], [32, 3]]) {
  const T = new A.Trainer({ arch: { H, K }, rl: { batch: 32 }, seed: 1 }), env = A.makeEnv(GAME, 'r178');
  const actor = { env, col: null, learn: true };
  let t0 = A.now(), steps = 0; while (A.now() - t0 < 1500) { T.actorStep(actor); steps++; }
  const sps = steps / ((A.now() - t0) / 1000);
  const G = env.G, col = A.decodeCanon(G, env.census[0]), enc = A.encode(G, col), grad = new Float32Array(T.net.size);
  t0 = A.now(); let fb = 0; while (A.now() - t0 < 1500) { const o = A.netForward(T.net, enc); A.netBackward(T.net, o, { v: 1 }, grad); fb++; }
  const fbs = fb / ((A.now() - t0) / 1000);
  t0 = A.now(); let fw = 0; while (A.now() - t0 < 1000) { A.netForward(T.net, enc); fw++; }
  console.log(`   H=${H} K=${K} params ${T.net.size}: env steps/s (afterstate scoring, n=34) ${sps.toFixed(0)}, forward/s ${(fw).toFixed(0)}, forward+backward/s ${fbs.toFixed(0)}`);
}
for (const [H, T] of [[16, 8], [32, 16]]) {
  const G = A.graphFromLevels(GAME, 'r178'), net = A.netCreate({ kind: 'rec', H, T }, 1), col = A.decodeCanon(G, Object.keys(G.dist)[5]);
  const enc = A.encode(G, col, { kind: 'rec' }), grad = new Float32Array(net.size), lg = new Float32Array(G.n * 4).fill(0.01);
  let t0 = A.now(), k = 0; while (A.now() - t0 < 1500) { const o = A.netForward(net, enc, { policy: true }); A.netBackward(net, o, { v: 1, logits: lg }, grad); k++; }
  const fb = k / ((A.now() - t0) / 1000); t0 = A.now(); let e = 0; while (A.now() - t0 < 500) { A.encode(G, col, { kind: 'rec' }); e++; }
  console.log(`   tabula-rasa rec H=${H} T=${T} params ${net.size}: forward+backward/s ${fb.toFixed(0)}, encode (with chain extraction for the move list)/s ${(e * 2).toFixed(0)}`);
}
{
  const R = A.rng(1); let t0 = A.now(), g = 0, lab = 0; const stats = {};
  for (const n of [14, 18, 24, 34]) {
    t0 = A.now(); g = 0; lab = 0;
    while (A.now() - t0 < 1500) { const G = A.generateGraph(n, R), c = A.randomBlocked(G, R); g++; if (n <= 18 && c && A.exactDistances(G, c, 3000)) lab++; }
    const s = (A.now() - t0) / 1000;
    console.log(`   generator n=${n}: ${(g / s).toFixed(0)} maps+colourings/s` + (n <= 18 ? ` (with exact BFS labels: ${(lab / s).toFixed(0)} labelled/s)` : ''));
  }
}

// 7. RL on tidy-trap
if (!QUICK && !SPEED) {
  console.log('7. short RL run on tidy-trap');
  const TB = new A.Testbench(GAME, ['trap17'], { maxHard: 100, maxEasy: 20 });
  const T = new A.Trainer({ arch: { H: 16, K: 3 }, rl: { lr: 2e-3, epsDecay: 400, episodeCap: 12, nstep: 2, startMix: 0.2 }, seed: 3 });
  const before = TB.run(A.netAgent(T.net, 'value', T.rl.gamma)).trap17;
  const env = A.makeEnv(GAME, 'trap17'), actors = Array.from({ length: 8 }, () => ({ env, col: null, learn: true }));
  const t0 = A.now();
  while (T.episodes < 1500) { for (const a of actors) { T.actorStep(a); if (T.steps % T.rl.updateEvery === 0) T.learn(); } }
  const after = TB.run(A.netAgent(T.net, 'value', T.rl.gamma)).trap17;
  console.log(`   agreement on d>=2 states: ${before.agree2.toFixed(2)} -> ${after.agree2.toFixed(2)} (all states ${before.agreeAll.toFixed(2)} -> ${after.agreeAll.toFixed(2)}), success ${after.success.toFixed(2)}, ${T.episodes} episodes, ${((A.now() - t0) / 1000).toFixed(1)}s`);
  ok(after.agree2 > before.agree2 + 0.1 || after.agree2 >= 0.9, 'RL did not improve agreement on tidy-trap');
}
console.log(bad ? bad + ' failures' : 'all good');
process.exit(bad ? 1 : 0);
