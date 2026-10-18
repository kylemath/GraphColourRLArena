// node tools/arena_probe.js [flags] : interpretability probe of the committed Maze Arena models (INTERPRETABILITY-1).
//
// Read-only with respect to the models: loads lab/arena/models/*.json (or docs/arena/models/ when lab/ has been moved),
// never trains, never writes into the models directory. Writes everything to --out (default: $TMPDIR/arena-probe).
//
// What it does:
//   1. State sets. CENSUS: every unfilled state of every 4-colour map in levels.js (whole state spaces by enumeration,
//      checked against the census distances). SPLICE (optional, --splice N attempts): 1-3 random splices of the census
//      TEST maps (c30 p32r11 p34r28 c60 six trap17 r210; never used by any model's training), kept when the variant has
//      states 3+ swaps deep; per kept map: every state with d >= 3, up to --splice-d2 states at d = 2 and --splice-d1 at
//      d = 1 (a declared sample, seeded).
//   2. Ground truth per state (exact): kappa = fewest pure Kempe swaps to a fill at the fixed hole (BFS over the whole
//      state space; the arena's distance d). ell = fewest mixed moves (Kempe swaps + singleton slides that move the
//      hole), computed by bounded BFS for kappa >= 3 states (kappa <= 2 gives ell = kappa by the compiled short-fill
//      theorem M3; kappa = 3 is checked to have no mixed fill within 2 moves). Theory features at the pentagonal hole
//      (definitions of QuarterFloor.lean / QuarterPi.lean / QuarterSigmaGroups.lean in ~/GraphColour): repeat index j,
//      Lock1, Lock2, DL, the pi map (R+3 / phiB^-1 / phiA / tau), lambda, pi-orbit length / filled count / winding
//      w = (L - 4F)/5, all-DL orbit (Gamma-cycle), DL run length along pi, pi-steps to a filled state, DD, the sigma
//      image (swap of the {alpha, mu}-component of x_{j+1}) and whether it is DL / lockless, Kempe class size and filled
//      share, rim-chain lengths, the rim link pattern (L1, L2, PA, PB: the four non-trivial rim pairs a two-colour chain
//      can join; L1/L2 join different colours = odd ring with the two spokes, PA/PB join the repeated colour = even
//      ring), move counts, the hole's degree word read from the repeat index.
//   3. Per model: value V(s) for every state, the greedy afterstate agent's choice (score -1 + gamma V(after), V = 0
//      at a fill, exactly as ARENA.netAgent), its correctness against the exact afterstate distance, the policy head's
//      choice, and a greedy rollout from every state (steps vs kappa). Hidden activations (per layer: mean pool, max
//      pool, hole, the 5 rim embeddings in relative rim order; plus the MLP layer z) for --acts models and for a
//      random-init network of each of their architectures (control). Node-level activations and node labels on a
//      stratified sample of states (--node-sample).
//
// Flags: --out DIR  --models a,b (default: every model in index.json)  --acts a,b  --splice 400  --splice-maps 40
//        --splice-seed 1  --splice-d2 150 --splice-d1 60  --node-sample 1500  --ell-cap 250000  --ell-max-states 400
//        --quick (census only, two models)  --no-input (skip the input baseline)  --models none --acts none (ground truth only)
'use strict';
const fs = require('fs'), path = require('path'), os = require('os');
const root = path.join(__dirname, '..');
global.window = global;
require(path.join(root, 'docs/levels.js'));
const ARENA_DIR = fs.existsSync(path.join(root, 'docs/arena/engine.js')) ? path.join(root, 'docs/arena') : path.join(root, 'docs/arena');
const A = require(path.join(ARENA_DIR, 'engine.js'));
const GAME = window.GAME;
const MODELS = path.join(ARENA_DIR, 'models');

const argv = process.argv.slice(2), flags = {};
for (let i = 0; i < argv.length; i++) { const a = argv[i]; if (!a.startsWith('--')) continue; const k = a.slice(2), v = argv[i + 1] && !argv[i + 1].startsWith('--') ? argv[++i] : true; flags[k] = v; }
const num = (k, d) => flags[k] === undefined ? d : +flags[k];
const QUICK = !!flags.quick;
const OUT = flags.out ? path.resolve(String(flags.out)) : path.join(os.tmpdir(), 'arena-probe');
fs.mkdirSync(OUT, { recursive: true });
const t0 = Date.now(), log = (...a) => console.log(`[${((Date.now() - t0) / 1000).toFixed(1)}s]`, ...a);

const TRAIN = ['w9', 'w13', 'w18', 'ico', 'p22', 'r1', 'r178', 'p32r111', 'p31r12'];
const TEST = ['c30', 'p32r11', 'p34r28', 'c60', 'six', 'trap17', 'r210'];
const index = JSON.parse(fs.readFileSync(path.join(MODELS, 'index.json'), 'utf8'));
const allModels = index.models.map(m => m.name);
const listFlag = k => flags[k] === 'none' ? [] : String(flags[k]).split(',');
const modelNames = QUICK ? ['priors-gnn-h16-k3-pool', 'nano-rl'] : flags.models ? listFlag('models') : allModels;
const actNames = QUICK ? ['priors-gnn-h16-k3-pool'] : flags.acts ? listFlag('acts') : ['priors-gnn-h16-k3-pool', 'mid-rl', 'nano-rl', 'colours-only-pretrained', 'tr-rec-h32-t16'];

// ------------------------------------------------------------------ chain helpers (hole given explicitly)
function comp(G, col, hole, a, b, seed, buf) {   // the {a,b}-component of seed in G - hole, as an array (empty if seed not a/b)
  if (seed === hole || (col[seed] !== a && col[seed] !== b)) return [];
  const seen = buf || new Uint8Array(G.n); seen.fill(0); const out = [seed]; seen[seed] = 1;
  for (let i = 0; i < out.length; i++) { const u = out[i]; for (let e = G.adjStart[u]; e < G.adjStart[u + 1]; e++) { const w = G.adjList[e]; if (w !== hole && !seen[w] && (col[w] === a || col[w] === b)) { seen[w] = 1; out.push(w); } } }
  return out;
}
function swapSet(col, vs, a, b) { const d = col.slice(); for (const u of vs) d[u] = d[u] === a ? b : a; return d; }
const fourth = (a, b, c) => 6 - a - b - c;   // colours 0..3
// rim reading: { unfilled, j } or { filled, i }; x = G.rim
function rimRead(G, col) {
  const r = [0, 1, 2, 3, 4].map(k => col[G.rim[k]]);
  if (A.rimCount(G, col) === 4) { for (let j = 0; j < 5; j++) if (r[j] === r[(j + 2) % 5]) return { unfilled: true, j, r }; throw new Error('no repeat index'); }
  for (let i = 0; i < 5; i++) { let uniq = true; for (let k = 0; k < 5; k++) if (k !== i && r[k] === r[i]) uniq = false; if (uniq) return { unfilled: false, i, r }; }
  throw new Error('no singleton');
}
const X = (G, k) => G.rim[((k % 5) + 5) % 5];
function locks(G, col, j, buf) {
  const m = X(G, j + 1), a = X(G, j + 3), b = X(G, j + 4), mu = col[m], cA = col[a], cB = col[b];
  const K1 = comp(G, col, G.hole, mu, cA, m, buf), l1 = K1.indexOf(a) >= 0;
  const K2 = comp(G, col, G.hole, mu, cB, m, buf), l2 = K2.indexOf(b) >= 0;
  return { l1, l2, K1, K2 };
}
// the pi map of QuarterPi.lean: returns { col, lam, kind }
function piMove(G, col, buf) {
  const R = rimRead(G, col);
  if (R.unfilled) {
    const j = R.j, al = col[X(G, j)], mu = col[X(G, j + 1)], cA = col[X(G, j + 3)], cB = col[X(G, j + 4)];
    const K2 = comp(G, col, G.hole, mu, cB, X(G, j + 1), buf);
    if (K2.indexOf(X(G, j + 4)) >= 0) return { col: swapSet(col, comp(G, col, G.hole, al, cA, X(G, j + 2), buf), al, cA), lam: 1, kind: 'R3' };
    return { col: swapSet(col, comp(G, col, G.hole, mu, cB, X(G, j + 4), buf), mu, cB), lam: -1, kind: 'phiBinv' };
  }
  const i = R.i, W = col[X(G, i)], Xc = col[X(G, i + 1)], Y = col[X(G, i + 2)], Z = fourth(W, Xc, Y);
  const KY = comp(G, col, G.hole, Y, Z, X(G, i + 2), buf);
  if (KY.indexOf(X(G, i + 4)) < 0) return { col: swapSet(col, KY, Y, Z), lam: -1, kind: 'phiA' };
  return { col: swapSet(col, comp(G, col, G.hole, W, Xc, X(G, i + 3), buf), W, Xc), lam: -3, kind: 'tau' };
}
function sigmaImage(G, col, j, buf) { const al = col[X(G, j)], mu = col[X(G, j + 1)]; return swapSet(col, comp(G, col, G.hole, al, mu, X(G, j + 1), buf), al, mu); }

// ------------------------------------------------------------------ mixed moves (slides) for ell
function mixedEll(G, col0, maxD, cap) {
  // state: hole h + colouring with col[h] = -1. Moves: non-web Kempe swaps of G - h, slides h -> u (c(u) unique on N(h)).
  const n = G.n, buf = new Uint8Array(n);
  const keyOf = (h, c) => { const mp = [-1, -1, -1, -1]; let s = h + ':', k = 0; for (let u = 0; u < n; u++) { if (u === h) continue; const x = c[u]; if (mp[x] < 0) mp[x] = k++; s += mp[x]; } return s; };
  const target = (h, c) => { let m = 0; for (let e = G.adjStart[h]; e < G.adjStart[h + 1]; e++) m |= 1 << c[G.adjList[e]]; return (m & 1) + (m >> 1 & 1) + (m >> 2 & 1) + (m >> 3 & 1) <= 3; };
  const seen = new Set(); let front = [{ h: G.hole, c: col0 }]; seen.add(keyOf(G.hole, col0)); let expanded = 0;
  for (let depth = 1; depth <= maxD; depth++) {
    const next = [];
    for (const s of front) {
      if (++expanded > cap) return { ell: null, capped: true, depth };
      const h = s.h, c = s.c;
      // Kempe swaps (non-web) in G - h
      for (let p = 0; p < 6; p++) {
        const a = A.PAIRS[p][0], b = A.PAIRS[p][1], mark = new Int32Array(n).fill(-1), comps = [];
        for (let u = 0; u < n; u++) { if (u === h || mark[u] >= 0 || (c[u] !== a && c[u] !== b)) continue; const K = comp(G, c, h, a, b, u, buf); K.forEach(v => { mark[v] = comps.length; }); comps.push(K); }
        if (comps.length < 2) continue;
        for (const K of comps) { const d = swapSet(c, K, a, b); if (target(h, d)) return { ell: depth }; const k = keyOf(h, d); if (!seen.has(k)) { seen.add(k); next.push({ h, c: d }); } }
      }
      // slides
      const cnt = [0, 0, 0, 0]; for (let e = G.adjStart[h]; e < G.adjStart[h + 1]; e++) cnt[c[G.adjList[e]]]++;
      for (let e = G.adjStart[h]; e < G.adjStart[h + 1]; e++) {
        const u = G.adjList[e]; if (cnt[c[u]] !== 1) continue;
        const d = c.slice(); d[h] = c[u]; d[u] = -1; if (target(u, d)) return { ell: depth };
        const k = keyOf(u, d); if (!seen.has(k)) { seen.add(k); next.push({ h: u, c: d }); }
      }
    }
    front = next;
  }
  return { ell: null, capped: false, depth: maxD };   // no mixed fill within maxD
}

// ------------------------------------------------------------------ per-map analysis
function analyseMap(G, opt) {
  const S = A.stateSpace(G, 400000); if (!S) throw new Error('state space too big: ' + G.key);
  const N = S.N, buf = new Uint8Array(G.n);
  // full Kempe adjacency (filled states too) for classes
  const parent = Int32Array.from({ length: N }, (_, i) => i), find = x => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
  const movesAll = new Array(N);
  for (let i = 0; i < N; i++) {
    const mv = A.movesOf(G, S.cols[i]); movesAll[i] = mv;
    for (const K of mv) { const j = S.idx.get(A.canon(G, A.flip(S.cols[i], K))); const a = find(i), b = find(j); if (a !== b) parent[a] = b; }
  }
  const cls = new Int32Array(N), clsN = new Map(), clsF = new Map();
  for (let i = 0; i < N; i++) { cls[i] = find(i); clsN.set(cls[i], (clsN.get(cls[i]) || 0) + 1); if (S.won[i]) clsF.set(cls[i], (clsF.get(cls[i]) || 0) + 1); }
  // pi map on every state
  const pi = new Int32Array(N), lam = new Int8Array(N), indeg = new Int32Array(N); let piErr = 0;
  for (let i = 0; i < N; i++) {
    const P = piMove(G, S.cols[i], buf), k = S.idx.get(A.canon(G, P.col));
    if (k === undefined) throw new Error('pi image not a state'); pi[i] = k; lam[i] = P.lam; indeg[k]++;
    if (cls[k] !== cls[i]) piErr++;
  }
  for (let i = 0; i < N; i++) if (indeg[i] !== 1) piErr++;
  // per-state lock / DL
  const unf = S.unfilled, isDL = new Uint8Array(N), jOf = new Int8Array(N).fill(-1), L1 = new Uint8Array(N), L2 = new Uint8Array(N);
  for (const i of unf) { const R = rimRead(G, S.cols[i]); jOf[i] = R.j; const L = locks(G, S.cols[i], R.j, buf); L1[i] = L.l1; L2[i] = L.l2; isDL[i] = L.l1 && L.l2 ? 1 : 0; }
  // orbits
  const orb = new Int32Array(N).fill(-1), orbits = [];
  for (let i = 0; i < N; i++) {
    if (orb[i] >= 0) continue; const cyc = []; let k = i; while (orb[k] < 0) { orb[k] = orbits.length; cyc.push(k); k = pi[k]; }
    let F = 0, dl = 0, sl = 0; for (const s of cyc) { F += S.won[s]; dl += isDL[s]; sl += lam[s]; }
    const L = cyc.length; orbits.push({ L, F, w: (L - 4 * F) / 5, allDL: dl === L, sumLam: sl, cyc });
    if ((L - 4 * F) % 5 !== 0 || sl !== L - 4 * F) piErr++;   // Theorem W bookkeeping (sum lambda = L - 4F, 5 | L - 4F)
  }
  // position along the orbit: steps to the next filled state, DL run length
  const toFill = new Int32Array(N).fill(-1), dlRun = new Int32Array(N);
  for (const O of orbits) {
    const c = O.cyc, L = c.length;
    if (O.F > 0) { // walk backwards twice around
      let last = -1; for (let t = 2 * L - 1; t >= 0; t--) { const s = c[t % L]; if (S.won[s]) last = t; if (t < L) toFill[s] = last - t; }
    }
    if (O.allDL) { for (const s of c) dlRun[s] = L; continue; }
    // runs of consecutive DL states (cyclic)
    let start = 0; while (isDL[c[start]]) start++;
    for (let t = 0; t < L; t++) {
      const s = c[(start + t) % L]; if (!isDL[s]) continue;
      if (t > 0 && isDL[c[(start + t - 1) % L]]) continue;
      let len = 0; while (len < L && isDL[c[(start + t + len) % L]]) len++;
      for (let q = 0; q < len; q++) dlRun[c[(start + t + q) % L]] = len;
    }
  }
  // position inside a DL run: pi-steps forward / backward to the first non-DL state (-1 on an all-DL orbit; 0 if not DL)
  const runFwd = new Int32Array(N), runBwd = new Int32Array(N);
  for (const O of orbits) {
    const c = O.cyc, L = c.length;
    if (O.allDL) { for (const s of c) { runFwd[s] = -1; runBwd[s] = -1; } continue; }
    for (let t = 0; t < L; t++) {
      const s = c[t]; if (!isDL[s]) continue;
      let f = 1; while (isDL[c[(t + f) % L]]) f++; let b = 1; while (isDL[c[(t - b + L) % L]]) b++;
      runFwd[s] = f; runBwd[s] = b;
    }
  }
  return { S, cls, clsN, clsF, pi, lam, isDL, jOf, L1, L2, orb, orbits, toFill, dlRun, runFwd, runBwd, piErr, movesAll };
}

// per-state feature row + per-move descriptors
function stateRow(G, M, i, setName, split, mapKey, ellInfo) {
  const S = M.S, col = S.cols[i], buf = new Uint8Array(G.n), d = S.dist[i], j = M.jOf[i];
  const x = k => X(G, j + k), al = col[x(0)], mu = col[x(1)], cA = col[x(3)], cB = col[x(4)];
  const cmp = (a, b, s) => comp(G, col, G.hole, a, b, s, buf);
  const K1 = cmp(mu, cA, x(1)), K2 = cmp(mu, cB, x(1)), KaA = cmp(al, cA, x(0)), KaA2 = cmp(al, cA, x(2)), KaB = cmp(al, cB, x(0)), KaB2 = cmp(al, cB, x(2)), Kam = cmp(al, mu, x(1)), KAB = cmp(cA, cB, x(3));
  const PA = KaA.indexOf(x(2)) >= 0 ? 1 : 0, PB = KaB.indexOf(x(2)) >= 0 ? 1 : 0;
  const O = M.orbits[M.orb[i]];
  // sigma image
  let sigDL = null, sigLockless = null, sigD = null, sigSame = null;
  if (M.isDL[i]) {
    const sc = sigmaImage(G, col, j, buf), k = S.idx.get(A.canon(G, sc));
    sigD = S.won[k] ? 0 : S.dist[k]; sigSame = M.orb[k] === M.orb[i] ? 1 : 0;
    if (!S.won[k]) { const R = rimRead(G, sc); const L = locks(G, sc, R.j, buf); sigDL = L.l1 && L.l2 ? 1 : 0; sigLockless = !L.l1 && !L.l2 ? 1 : 0; } else { sigDL = 0; sigLockless = 0; }
  }
  const piNext = M.pi[i], dd = M.isDL[i] && !S.won[piNext] && M.isDL[piNext] ? 1 : 0;
  // moves
  const rc = A.relClasses(G, col), rel = rc.rel, ref = rc.ref;
  const mv = M.movesAll[i], movesDesc = [];
  let nFill = 0, nOpt = 0, nRim = 0, maxSize = 0, sumSize = 0;
  const piCol = A.canon(G, piMove(G, col, buf).col), sigCanon = M.isDL[i] ? A.canon(G, sigmaImage(G, col, j, buf)) : null;
  for (const K of mv) {
    const after = A.flip(col, K), ck = A.canon(G, after), k = S.idx.get(ck), ad = S.won[k] ? 0 : S.dist[k];
    let rm = 0; for (let q = 0; q < 5; q++) if (K.rimMask >> ((ref + q) % 5) & 1) rm |= 1 << q;
    const pa = Math.min(rel[K.a], rel[K.b]), pb = Math.max(rel[K.a], rel[K.b]), pc = pa === 0 ? pb - 1 : pa === 1 ? pb + 1 : 5;
    const isPi = ck === piCol ? 1 : 0, isSig = sigCanon && ck === sigCanon ? 1 : 0;
    // does the chain meet a lock chain (shares a vertex with K1 or K2 when locked)?
    let meetsLock = 0; if (M.isDL[i]) { const set = new Set(K.vs); if (K1.some(u => set.has(u)) || K2.some(u => set.has(u))) meetsLock = 1; }
    // repeat-index shift of the afterstate relative to j (0..4), or -1 when the swap fills the hole
    const jShift = S.won[k] ? -1 : (rimRead(G, after).j - j + 5) % 5;
    movesDesc.push([ad, pc, K.vs.length, rm, isPi, isSig, meetsLock, k, jShift]);
    if (ad === 0) nFill++; if (ad === d - 1) nOpt++; if (rm) nRim++; maxSize = Math.max(maxSize, K.vs.length); sumSize += K.vs.length;
  }
  const chains = A.chainsOf(G, col), webs = chains.filter(K => K.web).length;
  const degWord = [0, 1, 2, 3, 4].map(k => G.deg[x(k)]).join('');
  const sig = [];   // rim link signature: for each pair, partition of the rim positions in those colours (relative)
  return {
    row: {
      set: setName, split, map: mapKey, n: G.n, key: S.keys[i], d, ell: ellInfo ? ellInfo.ell : null, ellNote: ellInfo ? ellInfo.note : null,
      j, degWord, lock1: M.L1[i], lock2: M.L2[i], DL: M.isDL[i], PA, PB, oddLinks: M.L1[i] + M.L2[i] + PA + PB, rimPattern: '' + M.L1[i] + M.L2[i] + PA + PB,
      lam: M.lam[i], runFwd: M.runFwd[i], runBwd: M.runBwd[i], runBound: M.runFwd[i] > 0 ? Math.min(M.runFwd[i], M.runBwd[i]) + 1 : null, piL: O.L, piF: O.F, piW: O.w, piAllDL: O.allDL ? 1 : 0, dlRun: M.dlRun[i], piToFill: M.toFill[i], DD: dd, piNextD: S.won[piNext] ? 0 : S.dist[piNext],
      sigDL, sigLockless, sigD, sigSame,
      classN: M.clsN.get(M.cls[i]), classF: M.clsF.get(M.cls[i]) || 0,
      lenL1: K1.length, lenL2: K2.length, lenAlA: KaA.length, lenAlA2: KaA2.length, lenAlB: KaB.length, lenAlB2: KaB2.length, lenSigma: Kam.length, lenAB: KAB.length,
      nMoves: mv.length, nWebs: webs, nRimMoves: nRim, nFill, nOpt, maxChain: maxSize, meanChain: mv.length ? sumSize / mv.length : 0,
      piMoveOpt: movesDesc.some(m => m[4] && m[0] === d - 1) ? 1 : 0, sigMoveOpt: movesDesc.some(m => m[5] && m[0] === d - 1) ? 1 : 0,
      optRimFree: movesDesc.some(m => m[0] === d - 1 && !m[3]) ? 1 : 0, optAllRimFree: nOpt > 0 && movesDesc.every(m => m[0] !== d - 1 || !m[3]) ? 1 : 0,
    },
    moves: movesDesc,
  };
}

// ------------------------------------------------------------------ networks
function loadModel(name) { const M = JSON.parse(fs.readFileSync(path.join(MODELS, name + '.json'), 'utf8')); const net = A.modelFromJSON(M); return { net, gamma: M.rl && M.rl.gamma != null ? M.rl.gamma : 0.97, meta: M }; }
function layerList(net) { const a = net.arch; if (a.kind === 'rec') { const T = a.T; return [...new Set([0, Math.round(T / 4), Math.round(T / 2), Math.round(3 * T / 4), T])]; } return Array.from({ length: a.K + 1 }, (_, k) => k); }
// forward on one state; returns { v, dh, logits(copy), pooled (Float32Array) , nodes (per layer n*H) }
function forward(net, G, col, wantActs, wantNodes) {
  const a = net.arch, enc = A.encode(G, col, a.kind === 'rec' ? { kind: 'rec', spectral: a.spectral } : a);
  const out = A.netForward(net, enc, { policy: true });
  const res = { v: out.v, dh: out.d != null ? out.d : null, enc };
  // policy scores per move (movesOf order)
  if (a.kind === 'rec') { const tab = A.actionTable(enc), sc = new Float32Array(enc.moves.length).fill(-Infinity); for (let k = 0; k < tab.length; k++) { const mi = tab[k]; if (mi >= 0 && out.logits[k] > sc[mi]) sc[mi] = out.logits[k]; } res.pol = sc; }
  else res.pol = Float32Array.from(out.logits.subarray(0, enc.moves.length));
  if (wantActs || wantNodes) {
    const H = a.H, n = G.n, Ls = layerList(net), w = a.kind === 'rec' ? net.wsr : net.ws, ref = A.relClasses(G, col).ref;
    const rimOrd = [0, 1, 2, 3, 4].map(k => G.rim[(ref + k) % 5]);
    if (wantActs) {
      const D = Ls.length * 8 * H + a.H1, v = new Float32Array(D); let o = 0;
      for (const L of Ls) {
        const Hm = w.Hs[L];
        for (let jj = 0; jj < H; jj++) { let s = 0, mx = -Infinity; for (let u = 0; u < n; u++) { const x = Hm[u * H + jj]; s += x; if (x > mx) mx = x; } v[o + jj] = s / n; v[o + H + jj] = mx; v[o + 2 * H + jj] = Hm[G.hole * H + jj]; }
        for (let k = 0; k < 5; k++) for (let jj = 0; jj < H; jj++) v[o + 3 * H + k * H + jj] = Hm[rimOrd[k] * H + jj];
        o += 8 * H;
      }
      for (let jj = 0; jj < a.H1; jj++) v[o + jj] = w.Z[jj];
      res.acts = v;
    }
    if (wantNodes) { const Ls2 = Ls; const v = new Float32Array(Ls2.length * n * H); Ls2.forEach((L, li) => v.set(w.Hs[L].subarray(0, n * H), li * n * H)); res.nodes = v; }
  }
  return res;
}

// ------------------------------------------------------------------ main
const maps = A.levelMaps(GAME).map(m => m.key);
const sets = [];   // { G, M, key, set, split, idxs (state indices into S) }
log('engine', path.relative(root, ARENA_DIR), '| out', OUT);
for (const key of maps) {
  const G = A.graphFromLevels(GAME, key), M = analyseMap(G);
  let mism = 0; for (const i of M.S.unfilled) { const d = G.dist[M.S.keys[i]]; if ((d === undefined ? -1 : d) !== M.S.dist[i]) mism++; }
  if (mism || M.piErr) throw new Error(key + ': census mismatch ' + mism + ' / pi errors ' + M.piErr);
  sets.push({ G, M, key, set: 'census', split: TRAIN.indexOf(key) >= 0 ? 'train' : 'test', idxs: M.S.unfilled.filter(i => M.S.dist[i] > 0) });
}
log('census maps', sets.length, 'unfilled states', sets.reduce((s, x) => s + x.idxs.length, 0), '(pi is a permutation of every class; Theorem W bookkeeping checked)');

// splice set from the TEST maps
const SPLICE = QUICK ? 0 : num('splice', 400), SPLICE_MAPS = num('splice-maps', 40);
if (SPLICE > 0) {
  const R = A.rng(num('splice-seed', 1) * 7919 + 3), seen = new Set(); let kept = 0, tried = 0;
  for (let t = 0; t < SPLICE && kept < SPLICE_MAPS; t++) {
    tried++;
    const parent = A.graphFromLevels(GAME, TEST[t % TEST.length]), tri = new A.Tri(parent), k = 1 + R.int(3), ops = [];
    for (let q = 0; q < k; q++) { const op = A.spliceRandom(tri, R); if (op) ops.push(op); }
    if (!ops.length) continue;
    const G = tri.toGraph(parent.key + '~s' + t); if (G.n > 40 || A.checkTriangulation(G)) continue;
    if (G.deg[G.hole] !== 5) continue;
    const S0 = A.stateSpace(G, 150000); if (!S0) continue;
    const h = A.hardness(S0.hist); if (h.maxD < 3) continue;
    const sig = G.n + ':' + JSON.stringify(S0.hist) + ':' + S0.N; if (seen.has(sig)) continue; seen.add(sig);
    const M = analyseMap(G); if (M.piErr) throw new Error('pi errors on ' + G.key);
    const RR = A.rng(A.hashStr(G.key)), deep = [], d2 = [], d1 = [];
    for (const i of M.S.unfilled) { const d = M.S.dist[i]; if (d >= 3) deep.push(i); else if (d === 2) d2.push(i); else if (d === 1) d1.push(i); }
    A.shuffle(d2, RR); A.shuffle(d1, RR);
    const idxs = deep.concat(d2.slice(0, num('splice-d2', 150)), d1.slice(0, num('splice-d1', 60)));
    sets.push({ G, M, key: G.key, set: 'splice', split: 'splice', idxs, parent: parent.key, ops, hist: S0.hist });
    kept++;
  }
  log('splice set: tried', tried, 'kept', kept, 'maps;', sets.filter(s => s.set === 'splice').reduce((s, x) => s + x.idxs.length, 0), 'sampled states');
}

// ground-truth rows (+ ell)
const rows = [], moveRecs = [], stateRef = [];
let ellDone = 0; const ELL_MAX = num('ell-max-states', 400), ELL_CAP = num('ell-cap', 250000), ellStats = {};
for (const st of sets) {
  for (const i of st.idxs) {
    const d = st.M.S.dist[i]; let ellInfo = null;
    if (d <= 2) ellInfo = { ell: d, note: 'M3' };
    else if (ellDone < ELL_MAX || st.set === 'census') {
      ellDone++;
      const r = mixedEll(st.G, st.M.S.cols[i], d - 1, ELL_CAP);
      ellInfo = r.ell != null ? { ell: r.ell, note: 'bfs' } : r.capped ? { ell: null, note: 'capped@' + r.depth } : { ell: d, note: 'bfs' };
      const kk = 'k' + d + ':' + (ellInfo.ell == null ? ellInfo.note : 'l' + ellInfo.ell); ellStats[kk] = (ellStats[kk] || 0) + 1;
    }
    const R = stateRow(st.G, st.M, i, st.set, st.split, st.key, ellInfo);
    if (st.set === 'splice') R.row.parent = st.parent;
    rows.push(R.row); moveRecs.push(R.moves); stateRef.push({ st, i });
  }
  log('rows', st.key, st.idxs.length);
}
log('ell (kappa >= 3):', JSON.stringify(ellStats));
fs.writeFileSync(path.join(OUT, 'states.json'), JSON.stringify(rows));
fs.writeFileSync(path.join(OUT, 'moves.json'), JSON.stringify(moveRecs));
fs.writeFileSync(path.join(OUT, 'maps.json'), JSON.stringify(sets.map(s => ({ key: s.key, set: s.set, split: s.split, n: s.G.n, parent: s.parent || null, ops: s.ops || null, hist: s.M.S.hist, N: s.M.S.N, classes: s.M.clsN.size, orbits: s.M.orbits.length, allDL: s.M.orbits.filter(o => o.allDL).length, posW: s.M.orbits.filter(o => o.w > 0).length, degWord: Array.from(s.G.rim).map(u => s.G.deg[u]).join(''), edges: s.G.edges, hole: s.G.hole, link: Array.from(s.G.rim) }))));
log('wrote states.json, moves.json, maps.json:', rows.length, 'states');

// node-level sample (stratified by depth, fixed)
const NS = num('node-sample', 1500), Rn = A.rng(12345), byD = {};
rows.forEach((r, q) => { const k = r.d >= 3 ? 3 : r.d; (byD[k] = byD[k] || []).push(q); });
const nodeSample = []; for (const k of [1, 2, 3]) { const arr = A.shuffle((byD[k] || []).slice(), Rn); nodeSample.push(...arr.slice(0, Math.round(NS / 3))); }
nodeSample.sort((a, b) => a - b);
{ // node labels
  const lab = [];
  for (const q of nodeSample) {
    const { st, i } = stateRef[q], G = st.G, col = st.M.S.cols[i], j = st.M.jOf[i], buf = new Uint8Array(G.n), rc = A.relClasses(G, col);
    const x = k => X(G, j + k), al = col[x(0)], mu = col[x(1)], cA = col[x(3)], cB = col[x(4)];
    const K1 = new Set(comp(G, col, G.hole, mu, cA, x(1), buf)), K2 = new Set(comp(G, col, G.hole, mu, cB, x(1), buf)), Ks = new Set(comp(G, col, G.hole, al, mu, x(1), buf)), Kp = new Set(comp(G, col, G.hole, al, cA, x(2), buf));
    const chains = A.chainsOf(G, col), at = new Int32Array(G.n * 4).fill(-1); chains.forEach((K, ci) => { for (const u of K.vs) at[u * 4 + (col[u] === K.a ? K.b : K.a)] = ci; });
    for (let u = 0; u < G.n; u++) {
      const sz = [0, 0, 0, 0], reach = [0, 0, 0, 0];
      if (u !== G.hole) for (let b = 0; b < 4; b++) { const ci = at[u * 4 + b]; if (ci < 0) continue; const K = chains[ci], s = rc.rel[b]; sz[s] = K.vs.length; let m = K.rimMask, c = 0; while (m) { c += m & 1; m >>= 1; } reach[s] = c; }
      lab.push([q, u, u === G.hole ? -1 : rc.rel[col[u]], G.rimIdx[u] >= 0 ? 1 : 0, G.dHole[u], G.deg[u], ...sz, ...reach, K1.has(u) ? 1 : 0, K2.has(u) ? 1 : 0, Ks.has(u) ? 1 : 0, Kp.has(u) ? 1 : 0, rows[q].DL]);
    }
  }
  fs.writeFileSync(path.join(OUT, 'node_labels.json'), JSON.stringify({ cols: ['q', 'u', 'relColour', 'isRim', 'dHole', 'deg', 'sz0', 'sz1', 'sz2', 'sz3', 'reach0', 'reach1', 'reach2', 'reach3', 'onL1', 'onL2', 'onSigma', 'onPiChain', 'stateDL'], rows: lab, sample: nodeSample }));
  log('node sample', nodeSample.length, 'states,', lab.length, 'nodes');
}

// input baseline (control): the encoder's own features pooled like a hidden layer (mean, max, hole, rim5 of the 42
// per-region features), with and without the chain features. Written as acts_input-priors.f32 / acts_input-colours.f32.
if (!flags['no-input']) {
  for (const [tag, chainFeats] of [['input-priors', true], ['input-colours', false]]) {
    const F = A.F, D = 8 * F, all = new Float32Array(rows.length * D);
    for (let q = 0; q < rows.length; q++) {
      const { st, i } = stateRef[q], G = st.G, col = st.M.S.cols[i], enc = A.encode(G, col, { chainFeats }), Xf = enc.X, n = G.n, o = q * D;
      for (let f = 0; f < F; f++) { let s = 0, mx = -Infinity; for (let u = 0; u < n; u++) { const x = Xf[u * F + f]; s += x; if (x > mx) mx = x; } all[o + f] = s / n; all[o + F + f] = mx; all[o + 2 * F + f] = Xf[G.hole * F + f]; }
      for (let k = 0; k < 5; k++) for (let f = 0; f < F; f++) all[o + 3 * F + k * F + f] = Xf[enc.rimOrd[k] * F + f];
    }
    fs.writeFileSync(path.join(OUT, 'acts_' + tag + '.f32'), Buffer.from(all.buffer));
    fs.writeFileSync(path.join(OUT, 'model_' + tag + '.json'), JSON.stringify({ name: tag, input: true, layers: ['input'], D, H: F }));
  }
  log('input baselines written');
}

// models
const runList = modelNames.map(n => ({ name: n, random: false })).concat(actNames.map(n => ({ name: n, random: true })));
const summary = {};
const nodeSet = new Set(nodeSample);
for (const R0 of runList) {
  const L = loadModel(R0.name); let net = L.net, tag = R0.name;
  if (R0.random) { net = A.netCreate(L.meta.arch, 777); tag = 'random-' + R0.name; }
  const wantActs = actNames.indexOf(R0.name) >= 0;
  const ts = Date.now();
  // value on every state of every set's state space that is needed: the sampled states and all their afterstates
  const vCache = sets.map(() => new Map());
  const V = (sIdx, i) => { const c = vCache[sIdx]; let v = c.get(i); if (v === undefined) { const st = sets[sIdx]; v = forward(net, st.G, st.M.S.cols[i], false, false).v; c.set(i, v); } return v; };
  const setIndex = new Map(sets.map((s, k) => [s, k]));
  const nR = rows.length, val = new Float32Array(nR), dh = new Float32Array(nR), choice = new Int16Array(nR), polChoice = new Int16Array(nR), margin = new Float32Array(nR), optGap = new Float32Array(nR);
  const D0 = wantActs ? (layerList(net).length * 8 * net.arch.H + net.arch.H1) : 0, acts = wantActs ? new Float32Array(nR * D0) : null;
  const nodeChunks = [];
  for (let q = 0; q < nR; q++) {
    const { st, i } = stateRef[q], sIdx = setIndex.get(st), wantN = wantActs && nodeSet.has(q);
    const f = forward(net, st.G, st.M.S.cols[i], wantActs, wantN);
    val[q] = f.v; dh[q] = f.dh == null ? NaN : f.dh; vCache[sIdx].set(i, f.v);
    if (wantActs) acts.set(f.acts, q * D0);
    if (wantN) nodeChunks.push(f.nodes);
    // afterstate agent
    // margin over DISTINCT afterstates (two components of a two-component pair give the same afterstate up to renaming)
    const mv = moveRecs[q]; let best = -Infinity, bi = -1, bestOpt = -Infinity; const scK = new Map();
    for (let m = 0; m < mv.length; m++) {
      const k = mv[m][7], sc = st.M.S.won[k] ? -1 : -1 + L.gamma * V(sIdx, k);
      scK.set(k, sc); if (sc > best) { best = sc; bi = m; }
      if (mv[m][0] === st.M.S.dist[i] - 1 && sc > bestOpt) bestOpt = sc;
    }
    let second = -Infinity; for (const [k, sc] of scK) if (k !== mv[bi][7] && sc > second) second = sc;
    choice[q] = bi; margin[q] = second === -Infinity ? 99 : best - second; optGap[q] = best - bestOpt;
    let pb = 0; for (let m = 1; m < f.pol.length; m++) if (f.pol[m] > f.pol[pb]) pb = m; polChoice[q] = pb;
  }
  // rollouts (greedy, deterministic): from every sampled state, follow the agent on the state space (cap max(3d, d + 10))
  const steps = new Int16Array(nR);
  for (let q = 0; q < nR; q++) {
    const { st, i } = stateRef[q], sIdx = setIndex.get(st), S = st.M.S, cap = Math.max(3 * S.dist[i], S.dist[i] + 10);
    let s = i, k = 0, ok = false; const visited = new Set([s]);
    while (k < cap) {
      const mv = st.M.movesAll[s]; let best = -Infinity, bk = -1;
      for (const K of mv) { const a = S.idx.get(A.canon(st.G, A.flip(S.cols[s], K))), sc = S.won[a] ? -1 : -1 + L.gamma * V(sIdx, a); if (sc > best) { best = sc; bk = a; } }
      if (bk < 0) break; s = bk; k++; if (S.won[s]) { ok = true; break; } if (visited.has(s)) { k = -1; break; } visited.add(s);
    }
    steps[q] = ok ? k : -1;
  }
  const res = { name: tag, model: R0.name, random: R0.random, gamma: L.gamma, arch: net.arch, layers: layerList(net), D: D0, val: Array.from(val), dh: Array.from(dh), choice: Array.from(choice), polChoice: Array.from(polChoice), margin: Array.from(margin), optGap: Array.from(optGap), steps: Array.from(steps) };
  fs.writeFileSync(path.join(OUT, 'model_' + tag + '.json'), JSON.stringify(res));
  if (wantActs) {
    fs.writeFileSync(path.join(OUT, 'acts_' + tag + '.f32'), Buffer.from(acts.buffer));
    const tot = nodeChunks.reduce((s, c) => s + c.length, 0), all = new Float32Array(tot); let o = 0; for (const c of nodeChunks) { all.set(c, o); o += c.length; }
    fs.writeFileSync(path.join(OUT, 'nodes_' + tag + '.f32'), Buffer.from(all.buffer));
  }
  let ok2 = 0, n2 = 0; for (let q = 0; q < nR; q++) if (rows[q].d >= 2) { n2++; ok2 += moveRecs[q][choice[q]][0] === rows[q].d - 1; }
  summary[tag] = { agreeD2: +(ok2 / n2).toFixed(4), seconds: +((Date.now() - ts) / 1000).toFixed(1) };
  log(tag.padEnd(36), 'agree(d>=2)', summary[tag].agreeD2, summary[tag].seconds + 's');
}
fs.writeFileSync(path.join(OUT, 'run.json'), JSON.stringify({ date: new Date().toISOString(), engine: path.relative(root, ARENA_DIR), flags, models: modelNames, acts: actNames, summary, nStates: rows.length, ellStats, seconds: (Date.now() - t0) / 1000 }, null, 1));
log('done');
