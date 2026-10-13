/* docs/arena/engine.js: the Maze Arena engine (puzzle core, map-independent encoding, a small graph neural network
 * with a hand-written backward pass, afterstate TD learning, pretraining objectives, a random-map generator and the
 * testbench). Pure JS, no dependencies: runs in the page, in a Web Worker (importScripts) and in Node (require).
 *
 * The puzzle (as in docs/core.js): a planar triangulation coloured with 4 elements, one degree-5 region (the hole)
 * left blank because its 5 rim neighbours use all 4. A move is a Kempe swap of a two-colour chain (a connected
 * component of the {a, b} subgraph, hole excluded). A pair with exactly one component is a "web": swapping it only
 * renames two colours, so it is not a move. Goal: the rim uses 3 colours (a fill). Won states are terminal.
 *
 *   ARENA.graphFromLevels(GAME, key)   a graph from levels.js (keeps its census: dist[canon] = fewest swaps to a fill)
 *   ARENA.generateGraph(n, rng)         a random sphere triangulation with a degree-5 hole (icosahedron + face
 *                                       insertions + random edge flips), see generateGraph below
 *   ARENA.randomBlocked(G, rng)         a random proper 4-colouring whose rim uses all 4 colours
 *   ARENA.chainsOf(G, col)              every chain (webs flagged); moves = the non-web ones
 *   ARENA.flip, won, canon, decodeCanon, distOf, exactDistances, stateGraph, layoutStateGraph
 *   ARENA.encode(G, col)                map-independent features (below) + the move list
 *   ARENA.netCreate(arch, seed)         the GNN; netForward / netBackward / netValue; Adam
 *   ARENA.Trainer(cfg)                  afterstate TD(n) with replay, target network, epsilon schedule, aux losses
 *   ARENA.Pretrainer(trainer, cfg)      objectives (a) distance, (b) good swap, (c) masked colour, (d) chain sizes
 *   ARENA.Testbench(GAME, keys, opt)    held-out evaluation: optimal-move agreement + greedy rollouts
 *   ARENA.modelToJSON / modelFromJSON   the 'mazearena-model-1' file format
 *   Round 2: arch.kind = 'rec' is the TABULA-RASA track (raw colours + hole flag, weight-tied recurrent message
 *   passing iterated T times, value and "click a region, pick a colour" heads; see its section below); the GNN
 *   above is the PRIORS track. stateSpace (every colouring of a map, exact distances), Tri (splices: flip / insert /
 *   remove away from the hole), mirrorGraph, hardness, and testbench metrics agree4, deepStarts, extra (hard pool).
 *
 * Map-independent encoding. Colours are symmetric, so each region's colour is given RELATIVE to the rim. In an unfilled
 * state the rim (link, cyclic order) shows the pattern D x D y z: one colour D twice (at rim positions i and i+2,
 * never adjacent) and three singles. The reference position is the D whose second copy is 2 steps ahead (i), so the
 * rim reads D(i) s1(i+1) D(i+2) s2(i+3) s3(i+4); classes: D -> 0, s1 -> 1, s2 -> 2, s3 -> 3, and rim positions are
 * numbered from i. This is unique and does not depend on where the link list starts. (Reflecting the link swaps
 * classes 2 and 3; the generator reverses links at random so the net sees both.) Won or other states use a fixed
 * fallback: classes by rim count (descending) then first rim position, unused colours last; reference = the first
 * position of class 0.
 *
 * Per-region features (F = 42):
 *   0-3   relative colour class one-hot (hole and masked regions: zeros)
 *   4     is the hole            5   is on the rim         6-10  relative rim position one-hot
 *   11    degree / 8             12  graph distance to the hole / 5 (capped at 1)
 *   13    masked (self-supervised colour objective)
 *   14-17 for each partner class b (slot b; own class slot 0): size of the {own, b} chain through the region / n
 *   18-37 for each partner class b: the rim positions that chain touches (5 bits, relative positions)
 *   38-41 for each partner class b: the chain is a web
 * Globals (3): n / 40, non-web moves / 30, webs / 6.
 * arch.chainFeats = false zeroes features 14-41 (a "colours only" net that must find chains by message passing).
 *
 * Network (all weights in one Float32Array; shape table in the model file):
 *   h0 = relu(Win x + bin);  h_k = relu(Ws_k h_{k-1} + Wn_k mean_{nbrs} h_{k-1} + b_k), k = 1..K   (width H)
 *   readout r = [mean pool, max pool, the 5 rim embeddings in relative rim order, hole embedding, globals] (8H + 3)
 *   z = relu(W1 r + b1) (width H1);  value v = wv.z + bv;  dist d = wd.z + bd
 *   policy (one logit per candidate chain): relu(Wp1 [mean of the chain's h_K, z, chain features (12)] + bp1) . wp2
 *   node heads on h_K: masked colour class (4 logits), chain sizes and rim reach per partner slot (8)
 * Agent: AFTERSTATE values. Transitions are deterministic, so the score of a swap m in state s is
 *   -1 + gamma * V(swap(s, m)), with V = 0 exactly for a fill. Greedy picks the best; epsilon-greedy or softmax explore.
 */
(function (root) {
  'use strict';

  // ---------------------------------------------------------------- utilities
  function rng(seed) {   // mulberry32
    let a = seed >>> 0;
    const f = () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ t >>> 15, t | 1); t ^= t + Math.imul(t ^ t >>> 7, t | 61); return ((t ^ t >>> 14) >>> 0) / 4294967296; };
    f.int = k => Math.floor(f() * k);
    f.pick = arr => arr[Math.floor(f() * arr.length)];
    f.normal = () => { let u = 0, v = 0; while (u === 0) u = f(); v = f(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
    return f;
  }
  function hashStr(s) { let h = 2166136261 >>> 0; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619) >>> 0; } return h >>> 0; }
  const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
  function shuffle(a, R) { for (let i = a.length - 1; i > 0; i--) { const j = R.int(i + 1); const t = a[i]; a[i] = a[j]; a[j] = t; } return a; }

  // ---------------------------------------------------------------- graphs
  // G = { key, name, n, hole, rim (link, cyclic), adjStart/adjList (CSR), deg, rimIdx[u] (-1 or rim position),
  //       dHole[u] (BFS distance to the hole), edges, dist (census object canon -> d) or null, distMap (Map) or null }
  function makeGraph(spec) {
    const n = spec.n, hole = spec.hole, rim = Int32Array.from(spec.link);
    const nb = Array.from({ length: n }, () => []);
    spec.edges.forEach(([a, b]) => { if (a !== b && nb[a].indexOf(b) < 0) { nb[a].push(b); nb[b].push(a); } });
    const adjStart = new Int32Array(n + 1), deg = new Int32Array(n);
    for (let u = 0; u < n; u++) { deg[u] = nb[u].length; adjStart[u + 1] = adjStart[u] + nb[u].length; }
    const adjList = new Int32Array(adjStart[n]);
    for (let u = 0; u < n; u++) adjList.set(nb[u], adjStart[u]);
    const rimIdx = new Int8Array(n).fill(-1); rim.forEach((u, k) => { rimIdx[u] = k; });
    const dHole = new Int32Array(n).fill(-1); dHole[hole] = 0; const q = [hole];
    for (let i = 0; i < q.length; i++) { const u = q[i]; for (let e = adjStart[u]; e < adjStart[u + 1]; e++) { const w = adjList[e]; if (dHole[w] < 0) { dHole[w] = dHole[u] + 1; q.push(w); } } }
    const edges = []; for (let u = 0; u < n; u++) for (const w of nb[u]) if (u < w) edges.push([u, w]);
    return { key: spec.key || spec.name || 'g', name: spec.name || spec.key || 'graph', n, hole, rim, adjStart, adjList, deg, rimIdx, dHole, edges,
      dist: spec.dist || null, distMap: spec.distMap || null, pos: spec.pos || null, rot: spec.rot || null };
  }
  function graphFromLevels(GAME, key) {
    const g = GAME.graphs[key];
    if ((g.ncol || 4) !== 4 || (g.goal || 3) !== 3) return null;
    return makeGraph({ key, name: g.name, n: g.n, hole: g.hole, link: g.link, edges: g.edges, dist: g.dist, pos: g.pos, rot: g.rot });
  }
  // the 4-colour maps of levels.js with their levels, in play order
  function levelMaps(GAME) {
    const out = [], by = {};
    GAME.levels.forEach(L => {
      const g = GAME.graphs[L.g]; if ((g.ncol || 4) !== 4 || (g.goal || 3) !== 3) return;
      if (!by[L.g]) { by[L.g] = { key: L.g, name: g.name, n: g.n, levels: [] }; out.push(by[L.g]); }
      by[L.g].levels.push(L);
    });
    return out;
  }

  // ---------------------------------------------------------------- colourings
  const PAIRS = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];
  const pairIndex = (a, b) => { if (a > b) { const t = a; a = b; b = t; } return a === 0 ? b - 1 : a === 1 ? b + 1 : 5; };
  function rimCount(G, col) { let m = 0; for (let k = 0; k < 5; k++) m |= 1 << col[G.rim[k]]; return (m & 1) + (m >> 1 & 1) + (m >> 2 & 1) + (m >> 3 & 1); }
  function won(G, col) { return rimCount(G, col) <= 3; }
  function canon(G, col) { const mp = [-1, -1, -1, -1]; let s = '', k = 0; for (let u = 0; u < G.n; u++) { if (u === G.hole) continue; const c = col[u]; if (mp[c] < 0) mp[c] = k++; s += mp[c]; } return s; }
  function decodeCanon(G, s) { const col = new Int8Array(G.n); let i = 0; for (let u = 0; u < G.n; u++) { if (u === G.hole) { col[u] = -1; continue; } col[u] = s.charCodeAt(i++) - 48; } return col; }
  // every chain: { a, b, vs (Int32Array), rimMask (bits by absolute rim position), web }
  const _seen = { n: 0, a: null, stack: null };
  function chainsOf(G, col) {
    const n = G.n, hole = G.hole, out = [];
    if (_seen.n < n) { _seen.n = n; _seen.a = new Int32Array(n); _seen.stack = new Int32Array(n); }
    const seen = _seen.a, st = _seen.stack; let stamp = (seen.stampBase || 0);
    for (let p = 0; p < 6; p++) {
      const a = PAIRS[p][0], b = PAIRS[p][1], first = out.length;
      stamp++;
      for (let s = 0; s < n; s++) {
        if (s === hole || seen[s] === stamp || (col[s] !== a && col[s] !== b)) continue;
        let top = 0, len = 0; st[top++] = s; seen[s] = stamp; const vs = [];
        let rm = 0;
        while (top) {
          const u = st[--top]; vs.push(u); len++;
          if (G.rimIdx[u] >= 0) rm |= 1 << G.rimIdx[u];
          for (let e = G.adjStart[u], E = G.adjStart[u + 1]; e < E; e++) { const w = G.adjList[e]; if (w !== hole && seen[w] !== stamp && (col[w] === a || col[w] === b)) { seen[w] = stamp; st[top++] = w; } }
        }
        out.push({ a, b, vs: Int32Array.from(vs), rimMask: rm, web: false });
      }
      if (out.length - first === 1) out[first].web = true;
    }
    seen.stampBase = stamp;
    if (stamp > 2e9) { seen.fill(0); seen.stampBase = 0; }
    return out;
  }
  const movesOf = (G, col) => chainsOf(G, col).filter(K => !K.web);
  function flip(col, K) { const d = col.slice(); const a = K.a, b = K.b, vs = K.vs; for (let i = 0; i < vs.length; i++) { const u = vs[i]; d[u] = d[u] === a ? b : a; } return d; }
  // fewest swaps to a fill: 0 when won, the census when present, else a BFS label map, else null (unknown)
  function distOf(G, col) {
    if (won(G, col)) return 0;
    const key = canon(G, col);
    if (G.dist) { const d = G.dist[key]; return d === undefined ? null : d; }
    if (G.distMap) { const d = G.distMap.get(key); return d === undefined ? null : d; }
    return null;
  }
  // exact distances over the Kempe class of col: BFS over unfilled states (won states are leaves), then a
  // multi-source BFS back from the fills. Returns { map: Map canon -> d (finite only), states, maxD } or null when the
  // class has more than cap states.
  function exactDistances(G, col, cap) {
    cap = cap || 4000;
    const keys = [], cols = [], idx = new Map(), wonA = [], nb = [];
    const add = c => { const k = canon(G, c); let i = idx.get(k); if (i !== undefined) return i; i = keys.length; idx.set(k, i); keys.push(k); cols.push(c); wonA.push(won(G, c)); nb.push([]); return i; };
    add(col);
    for (let qi = 0; qi < keys.length; qi++) {
      if (wonA[qi]) continue;
      for (const K of movesOf(G, cols[qi])) {
        const j = add(flip(cols[qi], K)); nb[qi].push(j); if (wonA[j]) nb[j].push(qi);
        if (keys.length > cap) return null;
      }
    }
    const D = new Int32Array(keys.length).fill(-1), q = [];
    for (let i = 0; i < keys.length; i++) if (wonA[i]) { D[i] = 0; q.push(i); }
    for (let h = 0; h < q.length; h++) { const i = q[h]; for (const j of nb[i]) if (D[j] < 0 && !wonA[j]) { D[j] = D[i] + 1; q.push(j); } }
    const map = new Map(); let maxD = 0;
    for (let i = 0; i < keys.length; i++) if (!wonA[i] && D[i] > 0) { map.set(keys[i], D[i]); if (D[i] > maxD) maxD = D[i]; }
    return { map, states: keys.length, maxD };
  }

  // ---------------------------------------------------------------- random maps (pretraining data)
  // Random sphere triangulation on n vertices: start from the icosahedron (12 vertices, all degree 5), insert vertices
  // into random faces until n, then do random edge flips (about 6n), keeping every degree >= 3 and no multi-edges.
  // Faces are kept as a rotation system: nxt[a].get(b) = c means the face (a, b, c) counter-clockwise.
  const ICO_FACES = [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 5, 1], [1, 6, 2], [2, 7, 3], [3, 8, 4], [4, 9, 5], [5, 10, 1],
    [2, 6, 7], [3, 7, 8], [4, 8, 9], [5, 9, 10], [1, 10, 6], [11, 7, 6], [11, 8, 7], [11, 9, 8], [11, 10, 9], [11, 6, 10]];
  function generateGraph(n, R, opt) {
    opt = opt || {};
    n = Math.max(12, n | 0);
    for (let attempt = 0; attempt < 50; attempt++) {
      const nxt = []; const faces = [];
      const addF = (a, b, c) => { nxt[a].set(b, c); nxt[b].set(c, a); nxt[c].set(a, b); };
      const delF = (a, b, c) => { nxt[a].delete(b); nxt[b].delete(c); nxt[c].delete(a); };
      for (let v = 0; v < n; v++) nxt.push(new Map());
      ICO_FACES.forEach(f => addF(f[0], f[2], f[1]));   // ICO_FACES are clockwise from outside; store counter-clockwise
      for (let v = 12; v < n; v++) {   // insert into a random face
        let a, b, c;
        a = R.int(v); const ks = [...nxt[a].keys()]; b = ks[R.int(ks.length)]; c = nxt[a].get(b);
        delF(a, b, c); addF(a, b, v); addF(b, c, v); addF(c, a, v);
      }
      const flips = opt.flips != null ? opt.flips : 6 * n, maxDeg = opt.maxDeg || 11;
      for (let t = 0; t < flips; t++) {
        const a = R.int(n), ks = [...nxt[a].keys()], b = ks[R.int(ks.length)];
        const c = nxt[a].get(b), d = nxt[b].get(a);
        if (c === d || nxt[c].has(d) || nxt[a].size <= 3 || nxt[b].size <= 3 || nxt[c].size >= maxDeg || nxt[d].size >= maxDeg) continue;
        delF(a, b, c); delF(b, a, d); addF(c, a, d); addF(d, b, c);
      }
      const five = []; for (let v = 0; v < n; v++) if (nxt[v].size === 5) five.push(v);
      if (!five.length) continue;
      const hole = five[R.int(five.length)];
      // cyclic neighbour order of every vertex
      const rot = nxt.map(m => { const first = m.keys().next().value, out = [first]; let u = m.get(first); while (u !== first) { out.push(u); u = m.get(u); } return out; });
      let link = rot[hole].slice(); if (R() < 0.5) link.reverse();
      const off = R.int(5); link = link.slice(off).concat(link.slice(0, off));
      const edges = []; for (let u = 0; u < n; u++) for (const w of rot[u]) if (u < w) edges.push([u, w]);
      return makeGraph({ key: 'gen' + n + '-' + Math.floor(R() * 1e9).toString(36), name: 'random ' + n, n, hole, link, edges, rot });
    }
    throw new Error('generateGraph: no degree-5 vertex after 50 attempts');
  }
  // is G a sphere triangulation (Euler: E = 3n - 6, every face of the rotation system a triangle, faces = 2n - 4)?
  function checkTriangulation(G) {
    const n = G.n, rot = G.rot; if (!rot) return 'no rotation system';
    if (G.edges.length !== 3 * n - 6) return 'E != 3n-6';
    const pos = rot.map(r => { const m = new Map(); r.forEach((w, i) => m.set(w, i)); return m; });
    const seen = new Set(); let faces = 0;
    for (let u = 0; u < n; u++) for (const v of rot[u]) {
      if (seen.has(u + ',' + v)) continue;
      let a = u, b = v, len = 0;
      do { seen.add(a + ',' + b); const i = pos[b].get(a); if (i === undefined) return 'asymmetric'; const c = rot[b][(i + rot[b].length - 1) % rot[b].length]; a = b; b = c; len++; } while (!(a === u && b === v) && len < 10);
      if (len !== 3) return 'face of length ' + len;
      faces++;
    }
    if (faces !== 2 * n - 4) return 'faces ' + faces + ' != 2n-4';
    for (let u = 0; u < n; u++) if (G.deg[u] < 3) return 'degree < 3';
    if (G.deg[G.hole] !== 5) return 'hole degree ' + G.deg[G.hole];
    for (let k = 0; k < 5; k++) { const a = G.rim[k], b = G.rim[(k + 1) % 5]; if (rot[a].indexOf(b) < 0) return 'rim not a cycle'; }
    return null;
  }
  // a random proper 4-colouring of G minus the hole whose rim uses all 4 colours (pattern D s1 D s2 s3 at a random
  // rotation, random colour names), completed by randomized backtracking (most-constrained region first)
  function randomBlocked(G, R, maxTries) {
    const n = G.n, hole = G.hole;
    for (let t = 0; t < (maxTries || 40); t++) {
      const col = new Int8Array(n).fill(-1), perm = shuffle([0, 1, 2, 3], R), r0 = R.int(5), pat = [0, 1, 0, 2, 3];
      for (let k = 0; k < 5; k++) col[G.rim[(r0 + k) % 5]] = perm[pat[k]];
      let bad = false;   // a chord between rim regions i and i + 2 can make the pattern improper
      for (let k = 0; k < 5 && !bad; k++) { const u = G.rim[k]; for (let e = G.adjStart[u]; e < G.adjStart[u + 1]; e++) { const w = G.adjList[e]; if (w !== hole && col[w] === col[u]) bad = true; } }
      if (bad) continue;
      let budget = 4000;
      const free = []; for (let u = 0; u < n; u++) if (u !== hole && col[u] < 0) free.push(u);
      const ok = (u, c) => { for (let e = G.adjStart[u]; e < G.adjStart[u + 1]; e++) if (col[G.adjList[e]] === c) return false; return true; };
      const solve = () => {
        if (--budget < 0) return false;
        let best = -1, bestOpts = null;
        for (const u of free) {
          if (col[u] >= 0) continue;
          const opts = []; for (let c = 0; c < 4; c++) if (ok(u, c)) opts.push(c);
          if (!opts.length) return false;
          if (!bestOpts || opts.length < bestOpts.length) { best = u; bestOpts = opts; if (opts.length === 1) break; }
        }
        if (best < 0) return true;
        shuffle(bestOpts, R);
        for (const c of bestOpts) { col[best] = c; if (solve()) return true; }
        col[best] = -1; return false;
      };
      if (solve()) { col[hole] = -1; return col; }
    }
    return null;
  }

  // ---------------------------------------------------------------- encoding
  const F = 42, GD = 3, CF = 12;
  // relative colour classes: rel[c] for colours 0..3, and ref = the reference rim position
  function relClasses(G, col) {
    const r = [0, 0, 0, 0, 0]; for (let k = 0; k < 5; k++) r[k] = col[G.rim[k]];
    const rel = [-1, -1, -1, -1];
    if (rimCount(G, col) === 4) {
      for (let i = 0; i < 5; i++) if (r[i] === r[(i + 2) % 5]) {
        rel[r[i]] = 0; rel[r[(i + 1) % 5]] = 1; rel[r[(i + 3) % 5]] = 2; rel[r[(i + 4) % 5]] = 3;
        return { rel, ref: i };
      }
    }
    const cnt = [0, 0, 0, 0], first = [9, 9, 9, 9];
    for (let k = 0; k < 5; k++) { cnt[r[k]]++; if (first[r[k]] === 9) first[r[k]] = k; }
    const order = [0, 1, 2, 3].sort((x, y) => (cnt[y] - cnt[x]) || (first[x] - first[y]) || (x - y));
    order.forEach((c, i) => { rel[c] = i; });
    return { rel, ref: first[order[0]] };
  }
  // enc = { G, col, n, X, glob, rimOrd (5 regions, relative rim order), hole, moves: [{K, feat (CF)}], chains, rel, ref }
  function encode(G, col, opt) {
    opt = opt || {};
    if (opt.kind === 'rec') return encodeRaw(G, col, opt);
    const n = G.n, hole = G.hole, X = new Float32Array(n * F), { rel, ref } = relClasses(G, col);
    const chains = chainsOf(G, col), at = new Int32Array(n * 4).fill(-1);
    chains.forEach((K, i) => { for (const u of K.vs) at[u * 4 + (col[u] === K.a ? K.b : K.a)] = i; });
    const relMask = m => { let o = 0; for (let k = 0; k < 5; k++) if (m >> ((ref + k) % 5) & 1) o |= 1 << k; return o; };
    const chainOn = opt.chainFeats !== false && !opt.dropChains;
    const masked = opt.mask || null;
    for (let u = 0; u < n; u++) {
      const o = u * F;
      if (u === hole) X[o + 4] = 1;
      else if (masked && masked[u]) X[o + 13] = 1;
      else X[o + rel[col[u]]] = 1;
      const rk = G.rimIdx[u];
      if (rk >= 0) { X[o + 5] = 1; X[o + 6 + (rk - ref + 5) % 5] = 1; }
      X[o + 11] = G.deg[u] / 8; X[o + 12] = Math.min(1, G.dHole[u] / 5);
      if (u === hole || !chainOn || (masked && masked[u])) continue;
      for (let b = 0; b < 4; b++) {
        const ci = at[u * 4 + b]; if (ci < 0) continue;
        const K = chains[ci], s = rel[b];
        X[o + 14 + s] = K.vs.length / n;
        const m = relMask(K.rimMask); for (let k = 0; k < 5; k++) if (m >> k & 1) X[o + 18 + s * 5 + k] = 1;
        if (K.web) X[o + 38 + s] = 1;
      }
    }
    const rimOrd = new Int32Array(5); for (let k = 0; k < 5; k++) rimOrd[k] = G.rim[(ref + k) % 5];
    const moves = []; let webs = 0;
    chains.forEach(K => {
      if (K.web) { webs++; return; }
      const f = new Float32Array(CF); f[0] = K.vs.length / n; const m = relMask(K.rimMask); for (let k = 0; k < 5; k++) if (m >> k & 1) f[1 + k] = 1;
      f[6 + pairIndex(rel[K.a], rel[K.b])] = 1;
      moves.push({ K, feat: f });
    });
    const glob = new Float32Array(GD); glob[0] = n / 40; glob[1] = moves.length / 30; glob[2] = webs / 6;
    return { G, col, n, X, glob, rimOrd, hole, moves, chains, rel, ref, at };
  }

  // ---------------------------------------------------------------- the network
  function archDefaults(a) {
    if (a && a.kind === 'rec') return recDefaults(a);
    a = Object.assign({ H: 16, K: 3, H1: 0, Hp: 0, chainFeats: true }, a || {});
    a.H = a.H | 0; a.K = a.K | 0; a.H1 = a.H1 || 2 * a.H; a.Hp = a.Hp || a.H; a.F = F; a.GD = GD; a.CF = CF;
    a.kind = 'gnn'; a.readout = 'mean,max,rim5,hole,glob'; a.heads = ['value', 'dist', 'policy', 'colour', 'chain']; a.feat = 'v1';
    return a;
  }
  function shapesOf(a) {
    if (a.kind === 'rec') return recShapes(a);
    const H = a.H, R = 8 * H + GD, S = [['Win', H, F], ['bin', H, 1]];
    for (let k = 1; k <= a.K; k++) S.push(['Ws' + k, H, H], ['Wn' + k, H, H], ['b' + k, H, 1]);
    S.push(['W1', a.H1, R], ['b1', a.H1, 1], ['wv', 1, a.H1], ['bv', 1, 1], ['wd', 1, a.H1], ['bd', 1, 1],
      ['Wp1', a.Hp, H + a.H1 + CF], ['bp1', a.Hp, 1], ['wp2', 1, a.Hp], ['bp2', 1, 1],
      ['Wc', 4, H], ['bc', 4, 1], ['Wch', 8, H], ['bch', 8, 1]);
    return S;
  }
  function netCreate(arch, seed, dtype) {
    const a = archDefaults(arch), AT = dtype || Float32Array, shapes = shapesOf(a), off = {};
    let tot = 0; shapes.forEach(([nm, r, c]) => { off[nm] = tot; tot += r * c; });
    const net = { arch: a, AT, shapes, off, size: tot, params: new AT(tot), ws: null };
    const R = rng(seed == null ? 1 : seed);
    shapes.forEach(([nm, r, c]) => {
      if (c === 1 || nm[0] === 'b') return;   // biases start at 0
      let fan = c; if (nm[0] === 'W' && (nm[1] === 's' || nm[1] === 'n')) fan = 2 * c;
      let lim = Math.sqrt(6 / fan);
      if (nm === 'wv' || nm === 'wd' || nm === 'wp2' || nm === 'Wc' || nm === 'Wch' || nm === 'Wo') lim = 0.1 / Math.sqrt(c);
      if (a.kind === 'rec' && (nm === 'Ws' || nm === 'Wn')) lim = 1.56 / Math.sqrt(2 * c);   // spectral radius ~0.9: stable when iterated
      const o = off[nm]; for (let i = 0; i < r * c; i++) net.params[o + i] = (R() * 2 - 1) * lim;
    });
    return net;
  }
  const paramCount = net => net.size;
  // workspace for one sample (forward cache, reused): sized for n regions and m candidate chains
  function workspace(net, n, m) {
    let w = net.ws; const a = net.arch, H = a.H, AT = net.AT;
    if (w && w.n >= n && w.m >= m) return w;
    n = Math.max(n, w ? w.n : 0, 40); m = Math.max(m, w ? w.m : 0, 64);
    w = { n, m, Hs: [], Ms: [], dH: new AT(n * H), dH2: new AT(n * H), dM: new AT(n * H), R: new AT(8 * H + GD), dR: new AT(8 * H + GD),
      zpre: new AT(a.H1), Z: new AT(a.H1), dZ: new AT(a.H1), arg: new Int32Array(H),
      pin: new AT(m * (H + a.H1 + CF)), ppre: new AT(m * a.Hp), logit: new AT(m), dpin: new AT(H + a.H1 + CF),
      colL: new AT(n * 4), chP: new AT(n * 8) };
    for (let k = 0; k <= a.K; k++) { w.Hs.push(new AT(n * H)); if (k < a.K) w.Ms.push(new AT(n * H)); }
    net.ws = w; return w;
  }
  // y[n x r] = x[n x c] W^T (+ b): W is r x c row-major at offset oW
  function linRows(P, oW, oB, x, n, c, r, y, accumulate) {
    for (let u = 0; u < n; u++) {
      const xo = u * c, yo = u * r;
      for (let i = 0; i < r; i++) {
        let s = accumulate ? y[yo + i] : (oB >= 0 ? P[oB + i] : 0); const wo = oW + i * c;
        for (let j = 0; j < c; j++) s += P[wo + j] * x[xo + j];
        y[yo + i] = s;
      }
    }
  }
  function meanNbr(G, Hm, H, M) {
    const n = G.n;
    for (let u = 0; u < n; u++) {
      const s = G.adjStart[u], e = G.adjStart[u + 1], inv = 1 / (e - s), mo = u * H;
      for (let j = 0; j < H; j++) M[mo + j] = 0;
      for (let q = s; q < e; q++) { const wo = G.adjList[q] * H; for (let j = 0; j < H; j++) M[mo + j] += Hm[wo + j]; }
      for (let j = 0; j < H; j++) M[mo + j] *= inv;
    }
  }
  // forward pass. need: { policy, nodes }. Returns { v, d, logits (if policy), colL/chP (if nodes) } (views into ws).
  function netForward(net, enc, need, params) {
    if (net.arch.kind === 'rec') return recForward(net, enc, need, params);
    need = need || {};
    const P = params || net.params, a = net.arch, H = a.H, n = enc.n, G = enc.G, m = need.policy ? enc.moves.length : 0;
    const w = workspace(net, n, m), Hs = w.Hs, Ms = w.Ms, off = net.off;
    // layer 0
    linRows(P, off.Win, off.bin, enc.X, n, F, H, Hs[0], false);
    for (let i = 0; i < n * H; i++) if (Hs[0][i] < 0) Hs[0][i] = 0;
    for (let k = 1; k <= a.K; k++) {
      const Hp = Hs[k - 1], M = Ms[k - 1], Hn = Hs[k];
      meanNbr(G, Hp, H, M);
      linRows(P, off['Ws' + k], off['b' + k], Hp, n, H, H, Hn, false);
      linRows(P, off['Wn' + k], -1, M, n, H, H, Hn, true);
      for (let i = 0; i < n * H; i++) if (Hn[i] < 0) Hn[i] = 0;
    }
    const HK = Hs[a.K], R = w.R;
    for (let j = 0; j < H; j++) { let s = 0, mx = -Infinity, am = 0; for (let u = 0; u < n; u++) { const x = HK[u * H + j]; s += x; if (x > mx) { mx = x; am = u; } } R[j] = s / n; R[H + j] = mx; w.arg[j] = am; }
    for (let k = 0; k < 5; k++) { const o = enc.rimOrd[k] * H; for (let j = 0; j < H; j++) R[2 * H + k * H + j] = HK[o + j]; }
    { const o = enc.hole * H; for (let j = 0; j < H; j++) R[7 * H + j] = HK[o + j]; }
    for (let g = 0; g < GD; g++) R[8 * H + g] = enc.glob[g];
    const RD = 8 * H + GD;
    linRows(P, off.W1, off.b1, R, 1, RD, a.H1, w.zpre, false);
    for (let i = 0; i < a.H1; i++) w.Z[i] = w.zpre[i] > 0 ? w.zpre[i] : 0;
    let v = P[off.bv], d = P[off.bd];
    for (let i = 0; i < a.H1; i++) { v += P[off.wv + i] * w.Z[i]; d += P[off.wd + i] * w.Z[i]; }
    const out = { v, d, n, m, enc };
    if (m) {
      const PI = H + a.H1 + CF;
      for (let c = 0; c < m; c++) {
        const mv = enc.moves[c], vs = mv.K.vs, po = c * PI;
        for (let j = 0; j < H; j++) w.pin[po + j] = 0;
        for (const u of vs) { const o = u * H; for (let j = 0; j < H; j++) w.pin[po + j] += HK[o + j]; }
        for (let j = 0; j < H; j++) w.pin[po + j] /= vs.length;
        for (let j = 0; j < a.H1; j++) w.pin[po + H + j] = w.Z[j];
        for (let j = 0; j < CF; j++) w.pin[po + H + a.H1 + j] = mv.feat[j];
        let lg = P[off.bp2];
        for (let i = 0; i < a.Hp; i++) {
          let s = P[off.bp1 + i]; const wo = off.Wp1 + i * PI;
          for (let j = 0; j < PI; j++) s += P[wo + j] * w.pin[po + j];
          w.ppre[c * a.Hp + i] = s; if (s > 0) lg += P[off.wp2 + i] * s;
        }
        w.logit[c] = lg;
      }
      out.logits = w.logit;
    }
    if (need.nodes) {
      linRows(P, off.Wc, off.bc, HK, n, H, 4, w.colL, false);
      linRows(P, off.Wch, off.bch, HK, n, H, 8, w.chP, false);
      out.colL = w.colL; out.chP = w.chP;
    }
    return out;
  }
  const netValue = (net, enc, params) => netForward(net, enc, null, params).v;
  // backward for the last forward. g = { v, d, logits (array m or null), colL (n*4 or null), chP (n*8 or null) };
  // accumulates into grad (same layout as params)
  function netBackward(net, out, g, grad) {
    if (net.arch.kind === 'rec') return recBackward(net, out, g, grad);
    const P = net.params, a = net.arch, H = a.H, enc = out.enc, n = enc.n, G = enc.G, w = net.ws, off = net.off, HK = w.Hs[a.K];
    const RD = 8 * H + GD, dZ = w.dZ, dH = w.dH;
    for (let i = 0; i < a.H1; i++) dZ[i] = 0;
    for (let i = 0; i < n * H; i++) dH[i] = 0;
    const gv = g.v || 0, gd = g.d || 0;
    if (gv || gd) {
      grad[off.bv] += gv; grad[off.bd] += gd;
      for (let i = 0; i < a.H1; i++) { grad[off.wv + i] += gv * w.Z[i]; grad[off.wd + i] += gd * w.Z[i]; dZ[i] += gv * P[off.wv + i] + gd * P[off.wd + i]; }
    }
    if (g.logits && out.m) {
      const PI = H + a.H1 + CF, dpin = w.dpin;
      for (let c = 0; c < out.m; c++) {
        const gl = g.logits[c]; if (!gl) continue;
        const po = c * PI; grad[off.bp2] += gl;
        for (let j = 0; j < PI; j++) dpin[j] = 0;
        for (let i = 0; i < a.Hp; i++) {
          const pre = w.ppre[c * a.Hp + i]; if (pre <= 0) continue;
          grad[off.wp2 + i] += gl * pre;
          const dp = gl * P[off.wp2 + i], wo = off.Wp1 + i * PI;
          grad[off.bp1 + i] += dp;
          for (let j = 0; j < PI; j++) { grad[wo + j] += dp * w.pin[po + j]; dpin[j] += dp * P[wo + j]; }
        }
        const vs = enc.moves[c].K.vs, inv = 1 / vs.length;
        for (const u of vs) { const o = u * H; for (let j = 0; j < H; j++) dH[o + j] += dpin[j] * inv; }
        for (let j = 0; j < a.H1; j++) dZ[j] += dpin[H + j];
      }
    }
    if (g.colL) for (let u = 0; u < n; u++) for (let i = 0; i < 4; i++) {
      const gg = g.colL[u * 4 + i]; if (!gg) continue;
      grad[off.bc + i] += gg; const wo = off.Wc + i * H, o = u * H;
      for (let j = 0; j < H; j++) { grad[wo + j] += gg * HK[o + j]; dH[o + j] += gg * P[wo + j]; }
    }
    if (g.chP) for (let u = 0; u < n; u++) for (let i = 0; i < 8; i++) {
      const gg = g.chP[u * 8 + i]; if (!gg) continue;
      grad[off.bch + i] += gg; const wo = off.Wch + i * H, o = u * H;
      for (let j = 0; j < H; j++) { grad[wo + j] += gg * HK[o + j]; dH[o + j] += gg * P[wo + j]; }
    }
    // MLP
    const dR = w.dR; for (let j = 0; j < RD; j++) dR[j] = 0;
    for (let i = 0; i < a.H1; i++) {
      if (w.zpre[i] <= 0 || !dZ[i]) continue;
      const dz = dZ[i], wo = off.W1 + i * RD; grad[off.b1 + i] += dz;
      for (let j = 0; j < RD; j++) { grad[wo + j] += dz * w.R[j]; dR[j] += dz * P[wo + j]; }
    }
    // readout
    for (let j = 0; j < H; j++) { const gm = dR[j] / n; for (let u = 0; u < n; u++) dH[u * H + j] += gm; dH[w.arg[j] * H + j] += dR[H + j]; }
    for (let k = 0; k < 5; k++) { const o = enc.rimOrd[k] * H; for (let j = 0; j < H; j++) dH[o + j] += dR[2 * H + k * H + j]; }
    { const o = enc.hole * H; for (let j = 0; j < H; j++) dH[o + j] += dR[7 * H + j]; }
    // message-passing layers
    let cur = dH, nxt = w.dH2;
    for (let k = a.K; k >= 1; k--) {
      const Hk = w.Hs[k], Hp = w.Hs[k - 1], M = w.Ms[k - 1], dM = w.dM, oS = off['Ws' + k], oN = off['Wn' + k], oB = off['b' + k];
      for (let i = 0; i < n * H; i++) { if (Hk[i] <= 0) cur[i] = 0; nxt[i] = 0; dM[i] = 0; }
      for (let u = 0; u < n; u++) {
        const o = u * H;
        for (let i = 0; i < H; i++) {
          const da = cur[o + i]; if (!da) continue;
          grad[oB + i] += da; const ws = oS + i * H, wn = oN + i * H;
          for (let j = 0; j < H; j++) { grad[ws + j] += da * Hp[o + j]; grad[wn + j] += da * M[o + j]; nxt[o + j] += da * P[ws + j]; dM[o + j] += da * P[wn + j]; }
        }
      }
      for (let u = 0; u < n; u++) {
        const s = G.adjStart[u], e = G.adjStart[u + 1], inv = 1 / (e - s), o = u * H;
        for (let q = s; q < e; q++) { const wo = G.adjList[q] * H; for (let j = 0; j < H; j++) nxt[wo + j] += dM[o + j] * inv; }
      }
      const t = cur; cur = nxt; nxt = t;
    }
    const H0 = w.Hs[0], X = enc.X;
    for (let u = 0; u < n; u++) {
      const o = u * H, xo = u * F;
      for (let i = 0; i < H; i++) {
        if (H0[o + i] <= 0) continue; const da = cur[o + i]; if (!da) continue;
        grad[off.bin + i] += da; const wo = off.Win + i * F;
        for (let j = 0; j < F; j++) { const x = X[xo + j]; if (x) grad[wo + j] += da * x; }
      }
    }
  }
  // Adam / AdamW with global-norm clipping; grad is the SUM over `count` samples
  function adamCreate(net) { return { m: new Float32Array(net.size), v: new Float32Array(net.size), t: 0 }; }
  function adamStep(net, opt, grad, count, hp) {
    const lr = hp.lr, b1 = 0.9, b2 = 0.999, eps = 1e-8, wd = hp.wd || 0, clip = hp.clip || 0, P = net.params, N = net.size, sc = 1 / Math.max(1, count);
    let nrm = 0; for (let i = 0; i < N; i++) { const gi = grad[i] * sc; nrm += gi * gi; }
    nrm = Math.sqrt(nrm);
    const k = clip && nrm > clip ? clip / nrm * sc : sc;
    opt.t++; const c1 = 1 - Math.pow(b1, opt.t), c2 = 1 - Math.pow(b2, opt.t);
    for (let i = 0; i < N; i++) {
      const gi = grad[i] * k;
      opt.m[i] = b1 * opt.m[i] + (1 - b1) * gi; opt.v[i] = b2 * opt.v[i] + (1 - b2) * gi * gi;
      P[i] -= lr * ((opt.m[i] / c1) / (Math.sqrt(opt.v[i] / c2) + eps) + wd * P[i]);
    }
    grad.fill(0);
    return nrm;
  }

  // ---------------------------------------------------------------- the tabula-rasa recurrent net (arch.kind = 'rec')
  // Inputs per region: RAW colour one-hot (4) and the hole flag (1); nothing else (no degree, rim index, chain or
  // rim-relative features). Colour symmetry is learned from colour-permutation augmentation (opt.perm).
  //   e = Wx x;  h_0 = tanh(e + b0);  h_t = tanh(Ws h_{t-1} + Wn (sum of neighbours' h_{t-1}) / 6 + e + b), t = 1..T
  // one weight-tied layer iterated T times with the input injected every iteration ("deep thinking" style), so T can be
  // raised at test time. Readout [mean, max, the marked hole's h_T] -> MLP -> value. Policy head: for every region u a
  // logit per colour b ("click region u, pick colour b"): relu(Wq [h_T(u), z] + bq) -> Wo -> 4 logits; the rules turn
  // (u, b) into the Kempe swap of the {col(u), b} chain through u (illegal when that chain is a web or b = col(u)).
  const FR = 5;
  function recDefaults(a) {
    a = Object.assign({ H: 32, T: 16, H1: 0, Hq: 0, holeRead: true }, a || {});
    a.kind = 'rec'; a.H = a.H | 0; a.T = a.T | 0; a.H1 = a.H1 || 2 * a.H; a.Hq = a.Hq || a.H; a.F = FR; a.holeRead = a.holeRead !== false;
    a.readout = 'mean,max' + (a.holeRead ? ',hole' : ''); a.heads = ['value', 'policy(u,b)']; a.feat = 'raw1'; a.chainFeats = false;
    return a;
  }
  function recShapes(a) {
    const H = a.H, R = (a.holeRead ? 3 : 2) * H;
    return [['Wx', H, FR], ['b0', H, 1], ['Ws', H, H], ['Wn', H, H], ['b', H, 1], ['W1', a.H1, R], ['b1', a.H1, 1], ['wv', 1, a.H1], ['bv', 1, 1],
      ['Wq', a.Hq, H + a.H1], ['bq', a.Hq, 1], ['Wo', 4, a.Hq], ['bo', 4, 1]];
  }
  // enc for the rec net: { G, col, n, hole, X (n x 5), perm, moves/chains/at unless opt.noMoves }
  function encodeRaw(G, col, opt) {
    const n = G.n, X = new Float32Array(n * FR), perm = opt.perm || null;
    for (let u = 0; u < n; u++) { if (u === G.hole) X[u * FR + 4] = 1; else X[u * FR + (perm ? perm[col[u]] : col[u])] = 1; }
    const enc = { G, col, n, hole: G.hole, X, perm, kind: 'rec' };
    if (!opt.noMoves) {
      const chains = chainsOf(G, col), at = new Int32Array(n * 4).fill(-1), moves = [];
      chains.forEach((K, i) => { for (const u of K.vs) at[u * 4 + (col[u] === K.a ? K.b : K.a)] = i; });
      chains.forEach(K => { if (!K.web) moves.push({ K }); });
      enc.chains = chains; enc.at = at; enc.moves = moves;
    }
    return enc;
  }
  // (region u, colour slot k) -> move index (or -1). Slots are in the net's colour frame (after perm).
  function actionTable(enc) {
    const n = enc.n, tab = new Int32Array(n * 4).fill(-1), mi = new Int32Array(enc.chains.length).fill(-1), perm = enc.perm;
    let k = 0; enc.chains.forEach((K, i) => { if (!K.web) mi[i] = k++; });
    for (let u = 0; u < n; u++) { if (u === enc.hole) continue; for (let b = 0; b < 4; b++) { const ci = enc.at[u * 4 + b]; if (ci >= 0 && mi[ci] >= 0) tab[u * 4 + (perm ? perm[b] : b)] = mi[ci]; } }
    return tab;
  }
  // listwise policy loss over legal clicks: -log P(a good click) = LSE(legal) - LSE(good); returns { loss, grad (n*4) }
  function policyLoss(logits, tab, goodMove, n) {
    const g = new Float32Array(n * 4); let mxL = -Infinity, mxG = -Infinity, any = false;
    for (let k = 0; k < n * 4; k++) { if (tab[k] < 0) continue; mxL = Math.max(mxL, logits[k]); if (goodMove[tab[k]]) { mxG = Math.max(mxG, logits[k]); any = true; } }
    if (!any) return { loss: 0, grad: g };
    let sL = 0, sG = 0;
    for (let k = 0; k < n * 4; k++) { if (tab[k] < 0) continue; sL += Math.exp(logits[k] - mxL); if (goodMove[tab[k]]) sG += Math.exp(logits[k] - mxG); }
    for (let k = 0; k < n * 4; k++) { if (tab[k] < 0) continue; g[k] = Math.exp(logits[k] - mxL) / sL - (goodMove[tab[k]] ? Math.exp(logits[k] - mxG) / sG : 0); }
    return { loss: (mxL + Math.log(sL)) - (mxG + Math.log(sG)), grad: g };
  }
  // a second handle on the same weights with its own workspace (two forwards kept for a ranking loss)
  const netShare = net => Object.assign({}, net, { ws: null, wsr: null });
  function recWorkspace(net, n, T) {
    let w = net.wsr; const a = net.arch, H = a.H, AT = net.AT;
    if (w && w.n >= n && w.T >= T) return w;
    n = Math.max(n, w ? w.n : 0, 40); T = Math.max(T, w ? w.T : 0, a.T);
    w = { n, T, Hs: [], Ms: [], E: new AT(n * H), cur: new AT(n * H), nxt: new AT(n * H), dM: new AT(n * H), dE: new AT(n * H), R: new AT(3 * H), dR: new AT(3 * H),
      zpre: new AT(a.H1), Z: new AT(a.H1), dZ: new AT(a.H1), arg: new Int32Array(H), qin: new AT(n * (H + a.H1)), qpre: new AT(n * a.Hq), logit: new AT(n * 4), dq: new AT(a.Hq), dqin: new AT(H + a.H1) };
    for (let t = 0; t <= T; t++) { w.Hs.push(new AT(n * H)); if (t < T) w.Ms.push(new AT(n * H)); }
    net.wsr = w; return w;
  }
  function recForward(net, enc, need, params) {
    need = need || {};
    const P = params || net.params, a = net.arch, H = a.H, T = need.T || a.T, n = enc.n, G = enc.G, off = net.off, w = recWorkspace(net, n, T);
    const E = w.E, Hs = w.Hs, Ms = w.Ms, X = enc.X;
    for (let u = 0; u < n; u++) for (let i = 0; i < H; i++) {   // e = Wx x (x is one-hot: one or two nonzeros)
      let s = 0; const xo = u * FR, wo = off.Wx + i * FR; for (let j = 0; j < FR; j++) if (X[xo + j]) s += P[wo + j] * X[xo + j];
      E[u * H + i] = s; Hs[0][u * H + i] = Math.tanh(s + P[off.b0 + i]);
    }
    const sc = 1 / 6;
    for (let t = 1; t <= T; t++) {
      const Hp = Hs[t - 1], M = Ms[t - 1], Hn = Hs[t];
      for (let u = 0; u < n; u++) {
        const mo = u * H; for (let j = 0; j < H; j++) M[mo + j] = 0;
        for (let q = G.adjStart[u], e = G.adjStart[u + 1]; q < e; q++) { const wo = G.adjList[q] * H; for (let j = 0; j < H; j++) M[mo + j] += Hp[wo + j]; }
        for (let j = 0; j < H; j++) M[mo + j] *= sc;
      }
      for (let u = 0; u < n; u++) {
        const o = u * H;
        for (let i = 0; i < H; i++) {
          let s0 = E[o + i] + P[off.b + i], s1 = 0, s2 = 0, s3 = 0; const ws = off.Ws + i * H, wn = off.Wn + i * H;
          let j = 0;
          for (; j + 3 < H; j += 4) {
            s0 += P[ws + j] * Hp[o + j] + P[wn + j] * M[o + j]; s1 += P[ws + j + 1] * Hp[o + j + 1] + P[wn + j + 1] * M[o + j + 1];
            s2 += P[ws + j + 2] * Hp[o + j + 2] + P[wn + j + 2] * M[o + j + 2]; s3 += P[ws + j + 3] * Hp[o + j + 3] + P[wn + j + 3] * M[o + j + 3];
          }
          for (; j < H; j++) s0 += P[ws + j] * Hp[o + j] + P[wn + j] * M[o + j];
          Hn[o + i] = Math.tanh(s0 + s1 + s2 + s3);
        }
      }
    }
    const HT = Hs[T], R = w.R, RD = (a.holeRead ? 3 : 2) * H;
    for (let j = 0; j < H; j++) { let s = 0, mx = -Infinity, am = 0; for (let u = 0; u < n; u++) { const x = HT[u * H + j]; s += x; if (x > mx) { mx = x; am = u; } } R[j] = s / n; R[H + j] = mx; w.arg[j] = am; }
    if (a.holeRead) { const o = enc.hole * H; for (let j = 0; j < H; j++) R[2 * H + j] = HT[o + j]; }
    let v = P[off.bv];
    for (let i = 0; i < a.H1; i++) {
      let s = P[off.b1 + i]; const wo = off.W1 + i * RD; for (let j = 0; j < RD; j++) s += P[wo + j] * R[j];
      w.zpre[i] = s; w.Z[i] = s > 0 ? s : 0; v += P[off.wv + i] * w.Z[i];
    }
    const out = { v, n, enc, T, rec: true };
    if (need.policy) {
      const QI = H + a.H1;
      for (let u = 0; u < n; u++) {
        const qo = u * QI; for (let j = 0; j < H; j++) w.qin[qo + j] = HT[u * H + j]; for (let j = 0; j < a.H1; j++) w.qin[qo + H + j] = w.Z[j];
        for (let i = 0; i < a.Hq; i++) { let s = P[off.bq + i]; const wo = off.Wq + i * QI; for (let j = 0; j < QI; j++) s += P[wo + j] * w.qin[qo + j]; w.qpre[u * a.Hq + i] = s; }
        for (let b = 0; b < 4; b++) { let s = P[off.bo + b]; const wo = off.Wo + b * a.Hq; for (let i = 0; i < a.Hq; i++) { const q = w.qpre[u * a.Hq + i]; if (q > 0) s += P[wo + i] * q; } w.logit[u * 4 + b] = s; }
      }
      out.logits = w.logit; out.policy = true;
    }
    return out;
  }
  function recBackward(net, out, g, grad) {
    const P = net.params, a = net.arch, H = a.H, T = out.T, enc = out.enc, n = enc.n, G = enc.G, w = net.wsr, off = net.off, HT = w.Hs[T];
    const RD = (a.holeRead ? 3 : 2) * H, dZ = w.dZ, cur = w.cur;
    for (let i = 0; i < a.H1; i++) dZ[i] = 0;
    for (let i = 0; i < n * H; i++) cur[i] = 0;
    const gv = g.v || 0;
    if (gv) { grad[off.bv] += gv; for (let i = 0; i < a.H1; i++) { grad[off.wv + i] += gv * w.Z[i]; dZ[i] += gv * P[off.wv + i]; } }
    if (g.logits && out.policy) {
      const QI = H + a.H1, dq = w.dq, dqin = w.dqin;
      for (let u = 0; u < n; u++) {
        let any = false; for (let b = 0; b < 4; b++) if (g.logits[u * 4 + b]) any = true; if (!any) continue;
        for (let i = 0; i < a.Hq; i++) dq[i] = 0;
        for (let b = 0; b < 4; b++) {
          const gl = g.logits[u * 4 + b]; if (!gl) continue; grad[off.bo + b] += gl; const wo = off.Wo + b * a.Hq;
          for (let i = 0; i < a.Hq; i++) { const q = w.qpre[u * a.Hq + i]; if (q > 0) { grad[wo + i] += gl * q; dq[i] += gl * P[wo + i]; } }
        }
        for (let j = 0; j < QI; j++) dqin[j] = 0;
        const qo = u * QI;
        for (let i = 0; i < a.Hq; i++) {
          if (w.qpre[u * a.Hq + i] <= 0 || !dq[i]) continue; const d = dq[i], wo = off.Wq + i * QI; grad[off.bq + i] += d;
          for (let j = 0; j < QI; j++) { grad[wo + j] += d * w.qin[qo + j]; dqin[j] += d * P[wo + j]; }
        }
        for (let j = 0; j < H; j++) cur[u * H + j] += dqin[j];
        for (let j = 0; j < a.H1; j++) dZ[j] += dqin[H + j];
      }
    }
    const dR = w.dR; for (let j = 0; j < RD; j++) dR[j] = 0;
    for (let i = 0; i < a.H1; i++) {
      if (w.zpre[i] <= 0 || !dZ[i]) continue; const dz = dZ[i], wo = off.W1 + i * RD; grad[off.b1 + i] += dz;
      for (let j = 0; j < RD; j++) { grad[wo + j] += dz * w.R[j]; dR[j] += dz * P[wo + j]; }
    }
    for (let j = 0; j < H; j++) { const gm = dR[j] / n; for (let u = 0; u < n; u++) cur[u * H + j] += gm; cur[w.arg[j] * H + j] += dR[H + j]; }
    if (a.holeRead) { const o = enc.hole * H; for (let j = 0; j < H; j++) cur[o + j] += dR[2 * H + j]; }
    const dE = w.dE; for (let i = 0; i < n * H; i++) dE[i] = 0;
    let dh = cur, nx = w.nxt; const dM = w.dM, sc = 1 / 6;
    for (let t = T; t >= 1; t--) {
      const Ht = w.Hs[t], Hp = w.Hs[t - 1], M = w.Ms[t - 1];
      for (let i = 0; i < n * H; i++) { dh[i] *= 1 - Ht[i] * Ht[i]; dE[i] += dh[i]; nx[i] = 0; dM[i] = 0; }
      for (let i = 0; i < H; i++) {   // weight gradients: one output row at a time (inner loop writes one row)
        const ws = off.Ws + i * H, wn = off.Wn + i * H; let gb = 0;
        for (let u = 0; u < n; u++) {
          const da = dh[u * H + i]; if (!da) continue; gb += da; const o = u * H;
          for (let j = 0; j < H; j++) { grad[ws + j] += da * Hp[o + j]; grad[wn + j] += da * M[o + j]; }
        }
        grad[off.b + i] += gb;
      }
      for (let u = 0; u < n; u++) {   // input gradients: nx = da Ws, dM = da Wn
        const o = u * H;
        for (let i = 0; i < H; i++) {
          const da = dh[o + i]; if (!da) continue; const ws = off.Ws + i * H, wn = off.Wn + i * H;
          for (let j = 0; j < H; j++) { nx[o + j] += da * P[ws + j]; dM[o + j] += da * P[wn + j]; }
        }
      }
      for (let u = 0; u < n; u++) { const o = u * H; for (let q = G.adjStart[u], e = G.adjStart[u + 1]; q < e; q++) { const wo = G.adjList[q] * H; for (let j = 0; j < H; j++) nx[wo + j] += dM[o + j] * sc; } }
      const tmp = dh; dh = nx; nx = tmp;
    }
    const H0 = w.Hs[0], X = enc.X;
    for (let u = 0; u < n; u++) for (let i = 0; i < H; i++) {
      const o = u * H + i, da = dh[o] * (1 - H0[o] * H0[o]); grad[off.b0 + i] += da; const de = dE[o] + da;
      if (!de) continue; const xo = u * FR, wo = off.Wx + i * FR; for (let j = 0; j < FR; j++) if (X[xo + j]) grad[wo + j] += de * X[xo + j];
    }
  }

  // ---------------------------------------------------------------- whole state spaces, splices, hardness
  // every proper 4-colouring of G minus the hole up to renaming (each once: a colour may be used only after all lower
  // ones, in a BFS order from the rim), as Int8Arrays; null if more than cap
  function allColourings(G, cap) {
    cap = cap || 200000;
    const n = G.n, hole = G.hole, order = [], seen = new Uint8Array(n); seen[hole] = 1;
    const q = [G.rim[0]]; seen[G.rim[0]] = 1;
    for (let i = 0; i < q.length; i++) { const u = q[i]; order.push(u); for (let e = G.adjStart[u]; e < G.adjStart[u + 1]; e++) { const w = G.adjList[e]; if (!seen[w]) { seen[w] = 1; q.push(w); } } }
    const col = new Int8Array(n).fill(-1), out = []; let over = false;
    const rec = (i, mx) => {
      if (over) return;
      if (i === order.length) { out.push(col.slice()); if (out.length > cap) over = true; return; }
      const u = order[i], lim = Math.min(3, mx + 1);
      for (let c = 0; c <= lim; c++) {
        let ok = true; for (let e = G.adjStart[u]; e < G.adjStart[u + 1]; e++) if (col[G.adjList[e]] === c) { ok = false; break; }
        if (!ok) continue; col[u] = c; rec(i + 1, Math.max(mx, c)); col[u] = -1;
      }
    };
    rec(0, -1);
    return over ? null : out;
  }
  // the whole state space: { keys, cols, won, dist (Int16Array, -1 = no fill reachable) for every colouring,
  // unfilled (indices), hist {d: count} over unfilled states, unreachable }; afterstate distances on demand
  function stateSpace(G, cap) {
    const cols = allColourings(G, cap); if (!cols) return null;
    const N = cols.length, keys = new Array(N), idx = new Map(), wonA = new Uint8Array(N);
    for (let i = 0; i < N; i++) { keys[i] = canon(G, cols[i]); idx.set(keys[i], i); wonA[i] = won(G, cols[i]) ? 1 : 0; }
    const nbr = new Array(N), dist = new Int16Array(N).fill(-1), q = [];
    for (let i = 0; i < N; i++) {
      if (wonA[i]) { dist[i] = 0; q.push(i); nbr[i] = []; continue; }
      nbr[i] = movesOf(G, cols[i]).map(K => idx.get(canon(G, flip(cols[i], K))));
    }
    const back = Array.from({ length: N }, () => []);   // swaps are involutions, so the swap graph is undirected
    for (let i = 0; i < N; i++) if (!wonA[i]) for (const j of nbr[i]) back[j].push(i);
    for (let h = 0; h < q.length; h++) { const i = q[h]; for (const j of back[i]) if (dist[j] < 0 && !wonA[j]) { dist[j] = dist[i] + 1; q.push(j); } }
    const hist = {}, unfilled = []; let unreachable = 0;
    for (let i = 0; i < N; i++) { if (wonA[i]) continue; unfilled.push(i); if (dist[i] < 0) unreachable++; else hist[dist[i]] = (hist[dist[i]] || 0) + 1; }
    return { N, keys, cols, idx, won: wonA, dist, nbr, unfilled, hist, unreachable };
  }
  // a hardness score from the depth histogram: states 3+ swaps from a fill, each level deeper weighted 8x, divided by
  // the square root of (unfilled states / 1000) so a big map with many shallow-ish states does not win on size alone
  function hardness(hist) {
    let s = 0, maxD = 0, tot = 0; for (const d in hist) { const k = +d; tot += hist[d]; if (k > maxD) maxD = k; if (k >= 3) s += hist[d] * Math.pow(8, k - 3); }
    return { score: +(s / Math.sqrt(Math.max(1, tot) / 1000)).toFixed(2), maxD };
  }
  const graphSpec = G => ({ key: G.key, name: G.name, n: G.n, hole: G.hole, link: Array.from(G.rim), edges: G.edges, rot: G.rot });
  // mirror image: every rotation reversed, the rim's cyclic order reversed (same adjacency, same labels)
  function mirrorGraph(G) {
    return makeGraph({ key: G.key + '~m', name: G.name + ' (mirror)', n: G.n, hole: G.hole, link: Array.from(G.rim).reverse(), edges: G.edges, rot: G.rot ? G.rot.map(r => r.slice().reverse()) : null, dist: G.dist, distMap: G.distMap });
  }
  // a triangulation under edits (rotation system: nxt[a].get(b) = c is the face (a, b, c)); the hole keeps its 5 edges
  function Tri(G) {
    const nxt = []; for (let v = 0; v < G.n; v++) { const m = new Map(), r = G.rot[v]; for (let i = 0; i < r.length; i++) m.set(r[i], r[(i + 1) % r.length]); nxt.push(m); }
    this.nxt = nxt; this.hole = G.hole; this.alive = new Uint8Array(G.n).fill(1); this.key = G.key; this.name = G.name;
  }
  Tri.prototype.addF = function (a, b, c) { const x = this.nxt; x[a].set(b, c); x[b].set(c, a); x[c].set(a, b); };
  Tri.prototype.delF = function (a, b, c) { const x = this.nxt; x[a].delete(b); x[b].delete(c); x[c].delete(a); };
  Tri.prototype.nearHole = function (v) { return v === this.hole || this.nxt[this.hole].has(v); };
  Tri.prototype.rotOf = function (v) { const m = this.nxt[v], f = m.keys().next().value, out = [f]; let u = m.get(f); while (u !== f && out.length <= m.size) { out.push(u); u = m.get(u); } return out; };
  // an edge flip away from the hole (neither end nor either opposite corner is the hole)
  Tri.prototype.flip = function (a, b) {
    const x = this.nxt; if (a === this.hole || b === this.hole || !x[a].has(b)) return false;
    const c = x[a].get(b), d = x[b].get(a);
    if (c === this.hole || d === this.hole || c === d || x[c].has(d) || x[a].size <= 3 || x[b].size <= 3) return false;
    this.delF(a, b, c); this.delF(b, a, d); this.addF(c, a, d); this.addF(d, b, c); return true;
  };
  // insert a new vertex into the face (a, b, nxt[a][b]) when the hole is not a corner of it
  Tri.prototype.insert = function (a, b) {
    const x = this.nxt, c = x[a].get(b); if (c === undefined || a === this.hole || b === this.hole || c === this.hole) return false;
    const v = x.length; x.push(new Map()); this.alive = Uint8Array.from([...this.alive, 1]);
    this.delF(a, b, c); this.addF(a, b, v); this.addF(b, c, v); this.addF(c, a, v); return true;
  };
  // remove a degree-3 or degree-4 vertex not next to the hole, retriangulating its link (diag 0: w0-w2, 1: w1-w3)
  Tri.prototype.remove = function (v, diag) {
    const x = this.nxt; if (!this.alive[v] || this.nearHole(v)) return false;
    const deg = x[v].size; if (deg !== 3 && deg !== 4) return false;
    const w = this.rotOf(v);
    if (deg === 3) { if (w.some(u => x[u].size <= 3)) return false; for (let i = 0; i < 3; i++) this.delF(v, w[i], w[(i + 1) % 3]); this.addF(w[0], w[1], w[2]); }
    else {
      const s = diag ? 1 : 0, p = w[s], r = w[s + 2], o1 = w[s + 1], o2 = w[(s + 3) % 4];
      if (x[p].has(r) || x[o1].size <= 3 || x[o2].size <= 3) return false;
      for (let i = 0; i < 4; i++) this.delF(v, w[i], w[(i + 1) % 4]);
      this.addF(p, o1, r); this.addF(p, r, o2);
    }
    this.alive[v] = 0; return true;
  };
  Tri.prototype.toGraph = function (key) {
    const ids = [], map = new Int32Array(this.nxt.length).fill(-1);
    for (let v = 0; v < this.nxt.length; v++) if (this.alive[v]) { map[v] = ids.length; ids.push(v); }
    const rot = ids.map(v => this.rotOf(v).map(u => map[u])), n = ids.length, hole = map[this.hole];
    const edges = []; for (let u = 0; u < n; u++) for (const w of rot[u]) if (u < w) edges.push([u, w]);
    return makeGraph({ key: key || this.key, name: this.name, n, hole, link: rot[hole], edges, rot });
  };
  // one random splice (flip / insert / remove), up to `tries` attempts; returns the op name or null
  function spliceRandom(tri, R, tries) {
    for (let t = 0; t < (tries || 40); t++) {
      const live = []; for (let v = 0; v < tri.nxt.length; v++) if (tri.alive[v] && v !== tri.hole) live.push(v);
      const a = R.pick(live), nb = [...tri.nxt[a].keys()], b = R.pick(nb), op = R();
      if (op < 0.6) { if (tri.flip(a, b)) return 'flip'; }
      else if (op < 0.8) { if (live.length < 44 && tri.insert(a, b)) return 'insert'; }
      else { if (live.length > 14 && tri.remove(a, R() < 0.5 ? 1 : 0)) return 'remove'; }
    }
    return null;
  }

  // ---------------------------------------------------------------- agents
  // score every move of state s by -1 + gamma V(after) (V = 0 at a fill). Returns { enc, scores, afters, wins }.
  function scoreMoves(net, G, col, gamma, params, encOpt) {
    const enc = encode(G, col, encOpt), m = enc.moves.length, scores = new Float32Array(m), afters = new Array(m), wins = new Uint8Array(m);
    const after = encOpt && encOpt.kind === 'rec' ? Object.assign({}, encOpt, { noMoves: true }) : encOpt;
    for (let i = 0; i < m; i++) {
      const c = flip(col, enc.moves[i].K); afters[i] = c;
      if (won(G, c)) { scores[i] = -1; wins[i] = 1; }
      else scores[i] = -1 + gamma * netValue(net, encode(G, c, after), params);
    }
    return { enc, scores, afters, wins };
  }
  const argmax = a => { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; };
  // policy-head scores (one forward pass): logits
  function policyScores(net, G, col, encOpt) {
    const enc = encode(G, col, encOpt), out = netForward(net, enc, { policy: true });
    let scores;
    if (net.arch.kind === 'rec') {   // a move scores the best (region, colour) click that triggers it
      scores = new Float32Array(enc.moves.length).fill(-Infinity); const tab = actionTable(enc);
      for (let k = 0; k < tab.length; k++) { const mi = tab[k]; if (mi >= 0 && out.logits[k] > scores[mi]) scores[mi] = out.logits[k]; }
    } else scores = Float32Array.from(out.logits.subarray(0, enc.moves.length));
    const afters = enc.moves.map(mv => flip(col, mv.K)), wins = Uint8Array.from(afters.map(c => won(G, c) ? 1 : 0));
    return { enc, scores, afters, wins };
  }
  // baseline agents return a move index for (G, col)
  const baselines = {
    random(G, col, R) { const mv = movesOf(G, col); return mv.length ? R.int(mv.length) : -1; },
    // hand rule: a swap that fills the hole now if there is one, else a random chain that touches the rim, else random
    hand(G, col, R) {
      const mv = movesOf(G, col); if (!mv.length) return -1;
      for (let i = 0; i < mv.length; i++) if (won(G, flip(col, mv[i]))) return i;
      const rimMv = []; mv.forEach((K, i) => { if (K.rimMask) rimMv.push(i); });
      return rimMv.length ? R.pick(rimMv) : R.int(mv.length);
    },
    // two-ply count: fill now if possible, else the swap after which the most swaps fill (ties random)
    twoply(G, col, R) {
      const mv = movesOf(G, col); if (!mv.length) return -1;
      let best = -1, bc = [];
      for (let i = 0; i < mv.length; i++) {
        const c = flip(col, mv[i]); if (won(G, c)) return i;
        let k = 0; for (const K2 of movesOf(G, c)) if (won(G, flip(c, K2))) k++;
        if (k > best) { best = k; bc = [i]; } else if (k === best) bc.push(i);
      }
      return R.pick(bc);
    },
  };

  // ---------------------------------------------------------------- the trainer (actors + learner)
  function rlDefaults(r) {
    return Object.assign({ lr: 1e-3, gamma: 0.97, epsStart: 0.5, epsEnd: 0.05, epsDecay: 3000, batch: 32, replay: 20000, targetSync: 200,
      nstep: 2, backup: 'td', auxWeight: 0, episodeCap: 20, updateEvery: 4, wd: 0, clip: 5, explore: 'egreedy', temp: 0.3, startMix: 0.3, huber: 1 }, r || {});
  }
  // env: { G, key, levels, rooms (unfilled cols of the drawn maze), census (array of canon strings) }
  function makeEnv(GAME, key, opt) {
    opt = opt || {};
    const G = graphFromLevels(GAME, key); if (!G) return null;
    const levels = GAME.levels.filter(L => L.g === key);
    const starts = levels.map(L => { const c = Int8Array.from(L.start); c[G.hole] = -1; return c; });
    const census = G.dist ? Object.keys(G.dist) : [];
    return { G, key, name: G.name, levels, starts, census };
  }
  function envStart(env, R, mix) {
    if (R() < mix || !env.census.length) return env.starts[R.int(env.starts.length)].slice();
    return decodeCanon(env.G, env.census[R.int(env.census.length)]);
  }
  function Trainer(cfg) {
    cfg = cfg || {};
    const T = this;
    T.arch = archDefaults(cfg.arch); T.rl = rlDefaults(cfg.rl); T.seed = cfg.seed == null ? 1 : cfg.seed;
    T.R = rng(T.seed * 7919 + 13);
    T.net = cfg.net || netCreate(T.arch, T.seed);
    T.arch = T.net.arch;
    T.tgt = Float32Array.from(T.net.params);
    T.opt = adamCreate(T.net); T.grad = new Float32Array(T.net.size);
    T.replay = []; T.rp = 0;
    T.episodes = 0; T.steps = 0; T.updates = 0; T.loss = 0; T.auxLoss = 0; T.wall = 0;
    T.mode = 'value';   // or 'policy' (act with the chain head)
    T.recent = [];   // last 400 episode results {key, ok, len, d0}
  }
  Trainer.prototype.eps = function () { const r = this.rl, f = Math.min(1, this.episodes / Math.max(1, r.epsDecay)); return r.epsStart + (r.epsEnd - r.epsStart) * f; };
  Trainer.prototype.syncTarget = function () { this.tgt.set(this.net.params); };
  Trainer.prototype.choose = function (G, col, greedy) {
    const T = this, r = T.rl;
    const sc = T.mode === 'policy' ? policyScores(T.net, G, col, T.arch) : scoreMoves(T.net, G, col, r.gamma, null, T.arch);
    const m = sc.scores.length; if (!m) return { i: -1, sc };
    let i;
    if (greedy) i = argmax(sc.scores);
    else if (r.explore === 'softmax') {
      const t = Math.max(1e-3, r.temp * (0.2 + 0.8 * (1 - Math.min(1, T.episodes / Math.max(1, r.epsDecay)))));
      let mx = -Infinity; for (let k = 0; k < m; k++) mx = Math.max(mx, sc.scores[k]);
      let s = 0; const p = new Float64Array(m); for (let k = 0; k < m; k++) { p[k] = Math.exp((sc.scores[k] - mx) / t); s += p[k]; }
      let x = T.R() * s; i = m - 1; for (let k = 0; k < m; k++) { x -= p[k]; if (x <= 0) { i = k; break; } }
    } else i = T.R() < T.eps() ? T.R.int(m) : argmax(sc.scores);
    return { i, sc };
  };
  // one step of one actor (a mouse). actor = { env, col, len, d0, q (n-step queue), learn (bool), greedy (bool),
  // startFn (optional: R -> a start colouring; default: a level start with probability startMix, else a census state) }
  // returns { ended, ok, from, to } (cols)
  Trainer.prototype.actorStep = function (actor) {
    const T = this, r = T.rl, env = actor.env, G = env.G;
    if (!actor.col) {
      let c = null;
      for (let t = 0; t < 20 && (!c || won(G, c)); t++) c = actor.startFn ? actor.startFn(T.R) : envStart(env, T.R, r.startMix);
      actor.col = c; actor.len = 0; actor.q = []; actor.d0 = distOf(G, c);
    }
    const from = actor.col, { i, sc } = T.choose(G, from, actor.greedy);
    if (i < 0) { actor.col = null; T.recent.push({ key: env.key, ok: false, len: actor.len, d0: actor.d0, learn: actor.learn }); return { ended: true, ok: false, from, to: from }; }
    const to = sc.afters[i];
    actor.len++; if (actor.learn) T.steps++;
    const done = won(G, to), trunc = !done && actor.len >= r.episodeCap;
    if (actor.learn) {
      actor.q.push(from);
      const flush = (bootCol, isDone) => {   // emit every queued transition, each with its partial return
        while (actor.q.length) { const s0 = actor.q.shift(), k = actor.q.length + 1; T.push(G, s0, k, isDone ? null : bootCol); }
      };
      if (done) flush(null, true);
      else if (trunc) flush(to, false);
      else if (actor.q.length >= r.nstep) { const s0 = actor.q.shift(); T.push(G, s0, r.nstep, to); }
    }
    let ended = false, ok = false;
    if (done || trunc) {
      ended = true; ok = done;
      if (actor.learn) T.episodes++;
      T.recent.push({ key: env.key, ok, len: actor.len, d0: actor.d0, learn: actor.learn }); if (T.recent.length > 600) T.recent.shift();
      actor.col = null;
    } else actor.col = to;
    return { ended, ok, from, to, len: actor.len };
  };
  // replay entry: state s, k steps of reward -1, bootstrap state (null = terminal)
  Trainer.prototype.push = function (G, s, k, boot) {
    const T = this, e = { G, s, k, boot };
    if (T.replay.length < T.rl.replay) T.replay.push(e); else { T.replay[T.rp] = e; T.rp = (T.rp + 1) % T.rl.replay; }
  };
  Trainer.prototype.ret = function (k) { const g = this.rl.gamma; let s = 0, p = 1; for (let i = 0; i < k; i++) { s -= p; p *= g; } return { R: s, disc: p }; };
  Trainer.prototype.learn = function () {
    const T = this, r = T.rl, net = T.net;
    if (T.replay.length < Math.max(r.batch, 200)) return null;
    let L = 0, LA = 0;
    for (let b = 0; b < r.batch; b++) {
      const e = T.replay[T.R.int(T.replay.length)], G = e.G;
      let y;
      if (r.backup === 'max') {   // model-based greedy backup over every swap from s, with the target network
        const sc = scoreMoves(net, G, e.s, r.gamma, T.tgt, T.arch); y = sc.scores.length ? Math.max(...sc.scores) : -1;
      } else {
        const { R, disc } = T.ret(e.k);
        y = R + (e.boot ? disc * netValue(net, encode(G, e.boot, T.arch), T.tgt) : 0);
      }
      const enc = encode(G, e.s, T.arch), out = netForward(net, enc, null);
      const err = out.v - y, hub = r.huber, gv = Math.abs(err) <= hub ? err : hub * Math.sign(err);
      L += Math.abs(err) <= hub ? 0.5 * err * err : hub * (Math.abs(err) - 0.5 * hub);
      netBackward(net, out, { v: gv }, T.grad);
      if (r.auxWeight > 0) LA += auxSSL(T, G, e.s, r.auxWeight, T.grad);
    }
    adamStep(net, T.opt, T.grad, r.batch, { lr: r.lr, wd: r.wd, clip: r.clip });
    T.updates++; if (T.updates % r.targetSync === 0) T.syncTarget();
    T.loss = T.loss ? 0.98 * T.loss + 0.02 * L / r.batch : L / r.batch;
    if (r.auxWeight > 0) T.auxLoss = T.auxLoss ? 0.98 * T.auxLoss + 0.02 * LA / r.batch : LA / r.batch;
    return L / r.batch;
  };
  // self-supervised objectives (c) masked colour and (d) chain sizes on one state: one corrupted forward with
  // k regions' colours hidden and every chain feature dropped. Returns the loss (already weighted) and adds gradients.
  function auxSSL(T, G, col, weight, grad, wc, wdd, R) {
    if (T.net.arch.kind === 'rec') return 0;   // the self-supervised heads belong to the priors GNN
    R = R || T.R; wc = wc == null ? 1 : wc; wdd = wdd == null ? 1 : wdd;
    const n = G.n, mask = new Uint8Array(n), cand = []; for (let u = 0; u < n; u++) if (u !== G.hole) cand.push(u);
    const k = Math.max(1, Math.round(0.15 * cand.length)); shuffle(cand, R); for (let i = 0; i < k; i++) mask[cand[i]] = 1;
    const full = encode(G, col, T.arch), enc = encode(G, col, { mask, dropChains: true, chainFeats: T.arch.chainFeats });
    const out = netForward(T.net, enc, { nodes: true });
    const gCol = new Float32Array(n * 4), gCh = new Float32Array(n * 8);
    let L = 0;
    if (wc) for (let u = 0; u < n; u++) {
      if (!mask[u]) continue;
      const y = full.rel[col[u]], o = u * 4; let mx = -Infinity; for (let i = 0; i < 4; i++) mx = Math.max(mx, out.colL[o + i]);
      let s = 0; const p = [0, 0, 0, 0]; for (let i = 0; i < 4; i++) { p[i] = Math.exp(out.colL[o + i] - mx); s += p[i]; }
      L += wc * weight * -Math.log(Math.max(1e-9, p[y] / s)) / k;
      for (let i = 0; i < 4; i++) gCol[o + i] = wc * weight * (p[i] / s - (i === y ? 1 : 0)) / k;
    }
    if (wdd) {
      let cnt = 0; for (let u = 0; u < n; u++) if (u !== G.hole) cnt += 3;
      for (let u = 0; u < n; u++) {
        if (u === G.hole) continue; const own = full.rel[col[u]];
        for (let s = 0; s < 4; s++) {
          if (s === own) continue;
          const xo = u * F; const size = full.X[xo + 14 + s]; let hits = 0; for (let q = 0; q < 5; q++) hits += full.X[xo + 18 + s * 5 + q];
          const e1 = out.chP[u * 8 + s] - size, e2 = out.chP[u * 8 + 4 + s] - hits / 5;
          L += wdd * weight * 0.5 * (e1 * e1 + e2 * e2) / cnt * 4;
          gCh[u * 8 + s] = wdd * weight * e1 / cnt * 4; gCh[u * 8 + 4 + s] = wdd * weight * e2 / cnt * 4;
        }
      }
    }
    netBackward(T.net, out, { colL: gCol, chP: gCh }, grad);
    return L;
  }

  // ---------------------------------------------------------------- pretraining
  // Data: random maps (generateGraph + randomBlocked), with exact distances when the Kempe class is small (n <= exactMax,
  // <= cap states); or census states of given maps (source 'maps'). Objectives per sample, each with its own head:
  //   (a) distance-to-fill regression (dist head, Huber), (b) good-swap classification (policy head, BCE per chain:
  //   does this swap lower the exact distance by one), (c) masked colour (node head, CE), (d) chain size and rim reach
  //   with chain features dropped (node head, MSE).
  function pretrainDefaults(p) {
    return Object.assign({ steps: 2000, batch: 16, lr: 2e-3, nMin: 12, nMax: 18, nMaxEnd: 30, exactMax: 20, cap: 3000, wA: 1, wB: 1, wC: 0.5, wD: 0.5,
      source: 'generated', graphReuse: 24, seed: 7 }, p || {});
  }
  function Pretrainer(trainer, cfg, maps) {
    const P = this; P.T = trainer; P.cfg = pretrainDefaults(cfg); P.R = rng(P.cfg.seed * 31 + 5); P.maps = maps || [];
    P.step = 0; P.graphs = 0; P.samples = 0; P.pool = null; P.loss = { a: 0, b: 0, c: 0, d: 0 }; P.genMs = 0; P.labelled = 0; P.wall = 0;
    P.opt = adamCreate(trainer.net);
  }
  Pretrainer.prototype.nRange = function () {
    const c = this.cfg, f = Math.min(1, this.step / Math.max(1, c.steps * 0.7));
    return [c.nMin, Math.round(c.nMax + (c.nMaxEnd - c.nMax) * f)];
  };
  // a fresh data source: { G, states: [{col, d}] }
  Pretrainer.prototype.newSource = function () {
    const P = this, c = P.cfg, R = P.R, t0 = now();
    if (c.source === 'maps' || (c.source === 'both' && R() < 0.5)) {
      const env = R.pick(P.maps), states = [];
      for (let i = 0; i < 64; i++) { const k = R.pick(env.census); states.push({ col: decodeCanon(env.G, k), d: env.G.dist[k] }); }
      return { G: env.G, states, labelled: true };
    }
    const [lo, hi] = P.nRange();
    for (;;) {
      const n = lo + R.int(hi - lo + 1), G = generateGraph(n, R), col = randomBlocked(G, R);
      if (!col) continue;
      P.graphs++;
      let states = [{ col, d: null }], labelled = false;
      if (n <= c.exactMax) {
        const ex = exactDistances(G, col, c.cap);
        if (ex && ex.map.size) {
          G.distMap = ex.map; labelled = true;
          const keys = [...ex.map.keys()], hard = keys.filter(k => ex.map.get(k) >= 2);
          states = [];
          for (let i = 0; i < c.graphReuse; i++) { const k = (hard.length && R() < 0.5) ? R.pick(hard) : R.pick(keys); states.push({ col: decodeCanon(G, k), d: ex.map.get(k) }); }
        }
      }
      if (!labelled) {   // unlabelled: a short random walk gives more states for (c) and (d)
        let cur = col; states = [];
        for (let i = 0; i < c.graphReuse; i++) { states.push({ col: cur, d: null }); const mv = movesOf(G, cur); if (!mv.length) break; const nx = flip(cur, R.pick(mv)); if (won(G, nx)) continue; cur = nx; }
      }
      P.genMs += now() - t0; if (labelled) P.labelled++;
      return { G, states, labelled };
    }
  };
  Pretrainer.prototype.sample = function () {
    const P = this;
    if (!P.pool || !P.pool.length) P.pool = [];
    while (P.pool.length < 6) P.pool.push(P.newSource());
    const i = P.R.int(P.pool.length), src = P.pool[i], s = src.states.pop();
    if (!src.states.length) P.pool.splice(i, 1);
    return { G: src.G, col: s.col, d: s.d };
  };
  Pretrainer.prototype.trainStep = function () {
    const P = this, c = P.cfg, T = P.T, net = T.net, t0 = now();
    const acc = { a: 0, b: 0, c: 0, d: 0 };
    for (let bi = 0; bi < c.batch; bi++) {
      const smp = P.sample(), G = smp.G, col = smp.col; P.samples++;
      if (smp.d != null && (c.wA || c.wB)) {
        const enc = encode(G, col, T.arch), m = enc.moves.length, out = netForward(net, enc, { policy: c.wB > 0 && m > 0 });
        const g = { d: 0, logits: null };
        if (c.wA) { const e = out.d - smp.d, gd = Math.abs(e) <= 1 ? e : Math.sign(e); acc.a += Math.abs(e) <= 1 ? 0.5 * e * e : Math.abs(e) - 0.5; g.d = c.wA * gd; }
        if (c.wB && m) {
          const gl = new Float32Array(m); let L = 0;
          for (let k = 0; k < m; k++) {
            const after = flip(col, enc.moves[k].K), dd = distOf(G, after), y = dd != null && dd === smp.d - 1 ? 1 : 0;
            const z = out.logits[k], p = 1 / (1 + Math.exp(-z));
            L += -(y ? Math.log(Math.max(p, 1e-9)) : Math.log(Math.max(1 - p, 1e-9)));
            gl[k] = c.wB * (p - y) / m;
          }
          acc.b += L / m; g.logits = gl;
        }
        netBackward(net, out, g, T.grad);
      }
      if (c.wC || c.wD) acc.c += auxSSL(T, G, col, 1, T.grad, c.wC, c.wD, P.R);
    }
    adamStep(net, P.opt, T.grad, c.batch, { lr: c.lr * (P.step < 100 ? (P.step + 1) / 100 : 1), clip: 5 });
    P.step++;
    for (const k of ['a', 'b', 'c']) P.loss[k] = P.loss[k] ? 0.97 * P.loss[k] + 0.03 * acc[k] / c.batch : acc[k] / c.batch;
    P.wall += (now() - t0) / 1000;
  };
  // after pretraining, start the value head as minus the distance head (V ~ -d), then RL fine-tunes it
  Pretrainer.prototype.handover = function () {
    const T = this.T, net = T.net, o = net.off, a = net.arch;
    for (let i = 0; i < a.H1; i++) net.params[o.wv + i] = -net.params[o.wd + i];
    net.params[o.bv] = -net.params[o.bd];
    T.syncTarget();
  };
  Pretrainer.prototype.summary = function () {
    const c = this.cfg; return { steps: this.step, batch: c.batch, lr: c.lr, source: c.source, nMin: c.nMin, nMax: c.nMax, nMaxEnd: c.nMaxEnd, exactMax: c.exactMax,
      weights: { dist: c.wA, goodSwap: c.wB, maskedColour: c.wC, chain: c.wD }, graphs: this.graphs, labelledGraphs: this.labelled, samples: this.samples,
      wallSeconds: +this.wall.toFixed(1), finalLoss: { dist: +this.loss.a.toFixed(4), goodSwap: +this.loss.b.toFixed(4), ssl: +this.loss.c.toFixed(4) } };
  };

  // ---------------------------------------------------------------- the testbench
  // For each map: a fixed sample (seeded by the map key) of census states: up to maxDeep states with d >= 3 (all of them
  // on every level map), maxHard with d = 2 and maxEasy with d = 1; rollout starts = the map's level starts + K census
  // states. agree2 = share of d >= 2 states where the agent's swap lowers the exact distance by one (the main score:
  // at d = 1 any one-step lookahead is perfect), agree3 = the same on d >= 3 only (rare, deeper planning).
  function Testbench(GAME, keys, opt) {
    opt = Object.assign({ maxDeep: 40, maxHard: 60, maxEasy: 20, rollouts: 12, capMul: 3 }, opt || {});
    const TB = this; TB.opt = opt; TB.maps = [];
    const envs = keys.map(key => makeEnv(GAME, key)).concat(opt.extra || []);   // extra: [{ key, G (with dist), starts: [], census: [] }]
    envs.forEach(env => {
      if (!env) return; const key = env.key;
      const R = rng(hashStr('tb:' + key)), deep = [], hard = [], easy = [];
      env.census.forEach(k => { const d = env.G.dist[k]; (d >= 3 ? deep : d === 2 ? hard : easy).push(k); });
      shuffle(deep, R); shuffle(hard, R); shuffle(easy, R);
      const pick = deep.slice(0, opt.maxDeep).concat(hard.slice(0, opt.maxHard), easy.slice(0, opt.maxEasy)).map(k => ({ col: decodeCanon(env.G, k), d: env.G.dist[k] }));
      const starts = env.starts.map((c, i) => ({ col: c, d: distOf(env.G, c), level: true, id: env.levels && env.levels[i] ? env.levels[i].id : null }));
      const pool = shuffle(env.census.slice(), R).slice(0, opt.rollouts);
      pool.forEach(k => starts.push({ col: decodeCanon(env.G, k), d: env.G.dist[k], level: false }));
      TB.maps.push({ key, env, G: env.G, samples: pick, starts });
    });
  }
  // agent(G, col) -> move index into movesOf(G, col). Returns per-map and the move-level detail.
  Testbench.prototype.run = function (agent, keys) {
    const TB = this, res = {};
    for (const M of TB.maps) {
      if (keys && keys.indexOf(M.key) < 0) continue;
      const G = M.G; let a1 = 0, n1 = 0, a2 = 0, n2 = 0, a3 = 0, n3 = 0, a4 = 0, n4 = 0; const levels = {};
      for (const s of M.samples) {
        const mv = movesOf(G, s.col), i = agent(G, s.col);
        const ok = i >= 0 && distOf(G, flip(s.col, mv[i])) === s.d - 1;
        if (s.d >= 2) { n2++; a2 += ok; } else { n1++; a1 += ok; }
        if (s.d >= 3) { n3++; a3 += ok; }
        if (s.d >= 4) { n4++; a4 += ok; }
      }
      let succ = 0, ratio = 0, nr = 0, lvlOk = 0, lvlN = 0;
      for (const st of M.starts) {
        if (st.d == null || st.d < 1) continue;
        const cap = TB.opt.capMul * st.d; let c = st.col, k = 0, ok = false;
        while (k < cap) { const mv = movesOf(G, c), i = agent(G, c); if (i < 0) break; c = flip(c, mv[i]); k++; if (won(G, c)) { ok = true; break; } }
        nr++; if (ok) { succ++; ratio += k / st.d; }
        if (st.id) levels[st.id] = { par: st.d, steps: ok ? k : null };
        if (st.level) { lvlN++; lvlOk += ok; }
      }
      res[M.key] = { agree2: n2 ? a2 / n2 : null, n2, agree3: n3 ? a3 / n3 : null, n3, agree4: n4 ? a4 / n4 : null, n4, levels, agree1: n1 ? a1 / n1 : null, n1, agreeAll: (n1 + n2) ? (a1 + a2) / (n1 + n2) : null,
        success: nr ? succ / nr : null, stepsRatio: succ ? ratio / succ : null, levelSuccess: lvlN ? lvlOk / lvlN : null, rollouts: nr };
    }
    return res;
  };
  // averages over a key set (maps with no d >= 2 states do not count toward agree2; agree3 pools the d >= 3 states)
  function aggregate(perMap, keys) {
    const o = { agree2: 0, agree3: 0, agree4: 0, agreeAll: 0, success: 0, stepsRatio: 0, levelSuccess: 0, maps: 0 }; let n2 = 0, nr = 0, nl = 0, ns = 0, s3 = 0, c3 = 0, s4 = 0, c4 = 0;
    keys.forEach(k => { const r = perMap[k]; if (!r) return; o.maps++;
      if (r.agree2 != null) { o.agree2 += r.agree2; n2++; }
      if (r.agree3 != null) { s3 += r.agree3 * r.n3; c3 += r.n3; }
      if (r.agree4 != null) { s4 += r.agree4 * r.n4; c4 += r.n4; }
      if (r.agreeAll != null) o.agreeAll += r.agreeAll;
      if (r.success != null) { o.success += r.success; nr++; }
      if (r.stepsRatio != null) { o.stepsRatio += r.stepsRatio; ns++; }
      if (r.levelSuccess != null) { o.levelSuccess += r.levelSuccess; nl++; } });
    o.agree2 = n2 ? o.agree2 / n2 : null; o.agree3 = c3 ? s3 / c3 : null; o.n3 = c3; o.agree4 = c4 ? s4 / c4 : null; o.n4 = c4; o.agreeAll = o.maps ? o.agreeAll / o.maps : null; o.success = nr ? o.success / nr : null;
    o.stepsRatio = ns ? o.stepsRatio / ns : null; o.levelSuccess = nl ? o.levelSuccess / nl : null;
    for (const k in o) if (typeof o[k] === 'number' && k !== 'maps' && k !== 'n3' && k !== 'n4') o[k] = +o[k].toFixed(4);
    return o;
  }
  // deep level starts: filled within 1x and 3x par (greedy), from Testbench.run results
  const DEEP_STARTS = ['six-ring-trap', 'second-pit', 'deepest-pit', 'depth-5'];
  function deepStarts(perMap, ids) {
    ids = ids || DEEP_STARTS; const o = { ids, within1: 0, within3: 0, n: 0, detail: {} };
    for (const k in perMap) { const L = perMap[k].levels || {}; for (const id of ids) if (L[id]) { const r = L[id]; o.n++; o.detail[id] = r.steps; if (r.steps != null && r.steps <= r.par) o.within1++; if (r.steps != null && r.steps <= 3 * r.par) o.within3++; } }
    o.within1 = o.n ? +(o.within1 / o.n).toFixed(4) : null; o.within3 = o.n ? +(o.within3 / o.n).toFixed(4) : null; return o;
  }
  // a greedy agent from a network (value = afterstate argmax, policy = chain-head argmax)
  function netAgent(net, mode, gamma) {
    return (G, col) => { const sc = mode === 'policy' ? policyScores(net, G, col, net.arch) : scoreMoves(net, G, col, gamma == null ? 0.97 : gamma, null, net.arch); return sc.scores.length ? argmax(sc.scores) : -1; };
  }

  // ---------------------------------------------------------------- the drawn maze (panels)
  // the state graph from a level start: BFS over non-web swaps, won states are leaves; whole when <= whole states,
  // else cut at radius R and cap rooms (as in lab/maze.html)
  function stateGraph(G, start, opt) {
    opt = opt || {};
    const WHOLE = opt.whole || 160, CAP = opt.cap || 300, R0 = opt.R == null ? 4 : opt.R;
    function bfs(R, cap) {
      const nodes = [], idx = new Map(); let truncated = false;
      const add = (c, d) => { const key = canon(G, c); idx.set(key, nodes.length); nodes.push({ key, c, d, won: won(G, c) }); };
      add(start, 0);
      for (let qi = 0; qi < nodes.length; qi++) {
        const N = nodes[qi]; if (N.won || N.d >= R) continue;
        for (const K of movesOf(G, N.c)) { const c = flip(N.c, K), key = canon(G, c); if (idx.has(key)) continue; if (nodes.length >= cap) { truncated = true; continue; } add(c, N.d + 1); }
      }
      return { nodes, idx, truncated };
    }
    let res = bfs(Infinity, WHOLE); const whole = !res.truncated;
    if (!whole) res = bfs(R0, CAP);
    const { nodes, idx } = res, edges = [], nb = nodes.map(() => []), seen = new Set();
    nodes.forEach((N, i) => {
      if (N.won) return;
      for (const K of movesOf(G, N.c)) {
        const j = idx.get(canon(G, flip(N.c, K))); if (j === undefined || j === i) continue;
        const pk = i < j ? i * 100000 + j : j * 100000 + i; if (seen.has(pk)) continue; seen.add(pk);
        nb[i].push(j); nb[j].push(i); edges.push([i, j]);
      }
    });
    nodes.forEach(N => { N.dist = distOf(G, N.c); });
    return { nodes, idx, edges, nb, whole };
  }
  // a cheap seeded spring layout (BFS rings as the start), positions normalised to [0, 1]^2
  function layoutStateGraph(SG, seed) {
    const N = SG.nodes.length, R = rng(seed || 1), x = new Float64Array(N), y = new Float64Array(N);
    const byD = {}; SG.nodes.forEach((v, i) => { (byD[v.d] = byD[v.d] || []).push(i); });
    SG.nodes.forEach((v, i) => { const L = byD[v.d], k = L.indexOf(i), a = 2 * Math.PI * (k + R() * 0.6) / L.length; x[i] = Math.cos(a) * v.d; y[i] = Math.sin(a) * v.d; });
    const fx = new Float64Array(N), fy = new Float64Array(N), IT = N > 200 ? 160 : 240;
    for (let it = 0; it < IT; it++) {
      const temp = 0.4 * (1 - it / IT) + 0.02; fx.fill(0); fy.fill(0);
      for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
        let dx = x[i] - x[j], dy = y[i] - y[j], d2 = dx * dx + dy * dy; if (d2 < 1e-6) { dx = R() - 0.5; dy = R() - 0.5; d2 = 0.01; }
        if (d2 > 25) continue; const f = 0.6 / d2; fx[i] += dx * f; fy[i] += dy * f; fx[j] -= dx * f; fy[j] -= dy * f;
      }
      for (const [a, b] of SG.edges) { const dx = x[a] - x[b], dy = y[a] - y[b], d = Math.hypot(dx, dy) || 1e-3, f = (d - 1) * 0.5; fx[a] -= dx / d * f; fy[a] -= dy / d * f; fx[b] += dx / d * f; fy[b] += dy / d * f; }
      for (let i = 0; i < N; i++) { fx[i] -= x[i] * 0.02; fy[i] -= y[i] * 0.02; const l = Math.hypot(fx[i], fy[i]) || 1, s = Math.min(l, temp) / l; x[i] += fx[i] * s; y[i] += fy[i] * s; }
    }
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (let i = 0; i < N; i++) { x0 = Math.min(x0, x[i]); x1 = Math.max(x1, x[i]); y0 = Math.min(y0, y[i]); y1 = Math.max(y1, y[i]); }
    const s = Math.max(x1 - x0, y1 - y0) || 1, P = new Float32Array(2 * N);
    for (let i = 0; i < N; i++) { P[2 * i] = 0.5 + (x[i] - (x0 + x1) / 2) / s; P[2 * i + 1] = 0.5 + (y[i] - (y0 + y1) / 2) / s; }
    return P;
  }

  // ---------------------------------------------------------------- model files
  function b64FromBytes(u8) {
    if (typeof Buffer !== 'undefined') return Buffer.from(u8.buffer, u8.byteOffset, u8.byteLength).toString('base64');
    let s = ''; for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000)); return btoa(s);
  }
  function bytesFromB64(b) {
    if (typeof Buffer !== 'undefined') { const buf = Buffer.from(b, 'base64'); return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength); }
    const s = atob(b), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return u;
  }
  function modelToJSON(net, meta) {
    meta = meta || {};
    const f32 = Float32Array.from(net.params), a = net.arch;
    return Object.assign({ format: 'mazearena-model-1', name: meta.name || 'model', date: meta.date || new Date().toISOString(),
      arch: a.kind === 'rec' ? { kind: 'rec', H: a.H, T: a.T, H1: a.H1, Hq: a.Hq, F: a.F, holeRead: a.holeRead, readout: a.readout, heads: a.heads, feat: a.feat, track: 'tabula rasa' }
        : { kind: 'gnn', H: a.H, K: a.K, H1: a.H1, Hp: a.Hp, F: a.F, GD: a.GD, CF: a.CF, chainFeats: a.chainFeats, readout: a.readout, heads: a.heads, feat: a.feat, track: 'priors' },
      paramCount: net.size, rl: meta.rl || null, training: meta.training || null, metrics: meta.metrics || null },
      meta.extra || {}, { weights: { dtype: 'float32', shapes: net.shapes.map(([nm, r, c]) => [nm, r, c]), b64: b64FromBytes(new Uint8Array(f32.buffer)) } });
  }
  function modelFromJSON(M) {
    if (!M || M.format !== 'mazearena-model-1') throw new Error('not a mazearena-model-1 file');
    const net = netCreate(M.arch, 0), bytes = bytesFromB64(M.weights.b64);
    const al = new Uint8Array(bytes.length); al.set(bytes);
    const f = new Float32Array(al.buffer);
    if (f.length !== net.size) throw new Error('weights have ' + f.length + ' values, arch needs ' + net.size);
    net.params.set(f); return net;
  }

  const ARENA = { rng, hashStr, shuffle, now, makeGraph, graphFromLevels, levelMaps, PAIRS, rimCount, won, canon, decodeCanon, chainsOf, movesOf, flip,
    distOf, exactDistances, generateGraph, checkTriangulation, randomBlocked, relClasses, encode, F, GD, CF, archDefaults, shapesOf, netCreate,
    paramCount, netForward, recDefaults, recShapes, encodeRaw, actionTable, policyLoss, netShare, DEEP_STARTS, deepStarts,
    allColourings, stateSpace, Tri, mirrorGraph, spliceRandom, hardness, graphSpec, netBackward, netValue, adamCreate, adamStep, scoreMoves, policyScores, argmax, baselines, rlDefaults, makeEnv,
    envStart, Trainer, auxSSL, pretrainDefaults, Pretrainer, Testbench, aggregate, netAgent, stateGraph, layoutStateGraph, modelToJSON, modelFromJSON };
  if (typeof module !== 'undefined' && module.exports) module.exports = ARENA;
  root.ARENA = ARENA;
})(typeof self !== 'undefined' ? self : typeof window !== 'undefined' ? window : globalThis);
