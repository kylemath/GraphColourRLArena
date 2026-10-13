#!/usr/bin/env python3
"""Analysis for INTERPRETABILITY-1 (exploratory, post hoc). Reads the output of tools/arena_probe.js and writes
figures (SVG) and a results JSON. numpy + matplotlib only.

    python3 tools/arena_probe_analysis.py --data <probe out dir> --out docs/interp1

Sections: behaviour by depth; theory features by depth; feature-only depth model; state-level linear probes
(leave-one-family-out: a census map and its splice variants form one family) with shuffled-label, random-init and
input-feature controls; per-unit correlation / mutual information; node-level probes; error analysis (named-feature
model of the errors, then k-means on the activations of the errors that model does not explain); candidate variables
computed from the colouring (lock thickness, theta overlap, rerouting cuts).
"""
import argparse, json, math, os, collections
import os as _os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'): _os.environ.setdefault(_v, '4')
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--alpha', type=float, default=30.0)
ap.add_argument('--seed', type=int, default=1)
args = ap.parse_args()
D, OUT = args.data, args.out
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(args.seed)
RES = {}

R = json.load(open(os.path.join(D, 'states.json')))
MV = json.load(open(os.path.join(D, 'moves.json')))
MAPS = {m['key']: m for m in json.load(open(os.path.join(D, 'maps.json')))}
NODE = json.load(open(os.path.join(D, 'node_labels.json')))
N = len(R)
d = np.array([r['d'] for r in R])
DL = np.array([r['DL'] for r in R]) == 1
fam = np.array([r.get('parent') or r['map'] for r in R])
grp = np.array(['train' if r['split'] == 'train' else 'test' if r['split'] == 'test' else 'splice' for r in R])
PC = ['al-mu', 'al-A', 'al-B', 'mu-A', 'mu-B', 'A-B']
def col(f, default=np.nan):
    return np.array([default if r.get(f) is None else r[f] for r in R], float)

# ------------------------------------------------------------------ graph-derived candidate variables
def graph_of(key):
    m = MAPS[key]; n = m['n']; adj = [[] for _ in range(n)]
    for a, b in m['edges']: adj[a].append(b); adj[b].append(a)
    return n, m['hole'], m['link'], adj
GRAPHS = {k: graph_of(k) for k in MAPS}
def decode(key, s):
    n, hole, link, adj = GRAPHS[key]; c = [-1] * n; i = 0
    for u in range(n):
        if u == hole: continue
        c[u] = ord(s[i]) - 48; i += 1
    return c
def component(adj, c, hole, a, b, seed):
    if c[seed] not in (a, b): return set()
    seen = {seed}; st = [seed]
    while st:
        u = st.pop()
        for w in adj[u]:
            if w != hole and w not in seen and c[w] in (a, b): seen.add(w); st.append(w)
    return seen
def local_conn(adj, allowed, s, t, cap=4):
    """internally vertex-disjoint s-t paths inside the vertex set `allowed` (unit capacities, node splitting)"""
    if s not in allowed or t not in allowed: return 0
    if t in adj[s]: direct = 1
    else: direct = 0
    # flow network on split nodes: v_in = 2v, v_out = 2v+1
    capm = collections.defaultdict(int); nb = collections.defaultdict(set)
    def add(x, y, k):
        capm[(x, y)] += k; nb[x].add(y); nb[y].add(x)
    for v in allowed:
        add(2 * v, 2 * v + 1, 10 ** 6 if v in (s, t) else 1)
        for w in adj[v]:
            if w in allowed and not (v == s and w == t) and not (v == t and w == s): add(2 * v + 1, 2 * w, 1)
    src, snk, flow = 2 * s + 1, 2 * t, 0
    while flow < cap:
        prev = {src: None}; q = [src]
        while q and snk not in prev:
            x = q.pop(0)
            for y in nb[x]:
                if y not in prev and capm[(x, y)] > 0: prev[y] = x; q.append(y)
        if snk not in prev: break
        y = snk
        while prev[y] is not None:
            x = prev[y]; capm[(x, y)] -= 1; capm[(y, x)] += 1; y = x
        flow += 1
    return flow + direct
def bfs_len(adj, allowed, s, t):
    dist = {s: 0}; q = [s]
    for x in q:
        if x == t: return dist[x]
        for y in adj[x]:
            if y in allowed and y not in dist: dist[y] = dist[x] + 1; q.append(y)
    return -1

thick1 = np.zeros(N); thick2 = np.zeros(N); theta = np.zeros(N); path1 = np.zeros(N); path2 = np.zeros(N)
for q, r in enumerate(R):
    if not r['DL']: continue
    n, hole, link, adj = GRAPHS[r['map']]; c = decode(r['map'], r['key']); j = r['j']
    x = lambda k: link[(j + k) % 5]
    al, mu, cA, cB = c[x(0)], c[x(1)], c[x(3)], c[x(4)]
    K1 = component(adj, c, hole, mu, cA, x(1)); K2 = component(adj, c, hole, mu, cB, x(1))
    thick1[q] = local_conn(adj, K1, x(1), x(3)); thick2[q] = local_conn(adj, K2, x(1), x(4))
    theta[q] = len(K1 & K2)
    path1[q] = bfs_len(adj, K1, x(1), x(3)); path2[q] = bfs_len(adj, K2, x(1), x(4))
# move-graph candidates
nOptDistinct = np.zeros(N); reroute = np.zeros(N); cutFrac = np.full(N, np.nan); exitsInterior = np.zeros(N); nAfter = np.zeros(N)
for q, r in enumerate(R):
    mv = MV[q]; byk = {}
    for m in mv:
        k = m[7]; e = byk.setdefault(k, [m[0], 0, 0, 0]); e[1] |= (m[3] != 0); e[2] |= m[6]; e[3] |= m[4]
    nAfter[q] = len(byk)
    opt = [k for k, e in byk.items() if e[0] == r['d'] - 1]; nOptDistinct[q] = len(opt)
    if r['DL']:
        cuts = [e for e in byk.values() if e[2]]
        reroute[q] = sum(1 for e in cuts if e[0] >= 2)
        cutFrac[q] = (sum(1 for e in cuts if e[0] <= 1) / len(cuts)) if cuts else np.nan
        exitsInterior[q] = 1 if (opt and all(not byk[k][1] for k in opt)) else 0
CAND = {'thick1': thick1, 'thick2': thick2, 'thetaShared': theta, 'path1': path1, 'path2': path2,
        'nOptDistinct': nOptDistinct, 'rerouteCuts': reroute, 'unlockingCutFrac': cutFrac, 'exitsAllInterior': exitsInterior, 'nAfter': nAfter}

# ------------------------------------------------------------------ helpers
def auc(score, y):
    y = np.asarray(y).astype(bool); s = np.asarray(score, float)
    if y.all() or (~y).all(): return float('nan')
    order = np.argsort(s, kind='mergesort'); ranks = np.empty(len(s)); sv = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sv[j + 1] == sv[i]: j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    npos = y.sum(); nneg = len(y) - npos
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))
def r2(pred, y):
    y = np.asarray(y, float); return float(1 - ((pred - y) ** 2).sum() / max(1e-12, ((y - y.mean()) ** 2).sum()))
def lofo_fit(X, Y, groups, alpha=None):
    """leave-one-family-out ridge with an unpenalised intercept, for several targets at once (Y: n x t).
    X is standardised once on the whole subset; per-fold Gram matrices are cached, so each fold is one solve."""
    alpha = args.alpha if alpha is None else alpha
    X = np.asarray(X, np.float64); Y = np.asarray(Y, np.float64)
    if Y.ndim == 1: Y = Y[:, None]
    mu = X.mean(0); sd = X.std(0); sd[sd < 1e-8] = 1
    Z = np.concatenate([(X - mu) / sd, np.ones((len(X), 1))], 1)
    gs = np.unique(groups); Gf = {}; Bf = {}
    for g in gs:
        m = groups == g; Gf[g] = Z[m].T @ Z[m]; Bf[g] = Z[m].T @ Y[m]
    Gt = sum(Gf.values()); Bt = sum(Bf.values()); P = alpha * np.eye(Z.shape[1]); P[-1, -1] = 0
    pred = np.full(Y.shape, np.nan)
    for g in gs:
        m = groups == g
        if (~m).sum() < 10: continue
        W = np.linalg.solve(Gt - Gf[g] + P + 1e-9 * np.eye(Z.shape[1]), Bt - Bf[g])
        pred[m] = Z[m] @ W
    return pred
def ridge_lofo(X, y, groups, alpha=None, binary=False):
    return lofo_fit(X, y, groups, alpha)[:, 0]
def metric(pred, y, binary):
    ok = ~np.isnan(pred)
    return auc(pred[ok], y[ok]) if binary else r2(pred[ok], y[ok])
def mi_bits(x, y, bins=10):
    xq = np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1]); xb = np.searchsorted(xq, x)
    yv = np.unique(y)
    if len(yv) > bins: yq = np.quantile(y, np.linspace(0, 1, bins + 1)[1:-1]); yb = np.searchsorted(yq, y)
    else: yb = np.searchsorted(yv, y)
    joint = np.zeros((xb.max() + 1, yb.max() + 1)); np.add.at(joint, (xb, yb), 1); p = joint / joint.sum()
    px = p.sum(1, keepdims=True); py = p.sum(0, keepdims=True); nz = p > 0
    return float((p[nz] * np.log2(p[nz] / (px @ py)[nz])).sum())
def kmeans(X, k, iters=100, seed=0):
    r = np.random.default_rng(seed); C = X[r.choice(len(X), k, replace=False)]
    for _ in range(iters):
        lab = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
        C2 = np.array([X[lab == i].mean(0) if (lab == i).any() else C[i] for i in range(k)])
        if np.allclose(C2, C): break
        C = C2
    return lab, C

# ------------------------------------------------------------------ models
def load_model(tag):
    M = json.load(open(os.path.join(D, 'model_' + tag + '.json')))
    if M.get('input'): return M
    g = M['gamma']; v = np.array(M['val'])
    M['dhat'] = -v if g == 1 else np.log(np.clip(1 + (1 - g) * v, 1e-9, None)) / math.log(g)
    M['afterD'] = np.array([MV[q][M['choice'][q]][0] for q in range(N)])
    M['polAfterD'] = np.array([MV[q][M['polChoice'][q]][0] if MV[q] else -9 for q in range(N)])
    M['steps'] = np.array(M['steps'])
    return M
MODEL_TAGS = [f[6:-5] for f in sorted(os.listdir(D)) if f.startswith('model_') and f.endswith('.json')]
MODELS = {t: load_model(t) for t in MODEL_TAGS}
behav_tags = [t for t in MODEL_TAGS if not MODELS[t].get('input')]

# ------------------------------------------------------------------ 1. behaviour
def dbin(x): return np.minimum(x, 4)
behaviour = {}
piAfterD = col('piNextD'); piSteps = col('piToFill')
for t in behav_tags + ['pi-agent']:
    out = {}
    for g in ['train', 'test', 'splice']:
        for k in [1, 2, 3, 4]:
            m = (grp == g) & (dbin(d) == k)
            if not m.any(): continue
            if t == 'pi-agent':
                agree = (piAfterD[m] == d[m] - 1).mean(); st = piSteps[m]; valerr = float('nan')
            else:
                M = MODELS[t]; agree = (M['afterD'][m] == d[m] - 1).mean(); st = M['steps'][m]
                valerr = (np.abs(M['dhat'][m] - d[m]) >= 0.5).mean()
            fail = (st < 0).mean(); ok = st >= 0
            out[f'{g}:{k}'] = {'n': int(m.sum()), 'agree': round(float(agree), 4), 'valErr': None if np.isnan(valerr) else round(float(valerr), 4),
                               'fail': round(float(fail), 4), 'gap': round(float((st[ok] - d[m][ok]).mean()) if ok.any() else float('nan'), 4),
                               'gapMax': int((st[ok] - d[m][ok]).max()) if ok.any() else None}
    behaviour[t] = out
RES['behaviour'] = behaviour
# gap distribution for the main model at d >= 3 (held-out: test + splice)
main = 'priors-gnn-h16-k3-pool'
RES['gapHist'] = {}
for t in [main, 'nano-rl', 'colours-only-pretrained', 'tr-rec-h32-t16', 'pi-agent']:
    st = piSteps if t == 'pi-agent' else MODELS[t]['steps']
    h = {}
    for k in [1, 2, 3, 4, 5, 6]:
        m = (grp != 'train') & (d == k)
        if not m.any(): continue
        g = np.where(st[m] < 0, -1, st[m] - d[m]); h[k] = dict(collections.Counter(int(x) for x in g))
    RES['gapHist'][t] = h

# ------------------------------------------------------------------ 2. theory features by depth (all sets)
feat_by_depth = {}
for k in [1, 2, 3, 4, 5, 6]:
    m = d == k
    if not m.any(): continue
    e = {'n': int(m.sum()), 'DL': float(DL[m].mean())}
    if k >= 2:
        for f in ['piAllDL', 'DD', 'sigDL', 'sigLockless', 'piMoveOpt']: e[f] = round(float(col(f)[m].mean()), 4)
        for f in ['dlRun', 'lenL1', 'lenL2', 'lenSigma', 'lenAB', 'classN', 'nMoves', 'nOpt']: e[f] = round(float(np.median(col(f)[m])), 2)
        pt = col('piToFill')[m]; e['piToFill_eq_d'] = round(float((pt == k).mean()), 4); e['piNoFill'] = round(float((pt < 0).mean()), 4)
        for f in ['thick1', 'thick2', 'thetaShared', 'rerouteCuts']: e[f] = round(float(np.median(CAND[f][m])), 2)
        e['ell'] = dict(collections.Counter(str(R[q]['ell']) for q in np.where(m)[0]))
    feat_by_depth[k] = e
RES['featByDepth'] = feat_by_depth
# deep share by Gamma membership and DL run (DL states)
RES['deepByGamma'] = {str(g): {'n': int((DL & (col('piAllDL') == g)).sum()), 'deep': float((d[DL & (col('piAllDL') == g)] >= 3).mean())} for g in [0, 1]}
RES['deepByRun'] = {}
run = col('dlRun')
for lo, hi in [(1, 1), (2, 2), (3, 4), (5, 9), (10, 10 ** 6)]:
    m = DL & (run >= lo) & (run <= hi)
    if m.any(): RES['deepByRun'][f'{lo}-{hi}'] = {'n': int(m.sum()), 'deep': round(float((d[m] >= 3).mean()), 4)}
# rim pattern by depth
RES['rimPattern'] = {f'{p}': dict(collections.Counter(int(min(R[q]['d'], 3)) for q in range(N) if R[q]['rimPattern'] == p)) for p in sorted(set(r['rimPattern'] for r in R))}
# map-level Gamma cycles
RES['mapsGamma'] = {k: {'allDL': v['allDL'], 'posW': v['posW'], 'hist': v['hist'], 'degWord': v['degWord'], 'set': v['set']} for k, v in MAPS.items() if v['set'] == 'census'}
# ell vs kappa
RES['ellKappa'] = dict(collections.Counter(f"k{R[q]['d']}:l{R[q]['ell']}" for q in range(N) if R[q]['d'] >= 3))

# ------------------------------------------------------------------ 3. named-feature models of depth (DL states)
NAMED = ['lenL1', 'lenL2', 'lenSigma', 'lenAB', 'lenAlA', 'lenAlA2', 'lenAlB', 'lenAlB2', 'classN', 'classF', 'piL', 'piF', 'piW', 'piAllDL',
         'dlRun', 'DD', 'sigDL', 'sigLockless', 'nMoves', 'nWebs', 'nRimMoves', 'maxChain', 'meanChain', 'n']
def named_matrix(extra=()):
    cols = [np.log1p(np.nan_to_num(col(f), nan=0)) if f in ('classN', 'classF', 'piL', 'piF', 'dlRun', 'lenL1', 'lenL2', 'lenSigma', 'lenAB', 'lenAlA', 'lenAlA2', 'lenAlB', 'lenAlB2') else np.nan_to_num(col(f), nan=0) for f in NAMED]
    pt = col('piToFill'); cols.append(np.where(pt < 0, 12, np.minimum(pt, 12)))
    jj = col('j'); cols += [(jj == k).astype(float) for k in range(5)]
    for e in extra: cols.append(np.nan_to_num(CAND[e], nan=0))
    return np.stack(cols, 1)
XN = named_matrix(); XC = named_matrix(['thick1', 'thick2', 'thetaShared', 'path1', 'path2', 'rerouteCuts', 'unlockingCutFrac', 'exitsAllInterior'])
deep = (d >= 3).astype(float)
STATIC = ['lenL1', 'lenL2', 'lenSigma', 'lenAB', 'lenAlA', 'lenAlA2', 'lenAlB', 'lenAlB2', 'nMoves', 'nWebs', 'nRimMoves', 'maxChain', 'meanChain', 'n']
XS = np.stack([np.log1p(col(f)) if f.startswith('len') else col(f) for f in STATIC] + [(col('j') == k).astype(float) for k in range(5)]
              + [CAND[e] for e in ['thick1', 'thick2', 'thetaShared', 'path1', 'path2']], 1)
XPI = np.stack([np.log1p(col('piL')), np.log1p(col('piF')), col('piW'), col('piAllDL'), np.log1p(col('dlRun')), np.where(col('piToFill') < 0, 12, np.minimum(col('piToFill'), 12)),
                col('DD'), np.nan_to_num(col('sigDL')), np.nan_to_num(col('runBound'), nan=12), np.where(col('runFwd') < 0, 12, col('runFwd')), np.where(col('runBwd') < 0, 12, col('runBwd')), np.nan_to_num(col('sigLockless')), np.log1p(col('classN')), np.log1p(col('classF'))], 1)
RES['depthFromNamed'] = {'AUC deep|DL (static colouring features: chain lengths, moves, lock thickness, theta; no swaps applied)': metric(ridge_lofo(XS[DL], deep[DL], fam[DL]), deep[DL], True),
                         'AUC deep|DL (pi/sigma/class features only; these apply specific swaps)': metric(ridge_lofo(XPI[DL], deep[DL], fam[DL]), deep[DL], True),
                         'R2 kappa|DL (static)': metric(ridge_lofo(XS[DL], d[DL].astype(float), fam[DL]), d[DL].astype(float), False),
                         'R2 kappa|DL (pi/sigma/class)': metric(ridge_lofo(XPI[DL], d[DL].astype(float), fam[DL]), d[DL].astype(float), False),
                         'R2 kappa|DL (named all)': metric(ridge_lofo(XC[DL], d[DL].astype(float), fam[DL]), d[DL].astype(float), False),
'AUC deep|DL (named theory features, no lookahead)': metric(ridge_lofo(XN[DL], deep[DL], fam[DL]), deep[DL], True),
                         'AUC deep|DL (+ candidate graph variables)': metric(ridge_lofo(XC[DL], deep[DL], fam[DL]), deep[DL], True),
                         'AUC deep|DL (dlRun alone)': auc(col('dlRun')[DL], deep[DL]),
                         'AUC deep|DL (piToFill alone, no-fill=inf)': auc(np.where(col('piToFill')[DL] < 0, 99, col('piToFill')[DL]), deep[DL])}
for f in ['thick1', 'thick2', 'thetaShared', 'rerouteCuts', 'path1', 'path2']:
    RES['depthFromNamed'][f'AUC deep|DL ({f} alone)'] = auc(CAND[f][DL], deep[DL])

# ------------------------------------------------------------------ 4. state-level probes
ACT_TAGS = [t for t in MODEL_TAGS if os.path.exists(os.path.join(D, 'acts_' + t + '.f32'))]
def acts(t):
    M = MODELS[t]; A = np.fromfile(os.path.join(D, 'acts_' + t + '.f32'), dtype=np.float32)
    if M.get('input'): return A.reshape(N, M['D']), [('input', 0, M['D'])], M['H']
    H = M['arch']['H']; Ls = M['layers']; Dd = A.size // N; A = A.reshape(N, Dd); blocks = []
    for i, L in enumerate(Ls): blocks.append((('t' if M['arch']['kind'] == 'rec' else 'L') + str(L), i * 8 * H, (i + 1) * 8 * H))
    blocks.append(('z', len(Ls) * 8 * H, Dd))
    return A, blocks, H
log1p = np.log1p
pt = col('piToFill')
TARGETS = [  # name, values, subset mask, binary
    ('DL (d>=2)', DL.astype(float), np.ones(N, bool), True),
    ('lock1', col('lock1'), np.ones(N, bool), True),
    ('lock2', col('lock2'), np.ones(N, bool), True),
    ('kappa (clip 4)', np.minimum(d, 4).astype(float), np.ones(N, bool), False),
    ('log class size', log1p(col('classN')), np.ones(N, bool), False),
    ('n moves', col('nMoves'), np.ones(N, bool), False),
    ('deep (d>=3) | DL', deep, DL, True),
    ('Gamma-cycle | DL', col('piAllDL'), DL, True),
    ('DD | DL', col('DD'), DL, True),
    ('sigma image DL | DL', col('sigDL'), DL, True),
    ('pi move optimal | DL', col('piMoveOpt'), DL, True),
    ('log DL run | DL', log1p(col('dlRun')), DL, False),
    ('pi steps to fill (cap 12) | DL', np.where(pt < 0, 12, np.minimum(pt, 12)), DL, False),
    ('log lock1 length | DL', log1p(col('lenL1')), DL, False),
    ('log lock2 length | DL', log1p(col('lenL2')), DL, False),
    ('log sigma-chain length | DL', log1p(col('lenSigma')), DL, False),
    ('n exits (optimal afterstates) | DL', np.log1p(nOptDistinct), DL, False),
    ('lock1 thickness | DL', thick1, DL, False),
    ('theta overlap | DL', log1p(theta), DL, False),
    ('reroute cuts | DL', log1p(reroute), DL, False),
]
probe = {}
MASKS = {}
for i, (tn, y, mask, binary) in enumerate(TARGETS): MASKS.setdefault(mask.tobytes(), (mask, []))[1].append(i)
for t in ACT_TAGS:
    A, blocks, H = acts(t); probe[t] = {}
    for (bname, lo, hi) in blocks:
        X = A[:, lo:hi]; probe[t][bname] = {}
        for key, (mask, idxs) in MASKS.items():
            Y = np.stack([TARGETS[i][1][mask] for i in idxs], 1)
            P_ = lofo_fit(X[mask], Y, fam[mask])
            for c, i in enumerate(idxs):
                tn, y, _, binary = TARGETS[i]; probe[t][bname][tn] = round(metric(P_[:, c], y[mask], binary), 4)
    print('probed', t, flush=True)
# shuffled-label control on the main model's last message-passing layer and z
ctrl = {}
A, blocks, H = acts(main)
for (bname, lo, hi) in blocks[-2:]:
    X = A[:, lo:hi].astype(np.float64); ctrl[bname] = {}
    for (tn, y, mask, binary) in TARGETS:
        ys = y[mask].copy(); rng.shuffle(ys)
        ctrl[bname][tn] = round(metric(ridge_lofo(X[mask], ys, fam[mask], binary=binary), ys, binary), 4)
RES['probe'] = probe; RES['probeShuffled'] = {main: ctrl}
# named-feature baseline for the same targets (what a probe gets from the theory features themselves)
RES['probeNamedBaseline'] = {}
for (tn, y, mask, binary) in TARGETS:
    if tn.startswith(('deep', 'n exits', 'kappa')):
        RES['probeNamedBaseline'][tn] = round(metric(ridge_lofo(XN[mask], y[mask], fam[mask]), y[mask], binary), 4)

# ------------------------------------------------------------------ 5. per-unit correlation / MI (main model and colours-only)
units = {}
for t in [main, 'colours-only-pretrained', 'nano-rl', 'tr-rec-h32-t16']:
    if t not in ACT_TAGS: continue
    A, blocks, H = acts(t); out = {}
    names = []
    for (bname, lo, hi) in blocks:
        if bname == 'z': names += [f'z.{i}' for i in range(hi - lo)]
        else:
            for blk in ['mean', 'max', 'hole', 'rim0', 'rim1', 'rim2', 'rim3', 'rim4']: names += [f'{bname}.{blk}.{i}' for i in range(H)]
    for (tn, y, mask, binary) in TARGETS[:12]:
        X = A[mask]; yy = y[mask]; sd = X.std(0); ok = sd > 1e-6
        r = np.zeros(X.shape[1]); r[ok] = ((X[:, ok] - X[:, ok].mean(0)) * (yy - yy.mean())[:, None]).mean(0) / (sd[ok] * yy.std())
        top = np.argsort(-np.abs(r))[:5]
        out[tn] = [{'unit': names[i], 'r': round(float(r[i]), 3), 'MI_bits': round(mi_bits(X[:, i], yy), 4)} for i in top]
        # best |r| per block
        out[tn + ' :: max|r| by layer'] = {b: round(float(np.abs(r[lo:hi]).max()), 3) for (b, lo, hi) in blocks}
    units[t] = out
RES['units'] = units

# ------------------------------------------------------------------ 6. node-level probes
ncols = NODE['cols']; NL = np.array(NODE['rows'], float); sample = NODE['sample']
nq = NL[:, ncols.index('q')].astype(int); nfam = fam[nq]
isHole = NL[:, ncols.index('relColour')] < 0
nodeDL = NL[:, ncols.index('stateDL')] == 1
relc = NL[:, ncols.index('relColour')]
def has(*cs): return np.isin(relc, cs)
NTARGETS = [('on lock1 chain | DL, colour in {mu,A}', NL[:, ncols.index('onL1')], nodeDL & has(1, 2), True),
            ('on lock2 chain | DL, colour in {mu,B}', NL[:, ncols.index('onL2')], nodeDL & has(1, 3), True),
            ('on sigma chain | DL, colour in {al,mu}', NL[:, ncols.index('onSigma')], nodeDL & has(0, 1), True),
            ('on pi (R+3) chain | DL, colour in {al,A}', NL[:, ncols.index('onPiChain')], nodeDL & has(0, 2), True),
            ('on lock1 chain | DL (unconditional)', NL[:, ncols.index('onL1')], nodeDL & ~isHole, True),
            ('log own {c,1} chain size', np.log1p(NL[:, ncols.index('sz1')]), ~isHole & (NL[:, ncols.index('sz1')] > 0), False),
            ('log own {c,2} chain size', np.log1p(NL[:, ncols.index('sz2')]), ~isHole & (NL[:, ncols.index('sz2')] > 0), False),
            ('rim reach of {c,1} chain', NL[:, ncols.index('reach1')], ~isHole & (NL[:, ncols.index('sz1')] > 0), False),
            ('graph distance to hole', NL[:, ncols.index('dHole')], ~isHole, False)]
nodeprobe = {}
for t in ACT_TAGS:
    M = MODELS[t]
    if M.get('input'): continue
    Araw = np.fromfile(os.path.join(D, 'nodes_' + t + '.f32'), dtype=np.float32); H = M['arch']['H']; nLay = len(M['layers'])
    # rebuild per-layer node matrices in node_labels order
    per = [[] for _ in range(nLay)]; o = 0
    for q in sample:
        n = R[q]['n']; blk = Araw[o:o + nLay * n * H].reshape(nLay, n, H); o += nLay * n * H
        for li in range(nLay): per[li].append(blk[li])
    per = [np.concatenate(p, 0).astype(np.float64) for p in per]
    nodeprobe[t] = {}
    for li, L in enumerate(M['layers']):
        nodeprobe[t][str(L)] = {}
        for (tn, y, mask, binary) in NTARGETS:
            p = ridge_lofo(per[li][mask], y[mask], nfam[mask], alpha=10.0, binary=binary)
            nodeprobe[t][str(L)][tn] = round(metric(p, y[mask], binary), 4)
    print('node-probed', t, flush=True)
# node-level input baseline (colours-only encoder features: relative colour one-hot, hole, rim, rim position, degree, distance)
NI = np.stack([relc == 0, relc == 1, relc == 2, relc == 3, isHole, NL[:, ncols.index('isRim')] == 1, NL[:, ncols.index('deg')] / 8, np.minimum(1, NL[:, ncols.index('dHole')] / 5)], 1).astype(float)
nodeprobe['input-colours'] = {'input': {}}
for (tn, y, mask, binary) in NTARGETS:
    nodeprobe['input-colours']['input'][tn] = round(metric(ridge_lofo(NI[mask], y[mask], nfam[mask], alpha=10.0), y[mask], binary), 4)
RES['nodeProbe'] = nodeprobe

# ------------------------------------------------------------------ 7. error analysis
def describe_move(q, m):
    e = MV[q][m]; rim = ''.join(str(i) for i in range(5) if e[3] >> i & 1)
    return {'pair': PC[e[1]], 'size': e[2], 'rim': rim or '-', 'afterD': e[0], 'pi': e[4], 'sigma': e[5], 'meetsLock': e[6]}
def best_opt(q, M):
    """the optimal move the network ranks highest"""
    g = M['gamma']; best = None
    for m, e in enumerate(MV[q]):
        if e[0] == R[q]['d'] - 1: best = m; break
    return best
ERR = {}
for t in [main, 'nano-rl', 'mid-rl', 'colours-only-pretrained']:
    M = MODELS[t]; res = M['dhat'] - d
    verr = (np.abs(res) >= 0.5) & DL
    perr = (M['afterD'] != d - 1) & DL
    e = {'valueErrors|DL': int(verr.sum()), 'policyErrors|DL': int(perr.sum()), 'DL states': int(DL.sum()),
         'valueErr by depth': {int(k): [int((verr & (d == k)).sum()), int((DL & (d == k)).sum())] for k in range(2, 7) if (d == k).any()},
         'policyErr by depth': {int(k): [int((perr & (d == k)).sum()), int((DL & (d == k)).sum())] for k in range(2, 7) if (d == k).any()},
         'valueErr sign (over-optimistic share)': round(float((res[verr] < 0).mean()), 4) if verr.any() else None}
    # named-feature model of the error indicator and of the residual (DL states), lofo
    for nm, X in [('named', XN), ('named+candidates', XC)]:
        p = ridge_lofo(X[DL], verr[DL].astype(float), fam[DL]); e[f'AUC value-error ~ {nm}'] = round(metric(p, verr[DL].astype(float), True), 4)
        p = ridge_lofo(X[DL], res[DL], fam[DL]); e[f'R2 residual ~ {nm}'] = round(metric(p, res[DL], False), 4)
        if perr.sum() >= 5:
            p = ridge_lofo(X[DL], perr[DL].astype(float), fam[DL]); e[f'AUC policy-error ~ {nm}'] = round(metric(p, perr[DL].astype(float), True), 4)
    # with depth itself as a feature (is the error just "being deep"?)
    Xd = np.concatenate([XC, np.minimum(d, 6)[:, None], (d >= 3)[:, None]], 1)
    p = ridge_lofo(Xd[DL], res[DL], fam[DL]); e['R2 residual ~ named+candidates+depth'] = round(metric(p, res[DL], False), 4)
    p = ridge_lofo(Xd[DL], verr[DL].astype(float), fam[DL]); e['AUC value-error ~ named+candidates+depth'] = round(metric(p, verr[DL].astype(float), True), 4)
    pfull = np.full(N, np.nan); pfull[DL] = p
    # per-feature error rate contrasts (DL states)
    contr = {}
    for f, x in [('piAllDL', col('piAllDL')), ('DD', col('DD')), ('sigDL', col('sigDL')), ('exitsAllInterior', exitsInterior), ('runBound tight', (col('runBound') == d).astype(float)),
                 ('nOptDistinct==1', (nOptDistinct == 1).astype(float)), ('thick1>=2', (thick1 >= 2).astype(float)), ('thick2>=2', (thick2 >= 2).astype(float))]:
        for k in [2, 3]:
            m = DL & (d == k)
            if not m.any(): continue
            contr[f'{f} @d={k}'] = {str(v): [int((verr & m & (x == v)).sum()), int((m & (x == v)).sum())] for v in [0, 1]}
    e['contrasts (value errors / states)'] = contr
    # unexplained errors: errors whose predicted error probability is below the median of all DL states' predictions
    E = np.where(verr | perr)[0]
    unexpl = E[pfull[E] < np.nanpercentile(pfull[DL], 75)] if len(E) else E
    e['errors'] = int(len(E)); e['unexplained (feature-model score below the 75th percentile of DL states)'] = int(len(unexpl))
    e['value residual by depth (mean dhat-d, share over-optimistic among value errors)'] = {int(k): [round(float(res[DL & (d == k)].mean()), 3), round(float((res[verr & (d == k)] < 0).mean()), 3) if (verr & (d == k)).any() else None] for k in range(2, 7) if (d == k).any()}
    if 'optGap' in M:
        og = np.array(M['optGap']); mg = np.array(M['margin'])
        e['policy errors: chosen-minus-best-optimal score (median, max)'] = [round(float(np.median(og[perr])), 4), round(float(og[perr].max()), 4)] if perr.any() else None
        e['policy errors with exact score tie between distinct afterstates'] = int((og[perr] == 0).sum())
        e['correct deep decisions: median margin to the next distinct afterstate'] = round(float(np.median(mg[(~perr) & (d >= 3)])), 4)
    ERR[t] = {'summary': e, 'E': E, 'unexpl': unexpl, 'verr': verr, 'perr': perr, 'pfull': pfull}
RES['errors'] = {t: v['summary'] for t, v in ERR.items()}

# cluster the error states of the main model and the colours-only model on the MLP layer z + the last layer's rim block
CLUST = {}
for t, K in [(main, 4), ('colours-only-pretrained', 5)]:
    if t not in ACT_TAGS: continue
    A, blocks, H = acts(t); E = ERR[t]['E']
    if len(E) < 8: continue
    zlo = blocks[-1][1]; last = blocks[-2]
    X = np.concatenate([A[E, zlo:], A[E, last[1] + 3 * H:last[2]]], 1).astype(np.float64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    U, S, Vt = np.linalg.svd(X, full_matrices=False); P = U[:, :8] * S[:8]
    lab, _ = kmeans(P, K, seed=args.seed)
    M = MODELS[t]; out = []
    for c in range(K):
        idx = E[lab == c]
        if not len(idx): continue
        cnt = collections.Counter(fam[idx]); dd = collections.Counter(int(x) for x in d[idx])
        res = M['dhat'][idx] - d[idx]
        info = {'cluster': c, 'size': int(len(idx)), 'top degree words': dict(collections.Counter(R[q]['degWord'] for q in idx).most_common(4)), 'depths': dict(dd), 'families': dict(cnt.most_common(5)),
                'valueErr': int(ERR[t]['verr'][idx].sum()), 'policyErr': int(ERR[t]['perr'][idx].sum()),
                'unexplained': int(np.isin(idx, ERR[t]['unexpl']).sum()), 'mean residual (dhat-d)': round(float(res.mean()), 3)}
        # feature means vs all DL states of the same depth mix
        ref = DL & np.isin(d, list(dd.keys()))
        fm = {}
        for f, x in [('piAllDL', col('piAllDL')), ('dlRun', col('dlRun')), ('DD', col('DD')), ('sigDL', col('sigDL')), ('classN', col('classN')),
                     ('lenL1', col('lenL1')), ('lenL2', col('lenL2')), ('lenSigma', col('lenSigma')), ('nMoves', col('nMoves')), ('nOptDistinct', nOptDistinct),
                     ('thick1', thick1), ('thick2', thick2), ('thetaShared', theta), ('rerouteCuts', reroute), ('exitsAllInterior', exitsInterior), ('n', col('n'))]:
            fm[f] = [round(float(np.nanmean(x[idx])), 3), round(float(np.nanmean(x[ref])), 3)]
        info['feature mean (cluster, reference)'] = fm
        # decisions: chosen vs first optimal move
        dec = collections.Counter(); ex = []
        for q in idx:
            ch = describe_move(q, M['choice'][q]); bo = best_opt(q, M); op = describe_move(q, bo) if bo is not None else None
            dec[(ch['pair'], 'rim' if ch['rim'] != '-' else 'int', 'pi' if ch['pi'] else '', 'ok' if ch['afterD'] == d[q] - 1 else 'bad')] += 1
            if len(ex) < 4:
                r = R[q]; ex.append({'map': r['map'], 'colouring': r['key'], 'kappa': int(d[q]), 'ell': r['ell'], 'dhat': round(float(M['dhat'][q]), 2), 'j': r['j'], 'degWord': r['degWord'],
                                     'chosen': ch, 'optimal': op, 'Gamma': r['piAllDL'], 'dlRun': r['dlRun'], 'piToFill': r['piToFill'], 'sigDL': r['sigDL'],
                                     'lenL1': r['lenL1'], 'lenL2': r['lenL2'], 'thick': [int(thick1[q]), int(thick2[q])], 'theta': int(theta[q]), 'rerouteCuts': int(reroute[q]), 'nExits': int(nOptDistinct[q])})
        info['chosen move types'] = {' '.join(k): v for k, v in dec.most_common(6)}
        info['examples'] = ex
        out.append(info)
    dw = np.array([R[q]['degWord'] for q in E]); tab = collections.defaultdict(collections.Counter)
    for w_, l_ in zip(dw, lab): tab[w_][l_] += 1
    out.append({'purity of cluster given the relative rim degree word': round(sum(max(c.values()) for c in tab.values()) / len(lab), 3),
                'chance purity (largest cluster share)': round(float(np.bincount(lab).max() / len(lab)), 3), 'distinct degree words': len(tab)})
    CLUST[t] = out
    # scatter figure data
    CLUST[t + ':xy'] = (P[:, :2].tolist(), lab.tolist(), d[E].tolist())
RES['clusters'] = {k: v for k, v in CLUST.items() if not k.endswith(':xy')}

# all policy errors of the main model, listed
pe = np.where(ERR[main]['perr'])[0]; M = MODELS[main]
RES['mainPolicyErrors'] = [{'map': R[q]['map'], 'set': R[q]['set'], 'colouring': R[q]['key'], 'kappa': int(d[q]), 'ell': R[q]['ell'], 'dhat': round(float(M['dhat'][q]), 2),
                            'chosen': describe_move(q, M['choice'][q]), 'chosenAfterDhat': None, 'optimal': describe_move(q, best_opt(q, M)) if best_opt(q, M) is not None else None,
                            'Gamma': R[q]['piAllDL'], 'dlRun': R[q]['dlRun'], 'piToFill': R[q]['piToFill'], 'nExits': int(nOptDistinct[q]), 'thick': [int(thick1[q]), int(thick2[q])],
                            'theta': int(theta[q]), 'rerouteCuts': int(reroute[q]), 'margin': round(float(M['margin'][q]), 3)} for q in pe]
# consensus-hard states: policy errors shared by >= 3 of the 6 priors models with chain features
priors = ['priors-gnn-h16-k3-pool', 'nano-rl', 'tiny-rl', 'mid-rl', 'mid-pretrained', 'mid-zero-shot']
cnt = np.zeros(N, int)
for t in priors: cnt += (MODELS[t]['afterD'] != d - 1) & DL
RES['consensusHard'] = {'hist (number of priors models wrong, DL states)': dict(collections.Counter(int(x) for x in cnt[DL])),
                        'states wrong for >=4 of 6': [{'map': R[q]['map'], 'colouring': R[q]['key'], 'kappa': int(d[q]), 'wrong': int(cnt[q]), 'Gamma': R[q]['piAllDL'], 'dlRun': R[q]['dlRun'],
                                                       'nExits': int(nOptDistinct[q]), 'thick': [int(thick1[q]), int(thick2[q])], 'rerouteCuts': int(reroute[q])} for q in np.where(cnt >= 4)[0]]}

# rotation direction: the repeat-index shift of each move's afterstate (+2 / +3 / 0 = same index, -1 = filled)
def shifts(q, pred):
    return sorted(set(e[8] for e in MV[q] if pred(e)))
gear = {}
for k in [2, 3, 4]:
    m = np.where(DL & (dbin(d) == k))[0]
    c = collections.Counter(tuple(shifts(q, lambda e, q=q: e[0] == d[q] - 1)) for q in m)
    gear[f'optimal-move shifts at kappa={k}{"+" if k == 4 else ""}'] = {str(list(t)): v for t, v in c.most_common()}
for t in priors + ['colours-only-pretrained']:
    M = MODELS[t]; pe = np.where((M['afterD'] != d - 1) & DL)[0]
    c = collections.Counter((MV[q][M['choice'][q]][8], tuple(shifts(q, lambda e, q=q: e[0] == d[q] - 1))) for q in pe)
    gear[f'{t}: policy errors (chosen shift, optimal shifts)'] = {f'{a} vs {list(b)}': v for (a, b), v in c.most_common(8)}
# do the +3 / +2 directions coincide with "towards the nearer end of the pi DL run" (forward = +3 = pi, backward = +2)?
fwd = col('runFwd'); bwd = col('runBwd')
def nearer(q):
    if fwd[q] <= 0 or fwd[q] == bwd[q]: return None
    return 3 if fwd[q] < bwd[q] else 2
dirstat = collections.Counter()
for q in np.where(DL & (d >= 3))[0]:
    opt = set(e[8] for e in MV[q] if e[0] == d[q] - 1); nr = nearer(q)
    if nr is None or not ({2, 3} & opt): continue
    dirstat['optimal set contains nearer-end direction' if nr in opt else 'optimal set has only the farther-end direction'] += 1
gear['deep non-Gamma states: optimal rotation vs nearer end of the DL run'] = dict(dirstat)
for t in priors + ['colours-only-pretrained']:
    M = MODELS[t]; c = collections.Counter()
    for q in np.where((M['afterD'] != d - 1) & DL)[0]:
        sh = MV[q][M['choice'][q]][8]; nr = nearer(q)
        if sh not in (2, 3) or nr is None: continue
        c['chose nearer-end direction' if sh == nr else 'chose farther-end direction'] += 1
    gear[f'{t}: rotation errors vs run ends'] = dict(c)
rb = col('runBound')
RES['runBound'] = {f'kappa={k}': dict(collections.Counter('Gamma' if np.isnan(rb[q]) else ('tight' if rb[q] == d[q] else f'+{int(rb[q] - d[q])}') for q in np.where(DL & (d == k))[0])) for k in range(2, 7) if (DL & (d == k)).any()}
RES['rotationDirection'] = gear

# example file: the maps and states named in the error lists (graph spec + every move of each listed state)
ex_states = sorted(set([q for q in np.where(ERR[main]['perr'])[0]] + [q for q in np.where(cnt >= 4)[0]]))
ex_maps = sorted(set(R[q]['map'] for q in ex_states))
EXAMPLES = {'format': 'colouring = colours of vertices 0..n-1 skipping the hole, renamed by first occurrence; rim = link in cyclic order x0..x4; '
                      'moves = [kappa after, pair (relative: al = repeated rim colour, mu = x_{j+1}, A = x_{j+3}, B = x_{j+4}), size, rim positions (relative to j), '
                      'is pi move, is sigma move, meets a lock chain, repeat-index shift (-1 = fills)]',
            'maps': {k: {f: MAPS[k][f] for f in ['n', 'hole', 'link', 'edges', 'parent', 'ops', 'hist', 'degWord']} for k in ex_maps},
            'states': [{'map': R[q]['map'], 'colouring': R[q]['key'], 'kappa': int(d[q]), 'ell': R[q]['ell'], 'j': R[q]['j'], 'runFwd': R[q]['runFwd'], 'runBwd': R[q]['runBwd'],
                        'Gamma': R[q]['piAllDL'], 'priorsModelsWrong': int(cnt[q]), 'poolChoice': int(MODELS[main]['choice'][q]),
                        'moves': [[e[0], PC[e[1]], e[2], ''.join(str(i) for i in range(5) if e[3] >> i & 1) or '-', e[4], e[5], e[6], e[8]] for e in MV[q]]} for q in ex_states]}
json.dump(EXAMPLES, open(os.path.join(OUT, 'examples.json'), 'w'))

# ------------------------------------------------------------------ 8. figures
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b',
                     'xtick.color': '#52514e', 'ytick.color': '#52514e', 'svg.fonttype': 'none', 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb'})
SER = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
GRID = '#e4e3dc'
# F1: agreement and failure by depth (held-out: test + splice)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
show = [(main, 'priors pool H16 K3'), ('nano-rl', 'nano-rl H4 K1'), ('colours-only-pretrained', 'colours-only H32 K4'), ('tr-rec-h32-t16', 'tabula rasa rec H32'), ('pi-agent', 'pi map (theory)')]
for i, (t, lbl) in enumerate(show):
    xs, ag, fl = [], [], []
    for k in [1, 2, 3, 4]:
        m = (grp != 'train') & (dbin(d) == k)
        if t == 'pi-agent': a = (piAfterD[m] == d[m] - 1).mean(); f = (piSteps[m] < 0).mean() + ((piSteps[m] >= 0) & (piSteps[m] > np.maximum(3 * d[m], d[m] + 10))).mean()
        else: a = (MODELS[t]['afterD'][m] == d[m] - 1).mean(); f = (MODELS[t]['steps'][m] < 0).mean()
        xs.append(k); ag.append(a); fl.append(f)
    ax[0].plot(xs, ag, '-o', color=SER[i], lw=2, ms=4, label=lbl); ax[1].plot(xs, fl, '-o', color=SER[i], lw=2, ms=4, label=lbl)
for a_, tt in zip(ax, ['optimal-move agreement', 'rollout failure (loop or > max(3d, d+10) swaps)']):
    a_.set_xticks([1, 2, 3, 4]); a_.set_xticklabels(['1', '2', '3', '4+']); a_.set_xlabel('kappa (swaps to a fill)'); a_.set_title(tt, loc='left', fontsize=9); a_.grid(axis='y', color=GRID, lw=0.6)
ax[1].legend(frameon=False, fontsize=8, loc='upper left')
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig1-behaviour-by-depth.svg')); plt.close(fig)

# F2: value compression: dhat by kappa (main model), box per depth
fig, ax = plt.subplots(figsize=(4.6, 3.2))
data = [MODELS[main]['dhat'][(d == k)] for k in [1, 2, 3, 4, 5]]
bp = ax.boxplot(data, positions=[1, 2, 3, 4, 5], widths=0.5, patch_artist=True, showfliers=True, flierprops={'markersize': 2, 'markerfacecolor': SER[0], 'markeredgecolor': SER[0]})
for b in bp['boxes']: b.set_facecolor('#cfe0f6'); b.set_edgecolor(SER[0])
for m_ in bp['medians']: m_.set_color('#0b0b0b')
ax.plot([0.5, 5.5], [0.5, 5.5], color='#9b9a94', lw=1, ls='--')
ax.set_xlabel('kappa (exact)'); ax.set_ylabel('value-implied distance'); ax.set_title('priors pool model: value is compressed past 2', loc='left', fontsize=9); ax.grid(axis='y', color=GRID, lw=0.6)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig2-value-compression.svg')); plt.close(fig)

# F3: probe heatmap (main model, random init, input baseline) for DL-conditional targets
tn_show = [x[0] for x in TARGETS]
cols_show = []
for t, lbl, bs in [('input-priors', 'input (priors)', None), (main, 'pool', None), ('random-' + main, 'random pool', ['L1', 'L3', 'z']),
                   ('input-colours', 'input (colours)', None), ('colours-only-pretrained', 'colours-only', ['L1', 'L4', 'z']), ('random-colours-only-pretrained', 'random colours-only', ['L4', 'z']),
                   ('tr-rec-h32-t16', 'tabula rasa', ['t4', 't16', 'z']), ('random-tr-rec-h32-t16', 'random rec', ['t16', 'z'])]:
    if t in probe:
        for b in probe[t]:
            if bs is None or b in bs: cols_show.append((t, b, lbl + ' ' + b if b != 'input' else lbl))
Mx = np.array([[probe[t][b][tn] for (t, b, _) in cols_show] for tn in tn_show])
# normalise: AUC -> 2*(AUC-0.5), R2 as is (clip 0)
binary = [x[3] for x in TARGETS]
Mn = np.array([[max(0, 2 * (v - 0.5)) if binary[i] else max(0, v) for v in row] for i, row in enumerate(Mx)])
fig, ax = plt.subplots(figsize=(10.5, 6.4))
im = ax.imshow(Mn, cmap='Blues', vmin=0, vmax=1, aspect='auto')
ax.set_yticks(range(len(tn_show))); ax.set_yticklabels([f"{t} [{'AUC' if binary[i] else 'R2'}]" for i, t in enumerate(tn_show)], fontsize=7.5)
ax.set_xticks(range(len(cols_show))); ax.set_xticklabels([c[2] for c in cols_show], rotation=60, ha='right', fontsize=7)
for i in range(Mx.shape[0]):
    for j in range(Mx.shape[1]):
        ax.text(j, i, f'{Mx[i, j]:.2f}', ha='center', va='center', fontsize=5.5, color='#ffffff' if Mn[i, j] > 0.6 else '#0b0b0b')
ax.set_title('linear probes, leave-one-family-out (shade: 2(AUC-0.5) or R2, clipped at 0)', loc='left', fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig3-probes.svg')); plt.close(fig)

# F4: error clusters (main model)
if main + ':xy' in CLUST:
    xy, lab, dd = CLUST[main + ':xy']; xy = np.array(xy); lab = np.array(lab); dd = np.array(dd)
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    for c in sorted(set(lab)):
        m = lab == c; ax.scatter(xy[m, 0], xy[m, 1], s=np.where(dd[m] >= 3, 22, 9), color=SER[c], edgecolor='#fcfcfb', linewidth=0.6, label=f'cluster {c} (n={m.sum()})')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.legend(frameon=False, fontsize=7); ax.set_title('pool model error states (z + last rim block); large = kappa>=3', loc='left', fontsize=8.5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig4-error-clusters.svg')); plt.close(fig)

# F5: deep share among DL states by DL-run length and by Gamma membership
fig, ax = plt.subplots(figsize=(4.8, 3.0))
ks = list(RES['deepByRun'].keys()); vals = [RES['deepByRun'][k]['deep'] for k in ks]
ax.bar(range(len(ks)), vals, color=SER[0], width=0.6)
for i, k in enumerate(ks): ax.text(i, vals[i], f"n={RES['deepByRun'][k]['n']}", ha='center', va='bottom', fontsize=7, color='#52514e')
ax.set_xticks(range(len(ks))); ax.set_xticklabels(ks); ax.set_xlabel('length of the DL run along pi containing the state'); ax.set_ylabel('share with kappa >= 3')
ax.set_title('deep states sit on long DL runs of the pi map', loc='left', fontsize=9); ax.grid(axis='y', color=GRID, lw=0.6)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig5-deep-by-dlrun.svg')); plt.close(fig)

def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating, float)): return None if (isinstance(o, float) or True) and (o != o) else float(o)
    return o
json.dump(clean(RES), open(os.path.join(OUT, 'results.json'), 'w'), indent=1)
print('wrote', OUT)
