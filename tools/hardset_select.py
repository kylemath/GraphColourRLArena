#!/usr/bin/env python3
"""Hard-set candidate selection (Studio Job 10, 14 Oct 2026; exploratory).

Reads a GraphColourDeductiveSolver checkout and writes candidate maps (sphere triangulations with a degree-5 hole)
that are hard for the four-colour *proof*, not just for players:
  1. floor-tight holes: some Kempe class at the hole has filled fraction F/N = 1/4 exactly (the quarter floor),
     from the frame census (Census29 orders 24-32 and Census33, field mf) and from the Studio gluing run (Job 2,
     graphs with separating 4/5-cycles);
  2. the two planar R0 counterexample maps (p27#4710 hole 14, the C40#14 fullerene dual hole 0);
  3. 66666 holes whose 2-ball (hole + 5 neighbours + second ring) is NOT D-reducible (Studio Job 9 results),
     located in census graphs.
Output: JSON list of {key, name, n, hole, link, edges, rot, source, why, meta}.

usage: hardset_select.py GRAPHCOLOUR_DIR OUT.json [--tight 150] [--glued 50] [--twoball 100]
"""
import sys, os, json, gzip, glob, re, argparse, collections

ap = argparse.ArgumentParser()
ap.add_argument('gc'); ap.add_argument('out')
ap.add_argument('--tight', type=int, default=150); ap.add_argument('--glued', type=int, default=50); ap.add_argument('--twoball', type=int, default=100)
a = ap.parse_args()
W = os.path.join(a.gc, 'SolvingFrameworkPlan/docs/working')
LT = os.path.join(a.gc, 'backgroundMaterial/planemap-structural/longtable')


def spec(key, name, rot, hole, source, why, meta):
    edges = sorted({(min(u, w), max(u, w)) for u, r in enumerate(rot) for w in r})
    return dict(key=key, name=name, n=len(rot), hole=hole, link=list(rot[hole]), edges=[list(e) for e in edges],
                rot=[list(r) for r in rot], source=source, why=why, meta=meta)


def read_rot(line):
    p = line.split()
    return p[0], [[int(x) for x in r.split(',')] for r in p[2].split(';')]


def load_census(names_by_file):
    out = {}
    for f, names in names_by_file.items():
        for l in open(f):
            nm = l.split(' ', 1)[0]
            if nm in names: out[nm] = read_rot(l)[1]
    return out


cands = []
# ---------------- 1a. floor-tight census holes (smallest orders first)
tight = []
for o in range(22, 34):
    ef = os.path.join(W, 'Census29/out/eval-%d.jsonl' % o) if o <= 32 else os.path.join(W, 'Census33/out/eval-33.jsonl.gz')
    if not os.path.exists(ef): continue
    op = gzip.open if ef.endswith('.gz') else open
    for l in op(ef, 'rt'):
        r = json.loads(l)
        if 'holes' in r:      # Census33 format
            hv = r['holes'].items()
        else:                 # Census29 format: minfrac / words / classes per hole
            hv = [(h, dict(mf=mf, w=r.get('words', {}).get(h, ''), S=sum(c[0] for c in r.get('classes', {}).get(h, []))))
                  for h, mf in r.get('minfrac', {}).items()]
        for h, v in hv:
            if abs(v.get('mf', 1) - 0.25) < 1e-6: tight.append((o, r['name'], int(h), v))
    if len(tight) >= 4 * a.tight: break
seen = set(); pick = []
for o, nm, h, v in tight:
    if nm in seen: continue          # one hole per graph
    seen.add(nm); pick.append((o, nm, h, v))
    if len(pick) >= a.tight: break
files = collections.defaultdict(set)
for o, nm, h, v in pick:
    files[os.path.join(W, 'Census29/out/frame-%d.txt' % o) if o <= 32 else os.path.join(W, 'Census33/out/frame-33.txt')].add(nm)
rots = load_census(files)
for o, nm, h, v in pick:
    cands.append(spec('ht_%s_h%d' % (nm.replace('#', '_').replace('.', '_'), h), '%s hole %d (floor-tight)' % (nm, h), rots[nm], h,
                      'census%d floor-tight' % o, 'a Kempe class at this hole sits exactly on the quarter floor F/N = 1/4 (hole word %s)' % v.get('w', ''),
                      dict(order=o, mf=v.get('mf'), S=v.get('S'), w=v.get('w'))))
# ---------------- 1b. floor-tight glued holes (Job 2; smallest n first)
gl = []
for m in ('glue4', 'wideB', 'tight33'):
    f = os.path.join(LT, 'gluing/studio-run/%s.jsonl.gz' % m)
    if not os.path.exists(f): continue
    for l in gzip.open(f, 'rt'):
        r = json.loads(l)
        if not r.get('frame'): continue
        for hh in r['holes']:
            if abs(hh['minFN'] - 0.25) < 1e-9:
                gl.append((r['n'], m, r['name'], hh['h'], hh, r['rot'])); break
gl.sort(key=lambda x: (x[0], x[1], x[2]))
for n, m, nm, h, hh, rs in gl[:a.glued]:
    rot = [[int(x) for x in s.split(',')] for s in rs.split(';')]
    cands.append(spec('hg_%s_%d' % (m, len(cands)), '%s hole %d (glued, floor-tight)' % (nm, h), rot, h, 'gluing/%s floor-tight' % m,
                      'glued along a separating %s-cycle; a Kempe class sits exactly on the quarter floor 1/4' % ('4' if m == 'glue4' else '5'),
                      dict(glued=nm, S=hh['S'], ncls=hh['ncls'], ld=hh['ld'])))
# ---------------- 2. R0 counterexample maps
r0 = []
for l in open(os.path.join(W, 'Census29/out/frame-27.txt')):
    if l.startswith('p27#4710 '): r0.append(('p27#4710', read_rot(l)[1], 14, 'census27 (frame class)'))
for l in open(os.path.join(W, 'TrackF/graphs/fall_20_46.txt')):
    if l.startswith('C40#14 '): r0.append(('C40#14', read_rot(l)[1], 0, 'fullerene C40#14 dual'))
for nm, rot, h, src in r0:
    cands.append(spec('hr0_%s_h%d' % (nm.replace('#', '_'), h), '%s hole %d (R0 counterexample)' % (nm, h), rot, h, 'R0 counterexample: ' + src,
                      'planar counterexample to Conjecture R0: a closed 10-step perfect-matching walk through a Hamiltonian pair in the Tait dual',
                      dict(witness='longtable/axioms/studio/out/R0_counterexample_%s_h%d.json' % ('p27_4710' if nm.startswith('p27') else 'C40_14', h))))
# ---------------- 3. 66666 holes with a NON-D-reducible 2-ball
res = {}
for l in gzip.open(os.path.join(LT, 'reducibility/wheels/studio/run9_results.txt.gz'), 'rt'):
    p = l.split()
    if 'D=' in l: res[p[0][1:]] = (int(p[1][2:]), p[6] == 'D=true', int(p[5].split('=')[1]), int(p[3].split('=')[1]))
sys.path.insert(0, os.path.join(LT, 'reducibility/wheels/studio'))
from gen_2ball import canon  # noqa: E402

def two_ball_pattern(rot, h):
    x = rot[h]
    if any(len(rot[v]) != 6 for v in x): return None
    deg = []
    for i in range(5):
        xi, xn, xp = x[i], x[(i + 1) % 5], x[(i - 1) % 5]
        r = rot[xi]; k = r.index(h)
        # around x_i starting after h: neighbours in rotation order; the three outer ones lie between x_{i-1} and x_{i+1}
        seq = [r[(k + j) % 6] for j in range(1, 6)]
        if seq[0] == xn: seq = seq[::-1]          # orient so that seq = x_{i-1}, s_{i-1}, p_i, s_i, x_{i+1}
        if seq[0] != xp or seq[4] != xn: return None
        deg.append((seq[2], seq[3]))               # (p_i, s_i)
    ys = [v for pr in deg for v in pr]
    if len(set(ys)) != 10 or h in ys or set(ys) & set(x): return None
    return canon(tuple(len(rot[v]) for v in ys))

tb = []; files = collections.defaultdict(set)
for o in range(22, 34):
    f = os.path.join(W, 'Census29/out/frame-%d.txt' % o) if o <= 32 else os.path.join(W, 'Census33/out/frame-33.txt')
    if not os.path.exists(f): continue
    for l in open(f):
        nm, rot = read_rot(l)
        for h in range(len(rot)):
            if len(rot[h]) != 5: continue
            pat = two_ball_pattern(rot, h)
            if pat is None: continue
            key = ''.join(map(str, pat)); rr = res.get(key)
            if rr and not rr[1]: tb.append((o, nm, h, key, rr, rot))
    if len(tb) >= 4 * a.twoball: break
seenp = set(); n2 = 0
for o, nm, h, key, rr, rot in tb:
    if (nm, key) in seenp: continue
    seenp.add((nm, key)); n2 += 1
    cands.append(spec('h2b_%s_h%d' % (nm.replace('#', '_').replace('.', '_'), h), '%s hole %d (66666, 2-ball not D-reducible)' % (nm, h), rot, h,
                      'census%d 66666 2-ball' % o,
                      'hole word 66666; its 2-ball (second ring %s, ring %d) is not D-reducible: %d of %d ring colourings fixable' % (key, rr[0], rr[2], rr[3]),
                      dict(order=o, pattern=key, ring=rr[0], reached=rr[2], classes=rr[3])))
    if n2 >= a.twoball: break
json.dump(cands, open(a.out, 'w'))
print(json.dumps(collections.Counter(c['source'].split(' ')[0] if not c['source'].startswith('R0') else 'R0' for c in cands)), len(cands))
