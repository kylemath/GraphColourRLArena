/* Two desks for the Maze Arena.

   Agent desk: a rule generator proposes a mazearena-rule-1 packet. The agent critic
   runs it against the admitted roster (search rules, earlier packets, nets).

   Map desk: a map generator proposes a graph grown from a blockage the roster just
   missed. The map critic keeps it, trains on it, or discards it.

   The critics talk to their own generator. The two generators exchange blockages
   and harder-requests. Nothing in this file is a kernel acceptance. See DUEL.md.
*/
'use strict';

const FORMAT = 'mazearena-rule-1';
const PICKS = new Set(['fill', 'rim', 'any']);
const CLAUSE_KEYS = new Set(['pick', 'minBits', 'shortest']);
const PACKET_KEYS = new Set(['format', 'name', 'admits', 'clauses', 'statement', 'reads']);
const ADMITS = new Set(['arena-only', 'lemma-candidate']);

// Census maps the agent critic may score and the map generator may not splice.
const CENSUS_TEST = ['c30', 'p32r11', 'p34r28', 'c60', 'six', 'trap17', 'r210'];

function popcount(m) {
  let b = 0;
  m |= 0;
  while (m) { b += m & 1; m >>>= 1; }
  return b;
}

function chainSig(K) {
  return { pair: [Math.min(K.a, K.b), Math.max(K.a, K.b)], size: K.vs.length, rimBits: popcount(K.rimMask), rimMask: K.rimMask };
}

function reject(reason) {
  const err = new Error(reason);
  err.ruleReject = true;
  return err;
}

function validateRule(packet) {
  if (!packet || typeof packet !== 'object') throw reject('packet is not an object');
  for (const k of Object.keys(packet)) {
    if (!PACKET_KEYS.has(k)) throw reject('unknown field ' + k);
  }
  if (packet.format !== FORMAT) throw reject('format is not ' + FORMAT);
  if (typeof packet.name !== 'string' || !/^[a-z0-9][a-z0-9-]{0,63}$/.test(packet.name)) throw reject('name must be a short hyphenated id');
  if (!ADMITS.has(packet.admits)) throw reject('admits must be arena-only or lemma-candidate');
  if (packet.script != null || packet.tactic != null) throw reject('a packet carries no script');
  if (!Array.isArray(packet.clauses) || packet.clauses.length < 1 || packet.clauses.length > 8) throw reject('clauses must be a list of 1 to 8');
  packet.clauses.forEach((c, i) => {
    if (!c || typeof c !== 'object') throw reject('clause ' + i + ' is not an object');
    for (const k of Object.keys(c)) if (!CLAUSE_KEYS.has(k)) throw reject('clause ' + i + ' has unknown field ' + k);
    if (!PICKS.has(c.pick)) throw reject('clause ' + i + ' pick is not fill, rim, or any');
    if (c.minBits != null && (c.pick !== 'rim' || !Number.isInteger(c.minBits) || c.minBits < 1 || c.minBits > 5)) throw reject('clause ' + i + ' minBits');
    if (c.shortest != null && typeof c.shortest !== 'boolean') throw reject('clause ' + i + ' shortest');
  });
  if (packet.admits === 'lemma-candidate') {
    if (typeof packet.statement !== 'string' || !/^[A-Za-z_][A-Za-z0-9_'.]*$/.test(packet.statement)) throw reject('lemma-candidate needs a type name in statement');
    if (/sorry|admit|axiom/i.test(packet.statement)) throw reject('statement names a type, not a tactic');
  } else if (packet.statement != null) {
    throw reject('arena-only packets have no statement');
  }
  if (packet.reads != null) {
    if (!Array.isArray(packet.reads) || packet.reads.some(r => typeof r !== 'string')) throw reject('reads must be a list of feature names');
  }
  return packet;
}

function clauseHits(A, G, col, moves, clause) {
  const hits = [];
  for (let i = 0; i < moves.length; i++) {
    const K = moves[i];
    if (clause.pick === 'fill') { if (A.won(G, A.flip(col, K))) hits.push(i); }
    else if (clause.pick === 'rim') { if (popcount(K.rimMask) >= (clause.minBits || 1)) hits.push(i); }
    else hits.push(i);
  }
  if (!hits.length) return hits;
  if (clause.shortest) hits.sort((i, j) => moves[i].vs.length - moves[j].vs.length || i - j);
  return hits;
}

// Returns (G, col) -> move index into movesOf. Ties keep the earliest movesOf index.
function compileRule(packet) {
  validateRule(packet);
  return function ruleAgent(A, G, col) {
    const moves = A.movesOf(G, col);
    if (!moves.length) return -1;
    for (let c = 0; c < packet.clauses.length; c++) {
      const hits = clauseHits(A, G, col, moves, packet.clauses[c]);
      if (hits.length) return hits[0];
    }
    return -1;
  };
}

// The roadblock handed to the map generator: the chain the agent played, and the chains that actually lower d.
function blockage(A, G, col, moveIndex) {
  const moves = A.movesOf(G, col);
  const d = A.distOf(G, col);
  const i = moveIndex;
  const picked = i >= 0 && i < moves.length ? moves[i] : null;
  const good = [];
  if (d != null) {
    for (let k = 0; k < moves.length; k++) {
      if (A.distOf(G, A.flip(col, moves[k])) === d - 1) good.push(chainSig(moves[k]));
    }
  }
  const after = picked ? A.distOf(G, A.flip(col, picked)) : null;
  return {
    map: G.key,
    heldOut: CENSUS_TEST.indexOf(G.key) >= 0,
    canon: A.canon(G, col),
    d,
    picked: picked ? chainSig(picked) : null,
    lowered: d != null && after === d - 1,
    good,
  };
}

function scoreOn(A, maps, agent) {
  let a2 = 0, n2 = 0, a3 = 0, n3 = 0, a4 = 0, n4 = 0;
  const dead = [];
  for (const M of maps) {
    for (const s of M.samples) {
      if (s.d == null || s.d < 1) continue;
      const moves = A.movesOf(M.G, s.col);
      const i = agent(M.G, s.col);
      const ok = i >= 0 && i < moves.length && A.distOf(M.G, A.flip(s.col, moves[i])) === s.d - 1;
      if (s.d >= 2) { n2++; a2 += ok; }
      if (s.d >= 3) { n3++; a3 += ok; }
      if (s.d >= 4) { n4++; a4 += ok; }
      if (!ok && s.d >= 2 && dead.length < 12) dead.push(blockage(A, M.G, s.col, i));
    }
  }
  const r4 = n => n == null ? null : +n.toFixed(4);
  return {
    agree2: n2 ? r4(a2 / n2) : null, n2,
    agree3: n3 ? r4(a3 / n3) : null, n3,
    agree4: n4 ? r4(a4 / n4) : null, n4,
    deadEnds: dead,
  };
}

function metricOf(row, prefer) {
  if (prefer === 'agree3' && row.n3 > 0) return row.agree3;
  if (row.n2 > 0) return row.agree2;
  return null;
}

// Champion: higher agree2, then higher agree3. A perfect agree3 on a handful of
// deep states does not crown an agent that misses the d = 2 mass.
function betterRow(a, b) {
  const a2 = a && a.agree2 != null ? a.agree2 : -1;
  const b2 = b && b.agree2 != null ? b.agree2 : -1;
  if (a2 !== b2) return a2 > b2;
  const a3 = a && a.agree3 != null ? a.agree3 : -1;
  const b3 = b && b.agree3 != null ? b.agree3 : -1;
  return a3 > b3;
}

function atCeiling(row, bar2, bar3) {
  if (!row || row.agree2 == null || row.agree2 < bar2) return false;
  if (row.n3 > 0 && (row.agree3 == null || row.agree3 < bar3)) return false;
  return true;
}

// before = best admitted score. after = best score once the candidate is in the roster.
// The bar is fixed for the round. Three outcomes from the brief, plus partial when the
// score moves and still sits under the bar.
function roundVerdict(before, after, bar) {
  if (before == null || after == null || !(bar > 0)) return { to: 'rule-gen', code: 'fail', detail: 'no score' };
  if (before >= bar) return { to: 'map-gen', code: 'harder', detail: 'roster is at the bar; the map desk owes a harder wall' };
  if (after >= bar) return { to: 'rule-gen', code: 'success', detail: 'the new rule carried the roster across the bar' };
  if (!(after > before)) return { to: 'rule-gen', code: 'fail', detail: 'the new rule did not move the score' };
  return { to: 'rule-gen', code: 'partial', detail: 'the score moved and the roster is still under the bar' };
}

function mapVerdict(o) {
  if (o.triError) return { to: 'map-gen', code: 'reject', detail: o.triError };
  if (o.censusTest) return { to: 'map-gen', code: 'reject', detail: 'census test maps are not splice parents' };
  if (o.exhaustive && o.completed === false) return { to: 'rule-gen', code: 'unrealizable', detail: 'every completion branch failed; this skeleton is a local lemma, not a colouring of a map' };
  if (!o.exhaustive && o.completed === false) return { to: 'map-gen', code: 'incomplete', detail: 'the builder stopped; that is not a disproof' };
  if (o.rimMaskOk === false) return { to: 'map-gen', code: 'discard', detail: 'rimMask changed; the filler dissolved the lock' };
  if (o.rosterClears) return { to: 'map-gen', code: 'discard', detail: 'the admitted roster already clears this map' };
  if (o.newRuleClears && o.oldMisses) return { to: 'map-gen', code: 'train', detail: 'the old roster misses it and the new rule clears it' };
  return { to: 'map-gen', code: 'test', detail: 'the roster is under the bar on this map' };
}

function crossTalk(verdict, deadEnds) {
  const walls = deadEnds.filter(d => !d.heldOut && !d.lowered);
  const held = deadEnds.filter(d => d.heldOut && !d.lowered);
  return {
    'rule-gen': {
      fromCritic: verdict.to === 'rule-gen' ? verdict.code : 'hold',
      unreadableTestWalls: held.length,
    },
    'map-gen': {
      fromCritic: verdict.code === 'harder' ? 'harder' : 'hold',
      blockages: walls,
    },
    betweenGenerators: verdict.code === 'harder'
      ? { from: 'rule-gen', to: 'map-gen', ask: 'harder', using: 'train-map blockages only' }
      : { from: 'map-gen', to: 'rule-gen', ask: 'a clause that picks one of good[]', blockages: walls.length },
  };
}

function protocolSelfCheck() {
  const cases = [
    [0.5, 0.95, 0.9, 'rule-gen', 'success'],
    [0.5, 0.5, 0.9, 'rule-gen', 'fail'],
    [0.5, 0.4, 0.9, 'rule-gen', 'fail'],
    [0.91, 0.99, 0.9, 'map-gen', 'harder'],
    [0.5, 0.7, 0.9, 'rule-gen', 'partial'],
  ];
  cases.forEach(([before, after, bar, to, code]) => {
    const v = roundVerdict(before, after, bar);
    if (v.to !== to || v.code !== code) throw new Error('verdict ' + before + '->' + after + ' got ' + v.to + '/' + v.code);
  });
  const maps = [
    [{ triError: 'E != 3n-6' }, 'reject'],
    [{ censusTest: true }, 'reject'],
    [{ exhaustive: true, completed: false }, 'unrealizable'],
    [{ exhaustive: false, completed: false }, 'incomplete'],
    [{ rimMaskOk: false }, 'discard'],
    [{ rosterClears: true }, 'discard'],
    [{ newRuleClears: true, oldMisses: true }, 'train'],
    [{}, 'test'],
  ];
  maps.forEach(([o, code]) => {
    const v = mapVerdict(o);
    if (v.code !== code) throw new Error('map verdict ' + JSON.stringify(o) + ' got ' + v.code);
  });
  let threw = false;
  try { validateRule({ format: FORMAT, name: 'x', admits: 'lemma-candidate', clauses: [{ pick: 'fill' }], script: 'exact sorry' }); }
  catch (e) { threw = e.ruleReject && /script/.test(e.message); }
  if (!threw) throw new Error('script was accepted');
  try { validateRule({ format: FORMAT, name: 'x', admits: 'lemma-candidate', clauses: [{ pick: 'fill' }] }); threw = false; }
  catch (e) { threw = e.ruleReject; }
  if (!threw) throw new Error('lemma without a statement was accepted');
  validateRule({ format: FORMAT, name: 'shortest-double-rim', admits: 'arena-only', clauses: [{ pick: 'fill' }, { pick: 'rim', minBits: 2, shortest: true }] });
  validateRule({ format: FORMAT, name: 'exists-missing-colour', admits: 'lemma-candidate', statement: 'exists_missing_colour', clauses: [{ pick: 'fill' }] });
  if (!betterRow({ agree2: 0.95, agree3: 0.5 }, { agree2: 0.7, agree3: 1 })) throw new Error('agree3 crowned a worse agree2 agent');
  if (atCeiling({ agree2: 0.99, agree3: 0.5, n3: 18 }, 0.95, 0.9)) throw new Error('depth bar ignored');
  if (!atCeiling({ agree2: 0.99, agree3: 0.91, n3: 18 }, 0.95, 0.9)) throw new Error('both bars should clear');
  return 'protocol ok';
}

module.exports = {
  FORMAT, CENSUS_TEST, popcount, chainSig, validateRule, compileRule, blockage, scoreOn, metricOf,
  roundVerdict, mapVerdict, crossTalk, protocolSelfCheck, betterRow, atCeiling,
};
