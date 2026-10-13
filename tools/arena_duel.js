// node tools/arena_duel.js [--protocol-only] [--bar-agree3 0.9] [--bar-agree2 0.95]
//
// Runs the two-desk round in docs/arena/DUEL.md on a fixed train slice (p22, r1)
// and reports the census-test map `six` without letting it vote.
'use strict';
const fs = require('fs'), path = require('path');
const root = path.join(__dirname, '..');
const D = require(path.join(root, 'docs/arena/duel.js'));

const argv = process.argv.slice(2);
const protocolOnly = argv.includes('--protocol-only');
function flag(name, d) {
  const i = argv.indexOf('--' + name);
  return i >= 0 ? +argv[i + 1] : d;
}
const BAR3 = flag('bar-agree3', 0.9);
const BAR2 = flag('bar-agree2', 0.95);

const protocol = D.protocolSelfCheck();
console.log(protocol);
if (protocolOnly) process.exit(0);

const enginePath = fs.existsSync(path.join(root, 'docs/arena/engine.js'))
  ? path.join(root, 'docs/arena/engine.js')
  : path.join(root, 'docs/arena/engine.js');
global.window = global;
require(path.join(root, 'docs/levels.js'));
const A = require(enginePath);
const GAME = window.GAME;

const TRAIN = ['p22', 'r1'];
const HELD = ['six'];
const packet = D.validateRule(JSON.parse(fs.readFileSync(path.join(root, 'docs/arena/rules/shortest-double-rim.json'), 'utf8')));
const rule = D.compileRule(packet);

function wrap(fn) {
  return (G, col) => fn(G, col);
}
const agents = {
  hand: wrap((G, col) => A.baselines.hand(G, col, A.rng(1))),
  twoply: wrap((G, col) => A.baselines.twoply(G, col, A.rng(1))),
  [packet.name]: (G, col) => rule(A, G, col),
};

function bench(keys) {
  const tb = new A.Testbench(GAME, keys, { maxDeep: 40, maxHard: 40, maxEasy: 10, rollouts: 0 });
  const out = {};
  for (const name of Object.keys(agents)) out[name] = D.scoreOn(A, tb.maps, agents[name]);
  return out;
}

console.log('scoring train ' + TRAIN.join(', ') + ' …');
const train = bench(TRAIN);
console.log('scoring held-out ' + HELD.join(', ') + ' …');
const held = bench(HELD);

function best(rows, names) {
  let name = names[0];
  for (const n of names.slice(1)) if (D.betterRow(rows[n], rows[name])) name = n;
  return { name, row: rows[name] };
}
function publicScore(row) {
  return { agree2: row.agree2, n2: row.n2, agree3: row.agree3, n3: row.n3, agree4: row.agree4, n4: row.n4, deadEnds: row.deadEnds.length };
}

const admitted = ['hand', 'twoply'];
const before = best(train, admitted);
const after = best(train, admitted.concat(packet.name));
const beforeCeil = D.atCeiling(before.row, BAR2, BAR3);
const afterCeil = D.atCeiling(after.row, BAR2, BAR3);
const moved = D.betterRow(after.row, before.row);
const verdict = D.roundVerdict(beforeCeil ? 1 : 0, afterCeil ? 1 : moved ? 0.5 : 0, 1);
const talk = D.crossTalk(verdict, train.twoply.deadEnds.concat(train[packet.name].deadEnds));

const round = {
  format: 'mazearena-duel-1',
  bar: { agree2: BAR2, agree3: BAR3 },
  train: TRAIN,
  heldOut: HELD,
  scores: {
    train: Object.fromEntries(Object.entries(train).map(([k, v]) => [k, publicScore(v)])),
    held: Object.fromEntries(Object.entries(held).map(([k, v]) => [k, publicScore(v)])),
  },
  before: { agent: before.name, agree2: before.row.agree2, agree3: before.row.agree3 },
  after: { agent: after.name, agree2: after.row.agree2, agree3: after.row.agree3 },
  verdict,
  crossTalk: {
    'rule-gen': talk['rule-gen'],
    'map-gen': { fromCritic: talk['map-gen'].fromCritic, blockages: talk['map-gen'].blockages.length },
    betweenGenerators: talk.betweenGenerators,
  },
  blockages: talk['map-gen'].blockages.slice(0, 4),
};

function line(name, row) {
  const a2 = row.agree2 == null ? '  -  ' : row.agree2.toFixed(3);
  const a3 = row.agree3 == null ? '  -  ' : row.agree3.toFixed(3);
  return '  ' + name.padEnd(22) + ' agree2 ' + a2 + ' (n2 ' + row.n2 + ')  agree3 ' + a3 + ' (n3 ' + row.n3 + ')';
}
console.log('train');
for (const name of Object.keys(train)) console.log(line(name, train[name]));
console.log('held-out (does not vote)');
for (const name of Object.keys(held)) console.log(line(name, held[name]));
console.log('verdict ' + verdict.to + ' / ' + verdict.code + '  (' + verdict.detail + ')');
console.log('bars agree2 >= ' + BAR2 + ' and agree3 >= ' + BAR3 + '   before ' + before.name + '   after ' + after.name);
console.log(JSON.stringify(round, null, 2));
