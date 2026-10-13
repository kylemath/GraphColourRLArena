# Levels, agents, and proofs

A note for the game team and the proof team. It describes two stalls, and a loop in which a level maker, an agent trainer, and a proof search test one another.

## The training wall

The arena trains agents to finish a four-colouring by flipping Kempe chains. A Kempe chain is a connected patch of the board that uses only two colours. Flipping it swaps those two colours and leaves the colouring legal. The goal is to free a colour for the blank region in the middle, the hole, whose five neighbours currently use all four colours.

Random boards almost never need a long sequence of flips. On a large sample, about three quarters of the unfinished positions are one flip from done, about a quarter are two flips away, and fewer than one in ten thousand are three or more. A search that mutates known boards, by flipping edges and adding or removing regions, did produce many more positions at distance three. Positions at distance four and beyond stayed rare, and the deepest ones still come from one known board and small variations of it.

The agents have learned the common case. With the useful features built in, even a very small network picks a good next flip on ordinary positions. On the rare deep positions, and on boards built to trap a short lookahead, the learned agents and the hand-written rules separate. There is not enough of that rare material to keep training. Making the network wider, on the same stream of random boards, practices the easy case again.

## The ideas wall

The four colour theorem has a checked proof. It does not have a short argument that a person can see in one picture. Hand-designed rules for Kempe flips work on a single chain and fail when two chains of the relevant colours meet. That meeting is the case we keep running into, and it is also the case random boards almost never contain.

A search for a proof, judged only by a proof checker, is under way in `bitter-prover`. It is not aimed at the full theorem yet. The first targets are small facts whose proofs are already known and are hidden from the search. Until those have a published record of success against a fixed budget of attempts, a clever colouring agent is not evidence for the theorem.

So both efforts are short of the same thing. The arena is short of positions that require a real decision. The proof search is short of a next idea that is allowed to count. Each has something the other needs.

## Three jobs

**Level maker.** Builds boards. Starts from a hard piece, then adds material beside it: a quiet outer ring, a longer route for the same chain, or a second hard piece. Also tries to build the chains directly, two chains that cross, then three, at every length and every attachment to the rim.

**Agent trainer.** The arena. Teaches networks to choose flips, and races them against older agents: a simple hand rule, a two-step lookahead, an exact shortest-path search when the board is small enough, and whatever rule the proof team last proposed.

**Proof search.** Looks for arguments a checker will accept. When a candidate rule or a candidate configuration gets stuck, that stuck case is handed back as a board.

## The loop

The three jobs run as one cycle.

**When the agents stop improving,** they ask the level maker for harder boards. That is the situation now. More training on random boards will not move the deep cases. The level maker answers with boards grown from the positions the current agents miss, not with another random sample.

**When the level maker needs to know what "hard" means,** it asks the proof search. A wall there is a configuration the latest solver candidate cannot reduce, or a sketch of crossing chains that cannot be completed. Those are the seeds. The level maker grows them into boards. It does not invent hardness by searching the whole tree of colourings of a large random graph. That search is the thing we already cannot finish.

**When the proof team wants to test a solver,** it sends the rule to the arena. The arena runs it on the same boards as the previous rules, the two-step lookahead, the exact search, and the trained networks. The report is several numbers, each answering a different question:

- On an ordinary position, did it pick a flip that shortens the solution?
- On a position that is still three or more flips from done, did it do the same?
- Did it actually finish the board?
- How many flips did it take, compared with the shortest solution?
- Where did it die: which board, which colouring, which chain it played, and which chain would have helped?

A rule that looks good on paper and loses these races goes back to the proof team with those dying positions attached. A rule that wins them becomes part of the roster the level maker has to beat.

Then the cycle repeats. Agents improve, the level maker has to move the wall, the proof search either explains the new wall or gets stuck on it, and the stuck case becomes the next seed.

## What "hard" means for each team

These are different, and mixing them up is how the last search went stale.

For a **player**, hard means a long chain that is satisfying to watch, plus a few short chains that look right and are wrong. The right flip is often a short chain in the middle that never touches the five neighbours of the hole. A board can be easy to prove and still be a good level. Extra material around a known hard piece is allowed to be something a proof would ignore, as long as it does not open a shortcut. If a splice opens a shortcut, the board looks busy and solves in one obvious move. That board is thrown away.

For an **agent**, hard means a position where the current roster plays the tempting chain and the helpful chain is a different one. The level maker is scored on that gap. "As far from finished as possible" is the wrong score. A long corridor can have an obvious next step. A position one flip from done can be a hard choice.

For a **proof**, hard means a statement the checker does not yet accept. A finished colouring, a high arena score, and a beautiful long chain are not that statement. The bare hard piece, without the outer ring built for the player, is the object a later theorem talks about.

## What already happened

A first race is in the arena. On a small training slice, the two-step lookahead beat both the hand rule and a new rule that always prefers the shortest chain touching the rim. The roster is not yet good enough on positions three flips from done, so the level maker is not being asked to raise an already-cleared bar. It is being handed the misses. In those misses the helpful chain often does not touch the rim at all.

The proof search has a working checker for a toy theory and for a pool of planted arithmetic proofs. The first real search, on a small colouring fact with the write-up hidden, is the next piece of that project. It should not be pointed at the full theorem first.

## What each team does next

**Level maker.** Take the misses from the current roster. Rebuild them as two crossing chains, then three. Add an outer ring only when the original chains are still the ones that meet the rim. Keep a second copy of each board with the ring stripped off, and send that bare copy toward the proof team.

**Agent trainer.** Keep racing every new rule against the hand rule, the two-step lookahead, the exact search, and the latest network. When the best of these clears the current boards, ask the level maker for the next wall. Until then, train on the misses, not on more random boards.

**Proof search.** Publish the first small search, the missing-colour fact, with the known write-up left out of the hints. When a later candidate stalls, send the stall to the level maker as a seed and to the arena as a rule. Do not put the arena's favourite flip into the list of hints.

## What does not count as progress

- A network that plays random boards perfectly and still misses the same deep boards.
- A level that is large to look at and easy to solve, because the splice opened a shortcut.
- An arena win reported as a proof.
- A proof search aimed at the full theorem before the small hidden proofs have a published record.
