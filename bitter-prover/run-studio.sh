#!/bin/zsh
# One Lean process per logical CPU, each single-threaded.
# On a Mac Studio the core count comes from the machine. A longer run is a
# larger --goals or --budget, not a different program.
# Summary: runs/studio/summary.json
cd "$(dirname "$0")"
workers=$(sysctl -n hw.ncpu)
exec ../../.venv/bin/python -m bitter_prover.scaled \
  --workers "$workers" \
  --goals 2000 \
  --budget 12 \
  --rounds 2 \
  --width 256 \
  --seed 0 \
  --output runs/studio
