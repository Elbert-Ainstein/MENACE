# MENACE — Machine Educable Noughts and Crosses Engine

A Python simulation of Donald Michie's 1961 reinforcement-learning system for tic-tac-toe, built for CS 580 Machine Learning (Spring 2026).

Note: I read through all the code to ensure the logic is correct.

## What it does

MENACE learns to play tic-tac-toe by adjusting "bead counts" in virtual drawers — one drawer per unique board state. Moves are chosen by weighted random sampling, and after each game the beads are updated: +3 for a win, +1 for a draw, -1 for a loss. Over thousands of games against a random opponent, MENACE converges toward strong play without any hardcoded strategy.

Board states are canonicalized using all 8 rotational/reflective symmetries (the dihedral group D4), collapsing ~5,478 reachable states into ~338 unique drawers.

## Project structure

| File | Purpose |
|------|---------|
| `main.py` | Entry point — runs training and produces the graph + summary |
| `menace_agent.py` | The MENACE agent class (drawers, move selection, reinforcement) |
| `board.py` | Board representation, legal moves, win/draw detection |
| `symmetry.py` | Canonicalization via D4 symmetry transformations |
| `game.py` | Game simulation and training loop |
| `opponent.py` | Random opponent strategy |

## Setup and running

Requires Python 3.8+.

```
pip install -r requirements.txt
python main.py
```

This will train MENACE for 5,000 games and output:
- `menace_results.png` — rolling-average win/draw/loss rates over training
- A printed summary of overall and final-100-game statistics

## Sample output

```
MENACE created 338 unique drawers (canonical states).

Total games played: 5000
  Wins:   3799  (76.0%)
  Draws:   532  (10.6%)
  Losses:  669  (13.4%)

Final 100 games:
  Wins:   83%
  Draws:  14%
  Losses: 3%
```
