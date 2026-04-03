"""
game.py — Game simulation and training loop.

Handles playing individual games (MENACE vs. any opponent) and running the
full training loop that lets MENACE accumulate experience over thousands
of games.  The opponent is passed in as a function, so the same loop works
for both the random and minimax opponents.
"""

from board import EMPTY_BOARD, check_winner
from opponent import random_opponent_move


def play_game(menace, opponent_fn=random_opponent_move):
    """
    Play one complete game: MENACE (X, first mover) vs. an opponent (O).

    The opponent is a callable that takes a board tuple and returns a move
    index.  This lets us swap between random, minimax, or any other strategy
    without changing the game logic.

    Args:
        menace:      a MENACE agent instance
        opponent_fn: function(board) -> move index  (default: random)

    Returns:
         1  — MENACE wins
        -1  — MENACE loses
         0  — draw
    """
    board = list(EMPTY_BOARD)  # mutable copy for in-place updates

    while True:
        # --- MENACE's turn (X = 1) ---
        move = menace.choose_move(tuple(board))
        board[move] = 1
        result = check_winner(tuple(board))
        if result is not None:
            return result

        # --- Opponent's turn (O = -1) ---
        move = opponent_fn(tuple(board))
        board[move] = -1
        result = check_winner(tuple(board))
        if result is not None:
            return result


def train(menace, num_games=5000, opponent_fn=random_opponent_move):
    """
    Train MENACE over many games, collecting outcome data.

    Each game:
      1. Play a full game of MENACE vs. the given opponent.
      2. Use the outcome to reinforce MENACE's drawers (adjust beads).
      3. Log the result for later analysis.

    Args:
        menace:      a MENACE agent instance
        num_games:   how many games to play (default 5000)
        opponent_fn: function(board) -> move index  (default: random)

    Returns:
        outcomes: list of results (1, 0, or -1) for each game
    """
    outcomes = []

    for game_num in range(num_games):
        result = play_game(menace, opponent_fn)
        menace.reinforce(result)
        outcomes.append(result)

    return outcomes
