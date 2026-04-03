"""
opponent.py — Opponent strategies for training MENACE.

Provides two opponents:

  1. Random — picks uniformly from legal moves.  Unpredictable but weak,
     so MENACE should learn to win the majority of games.

  2. Minimax — plays perfectly using exhaustive game-tree search.  It never
     makes a mistake, so the best MENACE can achieve is a draw.  Training
     against minimax tests whether MENACE can converge to truly optimal play.
"""

import random
from functools import lru_cache
from board import get_legal_moves, check_winner


# ──────────────────────────────────────────────────────────────────────────────
# Random opponent
# ──────────────────────────────────────────────────────────────────────────────

def random_opponent_move(board):
    """Pick uniformly at random from the available (empty) cells."""
    return random.choice(get_legal_moves(board))


# ──────────────────────────────────────────────────────────────────────────────
# Minimax opponent (perfect play)
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=None)
def _minimax(board, is_maximizing):
    """
    Classic minimax algorithm with memoization for tic-tac-toe.

    Recursively evaluates every possible future of the game tree to find
    the optimal move.  Because tic-tac-toe's tree is small (~9! leaves at
    most), this runs fast — and lru_cache ensures we never recompute a
    position we've already seen.

    Args:
        board:          current board state (tuple)
        is_maximizing:  True if it's X's turn (maximizer), False for O

    Returns:
        The best score reachable from this position:
            +1 if X (MENACE) can force a win
             0 if optimal play leads to a draw
            -1 if O (opponent) can force a win
    """
    result = check_winner(board)
    if result is not None:
        return result  # terminal state: 1, -1, or 0

    legal = get_legal_moves(board)

    if is_maximizing:
        # X wants the highest possible score
        best = -2
        for move in legal:
            new_board = list(board)
            new_board[move] = 1
            score = _minimax(tuple(new_board), False)
            best = max(best, score)
        return best
    else:
        # O wants the lowest possible score
        best = 2
        for move in legal:
            new_board = list(board)
            new_board[move] = -1
            score = _minimax(tuple(new_board), True)
            best = min(best, score)
        return best


def minimax_opponent_move(board):
    """
    Choose the optimal move for O (the opponent, minimizing player).

    Evaluates every legal move and picks the one that leads to the lowest
    minimax score — i.e., the move that is worst for MENACE.  If multiple
    moves tie for best, one is chosen at random to add slight variety.
    """
    legal = get_legal_moves(board)
    best_score = 2    # start worse than any real score
    best_moves = []

    for move in legal:
        new_board = list(board)
        new_board[move] = -1
        # After O plays, it's X's turn (maximizing)
        score = _minimax(tuple(new_board), True)
        if score < best_score:
            best_score = score
            best_moves = [move]
        elif score == best_score:
            best_moves.append(move)

    # Break ties randomly so MENACE sees some variety even against perfect play
    return random.choice(best_moves)
