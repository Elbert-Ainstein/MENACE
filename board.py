"""
board.py — Tic-tac-toe board representation and game logic.

The board is a tuple of 9 integers in row-major order:
    Index layout:       Values:
    0 | 1 | 2           0 = empty
    ---------            1 = X (MENACE)
    3 | 4 | 5          -1 = O (opponent)
    ---------
    6 | 7 | 8

We use a tuple (not a list) so board states are hashable and can serve
as dictionary keys in MENACE's drawer system.
"""

# A fresh board — all nine cells empty
EMPTY_BOARD = (0,) * 9


def print_board(board):
    """Pretty-print the board for debugging. Not used during training."""
    symbols = {0: ".", 1: "X", -1: "O"}
    for row in range(3):
        print(" ".join(symbols[board[row * 3 + col]] for col in range(3)))
    print()


def get_legal_moves(board):
    """Return a list of indices (0–8) that are still empty."""
    return [i for i in range(9) if board[i] == 0]


def check_winner(board):
    """
    Determine the current game state.

    Returns:
         1   — X (MENACE) has three in a row
        -1   — O (opponent) has three in a row
         0   — board is full, no winner (draw)
        None — game is still in progress

    We check all 8 possible winning lines (3 rows, 3 columns, 2 diagonals).
    Summing each line exploits our +1/-1 encoding: a sum of +3 means X owns
    all three cells, -3 means O does.
    """
    lines = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
        (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
        (0, 4, 8), (2, 4, 6),              # diagonals
    ]
    for a, b, c in lines:
        total = board[a] + board[b] + board[c]
        if total == 3:
            return 1   # X wins
        if total == -3:
            return -1  # O wins

    # No winner found — is the board full?
    if 0 not in board:
        return 0       # draw
    return None        # game still in progress
