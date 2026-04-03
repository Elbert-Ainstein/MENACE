"""
symmetry.py — Board canonicalization using rotational and reflective symmetry.

Tic-tac-toe has 8 symmetries (the dihedral group D4): 4 rotations (0°, 90°,
180°, 270°) times 2 reflections (identity and horizontal flip).  Two board
positions related by any of these transformations are strategically identical —
for example, opening in the top-left corner is the same as opening in the
bottom-right (just rotated 180°).

By mapping every board state to a single canonical representative, we let
equivalent positions share one drawer.  This collapses the state space from
~5,478 reachable states to ~765 (or ~338 that MENACE actually visits as X),
which means MENACE learns much faster.
"""

import numpy as np


def _board_to_grid(board):
    """Convert a flat 9-tuple into a 3×3 NumPy array for easy transformation."""
    return np.array(board).reshape(3, 3)


def _grid_to_board(grid):
    """Convert a 3×3 NumPy array back to a flat 9-tuple."""
    return tuple(grid.flatten())


def _all_symmetries(board):
    """
    Generate all 8 symmetry-equivalent boards.

    For each of the 4 rotation angles, we produce:
      - the rotated board
      - the rotated-then-horizontally-flipped board
    giving 4 × 2 = 8 total variants.
    """
    grid = _board_to_grid(board)
    variants = []
    for k in range(4):
        rotated = np.rot90(grid, k)
        variants.append(_grid_to_board(rotated))
        variants.append(_grid_to_board(np.fliplr(rotated)))
    return variants


def canonical(board):
    """
    Return the canonical form of a board state.

    We define "canonical" as the lexicographically smallest tuple among all 8
    symmetric variants.  This gives a unique, deterministic representative for
    every equivalence class of board positions.
    """
    return min(_all_symmetries(board))


def canonical_move(board, move):
    """
    Map a move chosen on the canonical board back to the original board.

    When MENACE picks a move on the canonical (transformed) board, we need to
    know which square that corresponds to on the real board.  We do this by:

      1. Building a 3×3 "index grid" where each cell holds its own index (0–8).
      2. Applying the same rotation/flip that maps the original board to its
         canonical form.
      3. Reading off which original index landed at the move's position.

    For example, if 90° rotation canonicalises the board, and MENACE picks
    cell 0 in the canonical view, the index grid (rotated the same way) tells
    us that cell 0 in the rotated view originally came from cell 6 on the
    real board — so we return 6.
    """
    index_grid = np.arange(9).reshape(3, 3)
    board_grid = _board_to_grid(board)
    canon = canonical(board)

    for k in range(4):
        rotated = np.rot90(board_grid, k)

        # Check if this rotation alone produces the canonical form
        if _grid_to_board(rotated) == canon:
            transformed_indices = np.rot90(index_grid, k)
            return int(transformed_indices.flatten()[move])

        # Check if rotation + horizontal flip produces the canonical form
        flipped = np.fliplr(rotated)
        if _grid_to_board(flipped) == canon:
            transformed_indices = np.fliplr(np.rot90(index_grid, k))
            return int(transformed_indices.flatten()[move])

    # Should never happen — one of the 8 symmetries must match
    raise RuntimeError("Could not find symmetry mapping (this is a bug)")
