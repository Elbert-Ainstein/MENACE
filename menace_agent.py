"""
menace_agent.py — The MENACE reinforcement-learning agent.

MENACE (Machine Educable Noughts and Crosses Engine) learns to play
tic-tac-toe through a simple but elegant mechanism:

  - Each unique board state has a "drawer" (dictionary) mapping legal moves
    to bead counts.
  - To choose a move, MENACE samples randomly, weighted by bead counts —
    moves with more beads are more likely to be picked.
  - After each game, bead counts are adjusted for every move MENACE played:
        Win:  +3 beads  (strongly reinforce winning lines of play)
        Draw: +1 bead   (mildly reinforce — draws are acceptable)
        Loss: -1 bead   (discourage losing lines of play)
  - Over many games, bead distributions shift toward strong play without
    any explicit strategy programming.
"""

import random
from board import get_legal_moves
from symmetry import canonical, canonical_move


class MENACE:
    """
    The MENACE agent.

    Attributes:
        drawers : dict[tuple, dict[int, int]]
            Maps canonical board states to drawers.  Each drawer maps
            move indices (in canonical coordinates) to bead counts.

        history : list[tuple[tuple, int]]
            The (canonical_state, canonical_move) pairs from the current game,
            used for post-game reinforcement.
    """

    # Bead adjustments applied after each game
    WIN_REWARD   =  3
    DRAW_REWARD  =  1
    LOSS_PENALTY = -1

    def __init__(self):
        self.drawers = {}   # created lazily as new states are encountered
        self.history = []   # reset after each game

    @staticmethod
    def _initial_beads_for_state(canon_board):
        """
        Determine how many starting beads each move gets in a new drawer.

        Early board positions (few pieces placed) have many legal moves and
        high strategic importance, so they get more beads to encourage broad
        exploration.  Later positions have fewer options and converge faster,
        so fewer starting beads suffice.

        We count how many of MENACE's pieces (X = 1) are on the board to
        determine the move number, then scale beads down accordingly:
            Move 1 (empty board)  → 8 beads per option
            Move 2 (2 pieces)     → 6 beads per option
            Move 3 (4 pieces)     → 4 beads per option
            Move 4 (6 pieces)     → 2 beads per option
            Move 5 (8 pieces)     → 1 bead  per option  (floor)

        This follows Michie's original insight: give the machine more room
        to explore where the decision tree is widest.
        """
        menace_pieces = canon_board.count(1)   # how many X's are placed
        # menace_pieces = 0 means it's MENACE's 1st move, 1 means 2nd, etc.
        beads = max(1, 8 - (menace_pieces * 2))
        return beads

    def _get_or_create_drawer(self, canon_board):
        """
        Retrieve the drawer for a canonical state, creating it if new.

        New drawers are initialized with a bead count that scales with
        how early in the game this state occurs — more beads for earlier,
        more consequential positions.
        """
        if canon_board not in self.drawers:
            legal = get_legal_moves(canon_board)
            beads = self._initial_beads_for_state(canon_board)
            self.drawers[canon_board] = {m: beads for m in legal}
        return self.drawers[canon_board]

    def choose_move(self, board):
        """
        Select a move for the given board state.

        Process:
          1. Canonicalize the board so we use the right drawer.
          2. Look up (or create) the drawer for this state.
          3. Sample a move weighted by bead counts (more beads = higher chance).
          4. Record the choice in history for post-game reinforcement.
          5. Map the canonical move back to the real board's coordinates.

        Returns:
            An index 0–8 in the original (non-canonical) board's frame.
        """
        canon = canonical(board)
        drawer = self._get_or_create_drawer(canon)

        # Separate the moves and weights for random.choices()
        moves = list(drawer.keys())
        weights = list(drawer.values())

        # Weighted random selection — the core of MENACE's decision-making
        canon_move = random.choices(moves, weights=weights, k=1)[0]

        # Log this decision for reinforcement later
        self.history.append((canon, canon_move))

        # Translate from canonical coordinates back to the actual board
        return canonical_move(board, canon_move)

    def reinforce(self, outcome):
        """
        Update bead counts for every move MENACE played this game.

        Args:
            outcome: 1 (win), 0 (draw), or -1 (loss)

        Beads are floored at 0 — they can never go negative.  If a drawer
        is completely emptied (all beads at 0), we reinitialize it with 1
        bead per move.  This prevents MENACE from ever being "stuck" with
        no options; it simply becomes uniformly random in that state until
        it discovers better play.
        """
        # Determine how many beads to add/remove per move
        if outcome == 1:
            delta = self.WIN_REWARD
        elif outcome == 0:
            delta = self.DRAW_REWARD
        else:
            delta = self.LOSS_PENALTY

        # Apply the adjustment to every move made this game
        for canon_state, canon_move in self.history:
            drawer = self.drawers[canon_state]
            drawer[canon_move] = max(0, drawer[canon_move] + delta)

            # Safety net: if all beads are gone, reset to uniform
            if all(v == 0 for v in drawer.values()):
                for key in drawer:
                    drawer[key] = 1

        # Clear history for the next game
        self.history = []
