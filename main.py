"""
main.py — Entry point for the MENACE simulation.

Trains two separate MENACE agents:
  1. MENACE vs. Random  — should converge to high win rate
  2. MENACE vs. Minimax — should converge to mostly draws (perfect play)

Produces a side-by-side comparison graph and prints summaries for both.

Usage:
    python main.py
"""

import numpy as np
import matplotlib.pyplot as plt
from menace_agent import MENACE
from game import train
from opponent import random_opponent_move, minimax_opponent_move


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_single(ax, outcomes, window, title):
    """
    Plot rolling-average win/draw/loss rates on a given matplotlib axes.

    Separated from the figure creation so we can reuse it for each subplot
    in the side-by-side comparison.
    """
    n = len(outcomes)

    # Convert outcomes to three binary indicator arrays
    wins   = np.array([1 if o == 1  else 0 for o in outcomes], dtype=float)
    draws  = np.array([1 if o == 0  else 0 for o in outcomes], dtype=float)
    losses = np.array([1 if o == -1 else 0 for o in outcomes], dtype=float)

    # Rolling average via convolution with a uniform kernel
    kernel = np.ones(window) / window
    win_rate  = np.convolve(wins,   kernel, mode="valid")
    draw_rate = np.convolve(draws,  kernel, mode="valid")
    loss_rate = np.convolve(losses, kernel, mode="valid")

    x = np.arange(window, n + 1)

    ax.plot(x, win_rate,  label="Win rate",  color="green",  linewidth=1.5)
    ax.plot(x, draw_rate, label="Draw rate", color="orange", linewidth=1.5)
    ax.plot(x, loss_rate, label="Loss rate", color="red",    linewidth=1.5)
    ax.set_xlabel("Game number")
    ax.set_ylabel(f"Rate (rolling avg, window={window})")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)  # consistent y-axis for easy comparison


def plot_comparison(outcomes_random, outcomes_minimax, window=100,
                    save_path="menace_results.png"):
    """
    Create a side-by-side figure comparing MENACE's performance against
    the random opponent (left) and the minimax opponent (right).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    plot_single(ax1, outcomes_random, window, "MENACE vs. Random Opponent")
    plot_single(ax2, outcomes_minimax, window, "MENACE vs. Minimax Opponent")

    fig.suptitle("MENACE Learning Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Graph saved to {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Summary statistics
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(outcomes, label):
    """Print overall totals and final-100-game performance snapshot."""
    total = len(outcomes)
    wins   = outcomes.count(1)
    draws  = outcomes.count(0)
    losses = outcomes.count(-1)

    print(f"\n{'=' * 50}")
    print(f"MENACE vs. {label}")
    print(f"{'=' * 50}")
    print(f"Total games played: {total}")
    print(f"  Wins:   {wins:>5}  ({100 * wins / total:.1f}%)")
    print(f"  Draws:  {draws:>5}  ({100 * draws / total:.1f}%)")
    print(f"  Losses: {losses:>5}  ({100 * losses / total:.1f}%)")

    # Final 100 games — a snapshot of how MENACE plays once "mature"
    last = outcomes[-100:]
    w = last.count(1)
    d = last.count(0)
    l = last.count(-1)
    print(f"\nFinal 100 games:")
    print(f"  Wins:   {w}%")
    print(f"  Draws:  {d}%")
    print(f"  Losses: {l}%")
    print("=" * 50)


# ──────────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    NUM_GAMES = 5000

    # --- Training session 1: MENACE vs. Random ---
    print(f"Training MENACE vs. Random for {NUM_GAMES} games...")
    menace_rand = MENACE()
    outcomes_random = train(menace_rand, NUM_GAMES, random_opponent_move)
    print(f"  Created {len(menace_rand.drawers)} drawers.")

    # --- Training session 2: MENACE vs. Minimax ---
    print(f"\nTraining MENACE vs. Minimax for {NUM_GAMES} games...")
    menace_mini = MENACE()
    outcomes_minimax = train(menace_mini, NUM_GAMES, minimax_opponent_move)
    print(f"  Created {len(menace_mini.drawers)} drawers.")

    # --- Output ---
    plot_comparison(outcomes_random, outcomes_minimax, window=100)
    print_summary(outcomes_random, "Random Opponent")
    print_summary(outcomes_minimax, "Minimax Opponent (Perfect Play)")
