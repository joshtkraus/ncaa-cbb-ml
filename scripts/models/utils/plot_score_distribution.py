"""Plot actual points by simulation rank across all backtest years.

Exposes plot_score_distributions() for use in ModelPipeline.py, and can also
be run as a standalone script if score_distributions.json already exists.

Usage (standalone):
    python plot_score_distribution.py
"""

import os

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — required when running headless
import json

import matplotlib.pyplot as plt
import numpy as np


def plot_score_distributions(score_distributions, out_path):
    """Plot actual points by simulation rank for all backtest years.

    Each year is a thin dashed line. The cross-year mean is a thick solid line.
    X-axis is simulation rank (1 = highest combined score, 1000 = lowest).
    Y-axis is actual points scored against real tournament results.

    Args:
        score_distributions: Dict keyed by test year (int) -> list of actual
            scores in simulation rank order (index 0 = top-ranked candidate).
        out_path: Path to write the PNG output.
    """
    data_by_year = {int(yr): np.array(scores) for yr, scores in score_distributions.items()}
    years = sorted(data_by_year.keys())
    n_candidates = len(next(iter(data_by_year.values())))

    # Smooth by averaging every 10 brackets
    bin_size = 10
    n_bins = n_candidates // bin_size
    bin_centers = np.arange(1, n_bins + 1) * bin_size - (bin_size // 2)

    def _smooth(arr):
        return arr[: n_bins * bin_size].reshape(n_bins, bin_size).mean(axis=1)

    matrix = np.vstack([data_by_year[yr] for yr in years])
    mean_curve = _smooth(matrix.mean(axis=0))

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(years)))
    for yr, color in zip(years, colors, strict=False):
        ax.plot(
            bin_centers,
            _smooth(data_by_year[yr]),
            color=color,
            linewidth=0.8,
            linestyle="--",
            alpha=0.6,
            label=str(yr),
        )

    ax.plot(
        bin_centers,
        mean_curve,
        color="black",
        linewidth=2.5,
        linestyle="-",
        label="Mean",
        zorder=5,
    )

    ax.set_xlabel("Simulation Rank (1 = highest combined score, averaged per 10)", fontsize=12)
    ax.set_ylabel("Actual Points Scored", fontsize=12)
    ax.set_title("Actual Points by Simulation Rank — All Backtest Years", fontsize=13)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(bin_centers[0], bin_centers[-1])

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    cwd = os.path.abspath(os.getcwd())
    dist_path = os.path.join(cwd, "results/backwards_test/score_distributions.json")
    out_path = os.path.join(cwd, "results/backwards_test/score_distribution.png")

    with open(dist_path) as f:
        raw = json.load(f)

    score_distributions = {int(yr): scores for yr, scores in raw.items()}
    plot_score_distributions(score_distributions, out_path)
