"""Generate interactive HTML bracket viewer from picks and actual results."""

import os

from models.utils.BracketViewer import generate

if __name__ == "__main__":
    cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    generate(
        picks_dir=os.path.join(cwd, "results/picks"),
        results_path=os.path.join(cwd, "data/processed/results.json"),
        data_path=os.path.join(cwd, "data/processed/data.csv"),
        output_path=os.path.join(cwd, "results/brackets/brackets.html"),
    )
