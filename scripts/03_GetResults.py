"""Get Results."""

import json
import os
import re

import pandas as pd
from models.ModelPipeline import combine_model


def _update_readme_results(points_path, accs_path, readme_path):
    """Rewrite the Backtested Results table in README.md from backtest CSVs.

    Args:
        points_path: Path to picks_points.csv.
        accs_path: Path to picks_accuracy.csv.
        readme_path: Path to README.md.
    """
    points_df = pd.read_csv(points_path)
    accs_df = pd.read_csv(accs_path)

    year_cols = [c for c in points_df.columns if c not in ("Mean", "SD")]
    mean = points_df["Mean"].iloc[0]
    rows = "\n".join(f"| {col} | {int(points_df[col].iloc[0])} |" for col in year_cols)

    games_per_round = {"R32": 32, "S16": 16, "E8": 8, "F4": 4, "NCG": 2, "Winner": 1}
    accs_df = accs_df.set_index("Round")
    total_correct = sum(
        accs_df.loc[rnd, year_cols].astype(float).sum() * games
        for rnd, games in games_per_round.items()
        if rnd in accs_df.index
    )
    total_picks = 63 * len(year_cols)
    weighted_acc = total_correct / total_picks

    new_block = (
        "## Backtested Results\n\n"
        "| Year | Points |\n"
        "|------|--------|\n"
        f"{rows}\n\n"
        f"**Mean: {mean:.0f} pts** &nbsp;|&nbsp; "
        f"**SD: {points_df['SD'].iloc[0]:.0f} pts** &nbsp;|&nbsp; "
        f"**Overall pick accuracy: {weighted_acc:.1%}**\n"
    )

    with open(readme_path, "r") as f:
        content = f.read()

    updated = re.sub(
        r"## Backtested Results\n.*?(?=\n## |\Z)",
        new_block,
        content,
        flags=re.DOTALL,
    )

    with open(readme_path, "w") as f:
        f.write(updated)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

cwd = os.path.abspath(os.getcwd())
data_path = os.path.join(cwd, "data/processed/data.csv")
bracket_path = os.path.join(cwd, "data/processed/results.json")
params_path = os.path.join(cwd, "model/autogluon_params.json")
matchup_params_path = os.path.join(cwd, "model/autogluon_matchup_params.json")
matchup_data_path = os.path.join(cwd, "data/processed/data_matchup.csv")
points_path = os.path.join(cwd, "results/backwards_test/picks_points.csv")
accs_path = os.path.join(cwd, "results/backwards_test/picks_accuracy.csv")
readme_path = os.path.join(cwd, "README.md")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

data = pd.read_csv(data_path)

with open(bracket_path, "r") as json_file:
    correct_picks = json.load(json_file)

with open(params_path, "r") as f:
    ag_params = json.load(f)
ag_params = {int(k): v for k, v in ag_params.items()}

# Load matchup params and data if available
matchup_params = None
matchup_data = None

if os.path.exists(matchup_params_path) and os.path.exists(matchup_data_path):
    with open(matchup_params_path, "r") as f:
        matchup_params = json.load(f)
    matchup_data = pd.read_csv(matchup_data_path)

# ---------------------------------------------------------------------------
# Backtest and update README
# ---------------------------------------------------------------------------

combine_model(
    data,
    ag_params,
    correct_picks,
    matchup_params=matchup_params,
    matchup_data=matchup_data,
)
_update_readme_results(points_path, accs_path, readme_path)
