"""Get Results."""

# Libraries
import json
import os
import re

import pandas as pd
from models.ModelPipeline import combine_model


def _update_readme_results(points_path, accs_path, readme_path):
    """Rewrite the Backtested Results table in README.md from backtest CSVs.

    Reads picks_points.csv and picks_accuracy.csv produced by combine_model,
    formats each year and its total points as a Markdown table row, and
    appends a weighted overall pick accuracy. Replaces the existing table block
    in the README in-place. The block is identified by the '## Backtested
    Results' heading and replaced up to (but not including) the next '## '
    heading or end of file.

    Weighted accuracy = sum(round_acc * games_in_round) / 63, averaged across
    all backtest years. Games per round: R32=32, S16=16, E8=8, F4=4, NCG=2,
    Winner=1.

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

# Load matchup params, data, and threshold if available
matchup_params = None
matchup_data = None
thresholds = dict.fromkeys(range(2, 8), 0.5)  # default if no tuned thresholds found

threshold_path = os.path.join(cwd, "model/matchup_threshold.json")
if os.path.exists(matchup_params_path) and os.path.exists(matchup_data_path):
    with open(matchup_params_path, "r") as f:
        matchup_params = json.load(f)
    matchup_data = pd.read_csv(matchup_data_path)
    if os.path.exists(threshold_path):
        with open(threshold_path, "r") as f:
            thresholds = {int(k): v for k, v in json.load(f)["thresholds"].items()}
        print(
            f"Matchup model found — bracket correction will be applied (thresholds={thresholds})."
        )
    else:
        print(
            "Matchup model found — bracket correction will be applied (threshold=0.5, not tuned)."
        )
else:
    print("No matchup model found — running without bracket correction.")

# ---------------------------------------------------------------------------
# Backtest and update README
# ---------------------------------------------------------------------------

combine_model(
    data,
    ag_params,
    correct_picks,
    matchup_params=matchup_params,
    matchup_data=matchup_data,
    thresholds=thresholds,
)
_update_readme_results(points_path, accs_path, readme_path)
