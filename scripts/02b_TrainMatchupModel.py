"""Entry point for matchup model tuning."""

import multiprocessing as mp


def run():
    """Build matchup dataset, tune AutoGluon, and save frozen params."""
    import json
    import os

    import pandas as pd
    from models.FitModels_matchup import train_matchup_model

    data_path = os.path.join(os.path.abspath(os.getcwd()), "data/processed/data.csv")
    data = pd.read_csv(data_path)

    results_path = os.path.join(os.path.abspath(os.getcwd()), "data/processed/results.json")
    with open(results_path, "r") as f:
        results = json.load(f)

    train_matchup_model(data, results)


if __name__ == "__main__":
    mp.freeze_support()
    run()
