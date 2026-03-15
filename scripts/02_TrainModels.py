"""Entry point for AutoGluon-based model tuning and backtesting."""

import multiprocessing as mp


def run():
    """Tune AutoGluon per round and run walk-forward backtesting."""
    import os

    import pandas as pd
    from models.FitModels import train_models

    data_path = os.path.join(os.path.abspath(os.getcwd()), "data/processed/data.csv")
    data = pd.read_csv(data_path)

    train_models(data)


if __name__ == "__main__":
    mp.freeze_support()
    run()
