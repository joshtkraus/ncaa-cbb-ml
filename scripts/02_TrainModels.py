"""Entry point for AutoGluon-based model tuning and backtesting."""

import multiprocessing as mp
import os

import pandas as pd
from models.FitModels import train_models


def run():
    """Tune AutoGluon per round and run walk-forward backtesting."""
    data_path = os.path.join(os.path.abspath(os.getcwd()), "data/processed/data_norm.csv")
    data = pd.read_csv(data_path)

    train_models(data)


if __name__ == "__main__":
    mp.freeze_support()
    run()
