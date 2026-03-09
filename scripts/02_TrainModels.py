"""Entry point for tuning models, voting classifier weights, and permutation importance."""

import multiprocessing as mp


def run():
    """Run the full model training pipeline: tune models, weights, and compute importance."""
    import os

    import pandas as pd

    from models.FitModels import train_models
    from models.PermImport import get_importance
    from models.VotingClassifier import tune_clf

    data_path = os.path.join(os.path.abspath(os.getcwd()), "data/processed/data.csv")
    data = pd.read_csv(data_path)

    # Validation start rounds
    split_dict = {2: 2022, 3: 2022, 4: 2019, 5: 2019, 6: 2016, 7: 2016}

    train_models(data, split_dict)
    tune_clf(data, split_dict)
    get_importance(data, split_dict)


if __name__ == "__main__":
    mp.freeze_support()
    run()
