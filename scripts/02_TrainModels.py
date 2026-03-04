"""Entry point for tuning models, voting classifier weights, and permutation importance."""

import multiprocessing as mp


def run():
    """Run the full model training pipeline: tune models, weights, and compute importance."""
    import os

    import pandas as pd

    from models.FeatureSelection import select_features
    from models.FitModels import train_models
    from models.PermImport import get_importance
    from models.VotingClassifier import tune_clf

    data_path = os.path.join(os.path.abspath(os.getcwd()), "data/processed/data.csv")
    data = pd.read_csv(data_path)

    # 18 years (2007-2025, excl. 2020): 15/3 val split for R32/S16,
    # 13/5 for E8/F4, 11/7 for NCG/Winner
    split_dict = {2: 0.8333, 3: 0.8333, 4: 0.7222, 5: 0.7222, 6: 0.6111, 7: 0.6111}

    train_models(data, split_dict)
    features_dict = select_features(data, split_dict)
    tune_clf(data, split_dict, features_dict=features_dict)
    get_importance(data, split_dict)


if __name__ == "__main__":
    mp.freeze_support()
    run()
