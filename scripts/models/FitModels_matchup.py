"""Matchup model hyperparameter tuning using walk-forward CV."""


def train_matchup_model(data, results):
    """Build the matchup dataset, tune AutoGluon, and save frozen params.

    Args:
        data: Full modeling DataFrame (data.csv).
        results: Loaded results.json dict keyed by year string.
    """
    import json
    import os

    from models.utils.autogluon_matchup import tune_matchup_autogluon
    from models.utils.cv_folds import make_folds
    from models.utils.DataProcessing_matchup import build_matchup_dataset

    matchup_data = build_matchup_dataset(data, results)

    matchup_path = os.path.join(os.path.abspath(os.getcwd()), "data/processed/data_matchup.csv")
    matchup_data.to_csv(matchup_path, index=False)

    folds = make_folds(matchup_data)

    print("\nTuning matchup model...")
    result = tune_matchup_autogluon(matchup_data, folds)

    params_path = os.path.join(os.path.abspath(os.getcwd()), "model/autogluon_matchup_params.json")
    os.makedirs(os.path.dirname(params_path), exist_ok=True)
    with open(params_path, "w") as f:
        json.dump(result, f, indent=2)
