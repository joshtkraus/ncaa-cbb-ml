"""AutoGluon-based hyperparameter tuning using walk-forward CV per round."""


def train_models(data):
    """Tune AutoGluon per round and save the best model config for each.

    For each round (2–7), runs AutoGluon with best_quality preset across
    walk-forward CV folds to identify the single best model type and exact
    hyperparameter configuration. The winner is selected by lowest mean
    val-set Brier score across folds.

    Results are saved to model/autogluon_params.json keyed by round number.
    This replaces the separate nn.json and gbm.json files.

    Args:
        data: Full modeling DataFrame.
    """
    import json
    import os

    from models.utils.autogluon import tune_autogluon
    from models.utils.cv_folds import make_folds

    print("Tuning Models...")

    folds = make_folds(data)
    print(f"  Using {len(folds)} walk-forward CV folds")
    for i, fold in enumerate(folds):
        print(f"    Fold {i + 1}: train {fold['train_years']}, val {fold['val_years']}")

    ag_params = {}

    for r in range(2, 8):
        print(f"\n  Round {r}")
        result = tune_autogluon(data, r, folds)
        ag_params[r] = result
        print(f"  Round {r} → model: {result['model_type']}, params: {result['hyperparameters']}")

    params_path = os.path.join(os.path.abspath(os.getcwd()), "model/autogluon_params.json")
    with open(params_path, "w") as f:
        json.dump(ag_params, f, indent=2)

    print(f"\nSaved AutoGluon params to {params_path}")
