"""AutoGluon tuning using walk-forward CV per round."""


def train_models(data):
    """Tune AutoGluon per round and save the best model config for each.

    Args:
        data: Full modeling DataFrame.
    """
    import json
    import os

    from models.utils.autogluon import tune_autogluon
    from models.utils.cv_folds import make_folds

    print("Tuning Models...")

    folds = make_folds(data, n_folds=3)

    ag_params = {}

    for r in range(2, 8):
        print(f"\n  Round {r}")
        result = tune_autogluon(data, r, folds)
        ag_params[r] = result

    params_path = os.path.join(os.path.abspath(os.getcwd()), "model/autogluon_params.json")
    with open(params_path, "w") as f:
        json.dump(ag_params, f, indent=2)
