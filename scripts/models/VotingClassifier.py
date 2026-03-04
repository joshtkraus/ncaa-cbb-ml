"""Wrapper to tune and export ensemble voting classifier weights."""


def tune_clf(data, split_dict, features_dict=None):
    """Load tuned model params, optimise ensemble blend weights, and save to disk.

    Args:
        data: Full modeling DataFrame.
        split_dict: Dict mapping round number to train/val split ratio.
        features_dict: Optional dict keyed by round number with 'nn' and 'gbm'
            feature lists from feature selection. When provided, each model is
            trained and evaluated on its own surviving feature subset.
    """
    import json
    import os

    from models.utils.voting_clf import tune_weights

    print("Tuning Weights...")

    nn_path = os.path.join(os.path.abspath(os.getcwd()), "models/components/nn.json")
    with open(nn_path, "r") as f:
        nn_params = json.load(f)
    nn_params = {int(k): v for k, v in nn_params.items()}

    gbm_path = os.path.join(os.path.abspath(os.getcwd()), "models/components/gbm.json")
    with open(gbm_path, "r") as f:
        gbm_params = json.load(f)
    gbm_params = {int(k): v for k, v in gbm_params.items()}

    weights = tune_weights(data, split_dict, nn_params, gbm_params, features_dict=features_dict)

    weights_path = os.path.join(os.path.abspath(os.getcwd()), "models/weights.json")
    with open(weights_path, "w") as f:
        json.dump(weights, f)
