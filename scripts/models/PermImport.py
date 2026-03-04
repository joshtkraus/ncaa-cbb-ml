"""Wrapper to compute and export permutation importance for all rounds."""


def get_importance(data, split_dict):
    """Load tuned model params and compute permutation importance.

    Loads the surviving feature sets from models/components/features.json
    so that each round/model is evaluated on its selected feature subset.

    Args:
        data: Full modeling DataFrame.
        split_dict: Dict mapping round number to train/val split ratio.
    """
    import json
    import os

    from models.utils.importance import get_importance as _get_importance

    print("Calculating Permutation Importance...")

    nn_path = os.path.join(os.path.abspath(os.getcwd()), "models/components/nn.json")
    with open(nn_path, "r") as f:
        nn_params = json.load(f)
    nn_params = {int(k): v for k, v in nn_params.items()}

    gbm_path = os.path.join(os.path.abspath(os.getcwd()), "models/components/gbm.json")
    with open(gbm_path, "r") as f:
        gbm_params = json.load(f)
    gbm_params = {int(k): v for k, v in gbm_params.items()}

    weights_path = os.path.join(os.path.abspath(os.getcwd()), "models/weights.json")
    with open(weights_path, "r") as f:
        weights = json.load(f)
    weights = {int(k): v for k, v in weights.items()}

    features_path = os.path.join(os.path.abspath(os.getcwd()), "models/components/features.json")
    with open(features_path, "r") as f:
        features_dict = json.load(f)
    features_dict = {int(k): v for k, v in features_dict.items()}

    _get_importance(data, split_dict, nn_params, gbm_params, weights, features_dict)
