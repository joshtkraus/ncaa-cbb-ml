"""Wrapper to compute and export permutation importance for all rounds."""


def get_importance(data, split_dict):
    """Load tuned model params and compute SHAP-based permutation importance.

    Args:
        data: Full modeling DataFrame.
        split_dict: Dict mapping round number to train/val split ratio.
    """
    import json
    import os

    from models.utils.importance import get_importance as _get_importance

    print("Calculating Permutation Importance...")

    nn_path = os.path.join(os.path.abspath(os.getcwd()), "models/components/nn.json")
    with open(nn_path, "r") as json_file:
        nn_params = json.load(json_file)
    nn_params = {int(key): value for key, value in nn_params.items()}

    gbm_path = os.path.join(os.path.abspath(os.getcwd()), "models/components/gbm.json")
    with open(gbm_path, "r") as json_file:
        gbm_params = json.load(json_file)
    gbm_params = {int(key): value for key, value in gbm_params.items()}

    weights_path = os.path.join(os.path.abspath(os.getcwd()), "models/weights.json")
    with open(weights_path, "r") as json_file:
        weights = json.load(json_file)
    weights = {int(key): value for key, value in weights.items()}

    _get_importance(data, split_dict, nn_params, gbm_params, weights)
