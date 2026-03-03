"""Wrapper to tune and export ensemble voting classifier weights."""


def tune_clf(data, split_dict):
    """Load tuned model params, optimise ensemble blend weights, and save to disk.

    Args:
        data: Full modeling DataFrame.
        split_dict: Dict mapping round number to train/val split ratio.
    """
    import json
    import os

    from models.utils.voting_clf import tune_weights

    print("Tuning Weights...")

    nn_path = os.path.join(os.path.abspath(os.getcwd()), "models/components/nn.json")
    with open(nn_path, "r") as json_file:
        nn_params = json.load(json_file)
    nn_params = {int(key): value for key, value in nn_params.items()}

    gbm_path = os.path.join(os.path.abspath(os.getcwd()), "models/components/gbm.json")
    with open(gbm_path, "r") as json_file:
        gbm_params = json.load(json_file)
    gbm_params = {int(key): value for key, value in gbm_params.items()}

    weights = tune_weights(data, split_dict, nn_params, gbm_params)

    weights_path = os.path.join(os.path.abspath(os.getcwd()), "models/weights.json")
    with open(weights_path, "w") as f:
        json.dump(weights, f)
