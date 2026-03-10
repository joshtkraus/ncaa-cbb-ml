"""Wrapper to tune and export ensemble voting classifier weights."""


def tune_clf(data):
    """Load tuned model params, optimise ensemble blend weights, and save to disk.

    Uses walk-forward CV across all available years so that the blend weight
    reflects performance across multiple val windows rather than a single
    fixed split.

    Args:
        data: Full modeling DataFrame.
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

    weights = tune_weights(data, nn_params, gbm_params)

    weights_path = os.path.join(os.path.abspath(os.getcwd()), "models/weights.json")
    with open(weights_path, "w") as f:
        json.dump(weights, f)
