"""Hyperparameter tuning for NN and GBM models for each tournament round."""


def _tune_round(r, data, split_dict, out_q, drop_cols_nn=None, drop_cols_gbm=None):
    """Tune NN and GBM hyperparameters for a single round in a subprocess.

    Args:
        r: Tournament round number (2–7).
        data: Full modeling DataFrame.
        split_dict: Dict mapping round number to train/val split ratio.
        out_q: Multiprocessing Queue to store results.
        drop_cols_nn: Optional list of feature names to exclude for NN tuning.
        drop_cols_gbm: Optional list of feature names to exclude for GBM tuning.
    """
    from models.utils.gbm import tune_gbm
    from models.utils.nn import tune_nn

    nn_result = tune_nn(data, r, split_dict, drop_cols=drop_cols_nn)
    gbm_result = tune_gbm(data, r, split_dict, drop_cols=drop_cols_gbm)
    out_q.put((r, nn_result, gbm_result))


def train_models(data, split_dict, features_dict=None):
    """Tune NN and GBM hyperparameters for all rounds and save results to disk.

    When features_dict is provided, each model is tuned on its own surviving
    feature subset identified during feature selection.

    Args:
        data: Full modeling DataFrame.
        split_dict: Dict mapping round number to train/val split ratio.
        features_dict: Optional dict keyed by round number with 'nn' and 'gbm'
            feature lists. When None, all features are used for both models.
    """
    import json
    import os
    from multiprocessing import Process, Queue

    from models.utils.importance import _dropped_cols

    print("Tuning Models...")

    nn_params = {}
    gbm_params = {}
    results_q = Queue()

    for r in range(2, 8):
        print("Round", r)
        drop_nn = _dropped_cols(data, r, features_dict, "nn") if features_dict else None
        drop_gbm = _dropped_cols(data, r, features_dict, "gbm") if features_dict else None
        p = Process(
            target=_tune_round,
            args=(r, data, split_dict, results_q),
            kwargs={"drop_cols_nn": drop_nn, "drop_cols_gbm": drop_gbm},
        )
        p.start()
        p.join()

        round_num, nn_result, gbm_result = results_q.get()
        nn_params[round_num] = nn_result
        gbm_params[round_num] = gbm_result

    nn_path = os.path.join(os.path.abspath(os.getcwd()), "models/components/nn.json")
    gbm_path = os.path.join(os.path.abspath(os.getcwd()), "models/components/gbm.json")
    with open(nn_path, "w") as f:
        json.dump(nn_params, f)
    with open(gbm_path, "w") as f:
        json.dump(gbm_params, f)
