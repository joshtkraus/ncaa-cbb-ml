"""Hyperparameter tuning for NN and GBM models for each tournament round."""


def _tune_round(r, data, split_dict, out_q):
    """Tune NN and GBM hyperparameters for a single round in a subprocess.

    Args:
        r: Tournament round number (2–7).
        data: Full modeling DataFrame.
        split_dict: Dict mapping round number to train/val split ratio.
        out_q: Multiprocessing Queue to store results.
    """
    from models.utils.gbm import tune_gbm
    from models.utils.nn import tune_nn

    nn_result = tune_nn(data, r, split_dict)
    gbm_result = tune_gbm(data, r, split_dict)
    out_q.put((r, nn_result, gbm_result))


def train_models(data, split_dict):
    """Tune NN and GBM hyperparameters for all rounds and save results to disk.

    Args:
        data: Full modeling DataFrame.
        split_dict: Dict mapping round number to train/val split ratio.
    """
    import json
    import os
    from multiprocessing import Process, Queue

    print("Tuning Models...")

    nn_params = {}
    gbm_params = {}
    results_q = Queue()

    for r in range(2, 8):
        print("Round", r)
        p = Process(target=_tune_round, args=(r, data, split_dict, results_q))
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
