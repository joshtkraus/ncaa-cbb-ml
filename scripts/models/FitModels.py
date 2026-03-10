"""Hyperparameter tuning for NN and GBM models using grouped round strategy."""


def _tune_group(group_name, feature_round, rounds, data, folds, out_q):
    """Tune NN and GBM hyperparameters for a round group in a subprocess.

    Tunes on the representative feature_round, then applies the resulting
    hyperparameters to all rounds in the group.

    Args:
        group_name: Label for the group (e.g. 'early', 'late').
        feature_round: The round used for feature construction during tuning.
        rounds: List of round numbers in this group.
        data: Full modeling DataFrame.
        folds: List of walk-forward CV fold dicts from make_folds().
        out_q: Multiprocessing Queue to store results.
    """
    from models.utils.gbm import tune_gbm
    from models.utils.nn import tune_nn

    print(f"  Tuning group '{group_name}' (rounds {rounds}, feature round {feature_round})")
    gbm_result = tune_gbm(data, feature_round, folds)
    nn_result = tune_nn(data, feature_round, folds)
    out_q.put((rounds, nn_result, gbm_result))


def train_models(data):
    """Tune NN and GBM hyperparameters using grouped rounds and walk-forward CV.

    Rounds 2-4 share hyperparameters tuned on round 2's feature space.
    Rounds 5-7 share hyperparameters tuned on round 5's feature space.
    Walk-forward CV across all available years gives stable estimates that
    are not dependent on any single validation window.

    Args:
        data: Full modeling DataFrame.
    """
    import json
    import os
    from multiprocessing import Process, Queue

    from models.utils.cv_folds import _ROUND_GROUPS, make_folds

    print("Tuning Models...")

    folds = make_folds(data)
    print(f"  Using {len(folds)} walk-forward CV folds")
    for i, fold in enumerate(folds):
        print(f"    Fold {i + 1}: train {fold['train_years']}, val {fold['val_years']}")

    nn_params = {}
    gbm_params = {}
    results_q = Queue()

    for group_name, group in _ROUND_GROUPS.items():
        p = Process(
            target=_tune_group,
            args=(
                group_name,
                group["feature_round"],
                group["rounds"],
                data,
                folds,
                results_q,
            ),
        )
        p.start()
        p.join()

        rounds, nn_result, gbm_result = results_q.get()
        for r in rounds:
            nn_params[r] = nn_result
            gbm_params[r] = gbm_result

    nn_path = os.path.join(os.path.abspath(os.getcwd()), "models/components/nn.json")
    gbm_path = os.path.join(os.path.abspath(os.getcwd()), "models/components/gbm.json")
    with open(nn_path, "w") as f:
        json.dump(nn_params, f)
    with open(gbm_path, "w") as f:
        json.dump(gbm_params, f)
