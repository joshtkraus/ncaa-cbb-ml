"""Voting classifier weight tuning using Optuna to minimize Brier Score."""


def get_pred(
    X_train_nn,
    X_train_gbm,
    X_val_nn,
    X_val_gbm,
    y_train_nn,
    y_train_gbm,
    y_val,
    nn_params,
    gbm_params,
):
    """Fit NN and GBM models and return their validation predictions.

    Args:
        X_train_nn: Training features for the NN.
        X_train_gbm: Training features for the GBM.
        X_val_nn: Validation features for the NN.
        X_val_gbm: Validation features for the GBM.
        y_train_nn: Training labels for the NN.
        y_train_gbm: Training labels for the GBM.
        y_val: Validation labels shared by both models.
        nn_params: Tuned NN hyperparameter dict.
        gbm_params: Tuned GBM hyperparameter dict.

    Returns:
        Tuple of (nn_probabilities, gbm_probabilities) on the validation set.
    """
    import xgboost as xgb

    from models.utils.gbm import tuned_gbm
    from models.utils.nn import tuned_nn

    nn = tuned_nn(nn_params, X_train_nn, y_train_nn, X_val_nn, y_val)
    prob_nn = nn.predict(X_val_nn, verbose=0)

    gbm = tuned_gbm(gbm_params, X_train_gbm, y_train_gbm, X_val_gbm, y_val)
    dval = xgb.DMatrix(X_val_gbm)
    prob_gbm = gbm.predict(dval)

    return prob_nn[:, 0], prob_gbm


def objective(trial, prob_nn, prob_gbm, y_val):
    """Optuna objective function to minimize Brier Score for ensemble weights.

    Args:
        trial: Optuna trial object.
        prob_nn: NN predicted probabilities on the validation set.
        prob_gbm: GBM predicted probabilities on the validation set.
        y_val: True validation labels.

    Returns:
        Brier score of the weighted ensemble on the validation set.
    """
    from sklearn.metrics import brier_score_loss

    w = trial.suggest_float("weight", 0, 1)
    combined_probs = w * prob_nn + (1 - w) * prob_gbm
    return brier_score_loss(y_val, combined_probs)


def tune_weights(data, split_dict, nn_params, gbm_params, n_trials=100):
    """Tune ensemble blend weights for all rounds using Optuna.

    Args:
        data: Full modeling DataFrame.
        split_dict: Dict mapping round number to train/val split ratio.
        nn_params: Dict of tuned NN hyperparameters keyed by round number.
        gbm_params: Dict of tuned GBM hyperparameters keyed by round number.
        n_trials: Number of Optuna trials per round (default 100).

    Returns:
        Dict of ensemble weights keyed by round number, each with 'NN' and 'GBM' keys.
    """
    import numpy as np
    import optuna

    from models.utils.DataProcessing import create_splits

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    weights = {}
    for r in range(2, 8):
        print("Round " + str(r))
        weights[r] = {}

        X_SMTL_nn, y_SMTL_nn = create_splits(data, r, train=True)
        X_nn, y = create_splits(data, r, train=False)
        X_SMTL_gbm, y_SMTL_gbm = create_splits(data, r, train=True)
        X_gbm, _ = create_splits(data, r, train=False)

        split_idx = int(split_dict[r] * len(X_nn))
        split_idx_SMTL = np.where((X_SMTL_nn == X_nn[split_idx]).all(axis=1))[0][0]

        X_train_nn, y_train_nn = X_SMTL_nn[:split_idx_SMTL], y_SMTL_nn[:split_idx_SMTL]
        X_val_nn, y_val = X_nn[split_idx:], y[split_idx:]
        X_train_gbm, y_train_gbm = X_SMTL_gbm[:split_idx_SMTL], y_SMTL_gbm[:split_idx_SMTL]
        X_val_gbm = X_gbm[split_idx:]

        prob_nn, prob_gbm = get_pred(
            X_train_nn,
            X_train_gbm,
            X_val_nn,
            X_val_gbm,
            y_train_nn,
            y_train_gbm,
            y_val,
            nn_params[r],
            gbm_params[r],
        )

        # Capture loop variables explicitly to avoid B023 late-binding closure issue
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial, pn=prob_nn, pg=prob_gbm, yv=y_val: objective(trial, pn, pg, yv),
            n_trials=n_trials,
        )

        weights[r]["NN"] = study.best_params["weight"]
        weights[r]["GBM"] = 1 - study.best_params["weight"]

    return weights
