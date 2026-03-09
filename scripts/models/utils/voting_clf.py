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

    w = trial.suggest_float("weight", 0.3, 0.7)
    combined_probs = w * prob_nn + (1 - w) * prob_gbm
    return brier_score_loss(y_val, combined_probs)


def tune_weights(data, split_dict, nn_params, gbm_params, n_trials=100, features_dict=None):
    """Tune ensemble blend weights for all rounds using Optuna.

    When features_dict is provided, each model is trained and evaluated on its
    own surviving feature subset so that the blended weight reflects performance
    on the same columns used during inference.

    Args:
        data: Full modeling DataFrame.
        split_dict: Dict mapping round number to validation start year.
        nn_params: Dict of tuned NN hyperparameters keyed by round number.
        gbm_params: Dict of tuned GBM hyperparameters keyed by round number.
        n_trials: Number of Optuna trials per round (default 100).
        features_dict: Optional dict keyed by round number with 'nn' and 'gbm'
            feature lists. When None, all features are used for both models.

    Returns:
        Dict of ensemble weights keyed by round number, each with 'NN' and 'GBM' keys.
    """
    import optuna

    from models.utils.DataProcessing import apply_smote, create_splits
    from models.utils.importance import _dropped_cols

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    weights = {}
    for r in range(2, 8):
        print("Round " + str(r))
        weights[r] = {}

        drop_nn = _dropped_cols(data, r, features_dict, "nn") if features_dict else None
        drop_gbm = _dropped_cols(data, r, features_dict, "gbm") if features_dict else None

        # NN fold
        X_train_nn, X_val_nn, y_train_nn, y_val, _ = create_splits(
            data, r, val_start=split_dict[r], drop_cols=drop_nn
        )
        X_train_nn_res, y_train_nn_res = apply_smote(X_train_nn, y_train_nn)

        # GBM fold
        X_train_gbm, X_val_gbm, y_train_gbm, _, _ = create_splits(
            data, r, val_start=split_dict[r], drop_cols=drop_gbm
        )
        X_train_gbm_res, y_train_gbm_res = apply_smote(X_train_gbm, y_train_gbm)

        prob_nn, prob_gbm = get_pred(
            X_train_nn_res,
            X_train_gbm_res,
            X_val_nn,
            X_val_gbm,
            y_train_nn_res,
            y_train_gbm_res,
            y_val,
            nn_params[r],
            gbm_params[r],
        )

        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial, pn=prob_nn, pg=prob_gbm, yv=y_val: objective(trial, pn, pg, yv),
            n_trials=n_trials,
        )

        weights[r]["NN"] = study.best_params["weight"]
        weights[r]["GBM"] = 1 - study.best_params["weight"]

    return weights
