"""XGBoost GBM model definition, Optuna tuning, and training utilities."""


def set_seed(seed=23):
    """Set random seeds for reproducibility.

    Args:
        seed: Integer seed value (default 23).
    """
    import random

    import numpy as np

    np.random.seed(seed)
    random.seed(seed)


def objective(trial, X_train, X_val, y_train, y_val):
    """Optuna objective function for XGBoost hyperparameter tuning.

    Args:
        trial: Optuna trial object.
        X_train: Training feature array.
        X_val: Validation feature array.
        y_train: Training labels.
        y_val: Validation labels.

    Returns:
        Best validation log-loss score from early stopping.
    """
    import xgboost as xgb

    set_seed()

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.1, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 1, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-3, 10.0),
        "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=trial.suggest_int("num_boost_round", 100, 1000),
        evals=[(dval, "validation")],
        early_stopping_rounds=10,
        verbose_eval=False,
    )
    return model.best_score


def tune_gbm(data, r, split_dict, n_trials=600):
    """Tune XGBoost hyperparameters using Optuna for a given round.

    Args:
        data: Full modeling DataFrame.
        r: Tournament round number.
        split_dict: Dict mapping round number to train/val split ratio.
        n_trials: Number of Optuna trials (default 600).

    Returns:
        Best hyperparameter dict from the Optuna study.
    """
    import os

    import numpy as np
    import optuna
    from optuna.visualization import plot_optimization_history

    from models.utils.DataProcessing import create_splits

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X_SMTL, y_SMTL = create_splits(data, r, train=True)
    X, y = create_splits(data, r, train=False)

    split_idx = int(split_dict[r] * len(X))
    split_idx_SMTL = np.where((X_SMTL == X[split_idx]).all(axis=1))[0][0]
    X_train, X_val = X_SMTL[:split_idx_SMTL], X[split_idx:]
    y_train, y_val = y_SMTL[:split_idx_SMTL], y[split_idx:]

    study = optuna.create_study(
        study_name=f"xgboost_round_{r}",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=23),
    )
    study.optimize(
        lambda trial: objective(trial, X_train, X_val, y_train, y_val),
        n_trials=n_trials,
        gc_after_trial=True,
    )

    fig = plot_optimization_history(study)
    path = os.path.join(os.path.abspath(os.getcwd()), f"results/models/gbm/round_{r}.png")
    fig.write_image(path)

    return study.best_params


def tuned_gbm(params, X_train, y_train, X_val=None, y_val=None):
    """Train an XGBoost model with pre-tuned hyperparameters.

    Args:
        params: Hyperparameter dict (including num_boost_round).
        X_train: Training feature array.
        y_train: Training labels.
        X_val: Optional validation feature array for early stopping.
        y_val: Optional validation labels for early stopping.

    Returns:
        Trained XGBoost Booster model.
    """
    import xgboost as xgb

    params_sub = {key: value for key, value in params.items() if key != "num_boost_round"}
    params_sub["objective"] = "binary:logistic"
    params_sub["eval_metric"] = "logloss"

    dtrain = xgb.DMatrix(X_train, label=y_train)
    if (X_val is not None) and (y_val is not None):
        dval = xgb.DMatrix(X_val, label=y_val)
        model = xgb.train(
            params_sub,
            dtrain,
            num_boost_round=params["num_boost_round"],
            evals=[(dval, "validation")],
            early_stopping_rounds=10,
            verbose_eval=False,
        )
    else:
        model = xgb.train(
            params_sub,
            dtrain,
            num_boost_round=params["num_boost_round"],
            evals=[(dtrain, "training")],
            early_stopping_rounds=10,
            verbose_eval=False,
        )
    return model
