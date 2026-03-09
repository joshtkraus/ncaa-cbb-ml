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
        "max_depth": trial.suggest_int("max_depth", 2, 7),
        "min_child_weight": trial.suggest_int("min_child_weight", 2, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 1, log=True),
        "gamma": trial.suggest_float("gamma", 0.1, 10.0),
        "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=trial.suggest_int("num_boost_round", 100, 1000),
        evals=[(dval, "validation")],
        early_stopping_rounds=30,
        verbose_eval=False,
    )
    return model.best_score


def tune_gbm(data, r, split_dict, n_trials=300, drop_cols=None):
    """Tune XGBoost hyperparameters using Optuna for a given round.

    Splits data first, fits the scaler on the training fold only, then
    applies SMOTE resampling to the training fold to prevent data leakage.

    Args:
        data: Full modeling DataFrame.
        r: Tournament round number.
        split_dict: Dict mapping round number to validation start year.
        n_trials: Number of Optuna trials (default 300).
        drop_cols: Optional list of feature column names to exclude before
            scaling. Used to restrict tuning to the selected feature subset.

    Returns:
        Best hyperparameter dict from the Optuna study.
    """
    import os

    import optuna
    from optuna.visualization import plot_optimization_history

    from models.utils.DataProcessing import apply_smote, create_splits

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X_train, X_val, y_train, y_val, _ = create_splits(
        data, r, val_start=split_dict[r], drop_cols=drop_cols
    )
    X_train, y_train = apply_smote(X_train, y_train)

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
