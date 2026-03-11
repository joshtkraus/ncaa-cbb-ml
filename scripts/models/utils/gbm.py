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


def objective(trial, data, r, folds, drop_cols=None):
    """Optuna objective function for XGBoost hyperparameter tuning using CV.

    Trains and evaluates the model on each walk-forward fold, returning the
    mean validation log-loss across all folds for stable hyperparameter selection.

    Args:
        trial: Optuna trial object.
        data: Full modeling DataFrame.
        r: Tournament round number (used for feature construction).
        folds: List of fold dicts from make_folds().
        drop_cols: Optional list of feature names to exclude.

    Returns:
        Mean validation log-loss across all CV folds.
    """
    import xgboost as xgb

    from models.utils.DataProcessing import apply_smote, create_fold_splits, get_class_weights

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

    fold_losses = []
    for fold_idx, fold in enumerate(folds):
        X_train, X_val, y_train, y_val = create_fold_splits(data, r, fold, drop_cols=drop_cols)
        X_train, y_train = apply_smote(X_train, y_train)
        sample_weights = get_class_weights(y_train)

        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
        dval = xgb.DMatrix(X_val, label=y_val)
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=trial.suggest_int("num_boost_round", 100, 1000),
            evals=[(dval, "validation")],
            early_stopping_rounds=30,
            verbose_eval=False,
        )
        fold_losses.append(model.best_score)
        print(
            f"      Fold {fold_idx + 1}/{len(folds)}: val_loss={model.best_score:.4f} "
            f"(rounds={model.best_iteration + 1})"
        )

    mean_loss = sum(fold_losses) / len(fold_losses)
    print(f"      Mean loss: {mean_loss:.4f}")
    return mean_loss


def tune_gbm(data, r, folds, n_trials=200, drop_cols=None):
    """Tune XGBoost hyperparameters using Optuna with walk-forward CV.

    Evaluates each trial across all CV folds and returns the hyperparameters
    that minimise mean validation log-loss, giving stable estimates that are
    not dependent on any single val window.

    Args:
        data: Full modeling DataFrame.
        r: Tournament round number used for feature construction.
        folds: List of fold dicts from make_folds().
        n_trials: Number of Optuna trials (default 200).
        drop_cols: Optional list of feature column names to exclude before
            scaling.

    Returns:
        Best hyperparameter dict from the Optuna study.
    """
    import os

    import optuna
    from optuna.visualization import plot_optimization_history

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        study_name=f"xgboost_round_{r}",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=23),
    )

    def _trial_callback(study, trial):
        print(
            f"    Trial {trial.number + 1}/{n_trials}: loss={trial.value:.4f} "
            f"(best={study.best_value:.4f})"
        )

    study.optimize(
        lambda trial: objective(trial, data, r, folds, drop_cols=drop_cols),
        n_trials=n_trials,
        gc_after_trial=True,
        callbacks=[_trial_callback],
    )

    fig = plot_optimization_history(study)
    path = os.path.join(os.path.abspath(os.getcwd()), f"results/models/gbm/round_{r}.png")
    fig.write_image(path)

    return study.best_params


def tuned_gbm(params, X_train, y_train, X_val=None, y_val=None, train_weights=None):
    """Train an XGBoost model with pre-tuned hyperparameters.

    Args:
        params: Hyperparameter dict (including num_boost_round).
        X_train: Training feature array.
        y_train: Training labels.
        X_val: Optional validation feature array for early stopping.
        y_val: Optional validation labels for early stopping.
        train_weights: Optional per-sample weight array for training rows.
            Used to pass class weights so the loss is scaled consistently
            with how the model was tuned.

    Returns:
        Trained XGBoost Booster model.
    """
    import xgboost as xgb

    params_sub = {key: value for key, value in params.items() if key != "num_boost_round"}
    params_sub["objective"] = "binary:logistic"
    params_sub["eval_metric"] = "logloss"

    early_stopping_rounds = None if params_sub.get("booster") == "dart" else 30

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=train_weights)
    if (X_val is not None) and (y_val is not None):
        dval = xgb.DMatrix(X_val, label=y_val)
        model = xgb.train(
            params_sub,
            dtrain,
            num_boost_round=params["num_boost_round"],
            evals=[(dval, "validation")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )
    else:
        model = xgb.train(
            params_sub,
            dtrain,
            num_boost_round=params["num_boost_round"],
            evals=[(dtrain, "training")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )
    return model
