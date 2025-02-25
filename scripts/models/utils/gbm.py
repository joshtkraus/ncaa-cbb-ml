def set_seed(seed=23):
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)

def objective(trial, X_train, X_val, y_train, y_val):
    import xgboost as xgb
    set_seed()
    
    # Params
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 0.3),
        "max_depth": trial.suggest_int("max_depth", 1, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.1, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-5, 1),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-3, 10.0),
        "gamma": trial.suggest_float("gamma", 1e-3, 10.0),
        "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    }
    
    # Train the model
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

def tune_gbm(data, r, split_dict, best_features=None, n_trials=600):
    import os
    import numpy as np
    from models.utils.DataProcessing import create_splits
    import optuna
    from optuna.visualization import plot_optimization_history

    # Supress Logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Create Data
    X_SMTL, y_SMTL = create_splits(data, r, train=True, best_features=best_features)
    X, y = create_splits(data, r, train=False, best_features=best_features)

    # Data Splits
    split_idx = int(split_dict[r] * len(X))
    split_idx_SMTL = np.where((X_SMTL == X[split_idx]).all(axis=1))[0][0]
    X_train, X_val = X_SMTL[:split_idx_SMTL], X[split_idx:]
    y_train, y_val = y_SMTL[:split_idx_SMTL], y[split_idx:]

    # Tuning
    study = optuna.create_study(
        study_name=f"xgboost_round_{r}",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=23)
    )
    study.optimize(lambda trial: objective(trial, X_train, X_val, y_train, y_val), 
                   n_trials=n_trials,
                   gc_after_trial=True)
    
    # Save Plot
    fig = plot_optimization_history(study)
    path = os.path.join(os.path.abspath(os.getcwd()), f"results/models/gbm/round_{r}.png")
    fig.write_image(path)
    
    return study.best_params

def tuned_gbm(params, X_train, y_train, X_val=None, y_val=None):
    import xgboost as xgb
    # Subset Params
    params_sub = {key: value for key, value in params.items() if key not in ['num_boost_round']}

    # Train the model
    dtrain = xgb.DMatrix(X_train, label=y_train)
    if (X_val is not None) and (y_val is not None):
        dval = xgb.DMatrix(X_val, label=y_val)
        model = xgb.train(
            params_sub, 
            dtrain, 
            num_boost_round=params['num_boost_round'],
            evals=[(dval, "validation")],
            early_stopping_rounds=10,
            verbose_eval=False,
        )
    else:
        model = xgb.train(
            params_sub, 
            dtrain, 
            num_boost_round=params['num_boost_round'],
            evals=[(dtrain, "training")],
            early_stopping_rounds=10,
            verbose_eval=False,
        )
    return model