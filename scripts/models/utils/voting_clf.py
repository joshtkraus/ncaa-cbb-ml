"""Voting classifier weight tuning using walk-forward CV."""


def objective(trial, prob_nn, prob_gbm, y_val):
    """Optuna objective minimizing Brier Score over ensemble weight.

    Args:
        trial: Optuna trial object.
        prob_nn: NN predicted probabilities on the validation set.
        prob_gbm: GBM predicted probabilities on the validation set.
        y_val: True validation labels.

    Returns:
        Brier score of the weighted ensemble.
    """
    from sklearn.metrics import brier_score_loss

    w = trial.suggest_float("weight", 0.3, 0.7)
    combined_probs = w * prob_nn + (1 - w) * prob_gbm
    return brier_score_loss(y_val, combined_probs)


def tune_weights(data, nn_params, gbm_params, n_trials=100):
    """Tune ensemble blend weights for all rounds using CV.

    For each round and each CV fold, trains both models on the fold's
    training data and evaluates on the fold's val data. The concatenated
    val predictions across all folds are used to tune the blend weight
    giving robust estimates not dependent on any single
    val window.

    Args:
        data: Full modeling DataFrame.
        nn_params: Dict of tuned NN hyperparameters keyed by round number.
        gbm_params: Dict of tuned GBM hyperparameters keyed by round number.
        n_trials: Number of Optuna trials per round (default 100).

    Returns:
        Dict keyed by round number, each with 'NN', 'GBM'
        keys.
    """
    import numpy as np
    import optuna
    import xgboost as xgb

    from models.utils.cv_folds import make_folds
    from models.utils.DataProcessing import apply_smote, create_fold_splits
    from models.utils.gbm import tuned_gbm
    from models.utils.nn import tuned_nn

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    folds = make_folds(data, n_folds=2)
    weights = {}

    for r in range(2, 8):
        print(f"Round {r}")

        all_prob_nn = []
        all_prob_gbm = []
        all_y_val = []

        for fold_idx, fold in enumerate(folds):
            print(
                f"  Fold {fold_idx + 1}/{len(folds)}: "
                f"train={fold['train_years']}, val={fold['val_years']}"
            )
            X_train, X_val, y_train, y_val = create_fold_splits(data, r, fold)
            X_train_res, y_train_res = apply_smote(X_train, y_train)

            nn = tuned_nn(nn_params[r], X_train_res, y_train_res, X_val, y_val)
            prob_nn = nn.predict(X_val, verbose=0).flatten()

            gbm = tuned_gbm(gbm_params[r], X_train_res, y_train_res, X_val, y_val)
            prob_gbm = gbm.predict(xgb.DMatrix(X_val))

            all_prob_nn.append(prob_nn)
            all_prob_gbm.append(prob_gbm)
            all_y_val.append(y_val)

        prob_nn_all = np.concatenate(all_prob_nn)
        prob_gbm_all = np.concatenate(all_prob_gbm)
        y_val_all = np.concatenate(all_y_val)

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=23 * r),
        )
        study.optimize(
            lambda trial, pn=prob_nn_all, pg=prob_gbm_all, yv=y_val_all: objective(
                trial, pn, pg, yv
            ),
            n_trials=n_trials,
        )

        best = study.best_params
        print(f"  Round {r}: NN={best['weight']:.3f}, GBM={1 - best['weight']:.3f}")

        weights[r] = {
            "NN": best["weight"],
            "GBM": 1 - best["weight"],
        }

    return weights
