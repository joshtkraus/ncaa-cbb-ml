"""Voting classifier weight computation using closed-form analytic solution."""


def tune_weights(data, nn_params, gbm_params):
    """Compute ensemble blend weights for all rounds using closed-form CV.

    For each round, trains both models on each walk-forward CV fold and
    collects out-of-fold val predictions. The concatenated predictions across
    all folds are passed to the closed-form analytic solution, which directly
    minimises Brier score on the full val set in one shot — deterministic,
    stable, and optimal by construction with no search overhead.

    Class weights are computed per fold from the post-SMOTE training labels
    and passed to both models so that training is consistent with how each
    model was tuned.

    Args:
        data: Full modeling DataFrame.
        nn_params: Dict of tuned NN hyperparameters keyed by round number.
        gbm_params: Dict of tuned GBM hyperparameters keyed by round number.

    Returns:
        Dict keyed by round number, each with 'NN' and 'GBM' keys.
    """
    import numpy as np
    import xgboost as xgb

    from models.utils.cv_folds import make_folds
    from models.utils.DataProcessing import apply_smote, create_fold_splits, get_class_weights
    from models.utils.gbm import tuned_gbm
    from models.utils.nn import tuned_nn

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

            # Compute per-sample class weights from post-SMOTE labels
            sample_weights = get_class_weights(y_train_res)
            unique_classes = np.unique(y_train_res)
            class_weight_dict = {
                int(c): float(sample_weights[y_train_res == c][0]) for c in unique_classes
            }

            nn = tuned_nn(
                nn_params[r],
                X_train_res,
                y_train_res,
                X_val,
                y_val,
                class_weight=class_weight_dict,
            )
            prob_nn = nn.predict(X_val, verbose=0).flatten()

            gbm = tuned_gbm(
                gbm_params[r],
                X_train_res,
                y_train_res,
                X_val,
                y_val,
                train_weights=sample_weights,
            )
            prob_gbm = gbm.predict(xgb.DMatrix(X_val))

            all_prob_nn.append(prob_nn)
            all_prob_gbm.append(prob_gbm)
            all_y_val.append(y_val)

        prob_nn_all = np.concatenate(all_prob_nn)
        prob_gbm_all = np.concatenate(all_prob_gbm)
        y_val_all = np.concatenate(all_y_val)

        brier_nn = float(np.mean((y_val_all - prob_nn_all) ** 2))
        brier_gbm = float(np.mean((y_val_all - prob_gbm_all) ** 2))
        print(f"  Round {r} Brier scores — NN: {brier_nn:.4f}, GBM: {brier_gbm:.4f}")

        w_nn = (1 / brier_nn) / (1 / brier_nn + 1 / brier_gbm)
        w_gbm = (1 / brier_gbm) / (1 / brier_nn + 1 / brier_gbm)
        print(f"  Round {r} weights   — NN: {w_nn:.3f}, GBM: {w_gbm:.3f}")

        weights[r] = {
            "NN": w_nn,
            "GBM": w_gbm,
        }

    return weights
