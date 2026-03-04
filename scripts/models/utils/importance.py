"""Permutation importance computation for ensemble models."""

_N_REPEATS = 10


def _get_models(X_train, X_val, y_train, y_val, nn_params, gbm_params):
    """Fit NN and GBM models on training data and return them with baseline val losses.

    Args:
        X_train: Scaled, SMOTE-resampled training feature array.
        X_val: Scaled validation feature array.
        y_train: Training labels.
        y_val: Validation labels.
        nn_params: Tuned NN hyperparameter dict.
        gbm_params: Tuned GBM hyperparameter dict.

    Returns:
        Tuple of (nn, gbm, nn_baseline_loss, gbm_baseline_loss).
    """
    import xgboost as xgb
    from sklearn.metrics import log_loss

    from models.utils.gbm import tuned_gbm
    from models.utils.nn import tuned_nn

    nn = tuned_nn(nn_params, X_train, y_train, X_val, y_val)
    gbm = tuned_gbm(gbm_params, X_train, y_train, X_val, y_val)

    nn_baseline = log_loss(y_val, nn.predict(X_val, verbose=0).flatten())
    gbm_baseline = log_loss(y_val, gbm.predict(xgb.DMatrix(X_val)))

    return nn, gbm, nn_baseline, gbm_baseline


def _permutation_losses(model, X_val, y_val, predict_fn, n_repeats, rng):
    """Compute mean increase in loss for each feature under random permutation.

    For each feature, permutes its values n_repeats times and records the mean
    loss increase over the baseline (unpermuted) loss.

    Args:
        model: Fitted model with a predict interface.
        X_val: Scaled validation feature array.
        y_val: Validation labels.
        predict_fn: Callable that takes (model, X) and returns predicted probabilities.
        n_repeats: Number of permutation repeats per feature.
        rng: numpy RandomState instance for reproducibility.

    Returns:
        Array of mean loss increases, one value per feature.
    """
    import numpy as np
    from sklearn.metrics import log_loss

    baseline = log_loss(y_val, predict_fn(model, X_val))
    n_features = X_val.shape[1]
    importances = np.zeros(n_features)

    for col in range(n_features):
        col_losses = np.zeros(n_repeats)
        for i in range(n_repeats):
            X_perm = X_val.copy()
            X_perm[:, col] = rng.permutation(X_perm[:, col])
            col_losses[i] = log_loss(y_val, predict_fn(model, X_perm))
        importances[col] = np.mean(col_losses) - baseline

    return importances


def _round_importances(
    data, r, split_dict, nn_params, gbm_params, rng, drop_cols_nn=None, drop_cols_gbm=None
):
    """Compute per-model permutation importances for a single round.

    This is the core computation shared by both the feature selection loop and
    the final export step. Each model is evaluated on its own feature subset
    when drop_cols are provided.

    Args:
        data: Full modeling DataFrame.
        r: Tournament round number.
        split_dict: Dict mapping round number to train/val split ratio.
        nn_params: Tuned NN hyperparameter dict for this round.
        gbm_params: Tuned GBM hyperparameter dict for this round.
        rng: numpy RandomState instance for reproducibility.
        drop_cols_nn: Optional list of feature names to exclude for the NN.
        drop_cols_gbm: Optional list of feature names to exclude for the GBM.

    Returns:
        Tuple of (nn_imp, gbm_imp, nn_features, gbm_features,
                  nn_baseline, gbm_baseline) where nn_imp and gbm_imp are
        arrays of mean loss increases aligned to their respective feature lists.
    """
    import xgboost as xgb

    from models.utils.DataProcessing import apply_smote, create_splits

    def _nn_predict(model, X):
        return model.predict(X, verbose=0).flatten()

    def _gbm_predict(model, X):
        return model.predict(xgb.DMatrix(X))

    # NN splits
    X_raw_nn, y_raw_nn = create_splits(data, r, drop_cols=drop_cols_nn)
    split_idx_nn = int(split_dict[r] * len(X_raw_nn))
    X_train_nn, X_val_nn, y_train_nn, y_val_nn, _ = create_splits(
        data, r, split_idx=split_idx_nn, drop_cols=drop_cols_nn
    )
    X_train_nn_res, y_train_nn_res = apply_smote(X_train_nn, y_train_nn)

    # GBM splits (may differ if drop_cols differ)
    X_raw_gbm, y_raw_gbm = create_splits(data, r, drop_cols=drop_cols_gbm)
    split_idx_gbm = int(split_dict[r] * len(X_raw_gbm))
    X_train_gbm, X_val_gbm, y_train_gbm, y_val_gbm, _ = create_splits(
        data, r, split_idx=split_idx_gbm, drop_cols=drop_cols_gbm
    )
    X_train_gbm_res, y_train_gbm_res = apply_smote(X_train_gbm, y_train_gbm)

    nn_features = create_splits(data, r, get_features=True, drop_cols=drop_cols_nn)
    gbm_features = create_splits(data, r, get_features=True, drop_cols=drop_cols_gbm)

    from models.utils.gbm import tuned_gbm
    from models.utils.nn import tuned_nn

    nn = tuned_nn(nn_params, X_train_nn_res, y_train_nn_res, X_val_nn, y_val_nn)
    gbm = tuned_gbm(gbm_params, X_train_gbm_res, y_train_gbm_res, X_val_gbm, y_val_gbm)

    from sklearn.metrics import log_loss

    nn_baseline = log_loss(y_val_nn, nn.predict(X_val_nn, verbose=0).flatten())
    gbm_baseline = log_loss(y_val_gbm, gbm.predict(xgb.DMatrix(X_val_gbm)))

    nn_imp = _permutation_losses(nn, X_val_nn, y_val_nn, _nn_predict, _N_REPEATS, rng)
    gbm_imp = _permutation_losses(gbm, X_val_gbm, y_val_gbm, _gbm_predict, _N_REPEATS, rng)

    return nn_imp, gbm_imp, nn_features, gbm_features, nn_baseline, gbm_baseline


def get_importance(data, split_dict, nn_params, gbm_params, weights, features_dict=None):
    """Compute and export permutation importance for all rounds.

    For each round, fits NN and GBM on the training fold (restricted to the
    surviving feature subset when features_dict is provided), then permutes
    each feature _N_REPEATS times and records the mean increase in log-loss.
    The weighted ensemble importance normalizes each model to [0, 1] before
    blending so loss-scale differences do not skew the weights.

    Args:
        data: Full modeling DataFrame.
        split_dict: Dict mapping round number to train/val split ratio.
        nn_params: Dict of tuned NN hyperparameters keyed by round number.
        gbm_params: Dict of tuned GBM hyperparameters keyed by round number.
        weights: Dict of ensemble weights keyed by round number.
        features_dict: Optional dict keyed by round number, each value a dict
            with keys 'nn' and 'gbm' containing lists of surviving feature
            names. When None, all features are used.
    """
    import os

    import numpy as np
    import pandas as pd

    rng = np.random.RandomState(23)

    for r in range(2, 8):
        drop_nn = _dropped_cols(data, r, features_dict, "nn") if features_dict else None
        drop_gbm = _dropped_cols(data, r, features_dict, "gbm") if features_dict else None

        nn_imp, gbm_imp, nn_features, gbm_features, nn_baseline, gbm_baseline = _round_importances(
            data,
            r,
            split_dict,
            nn_params[r],
            gbm_params[r],
            rng,
            drop_cols_nn=drop_nn,
            drop_cols_gbm=drop_gbm,
        )

        w_nn, w_gbm = weights[r]["NN"], weights[r]["GBM"]
        nn_imp_norm = nn_imp / nn_imp.max()
        gbm_imp_norm = gbm_imp / gbm_imp.max()
        weighted_imp = w_nn * nn_imp_norm + w_gbm * gbm_imp_norm

        nn_df = pd.DataFrame({
            "Feature": nn_features,
            "Importance": nn_imp,
            "Importance_Normalized": nn_imp_norm,
            "Baseline_Loss": nn_baseline,
        })
        gbm_df = pd.DataFrame({
            "Feature": gbm_features,
            "Importance": gbm_imp,
            "Importance_Normalized": gbm_imp_norm,
            "Baseline_Loss": gbm_baseline,
        })

        # Weighted output uses nn_features as the index — both must share the
        # same columns at this point (called after feature selection is done)
        weighted_baseline = w_nn * nn_baseline + w_gbm * gbm_baseline
        weight_df = pd.DataFrame({
            "Feature": nn_features,
            "Importance": weighted_imp,
            "Importance_Normalized": weighted_imp / weighted_imp.max(),
            "Baseline_Loss": weighted_baseline,
        })

        nn_df.sort_values(by="Importance", ascending=False, inplace=True)
        gbm_df.sort_values(by="Importance", ascending=False, inplace=True)
        weight_df.sort_values(by="Importance", ascending=False, inplace=True)

        base = os.path.abspath(os.getcwd())
        nn_df.to_csv(os.path.join(base, f"results/perm_importance/nn/round_{r}.csv"), index=False)
        gbm_df.to_csv(os.path.join(base, f"results/perm_importance/gbm/round_{r}.csv"), index=False)
        weight_df.to_csv(
            os.path.join(base, f"results/perm_importance/weighted/round_{r}.csv"), index=False
        )


def _dropped_cols(data, r, features_dict, model_key):
    """Derive the list of columns to drop given a surviving features dict.

    Args:
        data: Full modeling DataFrame (used to get the full feature list).
        r: Tournament round number.
        features_dict: Dict keyed by round number with 'nn'/'gbm' feature lists.
        model_key: Either 'nn' or 'gbm'.

    Returns:
        List of column name strings to pass as drop_cols, or None if no
        features dict entry exists for this round.
    """
    from models.utils.DataProcessing import create_splits

    all_features = create_splits(data, r, get_features=True)
    surviving = set(features_dict.get(r, {}).get(model_key, all_features))
    return [f for f in all_features if f not in surviving] or None
