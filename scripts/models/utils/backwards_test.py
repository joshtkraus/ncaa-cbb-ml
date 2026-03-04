"""Walk-forward backtesting for the ensemble model across historical tournament years."""


def run_test(
    data,
    X_raw,
    y_raw,
    years_raw,
    nn_params,
    gbm_params,
    weights,
    years,
    r,
    predictions,
):
    """Run walk-forward backtesting for a single round across all backtest years.

    For each test year, fits the scaler exclusively on prior-year rows, applies
    SMOTE only to that training fold, trains both models, and stores predictions.

    Args:
        data: Full modeling DataFrame.
        X_raw: Unscaled feature array for all rows.
        y_raw: Label array aligned to X_raw.
        years_raw: Year column array aligned to X_raw rows.
        nn_params: Tuned NN hyperparameter dict for this round.
        gbm_params: Tuned GBM hyperparameter dict for this round.
        weights: Ensemble weight dict with keys 'NN' and 'GBM'.
        years: List of backtest training years to iterate over.
        r: Tournament round number.
        predictions: Dict to store per-year round predictions.

    Returns:
        Updated predictions dict with Round_{r} arrays added for each test year.
    """
    import numpy as np
    import xgboost as xgb
    from sklearn.preprocessing import MinMaxScaler

    from models.utils.DataProcessing import apply_smote, create_splits
    from models.utils.gbm import tuned_gbm
    from models.utils.nn import tuned_nn

    full_years = [*range(data["Year"].min(), data["Year"].max() + 1)]
    full_years.remove(2020)

    # Unique scaled year values in chronological order
    _, _, years_scaled = create_splits(data, r, years_list=True)
    years_sorted = sorted(np.unique(np.array(years_scaled)))

    for year in years:
        test_year = 2021 if year == 2019 else year + 1
        idx = np.where(np.array(full_years) == test_year)[0][0]
        cutoff = years_sorted[idx]

        # Split on raw (unscaled) arrays by year — no leakage from future rows
        train_mask = years_raw < cutoff
        test_mask = years_raw == cutoff

        X_train_raw = X_raw[train_mask]
        X_test_raw = X_raw[test_mask]
        y_train = y_raw[train_mask]

        # Fit scaler only on this training window
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        # SMOTE on training fold only, single pass shared by both models
        X_train_res, y_train_res = apply_smote(X_train, y_train)

        nn = tuned_nn(nn_params, X_train_res, y_train_res)
        gbm = tuned_gbm(gbm_params, X_train_res, y_train_res)

        prob_nn = nn.predict(X_test, verbose=0).flatten()
        dtest = xgb.DMatrix(X_test)
        prob_gbm = gbm.predict(dtest)

        y_pred = prob_nn * weights["NN"] + prob_gbm * weights["GBM"]
        predictions[test_year]["Round_" + str(r)] = y_pred

    return predictions
