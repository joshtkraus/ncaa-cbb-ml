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
        X_raw: Unscaled feature array for all rows (Year already excluded).
        y_raw: Label array aligned to X_raw.
        years_raw: Integer year array aligned to X_raw rows.
        nn_params: Tuned NN hyperparameter dict for this round.
        gbm_params: Tuned GBM hyperparameter dict for this round.
        weights: Ensemble weight dict with keys 'NN' and 'GBM'.
        years: List of backtest training years to iterate over.
        r: Tournament round number.
        predictions: Dict to store per-year round predictions.

    Returns:
        Updated predictions dict with Round_{r} arrays added for each test year.
    """
    import xgboost as xgb
    from sklearn.preprocessing import MinMaxScaler

    from models.utils.DataProcessing import apply_smote
    from models.utils.gbm import tuned_gbm
    from models.utils.nn import tuned_nn

    for year in years:
        test_year = 2021 if year == 2019 else year + 1

        train_mask = years_raw < test_year
        test_mask = years_raw == test_year
        # Use the 2 most recent training years as an internal val set for early
        # stopping only — they are not withheld from the final model fit.
        import numpy as np

        sorted_train_years = sorted(set(years_raw[train_mask]))
        early_stop_years = sorted_train_years[-2:]
        early_stop_mask = np.isin(years_raw, early_stop_years)

        X_train_raw = X_raw[train_mask]
        X_test_raw = X_raw[test_mask]
        X_es_raw = X_raw[early_stop_mask]
        y_train = y_raw[train_mask]
        y_es = y_raw[early_stop_mask]

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        X_es = scaler.transform(X_es_raw)

        X_train_res, y_train_res = apply_smote(X_train, y_train)

        nn = tuned_nn(nn_params, X_train_res, y_train_res, X_es, y_es)
        gbm = tuned_gbm(gbm_params, X_train_res, y_train_res, X_es, y_es)

        prob_nn = nn.predict(X_test, verbose=0).flatten()
        prob_gbm = gbm.predict(xgb.DMatrix(X_test))

        from models.utils.voting_clf import apply_temperature

        combined = prob_nn * weights["NN"] + prob_gbm * weights["GBM"]
        predictions[test_year]["Round_" + str(r)] = apply_temperature(
            combined, weights["temperature"]
        )

    return predictions
