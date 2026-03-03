"""Walk-forward backtesting for the ensemble model across historical tournament years."""


def run_test(
    data,
    X_SMTL_nn,
    y_SMTL_nn,
    X_nn,
    X_SMTL_gbm,
    y_SMTL_gbm,
    X_gbm,
    y,
    nn_params,
    gbm_params,
    weights,
    years,
    r,
    predictions,
    years_SMTL_nn,
    years_nn,
    years_SMTL_gbm,
    years_gbm,
):
    """Run walk-forward backtesting for a single round across all backtest years.

    For each year, trains on all prior data and generates ensemble predictions
    for the test year, storing results in the predictions dict.

    Args:
        data: Full modeling DataFrame.
        X_SMTL_nn: SMOTE-resampled feature array for NN training.
        y_SMTL_nn: SMOTE-resampled labels for NN training.
        X_nn: Raw scaled feature array for NN inference.
        X_SMTL_gbm: SMOTE-resampled feature array for GBM training.
        y_SMTL_gbm: SMOTE-resampled labels for GBM training.
        X_gbm: Raw scaled feature array for GBM inference.
        y: Raw labels corresponding to X_nn/X_gbm.
        nn_params: Tuned NN hyperparameter dict for this round.
        gbm_params: Tuned GBM hyperparameter dict for this round.
        weights: Ensemble weight dict with keys 'NN' and 'GBM'.
        years: List of backtest years to iterate over.
        r: Tournament round number.
        predictions: Dict to store per-year round predictions.
        years_SMTL_nn: Year array aligned to X_SMTL_nn rows.
        years_nn: Year array aligned to X_nn rows.
        years_SMTL_gbm: Year array aligned to X_SMTL_gbm rows.
        years_gbm: Year array aligned to X_gbm rows.

    Returns:
        Updated predictions dict with Round_{r} arrays added for each test year.
    """
    import numpy as np
    import xgboost as xgb

    from models.utils.DataProcessing import create_splits
    from models.utils.gbm import tuned_gbm
    from models.utils.nn import tuned_nn

    full_years = [*range(data["Year"].min(), data["Year"].max() + 1)]
    full_years.remove(2020)
    _, _, years_scaled = create_splits(data, r, train=False, years_list=True)
    years_scaled = sorted(np.unique(np.array(years_scaled)))

    for year in years:
        test_year = 2021 if year == 2019 else year + 1
        idx = np.where(np.array(full_years) == test_year)[0][0]

        X_train_nn = X_SMTL_nn[years_SMTL_nn < years_scaled[idx]]
        X_test_nn = X_nn[years_nn == years_scaled[idx]]
        y_train_nn = y_SMTL_nn[years_SMTL_nn < years_scaled[idx]]
        X_train_gbm = X_SMTL_gbm[years_SMTL_gbm < years_scaled[idx]]
        X_test_gbm = X_gbm[years_gbm == years_scaled[idx]]
        y_train_gbm = y_SMTL_gbm[years_SMTL_gbm < years_scaled[idx]]

        nn = tuned_nn(nn_params, X_train_nn, y_train_nn)
        gbm = tuned_gbm(gbm_params, X_train_gbm, y_train_gbm)

        prob_nn = nn.predict(X_test_nn, verbose=0).flatten()
        dtest = xgb.DMatrix(X_test_gbm)
        prob_gbm = gbm.predict(dtest)

        y_pred = prob_nn * weights["NN"] + prob_gbm * weights["GBM"]
        predictions[test_year]["Round_" + str(r)] = y_pred

    return predictions
