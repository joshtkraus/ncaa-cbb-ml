"""Backtesting pipeline that combines NN and GBM predictions across historical years."""


def combine_model(data, nn_params, gbm_params, weights, correct_picks, backwards_year=2013):
    """Run walk-forward backtesting and export pick accuracy and points results.

    Args:
        data: Full modeling DataFrame.
        nn_params: Dict of tuned NN hyperparameters keyed by round number.
        gbm_params: Dict of tuned GBM hyperparameters keyed by round number.
        weights: Dict of ensemble weights keyed by round number.
        correct_picks: Dict of actual tournament results keyed by year string.
        backwards_year: First year to include in backtest (default 2013).
    """
    import os

    from models.utils.backwards_test import run_test
    from models.utils.DataProcessing import create_splits
    from models.utils.StandarizePredictions import standardize_predict

    print("Combining Models...")

    years = [*range(backwards_year - 1, 2024)]
    years.remove(2020)

    predictions = {}
    for year in years:
        test_year = 2021 if year == 2019 else year + 1
        predictions[test_year] = {}
        predictions[test_year]["Team"] = data.loc[data["Year"] == test_year, "Team"].values
        predictions[test_year]["Seed"] = data.loc[data["Year"] == test_year, "Seed"].values
        predictions[test_year]["Region"] = data.loc[data["Year"] == test_year, "Region"].values

    for r in range(2, 8):
        X_SMTL_nn, y_SMTL_nn, years_SMTL_nn = create_splits(data, r, train=True, years_list=True)
        X_nn, y, years_nn = create_splits(data, r, train=False, years_list=True)
        X_SMTL_gbm, y_SMTL_gbm, years_SMTL_gbm = create_splits(data, r, train=True, years_list=True)
        X_gbm, _, years_gbm = create_splits(data, r, train=False, years_list=True)

        predictions = run_test(
            data,
            X_SMTL_nn,
            y_SMTL_nn,
            X_nn,
            X_SMTL_gbm,
            y_SMTL_gbm,
            X_gbm,
            y,
            nn_params[r],
            gbm_params[r],
            weights[r],
            years,
            r,
            predictions,
            years_SMTL_nn,
            years_nn,
            years_SMTL_gbm,
            years_gbm,
        )

    points_df, accs_df = standardize_predict(years, predictions, correct_picks)

    path = os.path.join(os.path.abspath(os.getcwd()), "results/backwards_test/picks_accuracy.csv")
    accs_df.to_csv(path, index=False)
    path = os.path.join(os.path.abspath(os.getcwd()), "results/backwards_test/picks_points.csv")
    points_df.to_csv(path, index=False)
