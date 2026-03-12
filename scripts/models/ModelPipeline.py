"""Backtesting pipeline using AutoGluon frozen model configs."""


def combine_model(data, ag_params, correct_picks, backwards_year=2013):
    """Run walk-forward backtesting with AutoGluon and export results.

    For each round, passes the frozen AutoGluon config (model type +
    exact hyperparameters) to run_test, which refits one model per
    training window without any hyperparameter search.

    Args:
        data: Full modeling DataFrame.
        ag_params: Dict keyed by round number with 'model_type' and
            'hyperparameters' keys, as produced by tune_autogluon.
        correct_picks: Dict of actual tournament results keyed by year string.
        backwards_year: First year to include in backtest (default 2013).
    """
    import os

    from models.utils.backwards_test import run_test
    from models.utils.StandarizePredictions import standardize_predict

    max_train_year = data["Year"].max() - 1
    years = [*range(backwards_year - 1, max_train_year + 1)]
    years.remove(2020)

    predictions = {}
    for year in years:
        test_year = 2021 if year == 2019 else year + 1
        predictions[test_year] = {}
        predictions[test_year]["Team"] = data.loc[data["Year"] == test_year, "Team"].values
        predictions[test_year]["Seed"] = data.loc[data["Year"] == test_year, "Seed"].values
        predictions[test_year]["Region"] = data.loc[data["Year"] == test_year, "Region"].values

    for r in range(2, 8):
        print(f"Round {r}")
        predictions = run_test(data, ag_params, years, r, predictions)

    points_df, accs_df = standardize_predict(years, predictions, correct_picks)

    path = os.path.join(os.path.abspath(os.getcwd()), "results/backwards_test/picks_accuracy.csv")
    accs_df.to_csv(path, index=False)
    path = os.path.join(os.path.abspath(os.getcwd()), "results/backwards_test/picks_points.csv")
    points_df.to_csv(path, index=False)
