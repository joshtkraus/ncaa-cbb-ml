"""Backtesting pipeline using AutoGluon frozen model configs."""

import os
import shutil

from models.utils.autogluon_matchup import fit_matchup_autogluon
from models.utils.backwards_test import run_test
from models.utils.StandarizePredictions import standardize_predict


def combine_model(
    data,
    ag_params,
    correct_picks,
    backwards_year=2015,
    matchup_params=None,
    matchup_data=None,
):
    """Run walk-forward backtesting with AutoGluon and export results.

    Args:
        data: Full modeling DataFrame.
        ag_params: Dict keyed by round number with 'model_type' and 'hyperparameters' keys.
        correct_picks: Dict of actual tournament results keyed by year string.
        backwards_year: First year to include in backtest (default 2015).
        matchup_params: Optional dict with 'model_type' and 'hyperparameters'.
        matchup_data: Optional full matchup DataFrame from build_matchup_dataset.
    """
    use_matchup = matchup_params is not None and matchup_data is not None

    cwd = os.path.abspath(os.getcwd())
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

    print("\nFitting by-round model...")
    for r in range(2, 8):
        print(f"Round {r}")
        predictions = run_test(data, ag_params, years, r, predictions)

    # Build a per-year matchup predictor lookup
    matchup_predictors = {}
    matchup_base_dir = os.path.join(cwd, "model/autogluon_matchup_backtest")

    if use_matchup:
        # Clean up any previous backtest matchup models
        if os.path.exists(matchup_base_dir):
            shutil.rmtree(matchup_base_dir)

        print("\nFitting matchup model...")
        for year in years:
            test_year = 2021 if year == 2019 else year + 1
            train_mask = matchup_data["Year"].values < test_year
            save_path = os.path.join(matchup_base_dir, str(test_year))
            os.makedirs(save_path, exist_ok=True)
            predictor = fit_matchup_autogluon(
                matchup_data, train_mask, matchup_params, save_path=save_path
            )
            matchup_predictors[test_year] = predictor

    points_df, accs_df = standardize_predict(
        years,
        predictions,
        correct_picks,
        data=data if use_matchup else None,
        matchup_predictor=matchup_predictors if use_matchup else None,
    )

    # Clean up backtest matchup models after scoring is complete
    if use_matchup and os.path.exists(matchup_base_dir):
        shutil.rmtree(matchup_base_dir)

    path = os.path.join(cwd, "results/backwards_test/picks_accuracy.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    accs_df.to_csv(path, index=False)
    path = os.path.join(cwd, "results/backwards_test/picks_points.csv")
    points_df.to_csv(path, index=False)
