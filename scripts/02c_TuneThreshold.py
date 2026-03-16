"""Per-round threshold tuning for the matchup model bracket correction."""

import copy
import json
import multiprocessing as mp
import os
import shutil

import numpy as np
import pandas as pd
from models.utils.autogluon_matchup import fit_matchup_autogluon
from models.utils.backwards_test import run_test
from models.utils.BracketCorrection import (
    _correct_e8,
    _correct_f4,
    _correct_ncg,
    _correct_r32,
    _correct_s16,
    _correct_winner,
)
from models.utils.MakePicks import predict_bracket, real_Bracket
from models.utils.StandarizePredictions import standarize

_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
_ROUNDS = [2, 3, 4, 5, 6, 7]


def _build_predictions(data, ag_params, years, test_years):
    """Pre-compute backward selection predictions for all test years.

    Args:
        data: Full modeling DataFrame.
        ag_params: Frozen model params keyed by round.
        years: List of training years.
        test_years: List of test years aligned to years.

    Returns:
        Predictions dict keyed by test year.
    """
    print("Pre-computing backward selection predictions...")
    predictions = {}
    for test_year in test_years:
        predictions[test_year] = {
            "Team": data.loc[data["Year"] == test_year, "Team"].values,
            "Seed": data.loc[data["Year"] == test_year, "Seed"].values,
            "Region": data.loc[data["Year"] == test_year, "Region"].values,
        }
    for r in range(2, 8):
        print(f"  Round {r}")
        predictions = run_test(data, ag_params, years, r, predictions)
    return predictions


def _build_bracket_lookups(data, predictions, test_years, correct_picks, tuning_dir):
    """Build backward bracket picks, year data lookups, and baseline scores.

    Args:
        data: Full modeling DataFrame.
        predictions: Backward selection predictions dict.
        test_years: List of test years.
        correct_picks: Actual results dict keyed by year string.
        tuning_dir: Directory to save baseline CSV.

    Returns:
        Tuple of (backward_picks, year_data_lookup, baseline_mean, baseline_sd).
    """
    backward_picks = {}
    year_data_lookup = {}

    for test_year in test_years:
        pred_df = pd.DataFrame.from_dict(predictions[test_year])
        pred_df = standarize(pred_df)

        points_df = pred_df.copy()
        points_df["R32"] = pred_df["R32"] * 10
        points_df["S16"] = pred_df["R32"] * 10 + pred_df["S16"] * 20
        points_df["E8"] = pred_df["R32"] * 10 + pred_df["S16"] * 20 + pred_df["E8"] * 40
        points_df["F4"] = (
            pred_df["R32"] * 10 + pred_df["S16"] * 20 + pred_df["E8"] * 40 + pred_df["F4"] * 80
        )
        points_df["NCG"] = (
            pred_df["R32"] * 10
            + pred_df["S16"] * 20
            + pred_df["E8"] * 40
            + pred_df["F4"] * 80
            + pred_df["NCG"] * 160
        )
        points_df["Winner"] = (
            pred_df["R32"] * 10
            + pred_df["S16"] * 20
            + pred_df["E8"] * 40
            + pred_df["F4"] * 80
            + pred_df["NCG"] * 160
            + pred_df["Winner"] * 320
        )
        # backward_picks[test_year] = predict_bracket(points_df, calc_correct=False)
        backward_picks[test_year] = predict_bracket(pred_df, calc_correct=False, sim=True)
        year_data_lookup[test_year] = {
            "team_data": data[data["Year"] == test_year][["Team", "Seed", "Region"]],
            "full_data": data[data["Year"] == test_year],
        }

    baseline_points = {
        test_year: real_Bracket(backward_picks[test_year], correct_picks[str(test_year)])[0]
        for test_year in test_years
    }
    baseline_mean = float(np.mean(list(baseline_points.values())))
    baseline_sd = float(np.std(list(baseline_points.values())))
    print(f"\nBaseline (no correction): mean={baseline_mean:.0f}, SD={baseline_sd:.0f}")

    os.makedirs(tuning_dir, exist_ok=True)
    baseline_row = {str(y): p for y, p in baseline_points.items()}
    baseline_row["Mean"] = baseline_mean
    baseline_row["SD"] = baseline_sd
    pd.DataFrame([baseline_row]).to_csv(os.path.join(tuning_dir, "baseline.csv"), index=False)

    return backward_picks, year_data_lookup, baseline_mean, baseline_sd


def _build_matchup_predictors(matchup_data, matchup_params, test_years, matchup_base_dir):
    """Fit per-year matchup predictors using walk-forward discipline.

    Args:
        matchup_data: Full matchup DataFrame.
        matchup_params: Frozen matchup model params.
        test_years: List of test years.
        matchup_base_dir: Base directory to save predictors.

    Returns:
        Dict of fitted predictors keyed by test year.
    """
    print("\nFitting per-year matchup predictors...")
    if os.path.exists(matchup_base_dir):
        shutil.rmtree(matchup_base_dir)

    matchup_predictors = {}
    for test_year in test_years:
        train_mask = matchup_data["Year"].to_numpy() < test_year
        save_path = os.path.join(matchup_base_dir, str(test_year))
        os.makedirs(save_path, exist_ok=True)
        matchup_predictors[test_year] = fit_matchup_autogluon(
            matchup_data, train_mask, matchup_params, save_path=save_path
        )
    return matchup_predictors


def _score_thresholds(
    round_thresholds,
    test_years,
    backward_picks,
    year_data_lookup,
    matchup_predictors,
    correct_picks,
):
    """Apply per-round corrections and return mean points across test years.

    Args:
        round_thresholds: Dict mapping round number to threshold.
        test_years: List of test years.
        backward_picks: Initial backward selection bracket per year.
        year_data_lookup: Team data lookup per year.
        matchup_predictors: Fitted matchup predictor per year.
        correct_picks: Actual results dict keyed by year string.

    Returns:
        Tuple of (mean_points, year_points dict).
    """
    year_points = {}
    for test_year in test_years:
        picks = backward_picks[test_year]
        td = year_data_lookup[test_year]["team_data"]
        fd = year_data_lookup[test_year]["full_data"]
        pred = matchup_predictors[test_year]

        corrected = copy.deepcopy(picks)
        _correct_r32(corrected, td, fd, pred, round_thresholds[2])
        _correct_s16(corrected, fd, pred, round_thresholds[3])
        _correct_e8(corrected, fd, pred, round_thresholds[4])
        _correct_f4(corrected, fd, pred, round_thresholds[5])
        _correct_ncg(corrected, fd, pred, round_thresholds[6])
        _correct_winner(corrected, fd, pred, round_thresholds[7])

        pt, _ = real_Bracket(corrected, correct_picks[str(test_year)])
        year_points[test_year] = pt
    return float(np.mean(list(year_points.values()))), year_points


def _tune_round(r, best_thresholds, score_fn):
    """Tune a single round's threshold holding all others fixed.

    Args:
        r: Round number to tune.
        best_thresholds: Current best threshold dict.
        score_fn: Callable that takes a threshold dict and returns mean points.

    Returns:
        Tuple of (best_threshold, best_mean).
    """
    best_t = best_thresholds[r]
    best_mean, _ = score_fn(best_thresholds)
    for t in _THRESHOLDS:
        candidate = {**best_thresholds, r: t}
        mean_pts, _ = score_fn(candidate)
        if mean_pts > best_mean:
            best_mean = mean_pts
            best_t = t
    return best_t, best_mean


def run():
    """Tune per-round thresholds and save the optimal values."""
    cwd = os.path.abspath(os.getcwd())

    data = pd.read_csv(os.path.join(cwd, "data/processed/data.csv"))

    with open(os.path.join(cwd, "data/processed/results.json")) as f:
        correct_picks = json.load(f)

    with open(os.path.join(cwd, "model/autogluon_params.json")) as f:
        ag_params = json.load(f)
    ag_params = {int(k): v for k, v in ag_params.items()}

    with open(os.path.join(cwd, "model/autogluon_matchup_params.json")) as f:
        matchup_params = json.load(f)

    matchup_data = pd.read_csv(os.path.join(cwd, "data/processed/data_matchup.csv"))

    backwards_year = 2013
    max_train_year = data["Year"].max() - 1
    years = [*range(backwards_year - 1, max_train_year + 1)]
    years.remove(2020)
    test_years = [2021 if y == 2019 else y + 1 for y in years]
    tuning_dir = os.path.join(cwd, "results/threshold_tuning")
    matchup_base_dir = os.path.join(cwd, "model/autogluon_matchup_threshold")

    predictions = _build_predictions(data, ag_params, years, test_years)
    backward_picks, year_data_lookup, baseline_mean, baseline_sd = _build_bracket_lookups(
        data, predictions, test_years, correct_picks, tuning_dir
    )
    matchup_predictors = _build_matchup_predictors(
        matchup_data, matchup_params, test_years, matchup_base_dir
    )

    def score_fn(round_thresholds):
        return _score_thresholds(
            round_thresholds,
            test_years,
            backward_picks,
            year_data_lookup,
            matchup_predictors,
            correct_picks,
        )

    # Greedy forward pass
    print("\nForward pass (R2 → R7)...")
    best_thresholds = dict.fromkeys(_ROUNDS, 1.0)
    for r in _ROUNDS:
        best_t, best_mean = _tune_round(r, best_thresholds, score_fn)
        best_thresholds[r] = best_t
        print(f"  Round {r}: best threshold={best_t:.2f}  mean={best_mean:.0f}")

    # Single backward pass
    print("\nBackward pass (R6 → R2)...")
    for r in reversed(_ROUNDS[:-1]):
        prev_t = best_thresholds[r]
        best_t, best_mean = _tune_round(r, best_thresholds, score_fn)
        best_thresholds[r] = best_t
        if best_t != prev_t:
            print(f"  Round {r}: updated {prev_t:.2f} → {best_t:.2f}  mean={best_mean:.0f}")
        else:
            print(f"  Round {r}: unchanged ({best_t:.2f})  mean={best_mean:.0f}")

    # Final scoring and save
    final_mean, final_year_points = score_fn(best_thresholds)
    final_sd = float(np.std(list(final_year_points.values())))
    delta = final_mean - baseline_mean

    print("\nOptimal per-round thresholds:")
    for r in _ROUNDS:
        print(f"  Round {r}: {best_thresholds[r]:.2f}")
    print(f"\nFinal mean: {final_mean:.0f}  (baseline: {baseline_mean:.0f})")
    print(f"Final SD:   {final_sd:.0f}  (baseline: {baseline_sd:.0f})")
    print(f"Delta:      {delta:+.0f}")

    results_row = {str(y): p for y, p in final_year_points.items()}
    results_row["Mean"] = final_mean
    results_row["SD"] = final_sd
    pd.DataFrame([results_row]).to_csv(
        os.path.join(tuning_dir, "per_round_thresholds.csv"), index=False
    )

    threshold_path = os.path.join(cwd, "model/matchup_threshold.json")
    with open(threshold_path, "w") as f:
        json.dump(
            {
                "thresholds": best_thresholds,
                "mean": final_mean,
                "sd": final_sd,
                "delta_vs_baseline": delta,
                "by_year": final_year_points,
            },
            f,
            indent=2,
        )

    if os.path.exists(matchup_base_dir):
        shutil.rmtree(matchup_base_dir)


if __name__ == "__main__":
    mp.freeze_support()
    run()
