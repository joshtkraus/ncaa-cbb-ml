"""Threshold tuning for the matchup model bracket correction."""

import multiprocessing as mp


def run():
    """Tune the matchup correction threshold and save the optimal value."""
    import json
    import os
    import shutil

    import numpy as np
    import pandas as pd
    from models.utils.autogluon_matchup import fit_matchup_autogluon
    from models.utils.backwards_test import run_test
    from models.utils.BracketCorrection import correct_bracket
    from models.utils.MakePicks import predict_bracket, real_Bracket
    from models.utils.StandarizePredictions import standarize

    cwd = os.path.abspath(os.getcwd())

    # -----------------------------------------------------------------------
    # Candidate thresholds to evaluate
    # -----------------------------------------------------------------------
    thresholds = [0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60]

    # -----------------------------------------------------------------------
    # Load data and params
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # Step 1: Pre-compute backward selection predictions
    # -----------------------------------------------------------------------
    print("Pre-computing backward selection predictions...")
    predictions = {}
    for year in years:
        test_year = 2021 if year == 2019 else year + 1
        predictions[test_year] = {}
        predictions[test_year]["Team"] = data.loc[data["Year"] == test_year, "Team"].values
        predictions[test_year]["Seed"] = data.loc[data["Year"] == test_year, "Seed"].values
        predictions[test_year]["Region"] = data.loc[data["Year"] == test_year, "Region"].values

    for r in range(2, 8):
        print(f"  Round {r}")
        predictions = run_test(data, ag_params, years, r, predictions)

    # -----------------------------------------------------------------------
    # Step 2: Pre-compute per-year points DataFrames and initial picks
    # -----------------------------------------------------------------------
    print("\nPre-computing backward selection brackets...")
    year_points_dfs = {}
    backward_picks = {}

    for year in years:
        test_year = 2021 if year == 2019 else year + 1
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
        year_points_dfs[test_year] = points_df
        backward_picks[test_year] = predict_bracket(points_df, calc_correct=False)

    # Score baseline (no correction) for reference
    baseline_points = {}
    for year in years:
        test_year = 2021 if year == 2019 else year + 1
        point, _ = real_Bracket(backward_picks[test_year], correct_picks[str(test_year)])
        baseline_points[test_year] = point

    baseline_mean = float(np.mean(list(baseline_points.values())))
    baseline_sd = float(np.std(list(baseline_points.values())))
    print(f"\nBaseline (no correction): mean={baseline_mean:.0f}, SD={baseline_sd:.0f}")

    # Save baseline CSV
    tuning_dir = os.path.join(cwd, "results/threshold_tuning")
    os.makedirs(tuning_dir, exist_ok=True)
    baseline_row = {str(y): p for y, p in baseline_points.items()}
    baseline_row["Mean"] = baseline_mean
    baseline_row["SD"] = baseline_sd
    pd.DataFrame([baseline_row]).to_csv(os.path.join(tuning_dir, "baseline.csv"), index=False)

    # -----------------------------------------------------------------------
    # Step 3: Pre-compute per-year matchup predictors
    # -----------------------------------------------------------------------
    print("\nFitting per-year matchup predictors...")
    matchup_base_dir = os.path.join(cwd, "model/autogluon_matchup_threshold")
    if os.path.exists(matchup_base_dir):
        shutil.rmtree(matchup_base_dir)

    matchup_predictors = {}
    for year in years:
        test_year = 2021 if year == 2019 else year + 1
        train_mask = matchup_data["Year"].values < test_year
        save_path = os.path.join(matchup_base_dir, str(test_year))
        os.makedirs(save_path, exist_ok=True)
        predictor = fit_matchup_autogluon(
            matchup_data, train_mask, matchup_params, save_path=save_path
        )
        matchup_predictors[test_year] = predictor

    # -----------------------------------------------------------------------
    # Step 4: Test thresholds
    # -----------------------------------------------------------------------
    print("\nTesting thresholds...")
    results = {}

    for threshold in thresholds:
        year_points = {}
        for year in years:
            test_year = 2021 if year == 2019 else year + 1
            year_data = data[data["Year"] == test_year][["Team", "Seed", "Region"]]
            full_year_data = data[data["Year"] == test_year]
            predictor = matchup_predictors[test_year]

            corrected = correct_bracket(
                backward_picks[test_year],
                year_data,
                full_year_data,
                predictor,
                threshold=threshold,
            )
            point, _ = real_Bracket(corrected, correct_picks[str(test_year)])
            year_points[test_year] = point

        mean_pts = float(np.mean(list(year_points.values())))
        sd_pts = float(np.std(list(year_points.values())))
        delta = mean_pts - baseline_mean
        results[threshold] = {
            "mean": mean_pts,
            "sd": sd_pts,
            "delta_vs_baseline": delta,
            "by_year": year_points,
        }

        # Save per-threshold points CSV
        threshold_label = f"{threshold:.2f}".replace(".", "_")
        points_row = {str(y): p for y, p in year_points.items()}
        points_row["Mean"] = mean_pts
        points_row["SD"] = sd_pts
        threshold_df = pd.DataFrame([points_row])
        tuning_dir = os.path.join(cwd, "results/threshold_tuning")
        os.makedirs(tuning_dir, exist_ok=True)
        threshold_df.to_csv(
            os.path.join(tuning_dir, f"threshold_{threshold_label}.csv"), index=False
        )

    # -----------------------------------------------------------------------
    # Step 5: Select optimal threshold and report
    # -----------------------------------------------------------------------
    optimal_threshold = max(results, key=lambda t: results[t]["mean"])

    print(f"Optimal threshold: {optimal_threshold:.2f}")

    # Save optimal threshold
    threshold_path = os.path.join(cwd, "model/matchup_threshold.json")
    with open(threshold_path, "w") as f:
        json.dump({"threshold": optimal_threshold, "results": results}, f, indent=2)

    # Clean up per-year matchup models
    if os.path.exists(matchup_base_dir):
        shutil.rmtree(matchup_base_dir)


if __name__ == "__main__":
    mp.freeze_support()
    run()
