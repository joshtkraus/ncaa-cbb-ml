"""Utilities for normalizing model predictions and computing expected bracket points."""


def standarize(df):
    """Normalize raw model probabilities to sum to 1 within each matchup group.

    For each round, teams are grouped by their bracket pod and probabilities
    are rescaled so the group sums to 1, enforcing the constraint that exactly
    one team from each matchup advances.

    Args:
        df: DataFrame with columns Team, Seed, Region, and raw round probabilities
            (Round_2 through Round_7).

    Returns:
        DataFrame with normalized columns R32, S16, E8, F4, NCG, Winner.
    """
    import pandas as pd  # noqa: F401

    df.columns = ["Team", "Seed", "Region", "R32", "S16", "E8", "F4", "NCG", "Winner"]

    df["R32_Group"] = 0
    df.loc[df["Seed"].isin([1, 16]), "R32_Group"] = 1
    df.loc[df["Seed"].isin([2, 15]), "R32_Group"] = 2
    df.loc[df["Seed"].isin([3, 14]), "R32_Group"] = 3
    df.loc[df["Seed"].isin([4, 13]), "R32_Group"] = 4
    df.loc[df["Seed"].isin([5, 12]), "R32_Group"] = 5
    df.loc[df["Seed"].isin([6, 11]), "R32_Group"] = 6
    df.loc[df["Seed"].isin([7, 10]), "R32_Group"] = 7
    df.loc[df["Seed"].isin([8, 9]), "R32_Group"] = 8
    R32_Totals = df.groupby(["Region", "R32_Group"]).agg(Round_2_Sum=("R32", "sum")).reset_index()

    df["S16_Group"] = 0
    df.loc[df["Seed"].isin([1, 16, 8, 9]), "S16_Group"] = 1
    df.loc[df["Seed"].isin([2, 15, 7, 10]), "S16_Group"] = 2
    df.loc[df["Seed"].isin([3, 14, 6, 11]), "S16_Group"] = 3
    df.loc[df["Seed"].isin([4, 13, 5, 12]), "S16_Group"] = 4
    S16_Totals = df.groupby(["Region", "S16_Group"]).agg(Round_3_Sum=("S16", "sum")).reset_index()

    df["E8_Group"] = 0
    df.loc[df["Seed"].isin([1, 16, 8, 9, 4, 13, 5, 12]), "E8_Group"] = 1
    df.loc[df["Seed"].isin([2, 15, 7, 10, 3, 14, 6, 11]), "E8_Group"] = 2
    E8_Totals = df.groupby(["Region", "E8_Group"]).agg(Round_4_Sum=("E8", "sum")).reset_index()

    F4_Totals = df.groupby(["Region"]).agg(Round_5_Sum=("F4", "sum")).reset_index()

    df["NCG_Group"] = "0"
    df.loc[df["Region"].isin(["East", "West"]), "NCG_Group"] = "Left"
    df.loc[df["Region"].isin(["South", "Midwest"]), "NCG_Group"] = "Right"
    NCG_Totals = df.groupby(["NCG_Group"]).agg(Round_6_Sum=("NCG", "sum")).reset_index()

    Winner_Totals = df["Winner"].sum()

    df = df.merge(R32_Totals, on=["Region", "R32_Group"])
    df = df.merge(S16_Totals, on=["Region", "S16_Group"])
    df = df.merge(E8_Totals, on=["Region", "E8_Group"])
    df = df.merge(F4_Totals, on=["Region"])
    df = df.merge(NCG_Totals, on=["NCG_Group"])

    df["R32"] = round(df["R32"] / df["Round_2_Sum"], 6)
    df["S16"] = round(df["S16"] / df["Round_3_Sum"], 6)
    df["E8"] = round(df["E8"] / df["Round_4_Sum"], 6)
    df["F4"] = round(df["F4"] / df["Round_5_Sum"], 6)
    df["NCG"] = round(df["NCG"] / df["Round_6_Sum"], 6)
    df["Winner"] = round(df["Winner"] / Winner_Totals, 6)

    return df[["Team", "Seed", "Region", "R32", "S16", "E8", "F4", "NCG", "Winner"]]


def standardize_predict(
    years, predictions, correct_picks, data=None, matchup_predictor=None, thresholds=None
):
    """Normalize predictions, generate picks, apply matchup corrections, score them, export results.

    Args:
        years: List of backtest training years (test year = year + 1).
        predictions: Nested dict of raw model outputs keyed by test year and round.
        correct_picks: Dict of actual tournament results keyed by year string.
        data: Full modeling DataFrame.
        matchup_predictor: Optional fitted matchup TabularPredictor.
        thresholds: Optional dict mapping round number to threshold.

    Returns:
        Tuple of (points_df, accs_df) summarizing backtesting performance.
    """
    import json
    import os

    import pandas as pd
    from models.utils.MakePicks import predict_bracket

    use_correction = matchup_predictor is not None and data is not None

    if use_correction:
        from models.utils.BracketCorrection import correct_bracket

    points = {}
    pick_accs = {}

    for year in years:
        test_year = 2021 if year == 2019 else year + 1

        pred_df = pd.DataFrame.from_dict(predictions[test_year])
        pred_df = standarize(pred_df)

        path = os.path.join(os.path.abspath(os.getcwd()), f"results/probabilities/{test_year}.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pred_df.to_csv(path, index=False)

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

        # Generate initial backward-selection bracket
        picks, point, acc = predict_bracket(points_df, correct_picks[str(test_year)])
        assert isinstance(acc, dict)

        # Apply forward-pass matchup corrections if predictor is available.
        if use_correction:
            if isinstance(matchup_predictor, dict):
                predictor_for_year = matchup_predictor.get(test_year)
            else:
                predictor_for_year = matchup_predictor

            if predictor_for_year is not None:
                year_data = data[data["Year"] == test_year][["Team", "Seed", "Region"]]
                full_year_data = data[data["Year"] == test_year]
                picks = correct_bracket(
                    picks, year_data, full_year_data, predictor_for_year, thresholds=thresholds
                )
                # Re-score the corrected bracket
                from models.utils.MakePicks import real_Bracket

                point, acc = real_Bracket(picks, correct_picks[str(test_year)])

        path = os.path.join(os.path.abspath(os.getcwd()), f"results/picks/{test_year}.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(picks, f)

        points[test_year] = point
        pick_accs[test_year] = {}
        pick_accs[test_year]["R32"] = acc["R32"] / 32
        pick_accs[test_year]["S16"] = acc["S16"] / 16
        pick_accs[test_year]["E8"] = acc["E8"] / 8
        pick_accs[test_year]["F4"] = acc["F4"] / 4
        pick_accs[test_year]["NCG"] = acc["NCG"] / 2
        pick_accs[test_year]["Winner"] = acc["Winner"]

    points_df = pd.DataFrame([points])
    points_df["Mean"] = points_df.mean(axis=1).iloc[0]
    points_df["SD"] = points_df.std(axis=1).iloc[0]

    accs_df = pd.DataFrame(pick_accs).reset_index()
    accs_df.rename(columns={"index": "Round"}, inplace=True)
    accs_df["Mean"] = accs_df.iloc[:, 1:].mean(axis=1)
    accs_df["Standard Deviation"] = accs_df.iloc[:, 1:-1].std(axis=1)

    return points_df, accs_df
