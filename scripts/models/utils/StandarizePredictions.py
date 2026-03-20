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
    """Normalize predictions, generate picks via simulation, score them, export results.

    For each backtest year, generates n_top=1000 candidate brackets via simulate_picks
    and scores every candidate against actual results. This produces:
    - Top-1:  the best candidate by combined simulation score (primary metric)
    - Top-10: the best actual-scoring bracket among the top-10 simulation candidates
    - Top-25: the best actual-scoring bracket among the top-25 simulation candidates
    - Score distribution: all 1000 candidates scored against actuals, sorted descending

    Args:
        years: List of backtest training years (test year = year + 1).
        predictions: Nested dict of raw model outputs keyed by test year and round.
        correct_picks: Dict of actual tournament results keyed by year string.
        data: Full modeling DataFrame (required for simulation).
        matchup_predictor: Optional dict of fitted matchup TabularPredictors keyed by year,
            or a single predictor applied to all years.
        thresholds: Unused, retained for API compatibility.

    Returns:
        Tuple of (points_df, accs_df) where points_df includes Top1, Top10, Top25 rows.
    """
    import json
    import os

    import numpy as np
    import pandas as pd
    from models.utils.MakePicks import real_Bracket
    from models.utils.SimulatePicks import simulate_picks

    N_TOP = 1000  # total candidates to generate and rank
    K_DIST = 1000  # candidates to score against actuals for distribution plot

    points_top1 = {}
    points_top10 = {}
    points_top25 = {}
    pick_accs = {}
    score_distributions = {}  # {test_year: sorted array of K_DIST actual scores}

    for year in years:
        test_year = 2021 if year == 2019 else year + 1

        pred_df = pd.DataFrame.from_dict(predictions[test_year])
        pred_df = standarize(pred_df)

        path = os.path.join(os.path.abspath(os.getcwd()), f"results/probabilities/{test_year}.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pred_df.to_csv(path, index=False)

        # Resolve per-year matchup predictor
        if matchup_predictor is not None and data is not None:
            if isinstance(matchup_predictor, dict):
                predictor_for_year = matchup_predictor.get(test_year)
            else:
                predictor_for_year = matchup_predictor
        else:
            predictor_for_year = None

        full_year_data = data[data["Year"] == test_year] if data is not None else None

        # Generate top N_TOP candidates ranked by combined simulation score
        top_brackets = simulate_picks(
            pred_df,
            n_top=N_TOP,
            predictor=predictor_for_year,
            full_data=full_year_data,
        )

        # Score every candidate against actual results
        actual = correct_picks[str(test_year)]
        actual_scores = [real_Bracket(b, actual)[0] for b in top_brackets]
        actual_scores_arr = np.array(actual_scores)

        # Score distribution: all K_DIST candidates in simulation rank order
        # (rank 1 = highest combined simulation score, rank K_DIST = lowest)
        score_distributions[test_year] = actual_scores_arr[:K_DIST].tolist()

        # Top-1: best by simulation score (index 0 — already ranked)
        point_top1, acc = real_Bracket(top_brackets[0], actual)
        points_top1[test_year] = point_top1

        # Top-10: best actual score among simulation top-10
        points_top10[test_year] = int(actual_scores_arr[:10].max())

        # Top-25: best actual score among simulation top-25
        points_top25[test_year] = int(actual_scores_arr[:25].max())

        # Save best-scoring bracket (Top-1)
        path = os.path.join(os.path.abspath(os.getcwd()), f"results/picks/{test_year}.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(top_brackets[0], f)

        pick_accs[test_year] = {}
        pick_accs[test_year]["R32"] = acc["R32"] / 32
        pick_accs[test_year]["S16"] = acc["S16"] / 16
        pick_accs[test_year]["E8"] = acc["E8"] / 8
        pick_accs[test_year]["F4"] = acc["F4"] / 4
        pick_accs[test_year]["NCG"] = acc["NCG"] / 2
        pick_accs[test_year]["Winner"] = acc["Winner"]

    # Build points DataFrame with three rows: Top1, Top10, Top25
    def _make_row(pts_dict, label):
        row = {yr: pts_dict[yr] for yr in pts_dict}
        row["Mean"] = float(np.mean(list(pts_dict.values())))
        row["SD"] = float(np.std(list(pts_dict.values())))
        return pd.DataFrame([row], index=[label])

    points_df = pd.concat([
        _make_row(points_top1, "Top1"),
        _make_row(points_top10, "Top10"),
        _make_row(points_top25, "Top25"),
    ])

    accs_df = pd.DataFrame(pick_accs).reset_index()
    accs_df.rename(columns={"index": "Round"}, inplace=True)
    accs_df["Mean"] = accs_df.iloc[:, 1:].mean(axis=1)
    accs_df["Standard Deviation"] = accs_df.iloc[:, 1:-1].std(axis=1)

    return points_df, accs_df, score_distributions
