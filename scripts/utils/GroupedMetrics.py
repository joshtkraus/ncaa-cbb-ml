"""Bracket-pod-relative feature engineering for tournament modeling."""


def get_grouped_metrics(df):
    """Compute leave-one-out group averages for each team relative to its bracket pod.

    For each KenPom metric, calculates how a team compares to the average of
    other teams it could face at each round stage (R32, S16, E8, F4, NCG, Winner).
    Also computes a seed-group average for baseline comparison.

    Args:
        df: DataFrame containing all KenPom features, Year, Region, and Seed columns.

    Returns:
        DataFrame with additional difference columns for each metric and round stage.
    """
    from warnings import simplefilter

    import pandas as pd

    simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

    cols = [
        "Tempo",
        "RankTempo",
        "AdjTempo",
        "RankAdjTempo",
        "OE",
        "RankOE",
        "AdjOE",
        "RankAdjOE",
        "DE",
        "RankDE",
        "AdjDE",
        "RankAdjDE",
        "AdjEM",
        "RankAdjEM",
        "Off_1",
        "RankOff_1",
        "Off_2",
        "RankOff_2",
        "Off_3",
        "RankOff_3",
        "Def_1",
        "RankDef_1",
        "Def_2",
        "RankDef_2",
        "Def_3",
        "RankDef_3",
        "Size",
        "SizeRank",
        "Hgt5",
        "Hgt5Rank",
        "Hgt4",
        "Hgt4Rank",
        "Hgt3",
        "Hgt3Rank",
        "Hgt2",
        "Hgt2Rank",
        "Hgt1",
        "Hgt1Rank",
        "HgtEff",
        "HgtEffRank",
        "Exp",
        "ExpRank",
        "Bench",
        "BenchRank",
        "Pts5",
        "Pts5Rank",
        "Pts4",
        "Pts4Rank",
        "Pts3",
        "Pts3Rank",
        "Pts2",
        "Pts2Rank",
        "Pts1",
        "Pts1Rank",
        "OR5",
        "OR5Rank",
        "OR4",
        "OR4Rank",
        "OR3",
        "OR3Rank",
        "OR2",
        "OR2Rank",
        "OR1",
        "OR1Rank",
        "DR5",
        "DR5Rank",
        "DR4",
        "DR4Rank",
        "DR3",
        "DR3Rank",
        "DR2",
        "DR2Rank",
        "DR1",
        "DR1Rank",
    ]

    df["R32_Group"] = 0
    df.loc[df["Seed"].isin([1, 16]), "R32_Group"] = 1
    df.loc[df["Seed"].isin([2, 15]), "R32_Group"] = 2
    df.loc[df["Seed"].isin([3, 14]), "R32_Group"] = 3
    df.loc[df["Seed"].isin([4, 13]), "R32_Group"] = 4
    df.loc[df["Seed"].isin([5, 12]), "R32_Group"] = 5
    df.loc[df["Seed"].isin([6, 11]), "R32_Group"] = 6
    df.loc[df["Seed"].isin([7, 10]), "R32_Group"] = 7
    df.loc[df["Seed"].isin([8, 9]), "R32_Group"] = 8

    df["S16_Group"] = 0
    df.loc[df["Seed"].isin([1, 16, 8, 9]), "S16_Group"] = 1
    df.loc[df["Seed"].isin([2, 15, 7, 10]), "S16_Group"] = 2
    df.loc[df["Seed"].isin([3, 14, 6, 11]), "S16_Group"] = 3
    df.loc[df["Seed"].isin([4, 13, 5, 12]), "S16_Group"] = 4

    df["E8_Group"] = 0
    df.loc[df["Seed"].isin([1, 16, 8, 9, 4, 13, 5, 12]), "E8_Group"] = 1
    df.loc[df["Seed"].isin([2, 15, 7, 10, 3, 14, 6, 11]), "E8_Group"] = 2

    df["NCG_Group"] = "0"
    df.loc[df["Region"].isin(["East", "West"]), "NCG_Group"] = "Left"
    df.loc[df["Region"].isin(["South", "Midwest"]), "NCG_Group"] = "Right"

    for c in cols:
        for group_key, _, prefix in [
            (["Year", "Region", "R32_Group"], None, "R32"),
            (["Year", "Region", "S16_Group"], None, "S16"),
            (["Year", "Region", "E8_Group"], None, "E8"),
            (["Year", "Region", "Region"], None, "F4"),
            (["Year", "NCG_Group"], None, "NCG"),
            (["Year"], None, "Winner"),
        ]:
            grouped = df.groupby(group_key)
            n = grouped[c].transform("count")
            mean = grouped[c].transform("mean")
            df[prefix + "_" + c + "_Avg"] = (mean * n - df[c]) / (n - 1)

        grouped = df.groupby(["Year", "Seed"])
        n = grouped[c].transform("count")
        mean = grouped[c].transform("mean")
        df[c + "_Seed_Avg"] = (mean * n - df[c]) / (n - 1)

        avg_cols = [f"{p}_{c}_Avg" for p in ["R32", "S16", "E8", "F4", "NCG", "Winner"]]
        seed_col = c + "_Seed_Avg"
        if "Rank" in c:
            for col in avg_cols + [seed_col]:
                df[col] = df[col] - df[c]
        else:
            for col in avg_cols + [seed_col]:
                df[col] = df[c] - df[col]

    df.drop(columns=["R32_Group", "S16_Group", "E8_Group", "NCG_Group"], inplace=True)
    return df
