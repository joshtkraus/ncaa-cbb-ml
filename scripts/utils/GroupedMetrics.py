"""Bracket-pod-relative feature engineering for tournament modeling."""

from warnings import simplefilter

import pandas as pd

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

_COLS = [
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

_RANK_COLS = frozenset(c for c in _COLS if "Rank" in c)

_GROUP_KEYS = [
    (["Year", "Region", "R32_Group"], "R32"),
    (["Year", "Region", "S16_Group"], "S16"),
    (["Year", "Region", "E8_Group"], "E8"),
    (["Year", "Region"], "F4"),
    (["Year", "NCG_Group"], "NCG"),
    (["Year"], "Winner"),
]


def _leave_one_out_avgs(df, group_keys, prefix):
    """Compute leave-one-out group averages for all metric columns under one groupby.

    Args:
        df: DataFrame to operate on, modified in place.
        group_keys: Column name strings to group by.
        prefix: Output column name prefix (e.g. 'R32').

    Returns:
        DataFrame with new difference columns added.
    """
    grouped = df.groupby(group_keys)[_COLS]
    n = grouped.transform("count")
    mean = grouped.transform("mean")
    loo = (mean * n - df[_COLS]) / (n - 1)

    for c in _COLS:
        out = f"{prefix}_{c}_Avg"
        df[out] = loo[c] - df[c] if c in _RANK_COLS else df[c] - loo[c]

    return df


def _seed_avgs(df):
    """Compute leave-one-out seed-group averages for all metric columns.

    Args:
        df: DataFrame to operate on, modified in place.

    Returns:
        DataFrame with new seed difference columns added.
    """
    grouped = df.groupby(["Year", "Seed"])[_COLS]
    n = grouped.transform("count")
    mean = grouped.transform("mean")
    loo = (mean * n - df[_COLS]) / (n - 1)

    for c in _COLS:
        df[f"{c}_Seed_Avg"] = loo[c] - df[c] if c in _RANK_COLS else df[c] - loo[c]

    return df


def get_grouped_metrics(df):
    """Compute leave-one-out group averages for each team relative to its bracket pod.

    Args:
        df: DataFrame containing all KenPom features, Year, Region, and Seed columns.

    Returns:
        DataFrame with additional difference columns for each metric and round stage.
    """
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

    for group_keys, prefix in _GROUP_KEYS:
        df = _leave_one_out_avgs(df, group_keys, prefix)

    df = _seed_avgs(df)

    df.drop(columns=["R32_Group", "S16_Group", "E8_Group", "NCG_Group"], inplace=True)
    return df
