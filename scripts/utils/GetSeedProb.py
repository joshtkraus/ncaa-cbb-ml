"""Historical seed advancement probability calculations."""


def calc_seed_prob(df, lag=None, ind_col=True):
    """Compute historical seed advancement probabilities for each tournament year.

    For each year, calculates the rate at which each seed has advanced to each
    round in all prior years (optionally limited to a rolling window).

    Args:
        df: DataFrame containing Year, Seed, and Round columns.
        lag: Optional int limiting the lookback window to this many years.
            If None, uses all prior years.
        ind_col: If True, also returns a First_Year indicator column.

    Returns:
        DataFrame of seed advancement probabilities, one column per round.
        If ind_col is True, includes a First_Year column.
    """
    import pandas as pd

    R32_Full = []
    S16_Full = []
    E8_Full = []
    F4_Full = []
    NCG_Full = []
    Winner_Full = []

    suffix = "Full" if lag is None else str(lag)

    for year in df["Year"].unique():
        if lag is None:
            counts = (
                df[df["Year"] < year]
                .groupby(["Seed", "Round"])
                .size()
                .to_frame("Count")
                .reset_index()
            )
            total = len(df[df["Year"] < year]) / 16
        else:
            counts = (
                df[(df["Year"] < year) & (df["Year"] >= year - lag)]
                .groupby(["Seed", "Round"])
                .size()
                .to_frame("Count")
                .reset_index()
            )
            total = len(df[(df["Year"] < year) & (df["Year"] >= year - lag)]) / 16

        R32 = counts[counts["Round"] > 1].groupby("Seed")["Count"].sum().reset_index()
        S16 = counts[counts["Round"] > 2].groupby("Seed")["Count"].sum().reset_index()
        E8 = counts[counts["Round"] > 3].groupby("Seed")["Count"].sum().reset_index()
        F4 = counts[counts["Round"] > 4].groupby("Seed")["Count"].sum().reset_index()
        NCG = counts[counts["Round"] > 5].groupby("Seed")["Count"].sum().reset_index()
        Winner = counts[counts["Round"] > 6].groupby("Seed")["Count"].sum().reset_index()

        for frame in [R32, S16, E8, F4, NCG, Winner]:
            frame["Year"] = year

        R32.columns = ["Seed", "R32_Actual_" + suffix, "Year"]
        S16.columns = ["Seed", "S16_Actual_" + suffix, "Year"]
        E8.columns = ["Seed", "E8_Actual_" + suffix, "Year"]
        F4.columns = ["Seed", "F4_Actual_" + suffix, "Year"]
        NCG.columns = ["Seed", "NCG_Actual_" + suffix, "Year"]
        Winner.columns = ["Seed", "Winner_Actual_" + suffix, "Year"]

        R32["R32_Actual_" + suffix] = R32["R32_Actual_" + suffix] / total
        S16["S16_Actual_" + suffix] = S16["S16_Actual_" + suffix] / total
        E8["E8_Actual_" + suffix] = E8["E8_Actual_" + suffix] / total
        F4["F4_Actual_" + suffix] = F4["F4_Actual_" + suffix] / total
        NCG["NCG_Actual_" + suffix] = NCG["NCG_Actual_" + suffix] / (total / 2)
        Winner["Winner_Actual_" + suffix] = Winner["Winner_Actual_" + suffix] / (total / 4)

        R32_Full.append(R32)
        S16_Full.append(S16)
        E8_Full.append(E8)
        F4_Full.append(F4)
        NCG_Full.append(NCG)
        Winner_Full.append(Winner)

    R32 = pd.concat(R32_Full)
    S16 = pd.concat(S16_Full)
    E8 = pd.concat(E8_Full)
    F4 = pd.concat(F4_Full)
    NCG = pd.concat(NCG_Full)
    Winner = pd.concat(Winner_Full)

    # Drop any pre-existing _Actual_ columns to avoid merge conflicts when
    # called on a DataFrame that already contains these columns (e.g. when
    # recomputing derived features in 05_MakePredictions.py after concat).
    existing = [c for c in df.columns if "_Actual_" in c or c == "First_Year"]
    df = df.drop(columns=existing)

    df = df.merge(R32, on=["Year", "Seed"], how="left")
    df = df.merge(S16, on=["Year", "Seed"], how="left")
    df = df.merge(E8, on=["Year", "Seed"], how="left")
    df = df.merge(F4, on=["Year", "Seed"], how="left")
    df = df.merge(NCG, on=["Year", "Seed"], how="left")
    df = df.merge(Winner, on=["Year", "Seed"], how="left")
    df = df.fillna(0)

    cols = [
        "R32_Actual_" + suffix,
        "S16_Actual_" + suffix,
        "E8_Actual_" + suffix,
        "F4_Actual_" + suffix,
        "NCG_Actual_" + suffix,
        "Winner_Actual_" + suffix,
    ]

    if ind_col:
        df["First_Year"] = 0
        df.loc[df["Year"] == df["Year"].min(), "First_Year"] = 1
        return df[cols + ["First_Year"]]

    return df[cols]
