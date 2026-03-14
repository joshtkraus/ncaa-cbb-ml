"""Generate bracket predictions for the current tournament year."""

import json
import os
import tempfile

import pandas as pd
from models.utils.autogluon import _make_test_df, _make_train_df, fit_autogluon
from models.utils.MakePicks import predict_bracket
from models.utils.StandarizePredictions import standarize
from scrapers.GetData_SR import run_scraper
from utils.GetSeedProb import calc_seed_prob
from utils.GroupedMetrics import get_grouped_metrics
from utils.NameCleaner_KP import clean_KP
from utils.NameCleaner_SR import clean_SR

scraper_ind = True
year = 2026


def check_data_join(data, SR, KP):
    """Validate no rows were lost when merging SportsReference and KenPom data.

    Args:
        data: Merged DataFrame to validate.
        SR: SportsReference DataFrame.
        KP: KenPom DataFrame.

    Raises:
        ValueError: If the number of rows in SR differs from the merged data.
    """
    if len(SR) != len(data):
        print(
            "Missing KP Teams: ",
            [team for team in KP["Team"].unique() if team not in SR["Team"].unique()],
        )
        print(
            "Missing SR Teams: ",
            [team for team in SR["Team"].unique() if team not in KP["Team"].unique()],
        )
        raise ValueError("Data Loss in Join.")


# ---------------------------------------------------------------------------
# Data ingestion
# ---------------------------------------------------------------------------

if scraper_ind:
    SR = run_scraper(years=[year], export=False)
    assert SR is not None, "run_scraper returned None unexpectedly"
else:
    SR = pd.read_csv(
        os.path.join(os.path.abspath(os.getcwd()), "data/prediction/sportsreference.csv"),
        index_col=False,
    )

playin_KP = ["Saint Francis", "Texas", "American", "San Diego St."]

summary_temp = pd.read_csv(
    os.path.join(os.path.abspath(os.getcwd()), "data/prediction/KP/summary.csv"), index_col=False
)
points_temp = pd.read_csv(
    os.path.join(os.path.abspath(os.getcwd()), "data/prediction/KP/points.csv"), index_col=False
)
roster_temp = pd.read_csv(
    os.path.join(os.path.abspath(os.getcwd()), "data/prediction/KP/roster.csv"), index_col=False
)
roster_temp.drop(columns=["Continuity", "RankContinuity"], inplace=True)

summary_temp.columns = pd.Index([
    "Year",
    "Team",
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
])
points_temp.columns = pd.Index([
    "Year",
    "Team",
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
])
roster_temp.columns = pd.Index([
    "Year",
    "Team",
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
])

summary_temp = summary_temp[~summary_temp["Team"].isin(playin_KP)]
KP = summary_temp.merge(points_temp, on=["Year", "Team"])
KP = KP.merge(roster_temp, on=["Year", "Team"])

KP = clean_KP(KP)
SR = clean_SR(SR)
data = SR.merge(KP, on=["Team", "Year"])
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)
check_data_join(data, SR, KP)

# ---------------------------------------------------------------------------
# Merge with historical modeling data and compute derived features
# ---------------------------------------------------------------------------

data_path = os.path.join(os.path.abspath(os.getcwd()), "data/processed/data.csv")
modeling_data = pd.read_csv(data_path)
modeling_data = modeling_data[modeling_data["Year"] != year]

data = data[modeling_data.columns]
data = pd.concat([modeling_data, data], ignore_index=True)
data.drop_duplicates(inplace=True)

data[
    [
        "R32_Actual_Full",
        "S16_Actual_Full",
        "E8_Actual_Full",
        "F4_Actual_Full",
        "NCG_Actual_Full",
        "Winner_Actual_Full",
        "First_Year",
    ]
] = calc_seed_prob(data, lag=None, ind_col=True)
data[
    [
        "R32_Actual_12",
        "S16_Actual_12",
        "E8_Actual_12",
        "F4_Actual_12",
        "NCG_Actual_12",
        "Winner_Actual_12",
    ]
] = calc_seed_prob(data, lag=12, ind_col=False)
data[
    [
        "R32_Actual_6",
        "S16_Actual_6",
        "E8_Actual_6",
        "F4_Actual_6",
        "NCG_Actual_6",
        "Winner_Actual_6",
    ]
] = calc_seed_prob(data, lag=6, ind_col=False)

data = get_grouped_metrics(data)

data_path = os.path.join(os.path.abspath(os.getcwd()), "data/prediction/data.csv")
data.to_csv(data_path, index=False)

# ---------------------------------------------------------------------------
# Load frozen AutoGluon params
# ---------------------------------------------------------------------------

params_path = os.path.join(os.path.abspath(os.getcwd()), "model/autogluon_params.json")
with open(params_path, "r") as f:
    ag_params = json.load(f)
ag_params = {int(k): v for k, v in ag_params.items()}

# ---------------------------------------------------------------------------
# Generate predictions — refit one model per round on all historical data,
# predict on the current tournament year.
# ---------------------------------------------------------------------------

predictions = {}
predictions["Team"] = data.loc[data["Year"] == year, "Team"].values
predictions["Seed"] = data.loc[data["Year"] == year, "Seed"].values
predictions["Region"] = data.loc[data["Year"] == year, "Region"].values

for r in range(2, 8):
    print(f"Round {r}")
    train_mask = data["Year"].values < year
    test_mask = data["Year"].values == year

    train_df = _make_train_df(data, r, train_mask)
    test_df = _make_test_df(data, r, test_mask)

    with tempfile.TemporaryDirectory() as tmp_dir:
        predictor = fit_autogluon(train_df, ag_params[r], save_path=tmp_dir)
        predictions["Round_" + str(r)] = predictor.predict_proba(test_df)[1].values

# ---------------------------------------------------------------------------
# Standardize and generate bracket picks
# ---------------------------------------------------------------------------

pred_df = pd.DataFrame.from_dict(predictions)
pred_df = pred_df[
    [
        "Team",
        "Seed",
        "Region",
        "Round_2",
        "Round_3",
        "Round_4",
        "Round_5",
        "Round_6",
        "Round_7",
    ]
]

pred_df = standarize(pred_df)

points_df = pred_df.copy()
points_df["R32"] = points_df["R32"] * 10
points_df["S16"] = points_df["R32"] + points_df["S16"] * 20
points_df["E8"] = points_df["S16"] + points_df["E8"] * 40
points_df["F4"] = points_df["E8"] + points_df["F4"] * 80
points_df["NCG"] = points_df["F4"] + points_df["NCG"] * 160
points_df["Winner"] = points_df["NCG"] + points_df["Winner"] * 320

picks = predict_bracket(points_df, real_picks=None, calc_correct=False)

path = os.path.join(os.path.abspath(os.getcwd()), "prediction/probabilities.csv")
pred_df.to_csv(path, index=False)
path = os.path.join(os.path.abspath(os.getcwd()), "prediction/picks.json")
with open(path, "w") as f:
    json.dump(picks, f)
