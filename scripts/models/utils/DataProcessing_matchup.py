"""Matchup dataset construction for head-to-head tournament game prediction."""

import pandas as pd

_REGION_RANK = {"West": 1, "East": 4, "South": 2, "Midwest": 3}
_DROP_COLS = ["Round", "First_Year"]
_DROP_SUFFIXES = ("_Avg", "_Seed_Avg")
_DROP_CONTAINS = ("_Actual_",)
_REGIONS = ["West", "East", "South", "Midwest"]
_REGIONAL_ROUND_MAP = {2: ("R64", "R32"), 3: ("R32", "S16"), 4: ("S16", "E8")}


def _base_features(data):
    """Strip grouped averages, survival rates, and metadata from data.

    Args:
        data: Full modeling DataFrame.

    Returns:
        DataFrame with only base feature columns retained.
    """
    drop = set(_DROP_COLS)
    for col in data.columns:
        if any(col.endswith(s) for s in _DROP_SUFFIXES):
            drop.add(col)
        if any(s in col for s in _DROP_CONTAINS):
            drop.add(col)
    return data.drop(columns=[c for c in drop if c in data.columns])


def _assign_team_a(row_a, row_b):
    """Return (team_a, team_b) where team_a is the perceived better team.

    Args:
        row_a: Series for the first team.
        row_b: Series for the second team.

    Returns:
        Tuple of (team_a_row, team_b_row) with team_a being the better team.
    """
    seed_a, seed_b = int(row_a["Seed"]), int(row_b["Seed"])
    if seed_a != seed_b:
        return (row_a, row_b) if seed_a < seed_b else (row_b, row_a)
    rank_a = _REGION_RANK.get(str(row_a["Region"]), 99)
    rank_b = _REGION_RANK.get(str(row_b["Region"]), 99)
    return (row_a, row_b) if rank_a <= rank_b else (row_b, row_a)


def _make_matchup_row(team_a, team_b, round_num, winner, numeric_cols):
    """Build a single matchup row from two team feature Series.

    Args:
        team_a: Feature Series for the better team (Team A).
        team_b: Feature Series for the worse team (Team B).
        round_num: Tournament round number (2-7).
        winner: Name of the team that won this game.
        numeric_cols: List of numeric column names to difference.

    Returns:
        Dict representing one matchup row.
    """
    row: dict = {
        "Year": int(team_a["Year"]),
        "Round": round_num,
        "Team_A": team_a["Team"],
        "Team_B": team_b["Team"],
        "Seed_A": int(team_a["Seed"]),
        "Seed_B": int(team_b["Seed"]),
        "Conf_A": team_a["Conf"],
        "Conf_B": team_b["Conf"],
        "Outcome": 1 if winner == team_a["Team"] else 0,
    }
    for col in numeric_cols:
        row[f"{col}_diff"] = float(team_a[col]) - float(team_b[col])
    return row


def _get_team_row(lookup, team, year):
    """Retrieve a team's feature row from a year-keyed lookup dict.

    Args:
        lookup: Dict keyed by (year, team) -> pd.Series.
        team: Team name.
        year: Tournament year.

    Returns:
        Feature Series or None if not found.
    """
    return lookup.get((year, team))


def _collect_regional_games(bracket, year, lookup, numeric_cols):
    """Collect all Round 2-4 matchup rows for a single tournament year.

    Args:
        bracket: Single-year bracket dict from results.json.
        year: Tournament year.
        lookup: Dict keyed by (year, team) -> feature Series.
        numeric_cols: Numeric columns to difference.

    Returns:
        List of matchup row dicts.
    """
    rows = []
    for round_num, (prev_key, adv_key) in _REGIONAL_ROUND_MAP.items():
        for region in _REGIONS:
            prev_list = bracket[region][prev_key]
            adv_set = set(bracket[region][adv_key])
            for i in range(0, len(prev_list), 2):
                t1_name, t2_name = prev_list[i], prev_list[i + 1]
                t1 = _get_team_row(lookup, t1_name, year)
                t2 = _get_team_row(lookup, t2_name, year)
                if t1 is None or t2 is None:
                    continue
                winner = t1_name if t1_name in adv_set else t2_name
                team_a, team_b = _assign_team_a(t1, t2)
                rows.append(_make_matchup_row(team_a, team_b, round_num, winner, numeric_cols))
    return rows


def _collect_e8_games(bracket, year, lookup, numeric_cols):
    """Collect all Round 5 (E8) matchup rows for a single tournament year.

    Args:
        bracket: Single-year bracket dict from results.json.
        year: Tournament year.
        lookup: Dict keyed by (year, team) -> feature Series.
        numeric_cols: Numeric columns to difference.

    Returns:
        List of matchup row dicts (one per region).
    """
    rows = []
    for region in _REGIONS:
        e8 = bracket[region]["E8"]
        f4_winner = bracket[region]["F4"][0]
        t1 = _get_team_row(lookup, e8[0], year)
        t2 = _get_team_row(lookup, e8[1], year)
        if t1 is None or t2 is None:
            continue
        team_a, team_b = _assign_team_a(t1, t2)
        rows.append(_make_matchup_row(team_a, team_b, 5, f4_winner, numeric_cols))
    return rows


def _collect_f4_games(bracket, year, lookup, numeric_cols):
    """Collect both Round 6 (F4) matchup rows for a single tournament year.

    Args:
        bracket: Single-year bracket dict from results.json.
        year: Tournament year.
        lookup: Dict keyed by (year, team) -> feature Series.
        numeric_cols: Numeric columns to difference.

    Returns:
        List of matchup row dicts (one per F4 game).
    """
    rows = []
    ncg_set = set(bracket["NCG"])
    for reg_a, reg_b in [("West", "East"), ("South", "Midwest")]:
        t1_name = bracket[reg_a]["F4"][0]
        t2_name = bracket[reg_b]["F4"][0]
        t1 = _get_team_row(lookup, t1_name, year)
        t2 = _get_team_row(lookup, t2_name, year)
        if t1 is None or t2 is None:
            continue
        winner = t1_name if t1_name in ncg_set else t2_name
        team_a, team_b = _assign_team_a(t1, t2)
        rows.append(_make_matchup_row(team_a, team_b, 6, winner, numeric_cols))
    return rows


def _collect_ncg_game(bracket, year, lookup, numeric_cols):
    """Collect the Round 7 (NCG) matchup row for a single tournament year.

    Args:
        bracket: Single-year bracket dict from results.json.
        year: Tournament year.
        lookup: Dict keyed by (year, team) -> feature Series.
        numeric_cols: Numeric columns to difference.

    Returns:
        List containing one matchup row dict, or empty if teams not found.
    """
    ncg_list = bracket["NCG"]
    if len(ncg_list) != 2:
        return []
    t1_name, t2_name = ncg_list[0], ncg_list[1]
    t1 = _get_team_row(lookup, t1_name, year)
    t2 = _get_team_row(lookup, t2_name, year)
    if t1 is None or t2 is None:
        return []
    winner = bracket["Winner"]
    team_a, team_b = _assign_team_a(t1, t2)
    return [_make_matchup_row(team_a, team_b, 7, winner, numeric_cols)]


def build_matchup_dataset(data, results):
    """Construct the full historical matchup dataset from team data and results.

    Args:
        data: Full modeling DataFrame (data.csv).
        results: Loaded results.json dict keyed by year string.

    Returns:
        DataFrame where each row is one tournament game, with differential
        features, categorical team columns, Round, and Outcome.
    """
    base = _base_features(data)
    numeric_cols = [
        c
        for c in base.columns
        if c not in ["Year", "Team", "Conf", "Region", "Seed"]
        and pd.api.types.is_numeric_dtype(base[c])
    ]
    lookup = {(int(row["Year"]), row["Team"]): row for _, row in base.iterrows()}

    rows: list[dict] = []
    for year_str, bracket in results.items():
        year = int(year_str)
        rows.extend(_collect_regional_games(bracket, year, lookup, numeric_cols))
        rows.extend(_collect_e8_games(bracket, year, lookup, numeric_cols))
        rows.extend(_collect_f4_games(bracket, year, lookup, numeric_cols))
        rows.extend(_collect_ncg_game(bracket, year, lookup, numeric_cols))

    df = pd.DataFrame(rows)
    meta_cols = ["Year", "Round", "Team_A", "Team_B", "Seed_A", "Seed_B", "Conf_A", "Conf_B"]
    diff_cols = [c for c in df.columns if c.endswith("_diff")]
    return df[meta_cols + diff_cols + ["Outcome"]].reset_index(drop=True)


def make_matchup_train_df(matchup_data, year_mask):
    """Build a labeled training DataFrame for the matchup model.

    Args:
        matchup_data: Full matchup DataFrame from build_matchup_dataset.
        year_mask: Boolean array aligned to matchup_data rows.

    Returns:
        DataFrame with features and Outcome column.
    """
    subset = matchup_data[year_mask].copy()
    return subset.drop(columns=["Team_A", "Team_B", "Year"])


def make_matchup_pred_df(team_a_row, team_b_row, round_num, data):
    """Build a single-row prediction DataFrame for a prospective matchup.

    Args:
        team_a_row: Feature Series for the first team (from data.csv).
        team_b_row: Feature Series for the second team (from data.csv).
        round_num: Tournament round number for this game.
        data: Full modeling DataFrame (used to derive numeric_cols).

    Returns:
        Tuple of (single-row DataFrame, was_swapped) where was_swapped is
        True if team_b_row ended up as Team A due to seed/region ordering.
    """
    base = _base_features(data)
    numeric_cols = [
        c
        for c in base.columns
        if c not in ["Year", "Team", "Conf", "Region", "Seed"]
        and pd.api.types.is_numeric_dtype(base[c])
    ]

    team_a, team_b = _assign_team_a(team_a_row, team_b_row)
    was_swapped = team_a["Team"] != team_a_row["Team"]

    row = {
        "Round": round_num,
        "Seed_A": int(team_a["Seed"]),
        "Seed_B": int(team_b["Seed"]),
        "Conf_A": team_a["Conf"],
        "Conf_B": team_b["Conf"],
    }
    for col in numeric_cols:
        row[f"{col}_diff"] = float(team_a[col]) - float(team_b[col])

    return pd.DataFrame([row]), was_swapped
