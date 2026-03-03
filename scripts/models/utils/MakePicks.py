"""Bracket pick generation and scoring utilities."""


def R32(picks_dict, rg, sd, tm):
    """Assign a team to the correct R32 matchup slot based on seed.

    Args:
        picks_dict: Nested bracket picks dictionary.
        rg: Region name string.
        sd: Seed number.
        tm: Team name string.

    Returns:
        Updated picks_dict.
    """
    if str(sd) in ["1", "16"]:
        picks_dict[rg]["R32"]["1"].append(tm)
    elif str(sd) in ["8", "9"]:
        picks_dict[rg]["R32"]["8"].append(tm)
    elif str(sd) in ["5", "12"]:
        picks_dict[rg]["R32"]["5"].append(tm)
    elif str(sd) in ["4", "13"]:
        picks_dict[rg]["R32"]["4"].append(tm)
    elif str(sd) in ["6", "11"]:
        picks_dict[rg]["R32"]["6"].append(tm)
    elif str(sd) in ["3", "14"]:
        picks_dict[rg]["R32"]["3"].append(tm)
    elif str(sd) in ["7", "10"]:
        picks_dict[rg]["R32"]["7"].append(tm)
    else:
        picks_dict[rg]["R32"]["2"].append(tm)
    return picks_dict


def S16(picks_dict, rg, sd, tm):
    """Assign a team to the correct S16 matchup slot based on seed.

    Args:
        picks_dict: Nested bracket picks dictionary.
        rg: Region name string.
        sd: Seed number.
        tm: Team name string.

    Returns:
        Updated picks_dict.
    """
    if str(sd) in ["1", "16", "8", "9"]:
        picks_dict[rg]["S16"]["1"].append(tm)
    elif str(sd) in ["5", "12", "4", "13"]:
        picks_dict[rg]["S16"]["4"].append(tm)
    elif str(sd) in ["3", "14", "6", "11"]:
        picks_dict[rg]["S16"]["3"].append(tm)
    else:
        picks_dict[rg]["S16"]["2"].append(tm)
    return picks_dict


def E8(picks_dict, rg, sd, tm):
    """Assign a team to the correct E8 matchup slot (Upper/Lower half) based on seed.

    Args:
        picks_dict: Nested bracket picks dictionary.
        rg: Region name string.
        sd: Seed number.
        tm: Team name string.

    Returns:
        Updated picks_dict.
    """
    if str(sd) in ["1", "16", "8", "9", "5", "12", "4", "13"]:
        picks_dict[rg]["E8"]["Upper"].append(tm)
    else:
        picks_dict[rg]["E8"]["Lower"].append(tm)
    return picks_dict


def _init_picks_dict():
    """Initialise an empty nested bracket picks dictionary.

    Returns:
        Dict with empty slot lists for all four regions, NCG, and Winner.
    """

    def _region():
        return {
            "F4": [],
            "E8": {"Upper": [], "Lower": []},
            "S16": {"1": [], "4": [], "3": [], "2": []},
            "R32": {"1": [], "8": [], "5": [], "4": [], "6": [], "3": [], "7": [], "2": []},
        }

    return {
        "West": _region(),
        "East": _region(),
        "South": _region(),
        "Midwest": _region(),
        "NCG": [],
        "Winner": [],
    }


def _team_info(team_data, tm):
    """Look up seed and region for a team in the data.

    Args:
        team_data: DataFrame with Team, Seed, and Region columns.
        tm: Team name string to look up.

    Returns:
        Tuple of (seed, region).
    """
    sd = team_data.loc[team_data["Team"] == tm, "Seed"].item()
    rg = team_data.loc[team_data["Team"] == tm, "Region"].item()
    return sd, rg


def _propagate(picks_dict, rg, sd, tm, do_e8=True, do_s16=True, do_r32=True):
    """Propagate a team pick downward through earlier rounds.

    Args:
        picks_dict: Nested bracket picks dictionary, modified in place.
        rg: Region name string.
        sd: Seed number.
        tm: Team name string.
        do_e8: If True, also assign the team to the E8 slot.
        do_s16: If True, also assign the team to the S16 slot.
        do_r32: If True, also assign the team to the R32 slot.

    Returns:
        Updated picks_dict.
    """
    if do_e8:
        picks_dict = E8(picks_dict, rg, sd, tm)
    if do_s16:
        picks_dict = S16(picks_dict, rg, sd, tm)
    if do_r32:
        picks_dict = R32(picks_dict, rg, sd, tm)
    return picks_dict


def _sorted_pred(team_data, round_col):
    """Return team_data sorted descending by a round column, index reset.

    Args:
        team_data: DataFrame with Team, Seed, Region, and round score columns.
        round_col: Column name string to sort by.

    Returns:
        Sorted DataFrame with a fresh integer index.
    """
    pred = team_data[["Team", "Seed", "Region", round_col]].sort_values(
        by=round_col, ascending=False
    )
    pred.reset_index(inplace=True)
    return pred


def _pick_winner(team_data, picks_dict, counters):
    """Select the overall tournament winner and propagate into all earlier rounds.

    Args:
        team_data: DataFrame with round score columns.
        picks_dict: Nested bracket picks dictionary, modified in place.
        counters: Dict of round counters (f4, e8, s16, r32), modified in place.

    Returns:
        Tuple of (updated picks_dict, win_reg string).
    """
    pred = _sorted_pred(team_data, "Winner")
    tm = pred["Team"][0]
    sd, rg = _team_info(team_data, tm)
    picks_dict["Winner"].append(tm)
    picks_dict["NCG"].append(tm)
    picks_dict[rg]["F4"] = tm
    picks_dict = _propagate(picks_dict, rg, sd, tm)
    counters["f4"] += 1
    counters["e8"] += 1
    counters["s16"] += 1
    counters["r32"] += 1
    return picks_dict, rg


def _pick_ncg(team_data, picks_dict, counters, win_reg):
    """Select the runner-up from the opposite bracket side and propagate.

    Args:
        team_data: DataFrame with round score columns.
        picks_dict: Nested bracket picks dictionary, modified in place.
        counters: Dict of round counters, modified in place.
        win_reg: Region string of the already-selected winner.

    Returns:
        Updated picks_dict.
    """
    pred = _sorted_pred(team_data, "NCG")
    i = 0
    while len(picks_dict["NCG"]) < 2:
        tm = pred["Team"][i]
        sd, rg = _team_info(team_data, tm)
        opposite = (win_reg in ("West", "East") and rg not in ("West", "East")) or (
            win_reg in ("South", "Midwest") and rg not in ("South", "Midwest")
        )
        if opposite:
            picks_dict["NCG"].append(tm)
            picks_dict[rg]["F4"] = tm
            picks_dict = _propagate(picks_dict, rg, sd, tm)
            counters["f4"] += 1
            counters["e8"] += 1
            counters["s16"] += 1
            counters["r32"] += 1
        i += 1
    return picks_dict


def _pick_f4(team_data, picks_dict, counters):
    """Fill the remaining two Final Four spots (one per unfilled region).

    Args:
        team_data: DataFrame with round score columns.
        picks_dict: Nested bracket picks dictionary, modified in place.
        counters: Dict of round counters, modified in place.

    Returns:
        Updated picks_dict.
    """
    pred = _sorted_pred(team_data, "F4")
    i = 0
    while counters["f4"] < 4:
        tm = pred["Team"][i]
        sd, rg = _team_info(team_data, tm)
        if len(picks_dict[rg]["F4"]) == 0:
            picks_dict[rg]["F4"] = tm
            picks_dict = _propagate(picks_dict, rg, sd, tm)
            counters["f4"] += 1
            counters["e8"] += 1
            counters["s16"] += 1
            counters["r32"] += 1
        i += 1
    return picks_dict


def _pick_e8(team_data, picks_dict, counters):
    """Fill all eight Elite Eight slots (upper/lower half per region).

    Args:
        team_data: DataFrame with round score columns.
        picks_dict: Nested bracket picks dictionary, modified in place.
        counters: Dict of round counters, modified in place.

    Returns:
        Updated picks_dict.
    """
    pred = _sorted_pred(team_data, "E8")
    i = 0
    while counters["e8"] < 8:
        tm = pred["Team"][i]
        sd, rg = _team_info(team_data, tm)
        found = False
        if str(sd) in ["1", "16", "8", "9", "5", "12", "4", "13"]:
            if len(picks_dict[rg]["E8"]["Upper"]) == 0:
                picks_dict[rg]["E8"]["Upper"].append(tm)
                found = True
        else:
            if len(picks_dict[rg]["E8"]["Lower"]) == 0:
                picks_dict[rg]["E8"]["Lower"].append(tm)
                found = True
        if found:
            picks_dict = _propagate(picks_dict, rg, sd, tm, do_e8=False)
            counters["e8"] += 1
            counters["s16"] += 1
            counters["r32"] += 1
        i += 1
    return picks_dict


def _pick_s16(team_data, picks_dict, counters):
    """Fill all sixteen Sweet Sixteen slots (one per seed pod per region).

    Args:
        team_data: DataFrame with round score columns.
        picks_dict: Nested bracket picks dictionary, modified in place.
        counters: Dict of round counters, modified in place.

    Returns:
        Updated picks_dict.
    """
    pred = _sorted_pred(team_data, "S16")
    i = 0
    while counters["s16"] < 16:
        tm = pred["Team"][i]
        sd, rg = _team_info(team_data, tm)
        found = False
        if str(sd) in ["1", "16", "8", "9"]:
            if len(picks_dict[rg]["S16"]["1"]) == 0:
                picks_dict[rg]["S16"]["1"].append(tm)
                found = True
        elif str(sd) in ["5", "12", "4", "13"]:
            if len(picks_dict[rg]["S16"]["4"]) == 0:
                picks_dict[rg]["S16"]["4"].append(tm)
                found = True
        elif str(sd) in ["3", "14", "6", "11"]:
            if len(picks_dict[rg]["S16"]["3"]) == 0:
                picks_dict[rg]["S16"]["3"].append(tm)
                found = True
        else:
            if len(picks_dict[rg]["S16"]["2"]) == 0:
                picks_dict[rg]["S16"]["2"].append(tm)
                found = True
        if found:
            picks_dict = _propagate(picks_dict, rg, sd, tm, do_e8=False, do_s16=False)
            counters["s16"] += 1
            counters["r32"] += 1
        i += 1
    return picks_dict


def _pick_r32(team_data, picks_dict, counters):
    """Fill all thirty-two Round of 32 slots (one per seed matchup per region).

    Args:
        team_data: DataFrame with round score columns.
        picks_dict: Nested bracket picks dictionary, modified in place.
        counters: Dict of round counters, modified in place.

    Returns:
        Updated picks_dict.
    """
    slot_map = {
        frozenset(["1", "16"]): "1",
        frozenset(["8", "9"]): "8",
        frozenset(["5", "12"]): "5",
        frozenset(["4", "13"]): "4",
        frozenset(["6", "11"]): "6",
        frozenset(["3", "14"]): "3",
        frozenset(["7", "10"]): "7",
        frozenset(["2", "15"]): "2",
    }
    pred = _sorted_pred(team_data, "R32")
    i = 0
    while counters["r32"] < 32:
        tm = pred["Team"][i]
        sd, rg = _team_info(team_data, tm)
        slot = next((v for k, v in slot_map.items() if str(sd) in k), "2")
        if len(picks_dict[rg]["R32"][slot]) == 0:
            picks_dict[rg]["R32"][slot].append(tm)
            counters["r32"] += 1
        i += 1
    return picks_dict


def create_picks(team_data):
    """Generate a full bracket from expected-points scores, round by round.

    Iterates from Winner down to R32, at each round selecting the highest
    expected-points team that fits the required bracket slot.

    Args:
        team_data: DataFrame with columns Team, Seed, Region, and one column
            per round (Winner, NCG, F4, E8, S16, R32) containing expected points.

    Returns:
        Nested dict representing the filled bracket.
    """
    picks_dict = _init_picks_dict()
    counters = {"f4": 0, "e8": 0, "s16": 0, "r32": 0}

    picks_dict, win_reg = _pick_winner(team_data, picks_dict, counters)
    picks_dict = _pick_ncg(team_data, picks_dict, counters, win_reg)
    picks_dict = _pick_f4(team_data, picks_dict, counters)
    picks_dict = _pick_e8(team_data, picks_dict, counters)
    picks_dict = _pick_s16(team_data, picks_dict, counters)
    picks_dict = _pick_r32(team_data, picks_dict, counters)

    return picks_dict


def real_Bracket(picks, real):
    """Calculate total points and per-round accuracy against actual results.

    Args:
        picks: Nested bracket picks dict generated by create_picks.
        real: Nested dict of actual tournament results in the same structure.

    Returns:
        Tuple of (total points scored, accuracy dict keyed by round name).
    """
    point_totals = {"R32": 10, "S16": 20, "E8": 40, "F4": 80, "NCG": 160, "Winner": 320}
    accuracy = {"R32": 0, "S16": 0, "E8": 0, "F4": 0, "NCG": 0, "Winner": 0}
    total = 0

    for region in ["West", "East", "South", "Midwest"]:
        for rd in ["R32", "S16", "E8", "F4"]:
            team_list = picks[region][rd]
            if rd != "F4":
                team_list = [item for sublist in team_list.values() for item in sublist]
            else:
                team_list = [team_list]
            for team in team_list:
                if team in real[region][rd]:
                    total += point_totals[rd]
                    accuracy[rd] += 1

    for team in picks["NCG"]:
        if team in real["NCG"]:
            total += point_totals["NCG"]
            accuracy["NCG"] += 1

    team = picks["Winner"][0]
    if team == real["Winner"]:
        total += point_totals["Winner"]
        accuracy["Winner"] += 1

    return total, accuracy


def predict_bracket(team_data, real_picks=None, calc_correct=True):
    """Generate bracket picks and optionally score them against actual results.

    Args:
        team_data: DataFrame with expected points per round per team.
        real_picks: Optional dict of actual results for scoring.
        calc_correct: If True, score picks against real_picks and return accuracy.

    Returns:
        picks dict if calc_correct is False, else (picks, points, accuracy) tuple.
    """
    picks = create_picks(team_data)

    if calc_correct:
        points, accs = real_Bracket(picks, real_picks)
        return picks, points, accs

    return picks
