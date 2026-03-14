"""Forward-pass bracket correction using the matchup model."""

from __future__ import annotations

import copy

_R32_SEED_PAIRS = {
    "1": (1, 16),
    "8": (8, 9),
    "5": (5, 12),
    "4": (4, 13),
    "6": (6, 11),
    "3": (3, 14),
    "7": (7, 10),
    "2": (2, 15),
}
_S16_TO_E8 = {"1": "Upper", "4": "Upper", "3": "Lower", "2": "Lower"}
_F4_PAIRS = [("West", "East"), ("South", "Midwest")]
_REGIONS = ["West", "East", "South", "Midwest"]


def correct_bracket(picks_dict, team_data, full_data, predictor, threshold=0.5):
    """Apply forward-pass matchup corrections to a backward-selected bracket.

    Args:
        picks_dict: Nested bracket dict produced by create_picks.
        team_data: DataFrame for the current tournament year.
        full_data: Full modeling DataFrame for the current year.
        predictor: Fitted matchup TabularPredictor.
        threshold: Minimum probability for the matchup model.

    Returns:
        Corrected picks_dict with matchup-model overrides applied.
    """
    picks = copy.deepcopy(picks_dict)
    _correct_r32(picks, team_data, full_data, predictor, threshold)
    _correct_s16(picks, full_data, predictor, threshold)
    _correct_e8(picks, full_data, predictor, threshold)
    _correct_f4(picks, full_data, predictor, threshold)
    _correct_ncg(picks, full_data, predictor, threshold)
    _correct_winner(picks, full_data, predictor, threshold)
    return picks


def _correct_r32(picks, team_data, full_data, predictor, threshold):
    for region in _REGIONS:
        for slot, (seed_hi, seed_lo) in _R32_SEED_PAIRS.items():
            current = picks[region]["R32"][slot]
            if not current:
                continue
            current_team = current[0]
            opponent = _find_seed_in_region(team_data, region, seed_hi, seed_lo, current_team)
            if opponent is None:
                continue
            override = _run_correction(current_team, opponent, 2, full_data, predictor, threshold)
            if override is not None:
                picks[region]["R32"][slot] = [override]


def _correct_s16(picks, full_data, predictor, threshold):
    for region in _REGIONS:
        for slot_a, slot_b in [("1", "4"), ("3", "2")]:
            picks_a = picks[region]["S16"][slot_a]
            picks_b = picks[region]["S16"][slot_b]
            if not picks_a or not picks_b:
                continue
            team_a, team_b = picks_a[0], picks_b[0]
            override = _run_correction(team_a, team_b, 3, full_data, predictor, threshold)
            if override is None:
                continue
            loser = team_b if override == team_a else team_a
            winner_slot = slot_a if override == team_a else slot_b
            e8_half = _S16_TO_E8[winner_slot]
            if picks[region]["E8"][e8_half] and picks[region]["E8"][e8_half][0] == loser:
                picks[region]["E8"][e8_half] = [override]
                _propagate_override(picks, region, loser, override, do_e8=False)


def _correct_e8(picks, full_data, predictor, threshold):
    for region in _REGIONS:
        upper = picks[region]["E8"]["Upper"]
        lower = picks[region]["E8"]["Lower"]
        if not upper or not lower:
            continue
        override = _run_correction(upper[0], lower[0], 4, full_data, predictor, threshold)
        if override is None:
            continue
        loser = lower[0] if override == upper[0] else upper[0]
        current_f4 = picks[region]["F4"]
        if current_f4 == loser or (isinstance(current_f4, list) and loser in current_f4):
            picks[region]["F4"] = override
            _propagate_override(
                picks, region, loser, override, do_e8=False, do_s16=False, do_r32=False, do_f4=True
            )


def _correct_f4(picks, full_data, predictor, threshold):
    for reg_a, reg_b in _F4_PAIRS:
        f4_a = picks[reg_a]["F4"]
        f4_b = picks[reg_b]["F4"]
        if not f4_a or not f4_b:
            continue
        team_a = f4_a if isinstance(f4_a, str) else f4_a[0]
        team_b = f4_b if isinstance(f4_b, str) else f4_b[0]
        override = _run_correction(team_a, team_b, 5, full_data, predictor, threshold)
        if override is None:
            continue
        loser = team_b if override == team_a else team_a
        if loser in picks["NCG"]:
            picks["NCG"] = [override if t == loser else t for t in picks["NCG"]]
            if picks["Winner"] and picks["Winner"][0] == loser:
                picks["Winner"] = [override]


def _correct_ncg(picks, full_data, predictor, threshold):
    if len(picks["NCG"]) != 2:
        return
    ncg_a, ncg_b = picks["NCG"]
    override = _run_correction(ncg_a, ncg_b, 6, full_data, predictor, threshold)
    if override is not None:
        loser = ncg_b if override == ncg_a else ncg_a
        if picks["Winner"] and picks["Winner"][0] == loser:
            picks["Winner"] = [override]


def _correct_winner(picks, full_data, predictor, threshold):
    if len(picks["NCG"]) != 2:
        return
    ncg_a, ncg_b = picks["NCG"]
    override = _run_correction(ncg_a, ncg_b, 7, full_data, predictor, threshold)
    if override is not None:
        picks["Winner"] = [override]


def _run_correction(backward_pick, opponent, round_num, full_data, predictor, threshold):
    """Run the matchup model and return an override team name if warranted.

    Args:
        backward_pick: The backward selection's current pick for this game.
        opponent: The other team in this matchup.
        round_num: Forward round number (2-7).
        full_data: Full modeling DataFrame for feature construction.
        predictor: Fitted matchup TabularPredictor.
        threshold: Minimum confidence required to trigger an override.

    Returns:
        Name of the override team, or None if no override is warranted.
    """
    from models.utils.DataProcessing_matchup import make_matchup_pred_df

    year_data = full_data[full_data["Year"] == full_data["Year"].max()]
    row_a = year_data[year_data["Team"] == backward_pick]
    row_b = year_data[year_data["Team"] == opponent]

    if row_a.empty or row_b.empty:
        return None

    pred_df, was_swapped = make_matchup_pred_df(row_a.iloc[0], row_b.iloc[0], round_num, full_data)
    prob_team_a = float(predictor.predict_proba(pred_df)[1].iloc[0])

    if prob_team_a >= 0.5:
        model_winner_is_team_a, confidence = True, prob_team_a
    else:
        model_winner_is_team_a, confidence = False, 1.0 - prob_team_a

    model_pick = (
        (backward_pick if model_winner_is_team_a else opponent)
        if not was_swapped
        else (opponent if model_winner_is_team_a else backward_pick)
    )

    return model_pick if model_pick != backward_pick and confidence > threshold else None


def _find_seed_in_region(team_data, region, seed_hi, seed_lo, exclude_team):
    """Find the R64 opponent for a given R32 slot.

    Args:
        team_data: DataFrame with Team, Seed, Region columns.
        region: Region to search within.
        seed_hi: First seed in the pairing.
        seed_lo: Second seed in the pairing.
        exclude_team: The current backward pick to exclude.

    Returns:
        Team name of the opponent, or None if not found.
    """
    candidates = team_data[
        (team_data["Region"] == region)
        & (team_data["Seed"].isin([seed_hi, seed_lo]))
        & (team_data["Team"] != exclude_team)
    ]
    return candidates.iloc[0]["Team"] if not candidates.empty else None


def _propagate_override(
    picks, region, loser, winner, do_f4=False, do_e8=True, do_s16=True, do_r32=True
):
    """Replace loser with winner in all applicable round slots within a region.

    Args:
        picks: Nested bracket picks dict, modified in place.
        region: Region to update.
        loser: Team being replaced.
        winner: Team replacing the loser.
        do_f4: If True, update F4 and propagate to NCG/Winner.
        do_e8: If True, update E8 slots.
        do_s16: If True, update S16 slots.
        do_r32: If True, update R32 slots.
    """
    if do_f4:
        _swap_f4(picks, region, loser, winner)
    if do_e8:
        _swap_slots(picks[region]["E8"], loser, winner)
    if do_s16:
        _swap_slots(picks[region]["S16"], loser, winner)
    if do_r32:
        _swap_slots(picks[region]["R32"], loser, winner)


def _swap_f4(picks, region, loser, winner):
    if picks[region]["F4"] != loser:
        return
    picks[region]["F4"] = winner
    picks["NCG"] = [winner if t == loser else t for t in picks["NCG"]]
    if picks["Winner"] and picks["Winner"][0] == loser:
        picks["Winner"] = [winner]


def _swap_slots(slots, loser, winner):
    for slot, lst in slots.items():
        if lst and lst[0] == loser:
            slots[slot] = [winner]
