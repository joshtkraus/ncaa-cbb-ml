"""Tournament bracket simulation for joint-probability-aware pick selection.

Simulates N full tournaments using a two-signal blend of the matchup model
and by-round model conditional ratio. For each game, weights are drawn from
Uniform(0, 1) and normalised to sum to 1, independently per round per
simulation. This explores the full blend spectrum without constraints.

The bracket that maximises expected points across all simulations is selected,
accounting for the joint dependence of picks rather than maximising each
slot independently.

If no matchup predictor is provided, falls back to the conditional ratio only.
"""

import numpy as np
import pandas as pd

_R64_PODS = {
    "1": (1, 16),
    "8": (8, 9),
    "5": (5, 12),
    "4": (4, 13),
    "6": (6, 11),
    "3": (3, 14),
    "7": (7, 10),
    "2": (2, 15),
}
_S16_PODS = [("1", "4"), ("3", "2")]
_S16_TO_E8 = {"1": "Upper", "4": "Upper", "3": "Lower", "2": "Lower"}
_F4_PAIRS = [("West", "East"), ("South", "Midwest")]
_REGIONS = ["West", "East", "South", "Midwest"]
_R32_SLOTS = ["1", "8", "5", "4", "6", "3", "7", "2"]
_S16_SLOTS = ["1", "4", "3", "2"]
_E8_HALVES = ["Upper", "Lower"]

# Column advanced TO at each game stage (used for conditional ratio)
_STAGE_TO_COL = {
    "R64": "R32",
    "S16": "E8",
    "E8": "F4",
    "F4": "NCG",
    "NCG": "Winner",
    "Winner": "Winner",
}

# Round number passed to matchup model at each game stage
_STAGE_TO_ROUND = {"R64": 2, "S16": 3, "E8": 4, "F4": 5, "NCG": 6, "Winner": 7}

# Flat array slot ordering: R32(32) + S16(16) + E8(8) + F4(4) + NCG(2) + Winner(1) = 63
_POINTS = np.array(
    [10] * 32 + [20] * 16 + [40] * 8 + [80] * 4 + [160] * 2 + [320] * 1, dtype=np.float32
)


def _get_prob(probs_by_team, team, col):
    """Safely retrieve a probability value for a team.

    Args:
        probs_by_team: Dict keyed by team name -> probability Series.
        team: Team name string.
        col: Column name to retrieve (e.g. 'R32', 'E8').

    Returns:
        Float probability, defaulting to 0.0 if team not found.
    """
    row = probs_by_team.get(team)
    return float(row[col]) if row is not None else 0.0


def _conditional_ratio(team_a, team_b, col, probs_by_team):
    """p(A wins | both in game) from by-round marginal probabilities.

    Args:
        team_a: First team name.
        team_b: Second team name.
        col: Round probability column for the round being advanced to.
        probs_by_team: Dict keyed by team name -> probability Series.

    Returns:
        Float probability of team_a winning, in (0, 1).
    """
    pa = _get_prob(probs_by_team, team_a, col)
    pb = _get_prob(probs_by_team, team_b, col)
    total = pa + pb
    return pa / total if total > 0 else 0.5


def _matchup_prob(team_a, team_b, round_num, team_rows, predictor, numeric_cols, cache):
    """Get matchup model win probability for team_a, using lazy cache.

    Caches both (A,B) and (B,A) = 1 - (A,B) to avoid redundant calls.

    Args:
        team_a: First team name.
        team_b: Second team name.
        round_num: Tournament round number (2-7).
        team_rows: Dict keyed by team name -> feature Series from full_data.
        predictor: Fitted matchup TabularPredictor.
        numeric_cols: Pre-computed list of numeric feature column names.
        cache: Dict keyed by (team_a, team_b, round_num) -> float probability.

    Returns:
        Float probability of team_a winning in [0, 1].
    """
    from models.utils.DataProcessing_matchup import _assign_team_a

    key = (team_a, team_b, round_num)
    if key in cache:
        return cache[key]

    row_a = team_rows.get(team_a)
    row_b = team_rows.get(team_b)
    if row_a is None or row_b is None:
        cache[key] = 0.5
        cache[(team_b, team_a, round_num)] = 0.5
        return 0.5

    team_a_ord, team_b_ord = _assign_team_a(row_a, row_b)
    was_swapped = team_a_ord["Team"] != team_a

    row = {
        "Round": round_num,
        "Seed_A": int(team_a_ord["Seed"]),
        "Seed_B": int(team_b_ord["Seed"]),
        "Conf_A": team_a_ord["Conf"],
        "Conf_B": team_b_ord["Conf"],
    }
    for col in numeric_cols:
        row[f"{col}_diff"] = float(team_a_ord[col]) - float(team_b_ord[col])
    row["AdjEM_vs_Seed_diff"] = float(team_a_ord["AdjEM_Seed_Avg"]) - float(
        team_b_ord["AdjEM_Seed_Avg"]
    )
    row["Tempo_Mismatch"] = abs(float(team_a_ord["AdjTempo"]) - float(team_b_ord["AdjTempo"]))
    row["DefOff_Matchup_A"] = float(team_a_ord["AdjDE"]) - float(team_b_ord["AdjOE"])
    row["DefOff_Matchup_B"] = float(team_b_ord["AdjDE"]) - float(team_a_ord["AdjOE"])

    pred_df = pd.DataFrame([row])
    prob_ord = float(predictor.predict_proba(pred_df)[1].iloc[0])
    prob_a = 1.0 - prob_ord if was_swapped else prob_ord

    cache[key] = prob_a
    cache[(team_b, team_a, round_num)] = 1.0 - prob_a
    return prob_a


def _blend(
    team_a, team_b, stage, weights, probs_by_team, team_rows, predictor, numeric_cols, cache
):
    """Two-signal blended win probability: matchup model + conditional ratio.

    Args:
        team_a: First team name.
        team_b: Second team name.
        stage: Game stage string ('R64', 'S16', 'E8', 'F4', 'NCG', 'Winner').
        weights: Tuple of (w_matchup, w_ratio) summing to 1.0.
        probs_by_team: Dict keyed by team name -> pred_df probability Series.
        team_rows: Dict keyed by team name -> feature Series for matchup model.
        predictor: Fitted matchup TabularPredictor, or None.
        numeric_cols: Pre-computed numeric feature column names.
        cache: Matchup probability cache dict.

    Returns:
        Float probability of team_a winning in (0, 1).
    """
    w_matchup, w_ratio = weights
    col = _STAGE_TO_COL[stage]
    ratio = _conditional_ratio(team_a, team_b, col, probs_by_team)

    if predictor is None:
        return ratio

    rnum = _STAGE_TO_ROUND[stage]
    mp = _matchup_prob(team_a, team_b, rnum, team_rows, predictor, numeric_cols, cache)
    return w_matchup * mp + w_ratio * ratio


def _simulate_once(region_teams, probs_by_team, rng, team_rows, predictor, numeric_cols, cache):
    """Simulate one full tournament bracket outcome.

    All games use a two-signal blend of matchup model and conditional ratio,
    with weights drawn from Uniform(0, 1) normalised to sum to 1,
    independently per round per simulation.

    Args:
        region_teams: Dict mapping region -> dict mapping seed -> team name.
        probs_by_team: Dict keyed by team name -> probability Series.
        rng: numpy random Generator instance.
        team_rows: Dict keyed by team name -> feature Series from full_data.
        predictor: Fitted matchup TabularPredictor, or None for ratio-only.
        numeric_cols: Pre-computed numeric feature column names.
        cache: Shared matchup probability cache dict, modified in place.

    Returns:
        Dict representing one simulated tournament outcome with keys for each
        region (R32, S16, E8, F4), NCG, and Winner.
    """
    result = {r: {"F4": None, "E8": {}, "S16": {}, "R32": {}} for r in _REGIONS}
    result["NCG"] = []
    result["Winner"] = None

    # Draw weights from Uniform(0,1) Dirichlet per round
    def _dirichlet_weights():
        w = rng.random(2)
        return w / w.sum()

    w_r64 = _dirichlet_weights()
    w_s16 = _dirichlet_weights()
    w_e8 = _dirichlet_weights()
    w_f4 = _dirichlet_weights()
    w_ncg = _dirichlet_weights()
    w_win = _dirichlet_weights()

    f4_teams = {}

    for region in _REGIONS:
        teams = region_teams[region]

        # R64 -> R32
        r32_winners = {}
        for slot, (s_hi, s_lo) in _R64_PODS.items():
            t_hi = teams[s_hi]
            t_lo = teams[s_lo]
            p_hi = _blend(
                t_hi, t_lo, "R64", w_r64, probs_by_team, team_rows, predictor, numeric_cols, cache
            )
            w = t_hi if rng.random() < p_hi else t_lo
            r32_winners[slot] = w
            result[region]["R32"][slot] = w

        # R32 -> S16
        for slot_a, slot_b in _S16_PODS:
            ta = r32_winners[slot_a]
            tb = r32_winners[slot_b]
            result[region]["S16"][slot_a] = ta
            result[region]["S16"][slot_b] = tb
            p = _blend(
                ta, tb, "S16", w_s16, probs_by_team, team_rows, predictor, numeric_cols, cache
            )
            result[region]["E8"][_S16_TO_E8[slot_a]] = ta if rng.random() < p else tb

        # E8 game
        upper = result[region]["E8"]["Upper"]
        lower = result[region]["E8"]["Lower"]
        p = _blend(
            upper, lower, "E8", w_e8, probs_by_team, team_rows, predictor, numeric_cols, cache
        )
        f4w = upper if rng.random() < p else lower
        result[region]["F4"] = f4w
        f4_teams[region] = f4w

    # F4 games
    ncg_teams = []
    for reg_a, reg_b in _F4_PAIRS:
        ta = f4_teams[reg_a]
        tb = f4_teams[reg_b]
        p = _blend(ta, tb, "F4", w_f4, probs_by_team, team_rows, predictor, numeric_cols, cache)
        ncg_teams.append(ta if rng.random() < p else tb)
    result["NCG"] = ncg_teams

    # NCG game
    ta, tb = ncg_teams
    p = _blend(ta, tb, "NCG", w_ncg, probs_by_team, team_rows, predictor, numeric_cols, cache)
    champ = ta if rng.random() < p else tb

    # Championship game
    loser = tb if champ == ta else ta
    p = _blend(
        champ, loser, "Winner", w_win, probs_by_team, team_rows, predictor, numeric_cols, cache
    )
    result["Winner"] = champ if rng.random() < p else loser

    return result


def _bracket_to_flat(sim, team_to_int):
    """Convert simulation result dict to a flat int16 array of length 63.

    Slot ordering: R32 (32), S16 (16), E8 (8), F4 (4), NCG (2), Winner (1).

    Args:
        sim: Simulation result dict from _simulate_once.
        team_to_int: Dict mapping team name -> integer ID.

    Returns:
        numpy int16 array of length 63.
    """
    slots = []
    for r in _REGIONS:
        for s in _R32_SLOTS:
            slots.append(sim[r]["R32"].get(s))
    for r in _REGIONS:
        for s in _S16_SLOTS:
            slots.append(sim[r]["S16"].get(s))
    for r in _REGIONS:
        for h in _E8_HALVES:
            slots.append(sim[r]["E8"].get(h))
    for r in _REGIONS:
        slots.append(sim[r]["F4"])
    for t in sim["NCG"]:
        slots.append(t)
    slots.append(sim["Winner"])
    return np.array([team_to_int.get(t, 0) if t else 0 for t in slots], dtype=np.int16)


def _select_optimal_bracket(simulations, n_candidates, rng):
    """Return the simulated bracket with highest mean score across all sims.

    Candidates are randomly sampled without replacement and deduplicated
    to ensure genuine bracket diversity before scoring.

    Args:
        simulations: List of simulated outcome dicts from _simulate_once.
        n_candidates: Number of candidate brackets to evaluate.
        rng: numpy random Generator instance (shared with simulation loop).

    Returns:
        The best candidate bracket dict.
    """
    n_sims = len(simulations)
    all_teams = sorted(
        {
            t
            for sim in simulations
            for r in _REGIONS
            for t in [
                *sim[r]["R32"].values(),
                *sim[r]["S16"].values(),
                *sim[r]["E8"].values(),
                sim[r]["F4"],
            ]
            if t
        }
        | {t for sim in simulations for t in [*sim["NCG"], sim["Winner"]] if t}
    )
    team_to_int = {t: i + 1 for i, t in enumerate(all_teams)}

    sim_arrays = np.array([_bracket_to_flat(s, team_to_int) for s in simulations], dtype=np.int16)

    # Random sample without replacement for diversity
    sample_size = min(n_candidates * 4, n_sims)
    sampled_idx = rng.choice(n_sims, size=sample_size, replace=False)
    sampled_arrays = sim_arrays[sampled_idx]

    # Deduplicate
    _, unique_idx = np.unique(sampled_arrays, axis=0, return_index=True)
    unique_arrays = sampled_arrays[unique_idx]
    unique_sims = [simulations[sampled_idx[i]] for i in unique_idx]

    k = min(n_candidates, len(unique_arrays))
    cand_arrays = unique_arrays[:k]
    cand_sims = unique_sims[:k]

    n_dupes = sample_size - len(unique_arrays)
    print(
        f"    Candidates: {len(unique_arrays)} unique from {sample_size} sampled "
        f"({n_dupes} duplicates removed, {k} evaluated)"
    )

    scores = np.zeros((k, n_sims), dtype=np.float32)
    for s in range(63):
        scores += (cand_arrays[:, s : s + 1] == sim_arrays[np.newaxis, :, s]) * _POINTS[s]

    return cand_sims[int(np.argmax(scores.mean(axis=1)))]


def _format_picks(bracket):
    """Convert simulation result dict to standard picks_dict format.

    Args:
        bracket: Simulation result dict from _simulate_once.

    Returns:
        Picks dict compatible with real_Bracket, where each slot value is
        a list (e.g. ['Duke']) or empty list [].
    """
    picks = {}
    for region in _REGIONS:
        picks[region] = {
            "F4": bracket[region]["F4"] or "",
            "E8": {
                h: [bracket[region]["E8"].get(h)] if bracket[region]["E8"].get(h) else []
                for h in _E8_HALVES
            },
            "S16": {
                s: [bracket[region]["S16"].get(s)] if bracket[region]["S16"].get(s) else []
                for s in _S16_SLOTS
            },
            "R32": {
                s: [bracket[region]["R32"].get(s)] if bracket[region]["R32"].get(s) else []
                for s in _R32_SLOTS
            },
        }
    picks["NCG"] = [t for t in bracket["NCG"] if t]
    picks["Winner"] = [bracket["Winner"]] if bracket["Winner"] else []
    return picks


def _build_numeric_cols(full_data):
    """Pre-compute numeric feature columns for matchup prediction.

    Computed once before the simulation loop to avoid repeating _base_features
    on every predict_proba call.

    Args:
        full_data: Full modeling DataFrame for the current year.

    Returns:
        List of numeric feature column names.
    """
    from models.utils.DataProcessing_matchup import _base_features

    base = _base_features(full_data)
    return [
        c
        for c in base.columns
        if c not in ["Year", "Team", "Conf", "Region", "Seed"]
        and pd.api.types.is_numeric_dtype(base[c])
    ]


def simulate_picks(
    pred_df, n_sims=20000, n_candidates=1000, seed=23, predictor=None, full_data=None
):
    """Generate bracket picks using tournament simulation.

    Simulates n_sims full tournaments. For each game, blends the matchup
    model probability with the by-round model conditional ratio using weights
    drawn from Uniform(0, 1) normalised to sum to 1, independently per round
    per simulation.

    The bracket that maximises expected points across all simulations is
    returned, accounting for the joint dependence of picks.

    If predictor is None or full_data is None, falls back to the conditional
    ratio only.

    Args:
        pred_df: DataFrame with columns Team, Seed, Region, R32, S16, E8,
            F4, NCG, Winner as produced by standarize().
        n_sims: Number of tournament simulations. Default 20000.
        n_candidates: Number of candidate brackets to evaluate. Default 1000.
        seed: Random seed for reproducibility. Default 42.
        predictor: Fitted matchup TabularPredictor, or None.
        full_data: Current-year modeling DataFrame for matchup features, or None.

    Returns:
        Picks dict in the standard bracket format used by real_Bracket.
    """
    rng = np.random.default_rng(seed)

    probs_by_team = {row["Team"]: row for _, row in pred_df.iterrows()}
    region_teams = {
        region: {
            int(row["Seed"]): row["Team"]
            for _, row in pred_df[pred_df["Region"] == region].iterrows()
        }
        for region in _REGIONS
    }

    use_matchup = predictor is not None and full_data is not None
    team_rows = {row["Team"]: row for _, row in full_data.iterrows()} if use_matchup else {}
    numeric_cols = _build_numeric_cols(full_data) if use_matchup else []
    cache: dict = {}

    simulations = [
        _simulate_once(
            region_teams,
            probs_by_team,
            rng,
            team_rows,
            predictor if use_matchup else None,
            numeric_cols,
            cache,
        )
        for _ in range(n_sims)
    ]

    print(f"    Matchup cache: {len(cache)} unique predictions cached")

    best_bracket = _select_optimal_bracket(simulations, n_candidates, rng)
    return _format_picks(best_bracket)
