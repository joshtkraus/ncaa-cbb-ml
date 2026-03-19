"""Tournament bracket simulation for probability-aware pick selection."""

import numpy as np
import pandas as pd
from models.utils.DataProcessing_matchup import _assign_team_a, _base_features

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

# Column advanced to
_STAGE_TO_COL = {
    "R64": "R32",
    "S16": "E8",
    "E8": "F4",
    "F4": "NCG",
    "NCG": "Winner",
    "Winner": "Winner",
}

# Historical column for the round being advanced to
_STAGE_TO_ACTUAL = {
    "R64": "R32_Actual_Full",
    "S16": "S16_Actual_Full",
    "E8": "E8_Actual_Full",
    "F4": "F4_Actual_Full",
    "NCG": "NCG_Actual_Full",
    "Winner": "Winner_Actual_Full",
}

# Round number for matchup model
_STAGE_TO_ROUND = {"R64": 2, "S16": 3, "E8": 4, "F4": 5, "NCG": 6, "Winner": 7}

# Flat array slot ordering
_POINTS = np.array(
    [10] * 32 + [20] * 16 + [40] * 8 + [80] * 4 + [160] * 2 + [320] * 1, dtype=np.float32
)


def _get_prob(probs_by_team, team, col):
    """Safely retrieve a probability value for a team.

    Args:
        probs_by_team: Dict keyed by team name: probability Series.
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
        probs_by_team: Dict keyed by team name: probability Series.

    Returns:
        Float probability of team_a winning, in (0, 1).
    """
    pa = _get_prob(probs_by_team, team_a, col)
    pb = _get_prob(probs_by_team, team_b, col)
    total = pa + pb
    return pa / total if total > 0 else 0.5


def _actual_ratio(team_a, team_b, stage, actual_by_team):
    """p(A wins | both in game) from historical _Actual_Full seed survival rates.

    Args:
        team_a: First team name.
        team_b: Second team name.
        stage: Game stage string ('R64', 'S16', 'E8', 'F4', 'NCG', 'Winner').
        actual_by_team: Dict keyed by team name: Series with _Actual_Full.

    Returns:
        Float probability of team_a winning, in (0, 1).
    """
    col = _STAGE_TO_ACTUAL[stage]
    row_a = actual_by_team.get(team_a)
    row_b = actual_by_team.get(team_b)
    pa = float(row_a[col]) if row_a is not None else 0.0
    pb = float(row_b[col]) if row_b is not None else 0.0
    total = pa + pb
    return pa / total if total > 0 else 0.5


def _matchup_prob(team_a, team_b, round_num, team_rows, predictor, numeric_cols, cache):
    """Get matchup model win probability for team_a, using lazy cache.

    Args:
        team_a: First team name.
        team_b: Second team name.
        round_num: Tournament round number (2-7).
        team_rows: Dict keyed by team name: feature Series from full_data.
        predictor: Fitted matchup TabularPredictor.
        numeric_cols: Pre-computed list of numeric feature column names.
        cache: Dict keyed by (team_a, team_b, round_num): float probability.

    Returns:
        Float probability of team_a winning in [0, 1].
    """
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


def _blend_candidate(
    team_a, team_b, stage, weights, probs_by_team, team_rows, predictor, numeric_cols, cache
):
    """Two-signal blended win probability for candidate bracket generation.

    Args:
        team_a: First team name.
        team_b: Second team name.
        stage: Game stage string ('R64', 'S16', 'E8', 'F4', 'NCG', 'Winner').
        weights: Tuple of (w_matchup, w_ratio) summing to 1.0.
        probs_by_team: Dict keyed by team name: pred_df probability Series.
        team_rows: Dict keyed by team name: feature Series for matchup model.
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


def _blend_scoring(
    team_a,
    team_b,
    stage,
    w_model,
    w_actual,
    probs_by_team,
    actual_by_team,
    team_rows,
    predictor,
    numeric_cols,
    cache,
):
    """Two-stage blended win probability for scoring simulation generation.

    Args:
        team_a: First team name.
        team_b: Second team name.
        stage: Game stage string ('R64', 'S16', 'E8', 'F4', 'NCG', 'Winner').
        w_model: Weight for the model blend in Stage 2 (1 - w_actual).
        w_actual: Weight for the historical rate in Stage 2.
        probs_by_team: Dict keyed by team name: pred_df probability Series.
        actual_by_team: Dict keyed by team name: full_data feature Series.
        team_rows: Dict keyed by team name: feature Series for matchup model.
        predictor: Fitted matchup TabularPredictor, or None.
        numeric_cols: Pre-computed numeric feature column names.
        cache: Matchup probability cache dict (shared with candidate pool).

    Returns:
        Float probability of team_a winning in (0, 1).
    """
    # Stage 1: blend the two models on the log-odds scale
    col = _STAGE_TO_COL[stage]
    ratio = _conditional_ratio(team_a, team_b, col, probs_by_team)

    if predictor is not None:
        rnum = _STAGE_TO_ROUND[stage]
        mp = _matchup_prob(team_a, team_b, rnum, team_rows, predictor, numeric_cols, cache)
        model_prob = w_model * mp + (1.0 - w_model) * ratio
    else:
        model_prob = ratio

    # Stage 2: blend model probability with historical rate
    if actual_by_team:
        hist = _actual_ratio(team_a, team_b, stage, actual_by_team)
        return (1.0 - w_actual) * model_prob + w_actual * hist
    return model_prob


def _simulate_once_candidate(
    region_teams, probs_by_team, rng, team_rows, predictor, numeric_cols, cache
):
    """Simulate one candidate bracket using the two-signal model blend.

    Args:
        region_teams: Dict mapping region: dict mapping seed: team name.
        probs_by_team: Dict keyed by team name: probability Series.
        rng: numpy random Generator instance.
        team_rows: Dict keyed by team name: feature Series from full_data.
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

    def _dw():
        w = rng.random(2)
        return w / w.sum()

    w_r64 = _dw()
    w_s16 = _dw()
    w_e8 = _dw()
    w_f4 = _dw()
    w_ncg = _dw()
    w_win = _dw()

    f4_teams = {}
    for region in _REGIONS:
        teams = region_teams[region]

        r32_winners = {}
        for slot, (s_hi, s_lo) in _R64_PODS.items():
            t_hi = teams[s_hi]
            t_lo = teams[s_lo]
            p = _blend_candidate(
                t_hi, t_lo, "R64", w_r64, probs_by_team, team_rows, predictor, numeric_cols, cache
            )
            w = t_hi if rng.random() < p else t_lo
            r32_winners[slot] = w
            result[region]["R32"][slot] = w

        for slot_a, slot_b in _S16_PODS:
            ta = r32_winners[slot_a]
            tb = r32_winners[slot_b]
            result[region]["S16"][slot_a] = ta
            result[region]["S16"][slot_b] = tb
            p = _blend_candidate(
                ta, tb, "S16", w_s16, probs_by_team, team_rows, predictor, numeric_cols, cache
            )
            result[region]["E8"][_S16_TO_E8[slot_a]] = ta if rng.random() < p else tb

        upper = result[region]["E8"]["Upper"]
        lower = result[region]["E8"]["Lower"]
        p = _blend_candidate(
            upper, lower, "E8", w_e8, probs_by_team, team_rows, predictor, numeric_cols, cache
        )
        f4w = upper if rng.random() < p else lower
        result[region]["F4"] = f4w
        f4_teams[region] = f4w

    ncg_teams = []
    for reg_a, reg_b in _F4_PAIRS:
        ta = f4_teams[reg_a]
        tb = f4_teams[reg_b]
        p = _blend_candidate(
            ta, tb, "F4", w_f4, probs_by_team, team_rows, predictor, numeric_cols, cache
        )
        ncg_teams.append(ta if rng.random() < p else tb)
    result["NCG"] = ncg_teams

    ta, tb = ncg_teams
    p = _blend_candidate(
        ta, tb, "NCG", w_ncg, probs_by_team, team_rows, predictor, numeric_cols, cache
    )
    champ = ta if rng.random() < p else tb

    loser = tb if champ == ta else ta
    p = _blend_candidate(
        champ, loser, "Winner", w_win, probs_by_team, team_rows, predictor, numeric_cols, cache
    )
    result["Winner"] = champ if rng.random() < p else loser
    return result


def _simulate_once_scoring(
    region_teams, probs_by_team, actual_by_team, rng, team_rows, predictor, numeric_cols, cache
):
    """Simulate one scoring bracket using the two-stage historical blend.

    Args:
        region_teams: Dict mapping region: dict mapping seed: team name.
        probs_by_team: Dict keyed by team name: probability Series.
        actual_by_team: Dict keyed by team name: Series with _Actual_Full.
        rng: numpy random Generator instance.
        team_rows: Dict keyed by team name: feature Series from full_data.
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

    # Per-round weights
    def _stage_weights():
        w_model = rng.random()  # inner blend weight for matchup vs ratio
        w_actual = rng.random()  # Stage 2 weight: model_prob vs historical_rate
        return w_model, w_actual

    wr64 = _stage_weights()
    ws16 = _stage_weights()
    we8 = _stage_weights()
    wf4 = _stage_weights()
    wncg = _stage_weights()
    wwin = _stage_weights()

    f4_teams = {}
    for region in _REGIONS:
        teams = region_teams[region]

        r32_winners = {}
        for slot, (s_hi, s_lo) in _R64_PODS.items():
            t_hi = teams[s_hi]
            t_lo = teams[s_lo]
            p = _blend_scoring(
                t_hi,
                t_lo,
                "R64",
                *wr64,
                probs_by_team,
                actual_by_team,
                team_rows,
                predictor,
                numeric_cols,
                cache,
            )
            w = t_hi if rng.random() < p else t_lo
            r32_winners[slot] = w
            result[region]["R32"][slot] = w

        for slot_a, slot_b in _S16_PODS:
            ta = r32_winners[slot_a]
            tb = r32_winners[slot_b]
            result[region]["S16"][slot_a] = ta
            result[region]["S16"][slot_b] = tb
            p = _blend_scoring(
                ta,
                tb,
                "S16",
                *ws16,
                probs_by_team,
                actual_by_team,
                team_rows,
                predictor,
                numeric_cols,
                cache,
            )
            result[region]["E8"][_S16_TO_E8[slot_a]] = ta if rng.random() < p else tb

        upper = result[region]["E8"]["Upper"]
        lower = result[region]["E8"]["Lower"]
        p = _blend_scoring(
            upper,
            lower,
            "E8",
            *we8,
            probs_by_team,
            actual_by_team,
            team_rows,
            predictor,
            numeric_cols,
            cache,
        )
        f4w = upper if rng.random() < p else lower
        result[region]["F4"] = f4w
        f4_teams[region] = f4w

    ncg_teams = []
    for reg_a, reg_b in _F4_PAIRS:
        ta = f4_teams[reg_a]
        tb = f4_teams[reg_b]
        p = _blend_scoring(
            ta,
            tb,
            "F4",
            *wf4,
            probs_by_team,
            actual_by_team,
            team_rows,
            predictor,
            numeric_cols,
            cache,
        )
        ncg_teams.append(ta if rng.random() < p else tb)
    result["NCG"] = ncg_teams

    ta, tb = ncg_teams
    p = _blend_scoring(
        ta,
        tb,
        "NCG",
        *wncg,
        probs_by_team,
        actual_by_team,
        team_rows,
        predictor,
        numeric_cols,
        cache,
    )
    champ = ta if rng.random() < p else tb

    loser = tb if champ == ta else ta
    p = _blend_scoring(
        champ,
        loser,
        "Winner",
        *wwin,
        probs_by_team,
        actual_by_team,
        team_rows,
        predictor,
        numeric_cols,
        cache,
    )
    result["Winner"] = champ if rng.random() < p else loser
    return result


def _bracket_to_flat(sim, team_to_int):
    """Convert simulation result dict to a flat array of length 63.

    Args:
        sim: Simulation result dict
        team_to_int: Dict mapping team name to integer ID.

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


def _build_rate_tables(actual_by_team, team_to_seed):
    """Pre-compute seed to rate lookup tables for fast log-likelihood computation.

    Args:
        actual_by_team: Dict keyed by team name: full_data feature Series.
        team_to_seed: Dict mapping team name: seed number.

    Returns:
        Dict keyed by round name: dict keyed by seed: float rate.
        Returns empty dict if actual_by_team is empty.
    """
    if not actual_by_team:
        return {}

    _ROUND_ACTUAL_COL = {
        "R32": "R32_Actual_Full",
        "S16": "S16_Actual_Full",
        "E8": "E8_Actual_Full",
        "F4": "F4_Actual_Full",
        "NCG": "NCG_Actual_Full",
        "Winner": "Winner_Actual_Full",
    }
    tables: dict = {rd: {} for rd in _ROUND_ACTUAL_COL}
    for team, row in actual_by_team.items():
        seed = team_to_seed.get(team)
        if seed is None:
            continue
        for rd, col in _ROUND_ACTUAL_COL.items():
            if seed not in tables[rd]:
                try:
                    p = float(row[col])
                    tables[rd][seed] = max(1e-6, min(1.0 - 1e-6, p))
                except (KeyError, TypeError):
                    tables[rd][seed] = 0.5
    return tables


def _clamp(p):
    """Clamp probability to (1e-6, 1-1e-6) to avoid log(0).

    Args:
        p: Float probability.

    Returns:
        Float probability clamped away from 0 and 1.
    """
    return max(1e-6, min(1.0 - 1e-6, p))


def _ll_r32(candidate, team_to_seed, rate_tables):
    """Log-likelihood contribution from R32: did the favored seed (1-8) win?

    Args:
        candidate: Candidate bracket dict.
        team_to_seed: Dict mapping team name: seed number.
        rate_tables: Dict keyed by round: seed: rate.

    Returns:
        Float log-likelihood contribution from R32.
    """
    r32_rates = rate_tables.get("R32", {})
    log_lik = 0.0
    for region in _REGIONS:
        for slot, (s_hi, s_lo) in _R64_PODS.items():
            winner = candidate[region]["R32"].get(slot)
            if winner is None:
                continue
            winner_seed = team_to_seed.get(winner, s_lo)
            p = _clamp(r32_rates.get(s_hi, 0.5))
            log_lik += np.log(p) if winner_seed == s_hi else np.log(1.0 - p)
    return log_lik


def _ll_s16(candidate, team_to_seed, rate_tables):
    """Log-likelihood contribution from S16: did the pod's top seed (1-4) win?

    Args:
        candidate: Candidate bracket dict.
        team_to_seed: Dict mapping team name: seed number.
        rate_tables: Dict keyed by round: seed: rate.

    Returns:
        Float log-likelihood contribution from S16.
    """
    s16_rates = rate_tables.get("S16", {})
    _S16_FAVORED = {"1": 1, "4": 4, "3": 3, "2": 2}
    log_lik = 0.0
    for region in _REGIONS:
        for slot_a, _ in _S16_PODS:
            e8_winner = candidate[region]["E8"].get(_S16_TO_E8[slot_a])
            if e8_winner is None:
                continue
            winner_seed = team_to_seed.get(e8_winner, 99)
            fav_seed = _S16_FAVORED[slot_a]
            p = _clamp(s16_rates.get(fav_seed, 0.5))
            log_lik += np.log(p) if winner_seed == fav_seed else np.log(1.0 - p)
    return log_lik


def _ll_e8(candidate, team_to_seed, rate_tables):
    """Log-likelihood contribution from E8: did the upper-half winner advance?

    Args:
        candidate: Candidate bracket dict.
        team_to_seed: Dict mapping team name: seed number.
        rate_tables: Dict keyed by round: seed: rate.

    Returns:
        Float log-likelihood contribution from E8.
    """
    e8_rates = rate_tables.get("E8", {})
    log_lik = 0.0
    for region in _REGIONS:
        f4_team = candidate[region]["F4"]
        upper = candidate[region]["E8"].get("Upper")
        if f4_team is None or upper is None:
            continue
        upper_seed = team_to_seed.get(upper, 99)
        p = _clamp(e8_rates.get(upper_seed, 0.5))
        log_lik += np.log(p) if f4_team == upper else np.log(1.0 - p)
    return log_lik


def _ll_late(candidate, team_to_seed, rate_tables):
    """Log-likelihood contribution from F4, NCG, and Winner.

    Args:
        candidate: Candidate bracket dict.
        team_to_seed: Dict mapping team name: seed number.
        rate_tables: Dict keyed by round: seed: rate.

    Returns:
        Float log-likelihood contribution from F4, NCG, and Winner.
    """
    f4_rates = rate_tables.get("F4", {})
    ncg_rates = rate_tables.get("NCG", {})
    winner_rates = rate_tables.get("Winner", {})
    log_lik = 0.0

    for region in _REGIONS:
        f4_team = candidate[region]["F4"]
        if f4_team is None:
            continue
        p = _clamp(f4_rates.get(team_to_seed.get(f4_team, 99), 0.5))
        log_lik += np.log(p)

    for t in candidate["NCG"]:
        p = _clamp(ncg_rates.get(team_to_seed.get(t, 99), 0.5))
        log_lik += np.log(p)

    winner = candidate["Winner"]
    if winner:
        p = _clamp(winner_rates.get(team_to_seed.get(winner, 99), 0.5))
        log_lik += np.log(p)

    return log_lik


def _bracket_log_likelihood(candidate, team_to_seed, rate_tables):
    """Compute log-likelihood of a bracket using favored-seed binary outcomes.

    Args:
        candidate: Candidate bracket dict from _simulate_once_candidate.
        team_to_seed: Dict mapping team name: seed number.
        rate_tables: Dict keyed by round: seed: rate, from _build_rate_tables.

    Returns:
        Float log-likelihood. Higher (less negative) = more historically likely.
    """
    if not rate_tables:
        return 0.0
    return (
        _ll_r32(candidate, team_to_seed, rate_tables)
        + _ll_s16(candidate, team_to_seed, rate_tables)
        + _ll_e8(candidate, team_to_seed, rate_tables)
        + _ll_late(candidate, team_to_seed, rate_tables)
    )


def _select_optimal_bracket(candidates, scoring_sims, rate_tables=None, team_to_seed=None, n_top=1):
    """Score all candidate brackets against all scoring simulations.

    Args:
        candidates: List of candidate bracket dicts from _simulate_once_candidate.
        scoring_sims: List of scoring simulation dicts from _simulate_once_scoring.
        rate_tables: Pre-computed seed to rate lookup tables from _build_rate_tables.
        team_to_seed: Dict mapping team name to seed number.
        n_top: Number of top-ranked brackets to return. Default 1.

    Returns:
        List of the top n_top candidate bracket dicts, ranked by combined score
        descending. Length is min(n_top, len(candidates)).
    """
    n_cands = len(candidates)
    n_scores = len(scoring_sims)

    all_teams = sorted(
        {
            t
            for pool in [candidates, scoring_sims]
            for sim in pool
            for r in _REGIONS
            for t in [
                *sim[r]["R32"].values(),
                *sim[r]["S16"].values(),
                *sim[r]["E8"].values(),
                sim[r]["F4"],
            ]
            if t
        }
        | {
            t
            for pool in [candidates, scoring_sims]
            for sim in pool
            for t in [*sim["NCG"], sim["Winner"]]
            if t
        }
    )
    team_to_int = {t: i + 1 for i, t in enumerate(all_teams)}

    cand_arrays = np.array([_bracket_to_flat(s, team_to_int) for s in candidates], dtype=np.int16)
    score_arrays = np.array(
        [_bracket_to_flat(s, team_to_int) for s in scoring_sims], dtype=np.int16
    )

    print(f"    Scoring {n_cands} candidates against {n_scores} scoring sims")

    scores = np.zeros((n_cands, n_scores), dtype=np.float32)
    for s in range(63):
        scores += (cand_arrays[:, s : s + 1] == score_arrays[np.newaxis, :, s]) * _POINTS[s]

    mean_scores = scores.mean(axis=1)

    # Combine mean score with log-likelihood using z-score normalization
    if rate_tables and team_to_seed:
        log_liks = np.array(
            [_bracket_log_likelihood(c, team_to_seed, rate_tables) for c in candidates],
            dtype=np.float64,
        )

        def _zscore(arr):
            std = arr.std()
            return (arr - arr.mean()) / std if std > 0 else arr - arr.mean()

        z_scores = _zscore(mean_scores.astype(np.float64))
        z_liks = _zscore(log_liks)
        combined = 0.5 * z_scores + 0.5 * z_liks
    else:
        combined = mean_scores

    ranked_idx = np.argsort(combined)[::-1]
    return [candidates[i] for i in ranked_idx[:n_top]]


def _format_picks(bracket):
    """Convert simulation result dict to standard picks_dict format.

    Args:
        bracket: Simulation result dict from _simulate_once_candidate.

    Returns:
        Picks dict compatible with real_Bracket, where each slot value is
        a list (e.g. ['Duke']) or empty list [].
    """
    picks = {}
    for region in _REGIONS:
        picks[region] = {
            "R32": {
                s: [bracket[region]["R32"].get(s)] if bracket[region]["R32"].get(s) else []
                for s in _R32_SLOTS
            },
            "S16": {
                s: [bracket[region]["S16"].get(s)] if bracket[region]["S16"].get(s) else []
                for s in _S16_SLOTS
            },
            "E8": {
                h: [bracket[region]["E8"].get(h)] if bracket[region]["E8"].get(h) else []
                for h in _E8_HALVES
            },
            "F4": bracket[region]["F4"] or "",
        }
    picks["NCG"] = [t for t in bracket["NCG"] if t]
    picks["Winner"] = [bracket["Winner"]] if bracket["Winner"] else []
    return picks


def _build_numeric_cols(full_data):
    """Pre-compute numeric feature columns for matchup prediction.

    Args:
        full_data: Full modeling DataFrame for the current year.

    Returns:
        List of numeric feature column names.
    """
    base = _base_features(full_data)
    return [
        c
        for c in base.columns
        if c not in ["Year", "Team", "Conf", "Region", "Seed"]
        and pd.api.types.is_numeric_dtype(base[c])
    ]


def simulate_picks(
    pred_df,
    n_sims=5000,
    n_scoring_sims=20000,
    seed=23,
    n_top=1,
    predictor=None,
    full_data=None,
):
    """Generate bracket picks using candidate and scoring simulations.

    Args:
        pred_df: DataFrame as produced by standarize().
        n_sims: Number of candidate bracket simulations. Default 5000.
        n_scoring_sims: Number of scoring simulations. Default 20000.
        seed: Random seed for reproducibility. Default 23.
        n_top: Number of top-ranked brackets to return. Default 1.
        predictor: Fitted matchup TabularPredictor, or None.
        full_data: Current-year modeling DataFrame for matchup features.

    Returns:
        If n_top == 1: single picks dict in the standard bracket format.
        If n_top > 1:  list of picks dicts ranked by combined score descending.
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

    # _Actual_Full columns for scoring simulations
    actual_by_team = (
        {row["Team"]: row for _, row in full_data.iterrows()} if full_data is not None else {}
    )

    # Shared matchup cache across both simulation pools
    cache: dict = {}

    # --- Candidate brackets ---
    print(f"    Generating {n_sims} candidate brackets...")
    candidates = [
        _simulate_once_candidate(
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

    # --- Scoring simulations ---
    print(f"    Generating {n_scoring_sims} scoring simulations...")
    scoring_sims = [
        _simulate_once_scoring(
            region_teams,
            probs_by_team,
            actual_by_team,
            rng,
            team_rows,
            predictor if use_matchup else None,
            numeric_cols,
            cache,
        )
        for _ in range(n_scoring_sims)
    ]

    # Build team_to_seed and pre-compute rate tables for log-likelihood
    team_to_seed = {row["Team"]: int(row["Seed"]) for _, row in pred_df.iterrows()}
    rate_tables = _build_rate_tables(actual_by_team, team_to_seed)

    top_brackets = _select_optimal_bracket(
        candidates,
        scoring_sims,
        rate_tables=rate_tables,
        team_to_seed=team_to_seed,
        n_top=n_top,
    )
    formatted = [_format_picks(b) for b in top_brackets]
    return formatted[0] if n_top == 1 else formatted
