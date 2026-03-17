"""Tournament bracket simulation for joint-probability-aware pick selection.

Uses two separate simulation pools:

1. **Candidate brackets** (N simulations): generated using a two-signal blend
   of the matchup model and by-round model conditional ratio. Weights are drawn
   from Uniform(0, 1) normalised to sum to 1, independently per round per
   simulation. Every simulated bracket is evaluated as a candidate.

2. **Scoring simulations** (M simulations): generated using a two-stage blend
   that incorporates historical seed survival rates. Stage 1 blends the two
   models (as above). Stage 2 blends the Stage 1 result with historical
   _Actual_Full seed survival rates. Both blend weights are drawn independently
   from Uniform(0, 1). This produces a historically-grounded distribution that
   rewards realistic upset frequencies rather than chalk outcomes.

All N candidate brackets are scored against all M scoring simulations. The
candidate with the highest mean score is returned.

Separating the two pools breaks the convergence-to-chalk problem: increasing
M now stabilises the scoring distribution toward historical realism rather
than toward model-biased chalk.

If no matchup predictor or full_data is provided, both pools fall back to
the conditional ratio only.
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

# _Actual_Full column for the round being advanced TO at each game stage
_STAGE_TO_ACTUAL = {
    "R64": "R32_Actual_Full",
    "S16": "S16_Actual_Full",
    "E8": "E8_Actual_Full",
    "F4": "F4_Actual_Full",
    "NCG": "NCG_Actual_Full",
    "Winner": "Winner_Actual_Full",
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


def _actual_ratio(team_a, team_b, stage, actual_by_team):
    """p(A wins | both in game) from historical _Actual_Full seed survival rates.

    Uses the seed-level _Actual_Full column for the round being advanced to.
    All teams of the same seed share the same historical rate. This signal
    anchors scoring simulations to realistic upset frequencies.

    Args:
        team_a: First team name.
        team_b: Second team name.
        stage: Game stage string ('R64', 'S16', 'E8', 'F4', 'NCG', 'Winner').
        actual_by_team: Dict keyed by team name -> full_data feature Series
            containing _Actual_Full columns.

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


def _blend_candidate(
    team_a, team_b, stage, weights, probs_by_team, team_rows, predictor, numeric_cols, cache
):
    """Two-signal blended win probability for candidate bracket generation.

    Blends matchup model and conditional ratio using weights drawn from
    Uniform(0, 1) normalised to sum to 1.

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

    Stage 1: blend matchup model and conditional ratio.
    Stage 2: blend Stage 1 result with historical _Actual_Full seed survival rates.

    Both blend weights are drawn independently from Uniform(0, 1). This
    produces a historically-grounded win probability that rewards realistic
    upset frequencies rather than chalk outcomes.

    Args:
        team_a: First team name.
        team_b: Second team name.
        stage: Game stage string ('R64', 'S16', 'E8', 'F4', 'NCG', 'Winner').
        w_model: Weight for the model blend in Stage 2 (1 - w_actual).
        w_actual: Weight for the historical rate in Stage 2.
        probs_by_team: Dict keyed by team name -> pred_df probability Series.
        actual_by_team: Dict keyed by team name -> full_data feature Series.
        team_rows: Dict keyed by team name -> feature Series for matchup model.
        predictor: Fitted matchup TabularPredictor, or None.
        numeric_cols: Pre-computed numeric feature column names.
        cache: Matchup probability cache dict (shared with candidate pool).

    Returns:
        Float probability of team_a winning in (0, 1).
    """
    # Stage 1: blend the two models with their own random weights
    col = _STAGE_TO_COL[stage]
    ratio = _conditional_ratio(team_a, team_b, col, probs_by_team)

    if predictor is not None:
        rnum = _STAGE_TO_ROUND[stage]
        mp = _matchup_prob(team_a, team_b, rnum, team_rows, predictor, numeric_cols, cache)
        # Inner weights for model blend drawn earlier and stored in w_model/w_actual context
        # Use ratio of ratio vs mp drawn fresh — reuse w_model as inner alpha
        model_prob = w_model * mp + (1.0 - w_model) * ratio
    else:
        model_prob = ratio

    # Stage 2: blend model probability with historical rate
    # w_actual is the weight on the historical rate (beta ~ Uniform(0,1))
    # so historical gets equal expected weight to model across simulations
    if actual_by_team:
        hist = _actual_ratio(team_a, team_b, stage, actual_by_team)
        return (1.0 - w_actual) * model_prob + w_actual * hist
    return model_prob


def _simulate_once_candidate(
    region_teams, probs_by_team, rng, team_rows, predictor, numeric_cols, cache
):
    """Simulate one candidate bracket using the two-signal model blend.

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

    Stage 1 blends the two models with random weights. Stage 2 blends the
    Stage 1 result with historical _Actual_Full seed survival rates using
    independent random weights. This produces historically-grounded outcomes
    that reflect realistic upset frequencies.

    Args:
        region_teams: Dict mapping region -> dict mapping seed -> team name.
        probs_by_team: Dict keyed by team name -> probability Series.
        actual_by_team: Dict keyed by team name -> full_data feature Series
            containing _Actual_Full columns. May be empty dict if unavailable.
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

    # Per-round weights: w_model is inner model blend alpha, w_actual is Stage 2 beta
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
    """Convert simulation result dict to a flat int16 array of length 63.

    Slot ordering: R32 (32), S16 (16), E8 (8), F4 (4), NCG (2), Winner (1).

    Args:
        sim: Simulation result dict from _simulate_once_candidate or
            _simulate_once_scoring.
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


def _select_optimal_bracket(candidates, scoring_sims):
    """Score all candidate brackets against all scoring simulations.

    All N candidate brackets are evaluated against all M scoring simulations.
    The candidate with the highest mean score is returned. Because candidates
    and scoring simulations come from separate distributions, increasing M
    stabilises the scoring distribution toward historical realism rather than
    toward model-biased chalk.

    Args:
        candidates: List of candidate bracket dicts from _simulate_once_candidate.
        scoring_sims: List of scoring simulation dicts from _simulate_once_scoring.

    Returns:
        The best candidate bracket dict.
    """
    n_cands = len(candidates)
    n_scores = len(scoring_sims)

    # Build a unified team->int mapping across both pools
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

    print(
        f"    Scoring {n_cands} candidates against {n_scores} scoring sims "
        f"({n_cands * n_scores:,} comparisons)"
    )

    scores = np.zeros((n_cands, n_scores), dtype=np.float32)
    for s in range(63):
        scores += (cand_arrays[:, s : s + 1] == score_arrays[np.newaxis, :, s]) * _POINTS[s]

    return candidates[int(np.argmax(scores.mean(axis=1)))]


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
    pred_df,
    n_sims=5000,
    n_scoring_sims=20000,
    seed=23,
    predictor=None,
    full_data=None,
):
    """Generate bracket picks using decoupled candidate and scoring simulations.

    Generates N candidate brackets using the two-signal model blend, then
    generates M scoring brackets using a two-stage blend that incorporates
    historical _Actual_Full seed survival rates. All N candidates are scored
    against all M scoring simulations. The candidate with the highest mean
    score is returned.

    Increasing n_scoring_sims stabilises the scoring distribution toward
    historical realism rather than toward model-biased chalk, solving the
    convergence-to-chalk problem that occurs when candidates and scoring
    simulations share the same pool.

    If predictor is None or full_data is None, both pools fall back to the
    conditional ratio only, and scoring simulations use no historical blend.

    Args:
        pred_df: DataFrame with columns Team, Seed, Region, R32, S16, E8,
            F4, NCG, Winner as produced by standarize().
        n_sims: Number of candidate bracket simulations. Default 5000.
        n_scoring_sims: Number of scoring simulations. Default 20000.
        seed: Random seed for reproducibility. Default 23.
        predictor: Fitted matchup TabularPredictor, or None.
        full_data: Current-year modeling DataFrame for matchup features and
            _Actual_Full historical survival rates, or None.

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

    # actual_by_team provides _Actual_Full columns for scoring simulations
    actual_by_team = (
        {row["Team"]: row for _, row in full_data.iterrows()} if full_data is not None else {}
    )

    # Shared matchup cache across both simulation pools
    cache: dict = {}

    # --- Candidate brackets (two-signal model blend) ---
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

    # --- Scoring simulations (two-stage historical blend) ---
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

    print(f"    Matchup cache: {len(cache)} unique predictions cached")

    best_bracket = _select_optimal_bracket(candidates, scoring_sims)
    return _format_picks(best_bracket)
