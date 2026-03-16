"""Tournament bracket simulation for joint-probability-aware pick selection.

Replaces independent expected-points maximization with a simulation-based
approach that accounts for dependence between picks. N full tournaments are
simulated using model probabilities. A subset of K simulated brackets are
evaluated as candidates by scoring each against all N simulations. The
candidate with the highest mean score is selected as the final bracket.

This ensures the selected bracket is jointly optimal — a bracket with four
1-seeds in the F4 only scores well in the ~1.5% of simulations where that
actually happens, so its mean score is low. A bracket with realistic upsets
scores across more simulations and gets selected instead.
"""

import numpy as np

# R64 seed matchups: slot -> (favored_seed, underdog_seed)
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

# S16 pod pairs: which R32 slots play each other in the S16 game
_S16_PODS = [("1", "4"), ("3", "2")]

# S16 slot -> E8 half
_S16_TO_E8 = {"1": "Upper", "4": "Upper", "3": "Lower", "2": "Lower"}

# F4 region pairings
_F4_PAIRS = [("West", "East"), ("South", "Midwest")]

_REGIONS = ["West", "East", "South", "Midwest"]
_R32_SLOTS = ["1", "8", "5", "4", "6", "3", "7", "2"]
_S16_SLOTS = ["1", "4", "3", "2"]
_E8_HALVES = ["Upper", "Lower"]

# Slot ordering for flat array representation (63 total)
# R32: 32 slots, S16: 16, E8: 8, F4: 4, NCG: 2, Winner: 1
_POINTS = np.array(
    [10] * 32 + [20] * 16 + [40] * 8 + [80] * 4 + [160] * 2 + [320] * 1,
    dtype=np.float32,
)


def _get_prob(probs_by_team, team, col):
    """Safely retrieve a probability value for a team.

    Args:
        probs_by_team: Dict keyed by team name mapping to probability Series.
        team: Team name string.
        col: Column name (e.g. 'R32', 'S16').

    Returns:
        Float probability, defaulting to 0.0 if not found.
    """
    row = probs_by_team.get(team)
    return float(row[col]) if row is not None else 0.0


def _conditional_win_prob(team_a, team_b, col, probs_by_team):
    """Compute conditional probability that team_a beats team_b in a given round.

    Uses the ratio of marginal advancement probabilities as a proxy for
    head-to-head win probability given both teams have reached this game.

    Args:
        team_a: First team name.
        team_b: Second team name.
        col: Round probability column (e.g. 'S16', 'E8').
        probs_by_team: Dict keyed by team name mapping to probability Series.

    Returns:
        Float probability of team_a winning, in (0, 1).
    """
    pa = _get_prob(probs_by_team, team_a, col)
    pb = _get_prob(probs_by_team, team_b, col)
    total = pa + pb
    if total == 0:
        return 0.5
    return pa / total


def _simulate_once(region_teams, probs_by_team, rng):
    """Simulate one full tournament bracket outcome.

    Args:
        region_teams: Dict mapping region -> dict mapping seed -> team name.
        probs_by_team: Dict keyed by team name -> probability Series.
        rng: numpy random Generator instance.

    Returns:
        Dict representing one simulated tournament outcome with the same
        structure as a picks_dict (R32, S16, E8, F4, NCG, Winner).
    """
    result = {r: {"F4": None, "E8": {}, "S16": {}, "R32": {}} for r in _REGIONS}
    result["NCG"] = []
    result["Winner"] = None

    f4_teams = {}

    for region in _REGIONS:
        teams = region_teams[region]

        # R64 -> R32: use R32 prob directly (normalized per pod)
        r32_winners = {}
        for slot, (s_hi, s_lo) in _R64_PODS.items():
            team_hi = teams[s_hi]
            team_lo = teams[s_lo]
            p_hi = _get_prob(probs_by_team, team_hi, "R32")
            winner = team_hi if rng.random() < p_hi else team_lo
            r32_winners[slot] = winner
            result[region]["R32"][slot] = winner

        # R32 -> S16: both pod teams recorded in S16, winner advances to E8
        for slot_a, slot_b in _S16_PODS:
            team_a = r32_winners[slot_a]
            team_b = r32_winners[slot_b]
            result[region]["S16"][slot_a] = team_a
            result[region]["S16"][slot_b] = team_b
            p_a = _conditional_win_prob(team_a, team_b, "E8", probs_by_team)
            e8_winner = team_a if rng.random() < p_a else team_b
            e8_half = _S16_TO_E8[slot_a]
            result[region]["E8"][e8_half] = e8_winner

        # S16 -> E8: Upper vs Lower half winners compete to advance to F4
        upper = result[region]["E8"]["Upper"]
        lower = result[region]["E8"]["Lower"]
        p_upper = _conditional_win_prob(upper, lower, "F4", probs_by_team)
        f4_winner = upper if rng.random() < p_upper else lower
        result[region]["F4"] = f4_winner
        f4_teams[region] = f4_winner

    # E8 -> F4: two semifinal games, winners advance to NCG
    ncg_teams = []
    for reg_a, reg_b in _F4_PAIRS:
        team_a = f4_teams[reg_a]
        team_b = f4_teams[reg_b]
        p_a = _conditional_win_prob(team_a, team_b, "NCG", probs_by_team)
        winner = team_a if rng.random() < p_a else team_b
        ncg_teams.append(winner)
    result["NCG"] = ncg_teams

    # F4 -> Champion: NCG teams compete to become Winner
    team_a, team_b = ncg_teams
    p_a = _conditional_win_prob(team_a, team_b, "Winner", probs_by_team)
    result["Winner"] = team_a if rng.random() < p_a else team_b

    return result


def _bracket_to_flat(sim, team_to_int):
    """Convert a simulation result dict to a flat integer array.

    Slot ordering: R32 (32), S16 (16), E8 (8), F4 (4), NCG (2), Winner (1).

    Args:
        sim: Simulation result dict.
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


def _select_optimal_bracket(simulations, n_candidates):
    """Select the simulated bracket that maximises expected points across all sims.

    Evaluates n_candidates brackets (drawn from the simulation pool) by scoring
    each against all simulations. The candidate with the highest mean score is
    returned. Scoring is fully vectorised for speed.

    Args:
        simulations: List of simulated outcome dicts from _simulate_once.
        n_candidates: Number of simulated brackets to evaluate as candidates.

    Returns:
        The best candidate bracket dict.
    """
    n_sims = len(simulations)

    # Build team->int mapping from all simulations
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

    # Convert all simulations to flat integer arrays: (n_sims, 63)
    sim_arrays = np.array(
        [_bracket_to_flat(sim, team_to_int) for sim in simulations],
        dtype=np.int16,
    )

    # Candidate brackets are the first n_candidates simulations
    cand_arrays = sim_arrays[:n_candidates]  # (K, 63)

    # Vectorised scoring: for each slot, check if candidate matches each sim
    # scores[k, n] = total points candidate k scores against sim n
    scores = np.zeros((n_candidates, n_sims), dtype=np.float32)
    for s in range(63):
        scores += (cand_arrays[:, s : s + 1] == sim_arrays[np.newaxis, :, s]) * _POINTS[s]

    mean_scores = scores.mean(axis=1)
    best_idx = int(np.argmax(mean_scores))
    return simulations[best_idx]


def _format_picks(bracket):
    """Convert a simulation result dict to the standard picks_dict format.

    Args:
        bracket: Simulation result dict from _simulate_once.

    Returns:
        Picks dict compatible with correct_bracket and real_Bracket, where
        each slot value is a list (e.g. ['Duke']) or empty list [].
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


def simulate_picks(pred_df, n_sims=20000, n_candidates=1000, seed=23):
    """Generate bracket picks using tournament simulation.

    Simulates n_sims full tournaments using model probabilities, then selects
    the simulated bracket that maximises expected points across all simulations.
    This accounts for the joint dependence of picks — a bracket with all four
    1-seeds in the Final Four only scores well in the ~1.5% of simulations
    where that actually occurs, so its mean score is low relative to a bracket
    with realistic upsets that scores across more simulations.

    Args:
        pred_df: DataFrame with columns Team, Seed, Region, R32, S16, E8,
            F4, NCG, Winner — as produced by standarize().
        n_sims: Number of tournament simulations to run. Default 10000.
        n_candidates: Number of simulated brackets to evaluate as candidate
            picks. Drawn from the first n_candidates simulations. Default 500.
        seed: Random seed for reproducibility. Default 42.

    Returns:
        Picks dict in the standard bracket format used by correct_bracket
        and real_Bracket.
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

    simulations = [_simulate_once(region_teams, probs_by_team, rng) for _ in range(n_sims)]

    best_bracket = _select_optimal_bracket(simulations, n_candidates)
    return _format_picks(best_bracket)
