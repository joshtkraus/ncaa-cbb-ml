"""Generate interactive HTML bracket viewer from picks and results."""

import json
import os

import pandas as pd

_SLOT_ORDER = ["1", "8", "5", "4", "6", "3", "7", "2"]
_REGIONS = ["West", "East", "South", "Midwest"]
_SLOT_TO_R64_IDX = {slot: i for i, slot in enumerate(_SLOT_ORDER)}


def _load_seed_map(data_path):
    """Return dict keyed by (year, team) -> seed.

    Args:
        data_path: Path to data.csv.

    Returns:
        Dict mapping (year, team) tuples to integer seed values.
    """
    data = pd.read_csv(data_path)
    return {
        (int(row["Year"]), row["Team"]): int(row["Seed"])
        for _, row in data[["Year", "Team", "Seed"]].iterrows()
    }


def _pick_cell(pick, correct, seed_val, actual=None, actual_seed=None):
    """Render a single pick cell with optional correct answer above it if wrong.

    Args:
        pick: Team name string for the pick.
        correct: Whether the pick was correct.
        seed_val: Seed number for the picked team.
        actual: Actual correct team name to display above a wrong pick.
        actual_seed: Seed number for the actual correct team.

    Returns:
        HTML string for the pick cell.
    """
    mark = "✓" if correct else "✗"
    cls = "correct" if correct else "incorrect"
    html = ""
    if not correct and actual:
        html += (
            f'<div class="actual-pick">'
            f'<span class="seed">{actual_seed or ""}</span>'
            f'<span class="name">{actual}</span>'
            f"</div>"
        )
    html += f'<span class="pick-team {cls}">{mark} {seed_val} {pick or "—"}</span>'
    return html


def _build_region_html(region, picks, results_year, seed_map, year, side):
    """Build HTML for one region's rounds (R64, R32, S16, E8, F4).

    Args:
        region: Region name string.
        picks: Nested bracket picks dict.
        results_year: Single-year results dict from results.json.
        seed_map: Dict mapping (year, team) to seed number.
        year: Tournament year integer.
        side: Either "left" or "right" controlling round display order.

    Returns:
        HTML string for the full region block.
    """
    r = results_year
    r32_set = set(r[region]["R32"])
    s16_set = set(r[region]["S16"])
    e8_set = set(r[region]["E8"])
    f4_set = set(r[region]["F4"]) if isinstance(r[region]["F4"], list) else {r[region]["F4"]}
    r64 = r[region]["R64"]

    def seed(team):
        return seed_map.get((year, team), "?")

    # R64 column — just shows both teams playing, no pick
    r64_col = []
    for slot in _SLOT_ORDER:
        idx = _SLOT_TO_R64_IDX[slot]
        t1, t2 = r64[idx * 2], r64[idx * 2 + 1]
        r64_col.append(
            f'<div class="matchup r64-matchup">'
            f'<div class="team {"winner" if t1 in r32_set else "loser"}">'
            f'<span class="seed">{seed(t1)}</span><span class="name">{t1}</span>'
            f"</div>"
            f'<div class="team {"winner" if t2 in r32_set else "loser"}">'
            f'<span class="seed">{seed(t2)}</span><span class="name">{t2}</span>'
            f"</div>"
            f"</div>"
        )

    # R32 column — pick for each R64 game
    r32_col = []
    for slot in _SLOT_ORDER:
        idx = _SLOT_TO_R64_IDX[slot]
        t1, t2 = r64[idx * 2], r64[idx * 2 + 1]
        actual_r32 = t1 if t1 in r32_set else t2
        pick = (picks[region]["R32"][slot] or [None])[0]
        correct = pick in r32_set if pick else False
        actual_display = actual_r32 if not correct else None
        r32_col.append(
            '<div class="matchup r32-matchup">'
            + _pick_cell(
                pick,
                correct,
                seed(pick),
                actual_display,
                seed(actual_r32) if actual_display else None,
            )
            + "</div>"
        )

    # S16 column
    s16_col = []
    for pod in ["1", "4", "3", "2"]:
        pick = (picks[region]["S16"][pod] or [None])[0]
        correct = pick in s16_set if pick else False
        # actual S16 winner for this pod: find the real S16 team from the same pod group
        pod_r32_teams = {
            "1": r64[0:4],  # slots 1,8 -> r64 indices 0-3
            "4": r64[4:8],  # slots 5,4
            "3": r64[8:12],  # slots 6,3
            "2": r64[12:16],  # slots 7,2
        }
        pod_teams = set(pod_r32_teams[pod])
        actual_s16 = (
            next((t for t in r[region]["S16"] if t in pod_teams), None) if not correct else None
        )
        s16_col.append(
            '<div class="matchup s16-matchup">'
            + _pick_cell(
                pick, correct, seed(pick), actual_s16, seed(actual_s16) if actual_s16 else None
            )
            + "</div>"
        )

    # E8 column
    e8_col = []
    for half, _pods in [("Upper", ["1", "4"]), ("Lower", ["3", "2"])]:
        pick = (picks[region]["E8"][half] or [None])[0]
        correct = pick in e8_set if pick else False
        half_r64 = r64[0:8] if half == "Upper" else r64[8:16]
        actual_e8 = (
            next((t for t in r[region]["E8"] if t in set(half_r64)), None) if not correct else None
        )
        e8_col.append(
            '<div class="matchup e8-matchup">'
            + _pick_cell(
                pick, correct, seed(pick), actual_e8, seed(actual_e8) if actual_e8 else None
            )
            + "</div>"
        )

    # F4 column
    pick_f4 = picks[region]["F4"]
    if isinstance(pick_f4, list):
        pick_f4 = pick_f4[0] if pick_f4 else None
    f4_correct = pick_f4 in f4_set if pick_f4 else False
    actual_f4 = next(iter(f4_set), None) if not f4_correct else None
    f4_html = (
        '<div class="matchup f4-matchup">'
        + _pick_cell(
            pick_f4, f4_correct, seed(pick_f4), actual_f4, seed(actual_f4) if actual_f4 else None
        )
        + "</div>"
    )

    def wrap(label, cls, content):
        return (
            f'<div class="round-col">'
            f'<div class="round-label">{label}</div>'
            f'<div class="round {cls}">{content}</div>'
            f"</div>"
        )

    r64_block = wrap("R64", "r64", "".join(r64_col))
    r32_block = wrap("R32", "r32", "".join(r32_col))
    s16_block = wrap("S16", "s16", "".join(s16_col))
    e8_block = wrap("E8", "e8", "".join(e8_col))
    f4_block = wrap("F4", "f4", f4_html)

    if side == "right":
        rounds_html = f4_block + e8_block + s16_block + r32_block + r64_block
    else:
        rounds_html = r64_block + r32_block + s16_block + e8_block + f4_block

    return (
        f'<div class="region region-{side}" id="region-{region.lower()}">'
        f'<div class="region-label">{region}</div>'
        f'<div class="region-rounds">{rounds_html}</div>'
        f"</div>"
    )


def _build_year_bracket(year_str, picks, results, seed_map):
    """Build the full bracket HTML for one year.

    Args:
        year_str: Year as a string, e.g. "2025".
        picks: Nested bracket picks dict for this year.
        results: Full results dict from results.json.
        seed_map: Dict mapping (year, team) to seed number.

    Returns:
        HTML string for the full bracket layout.
    """
    r = results[year_str]
    year = int(year_str)

    def seed(team):
        return seed_map.get((year, team), "?")

    ncg_set = set(r["NCG"])
    winner_str = r["Winner"]

    ncg_picks = picks.get("NCG", [])
    ncg_pick_1 = ncg_picks[0] if len(ncg_picks) > 0 else None
    ncg_pick_2 = ncg_picks[1] if len(ncg_picks) > 1 else None
    winner_pick = picks.get("Winner", [])
    winner_pick = winner_pick[0] if winner_pick else None

    ncg1_correct = ncg_pick_1 in ncg_set if ncg_pick_1 else False
    ncg2_correct = ncg_pick_2 in ncg_set if ncg_pick_2 else False
    winner_correct = winner_pick == winner_str if winner_pick else False

    # Derive the correct finalist per slot by matching each pick's bracket side.
    # We look at which F4 pick the user made per region to determine which side
    # each NCG slot belongs to, then find the actual finalist from that side.
    # This is robust regardless of list order in picks["NCG"].
    f4_we = {r[reg]["F4"][0] for reg in ("West", "East")}
    f4_sm = {r[reg]["F4"][0] for reg in ("South", "Midwest")}
    actual_finalist_we = next((t for t in r["NCG"] if t in f4_we), None)
    actual_finalist_sm = next((t for t in r["NCG"] if t in f4_sm), None)

    # Determine which side each pick came from by matching against the user's F4 picks
    pick_f4_we = {
        picks[reg]["F4"]
        if isinstance(picks[reg]["F4"], str)
        else (picks[reg]["F4"][0] if picks[reg]["F4"] else None)
        for reg in ("West", "East")
    }
    pick_f4_sm = {
        picks[reg]["F4"]
        if isinstance(picks[reg]["F4"], str)
        else (picks[reg]["F4"][0] if picks[reg]["F4"] else None)
        for reg in ("South", "Midwest")
    }

    def _correct_finalist_for(pick):
        if pick in pick_f4_we:
            return actual_finalist_we
        if pick in pick_f4_sm:
            return actual_finalist_sm
        # Pick didn't make F4 — infer side from whichever finalist is not
        # already assigned to the other pick
        other_pick = ncg_pick_2 if pick == ncg_pick_1 else ncg_pick_1
        other_side = (
            actual_finalist_we
            if other_pick in pick_f4_we
            else actual_finalist_sm
            if other_pick in pick_f4_sm
            else None
        )
        if other_side == actual_finalist_we:
            return actual_finalist_sm
        return actual_finalist_we

    correct_finalist_1 = _correct_finalist_for(ncg_pick_1)
    correct_finalist_2 = _correct_finalist_for(ncg_pick_2)

    def ncg_pick_html(pick, correct, correct_finalist):
        mark = "✓" if correct else "✗"
        cls = "correct" if correct else "incorrect"
        html = ""
        if not correct and correct_finalist and correct_finalist != pick:
            html += (
                f'<div class="actual-pick" style="color:var(--text-dim)">'
                f'<span class="seed">{seed(correct_finalist)}</span>'
                f'<span class="name">{correct_finalist}</span>'
                f"</div>"
            )
        html += (
            f'<div class="ncg-pick {cls}">'
            f'{mark} <span class="seed">{seed(pick)}</span>'
            f'<span class="name">{pick or "—"}</span>'
            f"</div>"
        )
        return html

    winner_actual_html = ""
    if not winner_correct:
        winner_actual_html = (
            f'<div class="actual-pick" style="color:var(--text-dim);justify-content:center">'
            f'<span class="seed">{seed(winner_str)}</span>'
            f'<span class="name">{winner_str}</span>'
            f"</div>"
        )

    center_html = f"""
    <div class="center-column">
      <div class="center-label">Championship</div>
      <div class="ncg-box">
        {ncg_pick_html(ncg_pick_1, ncg1_correct, correct_finalist_1)}
        {ncg_pick_html(ncg_pick_2, ncg2_correct, correct_finalist_2)}
      </div>
      <div class="winner-box">
        <div class="center-label">Champion</div>
        <div class="ncg-trophy">🏆</div>
        {winner_actual_html}
        <div class="winner-pick {"correct" if winner_correct else "incorrect"}">
          {"✓" if winner_correct else "✗"}
          <span class="seed">{seed(winner_pick)}</span>
          <span class="name">{winner_pick or "—"}</span>
        </div>
      </div>
    </div>"""

    west = _build_region_html("West", picks, r, seed_map, year, "left")
    east = _build_region_html("East", picks, r, seed_map, year, "left")
    south = _build_region_html("South", picks, r, seed_map, year, "right")
    midwest = _build_region_html("Midwest", picks, r, seed_map, year, "right")

    return f"""
    <div class="bracket-year" id="bracket-{year_str}">
      <div class="bracket-grid">
        <div class="left-side">{west}{east}</div>
        {center_html}
        <div class="right-side">{south}{midwest}</div>
      </div>
    </div>"""


def generate(picks_dir, results_path, data_path, output_path):
    """Generate the bracket HTML file.

    Args:
        picks_dir: Path to directory containing per-year picks JSON files.
        results_path: Path to results.json.
        data_path: Path to data.csv (used for seed lookups).
        output_path: Path to write the output HTML file.

    Returns:
        None
    """
    with open(results_path) as f:
        results = json.load(f)

    seed_map = _load_seed_map(data_path)

    pick_years = {}
    if os.path.exists(picks_dir):
        for fname in sorted(os.listdir(picks_dir)):
            if fname.endswith(".json"):
                year_str = fname.replace(".json", "")
                if year_str in results:
                    with open(os.path.join(picks_dir, fname)) as f:
                        pick_years[year_str] = json.load(f)

    if not pick_years:
        print("No pick files found.")
        return

    bracket_sections = []
    for year_str in sorted(pick_years.keys()):
        html = _build_year_bracket(year_str, pick_years[year_str], results, seed_map)
        bracket_sections.append((year_str, html))

    years_list = [y for y, _ in bracket_sections]
    latest_year = years_list[-1]

    # Build options with latest year first so it's the default selected value
    options_html = "\n".join(
        f'<option value="{y}"{" selected" if y == latest_year else ""}>{y}</option>'
        for y in reversed(years_list)
    )

    brackets_html = "\n".join(html for _, html in bracket_sections)

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NCAA Bracket Results</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@400;500;600&display=swap');

  :root {{
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #242736;
    --border: #2e3247;
    --text: #e8eaf0;
    --text-dim: #7c8098;
    --correct: #22c55e;
    --correct-bg: rgba(34,197,94,0.12);
    --incorrect: #ef4444;
    --incorrect-bg: rgba(239,68,68,0.12);
    --accent: #f59e0b;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  html, body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
  }}

  header {{
    background: var(--surface);
    border-bottom: 2px solid var(--accent);
    padding: 8px 20px;
    display: flex;
    align-items: center;
    gap: 16px;
    height: 48px;
    flex-shrink: 0;
  }}

  header h1 {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.4rem;
    letter-spacing: 2px;
    color: var(--accent);
  }}

  select {{
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    padding: 5px 12px;
    border-radius: 6px;
    cursor: pointer;
    outline: none;
  }}

  select:focus {{ border-color: var(--accent); }}

  #bracket-container {{
    width: 100%;
    padding: 8px;
  }}

  .bracket-year {{
    display: none;
    width: 100%;
  }}

  .bracket-year.active {{ display: block; }}

  .bracket-grid {{
    display: grid;
    grid-template-columns: minmax(0,1fr) auto minmax(0,1fr);
    gap: 6px;
    align-items: start;
    width: 100%;
  }}

  .left-side, .right-side {{
    display: flex;
    flex-direction: column;
    gap: 6px;
  }}

  .region {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    flex: 1;
  }}

  .region-label {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: 0.85rem;
    letter-spacing: 2px;
    color: var(--accent);
    background: var(--surface2);
    padding: 4px 10px;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }}

  .region-rounds {{
    display: flex;
    flex: 1;
  }}

  .round-col {{
    display: flex;
    flex-direction: column;
    border-right: 1px solid var(--border);
    flex: 1;
    min-width: 110px;
  }}

  .round-col:last-child {{ border-right: none; }}

  /* Right side: F4 is leftmost so its right border should show */
  .region-right .round-col:last-child {{ border-right: none; }}

  .round-label {{
    font-size: 0.55rem;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: var(--text-dim);
    padding: 3px 6px;
    background: var(--surface2);
    border-bottom: 1px solid var(--border);
    text-align: center;
    flex-shrink: 0;
  }}

  .round {{
    display: flex;
    flex-direction: column;
    flex: 1;
  }}

  .matchup {{
    padding: 4px 6px;
    border-bottom: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    gap: 2px;
    flex: 1;
    justify-content: center;
  }}

  .matchup:last-child {{ border-bottom: none; }}

  .team {{
    display: flex;
    align-items: center;
    gap: 3px;
    padding: 1px 3px;
    border-radius: 3px;
    font-size: 0.6rem;
    white-space: nowrap;
    overflow: hidden;
  }}

  .team.winner {{ opacity: 1; }}
  .team.loser   {{ opacity: 0.35; text-decoration: line-through; }}

  .seed {{
    font-size: 0.55rem;
    font-weight: 700;
    color: var(--text-dim);
    min-width: 12px;
    text-align: right;
    flex-shrink: 0;
  }}

  .name {{
    flex: 1;
    font-weight: 500;
    overflow: hidden;
    text-overflow: ellipsis;
  }}

  .pick-team {{
    font-size: 0.62rem;
    font-weight: 600;
    padding: 2px 5px;
    border-radius: 2px;
    white-space: nowrap;
    display: block;
    margin-top: 2px;
  }}

  .pick-team.correct  {{ color: var(--correct);   background: var(--correct-bg); }}
  .pick-team.incorrect {{ color: var(--incorrect); background: var(--incorrect-bg); }}

  .actual-pick {{
    display: flex;
    align-items: center;
    gap: 3px;
    font-size: 0.52rem;
    font-weight: 600;
    color: var(--text-dim);
    margin-top: 2px;
    white-space: nowrap;
  }}

  .actual-pick + .pick-team {{
    font-size: 0.52rem;
    padding: 1px 4px;
    margin-top: 1px;
  }}

  .r64-matchup {{ flex: 1; }}
  .r32-matchup {{ flex: 1; }}
  .s16-matchup {{ flex: 2; }}
  .e8-matchup  {{ flex: 4; }}
  .f4-matchup  {{ flex: 8; }}

  /* Center column */
  .center-column {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 8px 6px;
    gap: 6px;
  }}

  .center-label {{
    font-size: 0.55rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--text-dim);
    text-align: center;
  }}

  .ncg-box {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 10px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    min-width: 140px;
    max-width: 160px;
  }}

  .winner-box {{
    background: var(--surface);
    border: 2px solid var(--accent);
    border-radius: 10px;
    padding: 10px 10px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    min-width: 140px;
    max-width: 160px;
  }}

  .ncg-pick, .winner-pick {{
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 5px 8px;
    border-radius: 5px;
    font-weight: 600;
    font-size: 0.7rem;
    width: 100%;
    justify-content: center;
    white-space: nowrap;
    overflow: hidden;
  }}

  .ncg-pick.correct, .winner-pick.correct {{
    background: var(--correct-bg);
    color: var(--correct);
    border: 1px solid var(--correct);
  }}
  .ncg-pick.incorrect, .winner-pick.incorrect {{
    background: var(--incorrect-bg);
    color: var(--incorrect);
    border: 1px solid var(--incorrect);
  }}

  .winner-pick {{ font-size: 0.8rem; padding: 7px 10px; }}

  .ncg-trophy {{ font-size: 1.4rem; line-height: 1; }}
</style>
</head>
<body>

<header>
  <h1>NCAA Tournament Brackets</h1>
  <select id="year-select" onchange="showYear(this.value)">
    {options_html}
  </select>
</header>

<div id="bracket-container">
{brackets_html}
</div>

<script>
  function showYear(year) {{
    document.querySelectorAll('.bracket-year').forEach(el => el.classList.remove('active'));
    const el = document.getElementById('bracket-' + year);
    if (el) el.classList.add('active');
    window.scrollTo(0, 0);
  }}

  showYear('{latest_year}');
</script>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(page)


if __name__ == "__main__":
    cwd = os.path.abspath(os.path.dirname(__file__))
    generate(
        picks_dir=os.path.join(cwd, "results/picks"),
        results_path=os.path.join(cwd, "data/processed/results.json"),
        data_path=os.path.join(cwd, "data/processed/data.csv"),
        output_path=os.path.join(cwd, "results/brackets/brackets.html"),
    )
