"""Scrape win streak stats for tournament teams using URLs from sportsreference.csv."""

import time

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        " (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com",
}


def _get_win_streak_stats(team_url):
    """Scrape win streak stats from a team's schedule page.

    Args:
        team_url: Full URL to the team's season page.

    Returns:
        Tuple of (win_streak, last10, streak_avg, streak_sd), or None on failure.
    """
    sched_url = team_url.replace(".html", "-schedule.html")
    try:
        sched_html = requests.get(sched_url, headers=_HEADERS, timeout=15).text
    except Exception as e:
        print(f"  Request failed: {e}")
        return None

    sched_soup = BeautifulSoup(sched_html, features="html.parser")
    games = sched_soup.select_one("div#all_schedule")
    if games is None:
        print("  No schedule div found")
        return None

    g_type = games.find_all("td", {"data-stat": "game_type"})
    type_data = [d.getText() for d in g_type]
    streak = games.find_all("td", {"data-stat": "game_streak"})

    mask = [d != "NCAA" for d in type_data]
    streak = [x for x, m in zip(list(streak), mask, strict=False) if m]
    streak_data = [
        0 if len(d.getText()) == 0 else int(d.getText()[2:]) if d.getText()[0] == "W" else 0
        for d in streak
    ]

    if not streak_data:
        print("  No streak data found")
        return None

    win_result = [
        1 if streak_data[i] < streak_data[i + 1] else 0 for i in range(len(streak_data) - 1)
    ]
    return (
        streak_data[-1],
        int(np.sum(win_result[-10:])),
        float(np.mean(streak_data)),
        float(np.std(streak_data)),
    )


def run(input_path, output_path):
    """Scrape win streak stats for all teams in input CSV and save updated file.

    Args:
        input_path: Path to sportsreference.csv with URL column.
        output_path: Path to write the updated CSV with win streak columns added.
    """
    df = pd.read_csv(input_path)

    df["WinStreak"] = None
    df["Last10"] = None
    df["WinStreak_Avg"] = None
    df["WinStreak_SD"] = None

    for i, row in df.iterrows():
        url = row["URL"]
        time.sleep(5)

        result = _get_win_streak_stats(url)
        if result is not None:
            df.at[i, "WinStreak"] = result[0]
            df.at[i, "Last10"] = result[1]
            df.at[i, "WinStreak_Avg"] = result[2]
            df.at[i, "WinStreak_SD"] = result[3]

        # Save after every team
        df.drop(columns=["URL"]).to_csv(output_path, index=False)

    missing = df[df["WinStreak"].isna()]["Team"].tolist()
    if missing:
        print(f"\nTeams missing win streak data: {missing}")


if __name__ == "__main__":
    run(
        input_path="data/prediction/sportsreference.csv",
        output_path="data/prediction/sportsreference.csv",
    )
