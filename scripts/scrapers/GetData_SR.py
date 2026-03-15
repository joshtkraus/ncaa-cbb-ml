"""Web scraper for Sports Reference NCAA tournament data."""

import json
import os
import re
import time
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        " (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com",
}

_DEFAULT_YEARS = [y for y in range(2007, 2026) if y != 2020]

_ALL_REGIONS = [
    "east",
    "west",
    "midwest",
    "south",
    "southeast",
    "southwest",
    "minneapolis",
    "atlanta",
    "oakland",
    "washington",
    "syracuse",
    "albuquerque",
    "austin",
    "chicago",
    "stlouis",
    "eastrutherford",
    "phoenix",
]

_REGIONS_CONVERT = {
    2025: {"south": "West", "west": "East", "east": "South", "midwest": "Midwest"},
    2024: {"east": "West", "west": "East", "south": "South", "midwest": "Midwest"},
    2023: {"south": "West", "east": "East", "midwest": "South", "west": "Midwest"},
    2022: {"west": "West", "east": "East", "south": "South", "midwest": "Midwest"},
    2021: {"west": "West", "east": "East", "south": "South", "midwest": "Midwest"},
    2019: {"east": "West", "west": "East", "south": "South", "midwest": "Midwest"},
    2018: {"south": "West", "west": "East", "east": "South", "midwest": "Midwest"},
    2017: {"east": "West", "west": "East", "midwest": "South", "south": "Midwest"},
    2016: {"south": "West", "west": "East", "east": "South", "midwest": "Midwest"},
    2015: {"midwest": "West", "west": "East", "east": "South", "south": "Midwest"},
    2014: {"south": "West", "east": "East", "west": "South", "midwest": "Midwest"},
    2013: {"midwest": "West", "west": "East", "south": "South", "east": "Midwest"},
    2012: {"south": "West", "west": "East", "east": "South", "midwest": "Midwest"},
    2011: {"east": "West", "west": "East", "southwest": "South", "southeast": "Midwest"},
    2010: {"midwest": "West", "west": "East", "east": "South", "south": "Midwest"},
    2009: {"midwest": "West", "west": "East", "east": "South", "south": "Midwest"},
    2008: {"east": "West", "midwest": "East", "south": "South", "west": "Midwest"},
    2007: {"midwest": "West", "west": "East", "east": "South", "south": "Midwest"},
    2006: {
        "atlanta": "West",
        "oakland": "East",
        "washington": "South",
        "minneapolis": "Midwest",
    },
    2005: {
        "chicago": "West",
        "albuquerque": "East",
        "syracuse": "South",
        "austin": "Midwest",
    },
    2004: {
        "stlouis": "West",
        "eastrutherford": "East",
        "atlanta": "South",
        "phoenix": "Midwest",
    },
    2003: {"midwest": "West", "west": "East", "south": "South", "east": "Midwest"},
    2002: {"south": "West", "west": "East", "east": "South", "midwest": "Midwest"},
}

_ROUND_DICT = {1: "R64", 2: "R32", 3: "S16", 4: "E8", 5: "F4"}


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def check_year_length_df(df):
    """Validate that exactly 64 teams are present for a given year.

    Args:
        df: DataFrame subset for a single tournament year.

    Raises:
        ValueError: If the number of rows is not exactly 64.
    """
    if len(df) < 64:
        raise ValueError("< 64 Teams, # of Teams is: " + str(len(df)))
    elif len(df) > 64:
        raise ValueError("> 64 Teams, # of Teams is: " + str(len(df)))


def _check_region_round(region, round_name, teams):
    """Validate team count for a single region/round combination.

    Args:
        region: Region name string used in error messages.
        round_name: Round identifier string (R64, R32, S16, E8, F4).
        teams: List or dict of teams for this round.

    Raises:
        ValueError: If the team count does not match the expected value for the round.
    """
    expected = {"F4": 1, "E8": 2, "S16": 4, "R32": 8, "R64": 16}
    n = expected.get(round_name, 16)
    if len(teams) != n:
        raise ValueError(f"{region} {round_name} Incorrect, # of Teams is: {len(teams)}")


def check_year_round_length_dict(picks_dict):
    """Validate team counts for every region and round in the bracket dict.

    Args:
        picks_dict: Nested dict of bracket results for a single year, keyed by
            region then round name.

    Raises:
        ValueError: If any region/round has an unexpected number of teams.
    """
    for region, rounds in picks_dict.items():
        if region == "Winner":
            if not isinstance(rounds, str):
                raise ValueError("Winner Incorrect, # of Teams is: " + str(len(rounds)))
        elif region == "NCG":
            if len(rounds) != 2:
                raise ValueError("NCG Incorrect, # of Teams is: " + str(len(rounds)))
        else:
            for round_name, teams in rounds.items():
                _check_region_round(region, round_name, teams)


# ---------------------------------------------------------------------------
# Scraping helpers
# ---------------------------------------------------------------------------


def _get_soup(url):
    """Fetch a URL and return a parsed BeautifulSoup object.

    Args:
        url: Full URL string to fetch.

    Returns:
        Parsed BeautifulSoup object for the page HTML.
    """
    time.sleep(5)
    html = requests.get(url, headers=_HEADERS).text
    return BeautifulSoup(html, features="html.parser")


def _team_names_from_round(round_tag):
    """Extract and format team name strings from a bracket round HTML element.

    Args:
        round_tag: BeautifulSoup tag for a single bracket round div.

    Returns:
        List of title-cased team name strings.
    """
    teams = round_tag.find_all("a", href=lambda href: href and "schools" in href)
    names = [urlparse(t["href"]).path.split("/")[3] for t in teams]
    return [re.sub(r"-", " ", name.title()) for name in names]


def _parse_f4(year_soup, year, bracket_data):
    """Parse Final Four and championship game results into bracket_data.

    Args:
        year_soup: Parsed BeautifulSoup object for the tournament year page.
        year: Integer tournament year.
        bracket_data: Dict to store results into, modified in place.
    """
    bracket_data[year]["NCG"] = []
    f4_region = year_soup.select_one("div#national")
    bracket = f4_region.find(id="bracket")
    rounds = bracket.find_all(class_="round")
    for idx, round_tag in enumerate(rounds, start=1):
        if idx == 1:
            continue
        team_names = _team_names_from_round(round_tag)
        if idx == 2:
            bracket_data[year]["NCG"] = team_names
        else:
            bracket_data[year]["Winner"] = "Connecticut" if year == 2024 else team_names[0]


def _parse_region_bracket(year_soup, year, region, bracket_data):
    """Parse all round results for a single bracket region.

    Args:
        year_soup: Parsed BeautifulSoup object for the tournament year page.
        year: Integer tournament year.
        region: Lowercase region name string (e.g. 'east').
        bracket_data: Dict to store results into, modified in place.

    Returns:
        The tourney_region BeautifulSoup tag, or None if the region was not found.
    """
    tourney_region = year_soup.select_one(f"div#{region}")
    if tourney_region is None:
        return None

    std_region = _REGIONS_CONVERT[year][region]
    bracket_data[year][std_region] = {}
    bracket = tourney_region.find(id="bracket")
    rounds = bracket.find_all(class_="round")
    for idx, round_tag in enumerate(rounds, start=1):
        team_names = _team_names_from_round(round_tag)
        bracket_data[year][std_region][_ROUND_DICT[idx]] = team_names

    return tourney_region


def _determine_round(homepage_text, year):
    """Determine the tournament round a team reached from their homepage text.

    Args:
        homepage_text: List of paragraph text strings from the team's SR page.
        year: Integer tournament year (affects round labelling for 2011-2015).

    Returns:
        Integer round number (1–7), or None if the round could not be determined.
    """
    if any("Won National Final" in t for t in homepage_text):
        return 7
    if any("Lost National Final" in t for t in homepage_text):
        return 6
    if any("National Semifinal" in t for t in homepage_text):
        return 5
    if any("Regional Final" in t for t in homepage_text):
        return 4
    if any("Regional Semifinal" in t for t in homepage_text):
        return 3
    if year in range(2011, 2016):
        if any("Third Round" in t for t in homepage_text):
            return 3
        if any("Second Round" in t for t in homepage_text):
            return 2
        return 1
    if any("Second Round" in t for t in homepage_text):
        return 2
    return 1


def _get_win_streak_stats(links, headers_dict):
    """Scrape and compute win streak statistics from a team's schedule page.

    Args:
        links: Relative URL path string to the team's season page.
        headers_dict: HTTP headers dict to use for the request.

    Returns:
        Tuple of (current_win_streak, last_10_wins, mean_streak, std_streak).
    """
    sched_url = "https://www.sports-reference.com" + links[:-5] + "-schedule.html"
    sched_html = requests.get(sched_url, headers=headers_dict).text
    sched_soup = BeautifulSoup(sched_html, features="html.parser")
    games = sched_soup.select_one("div#all_schedule")
    g_type = games.find_all("td", {"data-stat": "game_type"})
    type_data = [d.getText() for d in g_type]
    streak = games.find_all("td", {"data-stat": "game_streak"})
    mask = [d != "NCAA" for d in type_data]
    streak = [x for x, m in zip(list(streak), mask, strict=False) if m]
    streak_data = [
        0 if len(d.getText()) == 0 else int(d.getText()[2:]) if d.getText()[0] == "W" else 0
        for d in streak
    ]
    win_result = [
        1 if streak_data[i] < streak_data[i + 1] else 0 for i in range(len(streak_data) - 1)
    ]
    return (
        streak_data[-1],
        int(np.sum(win_result[-10:])),
        float(np.mean(streak_data)),
        float(np.std(streak_data)),
    )


def _scrape_team_page(team_url, year, region, regions_convert, seeddata):
    """Scrape a single team's SR page and append their data to seeddata.

    Args:
        team_url: Full URL string for the team's season page.
        year: Integer tournament year.
        region: Lowercase region name string.
        regions_convert: Dict mapping year to region name conversion dict.
        seeddata: DataFrame accumulating all scraped team rows.

    Returns:
        Updated seeddata DataFrame with the team's row appended if seed was found.
    """
    time.sleep(5)
    team_html = requests.get(team_url, headers=_HEADERS).text
    team_soup = BeautifulSoup(team_html, features="html.parser")

    part_url = re.sub("https://www.sports-reference.com/cbb/schools/", "", team_url)
    team_key = re.sub(r"/(\d+).html", "", part_url).title()

    homepage = team_soup.select_one("div#info")
    homepage_text = [hp.getText() for hp in homepage.findAll("p")]

    conf_match = re.search(r"\bin\s+(.*?)\s+MBB", homepage_text[2])
    conf = conf_match.group(1) if conf_match is not None else ""
    round_num = _determine_round(homepage_text, year)

    wins_raw = int(homepage_text[2][9:11]) - (round_num - 1)

    import_text = [im.getText() for im in homepage.findAll("a")]
    conf_tourney = 1 if any("Tourney Champ" in t for t in import_text) else 0

    std_region = regions_convert[year][region]

    homepage_text_last = homepage_text[-1]
    seeds = re.search(r"(\d+) seed", homepage_text_last)

    vcu_key = "Virginia-Commonwealth/Men"
    is_vcu_2021 = (year == 2021) and (team_key == vcu_key)
    if is_vcu_2021:
        seeds = "10"

    if seeds is None:
        return seeddata

    seed_val = "10" if is_vcu_2021 else seeds.group(0).replace(" seed", "")

    links_tag = team_soup.select_one("div#info a[href*='/cbb/schools/']")
    links = links_tag["href"] if links_tag else None

    if links is None:
        return seeddata

    win_streak, last10, streak_avg, streak_sd = _get_win_streak_stats(
        links,
        _HEADERS,
    )

    row = {
        "Year": year,
        "Team": team_key.replace("/Men", "").replace("-", " "),
        "Conf": conf,
        "Round": round_num,
        "Wins": wins_raw,
        "Conf Tourney": conf_tourney,
        "Region": std_region,
        "Seed": int(seed_val),
        "WinStreak": win_streak,
        "Last10": last10,
        "WinStreak_Avg": streak_avg,
        "WinStreak_SD": streak_sd,
    }
    seeddata = pd.concat([seeddata, pd.DataFrame([row])], ignore_index=True)
    return seeddata


def _scrape_region_teams(tourney_region, year, region, seeddata):
    """Scrape all 16 tournament teams from a single bracket region.

    Args:
        tourney_region: BeautifulSoup tag for the region's bracket div.
        year: Integer tournament year.
        region: Lowercase region name string.
        seeddata: DataFrame accumulating all scraped team rows.

    Returns:
        Updated seeddata DataFrame with all region teams appended.
    """
    first_round_div = tourney_region.select_one("div.round")
    for link_tag in first_round_div.find_all("a"):
        links = link_tag.get("href")
        if links.startswith("/cbb/s"):
            team_url = "https://www.sports-reference.com" + links
            seeddata = _scrape_team_page(team_url, year, region, _REGIONS_CONVERT, seeddata)
    return seeddata


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_scraper(years=None, export=True):
    """Scrape NCAA tournament data from Sports Reference for the given years.

    For each year, scrapes team stats, bracket results, seeds, win streaks,
    and conference tournament data. Optionally exports to CSV and JSON.

    Args:
        years: Optional list of integer years to scrape. Defaults to all years
            from 2007 to 2025, excluding 2020.
        export: If True, write results to disk and return None. If False,
            return the scraped DataFrame directly.

    Returns:
        DataFrame of scraped team data if export is False, otherwise None.
    """
    seeddata = pd.DataFrame()
    bracket_data = {}

    if years is None:
        years = list(_DEFAULT_YEARS)

    print("Scraping Sports Reference...")
    for year in years:
        print(year)
        time.sleep(5)
        year_url = f"https://www.sports-reference.com/cbb/postseason/{year}-ncaa.html"
        year_html = requests.get(year_url, headers=_HEADERS).text
        year_soup = BeautifulSoup(year_html, features="html.parser")

        bracket_data[year] = {}

        # Parse Final Four / championship
        _parse_f4(year_soup, year, bracket_data)

        # Parse each regional bracket
        for region in _ALL_REGIONS:
            tourney_region = _parse_region_bracket(year_soup, year, region, bracket_data)
            if tourney_region is None:
                continue
            seeddata = _scrape_region_teams(tourney_region, year, region, seeddata)

        # Unit tests
        check_year_length_df(seeddata[seeddata["Year"] == year])
        check_year_round_length_dict(bracket_data[year])

    if export:
        data_path = os.path.join(os.path.abspath(os.getcwd()), "data/raw/sportsreference.csv")
        bracket_path = os.path.join(os.path.abspath(os.getcwd()), "data/raw/results.json")
        seeddata.to_csv(data_path, index=False)
        with open(bracket_path, "w") as f:
            json.dump(bracket_data, f)
        return None

    return seeddata
