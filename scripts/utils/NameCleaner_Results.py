# Functions to Clean Names in Results Pickle
def clean_names(team):     
    import re                   
    team = re.sub(r'Mount St Marys', "Mount Saint Mary's", team)
    team = re.sub(r'^St ', 'Saint ', team)
    team = re.sub(r'Saint Marys Ca', "Saint Mary's", team)
    team = re.sub(r'Albany Ny', 'Albany', team)
    team = re.sub(r'Saint Josephs', "Saint Joseph's", team)
    team = re.sub(r'Saint Johns Ny', "Saint John's", team)
    team = re.sub(r'Saint Peters', "Saint Peter's", team)
    team = re.sub(r'Loyola Il', 'Loyola Chicago', team)
    team = re.sub(r'Texas Am Corpus Christi', 'Texas A&M Corpus Christi', team)
    team = re.sub(r'Texas Am', 'Texas A&M', team)
    team = re.sub(r'North Carolina At', 'North Carolina A&T', team)
    team = re.sub(r'Brigham Young', 'BYU', team)
    team = re.sub(r'California Irvine', 'UC Irvine', team)
    team = re.sub(r'Southern Methodist', 'SMU', team)
    team = re.sub(r'Alabama Birmingham', 'UAB', team)
    team = re.sub(r'Southern California', 'USC', team)
    team = re.sub(r'North Carolina Wilmington', 'UNC Wilmington', team)
    team = re.sub(r'North Carolina Asheville', 'UNC Asheville', team)
    team = re.sub(r'California Davis', 'UC Davis', team)
    team = re.sub(r'North Carolina Greensboro', 'UNC Greensboro', team)
    team = re.sub(r'Pennsylvania', 'Penn', team)
    team = re.sub(r'Texas Christian', 'TCU', team)
    team = re.sub(r'Maryland Baltimore County', 'UMBC', team)
    team = re.sub(r'Central Florida', 'UCF', team)
    team = re.sub(r'California Santa Barbara', 'UC Santa Barbara', team)
    team = re.sub(r'Virginia Commonwealth', 'VCU', team)
    team = re.sub(r'Louisiana State', 'LSU', team)
    team = re.sub(r'College Of Charleston', 'Charleston', team)
    team = re.sub(r'Louisiana Lafayette', 'Louisiana', team)
    team = re.sub(r'Ucla', 'UCLA', team)
    team = re.sub(r'Miami Fl', 'Miami FL', team)
    team = re.sub(r'Mcneese State', 'McNeese State', team)
    return team

# Function to Iterate Renaming
def clean_results(dict_teams):
    import copy 
    # Copy Dict
    dict_teams_copy = copy.deepcopy(dict_teams)

    # Iterate dict
    for year, regions in dict_teams_copy.items():
        for region, rounds in regions.items():
            if region not in ['NCG','Winner']:
                for round, teams in rounds.items():
                    for i, team in enumerate(teams):
                        teams[i] = clean_names(team)
            elif region == 'NCG':   
                for i, team in enumerate(rounds):
                    rounds[i] = clean_names(team)
            else:
                dict_teams_copy[year][region] = clean_names(dict_teams_copy[year][region])
    return dict_teams_copy