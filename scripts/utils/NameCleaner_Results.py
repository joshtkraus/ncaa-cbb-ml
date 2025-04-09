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
    team = re.sub(r' Am', ' A&M', team)
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
    team = re.sub(r'Mcneese State', 'McNeese', team)
    team = re.sub(r'McNeese State', 'McNeese', team)
    team = re.sub(r'Iupui', 'IUPUI', team)
    team = re.sub(r'Depaul', 'DePaul', team)
    team = re.sub(r'Miami Oh', 'Miami OH', team)
    team = re.sub(r'Loyola Md', 'Loyola MD', team)
    team = re.sub(r'Texas El Paso', 'UTEP', team)
    team = re.sub(r'Texas San Antonio', 'UTSA', team)
    team = re.sub(r'Nevada Las Vegas', 'UNLV', team)
    team = re.sub(r'Texas Arlington', 'UT Arlington', team)
    team = re.sub(r'Long Island University', 'Long Island', team)
    team = re.sub(r'Saint Francis Pa', 'Saint Francis', team)
    team = re.sub(r'Southern Illinois Edwardsville', 'SIUE', team)
    team = re.sub(r'Nebraska Omaha', 'Omaha', team)
    team = re.sub(r'California San Diego', 'UC San Diego', team)
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