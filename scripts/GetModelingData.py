# Libraries
import sys
sys.path.append('../scripts')
import os
import pandas as pd
import json
from utils.NameCleaner_KP import clean_KP
from utils.NameCleaner_SR import clean_SR
from utils.NameCleaner_Results import clean_results
from utils.GetSeedProb import calc_seed_prob

# Unit Tests
def check_KP_join(summary, joined):
     if len(summary) != len(joined):
          print('Missing Summary Teams: ',[team for team in summary['Team'].unique() if team not in joined['Team'].unique()])
          raise ValueError('Data Loss in Join.')
def check_data_join(data, SR, KP):
    if len(SR) != len(data):
        print('Missing KP Teams: ',[team for team in KP['Team'].unique() if team not in SR['Team'].unique()])
        print('Missing SR Teams: ',[team for team in SR['Team'].unique() if team not in KP['Team'].unique()])
        raise ValueError('Data Loss in Join.')
def check_results_naming(results_dict, SR, KP):
    # Iterate dict
    for year, regions in results_dict.items():
        for region, rounds in regions.items():
            if region not in ['NCG','Winner']:
                for round, teams in rounds.items():
                    for i, team in enumerate(teams):
                        if (team not in SR['Team'].values) | (team not in KP['Team'].values):
                            raise ValueError('Team missing: '+team)
            elif region == 'NCG':   
                for i, team in enumerate(rounds):
                    if (team not in SR['Team'].values) | (team not in KP['Team'].values):
                            raise ValueError('Team missing: '+team)
            else:
                if (team not in SR['Team'].values) | (team not in KP['Team'].values):
                            raise ValueError('Team missing: '+team)

# Run Web Scraper
from scrapers import GetData_SR

# Read Data
SR = pd.read_csv(os.path.join(os.path.abspath(os.getcwd()), 'data/raw/sportsreference.csv'))
with open(os.path.join(os.path.abspath(os.getcwd()), 'data/raw/results.json'), "r") as json_file:
    results = json.load(json_file)

# Read KenPom Data
# Teams who made play-in but lost
playin_dict = {
                2024:['Howard','Virginia','Montana St.','Boise St.'],
                2023:['Southeast Missouri St.','Texas Southern','Nevada','Mississippi St.'],
                2022:['Wyoming','Texas A&M Corpus Chris','Bryant','Rutgers'],
                2021:["Mount St. Mary's",'Michigan St.','Appalachian St.','Wichita St.'],
                2019:['North Carolina Central','Temple','Prairie View A&M',"St. John's"],
                2018:['LIU Brooklyn','UCLA','Arizona St.','North Carolina Central'],
                2017:['New Orleans','Providence','North Carolina Central','Wake Forest'],
                2016:['Fairleigh Dickinson','Tulsa','Vanderbilt','Southern'],
                2015:['Boise St.','Manhattan','North Florida','BYU'],
                2014:['Iowa','Texas Southern','Xavier',"Mount St. Mary's"],
                2013:['Long Island','Liberty','Middle Tennessee','Boise St.'],
                2012:['California','Lamar','Mississippi Valley St.','Iona'],
                2011:['UAB','Alabama St.','Arkansas Little Rock','USC'],
                2010:['Winthrop'],
                2009:['Alabama St.'],
                2008:['Coppin St.'],
                2007:['Florida A&M'],
                2006:['Hampton'],
                2005:['Alabama A&M'],
                2004:['Lehigh'],
                2003:['Texas Southern'],
                2002:['Alcorn St.']
                }
# List of years to include, excluding 2020
years = list(range(2002, 2025))
years.remove(2020)
# Initialize
KP = pd.DataFrame()
# Iterate years
for year in years:
    # Read data
    summary_temp = pd.read_csv(os.path.join(os.path.abspath(os.getcwd()), f'data/raw/KP/summary/{year}.csv'), index_col=False)
    points_temp = pd.read_csv(os.path.join(os.path.abspath(os.getcwd()), f'data/raw/KP/points/{year}.csv'), index_col=False)
    # Standardize Column Ordering
    summary_temp = summary_temp[['Season','TeamName','Tempo','RankTempo','AdjTempo','RankAdjTempo','OE','RankOE','AdjOE','RankAdjOE','DE','RankDE','AdjDE','RankAdjDE','AdjEM','RankAdjEM','seed']]
    points_temp = points_temp[['Season','TeamName','Off_1','RankOff_1','Off_2','RankOff_2','Off_3','RankOff_3','Def_1','RankDef_1','Def_2','RankDef_2','Def_3','RankDef_3']]
    # Rename Columns
    summary_temp.columns = ['Year','Team','Tempo','RankTempo','AdjTempo','RankAdjTempo','OE','RankOE','AdjOE','RankAdjOE','DE','RankDE','AdjDE','RankAdjDE','AdjEM','RankAdjEM','Seed']
    points_temp.columns = ['Year','Team','Off_1','RankOff_1','Off_2','RankOff_2','Off_3','RankOff_3','Def_1','RankDef_1','Def_2','RankDef_2','Def_3','RankDef_3']
    # Drop Non-Tournament Teams
    summary_temp = summary_temp.dropna(subset=['Seed'])
    # Drop Teams who Lost in Play-In
    summary_temp = summary_temp[~summary_temp['Team'].isin(playin_dict[year])]
    # Join
    temp_join = summary_temp.merge(points_temp, on=['Year','Team'])
    # Unit Test
    check_KP_join(summary_temp, temp_join)
    # Append
    KP = pd.concat([KP, temp_join], ignore_index=True)

# Clean Naming
KP = clean_KP(KP)
SR = clean_SR(SR)
results = clean_results(results)
# Join Dataframes
data = SR.merge(KP, on=['Team','Year','Seed'])
# Drop Any Duplicates or NAs Created
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)

# Unit Tests
check_data_join(data, SR, KP)
check_results_naming(results, SR, KP)

# Get Historical Seed Probabilities
data = calc_seed_prob(data)

# Export Data
# Get File Path
data_path = os.path.join(os.path.abspath(os.getcwd()), 'data/processed/data.csv')
bracket_path = os.path.join(os.path.abspath(os.getcwd()), 'data/processed/results.json')
# Export DF
data.to_csv(data_path,index=False)
# Export Dictionary
with open(bracket_path, 'w') as f:
    json.dump(results, f)