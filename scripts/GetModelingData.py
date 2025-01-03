# Libraries
import sys
sys.path.append('../scripts')
import os
import pandas as pd
import pickle
from utils.NameCleaner_KP import clean_KP
from utils.NameCleaner_SR import clean_SR
from utils.NameCleaner_Results import clean_results

# Unit Tests
def check_data_join(data, SR, KP):
    if (len(KP) != len(SR)) | (len(KP) != len(data)) | (len(SR) != len(data)):
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

# Run Web Scrapers
#from scrapers import GetData_KP, GetData_SR

# Read Data
KP = pd.read_csv(os.path.join(os.path.abspath(os.getcwd()), 'data/raw/kenpom.csv'))
SR = pd.read_csv(os.path.join(os.path.abspath(os.getcwd()), 'data/raw/sportsreference.csv'))
with open(os.path.join(os.path.abspath(os.getcwd()), 'data/raw/results.pkl'), 'rb') as f:
    results = pickle.load(f)

# Clean Naming
KP = clean_KP(KP)
SR = clean_SR(SR)
results = clean_results(results)

# Join Dataframes
data = SR.merge(KP, on=['Team','Year'])

# Drop Any Duplicates or NAs Created
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)

# Unit Tests
check_data_join(data, SR, KP)
check_results_naming(results, SR, KP)

# Export Data
# Get File Path
data_path = os.path.join(os.path.abspath(os.getcwd()), 'data/processed/data.csv')
bracket_path = os.path.join(os.path.abspath(os.getcwd()), 'data/processed/results.pkl')
# Export DF
data.to_csv(data_path,index=False)
# Export Dictionary
with open(bracket_path, 'wb') as f:
    pickle.dump(results, f)