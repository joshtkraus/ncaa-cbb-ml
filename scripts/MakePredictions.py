# Libraries
import os
import json
import joblib
import pandas as pd
from scrapers.GetData_SR import run_scraper
from utils.NameCleaner_KP import clean_KP
from utils.NameCleaner_SR import clean_SR
from utils.GetSeedProb import calc_seed_prob
from models.utils.DataProcessing import create_splits
from models.utils.StandarizePredictions import standarize, create_upset_picks
from models.utils.MakePicks import predict_bracket

# Year to Start Data At
year = 2024

# Run Web Scraper Ind
scraper = True

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
def check_results_naming(results_dict, SR):
    # Iterate dict
    for year, regions in results_dict.items():
        for region, rounds in regions.items():
            if region not in ['NCG','Winner']:
                for round, teams in rounds.items():
                    for i, team in enumerate(teams):
                        if team not in SR['Team'].values:
                            raise ValueError('Team missing: '+team)
            elif region == 'NCG':   
                for i, team in enumerate(rounds):
                    if team not in SR['Team'].values:
                            raise ValueError('Team missing: '+team)
            else:
                if team not in SR['Team'].values:
                            raise ValueError('Team missing: '+team)

# Get SR Data
SR = run_scraper(years=[year],
                 export=False)

# Read KenPom Data
# Teams who made play-in but lost
playin_dict = {
                2024:['Howard','Virginia','Montana St.','Boise St.'],
                }
# Read data
summary_temp = pd.read_csv(os.path.join(os.path.abspath(os.getcwd()), f'data/prediction/KP/summary.csv'), index_col=False)
points_temp = pd.read_csv(os.path.join(os.path.abspath(os.getcwd()), f'data/prediction/KP/points.csv'), index_col=False)
roster_temp = pd.read_csv(os.path.join(os.path.abspath(os.getcwd()), f'data/prediction/KP/roster.csv'), index_col=False)
# Rename Columns
summary_temp.columns = ['Year','Team','Tempo','RankTempo','AdjTempo','RankAdjTempo','OE','RankOE','AdjOE','RankAdjOE','DE','RankDE','AdjDE','RankAdjDE','AdjEM','RankAdjEM','Seed']
points_temp.columns = ['Year','Team','Off_1','RankOff_1','Off_2','RankOff_2','Off_3','RankOff_3','Def_1','RankDef_1','Def_2','RankDef_2','Def_3','RankDef_3']
roster_temp.columns = ['Year','Team','Size','SizeRank','Hgt5','Hgt5Rank','Hgt4','Hgt4Rank','Hgt3','Hgt3Rank','Hgt2','Hgt2Rank','Hgt1','Hgt1Rank','HgtEff','HgtEffRank',
                        'Exp','ExpRank','Bench','BenchRank','Pts5','Pts5Rank','Pts4','Pts4Rank','Pts3','Pts3Rank','Pts2','Pts2Rank','Pts1','Pts1Rank','OR5','OR5Rank',
                        'OR4','OR4Rank','OR3','OR3Rank','OR2','OR2Rank','OR1','OR1Rank','DR5','DR5Rank','DR4','DR4Rank','DR3','DR3Rank','DR2','DR2Rank','DR1','DR1Rank']
# Drop Non-Tournament Teams
summary_temp = summary_temp.dropna(subset=['Seed'])
# Drop Teams who Lost in Play-In
summary_temp = summary_temp[~summary_temp['Team'].isin(playin_dict[year])]
# Join
KP = summary_temp.merge(points_temp, on=['Year','Team'])
KP = KP.merge(roster_temp, on=['Year','Team'])
# Unit Test
check_KP_join(summary_temp, KP)

# Clean Naming
KP = clean_KP(KP)
SR = clean_SR(SR)
# Join Dataframes
data = SR.merge(KP, on=['Team','Year','Seed'])
# Drop Any Duplicates or NAs Created
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)

# Unit Tests
check_data_join(data, SR, KP)

# Get Historical Seed Probabilities
data[['R32_Actual_Full','S16_Actual_Full','E8_Actual_Full','F4_Actual_Full','NCG_Actual_Full','Winner_Actual_Full','First_Year']] = calc_seed_prob(data,lag=None,ind_col=True)
data[['R32_Actual_12','S16_Actual_12','E8_Actual_12','F4_Actual_12','NCG_Actual_12','Winner_Actual_12']] = calc_seed_prob(data,lag=12,ind_col=False)
data[['R32_Actual_6','S16_Actual_6','E8_Actual_6','F4_Actual_6','NCG_Actual_6','Winner_Actual_6']] = calc_seed_prob(data,lag=6,ind_col=False)

# Export Data
# Get File Path
data_path = os.path.join(os.path.abspath(os.getcwd()), 'data/prediction/data.csv')
# Export DF
data.to_csv(data_path,index=False)

# Get Modeling Data
data_path = os.path.join(os.path.abspath(os.getcwd()), 'data/processed/data.csv')
modeling_data = pd.read_csv(data_path)
modeling_data = modeling_data[modeling_data['Year']!=year]
# Join to Current Predictions
data = pd.concat([data, modeling_data], ignore_index=True)
# Drop Any Duplicates
data.drop_duplicates(inplace=True)

# Model Specifications
# Best Features
path = os.path.join(os.path.abspath(os.getcwd()), 'models/best_features.json')
with open(path, "r") as json_file:
    best_features = json.load(json_file)
best_features = {int(key): value for key, value in best_features.items()}
# Tuned Params
path = os.path.join(os.path.abspath(os.getcwd()), 'models/tuned_params.json')
with open(path, "r") as json_file:
    best_params = json.load(json_file)
best_params = {int(key): value for key, value in best_params.items()}
# Tuned Weights
path = os.path.join(os.path.abspath(os.getcwd()), 'models/tuned_weights.json')
with open(path, "r") as json_file:
    model_accs = json.load(json_file)
model_accs = {int(key): value for key, value in model_accs.items()}
# Upset Parameters
path = os.path.join(os.path.abspath(os.getcwd()), 'models/upset_parameters.json')
with open(path, "r") as json_file:
    upset_parameters = json.load(json_file)

# Make Picks
predictions = {}
for r in range(2,8):
    # Load Model
    path = os.path.join(os.path.abspath(os.getcwd()), 'models/model_'+str(r)+'.pkl')
    voting_clf = joblib.load(path)

    # Data Splits
    X, y = create_splits(data,r,best_features[r])

    # Subset Year
    X = X[data['Year']==year]
    y = y[data['Year']==year]

    # Get Prediction
    y_pred = voting_clf.predict_proba(X)[:, 1]
    predictions['Round_'+str(r)] = y_pred

# To DF
predictions['Team'] = data.loc[data['Year']==year,'Team'].values
predictions['Seed'] = data.loc[data['Year']==year,'Seed'].values
predictions['Region'] = data.loc[data['Year']==year,'Region'].values
pred_df = pd.DataFrame.from_dict(predictions)
pred_df = pred_df[['Team','Seed','Region','Round_2','Round_3','Round_4','Round_5','Round_6','Round_7']]

# Standardize
pred_df = standarize(pred_df)
# Apply Alphas
pred_df_upset = create_upset_picks(pred_df, 
                             upset_parameters['R32'], 
                             upset_parameters['S16'], 
                             upset_parameters['E8'], 
                             upset_parameters['F4'], 
                             upset_parameters['NCG'], 
                             upset_parameters['Winner'])

# Get Expected Points
# Regular
points_df = pred_df.copy()
points_df['R32'] = pred_df['R32'] * 10
points_df['S16'] = points_df['R32'] + pred_df['S16'] * 20
points_df['E8'] = points_df['S16'] + pred_df['E8'] * 40
points_df['F4'] = points_df['E8'] + pred_df['F4'] * 80
points_df['NCG'] = points_df['F4'] + pred_df['NCG'] * 160
points_df['Winner'] = points_df['NCG'] + pred_df['Winner'] * 320
# Upset
points_df_upset = pred_df_upset.copy()
points_df_upset['R32'] = points_df_upset['R32'] * 10
points_df_upset['S16'] = points_df_upset['R32'] + points_df_upset['S16'] * 20
points_df_upset['E8'] = points_df_upset['S16'] + points_df_upset['E8'] * 40
points_df_upset['F4'] = points_df_upset['E8'] + points_df_upset['F4'] * 80
points_df_upset['NCG'] = points_df_upset['F4'] + points_df_upset['NCG'] * 160
points_df_upset['Winner'] = points_df_upset['NCG'] + points_df_upset['Winner'] * 320

# Get Picks
picks = predict_bracket(points_df, real_picks=None, calc_correct=False)
picks_upset = predict_bracket(points_df_upset, real_picks=None, calc_correct=False)

# Export
# Probabilities
path = os.path.join(os.path.abspath(os.getcwd()), 'prediction/probabilities/standard.csv')
pred_df.to_csv(path,index=False)
path = os.path.join(os.path.abspath(os.getcwd()), 'prediction/probabilities/upset.csv')
pred_df_upset.to_csv(path,index=False)
# Picks
path = os.path.join(os.path.abspath(os.getcwd()), 'prediction/picks/standard.json')
with open(path, 'w') as f:
    json.dump(picks, f)
path = os.path.join(os.path.abspath(os.getcwd()), 'prediction/picks/upset.json')
with open(path, 'w') as f:
    json.dump(picks_upset, f)