# Libraries
import os
import json
import numpy as np
import pandas as pd
from scrapers.GetData_SR import run_scraper
from utils.NameCleaner_KP import clean_KP
from utils.NameCleaner_SR import clean_SR
from utils.GetSeedProb import calc_seed_prob
from models.utils.DataProcessing import create_splits
from models.utils.nn import tuned_nn
from models.utils.gbm import tuned_gbm
import xgboost as xgb
from models.utils.StandarizePredictions import standarize
from models.utils.MakePicks import predict_bracket
from utils.GroupedMetrics import get_grouped_metrics

scraper_ind = True

# Logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Year to Start Data At
year = 2025

# Unit Tests
def check_data_join(data, SR, KP):
    if len(SR) != len(data):
        print('Missing KP Teams: ',[team for team in KP['Team'].unique() if team not in SR['Team'].unique()])
        print('Missing SR Teams: ',[team for team in SR['Team'].unique() if team not in KP['Team'].unique()])
        raise ValueError('Data Loss in Join.')

# Get SR Data
if scraper_ind == True:
    SR = run_scraper(years=[year],
                export=False)
else:
    SR = pd.read_csv(os.path.join(os.path.abspath(os.getcwd()), f'data/prediction/sportsreference.csv'), index_col=False)

# Read KenPom Data
# Teams who made play-in but lost
playin_KP = ['Saint Francis','Texas','American','San Diego St.']

# Read data
summary_temp = pd.read_csv(os.path.join(os.path.abspath(os.getcwd()), f'data/prediction/KP/summary.csv'), index_col=False)
points_temp = pd.read_csv(os.path.join(os.path.abspath(os.getcwd()), f'data/prediction/KP/points.csv'), index_col=False)
roster_temp = pd.read_csv(os.path.join(os.path.abspath(os.getcwd()), f'data/prediction/KP/roster.csv'), index_col=False)
roster_temp.drop(columns=['Continuity','RankContinuity'],inplace=True)
# Rename Columns
summary_temp.columns = ['Year','Team','Tempo','RankTempo','AdjTempo','RankAdjTempo','OE','RankOE','AdjOE','RankAdjOE','DE','RankDE','AdjDE','RankAdjDE','AdjEM','RankAdjEM']
points_temp.columns = ['Year','Team','Off_1','RankOff_1','Off_2','RankOff_2','Off_3','RankOff_3','Def_1','RankDef_1','Def_2','RankDef_2','Def_3','RankDef_3']
roster_temp.columns = ['Year','Team','Size','SizeRank','Hgt5','Hgt5Rank','Hgt4','Hgt4Rank','Hgt3','Hgt3Rank','Hgt2','Hgt2Rank','Hgt1','Hgt1Rank','HgtEff','HgtEffRank',
                        'Exp','ExpRank','Bench','BenchRank','Pts5','Pts5Rank','Pts4','Pts4Rank','Pts3','Pts3Rank','Pts2','Pts2Rank','Pts1','Pts1Rank','OR5','OR5Rank',
                        'OR4','OR4Rank','OR3','OR3Rank','OR2','OR2Rank','OR1','OR1Rank','DR5','DR5Rank','DR4','DR4Rank','DR3','DR3Rank','DR2','DR2Rank','DR1','DR1Rank']
# Drop Teams who Lost in Play-In
summary_temp = summary_temp[~summary_temp['Team'].isin(playin_KP)]
# Join
KP = summary_temp.merge(points_temp, on=['Year','Team'])
KP = KP.merge(roster_temp, on=['Year','Team'])

# Clean Naming
KP = clean_KP(KP)
SR = clean_SR(SR)
# Join Dataframes
data = SR.merge(KP, on=['Team','Year'])
# Drop Any Duplicates or NAs Created
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)
# Unit Tests
check_data_join(data, SR, KP)

# Get Modeling Data (remove seed probs to recalculate)
data_path = os.path.join(os.path.abspath(os.getcwd()), 'data/processed/data.csv')
modeling_data = pd.read_csv(data_path)
modeling_data = modeling_data[modeling_data['Year']!=year]
modeling_data = modeling_data.loc[:,~modeling_data.columns.str.startswith(('R32_','S16_','E8_','F4_','NCG_','Winner_'))]
modeling_data = modeling_data.loc[:,~modeling_data.columns.str.contains('Seed_Avg')]
modeling_data.drop(columns=['First_Year'],inplace=True)

# Sort Columns
data = data[modeling_data.columns]
# Join to Current Predictions
data = pd.concat([modeling_data,data], ignore_index=True)
# Drop Any Duplicates
data.drop_duplicates(inplace=True)

# Get Historical Seed Probabilities
data[['R32_Actual_Full','S16_Actual_Full','E8_Actual_Full','F4_Actual_Full','NCG_Actual_Full','Winner_Actual_Full','First_Year']] = calc_seed_prob(data,lag=None,ind_col=True)
data[['R32_Actual_12','S16_Actual_12','E8_Actual_12','F4_Actual_12','NCG_Actual_12','Winner_Actual_12']] = calc_seed_prob(data,lag=12,ind_col=False)
data[['R32_Actual_6','S16_Actual_6','E8_Actual_6','F4_Actual_6','NCG_Actual_6','Winner_Actual_6']] = calc_seed_prob(data,lag=6,ind_col=False)

# Get Grouped Metrics
data = get_grouped_metrics(data)

# Export Data
# Get File Path
data_path = os.path.join(os.path.abspath(os.getcwd()), 'data/prediction/data.csv')
# Export DF
data.to_csv(data_path,index=False)

# Model Specifications
# Parameters
# NN
nn_path = os.path.join(os.path.abspath(os.getcwd()), 'models/components/nn.json')
with open(nn_path, "r") as json_file:
    nn_params = json.load(json_file)
nn_params = {int(key): value for key, value in nn_params.items()}
# GBM
gbm_path = os.path.join(os.path.abspath(os.getcwd()), 'models/components/gbm.json')
with open(gbm_path, "r") as json_file:
    gbm_params = json.load(json_file)
gbm_params = {int(key): value for key, value in gbm_params.items()}
# Weights
weights_path = os.path.join(os.path.abspath(os.getcwd()), 'models/weights.json')
with open(weights_path, "r") as json_file:
    weights = json.load(json_file)
weights = {int(key): value for key, value in weights.items()}

# Make Picks
predictions = {}
for r in range(2,8):
    # Get Years
    # Scaled Years
    full_years = [*range(data['Year'].min(),data['Year'].max()+1)]
    full_years.remove(2020)    
    _, _, years_scaled = create_splits(data, r, train=False,years_list=True)
    years_scaled = sorted(np.unique(years_scaled))
    idx = np.where(np.array(full_years)==year)[0][0]
    
    # Modeling Data
    # NN
    X_SMTL_nn, y_SMTL_nn, years_SMTL_nn = create_splits(data, r, train=True, years_list=True)
    X_nn, y, years_nn = create_splits(data, r, train=False, years_list=True)
    # GBM
    X_SMTL_gbm, y_SMTL_gbm, years_SMTL_gbm = create_splits(data, r, train=True, years_list=True)
    X_gbm, _, years_gbm = create_splits(data, r, train=False, years_list=True)

    # Create Splits                
    X_train_nn, X_test_nn = X_SMTL_nn[years_SMTL_nn < years_scaled[idx]], X_nn[years_nn == years_scaled[idx]]
    y_train_nn = y_SMTL_nn[years_SMTL_nn < years_scaled[idx]]
    X_train_gbm, X_test_gbm = X_SMTL_gbm[years_SMTL_gbm < years_scaled[idx]], X_gbm[years_gbm == years_scaled[idx]]
    y_train_gbm = y_SMTL_gbm[years_SMTL_gbm < years_scaled[idx]]

    # Create & Fit
    # NN
    nn = tuned_nn(nn_params[r],
                  X_train_nn, y_train_nn)
    # GBM
    gbm = tuned_gbm(gbm_params[r],
                    X_train_gbm, y_train_gbm)

    # Create Probabilities
    # NN
    prob_nn = nn.predict(X_test_nn, verbose=0).flatten()
    # GBM
    dtest = xgb.DMatrix(X_test_gbm)
    prob_gbm = gbm.predict(dtest)

    # Combine Probabilities
    y_pred = prob_nn * weights[r]['NN'] + prob_gbm * weights[r]['GBM']
    predictions['Round_'+str(r)] = y_pred

# To DF
# Standard
predictions['Team'] = data.loc[data['Year']==year,'Team'].values
predictions['Seed'] = data.loc[data['Year']==year,'Seed'].values
predictions['Region'] = data.loc[data['Year']==year,'Region'].values
pred_df = pd.DataFrame.from_dict(predictions)
pred_df = pred_df[['Team','Seed','Region','Round_2','Round_3','Round_4','Round_5','Round_6','Round_7']]

# Standardize
pred_df = standarize(pred_df)

# Get Expected Points
# Standard
points_df = pred_df.copy()
points_df['R32'] = points_df['R32'] * 10
points_df['S16'] = points_df['R32'] + points_df['S16'] * 20
points_df['E8'] = points_df['S16'] + points_df['E8'] * 40
points_df['F4'] = points_df['E8'] + points_df['F4'] * 80
points_df['NCG'] = points_df['F4'] + points_df['NCG'] * 160
points_df['Winner'] = points_df['NCG'] + points_df['Winner'] * 320

# Get Picks
picks = predict_bracket(points_df, real_picks=None, calc_correct=False)

# Export
# Probabilities
path = os.path.join(os.path.abspath(os.getcwd()), 'prediction/probabilities.csv')
pred_df.to_csv(path,index=False)
# Picks
path = os.path.join(os.path.abspath(os.getcwd()), 'prediction/picks.json')
with open(path, 'w') as f:
    json.dump(picks, f)