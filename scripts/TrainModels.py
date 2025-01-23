# Libraries
import os
import numpy as np
import pandas as pd
import joblib
import json
from models.FitModels import train_models, combine_model

# Tune Models Indicator
tune = False

# Data Paths
data_path = os.path.join(os.path.abspath(os.getcwd()), 'data/processed/data.csv')
bracket_path = os.path.join(os.path.abspath(os.getcwd()), 'data/processed/results.json')

# Load Data
teams = pd.read_csv(data_path)

# Load Real Results
with open(bracket_path, "r") as json_file:
    results = json.load(json_file)

if tune == True:
    # Tune Component Models
    best_params, model_accs = train_models(teams, validation_start=2017)
else:
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

# Combine Models
models, brier, points_df, accs_df = combine_model(teams,best_params,model_accs,results,backwards_test=2013,validation_year=2017)

# Validation Results
results = {}
results['R32'] = np.mean(brier[2])
results['S16'] = np.mean(brier[3])
results['E8'] = np.mean(brier[4])
results['F4'] = np.mean(brier[5])
results['NCG'] = np.mean(brier[6])
results['Winner'] = np.mean(brier[7])
results_df = pd.DataFrame(list(results.items()), columns=['Round', 'Brier Score'])

# Export

# Models
for r in [7,6,5,4,3,2]:
    # Path
    path = os.path.join(os.path.abspath(os.getcwd()), 'models/model_'+str(r)+'.pkl')
    # Save
    joblib.dump(models[r], path) 

# Parameters
path = os.path.join(os.path.abspath(os.getcwd()), 'models/tuned_params.json')
with open(path, 'w') as f:
    json.dump(best_params, f)

# Weights
path = os.path.join(os.path.abspath(os.getcwd()), 'models/tuned_weights.json')
with open(path, 'w') as f:
    json.dump(model_accs, f)
    
# Results
# Model Performance
path = os.path.join(os.path.abspath(os.getcwd()), 'results/model_performance.csv')
results_df.to_csv(path,index=False)
# Picks Accuracy
path = os.path.join(os.path.abspath(os.getcwd()), 'results/picks_accuracy.csv')
accs_df.to_csv(path,index=False)
# Picks Points
path = os.path.join(os.path.abspath(os.getcwd()), 'results/picks_points.csv')
points_df.to_csv(path,index=False)