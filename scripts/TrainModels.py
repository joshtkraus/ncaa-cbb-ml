# Libraries
import os
import numpy as np
import pandas as pd
import joblib
import json
from models.utils.ModelPipeline import feature_selection
from models.FitModels import train_models, combine_model

# Tune Models Indicator
tune = True
tune_upset = True

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
    best_params, model_accs = train_models(team_data=teams)
    # Perform Feature Selection
    best_features = feature_selection(teams,best_params,model_accs)
    # Export
    # Best Features
    path = os.path.join(os.path.abspath(os.getcwd()), 'models/best_features.json')
    with open(path, 'w') as f:
        json.dump(best_features, f)
else:
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

# Retune After Selecting Features
if tune == True:
    # Tune Component Models
    best_params, model_accs = train_models(teams, best_features)
    # Export
    # Parameters
    path = os.path.join(os.path.abspath(os.getcwd()), 'models/tuned_params.json')
    with open(path, 'w') as f:
        json.dump(best_params, f)
    # Weights
    path = os.path.join(os.path.abspath(os.getcwd()), 'models/tuned_weights.json')
    with open(path, 'w') as f:
        json.dump(model_accs, f)

# Combine Models
if tune_upset == True:
    models, upset_parameters, prec = combine_model(teams,
                                                    best_params,
                                                    model_accs,
                                                    results,
                                                    best_features,
                                                    backwards_year=2013,
                                                    validation_year=2017)
else:
    models, upset_parameters, prec = combine_model(teams,
                                                    best_params,
                                                    model_accs,
                                                    results,
                                                    best_features,
                                                    backwards_year=2013,
                                                    validation_year=2017,
                                                    upset_parameters=upset_parameters,
                                                    tune=False)

# Validation Results
results = {}
results['R32'] = np.mean(prec[2])
results['S16'] = np.mean(prec[3])
results['E8'] = np.mean(prec[4])
results['F4'] = np.mean(prec[5])
results['NCG'] = np.mean(prec[6])
results['Winner'] = np.mean(prec[7])
results_df = pd.DataFrame(list(results.items()), columns=['Round', 'Average Precision'])

# Export
# Models
for r in [7,6,5,4,3,2]:
    # Path
    path = os.path.join(os.path.abspath(os.getcwd()), 'models/model_'+str(r)+'.pkl')
    # Save
    joblib.dump(models[r], path) 
# Upset Parameters
path = os.path.join(os.path.abspath(os.getcwd()), 'models/upset_parameters.json')
with open(path, 'w') as f:
    json.dump(upset_parameters, f)
# Model Performance
path = os.path.join(os.path.abspath(os.getcwd()), 'results/backwards_test/model_performance.csv')
results_df.to_csv(path,index=False)