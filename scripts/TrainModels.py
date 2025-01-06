# Libraries
import os
import numpy as np
import pandas as pd
import joblib
import json
from models.FitModels import backwards_model, combine_model

# Data Paths
data_path = os.path.join(os.path.abspath(os.getcwd()), 'data/processed/data.csv')
bracket_path = os.path.join(os.path.abspath(os.getcwd()), 'data/processed/results.pkl')

# Load Data
teams = pd.read_csv(data_path)

# Tune Component Models
best_params, model_accs = backwards_model(teams)

# Combine Models
models, accs = combine_model(teams,best_params,model_accs)

# Cross Validated Results
results = {}
results['R32'] = np.mean(accs[2])
results['S16'] = np.mean(accs[3])
results['E8'] = np.mean(accs[4])
results['F4'] = np.mean(accs[5])
results['NCG'] = np.mean(accs[6])
results['Winner'] = np.mean(accs[7])
results_df = pd.DataFrame(list(results.items()), columns=['Round', 'Accuracy'])

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
path = os.path.join(os.path.abspath(os.getcwd()), 'results/model_accuracy.csv')
results_df.to_csv(path,index=False)