# Libraries
import os
import pandas as pd
import json
from models.ModelPipeline import combine_model

# Load
# Paths
data_path = os.path.join(os.path.abspath(os.getcwd()), 'data/processed/data.csv')
bracket_path = os.path.join(os.path.abspath(os.getcwd()), 'data/processed/results.json')
# Data
data = pd.read_csv(data_path)
# Real Results
with open(bracket_path, "r") as json_file:
    correct_picks = json.load(json_file)

# Params
# NN
nn_path = os.path.join(os.path.abspath(os.getcwd()), 'models/pre_fs/nn.json')
with open(nn_path, "r") as json_file:
    nn_params = json.load(json_file)
nn_params = {int(key): value for key, value in nn_params.items()}
# GBM
gbm_path = os.path.join(os.path.abspath(os.getcwd()), 'models/pre_fs/gbm.json')
with open(gbm_path, "r") as json_file:
    gbm_params = json.load(json_file)
gbm_params = {int(key): value for key, value in gbm_params.items()}

# # Features
# # NN
# nn_path = os.path.join(os.path.abspath(os.getcwd()), 'models/features/nn.json')
# with open(nn_path, "r") as json_file:
#     nn_feat = json.load(json_file)
# nn_feat = {int(key): value for key, value in nn_feat.items()}
# # GBM
# gbm_path = os.path.join(os.path.abspath(os.getcwd()), 'models/features/gbm.json')
# with open(gbm_path, "r") as json_file:
#     gbm_feat = json.load(json_file)
# gbm_feat = {int(key): value for key, value in gbm_feat.items()}

# Weights
weights_path = os.path.join(os.path.abspath(os.getcwd()), 'models/weights.json')
with open(weights_path, "r") as json_file:
    weights = json.load(json_file)
weights = {int(key): value for key, value in weights.items()}

# Combine Models & Predict
combine_model(data,nn_params,gbm_params,weights,correct_picks,backwards_year=2013)