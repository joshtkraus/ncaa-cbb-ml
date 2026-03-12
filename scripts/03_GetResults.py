"""Get Results."""

# Libraries
import json
import os

import pandas as pd
from models.ModelPipeline import combine_model

# Load
# Paths
data_path = os.path.join(os.path.abspath(os.getcwd()), "data/processed/data.csv")
bracket_path = os.path.join(os.path.abspath(os.getcwd()), "data/processed/results.json")
# Data
data = pd.read_csv(data_path)
# Real Results
with open(bracket_path, "r") as json_file:
    correct_picks = json.load(json_file)

# Load the tuned params
params_path = os.path.join(os.path.abspath(os.getcwd()), "model/autogluon_params.json")
with open(params_path, "r") as f:
    ag_params = json.load(f)
ag_params = {int(k): v for k, v in ag_params.items()}

# Backtest using frozen model configs
combine_model(data, ag_params, correct_picks)
