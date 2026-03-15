# NCAA College Basketball Tournament Predictive Modeling

## Overview
This project builds a predictive model for the NCAA Men's Basketball Tournament. Rather than predicting head-to-head matchups, it models each round independently and fills the bracket recursively from champion backward — maximizing **expected points** under the standard scoring structure where point values double each round.

## Modeling Approach

### Round-by-Round Classification
A binary classifier is trained per round (R32 through Championship). The outcome for each team is whether they reached that round or beyond. This produces a probability for each team at each stage, which feeds directly into the pick selection strategy.

### Model
[AutoGluon](https://auto.ml/autogluon) `TabularPredictor` with `best_quality` preset. Bagging and stacking are disabled so an explicit walk-forward validation split can be used. The best-performing base model type and exact hyperparameters are selected via walk-forward cross-validation, then frozen for backtesting. Class imbalance is handled via balanced class weights.

### Cross-Validation
Walk-forward CV with 3 folds. Each fold's training set consists of all years prior to the validation window, respecting temporal ordering and preventing data leakage. The model with the highest mean val-set ROC-AUC across folds is selected per round.

### Backtesting
Walk-forward backtesting from 2013 onward. For each test year, the frozen model config is refit on all prior years and evaluated on the held-out year. 2020 is excluded (no tournament).

### Pick Selection
Each team's expected points are computed as:

```
E = p(R32)×10 + p(S16)×20 + p(E8)×40 + p(F4)×80 + p(NCG)×160 + p(Winner)×320
```

The bracket is filled by selecting the highest expected-points team at each position, working from champion backward.

### Data
Per-team features for each tournament year scraped from [SportsReference](https://www.sports-reference.com/cbb/) and [KenPom](https://kenpom.com/):
- **Metadata:** conference, seed, region, wins, conference tournament result
- **Efficiency:** adjusted offensive/defensive efficiency, tempo, strength of schedule
- **Points/Roster:** offensive/defensive point distributions, height, experience, bench depth
- **Derived:** historical seed survival rates (full history, 12-year, 6-year windows), grouped KenPom averages by tournament round and region

---

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for environment and dependency management.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install dependencies
uv sync

# Activate
source .venv/bin/activate       # macOS/Linux
.venv\Scripts\activate          # Windows

# Install pre-commit hooks (ruff, mypy, pydoclint)
uv run pre-commit install
```

---

## Usage

Scripts are run from the `scripts/` directory.

### 1. Build the dataset
Scrapes and processes historical tournament data into `data/processed/data.csv`.
```bash
python 01_GetData.py
```

### 2. Tune models
Runs walk-forward CV per round, selects the best AutoGluon model config, and saves frozen hyperparameters to `model/autogluon_params.json`. Re-run this when new tournament data is added.
```bash
python 02_TrainModels.py
```

### 3. Backtest
Refits the frozen model configs in a walk-forward backtest from 2013 onward and exports accuracy and points results to `results/backwards_test/`.
```bash
python 03_GetResults.py
```

### 4. Generate predictions
Scrapes current-year data, refits models on all historical data, and outputs bracket probabilities and picks to `prediction/`.
```bash
python 04_MakePredictions.py
```

> Update `year` and `playin_KP` at the top of `04_MakePredictions.py` each tournament year.

---

## Backtested Results

| Year | Points |
|------|--------|
| 2015 | 940 |
| 2016 | 680 |
| 2017 | 880 |
| 2018 | 960 |
| 2019 | 1330 |
| 2021 | 940 |
| 2022 | 520 |
| 2023 | 570 |
| 2024 | 860 |
| 2025 | 1670 |

**Mean: 935 pts** &nbsp;|&nbsp; **SD: 328 pts** &nbsp;|&nbsp; **Overall pick accuracy: 63.7%**
