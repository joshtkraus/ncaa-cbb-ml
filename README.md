# NCAA Basketball Tournament Predictive Model

A machine learning pipeline to predict NCAA tournament outcomes and generate optimized bracket picks using Automated Machine Learning and Monte Carlo simulation.

## Overview
This project builds a predictive model for the NCAA Men's Basketball Tournament using a two-model architecture and a simulation-based bracket selection strategy that maximizes **expected points** under the standard scoring structure where point values double each round.

## Modeling Approach

### Round-by-Round Classification (By-Round Model)
A binary classifier is trained per round (R32 through Championship). The outcome for each team is whether they reached that round or beyond. This produces a marginal advancement probability for each team at each stage, which feeds into the simulation pipeline.

### Matchup Model
A head-to-head binary classifier trained on differential features between two tournament opponents. Predicts the win probability for a specific matchup directly, capturing features that the by-round model cannot.

### Model
[AutoGluon](https://auto.ml/autogluon) `TabularPredictor` with `best_quality` preset. The best-performing base model type and exact hyperparameters are selected via walk-forward cross-validation, then frozen for backtesting.

### Cross-Validation
Walk-forward CV with 3 folds. Each fold's training set consists of all years prior to the validation window, respecting temporal ordering and preventing data leakage.

### Backtesting
Walk-forward backtesting from 2015 onward. For each test year, the frozen model configs are refit on all prior years and evaluated on the held-out year. 2020 is excluded (no tournament).

---

## Pick Selection

Bracket picks are generated via a decoupled two-pool simulation architecture in `SimulatePicks.py`.

### Candidate Brackets (N simulations)
Each simulation produces one complete bracket outcome using a two-signal blend:

```
p(A wins) = α × matchup_prob + (1−α) × conditional_ratio
```

where `α ~ Uniform(0, 1)` is drawn independently per round per simulation, and `conditional_ratio = p_A / (p_A + p_B)` from the by-round model's marginal probabilities. This explores the full blend spectrum without constraints. Every simulated bracket is a candidate.

### Scoring Simulations (M simulations)
A separate pool of M scoring simulations is generated using a two-stage blend that incorporates historical seed survival rates:

- **Stage 1:** blend matchup model and conditional ratio with random weight `α`
- **Stage 2:** blend Stage 1 result with historical `_Actual_Full` seed survival rates with independent random weight `β ~ Uniform(0, 1)`

This produces a historically-grounded scoring distribution that reflects realistic upset frequencies. Because the two pools are decoupled, increasing M stabilises the scoring distribution toward historical realism rather than toward model-biased chalk.

### Candidate Selection
All N candidates are scored against all M scoring simulations. Each candidate's final score combines two signals via z-score normalisation:

```
combined = 0.5 × z_score(mean_points) + 0.5 × z_score(log_likelihood)
```

The **log-likelihood** is computed using favored-seed binary outcomes: for each pod at each round, it checks whether the lower-numbered (favored) seed advanced and contributes `log(p)` if they did, `log(1−p)` if not. This penalises brackets whose seed distribution deviates from historical expectations without discarding team-specific signal from the scoring step.

Pods evaluated:
- **R32:** seeds 1–8 (favored seed in each of the 8 matchup pairs per region)
- **S16:** seeds 1–4 (top seed from each of the 4 pods per region)
- **E8:** seeds 1–2 (upper-half winner per region)
- **F4 / NCG / Winner:** log-probability of whoever advanced at each stage

---

## Data

Per-team features for each tournament year scraped from [SportsReference](https://www.sports-reference.com/cbb/) and [KenPom](https://kenpom.com/):

- **Metadata:** conference, seed, region, wins, conference tournament result
- **Efficiency:** adjusted offensive/defensive efficiency, tempo, strength of schedule
- **Points/Roster:** offensive/defensive point distributions, height, experience, bench depth
- **Derived:** historical seed survival rates, grouped KenPom averages by tournament round and region

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

### 2. Train by-round models
Runs walk-forward CV per round, selects the best AutoGluon model config, and saves frozen hyperparameters to `model/autogluon_params.json`. Re-run this when new tournament data is added.
```bash
python 02_TrainModels.py
```

### 3. Train matchup model
Trains the head-to-head matchup classifier and saves to `model/matchup/`.
```bash
python 02b_TrainMatchupModel.py
```

### 4. Backtest
Refits the frozen model configs in a walk-forward backtest from 2013 onward and exports accuracy and points results to `results/backwards_test/`.
```bash
python 03_GetResults.py
```

### 4. Visualize Brackets (Optional)
Visualized the backtested picks in a typical bracket format, denoting when picks are correct or not. Saves brackets to `results/brackets/`
```bash
python 04_CreateBrackets.py
```

### 5. Generate predictions
Scrapes current-year data, refits models on all historical data, and outputs bracket probabilities and picks to `prediction/`.
```bash
python 05_MakePredictions.py
```

> Update `year` and `playin_KP` at the top of `05_MakePredictions.py` each tournament year.

---

## Backtested Results

| Year | Points |
|------|--------|
| 2015 | 1350 |
| 2016 | 670 |
| 2017 | 1430 |
| 2018 | 1070 |
| 2019 | 960 |
| 2021 | 850 |
| 2022 | 580 |
| 2023 | 480 |
| 2024 | 1360 |
| 2025 | 1200 |

**Mean: 995 pts** &nbsp;|&nbsp; **SD: 326 pts** &nbsp;|&nbsp; **Overall pick accuracy: 63.2%**
