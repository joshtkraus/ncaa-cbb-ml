# NCAA College Basketball Tournament Predictive Modeling

## Overview
This project aims to build a predictive model for selecting winners in the NCAA Men's Basketball Tournament. A common approach is to predict the likelihood that *Team X* defeats *Team Y* and use these probabilities iteratively to determine the winner of each matchup. However, this method overlooks a critical aspect of traditional tournament scoring: **the points awarded for a correct pick double in each subsequent round**. This means that correctly picking the national champion is worth **32 times** as many points as selecting a single Round of 64 winner.

## Approach
To account for the scoring structure, this project takes a probabilistic approach by modeling each round of the tournament separately. Instead of selecting winners matchup by matchup, the bracket is filled out recursively, starting with the champion and working backward through each round. This ensures that selections maximize expected points rather than just the likelihood of winning individual games.

## Data
The dataset constructed consists of the teams and results of the past 18 NCAA Tournaments, and is gathered from [SportsReference](https://www.sports-reference.com/cbb/) using *BeautifulSoup* and [KenPom](https://kenpom.com/). Data collected consists of:
- Team Metadata: *Name, Conference, # of Wins, Won Conference Tournament (Y/N), Bracket Region, Seed*
- Efficiency: *Offensive & Defensive Efficiency, Tempo, Luck, Strength of Schedule*
- Points: *Offensive & Defensive Point Rankings*
- Roster: *Height, Experience, Bench Rating*
- Computed Fields: *Historical Seed Performance, Grouped Metrics by Tournament Round & Reigon*

## Models
Classification models were built by-round using two primary models: Gradient Boosting Machines (*XGBoost*) & Multilayer Perceptron (*Keras*) models. Hyperparameters were tuned for each using *Optuna*, with the final models being a weighted ensemble of the two components (the optimal weights which minimizes **Brier Score**). 

## Pick Selection Strategy
The selection metric used to determine each pick to make in each round is the **Expected Points** that would be garnered if the pick was corrected. Using the standard scoring method, this means that *E(Winner)* would be as follows:  
*E(Winner) = p(R32)\*10 + p(S16)\*20 + p(E8)\*40 + p(F4)\*80 + p(NCG)\*160 + p(Winner)\*320*  

Thus, the first pick made would be the team with the most expected points for winning the tournament and this team would be selected throughout each round of the tournament. This process would then continue until the entire bracket is completed. 

## Evalutation
To evaluate the peformance of this strategy, backtesting was implemented starting with the *2013 tournament*. While these results may be overly optimistic given the overlap between training, validation, and backtesting data, the results are as follows: 

Year | Points
--- | ---
2013 | 1150
2014 | 690
2015 | 1270
2016 | 830
2017 | 1490
2018 | 1420
2019 | 1690
2021 | 1510
2022 | 1140
2023 | 1120
2024 | 1370
2025 | Coming Soon

Overall, this method has correctly selected the winner 82% of the time.

## Set-Up
Below are basic commands to create your virtual environement:  
python3 -m venv .venv  
source .venv/bin/activate  
pip install -r requirements.txt

## Contact
Author: Josh Kraus  
Email: joshtkraus@gmail.com
