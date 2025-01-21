# Tune Models
def train_models(team_data, validation_start=2017):
    print('Tuning Models...')
    # Libraries
    from models.Models import Logistic_Fit, RF_Fit, GB_Fit, NN_Fit
    
    # Initialize
    best_params = {}
    accs = {}
    
    # Iterate Rounds
    for r in range(2,8):
        print('Round',r)
        # Initalize
        best_params[r] = {}
        accs[r] = {}

        # Fit Models
        print('Fitting Logistic...')
        best_params[r]['Log'],accs[r]['Log'] = Logistic_Fit(team_data,r,validation_start)
        print('Fitting RF...')
        best_params[r]['RF'],accs[r]['RF'] = RF_Fit(team_data,r,validation_start)
        print('Fitting GB...')
        best_params[r]['GB'],accs[r]['GB'] = GB_Fit(team_data,r,validation_start)
        print('Fitting NN...')
        best_params[r]['NN'],accs[r]['NN'] = NN_Fit(team_data,r,validation_start)

        # Normalize Neg Brier Loss
        # Total Loss
        total_loss = sum([accs[r]['Log'],accs[r]['RF'],accs[r]['GB'],accs[r]['NN']])
        # Normalize
        accs[r]['Log'] = accs[r]['Log'] / total_loss
        accs[r]['RF'] = accs[r]['RF'] / total_loss
        accs[r]['GB'] = accs[r]['GB'] / total_loss
        accs[r]['NN'] = accs[r]['NN'] / total_loss

    return best_params, accs

# To Plot Calibration Curves
def save_calibration_curve(df_joined):
    # Libraries
    import os
    from sklearn.calibration import calibration_curve
    import matplotlib.pyplot as plt

    # Map Round to Column Name
    col_map = {
        2:'Round_2',
        3:'Round_3',
        4:'Round_4',
        5:'Round_5',
        6:'Round_6',
        7:'Round_7'
    }

    # Iterate Rounds
    for r in [2,3,4,5,6,7]:
        # Outcome
        df_joined['Outcome'] = 0
        df_joined.loc[df_joined['Round']==r,'Outcome'] = 1

        # Calibration Curves
        prob_true, prob_pred = calibration_curve(df_joined['Outcome'], df_joined[col_map[r]], pos_label=1, n_bins=8)

        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, marker='o', label='Calibration Curve')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
        plt.xlabel('Predicted Probability')
        plt.ylabel('True Probability')
        plt.title('Calibration Curve: Round '+str(r))
        plt.legend()
        # Export
        path = os.path.join(os.path.abspath(os.getcwd()), 'results/calibration_curves/Round_'+str(r)+'.png')
        plt.savefig(path, bbox_inches='tight')
        plt.close()

# Combine Component Models
def combine_model(team_data,best_params,model_accs,correct_picks,backwards_test=2013,validation_year=2017):
    print('Combining Models...')
    # Libraries
    import os
    import json
    import numpy as np
    import pandas as pd
    from imblearn.pipeline import Pipeline as ImbPipeline
    from sklearn.preprocessing import StandardScaler
    from imblearn.over_sampling import BorderlineSMOTE
    from imblearn.under_sampling import TomekLinks
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import VotingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import precision_score, make_scorer
    from models.utils.DataProcessing import create_splits
    from models.utils.StandarizePredictions import standarize
    from models.utils.MakePicks import predict_bracket
    from sklearn.calibration import CalibratedClassifierCV
    import warnings
    warnings.filterwarnings("ignore", message="X has feature names, but StandardScaler was fitted without feature names")

    # Years to Backwards Test
    years = [*range(backwards_test-1,2024)]
    years.remove(2020)

    # Initialize
    precision_scorer = make_scorer(precision_score, pos_label=1,average='binary',zero_division=0.0)
    prec_list = {}
    models = {}
    predictions = {}
    for year in years:
        if year == 2019:
            test_year = 2021
        else:
            test_year = year+1
        predictions[test_year] = {}
        predictions[test_year]['Team'] = team_data.loc[team_data['Year']==test_year,'Team'].values
        predictions[test_year]['Seed'] = team_data.loc[team_data['Year']==test_year,'Seed'].values
        predictions[test_year]['Region'] = team_data.loc[team_data['Year']==test_year,'Region'].values

    # Iterate Rounds
    for r in range(2,8):
        # Initialize
        prec_list[r] = []

        # Data Splits
        X, y = create_splits(team_data,r)

        # Test Set for Permutation Importance
        X_test_full = X[team_data['Year']>=validation_year]
        y_test_full = y[team_data['Year']>=validation_year]

        # Iterate years
        for year in years:
            if year == 2019:
                test_year = 2021
            else:
                test_year = year+1

            # Create Splits
            X_train = X[team_data['Year']<=year]
            y_train = y[team_data['Year']<=year]

            # Model Weights
            weights = [model_accs[r]['Log'],
                    model_accs[r]['RF'],
                    model_accs[r]['GB'],
                    model_accs[r]['NN']]
                       
            # Subset Testing Year
            X_test = X[team_data['Year']==test_year]
            y_test = y[team_data['Year']==test_year]

            # Iterate random states
            # Max Iterations
            max_iter = 1
            # Initialize
            prec_list_avg = []
            prob_list_avg = []
            import_list_avg = []
            for state in range(0,max_iter):
                # Tuned Models
                log = LogisticRegression(**best_params[r]['Log'], random_state=state)
                rf = RandomForestClassifier(**best_params[r]['RF'], random_state=state)
                gb = GradientBoostingClassifier(**best_params[r]['GB'], random_state=state)
                nn = MLPClassifier(**best_params[r]['NN'], random_state=state)
                # Create Voting Classifier
                voting_clf = ImbPipeline([
                                ('scaler', StandardScaler()),
                                ('smote', BorderlineSMOTE(sampling_strategy='not majority', random_state=state)),
                                ('tomek', TomekLinks(sampling_strategy='not minority')),
                                ('clf', VotingClassifier(estimators=[
                                                                        ('lr', log),
                                                                        ('rf', rf),
                                                                        ('gb', gb),
                                                                        ('mlp', nn),
                                                                    ], voting='soft',weights=weights))
                            ])
                # Fit
                voting_clf.fit(X_train.to_numpy(), y_train.to_numpy())

                # If end of Training meets w/ Validation Set
                if year == validation_year-1:
                    # Permutation Importance
                    perm_importance = permutation_importance(voting_clf,
                                                            X_test_full,
                                                            y_test_full,
                                                            n_repeats=10,
                                                            scoring=precision_scorer, 
                                                            random_state=0)
                    import_list_avg.append(perm_importance.importances_mean)

                # Get Prediction
                y_pred = voting_clf.predict_proba(X_test.to_numpy())[:, 1]

                # Get Precision, Probabilities
                prec_sub = precision_score(y_test,
                                           [1 if prob >= 0.5 else 0 for prob in y_pred],
                                           pos_label=1,
                                           average='binary',
                                           zero_division=0.0)

                # Store
                prec_list_avg.append(prec_sub)
                prob_list_avg.append(y_pred)
            
            # Get Averaged Results
            prec = np.mean(prec_list_avg,axis=0)
            prob = np.mean(prob_list_avg,axis=0)

            # Store Averaged Results
            # Precision
            prec_list[r].append(prec)
            # Predictions
            predictions[test_year]['Round_'+str(r)] = prob
            # Permutation Importance
            if year == validation_year-1:
                # Get Average Results
                importance = np.mean(import_list_avg,axis=0)
                # To DF
                feature_names = X.columns
                importances_mean =importance
                importance_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance Mean": importances_mean,
                })
                importance_df = importance_df.sort_values(by="Importance Mean", ascending=False)
                path = os.path.join(os.path.abspath(os.getcwd()), 'results/feature_importance/Round_'+str(r)+'.csv')
                importance_df.to_csv(path,index=False)
        
        # Get Full Model
        voting_clf = ImbPipeline([
                        ('scaler', StandardScaler()),
                        ('smote', BorderlineSMOTE(sampling_strategy='not majority', random_state=state)),
                        ('tomek', TomekLinks(sampling_strategy='not minority')),
                        ('clf', VotingClassifier(estimators=[
                                                                ('lr', log),
                                                                ('rf', rf),
                                                                ('gb', gb),
                                                                ('mlp', nn),
                                                            ], voting='soft',weights=weights))
                        ])
        voting_clf.fit(X,y)
        models[r] = voting_clf
    
    # Standardize Predictions & Export
    # Initialize
    points = {}
    pick_accs = {}
    pred_raw = pd.DataFrame()
    # Iterate Years
    for year in years:
        if year == 2019:
            test_year = 2021
        else:
            test_year = year+1
        # To DF
        pred_df = pd.DataFrame.from_dict(predictions[test_year])
        # Add Year
        pred_df['Year'] = test_year
        # Combine DF
        pred_raw = pd.concat([pred_raw,pred_df],ignore_index=True)
        # Standardize
        pred_df = standarize(pred_df)
        # Export
        path = os.path.join(os.path.abspath(os.getcwd()), 'results/predictions/'+str(test_year)+'.csv')
        pred_df.to_csv(path,index=False)

        # Get Expected Points
        points_df = pred_df.copy()
        points_df['R32'] = pred_df['R32']*10
        points_df['S16'] = pred_df['R32']*10 + pred_df['S16']*20
        points_df['E8'] = pred_df['R32']*10 + pred_df['S16']*20 + pred_df['E8']*40
        points_df['F4'] = pred_df['R32']*10 + pred_df['S16']*20 + pred_df['E8']*40 + pred_df['F4']*80
        points_df['NCG'] = pred_df['R32']*10 + pred_df['S16']*20 + pred_df['E8']*40 + pred_df['F4']*80 + pred_df['NCG']*160
        points_df['Winner'] = pred_df['R32']*10 + pred_df['S16']*20 + pred_df['E8']*40 + pred_df['F4']*80 + pred_df['NCG']*160 + pred_df['Winner']*320

        # Make Picks
        pick_accs[test_year] = {}
        # Make Predictions
        picks, point, acc = predict_bracket(points_df,
                                            correct_picks[str(test_year)])
        # Save Predictions
        path = os.path.join(os.path.abspath(os.getcwd()), 'results/picks/'+str(test_year)+'.json')
        with open(path, 'w') as f:
            json.dump(picks, f)
        # Store
        points[test_year] = point
        pick_accs[test_year]['R32'] = acc['R32'] / 32
        pick_accs[test_year]['S16'] = acc['S16'] / 16
        pick_accs[test_year]['E8'] = acc['E8'] / 8
        pick_accs[test_year]['F4'] = acc['F4'] / 4
        pick_accs[test_year]['NCG'] = acc['NCG'] / 2
        pick_accs[test_year]['Winner'] = acc['Winner']

    # Calibration Curves
    # Read Historical Data
    path = os.path.join(os.path.abspath(os.getcwd()), 'data/processed/data.csv')
    historical = pd.read_csv(path)
    # Join
    df_joined = pred_raw.merge(historical,on=['Team','Seed','Region','Year'])
    # Subset
    df_joined = df_joined[['Year','Team','Seed','Region','Round','Round_2','Round_3','Round_4','Round_5','Round_6','Round_7']]
    # Calibration Curve
    save_calibration_curve(df_joined)

    # Create DFs
    # Points
    points_df = pd.DataFrame([points])
    points_df['Mean'] = points_df.mean(axis=1).iloc[0]
    points_df['SD'] = points_df.std(axis=1).iloc[0]
    # Accuracy
    accs_df = pd.DataFrame(pick_accs).reset_index()
    accs_df.rename(columns={'index': 'Round'}, inplace=True)
    accs_df['Mean'] = accs_df.iloc[:, 1:].mean(axis=1)
    accs_df['Standard Deviation'] = accs_df.iloc[:, 1:-1].std(axis=1)
    return models, prec_list, points_df, accs_df