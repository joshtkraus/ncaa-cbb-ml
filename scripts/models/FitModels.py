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
    from models.utils.metrics import calculate_precision
    from sklearn.metrics import precision_score, make_scorer
    from models.utils.DataProcessing import create_splits
    from models.utils.StandarizePredictions import standarize
    from models.utils.MakePicks import predict_bracket

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

            # Tuned Models
            log = LogisticRegression(**best_params[r]['Log'], random_state=0)
            rf = RandomForestClassifier(**best_params[r]['RF'], random_state=0)
            gb = GradientBoostingClassifier(**best_params[r]['GB'], random_state=0)
            nn = MLPClassifier(**best_params[r]['NN'], random_state=0)

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
                voting_clf.fit(X_train, y_train)

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
                    

                # Get Precision, Probabilities
                prec_sub = precision_score(y_test,
                                           voting_clf.predict(X_test),
                                           pos_label=1,
                                           average='binary',
                                           zero_division=0.0)
                probs_sub = voting_clf.predict_proba(X_test)[:,1]

                # Store
                prec_list_avg.append(prec_sub)
                prob_list_avg.append(probs_sub)
            
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
                        ('smote', BorderlineSMOTE(sampling_strategy='not majority', random_state=0)),
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
    # Iterate Years
    for year in years:
        if year == 2019:
            test_year = 2021
        else:
            test_year = year+1
        # To DF
        pred_df = pd.DataFrame.from_dict(predictions[test_year])
        # Standardize
        pred_df = standarize(pred_df)
        # Export
        path = os.path.join(os.path.abspath(os.getcwd()), 'results/predictions/'+str(test_year)+'.csv')
        pred_df.to_csv(path,index=False)

        # Make Picks
        pick_accs[test_year] = {}
        # Make Predictions
        picks, point, acc = predict_bracket(pred_df,
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