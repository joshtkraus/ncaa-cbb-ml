def backwards_model(team_data, validation_start=2016):
    print('Tuning Models...')
    # Libraries
    from models.Models import Logistic_Fit, RF_Fit, GB_Fit, NN_Fit
    from sklearn.exceptions import ConvergenceWarning
    from warnings import simplefilter
    simplefilter('ignore', category=ConvergenceWarning)
    
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

        # Neg Log Loss
        # Total
        total_loss = sum([accs[r]['Log'],accs[r]['RF'],accs[r]['GB'],accs[r]['NN']])
        # Normalize
        accs[r]['Log'] = accs[r]['Log'] / total_loss
        accs[r]['RF'] = accs[r]['RF'] / total_loss
        accs[r]['GB'] = accs[r]['GB'] / total_loss
        accs[r]['NN'] = accs[r]['NN'] / total_loss

    return best_params, accs

def combine_model(team_data,best_params,model_accs,correct_picks,backwards_test=2013,validation_year=2017):
    print('Combining Models...')
    # Libraries
    import os
    import pandas as pd
    import json
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
    from models.utils.DataProcessing import create_splits
    from models.utils.StandarizePredictions import standarize
    from models.utils.MakePicks import predict_bracket
    from sklearn.exceptions import ConvergenceWarning
    from warnings import simplefilter
    simplefilter('ignore', category=ConvergenceWarning)
    # Years
    years = [*range(backwards_test-1,2024)]
    years.remove(2020)

    # Initialize
    accs = {}
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
        accs[r] = []

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

            # Best Model w/ Elastic  et
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
            
            # Cross Validation
            # Subset Testing Year
            X_test = X[team_data['Year']==test_year]
            y_test = y[team_data['Year']==test_year]

            # Fit
            voting_clf.fit(X_train, y_train)

            # If training year is one less that validation year
            if year == validation_year-1:
                # Feature Importance
                perm_importance = permutation_importance(voting_clf, X_test_full, y_test_full, n_repeats=10, random_state=0)
                # To DF
                feature_names = X.columns
                importances_mean = perm_importance.importances_mean
                importances_std = perm_importance.importances_std
                importance_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance Mean": importances_mean,
                    "Importance Std": importances_std
                })
                importance_df = importance_df.sort_values(by="Importance Mean", ascending=False)
                # Export
                path = os.path.join(os.path.abspath(os.getcwd()), 'results/feature_importance/Round_'+str(r)+'.csv')
                importance_df.to_csv(path,index=False)

            # Evaluate, store
            accuracy = calculate_precision(y_test, voting_clf.predict(X_test))
            accs[r].append(accuracy)

            # Store Probabilities
            predictions[test_year]['Round_'+str(r)] = voting_clf.predict_proba(X_test)[:,1]

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
    return models, accs, points_df, accs_df