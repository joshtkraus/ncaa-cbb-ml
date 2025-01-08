def backwards_model(team_data):
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
        best_params[r]['Log'],accs[r]['Log'] = Logistic_Fit(team_data,r)
        print('Fitting RF...')
        best_params[r]['RF'],accs[r]['RF'] = RF_Fit(team_data,r)
        print('Fitting GB...')
        best_params[r]['GB'],accs[r]['GB'] = GB_Fit(team_data,r)
        print('Fitting NN...')
        best_params[r]['NN'],accs[r]['NN'] = NN_Fit(team_data,r)

    return best_params, accs

def combine_model(team_data,best_params,model_accs,correct_picks):
    print('Combining Models...')
    # Libraries
    import os
    import pandas as pd
    import json
    from imblearn.pipeline import Pipeline as ImbPipeline
    from sklearn.preprocessing import StandardScaler
    from imblearn.over_sampling import BorderlineSMOTE
    from imblearn.under_sampling import TomekLinks
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import VotingClassifier
    from models.utils.metrics import calculate_precision
    from models.utils.DataProcessing import create_splits
    from models.utils.StandarizePredictions import standarize
    from models.utils.MakePicks import predict_bracket
    from sklearn.exceptions import ConvergenceWarning
    from warnings import simplefilter
    simplefilter('ignore', category=ConvergenceWarning)

    # Initialize
    accs = {}
    cv_models = {}
    models = {}
    predictions = {}
    for year in team_data['Year']:
        predictions[year] = {}
        predictions[year]['Team'] = team_data.loc[team_data['Year']==year,'Team'].values
        predictions[year]['Seed'] = team_data.loc[team_data['Year']==year,'Seed'].values
        predictions[year]['Region'] = team_data.loc[team_data['Year']==year,'Region'].values

    # Iterate Rounds
    for r in range(2,8):
        # Initialize
        accs[r] = []
        cv_models[r] = {}

        # Data Splits
        X, y = create_splits(team_data,r)

        # Tuned Models
        log = LogisticRegression(**best_params[r]['Log'])
        rf = RandomForestClassifier(**best_params[r]['RF'])
        gb = GradientBoostingClassifier(**best_params[r]['GB'])
        nn = MLPClassifier(**best_params[r]['NN'])

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
        for train_idx, test_idx in LeaveOneGroupOut().split(X, y, team_data['Year']):
            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Year in Holdout
            year = X_test['Year'].unique()[0]

            # Fit
            voting_clf.fit(X_train, y_train)

            # Store
            cv_models[r][year] = voting_clf

            # Evaluate, store
            accuracy = calculate_precision(y_test, voting_clf.predict(X_test))
            accs[r].append(accuracy)

            # Store Probabilities
            predictions[year]['Round_'+str(r)] = voting_clf.predict_proba(X_test)[:,1]

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
    for year in team_data['Year'].unique():
        # To DF
        pred_df = pd.DataFrame.from_dict(predictions[year])
        # Standardize
        pred_df = standarize(pred_df)
        # Export
        path = os.path.join(os.path.abspath(os.getcwd()), 'results/predictions/'+str(year)+'.csv')
        pred_df.to_csv(path,index=False)

        # Make Picks
        pick_accs[year] = {}
        # Make Predictions
        picks, point, acc = predict_bracket(pred_df,
                                            correct_picks[str(year)])
        # Save Predictions
        path = os.path.join(os.path.abspath(os.getcwd()), 'results/picks/'+str(year)+'.json')
        with open(path, 'w') as f:
            json.dump(picks, f)
        # Store
        points[year] = point
        pick_accs[year]['R32'] = acc['R32'] / 32
        pick_accs[year]['S16'] = acc['S16'] / 16
        pick_accs[year]['E8'] = acc['E8'] / 8
        pick_accs[year]['F4'] = acc['F4'] / 4
        pick_accs[year]['NCG'] = acc['NCG'] / 2
        pick_accs[year]['Winner'] = acc['Winner']

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