# Feature Selection
def feature_selection(team_data,best_params,model_accs):
    print('Feature Selection...')
     # Libraries
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
    from sklearn.model_selection import cross_val_score
    from models.utils.DataProcessing import create_splits
    from models.utils.CV import custom_time_series_split

    # Initialize
    best_features = {}
    custom_cv = list(custom_time_series_split(1088, 5, 640, 192, 64))

    # Iterate Rounds
    for r in range(2,8):
        print('Round '+str(r))
        # Create Splits
        X, y = create_splits(team_data,r)

        # Model Weights
        weights = [model_accs[r]['Log'],
                model_accs[r]['RF'],
                model_accs[r]['GB'],
                model_accs[r]['NN']]
        
        # Tuned Models
        log = LogisticRegression(**best_params[r]['Log'], max_iter = 50000, solver = 'saga', random_state=0)
        rf = RandomForestClassifier(**best_params[r]['RF'], random_state=0)
        gb = GradientBoostingClassifier(**best_params[r]['GB'], random_state=0)
        nn = MLPClassifier(**best_params[r]['NN'], max_iter = 50000, learning_rate='constant', solver='adam', random_state=0)
        smote = BorderlineSMOTE(**best_params[r]['SMOTE'], sampling_strategy='not majority', random_state=0)
        # Create Voting Classifier
        voting_clf = ImbPipeline([
                        ('scaler', StandardScaler()),
                        ('smote', smote),
                        ('tomek', TomekLinks(sampling_strategy='not minority')),
                        ('clf', VotingClassifier(estimators=[
                                                                ('lr', log),
                                                                ('rf', rf),
                                                                ('gb', gb),
                                                                ('mlp', nn),
                                                            ], voting='soft',weights=weights))
                    ])

        # Initialize
        selected_features = []
        remaining_features = list(X.columns)
        best_score = float('-inf')
        improved = True

        while improved:
            improved = False
            # Forward Selection
            best_addition = None
            for feature in remaining_features:
                candidate_features = selected_features + [feature]
                score = np.mean(cross_val_score(voting_clf, 
                                                X[candidate_features], 
                                                y, 
                                                cv=custom_cv, 
                                                scoring='average_precision'))
                if score > best_score:
                    best_score = score
                    best_addition = feature
                    improved = True
            if best_addition:
                selected_features.append(best_addition)
                remaining_features.remove(best_addition)

            # Backward Elimination
            if len(selected_features) > 1:
                best_removal = None
                for feature in selected_features:
                    candidate_features = [f for f in selected_features if f != feature]
                    score = np.mean(cross_val_score(voting_clf, 
                                                    X[candidate_features], 
                                                    y,
                                                    cv=custom_cv, 
                                                    scoring='average_precision'))
                    if score > best_score:
                        best_score = score
                        best_removal = feature
                        improved = True

                if best_removal:
                    selected_features.remove(best_removal)
                    remaining_features.append(best_removal)
        
        # Store
        best_features[r] = selected_features

    return best_features

def backwards_test(years,validation_year,r,team_data,X,y,model_accs,best_params,prec_list,predictions,models):
    # Libraries
    import os
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
    from sklearn.metrics import average_precision_score

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

        # Tuned Models
        log = LogisticRegression(**best_params[r]['Log'], max_iter = 50000, solver = 'saga', random_state=0)
        rf = RandomForestClassifier(**best_params[r]['RF'], random_state=0)
        gb = GradientBoostingClassifier(**best_params[r]['GB'], random_state=0)
        nn = MLPClassifier(**best_params[r]['NN'], max_iter = 50000, learning_rate='constant', solver='adam', random_state=0)
        smote = BorderlineSMOTE(**best_params[r]['SMOTE'], sampling_strategy='not majority', random_state=0)
        # Create Voting Classifier
        voting_clf = ImbPipeline([
                        ('scaler', StandardScaler()),
                        ('smote', smote),
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

        # Get Prediction
        y_pred = voting_clf.predict_proba(X_test)[:, 1]

        # Permutation Importance
        perm_importance = permutation_importance(voting_clf,
                                                X_test,
                                                y_test,
                                                n_repeats=100,
                                                scoring='average_precision', 
                                                random_state=0,
                                                n_jobs=-1)

        # Get Precision
        precision = average_precision_score(y_test,y_pred)
        
        # Store Averaged Results
        # Precision
        prec_list[r].append(precision)
        # Predictions
        predictions[test_year]['Round_'+str(r)] = y_pred
        # Permutation Importance
        if year == validation_year-1:
            # To DF
            feature_names = X.columns
            importances_mean =perm_importance.importances_mean
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
                    ('smote', smote),
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
    return models, prec_list, predictions