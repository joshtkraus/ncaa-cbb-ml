def Logistic_Fit(team_data, r, validation_start=2016):    
    # Libraries
    import pandas as pd
    import os
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, PredefinedSplit
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import BorderlineSMOTE
    from imblearn.under_sampling import TomekLinks
    from models.utils.DataProcessing import create_splits
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import brier_score_loss
    import warnings
    warnings.filterwarnings("ignore", message="X has feature names, but StandardScaler was fitted without feature names")

    # Data Splits
    X, y = create_splits(team_data,r)

    # Create Training/Valdation Splits
    team_data['Split'] = -1
    team_data.loc[team_data['Year']>=validation_start,'Split'] = 0

    # Parameter grid
    param_grid = {
        'clf__max_iter': [10000],
        'clf__C': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1],
        'clf__solver': ['saga'],
        'clf__penalty': ['elasticnet'],
        'clf__l1_ratio':[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    }

    # Pipeline
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', BorderlineSMOTE(sampling_strategy='not majority', random_state=0)),
        ('tomek', TomekLinks(sampling_strategy='not minority')),
        ('clf', LogisticRegression(random_state=0))
    ])

    # Grid Search
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring='neg_brier_score',
        cv=PredefinedSplit(test_fold=team_data['Split'].values),
        n_jobs=-1
    )
    grid_search.fit(X.to_numpy(), y.to_numpy())

    # Get best parameters and performance
    best_params = grid_search.best_params_

    # Calibrate Probability
    # Validation Set
    X_val = X[team_data['Split']==0]
    y_val = y[team_data['Split']==0]
    # Isotonic Regression
    calibrate = CalibratedClassifierCV(estimator=grid_search.best_estimator_.named_steps['clf'], 
                                              cv='prefit', 
                                              method='isotonic')
    calibrate.fit(X_val.to_numpy(), y_val.to_numpy())
    # Evaluate
    calibrate_y = calibrate.predict_proba(X_val.to_numpy())[:, 1]
    best_perform = 1 - brier_score_loss(y_val, calibrate_y)
    
    # Remove prefix from tuned param
    best_params = dict(zip([key[5:] for key in best_params.keys()],best_params.values()))

    # Export Validation Results
    path_dict = {2:'R32',
                 3:'S16',
                 4:'E8',
                 5:'F4',
                 6:'NCG',
                 7:'Winner'}
    Val_df =  pd.DataFrame.from_dict(grid_search.cv_results_)
    path = os.path.join(os.path.abspath(os.getcwd()), 'results/models/'+path_dict[r]+'/Logistic.csv')
    Val_df.to_csv(path,index=False)
    return best_params, best_perform

def RF_Fit(team_data, r, validation_start=2016):
    # Libraries
    import pandas as pd
    import os
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV, PredefinedSplit
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import BorderlineSMOTE
    from imblearn.under_sampling import TomekLinks
    from models.utils.DataProcessing import create_splits
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import brier_score_loss
    import warnings
    warnings.filterwarnings("ignore", message="X has feature names, but StandardScaler was fitted without feature names")

    # Data Splits
    X, y = create_splits(team_data,r)

    # Create Training/Valdation Splits
    team_data['Split'] = -1
    team_data.loc[team_data['Year']>=validation_start,'Split'] = 0

    # Parameter grid
    param_grid = {
        'clf__n_estimators':[25, 50, 100, 150],
        'clf__criterion': ['gini', 'entropy'],
        'clf__class_weight': ['balanced', None],
        'clf__min_samples_split': [2, 5, 10, 75],
        'clf__min_samples_leaf': [1, 2, 5],
        'clf__max_depth': [5, 10, 15, 20, 25]
    }

    # Pipeline
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', BorderlineSMOTE(sampling_strategy='not majority', random_state=0)),
        ('tomek', TomekLinks(sampling_strategy='not minority')),
        ('clf', RandomForestClassifier(random_state=0))
    ])

    # Grid Search
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring='neg_brier_score',
        cv=PredefinedSplit(test_fold=team_data['Split'].values),
        n_jobs=-1
    )
    grid_search.fit(X.to_numpy(), y.to_numpy())

    # Get best parameters and performance
    best_params = grid_search.best_params_

    # Calibrate Probability
    # Validation Set
    X_val = X[team_data['Split']==0]
    y_val = y[team_data['Split']==0]
    # Isotonic Regression
    calibrate = CalibratedClassifierCV(estimator=grid_search.best_estimator_.named_steps['clf'], 
                                              cv='prefit', 
                                              method='isotonic')
    calibrate.fit(X_val.to_numpy(), y_val.to_numpy())
    # Evaluate
    calibrate_y = calibrate.predict_proba(X_val.to_numpy())[:, 1]
    best_perform = 1 - brier_score_loss(y_val, calibrate_y)

    # Remove prefix from tuned param
    best_params = dict(zip([key[5:] for key in best_params.keys()],best_params.values()))

    # Export Validation Results
    path_dict = {2:'R32',
                 3:'S16',
                 4:'E8',
                 5:'F4',
                 6:'NCG',
                 7:'Winner'}
    Val_df =  pd.DataFrame.from_dict(grid_search.cv_results_)
    path = os.path.join(os.path.abspath(os.getcwd()), 'results/models/'+path_dict[r]+'/RandomForest.csv')
    Val_df.to_csv(path,index=False)
    return best_params, best_perform

def GB_Fit(team_data, r, validation_start=2016):
    # Libraries
    import pandas as pd
    import os
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import GridSearchCV, PredefinedSplit
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import BorderlineSMOTE
    from imblearn.under_sampling import TomekLinks
    from models.utils.DataProcessing import create_splits
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import brier_score_loss
    import warnings
    warnings.filterwarnings("ignore", message="X has feature names, but StandardScaler was fitted without feature names")

    # Data Splits
    X, y = create_splits(team_data,r)

    # Create Training/Valdation Splits
    team_data['Split'] = -1
    team_data.loc[team_data['Year']>=validation_start,'Split'] = 0

    # Parameter grid
    param_grid = {
        'clf__n_estimators': [10, 25, 50, 100, 200, 300],
        'clf__min_samples_split': [2, 3, 5],
        'clf__min_samples_leaf': [20, 50, 75, 100, 150],
        'clf__max_depth': [5, 10, 15, 20, 25]
    }

    # Pipeline
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', BorderlineSMOTE(sampling_strategy='not majority', random_state=0)),
        ('tomek', TomekLinks(sampling_strategy='not minority')),
        ('clf', GradientBoostingClassifier(random_state=0))
    ])

    # Grid Search
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring='neg_brier_score',
        cv=PredefinedSplit(test_fold=team_data['Split'].values),
        n_jobs=-1
    )
    grid_search.fit(X.to_numpy(), y.to_numpy())

    # Get best parameters and performance
    best_params = grid_search.best_params_

    # Calibrate Probability
    # Validation Set
    X_val = X[team_data['Split']==0]
    y_val = y[team_data['Split']==0]
    # Isotonic Regression
    calibrate = CalibratedClassifierCV(estimator=grid_search.best_estimator_.named_steps['clf'], 
                                              cv='prefit', 
                                              method='isotonic')
    calibrate.fit(X_val.to_numpy(), y_val.to_numpy())
    # Evaluate
    calibrate_y = calibrate.predict_proba(X_val.to_numpy())[:, 1]
    best_perform = 1 - brier_score_loss(y_val, calibrate_y)

    # Remove prefix from tuned param
    best_params = dict(zip([key[5:] for key in best_params.keys()],best_params.values()))

    # Export Validation Results
    path_dict = {2:'R32',
                 3:'S16',
                 4:'E8',
                 5:'F4',
                 6:'NCG',
                 7:'Winner'}
    Val_df =  pd.DataFrame.from_dict(grid_search.cv_results_)
    path = os.path.join(os.path.abspath(os.getcwd()), 'results/models/'+path_dict[r]+'/GradientBoosting.csv')
    Val_df.to_csv(path,index=False)
    return best_params, best_perform

def NN_Fit(team_data, r, validation_start=2016):
    # Libraries
    import pandas as pd
    import os
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import GridSearchCV, PredefinedSplit
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import BorderlineSMOTE
    from imblearn.under_sampling import TomekLinks
    from models.utils.DataProcessing import create_splits
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import brier_score_loss
    import warnings
    warnings.filterwarnings("ignore", message="X has feature names, but StandardScaler was fitted without feature names")

    # Data Splits
    X, y = create_splits(team_data,r)

    # Create Training/Valdation Splits
    team_data['Split'] = -1
    team_data.loc[team_data['Year']>=validation_start,'Split'] = 0

    # Parameter grid
    param_grid = {
        'clf__max_iter':[10000],
        'clf__hidden_layer_sizes': [(40,), (60,), (80,), (120,), (160,), (200,), (240,) ,
                                    (40,20), (60, 30), (80,40), (120,60), (160,80), (200,100), (240,120)],
        'clf__activation': ['tanh', 'relu'],
        'clf__alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 15],
        'clf__learning_rate': ['constant'],
        'clf__solver': ['adam']
    }

    # Pipeline
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', BorderlineSMOTE(sampling_strategy='not majority', random_state=0)),
        ('tomek', TomekLinks(sampling_strategy='not minority')),
        ('clf', MLPClassifier(random_state=0))
    ])

    # Grid Search
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring='neg_brier_score',
        cv=PredefinedSplit(test_fold=team_data['Split'].values),
        n_jobs=-1
    )
    grid_search.fit(X.to_numpy(), y.to_numpy())

    # Get best parameters and performance
    best_params = grid_search.best_params_

    # Calibrate Probability
    # Validation Set
    X_val = X[team_data['Split']==0]
    y_val = y[team_data['Split']==0]
    # Isotonic Regression
    calibrate = CalibratedClassifierCV(estimator=grid_search.best_estimator_.named_steps['clf'], 
                                              cv='prefit', 
                                              method='isotonic')
    calibrate.fit(X_val.to_numpy(), y_val.to_numpy())
    # Evaluate
    calibrate_y = calibrate.predict_proba(X_val.to_numpy())[:, 1]
    best_perform = 1 - brier_score_loss(y_val, calibrate_y)

    # Remove prefix from tuned param
    best_params = dict(zip([key[5:] for key in best_params.keys()],best_params.values()))

    # Export Validation Results
    path_dict = {2:'R32',
                 3:'S16',
                 4:'E8',
                 5:'F4',
                 6:'NCG',
                 7:'Winner'}
    Val_df =  pd.DataFrame.from_dict(grid_search.cv_results_)
    path = os.path.join(os.path.abspath(os.getcwd()), 'results/models/'+path_dict[r]+'/NeuralNetwork.csv')
    Val_df.to_csv(path,index=False)
    return best_params, best_perform