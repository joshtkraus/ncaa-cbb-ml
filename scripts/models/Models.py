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
    from sklearn.metrics import make_scorer, precision_score

    # Data Splits
    X, y = create_splits(team_data,r)

    # Create Training/Valdation Splits
    team_data['Split'] = -1
    team_data.loc[team_data['Year']>=validation_start,'Split'] = 0

    # Parameter grid
    param_grid = {
        'clf__max_iter': [10000],
        'clf__C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10],
        'clf__solver': ['saga'],
        'clf__penalty': ['elasticnet'],
        'clf__l1_ratio':[0.1, 0.3, 0.5, 0.7, 0.9]
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
<<<<<<< HEAD
        scoring=custom_precision_scorer,
=======
        scoring='neg_brier_score',
>>>>>>> add_years
        cv=PredefinedSplit(test_fold=team_data['Split'].values),
        n_jobs=-1
    )
    grid_search.fit(X, y)

    # Get best parameters and precision
    best_acc = 1 / (-1*grid_search.best_score_)
    best_params = grid_search.best_params_
    
    # Remove prefix from tuned param
    best_params = dict(zip([key[5:] for key in best_params.keys()],best_params.values()))

    # Export CV Results
    path_dict = {2:'R32',
                 3:'S16',
                 4:'E8',
                 5:'F4',
                 6:'NCG',
                 7:'Winner'}
    CV_df =  pd.DataFrame.from_dict(grid_search.cv_results_)
    path = os.path.join(os.path.abspath(os.getcwd()), 'results/models/'+path_dict[r]+'/Logistic.csv')
    CV_df.to_csv(path,index=False)
    return best_params, best_acc

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
    from sklearn.metrics import make_scorer, precision_score

    # Data Splits
    X, y = create_splits(team_data,r)

    # Create Training/Valdation Splits
    team_data['Split'] = -1
    team_data.loc[team_data['Year']>=validation_start,'Split'] = 0

    # Parameter grid
    param_grid = {
        'clf__n_estimators':[50, 100, 150, 1500, 2000, 2500],
        'clf__criterion': ['gini', 'entropy'],
        'clf__class_weight': ['balanced', None],
        'clf__min_samples_split': [5, 10, 20, 50, 75],
        'clf__min_samples_leaf': [1, 2, 5, 10, 15, 20],
        'clf__max_depth': [1, 5, 20, 40]
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
<<<<<<< HEAD
        scoring=custom_precision_scorer,
=======
        scoring='neg_brier_score',
>>>>>>> add_years
        cv=PredefinedSplit(test_fold=team_data['Split'].values),
        n_jobs=-1
    )
    grid_search.fit(X, y)

    # Get best parameters and precision
    best_acc = 1 / (-1*grid_search.best_score_)
    best_params = grid_search.best_params_

    # Remove prefix from tuned param
    best_params = dict(zip([key[5:] for key in best_params.keys()],best_params.values()))

    # Export CV Results
    path_dict = {2:'R32',
                 3:'S16',
                 4:'E8',
                 5:'F4',
                 6:'NCG',
                 7:'Winner'}
    CV_df =  pd.DataFrame.from_dict(grid_search.cv_results_)
    path = os.path.join(os.path.abspath(os.getcwd()), 'results/models/'+path_dict[r]+'/RandomForest.csv')
    CV_df.to_csv(path,index=False)
    return best_params, best_acc

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
    from sklearn.metrics import make_scorer, precision_score

    # Data Splits
    X, y = create_splits(team_data,r)

    # Create Training/Valdation Splits
    team_data['Split'] = -1
    team_data.loc[team_data['Year']>=validation_start,'Split'] = 0

    # Parameter grid
    param_grid = {
        'clf__n_estimators': [50, 100, 150, 500, 1000, 1500, 2000, 2500],
        'clf__min_samples_split': [2, 5, 10, 15, 20],
        'clf__min_samples_leaf': [5, 10, 20, 50, 75],
        'clf__max_depth': [1, 2, 5, 8, 10]
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
<<<<<<< HEAD
        scoring=custom_precision_scorer,
=======
        scoring='neg_brier_score',
>>>>>>> add_years
        cv=PredefinedSplit(test_fold=team_data['Split'].values),
        n_jobs=-1
    )
    grid_search.fit(X, y)

    # Get best parameters and precision
    best_acc = 1 / (-1*grid_search.best_score_)
    best_params = grid_search.best_params_

    # Remove prefix from tuned param
    best_params = dict(zip([key[5:] for key in best_params.keys()],best_params.values()))

    # Export CV Results
    path_dict = {2:'R32',
                 3:'S16',
                 4:'E8',
                 5:'F4',
                 6:'NCG',
                 7:'Winner'}
    CV_df =  pd.DataFrame.from_dict(grid_search.cv_results_)
    path = os.path.join(os.path.abspath(os.getcwd()), 'results/models/'+path_dict[r]+'/GradientBoosting.csv')
    CV_df.to_csv(path,index=False)
    return best_params, best_acc

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
    from sklearn.metrics import make_scorer, precision_score

    # Data Splits
    X, y = create_splits(team_data,r)

    # Create Training/Valdation Splits
    team_data['Split'] = -1
    team_data.loc[team_data['Year']>=validation_start,'Split'] = 0

    # Parameter grid
    param_grid = {
        'clf__max_iter':[10000],
        'clf__hidden_layer_sizes': [(20,), (40,), (80,), (20,10), (40,20)],
        'clf__activation': ['tanh', 'relu'],
        'clf__alpha': [0.001, 0.01, 0.1, 1, 10],
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
<<<<<<< HEAD
        scoring=custom_precision_scorer,
=======
        scoring='neg_brier_score',
>>>>>>> add_years
        cv=PredefinedSplit(test_fold=team_data['Split'].values),
        n_jobs=-1
    )
    grid_search.fit(X, y)

    # Get best parameters and precision
    best_acc = 1 / (-1*grid_search.best_score_)
    best_params = grid_search.best_params_
    # Remove prefix from tuned param
    best_params = dict(zip([key[5:] for key in best_params.keys()],best_params.values()))

    # Export CV Results
    path_dict = {2:'R32',
                 3:'S16',
                 4:'E8',
                 5:'F4',
                 6:'NCG',
                 7:'Winner'}
    CV_df =  pd.DataFrame.from_dict(grid_search.cv_results_)
    path = os.path.join(os.path.abspath(os.getcwd()), 'results/models/'+path_dict[r]+'/NeuralNetwork.csv')
    CV_df.to_csv(path,index=False)
    return best_params, best_acc