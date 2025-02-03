def Logistic_Fit(team_data, r, best_features=None):    
    # Libraries
    import numpy as np
    import pandas as pd
    import os
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import BorderlineSMOTE
    from imblearn.under_sampling import TomekLinks
    from models.utils.DataProcessing import create_splits
    from models.utils.CV import custom_time_series_split
    from models.utils.Hyperparameters import logistic_params, smote
    import warnings
    warnings.filterwarnings("ignore", message="X has feature names, but StandardScaler was fitted without feature names")

    # Data Splits
    if best_features != None:
        X, y = create_splits(team_data,r,best_features)
    else:
        X, y = create_splits(team_data,r)

    # Get Hyperparameters
    if best_features != None:
        alpha, l1_ratio, class_weight, penalty = logistic_params(step='Post')
    else:
        alpha, l1_ratio, class_weight, penalty = logistic_params(step='Pre')
    k_neighbors = smote()

    # Parameter grid
    param_grid = {
        'smote__k_neighbors': k_neighbors,
        'clf__class_weight': class_weight,
        'clf__C': alpha[r],
        'clf__penalty': penalty,
        'clf__l1_ratio': l1_ratio
    }

    # Pipeline
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', BorderlineSMOTE(sampling_strategy='not majority', random_state=0)),
        ('tomek', TomekLinks(sampling_strategy='not minority')),
        ('clf', LogisticRegression(max_iter = 50000, solver = 'saga', random_state=0))
      
    ])

    # Grid Search
    custom_cv = custom_time_series_split(1088, 5, 640, 192, 64)
    scoring = {'neg_log_loss':'neg_log_loss','average_precision':'average_precision'}
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring=scoring,
        refit='neg_log_loss',
        cv=custom_cv,
        n_jobs=-1
    )
    grid_search.fit(X.to_numpy(), y.to_numpy())

    # Get best parameters and performance
    best_params = grid_search.best_params_
    best_perform = grid_search.cv_results_['mean_test_average_precision'][grid_search.best_index_ ]

    # Remove prefix from tuned param
    best_params = dict(
        zip(
            [
                key[7:] if key.endswith(('k_neighbors')) else key[5:]
                for key in best_params.keys()
            ],
            best_params.values(),
        )
    )
    # Filter Parameters
    model_params = {key: value for key, value in best_params.items() if key not in ['k_neighbors']}
    smote_params = {key: value for key, value in best_params.items() if key in ['k_neighbors']}

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
    return model_params, smote_params, best_perform

def RF_Fit(team_data, r, best_features=None):
    # Libraries
    import numpy as np
    import pandas as pd
    import os
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import BorderlineSMOTE
    from imblearn.under_sampling import TomekLinks
    from models.utils.DataProcessing import create_splits
    from models.utils.CV import custom_time_series_split
    from models.utils.Hyperparameters import random_forest, smote
    import warnings
    warnings.filterwarnings("ignore", message="X has feature names, but StandardScaler was fitted without feature names")

    # Data Splits
    if best_features != None:
        X, y = create_splits(team_data,r,best_features)
    else:
        X, y = create_splits(team_data,r)

    # Get Hyperparameters
    if best_features != None:
        max_depth, n_estimators, min_samples_split, criterion, class_weight, min_samples_leaf = random_forest(step='Post')
    else:
        max_depth, n_estimators, min_samples_split, criterion, class_weight, min_samples_leaf = random_forest(step='Pre')
    k_neighbors = smote()

    # Parameter grid
    param_grid = {
        'smote__k_neighbors': k_neighbors,
        'clf__n_estimators': n_estimators[r],
        'clf__criterion': criterion,
        'clf__class_weight': class_weight,
        'clf__min_samples_split': min_samples_split[r],
        'clf__min_samples_leaf': min_samples_leaf[r], 
        'clf__max_depth': max_depth[r]
    }

    # Pipeline
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', BorderlineSMOTE(sampling_strategy='not majority', random_state=0)),
        ('tomek', TomekLinks(sampling_strategy='not minority')),
        ('clf', RandomForestClassifier(random_state=0))
    ])

    # Grid Search
    custom_cv = custom_time_series_split(1088, 5, 640, 192, 64)
    scoring = {'neg_log_loss':'neg_log_loss','average_precision':'average_precision'}
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring=scoring,
        refit='neg_log_loss',
        cv=custom_cv,
        n_jobs=-1
    )
    grid_search.fit(X.to_numpy(), y.to_numpy())

    # Get best parameters and performance
    best_params = grid_search.best_params_
    best_perform = grid_search.cv_results_['mean_test_average_precision'][grid_search.best_index_ ]

    # Remove prefix from tuned param
    best_params = dict(
        zip(
            [
                key[7:] if key.endswith(('k_neighbors')) else key[5:]
                for key in best_params.keys()
            ],
            best_params.values(),
        )
    )
    # Filter Parameters
    model_params = {key: value for key, value in best_params.items() if key not in ['k_neighbors']}
    smote_params = {key: value for key, value in best_params.items() if key in ['k_neighbors']}

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
    return model_params, smote_params, best_perform

def GB_Fit(team_data, r, best_features=None):
    # Libraries
    import numpy as np
    import pandas as pd
    import os
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import GridSearchCV
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import BorderlineSMOTE
    from imblearn.under_sampling import TomekLinks
    from models.utils.DataProcessing import create_splits
    from models.utils.CV import custom_time_series_split
    from models.utils.Hyperparameters import gradient_boosting, smote
    import warnings
    warnings.filterwarnings("ignore", message="X has feature names, but StandardScaler was fitted without feature names")

    # Data Splits
    if best_features != None:
        X, y = create_splits(team_data,r,best_features)
    else:
        X, y = create_splits(team_data,r)

    # Get Hyperparameters
    max_depth, n_estimators, min_samples_leaf, min_samples_split = gradient_boosting()
    k_neighbors = smote()

    # Parameter grid
    param_grid = {
        'smote__k_neighbors': k_neighbors,
        'clf__n_estimators': n_estimators[r],
        'clf__min_samples_split': min_samples_split[r],
        'clf__min_samples_leaf': min_samples_leaf[r],
        'clf__max_depth': max_depth[r]
    }

    # Pipeline
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', BorderlineSMOTE(sampling_strategy='not majority', random_state=0)),
        ('tomek', TomekLinks(sampling_strategy='not minority')),
        ('clf', GradientBoostingClassifier(random_state=0))
    ])

    # Grid Search
    custom_cv = custom_time_series_split(1088, 5, 640, 192, 64)
    scoring = {'neg_log_loss':'neg_log_loss','average_precision':'average_precision'}
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring=scoring,
        refit='neg_log_loss',
        cv=custom_cv,
        n_jobs=-1
    )
    grid_search.fit(X.to_numpy(), y.to_numpy())

    # Get best parameters and performance
    best_params = grid_search.best_params_
    best_perform = grid_search.cv_results_['mean_test_average_precision'][grid_search.best_index_ ]

    # Remove prefix from tuned param
    best_params = dict(
        zip(
            [
                key[7:] if key.endswith(('k_neighbors')) else key[5:]
                for key in best_params.keys()
            ],
            best_params.values(),
        )
    )
    # Filter Parameters
    model_params = {key: value for key, value in best_params.items() if key not in ['k_neighbors']}
    smote_params = {key: value for key, value in best_params.items() if key in ['k_neighbors']}

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
    return model_params, smote_params, best_perform

def NN_Fit(team_data, r, best_features=None):
    # Libraries
    import numpy as np
    import pandas as pd
    import os
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import GridSearchCV
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import BorderlineSMOTE
    from imblearn.under_sampling import TomekLinks
    from models.utils.DataProcessing import create_splits
    from models.utils.CV import custom_time_series_split
    from models.utils.Hyperparameters import neural_network, smote
    import warnings
    warnings.filterwarnings("ignore", message="X has feature names, but StandardScaler was fitted without feature names")

    # Data Splits
    if best_features != None:
        X, y = create_splits(team_data,r,best_features)
    else:
        X, y = create_splits(team_data,r)

    # Get Hyperparameters
    hidden_layer_size, alpha, activation = neural_network()
    k_neighbors = smote()

    # Parameter grid
    # If Feature Selection has been done, then regularize
    param_grid = {
        'smote__k_neighbors': k_neighbors,
        'clf__hidden_layer_sizes': hidden_layer_size[r],
        'clf__activation': activation,
        'clf__alpha': alpha[r],
    }

    # Pipeline
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', BorderlineSMOTE(sampling_strategy='not majority', random_state=0)),
        ('tomek', TomekLinks(sampling_strategy='not minority')),
        ('clf', MLPClassifier(max_iter = 50000, learning_rate='constant', solver='adam', random_state=0))
    ])

    # Grid Search
    custom_cv = custom_time_series_split(1088, 5, 640, 192, 64)
    scoring = {'neg_log_loss':'neg_log_loss','average_precision':'average_precision'}
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring=scoring,
        refit='neg_log_loss',
        cv=custom_cv,
        n_jobs=-1
    )
    grid_search.fit(X.to_numpy(), y.to_numpy())

    # Get best parameters and performance
    best_params = grid_search.best_params_
    best_perform = grid_search.cv_results_['mean_test_average_precision'][grid_search.best_index_ ]

    # Remove prefix from tuned param
    best_params = dict(
        zip(
            [
                key[7:] if key.endswith(('k_neighbors')) else key[5:]
                for key in best_params.keys()
            ],
            best_params.values(),
        )
    )
    # Filter Parameters
    model_params = {key: value for key, value in best_params.items() if key not in ['k_neighbors']}
    smote_params = {key: value for key, value in best_params.items() if key in ['k_neighbors']}

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
    return model_params, smote_params, best_perform