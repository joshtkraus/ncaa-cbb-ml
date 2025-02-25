# Feature Selection
def get_score(data, r, cv, candidate_features, model_type, params):
    import numpy as np
    from sklearn.metrics import average_precision_score
    from models.utils.DataProcessing import create_splits
    from models.utils.nn import tuned_nn
    from models.utils.gbm import tuned_gbm
    import xgboost as xgb

    # Create Splits
    X_SMTL, y_SMTL = create_splits(data, r, train=True, best_features=candidate_features)
    X, y = create_splits(data, r, train=False, best_features=candidate_features)
    
    # Cross Validation
    score_list = []
    for train_idx, val_idx in cv:
        # Data Splits
        train_idx_SMTL = np.where((X_SMTL == X[train_idx[-1]]).all(axis=1))[0][0]
        X_train, y_train = X_SMTL[:train_idx_SMTL+1], y_SMTL[:train_idx_SMTL+1]
        X_val, y_val = X[val_idx], y[val_idx]

        # Create & Fit Model
        if model_type == 'NN':
            # Create Model
            model = tuned_nn(params[r],
                            X_train, y_train,
                            X_val, y_val)
            # Predict
            pred_prob = model.predict(X_val, verbose=0)
            pred = pred_prob > 0.5
            pred = pred.astype(int).flatten()
        else:
            model = tuned_gbm(params[r],
                                X_train, y_train,
                                X_val, y_val)
            # Predict
            dval = xgb.DMatrix(X_val)
            pred = model.predict(dval)
        # Score
        score_list.append(average_precision_score(y_val, pred))
    # Average Score
    return np.mean(score_list)

def backwards_selection(data, model_type, params):
     # Libraries
    from models.utils.CV import custom_time_series_split
    from models.utils.DataProcessing import get_modeling_cols

    # Initialize
    best_features = {}
    cv = list(custom_time_series_split(1088, 5, 640, 192, 64))

    # Iterate Rounds
    for r in range(2,8):
        print('Round '+str(r))

        # Initialize
        selected_features = get_modeling_cols(data)
        best_score = float('-inf')
        improved = True

        while improved:
            improved = False
            best_removal = None
            for feature in selected_features:
                candidate_features = [f for f in selected_features if f != feature]

                score = get_score(data, r, cv, candidate_features, model_type, params)

                if score > best_score:
                    best_score = score
                    best_removal = feature
                    improved = True

            if best_removal:
                selected_features.remove(best_removal)
        
        # Store
        best_features[r] = selected_features

    return best_features